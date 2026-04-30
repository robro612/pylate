"""Token retrieval analysis: P(Gold | rank) and P(Lexical | rank).

For each query token, retrieves k nearest neighbor document tokens from a
pre-built ScaNN index and records the raw (qid, query_token_id, rank,
doc_id, doc_token_id, score) triples as a gzipped TSV.  Then derives
is_lexical and is_gold signals post-hoc and plots P(signal | rank).

Usage:
    python scripts/token_retrieval_analysis.py
    python scripts/token_retrieval_analysis.py dataset=beir/fiqa/test k_token=500
"""

from __future__ import annotations

import gc
import gzip
import logging
import os

import hydra
import ir_datasets
import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from plot_style import get_style
from pylate import indexes, models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_id: str,
) -> tuple[dict[str, str], dict[str, dict[str, int]]]:
    """Load queries and qrels from an ir_datasets collection.

    Returns (queries, qrels) where:
    - queries: dict mapping query_id -> query_text
    - qrels: dict mapping query_id -> {doc_id -> relevance_score}
    """
    logger.info("Loading dataset: %s", dataset_id)
    dataset = ir_datasets.load(dataset_id)

    queries = {}
    for query in tqdm(dataset.queries_iter(), desc="Loading queries"):
        queries[query.query_id] = query.text.strip()

    qrels: dict[str, dict[str, int]] = {}
    if dataset.has_qrels():
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    logger.info("Loaded %d queries, %d queries with qrels", len(queries), len(qrels))
    return queries, qrels


# ---------------------------------------------------------------------------
# Raw retrieval collection
# ---------------------------------------------------------------------------

TSV_COLUMNS = ["qid", "query_token_id", "rank", "doc_id", "doc_token_id", "score"]


def collect_raw_retrieval(
    model_cfg: DictConfig,
    queries: dict[str, str],
    k_token: int,
    query_length: int,
    batch_size: int,
    output_path: str,
) -> None:
    """Run ANN retrieval and write raw results to a gzipped TSV.

    Each row is one (query, query_token, rank) triple with columns:
    qid, query_token_id, rank, doc_id, doc_token_id, score
    """
    # 1. Load ScaNN index
    index_path = model_cfg.index_path
    index_folder = str(os.path.dirname(index_path))
    index_name = str(os.path.basename(index_path))
    logger.info("Loading ScaNN index: %s", index_path)
    index = indexes.ScaNN(
        index_folder=index_folder,
        index_name=index_name,
        override=False,
    )
    if index.position_to_token_id is None:
        raise ValueError(
            f"ScaNN index at {index_path} "
            "does not have token IDs. Rebuild with encode.save_token_ids=true."
        )

    # 2. Encode queries with token IDs
    logger.info("Encoding queries with model: %s", model_cfg.path)
    model = models.ColBERT(
        model_name_or_path=model_cfg.path,
        query_length=query_length,
    )
    query_ids = list(queries.keys())
    query_texts = list(queries.values())

    query_embs, query_token_ids = model.encode(
        sentences=query_texts,
        batch_size=batch_size,
        is_query=True,
        show_progress_bar=True,
        return_token_ids=True,
    )

    # 3. Capture mask token ID for filtering expansion tokens
    mask_token_id = model.tokenizer.mask_token_id
    logger.info("Mask token ID: %d", mask_token_id)

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 4. Build real-token masks (exclude expansion/padding)
    real_masks = [qtids != mask_token_id for qtids in query_token_ids]
    total_real_tokens = sum(m.sum() for m in real_masks)
    logger.info(
        "Total query tokens: %d real out of %d (%.1f%% are expansion)",
        total_real_tokens,
        sum(len(m) for m in real_masks),
        100 * (1 - total_real_tokens / max(sum(len(m) for m in real_masks), 1)),
    )

    # 5. Run ANN retrieval
    logger.info("Running ANN retrieval with k_token=%d...", k_token)
    query_embs_np = [
        e.cpu().numpy() if isinstance(e, torch.Tensor) else np.asarray(e, dtype=np.float32)
        for e in query_embs
    ]
    result = index(query_embs_np, k=k_token)

    # 6. Write TSV
    logger.info("Writing raw retrieval data to %s", output_path)
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    rows_written = 0
    with gzip.open(output_path, "wt") as f:
        f.write("\t".join(TSV_COLUMNS) + "\n")

        for qi in tqdm(range(len(query_ids)), desc="Writing TSV"):
            qid = query_ids[qi]
            q_real_mask = real_masks[qi]
            q_query_tids = query_token_ids[qi]
            q_doc_ids = result["documents_ids"][qi]    # list of arrays (T_q, k)
            q_doc_tids = result["token_ids"][qi]       # list of arrays (T_q, k)
            q_scores = result["distances"][qi]          # list of arrays (T_q, k)

            for ti in range(len(q_real_mask)):
                if not q_real_mask[ti]:
                    continue

                qtid = int(q_query_tids[ti])
                neighbor_doc_ids = q_doc_ids[ti]
                neighbor_doc_tids = q_doc_tids[ti]
                neighbor_scores = q_scores[ti]

                for rank_idx in range(len(neighbor_doc_ids)):
                    f.write(
                        f"{qid}\t{qtid}\t{rank_idx + 1}\t"
                        f"{neighbor_doc_ids[rank_idx]}\t"
                        f"{int(neighbor_doc_tids[rank_idx])}\t"
                        f"{float(neighbor_scores[rank_idx]):.6f}\n"
                    )
                    rows_written += 1

    logger.info("Wrote %d rows to %s", rows_written, output_path)


# ---------------------------------------------------------------------------
# Analysis and plotting
# ---------------------------------------------------------------------------


def load_and_annotate(
    tsv_path: str,
    qrels: dict[str, dict[str, int]],
) -> pd.DataFrame:
    """Load a raw retrieval TSV and add is_lexical / is_gold columns."""
    df = pd.read_csv(tsv_path, sep="\t", dtype={"qid": str, "doc_id": str})

    # is_lexical: same vocab token ID
    df["is_lexical"] = df["query_token_id"] == df["doc_token_id"]

    # is_gold: doc_id appears in qrels for this query with relevance > 0
    # Build a set of (qid, doc_id) pairs for fast lookup
    gold_pairs = set()
    for qid, doc_rels in qrels.items():
        for doc_id, rel in doc_rels.items():
            if rel > 0:
                gold_pairs.add((qid, doc_id))

    df["is_gold"] = [
        (row.qid, row.doc_id) in gold_pairs for row in df.itertuples()
    ]

    return df


def _apply_plot_cfg(ax, plot_cfg: DictConfig | None) -> None:
    """Apply xscale, yscale, xlim, ylim from config to an axis."""
    if plot_cfg is None:
        return
    xscale = plot_cfg.get("xscale", "linear")
    yscale = plot_cfg.get("yscale", "linear")
    if xscale and xscale != "linear":
        ax.set_xscale(xscale)
    if yscale and yscale != "linear":
        ax.set_yscale(yscale)
    xlim = plot_cfg.get("xlim")
    ylim = plot_cfg.get("ylim")
    if xlim is not None:
        ax.set_xlim(list(xlim))
    if ylim is not None:
        ax.set_ylim(list(ylim))


def plot_rank_probabilities(
    data_by_label: dict[str, pd.DataFrame],
    output_dir: str,
    dpi: int = 300,
    plot_cfg: DictConfig | None = None,
    label_styles: dict[str, dict] | None = None,
) -> None:
    """Plot P(Gold | rank) and P(Lexical | rank) curves."""
    import matplotlib.pyplot as plt

    styles = label_styles or {}

    # --- Combined 1x2 subplot ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    for label, df in data_by_label.items():
        p_gold = df.groupby("rank")["is_gold"].mean()
        p_lexical = df.groupby("rank")["is_lexical"].mean()
        kw = styles.get(label, {})

        ax1.plot(p_gold.index, p_gold.values, label=label, linewidth=2, **kw)
        ax2.plot(p_lexical.index, p_lexical.values, label=label, linewidth=2, **kw)

    ax1.set_xlabel("Rank")
    ax1.set_ylabel("P(Gold | rank)")
    ax1.legend()
    ax1.set_title("Fraction of ANN hits from relevant documents")
    _apply_plot_cfg(ax1, plot_cfg)

    ax2.set_xlabel("Rank")
    ax2.set_ylabel("P(Lexical | rank)")
    ax2.legend()
    ax2.set_title("Fraction of ANN hits that are lexical matches")
    _apply_plot_cfg(ax2, plot_cfg)

    fig.tight_layout()
    combined_path = os.path.join(output_dir, "token_retrieval_analysis.pdf")
    fig.savefig(combined_path, dpi=dpi)
    logger.info("Saved combined plot to %s", combined_path)
    plt.close(fig)

    # --- Separate single-panel plots ---
    for metric, ylabel in [("is_gold", "P(Gold | rank)"), ("is_lexical", "P(Lexical | rank)")]:
        fig, ax = plt.subplots(figsize=(7, 5))
        for label, df in data_by_label.items():
            p = df.groupby("rank")[metric].mean()
            kw = styles.get(label, {})
            ax.plot(p.index, p.values, label=label, linewidth=2, **kw)
        ax.set_xlabel("Rank")
        ax.set_ylabel(ylabel)
        ax.legend()
        _apply_plot_cfg(ax, plot_cfg)
        fig.tight_layout()

        fname = f"p_{'gold' if metric == 'is_gold' else 'lexical'}_vs_rank.pdf"
        path = os.path.join(output_dir, fname)
        fig.savefig(path, dpi=dpi)
        logger.info("Saved %s", path)
        plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../conf/token_retrieval_analysis",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    out_dir = cfg.output.dir
    raw_dir = os.path.join(out_dir, "raw")
    os.makedirs(raw_dir, exist_ok=True)

    # Load queries and qrels
    queries, qrels = load_dataset(cfg.dataset)

    # Build name -> label mapping for all models
    name_to_label = {}
    for model_cfg in cfg.models:
        name_to_label[model_cfg.name] = model_cfg.get("label", model_cfg.name)

    # Collect raw retrieval data for each model
    tsv_paths: dict[str, str] = {}  # name -> tsv_path
    for model_cfg in cfg.models:
        name = model_cfg.name
        tsv_path = os.path.join(raw_dir, f"{name}.tsv.gz")
        tsv_paths[name] = tsv_path

        if os.path.exists(tsv_path):
            logger.info("Found cached TSV for '%s': %s", name_to_label[name], tsv_path)
            continue

        logger.info("Collecting raw retrieval for '%s'...", name_to_label[name])
        collect_raw_retrieval(
            model_cfg=model_cfg,
            queries=queries,
            k_token=cfg.k_token,
            query_length=cfg.query_length,
            batch_size=cfg.batch_size,
            output_path=tsv_path,
        )

    # Determine which models to include in plots
    plot_cfg = cfg.get("plot")
    include_names = list(plot_cfg.get("include")) if plot_cfg and plot_cfg.get("include") else list(tsv_paths.keys())

    # Build label -> style mapping from canonical plot_style module
    label_styles = {}
    for model_cfg in cfg.models:
        label = model_cfg.get("label", model_cfg.name)
        label_styles[label] = get_style(model_cfg.name)

    # Load and annotate TSVs for included models only
    logger.info("Loading and annotating TSVs...")
    data_by_label = {}
    for name in include_names:
        if name not in tsv_paths:
            logger.warning("Model '%s' in plot.include not found in models, skipping.", name)
            continue
        label = name_to_label[name]
        df = load_and_annotate(tsv_paths[name], qrels)
        logger.info(
            "  %s: %d rows, P(Gold)=%.4f, P(Lexical)=%.4f",
            label, len(df), df["is_gold"].mean(), df["is_lexical"].mean(),
        )
        data_by_label[label] = df

    # Plot
    plot_rank_probabilities(
        data_by_label=data_by_label,
        output_dir=out_dir,
        dpi=cfg.output.get("dpi", 300),
        plot_cfg=plot_cfg,
        label_styles=label_styles,
    )

    logger.info("Done.")


if __name__ == "__main__":
    main()
