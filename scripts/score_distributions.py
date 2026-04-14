"""Token-level similarity score distribution experiment.

Encodes queries and documents for each configured model, collects the raw
token-level cosine similarity scores produced during retrieval, and plots
their kernel density estimates.

Retrieval modes
---------------
exact
    Brute-force exact matmul of query tokens vs all doc tokens.
    Best for small datasets (NFCorpus, SciFact, FIQA).  No index required.
scann
    Load a pre-built ScaNN index (built via benchmark_indexes.py) and run
    retrieval to collect the raw ANN token distances.  Set ``scann_index``
    per model entry in the config.

Usage
-----
    python scripts/score_distributions.py
    python scripts/score_distributions.py dataset=beir/fiqa/test num_queries=200
    python scripts/score_distributions.py retrieval.mode=scann \\
        'models=[{name: xtr, path: robro612/xtr-base-en-pylate, \\
                  label: XTR, scann_index: benchmark_indexes/.../scann}]'
"""

from __future__ import annotations

import logging
import os

import hydra
import ir_datasets
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from plot_style import get_style
from pylate import indexes, models

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Dataset loading
# ---------------------------------------------------------------------------


def load_dataset(dataset_id: str) -> tuple[list[dict], list[tuple[str, str]]]:
    """Load documents and queries from an ir_datasets collection.

    Returns (documents, queries) where documents is a list of dicts with
    'id' and 'text' keys, and queries is a list of (query_id, query_text).
    """
    logger.info("Loading dataset: %s", dataset_id)
    dataset = ir_datasets.load(dataset_id)

    documents = []
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents"):
        if hasattr(doc, "title") and doc.title:
            text = f"{doc.title}\n\n{doc.text}".strip()
        else:
            text = doc.text.strip()
        documents.append({"id": doc.doc_id, "text": text})

    queries = []
    for query in tqdm(dataset.queries_iter(), desc="Loading queries"):
        queries.append((query.query_id, query.text.strip()))

    logger.info("Loaded %d documents, %d queries", len(documents), len(queries))
    return documents, queries


# ---------------------------------------------------------------------------
# Score collection
# ---------------------------------------------------------------------------


def collect_exact_scores(
    query_embs: list[torch.Tensor],
    doc_embs_flat: torch.Tensor,
    k_token: int,
) -> np.ndarray:
    """Collect top-k_token cosine similarities per query token via exact matmul.

    Parameters
    ----------
    query_embs:
        List of (T_q, D) tensors on device, one per query.
    doc_embs_flat:
        (N_tokens, D) tensor on device of all document token embeddings.
    k_token:
        Number of top scores to keep per query token (mirrors ANN retrieval).

    Returns
    -------
    np.ndarray
        Flat array of all collected token similarity scores.
    """
    k = min(k_token, doc_embs_flat.shape[0])
    all_scores = []
    for q_emb in tqdm(query_embs, desc="Collecting exact scores"):
        # (T_q, N_tokens) — cosine sim (embeddings assumed unit-normed)
        sim = q_emb @ doc_embs_flat.T
        topk_scores = sim.topk(k, dim=1).values  # (T_q, k)
        all_scores.append(topk_scores.cpu().numpy().flatten())

    return np.concatenate(all_scores)


def collect_scann_scores(
    index: indexes.ScaNN,
    query_embs: list[np.ndarray],
    k_token: int,
) -> np.ndarray:
    """Collect raw token similarity scores from a pre-built ScaNN index.

    Parameters
    ----------
    index:
        A loaded ScaNN index.
    query_embs:
        List of (T_q, D) arrays, one per query.
    k_token:
        Top-k tokens per query token to retrieve.

    Returns
    -------
    np.ndarray
        Flat array of all collected token similarity scores.
    """
    results = index(query_embs, k=k_token)
    # results["distances"]: list (per query) of (T_q, k_token) arrays
    all_scores = []
    for per_query_distances in results["distances"]:
        all_scores.append(np.asarray(per_query_distances, dtype=np.float32).flatten())
    return np.concatenate(all_scores)


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------


def plot_distributions(
    scores_by_label: dict[str, np.ndarray],
    output_path: str,
    dpi: int = 150,
    max_kde_samples: int = 100_000,
    label_styles: dict[str, dict] | None = None,
) -> None:
    """Plot KDE score distributions, one curve per model."""
    import matplotlib.pyplot as plt
    from scipy.stats import gaussian_kde

    styles = label_styles or {}
    rng = np.random.default_rng(0)
    fig, ax = plt.subplots(figsize=(8, 5))

    for label, scores in scores_by_label.items():
        scores = scores[np.isfinite(scores)]
        if len(scores) == 0:
            logger.warning("No finite scores for '%s', skipping.", label)
            continue
        if len(scores) > max_kde_samples:
            scores = rng.choice(scores, size=max_kde_samples, replace=False)
        kde = gaussian_kde(scores, bw_method="scott")
        xs = np.linspace(float(scores.min()), float(scores.max()), 500)
        ys = kde(xs)
        kw = styles.get(label, {})
        color = ax.plot(xs, ys, label=label, linewidth=2, **kw)[0].get_color()
        ax.fill_between(xs, ys, facecolor=color, alpha=0.15)

    ax.set_xlabel("Token Similarity Score")
    ax.set_ylabel("Density")
    ax.legend()
    fig.tight_layout()

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    fig.savefig(output_path, dpi=dpi)
    logger.info("Saved plot to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(
    config_path="../conf/score_distributions",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    device = cfg.get("device") or ("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("Device: %s", device)

    rng = np.random.default_rng(cfg.seed)

    out_dir = cfg.output.dir
    scores_dir = os.path.join(out_dir, "scores")
    os.makedirs(scores_dir, exist_ok=True)

    # Determine which models need encoding
    models_to_encode = []
    scores_by_label: dict[str, np.ndarray] = {}

    for model_cfg in cfg.models:
        name = model_cfg.name
        label = model_cfg.get("label", name)
        scores_path = os.path.join(scores_dir, f"{name}.npy")
        if os.path.exists(scores_path):
            logger.info("Loading cached scores for '%s' from %s", label, scores_path)
            scores_by_label[label] = np.load(scores_path)
        else:
            models_to_encode.append(model_cfg)

    # Encode missing models
    if models_to_encode:
        documents, all_queries = load_dataset(cfg.dataset)

        if cfg.num_queries is not None and cfg.num_queries < len(all_queries):
            idx = rng.choice(len(all_queries), size=cfg.num_queries, replace=False)
            idx.sort()
            queries = [all_queries[i] for i in idx]
        else:
            queries = all_queries
        query_texts = [text for _, text in queries]
        logger.info("Using %d / %d queries", len(query_texts), len(all_queries))

        doc_texts = [doc["text"] for doc in documents]

        for model_cfg in models_to_encode:
            name = model_cfg.name
            path = model_cfg.path
            label = model_cfg.get("label", name)
            scann_index_path = model_cfg.get("scann_index", None)

            logger.info("=== Model: %s (%s) ===", label, path)

            model = models.ColBERT(
                model_name_or_path=path,
                document_length=cfg.doc_length,
            )

            logger.info("Encoding %d queries...", len(query_texts))
            query_embs = model.encode(
                sentences=query_texts,
                batch_size=cfg.encode.batch_size,
                is_query=True,
                show_progress_bar=True,
            )
            query_embs = [
                e.to(device) if isinstance(e, torch.Tensor) else torch.from_numpy(np.asarray(e, dtype=np.float32)).to(device)
                for e in query_embs
            ]

            if cfg.retrieval.mode == "exact":
                logger.info("Encoding %d documents (exact mode)...", len(doc_texts))
                doc_embs = model.encode(
                    sentences=doc_texts,
                    batch_size=cfg.encode.batch_size,
                    is_query=False,
                    show_progress_bar=True,
                )
                doc_embs_flat = torch.cat([
                    e.to(device) if isinstance(e, torch.Tensor) else torch.from_numpy(np.asarray(e, dtype=np.float32)).to(device)
                    for e in doc_embs
                ], dim=0)
                logger.info(
                    "Doc token matrix: %s (%.1f MB)",
                    tuple(doc_embs_flat.shape),
                    doc_embs_flat.element_size() * doc_embs_flat.nelement() / 1e6,
                )
                scores = collect_exact_scores(
                    query_embs=query_embs,
                    doc_embs_flat=doc_embs_flat,
                    k_token=cfg.retrieval.k_token,
                )
                del doc_embs, doc_embs_flat

            elif cfg.retrieval.mode == "scann":
                if not scann_index_path:
                    raise ValueError(
                        f"Model '{name}' has no scann_index path configured. "
                        "Set scann_index per model or use retrieval.mode=exact."
                    )
                logger.info("Loading ScaNN index from %s", scann_index_path)
                index = indexes.ScaNN(index_folder=scann_index_path, override=False)
                scores = collect_scann_scores(
                    index=index,
                    query_embs=query_embs,
                    k_token=cfg.retrieval.k_token,
                )
                del index

            else:
                raise ValueError(
                    f"Unknown retrieval.mode: {cfg.retrieval.mode!r}. Use 'exact' or 'scann'."
                )

            logger.info("Collected %d token scores for '%s'", len(scores), label)
            scores_by_label[label] = scores

            scores_path = os.path.join(scores_dir, f"{name}.npy")
            np.save(scores_path, scores)
            logger.info("Saved scores to %s", scores_path)

            del model, query_embs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # --- Plot ---
    label_styles = {}
    for model_cfg in cfg.models:
        label = model_cfg.get("label", model_cfg.name)
        label_styles[label] = get_style(model_cfg.name)

    plot_path = os.path.join(out_dir, cfg.output.plot_file)
    plot_distributions(
        scores_by_label=scores_by_label,
        output_path=plot_path,
        dpi=cfg.output.get("dpi", 150),
        label_styles=label_styles,
    )


if __name__ == "__main__":
    main()
