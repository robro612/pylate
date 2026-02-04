"""
Analyze P(Gold | rank k) for token-level retrieval.

This script reproduces the analysis from the XTR paper showing the probability
that a token retrieved at rank k comes from a gold (relevant) document.

Usage:
    python analyze_token_rank.py
    python analyze_token_rank.py model.name_or_path='path/to/model'
    python analyze_token_rank.py dataset.names='[beir/nfcorpus/test]'
"""

from __future__ import annotations

import itertools
import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Set, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pylate import indexes, models

# Reuse helpers from eval_model_irds_v2
from eval_model_irds_v2 import (
    build_cache_paths,
    build_index_configs,
    build_model,
    encode_queries_with_cache,
    expand_model_paths,
    get_embedding_size,
    get_torch_dtype,
    load_dataset,
    resolve_query_length,
    sanitize_dataset_name,
    sanitize_model_name,
)

logger = logging.getLogger(__name__)


def collect_gold_rank_statistics(
    index: indexes.Base,
    queries_embeddings: List[torch.Tensor],
    query_ids: List[str],
    qrels: Dict[str, Dict[str, int]],
    k_token: int,
    relevance_threshold: int,
    batch_size: int,
) -> Tuple[np.ndarray, np.ndarray, int]:
    """
    Collect statistics on gold document presence at each rank position.

    Parameters
    ----------
    index
        The index to query.
    queries_embeddings
        List of query embeddings (one tensor per query, shape [num_tokens, dim]).
    query_ids
        List of query IDs corresponding to queries_embeddings.
    qrels
        Mapping from query_id -> {doc_id -> relevance}.
    k_token
        Number of neighbors to retrieve per query token.
    relevance_threshold
        Minimum relevance score to consider a document as "gold".
    batch_size
        Batch size for index retrieval.

    Returns
    -------
    gold_counts : np.ndarray
        Array of shape (k_token,) with count of gold documents at each rank.
    total_counts : np.ndarray
        Array of shape (k_token,) with total count of tokens at each rank.
    total_tokens : int
        Total number of query tokens processed.
    """
    gold_counts = np.zeros(k_token, dtype=np.int64)
    total_counts = np.zeros(k_token, dtype=np.int64)
    total_tokens = 0

    # Build gold document sets for each query
    gold_docs_per_query: Dict[str, Set[str]] = {}
    for query_id in query_ids:
        if query_id in qrels:
            gold_docs_per_query[query_id] = {
                doc_id
                for doc_id, rel in qrels[query_id].items()
                if rel >= relevance_threshold
            }
        else:
            gold_docs_per_query[query_id] = set()

    # Process queries in batches
    num_queries = len(queries_embeddings)
    num_batches = (num_queries + batch_size - 1) // batch_size

    for batch_idx in tqdm(range(num_batches), desc="Processing query batches"):
        start_idx = batch_idx * batch_size
        end_idx = min(start_idx + batch_size, num_queries)

        batch_embeddings = queries_embeddings[start_idx:end_idx]
        batch_query_ids = query_ids[start_idx:end_idx]

        # Query the index
        index_results = index(batch_embeddings, k=k_token)

        # Process results for each query in the batch
        for query_offset, (query_id, query_doc_ids) in enumerate(
            zip(batch_query_ids, index_results["documents_ids"])
        ):
            gold_docs = gold_docs_per_query[query_id]

            # Process each token's results
            for token_doc_ids in query_doc_ids:
                # token_doc_ids is an array of doc_ids at ranks 0..k_token-1
                num_results = len(token_doc_ids)
                actual_k = min(num_results, k_token)

                for rank in range(actual_k):
                    doc_id = token_doc_ids[rank]
                    if doc_id in gold_docs:
                        gold_counts[rank] += 1
                    total_counts[rank] += 1

                total_tokens += 1

    return gold_counts, total_counts, total_tokens


def plot_p_gold_at_rank(
    p_gold: np.ndarray,
    output_path: Path,
    model_name: str,
    dataset_name: str,
    k_token: int,
    figure_format: str = "png",
) -> None:
    """
    Plot P(Gold | rank k) and save to file.

    Parameters
    ----------
    p_gold
        Array of P(Gold | rank k) values.
    output_path
        Directory to save the figure.
    model_name
        Name of the model (for title).
    dataset_name
        Name of the dataset (for title).
    k_token
        Maximum rank analyzed.
    figure_format
        Output format (png or pdf).
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    ranks = np.arange(1, len(p_gold) + 1)  # 1-indexed ranks for display

    ax.plot(ranks, p_gold, linewidth=1.5)
    ax.set_xlabel("Rank k", fontsize=12)
    ax.set_ylabel("P(Gold | rank k)", fontsize=12)
    ax.set_title(
        f"Token-Level Retrieval: P(Gold | rank k)\n{sanitize_model_name(model_name)} on {dataset_name}",
        fontsize=14,
    )
    ax.set_xlim(1, k_token)
    ax.set_ylim(0, None)
    ax.grid(True, alpha=0.3)

    # Add log-scale x-axis version as inset or secondary plot
    ax.set_xscale("log")

    plt.tight_layout()

    # Save figure
    output_path.mkdir(parents=True, exist_ok=True)
    model_slug = sanitize_model_name(model_name)
    dataset_slug = sanitize_dataset_name(dataset_name)
    fig_path = output_path / f"p_gold_at_rank_{model_slug}_{dataset_slug}.{figure_format}"
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()

    logger.info("Figure saved to: %s", fig_path)


def run_analysis(
    cfg: DictConfig,
    model_name: str,
    dataset_id: str,
) -> Dict:
    """
    Run token rank analysis for a single model/dataset combination.

    Returns
    -------
    dict
        Analysis results including p_gold array and metadata.
    """
    logger.info("")
    logger.info("=" * 80)
    logger.info("Analyzing: %s on %s", model_name, dataset_id)
    logger.info("=" * 80)

    # Resolve query length
    query_length = resolve_query_length(dataset_id, cfg.model.query_len)
    doc_length = cfg.model.doc_len
    embedding_dtype = get_torch_dtype(cfg.cache.embedding_dtype)

    # Load model
    model = build_model(cfg, model_name, query_length, doc_length)
    embedding_size = get_embedding_size(model)

    # Load dataset
    documents, queries, qrels = load_dataset(dataset_id, lowercase=cfg.dataset.lowercase)
    documents_ids = [doc["id"] for doc in documents]
    query_ids = list(queries.keys())

    # Filter to queries that have qrels
    queries_with_qrels = {qid: queries[qid] for qid in query_ids if qid in qrels}
    query_ids_filtered = list(queries_with_qrels.keys())
    logger.info(
        "Filtered to %d queries with qrels (from %d total)",
        len(query_ids_filtered),
        len(query_ids),
    )

    # Build cache paths (same as eval_v2 for compatibility)
    dataset_slug = sanitize_dataset_name(dataset_id)
    cache_paths = build_cache_paths(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        doc_length=doc_length,
        query_length=query_length,
        lowercase=cfg.dataset.lowercase,
    )

    # Build index config to load existing index
    index_configs = build_index_configs(
        cfg=cfg,
        dataset_slug=dataset_slug,
        model_name=model_name,
        embedding_size=embedding_size,
        doc_embed_key=cache_paths.doc_hash,
    )

    # We expect exactly one index config (ScaNN)
    if not index_configs:
        raise ValueError("No index configuration found. Check your config.")

    index_config = index_configs[0]
    logger.info("Loading index: %s", index_config["name"])
    logger.info("Index init kwargs: %s", index_config["init_kwargs"])

    # Initialize index (will load from disk if exists)
    index: indexes.Base = index_config["index_class"](**index_config["init_kwargs"])

    if not getattr(index, "_documents_added", False):
        raise ValueError(
            f"Index not found or empty. Please run eval_model_irds_v2.py first to build the index.\n"
            f"Expected index at: {index_config['init_kwargs'].get('index_folder')}/{index_config['init_kwargs'].get('name')}"
        )

    logger.info("Index loaded successfully with %d documents", len(index.doc_id_to_embedding_range))

    # Encode queries (or load from cache)
    logger.info("Encoding queries...")
    queries_embeddings = encode_queries_with_cache(
        model=model,
        queries=queries_with_qrels,
        batch_size=cfg.encode.batch_size,
        embedding_dtype=embedding_dtype,
        move_to_cpu=cfg.encode.move_embeddings_to_cpu,
        cache_paths=cache_paths,
        cache_enabled=cfg.cache.enable,
    )

    # Collect statistics
    logger.info("Collecting gold rank statistics...")
    logger.info("  k_token=%d, relevance_threshold=%d", cfg.analysis.k_token, cfg.analysis.relevance_threshold)

    gold_counts, total_counts, total_tokens = collect_gold_rank_statistics(
        index=index,
        queries_embeddings=queries_embeddings,
        query_ids=query_ids_filtered,
        qrels=qrels,
        k_token=cfg.analysis.k_token,
        relevance_threshold=cfg.analysis.relevance_threshold,
        batch_size=cfg.analysis.batch_size,
    )

    # Compute P(Gold | rank k)
    # Avoid division by zero
    with np.errstate(divide="ignore", invalid="ignore"):
        p_gold = np.where(total_counts > 0, gold_counts / total_counts, 0.0)

    logger.info("Total query tokens processed: %d", total_tokens)
    logger.info("P(Gold | rank 1): %.4f", p_gold[0] if len(p_gold) > 0 else 0)
    logger.info("P(Gold | rank 10): %.4f", p_gold[9] if len(p_gold) > 9 else 0)
    logger.info("P(Gold | rank 100): %.4f", p_gold[99] if len(p_gold) > 99 else 0)
    logger.info("P(Gold | rank 1000): %.4f", p_gold[999] if len(p_gold) > 999 else 0)

    # Build results dict
    results = {
        "model": model_name,
        "dataset": dataset_id,
        "k_token": cfg.analysis.k_token,
        "relevance_threshold": cfg.analysis.relevance_threshold,
        "total_queries": len(query_ids_filtered),
        "total_tokens": total_tokens,
        "p_gold": p_gold.tolist(),
        "gold_counts": gold_counts.tolist(),
        "total_counts": total_counts.tolist(),
        "timestamp": datetime.now().isoformat(),
    }

    # Save results
    output_dir = Path(cfg.output.results_dir) / dataset_slug / sanitize_model_name(model_name)
    output_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output.save_data:
        data_path = output_dir / "token_rank_data.json"
        with open(data_path, "w") as f:
            json.dump(results, f, indent=2)
        logger.info("Data saved to: %s", data_path)

    # Plot
    plot_p_gold_at_rank(
        p_gold=p_gold,
        output_path=output_dir,
        model_name=model_name,
        dataset_name=dataset_id,
        k_token=cfg.analysis.k_token,
        figure_format=cfg.output.figure_format,
    )

    return results


@hydra.main(version_base=None, config_path="conf/eval", config_name="experiment_4_rank_analysis")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger.info("Token Rank Analysis")
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    model_paths = expand_model_paths(cfg.model.name_or_path)
    dataset_names = list(cfg.dataset.names)

    logger.info("Models to analyze: %s", model_paths)
    logger.info("Datasets to analyze: %s", dataset_names)

    all_results = []

    with logging_redirect_tqdm():
        for dataset_id, model_name in itertools.product(dataset_names, model_paths):
            try:
                results = run_analysis(cfg, model_name, dataset_id)
                all_results.append(results)
            except Exception as e:
                logger.error("Failed to analyze %s on %s: %s", model_name, dataset_id, e)
                raise

    # Save combined results
    if len(all_results) > 1:
        combined_path = Path(cfg.output.results_dir) / "all_token_rank_results.json"
        combined_path.parent.mkdir(parents=True, exist_ok=True)
        with open(combined_path, "w") as f:
            json.dump(all_results, f, indent=2)
        logger.info("Combined results saved to: %s", combined_path)

    logger.info("Analysis complete.")


if __name__ == "__main__":
    main()
