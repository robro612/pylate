"""
Token Score Distribution Analysis

Analyzes the distribution of token-level similarity scores for positive (relevant)
and negative (non-relevant) documents across different models.

This replicates the analysis from Figure 2 of the XTR paper, which shows that
ColBERT training causes many document tokens to have extremely high scores
regardless of their actual relevance, while XTR mitigates this with a better
training objective.

Prerequisites:
    - Existing index built with eval_model_irds_v3.py
    - Query embeddings (will be cached)

Usage:
    python analyze_token_scores.py
    python analyze_token_scores.py model.name_or_path=[model1,model2] dataset.names=[dataset1]
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import hydra
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from pylate import indexes, models

# Import helpers from eval script
from eval_model_irds_v3 import (
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


@dataclass
class TokenScoreData:
    """Container for token score distribution data."""
    model_name: str
    dataset_name: str
    # All token scores (flattened)
    all_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    # Positive document token scores
    pos_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    # Negative document token scores
    neg_scores: np.ndarray = field(default_factory=lambda: np.array([]))
    # Metadata
    num_queries: int = 0
    num_pos_tokens: int = 0
    num_neg_tokens: int = 0


def extract_token_scores_from_retrieval(
    query_id: str,
    token_doc_ids: List[np.ndarray],  # List of arrays, one per query token
    token_scores: List[np.ndarray],   # List of arrays, one per query token
    qrels_for_query: Dict[str, int],
    relevance_threshold: int = 1,
) -> Tuple[List[float], List[float]]:
    """
    Extract and segment token-level scores from index retrieval results.

    Parameters
    ----------
    query_id
        The query ID (for logging/debugging).
    token_doc_ids
        For each query token, the document IDs of retrieved tokens.
        Shape: (num_query_tokens, k) where k is top-k retrieved per token.
    token_scores
        For each query token, the similarity scores of retrieved tokens.
        Shape: (num_query_tokens, k).
    qrels_for_query
        Mapping from doc_id -> relevance score for this query.
    relevance_threshold
        Minimum relevance score to consider a document as positive.

    Returns
    -------
    pos_scores : List[float]
        Token-level scores from relevant documents.
    neg_scores : List[float]
        Token-level scores from non-relevant documents.
    """
    pos_scores = []
    neg_scores = []

    # For each query token
    for doc_ids_arr, scores_arr in zip(token_doc_ids, token_scores):
        # For each retrieved document token (top-k)
        for doc_id, score in zip(doc_ids_arr, scores_arr):
            # Check if document is relevant
            is_relevant = doc_id in qrels_for_query and qrels_for_query[doc_id] >= relevance_threshold

            if is_relevant:
                pos_scores.append(float(score))
            else:
                neg_scores.append(float(score))

    return pos_scores, neg_scores


def collect_token_scores(
    index: indexes.Base,
    queries: Dict[str, str],
    queries_embeddings: List[torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    k_token: int = 1000,
    batch_size: int = 32,
    relevance_threshold: int = 1,
    sample_queries: Optional[int] = None,
    verbose: bool = True,
) -> TokenScoreData:
    """
    Collect token-level similarity scores from index retrieval results.

    For each query:
    1. Retrieve top-k_token tokens per query token using the index
    2. Extract scores directly from index results
    3. Segment scores into positive (relevant) and negative (non-relevant) based on qrels

    Parameters
    ----------
    index
        The pre-built index to query.
    queries
        Mapping from query_id -> query text.
    queries_embeddings
        List of query embeddings (one tensor per query).
    qrels
        Mapping from query_id -> {doc_id -> relevance}.
    k_token
        Number of top tokens to retrieve per query token.
    batch_size
        Batch size for index querying.
    relevance_threshold
        Minimum relevance score to consider a document as positive.
    sample_queries
        If set, randomly sample this many queries (for faster analysis).
    verbose
        Whether to show progress bar.

    Returns
    -------
    TokenScoreData
        Container with positive/negative token score distributions.
    """
    all_pos_scores = []
    all_neg_scores = []
    num_pos_tokens = 0
    num_neg_tokens = 0

    query_ids = list(queries.keys())

    # Sample queries if requested
    if sample_queries is not None and sample_queries < len(query_ids):
        logger.info("Sampling %d out of %d queries", sample_queries, len(query_ids))
        import random
        random.seed(42)  # For reproducibility
        sampled_indices = random.sample(range(len(query_ids)), sample_queries)
        query_ids = [query_ids[i] for i in sampled_indices]
        queries_embeddings = [queries_embeddings[i] for i in sampled_indices]
        logger.info("Sampled query IDs: %s", query_ids[:10])

    # Process in batches
    for batch_start in tqdm(range(0, len(queries_embeddings), batch_size),
                           desc="Retrieving tokens", disable=not verbose):
        batch_end = min(batch_start + batch_size, len(queries_embeddings))
        batch_query_embeddings = queries_embeddings[batch_start:batch_end]
        batch_query_ids = query_ids[batch_start:batch_end]

        # Query the index (returns doc IDs and scores for each token)
        index_results = index(batch_query_embeddings, k=k_token)

        # For each query in batch
        for q_idx, query_id in enumerate(batch_query_ids):
            # Get qrels for this query
            qrels_for_query = qrels.get(query_id, {})

            # Extract token-level doc IDs and scores
            token_doc_ids = index_results["documents_ids"][q_idx]  # (num_query_tokens, k_token)
            token_distances = index_results["distances"][q_idx]     # (num_query_tokens, k_token)

            # Convert distances to similarity scores (ScaNN returns squared L2 distances for inner product)
            # For cosine similarity with normalized embeddings, distance = 2 * (1 - cosine_sim)
            # So: cosine_sim = 1 - distance/2
            # But actually, ScaNN configured for max inner product returns negative distances
            # Let's just use the distances as-is (higher = more similar)
            token_scores = token_distances  # Keep as-is for now

            # Extract and segment scores
            pos_scores, neg_scores = extract_token_scores_from_retrieval(
                query_id=query_id,
                token_doc_ids=token_doc_ids,
                token_scores=token_scores,
                qrels_for_query=qrels_for_query,
                relevance_threshold=relevance_threshold,
            )

            all_pos_scores.extend(pos_scores)
            all_neg_scores.extend(neg_scores)
            num_pos_tokens += len(pos_scores)
            num_neg_tokens += len(neg_scores)

    # Convert to numpy arrays
    pos_scores_arr = np.array(all_pos_scores, dtype=np.float32)
    neg_scores_arr = np.array(all_neg_scores, dtype=np.float32)
    all_scores_arr = np.concatenate([pos_scores_arr, neg_scores_arr])

    return TokenScoreData(
        model_name="",  # Will be set by caller
        dataset_name="",  # Will be set by caller
        all_scores=all_scores_arr,
        pos_scores=pos_scores_arr,
        neg_scores=neg_scores_arr,
        num_queries=len(query_ids),
        num_pos_tokens=num_pos_tokens,
        num_neg_tokens=num_neg_tokens,
    )


def plot_single_model_distribution(
    data: TokenScoreData,
    cfg: DictConfig,
    output_path: Path,
) -> None:
    """
    Plot token score distribution for a single model (pos vs neg).
    Similar to Figure 2 in the XTR paper.
    """
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize_single))

    # Sample if too many points (for performance)
    max_points = 500_000
    pos_scores = data.pos_scores
    neg_scores = data.neg_scores

    if len(pos_scores) > max_points:
        pos_scores = np.random.choice(pos_scores, max_points, replace=False)
    if len(neg_scores) > max_points:
        neg_scores = np.random.choice(neg_scores, max_points, replace=False)

    if cfg.plot.use_kde:
        # KDE plot
        if len(pos_scores) > 0:
            sns.kdeplot(
                pos_scores,
                ax=ax,
                label=f"Positive (n={data.num_pos_tokens:,})",
                color=cfg.plot.pos_color,
                fill=True,
                alpha=cfg.plot.alpha,
                bw_adjust=cfg.plot.kde_bw_adjust,
            )
        if len(neg_scores) > 0:
            sns.kdeplot(
                neg_scores,
                ax=ax,
                label=f"Negative (n={data.num_neg_tokens:,})",
                color=cfg.plot.neg_color,
                fill=True,
                alpha=cfg.plot.alpha,
                bw_adjust=cfg.plot.kde_bw_adjust,
            )
    else:
        # Histogram
        if len(pos_scores) > 0:
            ax.hist(
                pos_scores,
                bins=cfg.plot.bins,
                alpha=cfg.plot.alpha,
                label=f"Positive (n={data.num_pos_tokens:,})",
                color=cfg.plot.pos_color,
                density=True,
            )
        if len(neg_scores) > 0:
            ax.hist(
                neg_scores,
                bins=cfg.plot.bins,
                alpha=cfg.plot.alpha,
                label=f"Negative (n={data.num_neg_tokens:,})",
                color=cfg.plot.neg_color,
                density=True,
            )

    ax.set_xlabel("Token Similarity Score (Cosine)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Token Score Distribution: {data.model_name}\n({data.dataset_name})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.plot.dpi, format=cfg.plot.format, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", output_path)


def plot_overall_distribution(
    data: TokenScoreData,
    cfg: DictConfig,
    output_path: Path,
) -> None:
    """Plot overall token score distribution (without pos/neg segmentation)."""
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize_single))

    # Sample if too many points
    max_points = 500_000
    all_scores = data.all_scores
    if len(all_scores) > max_points:
        all_scores = np.random.choice(all_scores, max_points, replace=False)

    if cfg.plot.use_kde:
        sns.kdeplot(
            all_scores,
            ax=ax,
            label=f"All tokens (n={len(data.all_scores):,})",
            color="#3498db",
            fill=True,
            alpha=cfg.plot.alpha,
            bw_adjust=cfg.plot.kde_bw_adjust,
        )
    else:
        ax.hist(
            all_scores,
            bins=cfg.plot.bins,
            alpha=cfg.plot.alpha,
            label=f"All tokens (n={len(data.all_scores):,})",
            color="#3498db",
            density=True,
        )

    ax.set_xlabel("Token Similarity Score (Cosine)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"Overall Token Score Distribution: {data.model_name}\n({data.dataset_name})", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.plot.dpi, format=cfg.plot.format, bbox_inches='tight')
    plt.close()
    logger.info("Saved figure: %s", output_path)


def plot_model_comparison(
    all_data: List[TokenScoreData],
    cfg: DictConfig,
    output_path: Path,
    title_suffix: str = "",
) -> None:
    """
    Plot comparison of token score distributions across multiple models.
    Creates a figure with one subplot per model.
    """
    n_models = len(all_data)
    fig, axes = plt.subplots(1, n_models, figsize=(8 * n_models, 6))

    if n_models == 1:
        axes = [axes]

    max_points = 500_000

    for ax, data in zip(axes, all_data):
        pos_scores = data.pos_scores
        neg_scores = data.neg_scores

        if len(pos_scores) > max_points:
            pos_scores = np.random.choice(pos_scores, max_points, replace=False)
        if len(neg_scores) > max_points:
            neg_scores = np.random.choice(neg_scores, max_points, replace=False)

        if cfg.plot.use_kde:
            if len(pos_scores) > 0:
                sns.kdeplot(
                    pos_scores,
                    ax=ax,
                    label=f"Positive",
                    color=cfg.plot.pos_color,
                    fill=True,
                    alpha=cfg.plot.alpha,
                    bw_adjust=cfg.plot.kde_bw_adjust,
                )
            if len(neg_scores) > 0:
                sns.kdeplot(
                    neg_scores,
                    ax=ax,
                    label=f"Negative",
                    color=cfg.plot.neg_color,
                    fill=True,
                    alpha=cfg.plot.alpha,
                    bw_adjust=cfg.plot.kde_bw_adjust,
                )
        else:
            if len(pos_scores) > 0:
                ax.hist(pos_scores, bins=cfg.plot.bins, alpha=cfg.plot.alpha,
                       label="Positive", color=cfg.plot.pos_color, density=True)
            if len(neg_scores) > 0:
                ax.hist(neg_scores, bins=cfg.plot.bins, alpha=cfg.plot.alpha,
                       label="Negative", color=cfg.plot.neg_color, density=True)

        # Shorten model name for display
        short_name = sanitize_model_name(data.model_name)
        ax.set_xlabel("Token Similarity Score", fontsize=11)
        ax.set_ylabel("Density", fontsize=11)
        ax.set_title(f"{short_name}", fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)

    fig.suptitle(f"Token Score Distribution Comparison{title_suffix}", fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.plot.dpi, format=cfg.plot.format, bbox_inches='tight')
    plt.close()
    logger.info("Saved comparison figure: %s", output_path)


def plot_overlay_comparison(
    all_data: List[TokenScoreData],
    cfg: DictConfig,
    output_path: Path,
    score_type: str = "pos",  # "pos", "neg", or "all"
) -> None:
    """
    Plot overlaid distributions from multiple models on the same axes.
    """
    fig, ax = plt.subplots(figsize=tuple(cfg.plot.figsize_single))

    colors = plt.cm.tab10(np.linspace(0, 1, len(all_data)))
    max_points = 500_000

    for data, color in zip(all_data, colors):
        if score_type == "pos":
            scores = data.pos_scores
            label_suffix = " (pos)"
        elif score_type == "neg":
            scores = data.neg_scores
            label_suffix = " (neg)"
        else:
            scores = data.all_scores
            label_suffix = ""

        if len(scores) > max_points:
            scores = np.random.choice(scores, max_points, replace=False)

        short_name = sanitize_model_name(data.model_name)

        if cfg.plot.use_kde:
            sns.kdeplot(
                scores,
                ax=ax,
                label=f"{short_name}{label_suffix}",
                color=color,
                fill=True,
                alpha=0.4,
                bw_adjust=cfg.plot.kde_bw_adjust,
            )
        else:
            ax.hist(
                scores,
                bins=cfg.plot.bins,
                alpha=0.4,
                label=f"{short_name}{label_suffix}",
                color=color,
                density=True,
            )

    score_type_label = {"pos": "Positive", "neg": "Negative", "all": "All"}[score_type]
    ax.set_xlabel("Token Similarity Score (Cosine)", fontsize=12)
    ax.set_ylabel("Density", fontsize=12)
    ax.set_title(f"{score_type_label} Token Scores: Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=cfg.plot.dpi, format=cfg.plot.format, bbox_inches='tight')
    plt.close()
    logger.info("Saved overlay figure: %s", output_path)


def save_score_data(
    data: TokenScoreData,
    output_path: Path,
) -> None:
    """Save token score data for later analysis."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Save as compressed numpy
    np.savez_compressed(
        output_path,
        all_scores=data.all_scores,
        pos_scores=data.pos_scores,
        neg_scores=data.neg_scores,
    )

    # Save metadata as JSON
    metadata = {
        "model_name": data.model_name,
        "dataset_name": data.dataset_name,
        "num_queries": data.num_queries,
        "num_pos_tokens": data.num_pos_tokens,
        "num_neg_tokens": data.num_neg_tokens,
        "created_at": datetime.now().isoformat(),
    }
    metadata_path = output_path.with_suffix(".json")
    metadata_path.write_text(json.dumps(metadata, indent=2))

    logger.info("Saved score data: %s", output_path)


@hydra.main(version_base=None, config_path="conf/eval", config_name="experiment_3_token_scores")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    # Expand model paths (handle globs)
    model_paths = expand_model_paths(cfg.model.name_or_path)
    dataset_names = list(cfg.dataset.names)
    embedding_dtype = get_torch_dtype(cfg.cache.embedding_dtype)

    logger.info("Models to analyze: %s", model_paths)
    logger.info("Datasets: %s", dataset_names)

    # Create output directories
    figures_dir = Path(cfg.output.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    if cfg.output.save_data:
        data_dir = Path(cfg.output.data_dir)
        data_dir.mkdir(parents=True, exist_ok=True)

    # Process each dataset
    for dataset_id in dataset_names:
        logger.info("\n" + "=" * 80)
        logger.info("Processing dataset: %s", dataset_id)
        logger.info("=" * 80)

        dataset_slug = sanitize_dataset_name(dataset_id)

        # Load dataset once (shared across models)
        documents, queries, qrels = load_dataset(dataset_id, lowercase=cfg.dataset.lowercase)

        # Collect results for all models on this dataset
        all_model_data: List[TokenScoreData] = []

        for model_name in model_paths:
            logger.info("\n" + "-" * 60)
            logger.info("Processing model: %s", model_name)
            logger.info("-" * 60)

            model_slug = sanitize_model_name(model_name)
            query_length = resolve_query_length(dataset_id, cfg.model.query_len)
            doc_length = cfg.model.doc_len

            # Build model (for query encoding only)
            model = build_model(cfg, model_name, query_length, doc_length)
            embedding_size = get_embedding_size(model)

            # Build cache paths
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
                doc_length=doc_length,
                lowercase=cfg.dataset.lowercase,
            )

            if not index_configs:
                raise ValueError("No index configuration found. Check your config.")

            # Load existing index (fail if not found)
            index_config = index_configs[0]
            logger.info("Loading index: %s", index_config["name"])
            logger.info("Index init kwargs: %s", index_config["init_kwargs"])

            index: indexes.Base = index_config["index_class"](**index_config["init_kwargs"])

            if not getattr(index, "_documents_added", False):
                raise ValueError(
                    f"Index not found or empty. Please run eval_model_irds_v3.py first to build the index.\n"
                    f"Expected index at: {index_config['init_kwargs'].get('index_folder')}/{index_config['init_kwargs'].get('name')}"
                )

            logger.info("Index loaded successfully with %d documents", len(index.doc_id_to_embedding_range))

            # Encode queries (or load from cache)
            logger.info("Encoding queries...")
            queries_embeddings = encode_queries_with_cache(
                model=model,
                queries=queries,
                batch_size=cfg.encode.batch_size,
                embedding_dtype=embedding_dtype,
                move_to_cpu=cfg.encode.move_embeddings_to_cpu,
                cache_paths=cache_paths,
                cache_enabled=cfg.cache.enable_queries,
            )

            # Collect token scores from retrieval
            logger.info("Collecting token scores from retrieval...")
            score_data = collect_token_scores(
                index=index,
                queries=queries,
                queries_embeddings=queries_embeddings,
                qrels=qrels,
                k_token=cfg.retrieve.k_token,
                batch_size=cfg.retrieve.batch_size,
                relevance_threshold=cfg.analysis.get("relevance_threshold", 1),
                sample_queries=cfg.analysis.get("sample_queries", None),
                verbose=True,
            )

            # Set metadata
            score_data.model_name = model_name
            score_data.dataset_name = dataset_id

            # Log statistics
            logger.info("Token score statistics:")
            logger.info("  - Queries: %d", score_data.num_queries)
            logger.info("  - Positive tokens: %d", score_data.num_pos_tokens)
            logger.info("  - Negative tokens: %d", score_data.num_neg_tokens)
            if len(score_data.pos_scores) > 0:
                logger.info("  - Positive score mean: %.4f, std: %.4f",
                           score_data.pos_scores.mean(), score_data.pos_scores.std())
            if len(score_data.neg_scores) > 0:
                logger.info("  - Negative score mean: %.4f, std: %.4f",
                           score_data.neg_scores.mean(), score_data.neg_scores.std())

            all_model_data.append(score_data)

            # Save individual model plots
            dataset_fig_dir = figures_dir / dataset_slug
            dataset_fig_dir.mkdir(parents=True, exist_ok=True)

            # Overall distribution
            plot_overall_distribution(
                score_data,
                cfg,
                dataset_fig_dir / f"{model_slug}_overall.{cfg.plot.format}",
            )

            # Pos/neg segmented distribution
            plot_single_model_distribution(
                score_data,
                cfg,
                dataset_fig_dir / f"{model_slug}_pos_neg.{cfg.plot.format}",
            )

            # Save raw data
            if cfg.output.save_data:
                save_score_data(
                    score_data,
                    data_dir / dataset_slug / f"{model_slug}_scores.npz",
                )

            # Free memory
            del model
            torch.cuda.empty_cache() if torch.cuda.is_available() else None

        # Plot model comparisons for this dataset
        if len(all_model_data) > 1:
            logger.info("\nGenerating comparison plots...")

            # Side-by-side comparison
            plot_model_comparison(
                all_model_data,
                cfg,
                figures_dir / dataset_slug / f"comparison_pos_neg.{cfg.plot.format}",
                title_suffix=f"\n({dataset_id})",
            )

            # Overlay plots
            plot_overlay_comparison(
                all_model_data,
                cfg,
                figures_dir / dataset_slug / f"overlay_pos.{cfg.plot.format}",
                score_type="pos",
            )
            plot_overlay_comparison(
                all_model_data,
                cfg,
                figures_dir / dataset_slug / f"overlay_neg.{cfg.plot.format}",
                score_type="neg",
            )
            plot_overlay_comparison(
                all_model_data,
                cfg,
                figures_dir / dataset_slug / f"overlay_all.{cfg.plot.format}",
                score_type="all",
            )

    logger.info("\n" + "=" * 80)
    logger.info("Analysis complete!")
    logger.info("Figures saved to: %s", figures_dir)
    if cfg.output.save_data:
        logger.info("Data saved to: %s", data_dir)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
