#!/usr/bin/env python3
"""Measure (an)isotropy of ColBERT models.

Encodes a fixed random sample of N documents (identical for every model in the list),
then computes two complementary metrics on the resulting token embeddings:

  avg_cosine_sim      -- Mean cosine similarity between random token-vector pairs.
                         0 = perfectly isotropic, 1 = fully anisotropic.
  effective_rank      -- exp(H) where H = Shannon entropy of the normalized singular-
                         value distribution (Roy & Vetterli 2007).  Range [1, d].
  participation_ratio -- (Σσ_i)² / (d · Σσ_i²).  Range (0, 1].

Singular values are computed exactly via the d×d covariance matrix (E^T E),
which is fast even for large N because d ≤ 128 for most ColBERT models.

Both metrics are reported twice: over all token vectors and with per-doc CLS excluded.
Token IDs are saved so token-type breakdown can be done in post-processing.

Caches are keyed per {dataset}/{model}/{n_docs}_{seed} so re-runs reuse embeddings.

Usage:
    python scripts/measure_anisotropy.py                                # defaults (all models)
    python scripts/measure_anisotropy.py dataset=beir/nfcorpus/test    # swap dataset
    python scripts/measure_anisotropy.py sample.n_docs=2000 sample.seed=7
    python scripts/measure_anisotropy.py 'models=[robro612/ModernBERT-XTR]'  # ad-hoc subset
"""

from __future__ import annotations

import datetime
import gc
import logging
import random
import sys
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).parent))
from benchmark_indexes import (
    append_jsonl,
    encode_documents_sharded,
    get_hardware_info,
    load_dataset,
    load_shards_flat,
    sanitize_name,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def sample_documents(documents: list[dict], n: int, seed: int) -> list[dict]:
    """Return n documents drawn without replacement at a fixed seed."""
    if n >= len(documents):
        return list(documents)
    rng = random.Random(seed)
    indices = sorted(rng.sample(range(len(documents)), n))
    return [documents[i] for i in indices]


def _singular_values(E_norm: np.ndarray) -> np.ndarray:
    """Return exact singular values of E_norm via the d×d covariance matrix.

    For ColBERT d ≤ 128, so E^T E is tiny even when N is large.
    Returns values in descending order.
    """
    cov = E_norm.T @ E_norm  # (d, d)
    eigenvalues = np.linalg.eigvalsh(cov)  # ascending
    return np.sqrt(np.maximum(eigenvalues[::-1], 0.0))  # descending


def compute_anisotropy(
    flat_embs: np.ndarray,
    flat_doclens: np.ndarray,
    n_pairs: int,
    include_cls: bool,
    pair_seed: int = 0,
) -> dict:
    """Compute anisotropy metrics on a flat (N, d) embedding matrix.

    Args:
        flat_embs:    (N, d) float array of concatenated token vectors.
        flat_doclens: (num_docs,) token counts per document.
        n_pairs:      random pairs to sample for avg cosine similarity.
        include_cls:  if False, mask out the first token of every document.
        pair_seed:    seed for reproducible pair sampling.
    """
    rng = np.random.default_rng(pair_seed)
    E = flat_embs.astype(np.float32)

    if not include_cls:
        mask = np.ones(len(E), dtype=bool)
        offset = 0
        for dl in flat_doclens:
            mask[offset] = False
            offset += int(dl)
        E = E[mask]

    n_tokens, emb_dim = E.shape

    # L2-normalize row-wise
    norms = np.linalg.norm(E, axis=1, keepdims=True)
    E_norm = E / np.maximum(norms, 1e-8)

    # avg cosine similarity on random pairs
    actual_pairs = min(n_pairs, n_tokens // 2)
    idx = rng.choice(n_tokens, size=actual_pairs * 2, replace=False)
    avg_cos = float(
        (E_norm[idx[:actual_pairs]] * E_norm[idx[actual_pairs:]]).sum(axis=1).mean()
    )

    # singular-value-based metrics
    S = _singular_values(E_norm)
    s_sum = S.sum() + 1e-12

    # Roy & Vetterli (2007) effective rank = exp(Shannon entropy of p_i = σ_i / Σσ)
    p = S / s_sum
    log_p = np.where(p > 1e-15, np.log(p), 0.0)
    effective_rank = float(np.exp(-(p * log_p).sum()))

    # Participation ratio = (Σσ)² / (d · Σσ²)
    participation_ratio = float((S.sum() ** 2) / (emb_dim * (S ** 2).sum()))

    return {
        "n_tokens": n_tokens,
        "emb_dim": emb_dim,
        "avg_cosine_sim": round(avg_cos, 6),
        "effective_rank": round(effective_rank, 3),
        "participation_ratio": round(participation_ratio, 6),
        "n_pairs": actual_pairs,
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf/anisotropy", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    n_docs = cfg.sample.n_docs
    seed = cfg.sample.seed

    # Load corpus once; the sample is identical for every model in the list
    documents, _, _ = load_dataset(cfg.dataset)
    sampled_docs = sample_documents(documents, n_docs, seed)
    logger.info(
        "Sampled %d / %d documents (seed=%d)",
        len(sampled_docs), len(documents), seed,
    )
    del documents
    gc.collect()

    dataset_slug = sanitize_name(cfg.dataset)
    sample_tag = f"n{len(sampled_docs)}_s{seed}"
    hw = get_hardware_info()

    for model_name in cfg.models:
        if not model_name:
            continue
        model_slug = sanitize_name(model_name)
        # Cache keyed by model and sample params — safe to reuse across runs
        cache_dir = Path(cfg.cache.dir) / dataset_slug / model_slug / sample_tag

        logger.info("Encoding %s → %s", model_name, cache_dir)
        # cfg supplies all keys encode_documents_sharded expects:
        # doc_length, encode.*, model.model_kwargs, compile
        shard_dir, encode_time = encode_documents_sharded(
            model_name=model_name,
            documents=sampled_docs,
            cache_dir=cache_dir,
            cfg=cfg,
        )

        logger.info("Loading embeddings from %s", shard_dir)
        flat_embs, flat_doclens, _ = load_shards_flat(shard_dir)
        logger.info(
            "  %d tokens across %d docs, dim=%d",
            len(flat_embs), len(flat_doclens), flat_embs.shape[1],
        )

        stats_all = compute_anisotropy(
            flat_embs, flat_doclens, n_pairs=cfg.n_pairs, include_cls=True,
        )
        stats_no_cls = compute_anisotropy(
            flat_embs, flat_doclens, n_pairs=cfg.n_pairs, include_cls=False,
        )

        row = {
            "timestamp": datetime.datetime.now().isoformat(),
            "model": model_name,
            "dataset": cfg.dataset,
            "n_docs_sampled": len(sampled_docs),
            "sample_seed": seed,
            "doc_length": cfg.doc_length,
            "encode_time_s": round(encode_time, 2),
            **{f"all_{k}": v for k, v in stats_all.items()},
            **{f"no_cls_{k}": v for k, v in stats_no_cls.items()},
            **hw,
        }
        append_jsonl(cfg.output.results_file, row)

        print(f"\n{'='*60}")
        print(f"Model: {model_name}")
        print(
            f"  All tokens:  "
            f"avg_cos={stats_all['avg_cosine_sim']:.4f}  "
            f"eff_rank={stats_all['effective_rank']:.1f}/{stats_all['emb_dim']}  "
            f"PR={stats_all['participation_ratio']:.4f}"
        )
        print(
            f"  No CLS:      "
            f"avg_cos={stats_no_cls['avg_cosine_sim']:.4f}  "
            f"eff_rank={stats_no_cls['effective_rank']:.1f}/{stats_no_cls['emb_dim']}  "
            f"PR={stats_no_cls['participation_ratio']:.4f}"
        )

        del flat_embs, flat_doclens
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nResults appended to {cfg.output.results_file}")


if __name__ == "__main__":
    main()
