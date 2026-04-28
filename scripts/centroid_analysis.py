"""Analyse why PLAID search is slower for ColBERT vs XTR embeddings.

Simulates PLAID's candidate generation: for each query, finds top-nprobe
centroids per query token, maps those centroids to documents, and counts
how many unique candidate documents need MaxSim scoring.

Usage:
    python scripts/centroid_analysis.py <dataset_slug> <model_slugs...> [--n-centroids N] [--nprobes 2,4,8]

Example:
    python scripts/centroid_analysis.py beir_fiqa_test \
        output_modernbert_colbert_kd_final \
        output_modernbert_xtr_kd_k256_final
"""

import argparse
import math
from pathlib import Path

import numpy as np
import torch
from flash_kmeans import FlashKMeans


CACHE_BASE = Path("embeddings_cache")
ENCODING_SUBDIR = "nopool_fp16"


def plaid_n_centroids(n_docs: int) -> int:
    """Same formula as benchmark_indexes.py / xtr-warp for n_samples_kmeans."""
    return min(1 + int(16 * math.sqrt(120 * n_docs)), n_docs)


def load_embeddings(cache_base: Path, dataset_slug: str, model_slug: str):
    """Load doc and query embeddings from the cache."""
    model_dir = cache_base / dataset_slug / model_slug / ENCODING_SUBDIR

    docs_dir = model_dir / "docs"
    shard_files = sorted(docs_dir.glob("doc_shard_*.npy"))
    shard_files = [f for f in shard_files if "doclens" not in f.name and "token_ids" not in f.name]
    doc_embs = np.concatenate([np.load(f) for f in shard_files], axis=0).astype(np.float32)
    doc_doclens = np.concatenate([
        np.load(f.with_suffix("").with_suffix(".doclens.npy"))
        for f in shard_files
    ])

    q_dir = model_dir / "queries"
    query_embs = np.load(q_dir / "query_emb.npy").astype(np.float32)
    query_doclens = np.load(q_dir / "query_emb.doclens.npy")

    return doc_embs, doc_doclens, query_embs, query_doclens


def build_token_to_doc(doc_doclens: np.ndarray) -> np.ndarray:
    """Map each token index to its document index."""
    return np.repeat(np.arange(len(doc_doclens)), doc_doclens)


def analyse_model(
    dataset_slug: str,
    model_slug: str,
    n_centroids: int | None,
    nprobes: list[int],
):
    """Run the analysis for one model and return a stats dict."""
    print(f"\n{'='*60}")
    print(f"Model: {model_slug}")
    print(f"{'='*60}")

    doc_embs, doc_doclens, query_embs, query_doclens = load_embeddings(
        CACHE_BASE, dataset_slug, model_slug,
    )
    n_docs = len(doc_doclens)
    n_queries = len(query_doclens)
    print(f"  Docs: {n_docs:,} ({doc_embs.shape[0]:,} tokens, "
          f"mean {doc_doclens.mean():.1f} tok/doc)")
    print(f"  Queries: {n_queries:,} ({query_embs.shape[0]:,} tokens, "
          f"mean {query_doclens.mean():.1f} tok/query)")

    # k-means
    actual_k = n_centroids if n_centroids is not None else plaid_n_centroids(n_docs)
    actual_k = min(actual_k, doc_embs.shape[0])
    print(f"  Fitting k-means (k={actual_k})...")
    doc_embs_gpu = torch.from_numpy(doc_embs).cuda()
    kmeans = FlashKMeans(d=doc_embs.shape[1], k=actual_k, niter=4, seed=42)
    kmeans.fit(doc_embs_gpu)
    centroids = kmeans.centroids_b.squeeze(0).cpu().numpy()  # (k, 128)

    # Assign each doc token to its nearest centroid
    labels = kmeans.predict(doc_embs_gpu).cpu().numpy()  # (n_tokens,)
    del doc_embs_gpu

    # Map: token -> doc
    token_to_doc = build_token_to_doc(doc_doclens)

    # Build inverted index: centroid_id -> set of doc_ids
    print("  Building centroid -> doc inverted index...")
    centroid_to_docs = [set() for _ in range(actual_k)]
    for tok_idx, cid in enumerate(labels):
        centroid_to_docs[cid].add(token_to_doc[tok_idx])
    centroid_doc_counts = np.array([len(s) for s in centroid_to_docs])
    print(f"  Docs per centroid: mean={centroid_doc_counts.mean():.1f}  "
          f"median={np.median(centroid_doc_counts):.0f}  "
          f"max={centroid_doc_counts.max()}")

    # Normalise centroids for cosine scoring
    centroid_norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    centroids_normed = centroids / np.maximum(centroid_norms, 1e-8)

    # For each query, simulate PLAID candidate generation:
    #   1. For each query token, find top-nprobe centroids
    #   2. Union all docs in those centroids across all query tokens
    #   3. Count unique candidate docs
    stats = {}
    for nprobe in nprobes:
        print(f"\n  --- nprobe={nprobe} ---")
        candidate_counts = []
        q_offset = 0
        for qi in range(n_queries):
            q_len = query_doclens[qi]
            q_toks = query_embs[q_offset:q_offset + q_len]  # (q_len, 128)
            q_offset += q_len

            # Score query tokens against centroids
            q_scores = q_toks @ centroids_normed.T  # (q_len, k)
            # Top-nprobe centroids per query token
            top_centroids = np.argpartition(q_scores, -nprobe, axis=1)[:, -nprobe:]

            # Union of docs across all query tokens
            candidate_docs = set()
            for tok_top in top_centroids:
                for cid in tok_top:
                    candidate_docs.update(centroid_to_docs[cid])
            candidate_counts.append(len(candidate_docs))

        candidate_counts = np.array(candidate_counts)
        pct = candidate_counts / n_docs * 100
        print(f"  Candidate docs per query:")
        print(f"    mean={candidate_counts.mean():.0f} ({pct.mean():.1f}% of corpus)  "
              f"median={np.median(candidate_counts):.0f}  "
              f"p95={np.percentile(candidate_counts, 95):.0f}  "
              f"max={candidate_counts.max()}")

        stats[f"candidates_mean_np{nprobe}"] = candidate_counts.mean()
        stats[f"candidates_pct_np{nprobe}"] = pct.mean()
        stats[f"candidates_p95_np{nprobe}"] = np.percentile(candidate_counts, 95)

    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_slug", help="e.g. beir_fiqa_test")
    parser.add_argument("models", nargs="+", help="model slugs to compare")
    parser.add_argument("--n-centroids", type=int, default=None,
                        help="Override centroid count (default: PLAID formula)")
    parser.add_argument("--nprobes", type=str, default="2,4,8,16,32",
                        help="Comma-separated nprobe values to test (default: 2,4,8,16,32)")
    args = parser.parse_args()

    nprobes = [int(x) for x in args.nprobes.split(",")]

    all_stats = {}
    for model in args.models:
        all_stats[model] = analyse_model(
            args.dataset_slug, model, args.n_centroids, nprobes,
        )

    # Summary comparison
    if len(args.models) >= 2:
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY")
        print(f"{'='*60}")

        # Shorter labels from model slugs
        labels = []
        for m in args.models:
            parts = m.split("_")
            # e.g. output_modernbert_colbert_kd_final -> colbert_kd
            label = "_".join(parts[2:-1])
            labels.append(label)

        header = f"{'Metric':<30s}" + "".join(f"  {l:>20s}" for l in labels)
        print(header)
        print("-" * len(header))
        for key in all_stats[args.models[0]]:
            row = f"{key:<30s}"
            for m in args.models:
                v = all_stats[m][key]
                row += f"  {v:>20.1f}"
            print(row)


if __name__ == "__main__":
    main()
