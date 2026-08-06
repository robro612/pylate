#!/usr/bin/env python3
"""Isolated coarse-clustering latency: TAC vs PGC.

The benchmark's ``build_time_s`` folds clustering together with PQ + HNSW
construction, so it cannot separate the cost of the coarse-centroid step (the
only part that differs between TAC and PGC).  This script times *only* that
step by calling the ``tachiom.time_tac_clustering`` / ``time_pgc_clustering``
bindings, which reproduce the exact clustering call made by the real build
path (``build_from_arrays`` / ``build_with_pgc``) but skip PQ/HNSW.  The
numpy->Rust copy happens outside the timer.

Ecological validity:
  * Uses already-encoded document shards (the same ``doc_shard_*.npy`` the
    benchmark built its indexes from) -- no synthetic/random vectors.  Real
    ColBERT/XTR token embeddings are strongly anisotropic and live near a
    low-dimensional manifold; PGC's HNSW-based assignment and TAC's per-token
    k-means both behave very differently on isotropic random data, so random
    inputs would not be representative.
  * ``total_centroids=None`` lets the Rust resolver pick the identical centroid
    budget the real build used (e.g. 50000 for nfcorpus).
  * Defaults below match the clustering configs recorded in results.jsonl
    (n_iter=5, PGC empty_strategy="split", iter_hnsw_m=16, etc.).

Usage:
    # one dataset, append a row per (method, repeat) to JSONL
    python scripts/measure_clustering_latency.py \
        --shard-dir embeddings_cache/beir_nfcorpus_test/lightonai_GTE-ModernColBERT-v1/nopool_fp16/docs \
        --dataset beir/nfcorpus/test --repeats 3

    # several datasets in one go
    python scripts/measure_clustering_latency.py \
        --shard-dir <dir1> <dir2> ... --repeats 3 --out results/clustering_latency.jsonl

Run under srun (GPU not required, but use a real compute node, not login):
    srunl40s python scripts/measure_clustering_latency.py ...
"""

from __future__ import annotations

import argparse
import datetime
import json
import platform
from pathlib import Path

import numpy as np

# Defaults mirror the clustering subconfig used in results.jsonl.
TAC_DEFAULTS = dict(tac_n_iter=5)
PGC_DEFAULTS = dict(
    pgc_n_iter=5,
    pgc_sample_multiplier=2**63 - 1,  # null multiplier -> no sampling cap
    pgc_empty_strategy="split",
    pgc_iter_hnsw_m=16,
    pgc_iter_ef_construction=200,
    pgc_iter_ef_search=50,
    pgc_seed=42,
)


def load_shards(shard_dir: Path, glob_pattern: str = "doc_shard_*.npy"):
    """Load encoded shards into flat in-memory arrays.

    Mirrors ``TachiomIndex.add_documents_from_shards``: a doclens pass to
    pre-allocate, then a fill pass.  Returns (vectors_u16[N,dim], token_ids_u32,
    doclens_i32, n_docs).
    """
    vec_paths = sorted(shard_dir.glob(glob_pattern))
    vec_paths = [
        p
        for p in vec_paths
        if not any(
            p.name.endswith(s)
            for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy")
        )
    ]
    if not vec_paths:
        raise FileNotFoundError(f"No embedding shards matching {glob_pattern!r} in {shard_dir}")

    doclens_paths = [p.parent / p.name.replace(".npy", ".doclens.npy") for p in vec_paths]
    token_ids_paths = [p.parent / p.name.replace(".npy", ".token_ids.npy") for p in vec_paths]

    all_doclens = [np.load(str(p)) for p in doclens_paths]
    shard_tok_counts = [int(d.sum()) for d in all_doclens]
    total_tokens = sum(shard_tok_counts)
    total_docs = sum(len(d) for d in all_doclens)
    dim = np.load(str(vec_paths[0]), mmap_mode="r").shape[1]

    flat_vecs = np.empty((total_tokens, dim), dtype=np.float16)
    flat_tids = np.zeros(total_tokens, dtype=np.uint32)
    flat_doclens = np.empty(total_docs, dtype=np.int32)
    have_tids = all(p.exists() for p in token_ids_paths)

    tok_off = 0
    doc_off = 0
    for i, (vp, tp) in enumerate(zip(vec_paths, token_ids_paths)):
        n_tok = shard_tok_counts[i]
        n_doc = len(all_doclens[i])
        flat_vecs[tok_off : tok_off + n_tok] = np.load(str(vp), mmap_mode="r")
        flat_doclens[doc_off : doc_off + n_doc] = all_doclens[i]
        if have_tids:
            flat_tids[tok_off : tok_off + n_tok] = np.load(str(tp), mmap_mode="r")
        tok_off += n_tok
        doc_off += n_doc

    if not have_tids:
        raise FileNotFoundError(
            f"token_ids shards missing in {shard_dir}; TAC needs them and the "
            "centroid budget is derived from them."
        )

    vectors_u16 = np.ascontiguousarray(flat_vecs).view(np.uint16)
    token_ids = np.ascontiguousarray(flat_tids, dtype=np.uint32)
    doclens = np.ascontiguousarray(flat_doclens, dtype=np.int32)
    return vectors_u16, token_ids, doclens, total_docs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--shard-dir", nargs="+", required=True, type=Path,
                    help="One or more directories of doc_shard_*.npy (+ .doclens/.token_ids).")
    ap.add_argument("--dataset", nargs="*", default=None,
                    help="Optional dataset label(s), aligned with --shard-dir (else inferred from path).")
    ap.add_argument("--glob", default="doc_shard_*.npy")
    ap.add_argument("--methods", nargs="+", default=["tac", "pgc"], choices=["tac", "pgc"])
    ap.add_argument("--repeats", type=int, default=3, help="Timed repeats per method.")
    ap.add_argument("--warmup", type=int, default=0, help="Untimed warmup runs per method.")
    ap.add_argument("--total-centroids", type=int, default=None,
                    help="Override resolved centroid budget (default: same as index build).")
    ap.add_argument("--out", type=Path, default=Path("results/clustering_latency.jsonl"))
    args = ap.parse_args()

    import tachiom  # imported here so --help works without the extension

    for fn in ("time_tac_clustering", "time_pgc_clustering"):
        if not hasattr(tachiom, fn):
            raise SystemExit(
                f"tachiom.{fn} not found -- rebuild the local tachiom extension "
                "(maturin develop / uv pip install -e ../tachiom) to pick up the "
                "clustering-timing bindings."
            )

    labels = args.dataset or [None] * len(args.shard_dir)
    if len(labels) != len(args.shard_dir):
        raise SystemExit("--dataset count must match --shard-dir count")

    args.out.parent.mkdir(parents=True, exist_ok=True)
    host = platform.node()

    def run(method, vectors, token_ids, doclens):
        if method == "tac":
            return tachiom.time_tac_clustering(
                vectors, token_ids, doclens,
                total_centroids=args.total_centroids, **TAC_DEFAULTS,
            )
        return tachiom.time_pgc_clustering(
            vectors, token_ids, doclens,
            total_centroids=args.total_centroids, **PGC_DEFAULTS,
        )

    with args.out.open("a") as fout:
        for shard_dir, label in zip(args.shard_dir, labels):
            label = label or shard_dir.as_posix()
            print(f"\n=== {label}  ({shard_dir}) ===")
            vectors, token_ids, doclens, n_docs = load_shards(shard_dir, args.glob)
            print(f"loaded {n_docs} docs, {vectors.shape[0]} tokens, dim={vectors.shape[1]}")

            for method in args.methods:
                params = TAC_DEFAULTS if method == "tac" else PGC_DEFAULTS
                for _ in range(args.warmup):
                    run(method, vectors, token_ids, doclens)
                times = []
                for r in range(args.repeats):
                    res = run(method, vectors, token_ids, doclens)
                    times.append(res["elapsed_s"])
                    rec = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "dataset": label,
                        "shard_dir": shard_dir.as_posix(),
                        "clustering": method,
                        "repeat": r,
                        "n_documents": n_docs,
                        "n_tokens": res["n_tokens"],
                        "dim": res["dim"],
                        "n_centroids": res["n_centroids"],
                        "requested_centroids": res["requested_centroids"],
                        "elapsed_s": round(res["elapsed_s"], 4),
                        "params": params,
                        "host": host,
                    }
                    fout.write(json.dumps(rec) + "\n")
                    fout.flush()
                best = min(times)
                med = sorted(times)[len(times) // 2]
                print(f"  {method:>3}: best={best:.4f}s  median={med:.4f}s  "
                      f"centroids={res['n_centroids']}  (n={len(times)})")

    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
