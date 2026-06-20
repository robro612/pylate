#!/usr/bin/env python3
"""Run TAC/PGC clustering ONCE and save (centroids, assignments) for reuse.

Decouples clustering from indexing: clustering is the expensive step (PGC
minutes-to-hours, TAC up to hours at high centroid budgets), so caching its
output lets you build many downstream indexes (varying PQ / HNSW / search
params) via `clustering=external` without ever re-clustering.

Outputs (sorted(glob) token order, matching TachiomIndex.add_documents_from_shards):
  centroids.npy   [n_centroids, dim] f32
  assignments.npy [n_tokens] u32     (token i -> centroid)

CPU-only (uses the tachiom Rust cluster_tac/cluster_pgc bindings). Reuse with:
  python scripts/benchmark_indexes.py ... index/clustering=external \
    index.clustering.centroids_path=OUT/centroids.npy \
    index.clustering.assignments_path=OUT/assignments.npy

Usage:
  python scripts/cluster_only.py --shard-dir <docs> --method pgc --total-centroids 262144 --out-dir OUT
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np


def load_flat_shards(shard_dir: Path, glob="doc_shard_*.npy"):
    vec_paths = sorted(p for p in shard_dir.glob(glob)
                       if not any(p.name.endswith(s) for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy")))
    if not vec_paths:
        raise FileNotFoundError(f"no shards in {shard_dir}")
    dls = [np.load(str(p.parent / p.name.replace(".npy", ".doclens.npy"))) for p in vec_paths]
    counts = [int(d.sum()) for d in dls]
    n_tok, n_doc = sum(counts), sum(len(d) for d in dls)
    dim = np.load(str(vec_paths[0]), mmap_mode="r").shape[1]
    vecs = np.empty((n_tok, dim), dtype=np.float16)
    tids = np.zeros(n_tok, dtype=np.uint32)
    dl = np.empty(n_doc, dtype=np.int32)
    to = do = 0
    for i, p in enumerate(vec_paths):
        nt, nd = counts[i], len(dls[i])
        vecs[to:to+nt] = np.load(str(p), mmap_mode="r")
        dl[do:do+nd] = dls[i]
        tp = p.parent / p.name.replace(".npy", ".token_ids.npy")
        if tp.exists():
            tids[to:to+nt] = np.load(str(tp), mmap_mode="r")
        to += nt; do += nd
    return vecs, tids, dl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--method", choices=["pgc", "tac"], required=True)
    ap.add_argument("--total-centroids", type=int, default=None, help="None -> auto resolver")
    ap.add_argument("--n-iter", type=int, default=5)
    ap.add_argument("--pgc-sample-multiplier", type=int, default=40)
    ap.add_argument("--pgc-empty-strategy", default="split")
    ap.add_argument("--pgc-assign-topm", type=int, default=1)
    ap.add_argument("--pgc-assign-temp", type=float, default=0.1)
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    import tachiom
    t0 = time.time()
    print(f"loading shards from {a.shard_dir} ...")
    vecs, tids, dl = load_flat_shards(a.shard_dir)
    vectors_u16 = np.ascontiguousarray(vecs).view(np.uint16)
    tids = np.ascontiguousarray(tids, dtype=np.uint32)
    dl = np.ascontiguousarray(dl, dtype=np.int32)
    print(f"  {len(dl)} docs, {vecs.shape[0]} tokens, dim={vecs.shape[1]}  ({time.time()-t0:.0f}s)")

    tc = time.time()
    if a.method == "pgc":
        centroids, assignments = tachiom.cluster_pgc(
            vectors_u16, tids, dl,
            total_centroids=a.total_centroids, pgc_n_iter=a.n_iter,
            pgc_sample_multiplier=a.pgc_sample_multiplier, pgc_empty_strategy=a.pgc_empty_strategy,
            pgc_assign_topm=a.pgc_assign_topm, pgc_assign_temp=a.pgc_assign_temp,
            verbose=True,
        )
    else:
        centroids, assignments = tachiom.cluster_tac(
            vectors_u16, tids, dl, total_centroids=a.total_centroids, tac_n_iter=a.n_iter,
            verbose=True,
        )
    print(f"  {a.method} clustering done in {time.time()-tc:.1f}s -> {centroids.shape[0]} centroids")
    np.save(a.out_dir / "centroids.npy", np.ascontiguousarray(centroids, dtype=np.float32))
    np.save(a.out_dir / "assignments.npy", np.ascontiguousarray(assignments, dtype=np.uint32))
    print(f"wrote {a.out_dir}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
