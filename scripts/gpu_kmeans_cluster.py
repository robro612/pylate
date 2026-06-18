#!/usr/bin/env python3
"""GPU spherical k-means coarse clustering, decoupled from the tachiom index.

Separates clustering from indexing: produces centroids + per-token assignments
that tachiom's `Tachiom.build_from_tac(...)` ingests to run the identical
downstream (PQ -> HNSW -> IVF) as PGC/TAC. This lets us test whether the
clustering method (vs the downstream use of centroids) drives retrieval quality.

Spherical k-means (L2-normalized): for unit vectors, argmin ||x-c||^2 == argmax x·c,
so euclidean-on-the-sphere == cosine, matching tachiom's inner-product search and
PGC's normalized-mean updates.

Backends:
  --backend torch  : self-contained tiled Lloyd (tiles over N and K, so any K).
  --backend flash  : svg-project flash-kmeans (faster if it supports the K regime).

Outputs into --out-dir:
  centroids.npy [K, d] f32, assignments.npy [n_tok] u32

Token order is sorted(glob) over the shard dir, matching
TachiomIndex.add_documents_from_shards, so assignments line up with the index's
token order (clustering="external"). No flat concatenation of the corpus needed.

Usage (GPU node):
  python scripts/gpu_kmeans_cluster.py \
     --shard-dir embeddings_cache/beir_trec-covid/lightonai_LateOn-regularized/nopool_fp16/docs \
     --k 262144 --iters 10 --backend torch --out-dir /scratch/.../km_treccovid
"""
from __future__ import annotations
import argparse, time
from pathlib import Path
import numpy as np
import torch


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


def kmeans_torch(X, K, iters, seed=42, chunk=65536, kblock=16384, log=print):
    """Tiled spherical Lloyd. X: [N,D] f16 unit-norm on GPU. Returns (centroids f32 [K,D], labels u32 [N]).

    Tiles assignment over both N (chunk) and K (kblock) so it never materializes
    the N x K similarity matrix, and accumulates the update in N-chunks too, so
    peak VRAM ~= |X| + one [chunk, kblock] fp16 tile. Defaults are sized for a
    16GB V100 (7.5GB trec-covid data + ~2GB tile). A per-iteration tqdm bar over
    the assignment chunks gives a live throughput/ETA estimate.
    """
    from tqdm import tqdm
    g = torch.Generator(device=X.device).manual_seed(seed)
    N, D = X.shape
    C = X[torch.randperm(N, generator=g, device=X.device)[:K]].float()      # random init from data
    C = torch.nn.functional.normalize(C, dim=1)
    labels = torch.empty(N, dtype=torch.long, device=X.device)
    NEG = float("-inf")
    for it in range(iters):
        t = time.time()
        Ch = C.half()
        # ── Assignment: tiled over N (chunk) x K (kblock), running argmax in fp16 ──
        for s in tqdm(range(0, N, chunk), desc=f"iter {it+1}/{iters} assign", leave=False):
            xb = X[s:s+chunk]                                               # [c, D] f16
            best = torch.full((xb.shape[0],), NEG, device=X.device, dtype=torch.float16)
            arg = torch.zeros(xb.shape[0], dtype=torch.long, device=X.device)
            for kc in range(0, K, kblock):
                sims = xb @ Ch[kc:kc+kblock].T                              # [c, kb] f16
                m, a = sims.max(dim=1)
                upd = m > best
                best = torch.where(upd, m, best)
                arg = torch.where(upd, a + kc, arg)
            labels[s:s+chunk] = arg
        # ── Update: chunked scatter-add (fp32 accum), spherical (renormalize) ──
        sums = torch.zeros(K, D, device=X.device, dtype=torch.float32)
        cnt = torch.zeros(K, device=X.device, dtype=torch.float32)
        for s in range(0, N, chunk):
            lab = labels[s:s+chunk]
            sums.index_add_(0, lab, X[s:s+chunk].float())
            cnt.index_add_(0, lab, torch.ones(lab.shape[0], device=X.device))
        nz = cnt > 0
        C[nz] = torch.nn.functional.normalize(sums[nz] / cnt[nz, None], dim=1)
        if (~nz).any():                                                     # reinit empties from random points
            ridx = torch.randint(0, N, ((~nz).sum().item(),), generator=g, device=X.device)
            C[~nz] = X[ridx].float()
        log(f"  iter {it+1}/{iters}: {int((~nz).sum())} empty, {time.time()-t:.1f}s")
    return C.cpu().numpy().astype(np.float32), labels.cpu().numpy().astype(np.uint32)


def kmeans_flash(X, K, iters, seed=42, log=print):
    import flash_kmeans as fk
    log(f"  flash_kmeans backend, K={K}")
    out = fk.batch_kmeans_Euclid(X.unsqueeze(0), n_clusters=K, max_iters=iters, verbose=True)
    labels, centroids = out[0], out[1]                                      # (assignments, centroids, ...)
    centroids = torch.nn.functional.normalize(centroids.squeeze(0).float(), dim=1)
    return centroids.cpu().numpy().astype(np.float32), labels.squeeze(0).cpu().numpy().astype(np.uint32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--shard-dir", type=Path, required=True)
    ap.add_argument("--k", type=int, required=True)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--backend", choices=["torch", "flash"], default="torch")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--chunk", type=int, default=65536, help="torch backend: N-tile size (VRAM)")
    ap.add_argument("--kblock", type=int, default=16384, help="torch backend: K-tile size (VRAM)")
    ap.add_argument("--out-dir", type=Path, required=True)
    a = ap.parse_args()
    a.out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    print(f"loading shards from {a.shard_dir} ...")
    vecs, _tids, _dl = load_flat_shards(a.shard_dir)
    print(f"  {len(_dl)} docs, {vecs.shape[0]} tokens, dim={vecs.shape[1]}  ({time.time()-t0:.0f}s)")

    dev = "cuda"
    X = torch.from_numpy(vecs).to(dev)                                      # [N,D] f16
    # Unit-normalize in-place, chunked, to avoid a full-corpus fp32 copy (OOM on 32GB).
    for s in range(0, X.shape[0], 1_000_000):
        X[s:s+1_000_000] = torch.nn.functional.normalize(X[s:s+1_000_000].float(), dim=1).half()
    print(f"clustering: backend={a.backend} K={a.k} iters={a.iters}")
    tk = time.time()
    if a.backend == "flash":
        centroids, labels = kmeans_flash(X, a.k, a.iters, seed=a.seed)
    else:
        centroids, labels = kmeans_torch(X, a.k, a.iters, seed=a.seed, chunk=a.chunk, kblock=a.kblock)
    print(f"  kmeans done in {time.time()-tk:.1f}s -> {centroids.shape[0]} centroids")
    np.save(a.out_dir / "centroids.npy", centroids)
    np.save(a.out_dir / "assignments.npy", labels.astype(np.uint32))
    print(f"wrote {a.out_dir}  (total {time.time()-t0:.0f}s)")


if __name__ == "__main__":
    main()
