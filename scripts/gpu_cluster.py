#!/usr/bin/env python3
"""Unified GPU coarse clustering for tachiom, decoupled from the index build.

Produces (centroids, assignments) that tachiom ingests via `clustering=gpu`
(an alias of the `external` pathway), running the identical downstream
(PQ -> HNSW -> IVF) as CPU PGC/TAC. This isolates the *clustering method* from
its downstream use, so GPU clustering is a pure build-time speedup, not a
quality change.

Spherical k-means (L2-normalized): for unit vectors argmin||x-c||^2 == argmax x·c,
so sqeuclidean-on-the-sphere == cosine, matching tachiom's inner-product search
and PGC's normalized-mean updates.

Backends (Hydra: `clustering.backend=`):
  brute : self-contained tiled exact Lloyd in pytorch (tiles over N and K, any K).
  cagra : CAGRA-guided Lloyd — the GPU analogue of PGC. Each iteration builds a
          cuVS CAGRA graph over the current centroids and does approximate
          nearest-centroid assignment on the GPU. Pays at huge K, where an exact
          N×K assignment is the bottleneck. Needs cuvs (cu13 main venv on L40S/A100,
          or cu12 sidecar on V100 — see scripts/setup_cu12_venv.sh).
  flash : svg-project flash-kmeans (exact, batched). Standalone.

Scale: `clustering.train_sample` controls memory. null = load the whole corpus onto
the GPU and cluster on all of it (fine up to ~10s of GB). An int N_train = subsample
that many tokens for the Lloyd iterations (mmap'd, never materializing the full corpus
in host RAM), then stream a final hard assignment over ALL tokens chunk-by-chunk. This
is what makes LoTTE/MSMARCO-scale (100s of millions of tokens) fit, and it mirrors
PGC's sample_multiplier — set train_sample = sample_multiplier * K for an apples-to-apples
clustering-speed comparison vs CPU-PGC.

Outputs into `out_dir` (sorted(glob) token order, matching
TachiomIndex.add_documents_from_shards so assignments line up under clustering=gpu):
  centroids.npy   [K, dim] f32
  assignments.npy [n_tokens] u32   (token i -> centroid)
  meta.json       backend + params + timings + peak GPU mem

Usage:
  # in-memory (small corpus), L40S/A100 main venv:
  uv run --no-sync python scripts/gpu_cluster.py shard_dir=<docs> out_dir=<o> \
    clustering.backend=cagra clustering.k=262144
  # streamed/subsampled (large corpus), V100 sidecar:
  srunv100 .venv-cu126/bin/python scripts/gpu_cluster.py shard_dir=<docs> out_dir=<o> \
    clustering.backend=cagra clustering.k=2792304 clustering.train_sample=11200000
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf


# ---------------------------------------------------------------------------
# Lazy sharded vector access (mmap) — for corpora too large to hold on GPU/host.
# Token order = sorted(glob), matching TachiomIndex.add_documents_from_shards.
# ---------------------------------------------------------------------------
def _shard_vec_paths(shard_dir: Path, glob="doc_shard_*.npy"):
    return sorted(
        p for p in shard_dir.glob(glob)
        if not any(p.name.endswith(s) for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy"))
    )


class ShardedVectors:
    """Read-only mmap view over the per-shard token vectors, in global token order."""

    def __init__(self, shard_dir: Path, glob="doc_shard_*.npy"):
        self.paths = _shard_vec_paths(shard_dir, glob)
        if not self.paths:
            raise FileNotFoundError(f"no shards in {shard_dir}")
        self.counts = []
        for p in self.paths:
            dl = p.parent / p.name.replace(".npy", ".doclens.npy")
            self.counts.append(int(np.load(str(dl)).sum()) if dl.exists()
                               else int(np.load(str(p), mmap_mode="r").shape[0]))
        self.offsets = np.cumsum([0] + self.counts)
        self.N = int(self.offsets[-1])
        self.dim = int(np.load(str(self.paths[0]), mmap_mode="r").shape[1])

    def sample(self, m: int, seed: int = 42) -> np.ndarray:
        """Gather m random tokens (f16 [m, dim]) without materializing the full corpus."""
        m = min(m, self.N)
        rng = np.random.default_rng(seed)
        idx = np.sort(rng.choice(self.N, size=m, replace=False))
        out = np.empty((m, self.dim), dtype=np.float16)
        shard_of = np.searchsorted(self.offsets, idx, side="right") - 1
        w = 0
        for s, p in enumerate(self.paths):
            sel = idx[shard_of == s]
            if len(sel) == 0:
                continue
            local = sel - self.offsets[s]
            mm = np.load(str(p), mmap_mode="r")
            out[w:w + len(sel)] = mm[local]
            w += len(sel)
        return out

    def iter_chunks(self, chunk: int):
        """Yield (global_start, f16 ndarray [c, dim]) covering all tokens in order."""
        for s, p in enumerate(self.paths):
            mm = np.load(str(p), mmap_mode="r")
            base = int(self.offsets[s])
            for lo in range(0, self.counts[s], chunk):
                hi = min(lo + chunk, self.counts[s])
                yield base + lo, np.ascontiguousarray(mm[lo:hi])


def load_flat_shards(shard_dir: Path, glob="doc_shard_*.npy"):
    """Materialize the full corpus (in-memory path). Returns (vecs f16, tids u32, doclens i32)."""
    vec_paths = _shard_vec_paths(shard_dir, glob)
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
        vecs[to:to + nt] = np.load(str(p), mmap_mode="r")
        dl[do:do + nd] = dls[i]
        tp = p.parent / p.name.replace(".npy", ".token_ids.npy")
        if tp.exists():
            tids[to:to + nt] = np.load(str(tp), mmap_mode="r")
        to += nt
        do += nd
    return vecs, tids, dl


def _normalize_inplace(X: torch.Tensor, block: int = 1_000_000) -> None:
    for s in range(0, X.shape[0], block):
        X[s:s + block] = torch.nn.functional.normalize(X[s:s + block].float(), dim=1).half()


# ---------------------------------------------------------------------------
# Assignment kernels: (X[N,D] f16, C_half[K,D] f16) -> labels[N] long on GPU.
# ---------------------------------------------------------------------------
def _argmax_brute(xb, Ch, kblock):
    """Exact tiled argmax for one batch xb[c,D] vs all centroids Ch[K,D]; returns labels[c]."""
    K = Ch.shape[0]
    best = torch.full((xb.shape[0],), float("-inf"), device=xb.device, dtype=torch.float16)
    arg = torch.zeros(xb.shape[0], dtype=torch.long, device=xb.device)
    for kc in range(0, K, kblock):
        sims = xb @ Ch[kc:kc + kblock].T
        m, a = sims.max(dim=1)
        upd = m > best
        best = torch.where(upd, m, best)
        arg = torch.where(upd, a + kc, arg)
    return arg


def _assign_brute(X, Ch, chunk, kblock, desc=None):
    N = X.shape[0]
    labels = torch.empty(N, dtype=torch.long, device=X.device)
    rng = range(0, N, chunk)
    if desc is not None:
        from tqdm import tqdm
        rng = tqdm(rng, desc=desc, unit="batch", leave=False, mininterval=2.0)
    for s in rng:
        labels[s:s + chunk] = _argmax_brute(X[s:s + chunk], Ch, kblock)
    return labels


def _build_cagra(Ch, cfg_cagra):
    from cuvs.neighbors import cagra
    params = cagra.IndexParams(
        metric=cfg_cagra.get("metric", "sqeuclidean"),
        intermediate_graph_degree=cfg_cagra.intermediate_graph_degree,
        graph_degree=cfg_cagra.graph_degree,
        build_algo=cfg_cagra.build_algo,
    )
    return cagra.build(params, Ch.contiguous())


def _cagra_search(index, xb, cfg_cagra):
    from cuvs.neighbors import cagra
    sp = cagra.SearchParams(itopk_size=cfg_cagra.itopk_size,
                            search_width=cfg_cagra.get("search_width", 1))
    cn = xb.shape[0]
    nb = torch.empty((cn, 1), dtype=torch.uint32, device=xb.device)
    dist = torch.empty((cn, 1), dtype=torch.float32, device=xb.device)
    cagra.search(sp, index, xb.contiguous(), 1, neighbors=nb, distances=dist)
    return nb[:, 0].long()


def _assign_cagra(X, Ch, cfg_cagra, chunk, desc=None):
    index = _build_cagra(Ch, cfg_cagra)
    N = X.shape[0]
    labels = torch.empty(N, dtype=torch.long, device=X.device)
    rng = range(0, N, chunk)
    if desc is not None:
        from tqdm import tqdm
        rng = tqdm(rng, desc=desc, unit="batch", leave=False, mininterval=2.0)
    for s in rng:
        labels[s:s + chunk] = _cagra_search(index, X[s:s + chunk], cfg_cagra)
    return labels


# ---------------------------------------------------------------------------
# Lloyd loop shared by brute / cagra. Returns GPU centroids (f32 [K,D]) and
# optional GPU labels (final hard assignment over X), or None if final_assign=False.
# ---------------------------------------------------------------------------
def lloyd(X, K, iters, assign_fn, seed=42, chunk=65536, log=print, final_assign=True):
    g = torch.Generator(device=X.device).manual_seed(seed)
    N, D = X.shape
    C = X[torch.randperm(N, generator=g, device=X.device)[:K]].float()
    C = torch.nn.functional.normalize(C, dim=1)
    for it in range(iters):
        t = time.time()
        labels = assign_fn(X, C.half(), f"iter {it + 1}/{iters} assign")
        sums = torch.zeros(K, D, device=X.device, dtype=torch.float32)
        cnt = torch.zeros(K, device=X.device, dtype=torch.float32)
        for s in range(0, N, chunk):
            lab = labels[s:s + chunk]
            sums.index_add_(0, lab, X[s:s + chunk].float())
            cnt.index_add_(0, lab, torch.ones(lab.shape[0], device=X.device))
        nz = cnt > 0
        C[nz] = torch.nn.functional.normalize(sums[nz] / cnt[nz, None], dim=1)
        if (~nz).any():  # reinit empties from random points (PGC 'resample')
            ridx = torch.randint(0, N, ((~nz).sum().item(),), generator=g, device=X.device)
            C[~nz] = X[ridx].float()
        log(f"  iter {it + 1}/{iters}: {int((~nz).sum())} empty, {time.time() - t:.1f}s")
    labels = assign_fn(X, C.half(), "final assign") if final_assign else None
    return C, labels


def _stream_assign(sv: ShardedVectors, C_half, backend, cfg, chunk, log=print):
    """Final hard top-1 assignment over ALL tokens, streamed from mmap'd shards.

    Builds the CAGRA index once (cagra) over the final centroids, or tiles over K
    (brute), and walks the corpus chunk-by-chunk so the full corpus never lands in
    host RAM or GPU memory at once.
    """
    from tqdm import tqdm

    dev = C_half.device
    labels = np.empty(sv.N, dtype=np.uint32)
    log(f"  building {backend} index over {C_half.shape[0]} centroids for final assignment ...")
    index = _build_cagra(C_half, cfg.cagra) if backend == "cagra" else None
    t = time.time()
    done = 0
    pbar = tqdm(total=sv.N, desc="stream-assign", unit="tok", unit_scale=True, mininterval=2.0)
    for start, arr in sv.iter_chunks(chunk):
        xb = torch.nn.functional.normalize(
            torch.from_numpy(arr).to(dev, non_blocking=True).float(), dim=1).half()
        if backend == "cagra":
            lab = _cagra_search(index, xb, cfg.cagra)
        else:
            lab = _argmax_brute(xb, C_half, cfg.kblock)
        labels[start:start + arr.shape[0]] = lab.to(torch.int32).cpu().numpy().astype(np.uint32)
        done += arr.shape[0]
        pbar.update(arr.shape[0])
    pbar.close()
    log(f"  streamed final assignment: {done} tokens in {time.time() - t:.1f}s")
    return labels


def kmeans_flash(X, K, iters, seed=42, log=print):
    import flash_kmeans as fk

    log(f"  flash_kmeans backend, K={K}")
    out = fk.batch_kmeans_Euclid(X.unsqueeze(0), n_clusters=K, max_iters=iters, verbose=True)
    labels, centroids = out[0], out[1]
    centroids = torch.nn.functional.normalize(centroids.squeeze(0).float(), dim=1)
    return centroids.cpu().numpy().astype(np.float32), labels.squeeze(0).cpu().numpy().astype(np.uint32)


# ---------------------------------------------------------------------------
def cluster_tokens(shard_dir: Path, out_dir: Path, cl: DictConfig):
    """Run GPU coarse clustering over the token shards in `shard_dir`.

    Writes centroids.npy / assignments.npy / meta.json into `out_dir` and returns
    (centroids_path, assignments_path). This is the importable core shared by the
    standalone CLI (`main`, below) and the `cluster` stage of benchmark_indexes.py, so
    both run identical code. K resolution: `clustering.k` if set, else
    tachiom.auto_build_params on the token_ids (in-memory) or 1% of tokens (streaming).
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    backend = cl.backend
    if cl.get("search_topm", 1) and cl.get("search_topm", 1) > 1:
        raise NotImplementedError(
            "soft top-m update (search_topm>1) not implemented; index assignment is hard top-1."
        )
    train_sample = cl.get("train_sample", None)
    streaming = train_sample is not None

    dev = "cuda"
    if not torch.cuda.is_available():
        raise RuntimeError("no CUDA device — run on a GPU node (srunl40s/srunv100).")

    t0 = time.time()
    if streaming:
        if backend == "flash":
            raise ValueError("flash backend does not support streaming/train_sample; use null.")
        sv = ShardedVectors(shard_dir)
        n_tok, dim = sv.N, sv.dim
        print(f"  (streaming) {n_tok} tokens, dim={dim}, {len(sv.paths)} shards "
              f"({time.time() - t0:.0f}s index)")
    else:
        print(f"loading shards from {shard_dir} ...")
        vecs, tids, _dl = load_flat_shards(shard_dir)
        n_tok, dim = vecs.shape
        print(f"  {n_tok} tokens, dim={dim}  ({time.time() - t0:.0f}s)")

    # Resolve K. null -> tachiom auto resolver (needs token_ids -> in-memory only).
    K = cl.get("k", None)
    if K is None:
        if streaming:
            # Streaming never loads token_ids; use the established 1%-of-tokens rule
            # (the LateOn lotte_cagra K convention) as the auto default.
            K = round(0.01 * n_tok)
            print(f"  resolved K = round(1% of {n_tok} tokens) = {K}")
        else:
            import tachiom
            K = int(tachiom.auto_build_params(
                np.ascontiguousarray(tids, dtype=np.uint32), total_centroids=None)["total_centroids"])
            print(f"  resolved K (auto_build_params) = {K}")
    K = int(K)

    print("torch", torch.__version__, "device", torch.cuda.get_device_name(0))
    torch.cuda.reset_peak_memory_stats()
    print(f"clustering: backend={backend} K={K} iters={cl.iters} "
          f"{'train_sample=' + str(train_sample) + ' (streamed)' if streaming else '(full, in-memory)'}")
    tk = time.time()

    if streaming:
        # Train on a GPU-resident subsample; final assignment streamed over all tokens.
        print(f"  gathering {int(train_sample)}-token training subsample from mmap'd shards ...")
        Xs = torch.from_numpy(sv.sample(int(train_sample), seed=cl.seed)).to(dev)
        _normalize_inplace(Xs)
        print(f"  training subsample on GPU: {Xs.shape[0]} tokens "
              f"({time.time() - tk:.0f}s to gather)")
        if backend == "cagra":
            if K <= cl.cagra.intermediate_graph_degree:
                raise ValueError(f"K={K} too small for CAGRA; use backend=brute.")
            assign = lambda Xx, Ch, desc: _assign_cagra(Xx, Ch, cl.cagra, cl.chunk, desc=desc)  # noqa: E731
        else:
            assign = lambda Xx, Ch, desc: _assign_brute(Xx, Ch, cl.chunk, cl.kblock, desc=desc)  # noqa: E731
        C, _ = lloyd(Xs, K, cl.iters, assign, seed=cl.seed, chunk=cl.chunk, final_assign=False)
        del Xs
        torch.cuda.empty_cache()
        labels = _stream_assign(sv, C.half(), backend, cl, cl.chunk)
        centroids = C.cpu().numpy().astype(np.float32)
    else:
        X = torch.from_numpy(vecs).to(dev)
        _normalize_inplace(X)
        if backend == "brute":
            assign = lambda Xx, Ch, desc: _assign_brute(Xx, Ch, cl.chunk, cl.kblock, desc=desc)  # noqa: E731
            C, lab = lloyd(X, K, cl.iters, assign, seed=cl.seed, chunk=cl.chunk)
        elif backend == "cagra":
            if K <= cl.cagra.intermediate_graph_degree:
                raise ValueError(f"K={K} too small for CAGRA; use backend=brute.")
            assign = lambda Xx, Ch, desc: _assign_cagra(Xx, Ch, cl.cagra, cl.chunk, desc=desc)  # noqa: E731
            C, lab = lloyd(X, K, cl.iters, assign, seed=cl.seed, chunk=cl.chunk)
        elif backend == "flash":
            centroids, labels = kmeans_flash(X, K, cl.iters, seed=cl.seed)
            C = None
        else:
            raise ValueError(f"unknown backend {backend!r} (brute|cagra|flash)")
        if C is not None:
            centroids = C.cpu().numpy().astype(np.float32)
            labels = lab.cpu().numpy().astype(np.uint32)

    cluster_s = time.time() - tk
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    print(f"  {backend} clustering done in {cluster_s:.1f}s -> {centroids.shape[0]} centroids, "
          f"peak GPU {peak_gb:.1f} GB")

    np.save(out_dir / "centroids.npy", np.ascontiguousarray(centroids, dtype=np.float32))
    np.save(out_dir / "assignments.npy", np.ascontiguousarray(labels, dtype=np.uint32))
    meta = {
        "backend": backend, "k": int(centroids.shape[0]), "n_tokens": int(n_tok), "dim": int(dim),
        "iters": int(cl.iters), "seed": int(cl.seed), "streaming": bool(streaming),
        "train_sample": int(train_sample) if streaming else None,
        "clustering_seconds": round(cluster_s, 2), "peak_gpu_gb": round(peak_gb, 2),
        "gpu": torch.cuda.get_device_name(0), "shard_dir": str(shard_dir),
        "clustering_cfg": OmegaConf.to_container(cl, resolve=True),
    }
    (out_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"wrote {out_dir}  (total {time.time() - t0:.0f}s)")
    return out_dir / "centroids.npy", out_dir / "assignments.npy"


@hydra.main(config_path="../conf/eval", config_name="cluster", version_base=None)
def main(cfg: DictConfig) -> None:
    # Line-buffer stdout so progress shows up live under slurm (-o redirects stdout to a
    # file, where the default block buffering hides every print until the buffer fills).
    try:
        import sys
        sys.stdout.reconfigure(line_buffering=True)
    except Exception:
        pass
    shard_dir = Path(hydra.utils.to_absolute_path(cfg.shard_dir))
    out_dir = Path(hydra.utils.to_absolute_path(cfg.out_dir))
    cluster_tokens(shard_dir, out_dir, cfg.clustering)


if __name__ == "__main__":
    main()
