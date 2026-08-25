"""Chimera build cost vs corpus size, on a prefix of the lotte shards."""
import gc
import glob
import resource
import sys
import time

import chimera
import numpy as np

sys.stdout.reconfigure(line_buffering=True)

D = "embeddings_cache/lotte_pooled_dev_search/lightonai_LateOn/nopool_fp16/docs"
vecs = [f for f in sorted(glob.glob(f"{D}/doc_shard_*.npy"))
        if not f.endswith((".doclens.npy", ".token_ids.npy"))]

def peak_gib():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 2**20

for n_shards in (1, 2, 4):
    paths = vecs[:n_shards]
    doclens = np.concatenate([np.load(p.replace(".npy", ".doclens.npy")) for p in paths])
    counts = [int(np.load(p.replace(".npy", ".doclens.npy")).sum()) for p in paths]
    total = sum(counts)

    t0 = time.perf_counter()
    emb = np.empty((total, 128), dtype=np.float32)
    off = 0
    for p, c in zip(paths, counts):
        emb[off:off + c] = np.load(p, mmap_mode="r")
        off += c
    load_s = time.perf_counter() - t0

    n_clusters = total // 150
    t0 = time.perf_counter()
    idx = chimera.ChimeraIndex.build(
        emb, doclens.astype(np.int32).tolist(),
        n_clusters=n_clusters, ex_bits=4,
    )
    build_s = time.perf_counter() - t0

    print(f"shards={n_shards:2d}  docs={len(doclens):>9,}  tokens={total:>12,}  "
          f"n_clusters={n_clusters:>9,}  load={load_s:7.1f}s  build={build_s:8.1f}s  "
          f"peakRSS={peak_gib():6.1f} GiB")
    del idx, emb
    gc.collect()
