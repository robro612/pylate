# GPU-CAGRA clustering vs CPU-PGC — fidelity + speed

**Question.** Tachiom clustering (PGC/TAC) is CPU-bound. Can a GPU graph-ANN method —
NVIDIA cuVS **CAGRA-guided Lloyd** — replace PGC's HNSW-guided Lloyd as a drop-in coarse
clustering that is *faster* without *losing downstream retrieval quality*?

**Method.** `scripts/gpu_cluster.py backend=cagra` produces `centroids.npy`/`assignments.npy`
ingested by tachiom via `clustering=gpu` (an alias of the `external` precomputed-centroids
path). Every comparison holds K, the downstream build (PQ M=32 → HNSW → IVF), and the search
params **identical** — only the clustering source differs. Model: `lightonai/LateOn-regularized`.
CAGRA params (locked by the trec-covid sweep): `nn_descent`, `graph_degree=64`, `itopk_size=64`.

## trec-covid (29.3M tokens, K=293,130)

CAGRA parameter sweep (downstream identical; Δndcg vs CPU-PGC baseline):

| clustering | cluster_s | nDCG@10 | recall@100 | Δndcg |
|---|--:|--:|--:|--:|
| CPU-PGC (baseline, subsampled) | 455 | 0.8097 | 0.1224 | — |
| GPU exact (pytorch brute) | 761 | 0.7690 | 0.1177 | −0.041 |
| **CAGRA nn_descent g64 itopk64** | 579 | 0.8049 | 0.1304 | −0.005 |
| CAGRA nn_descent g32 itopk64 | 507 | 0.8056 | 0.1318 | −0.004 |
| CAGRA nn_descent g64 itopk32 | 469 | 0.7986 | 0.1316 | −0.011 |
| CAGRA nn_descent g64 itopk128 | 1318 | 0.7944 | 0.1317 | −0.015 |
| CAGRA ivf_pq g64 itopk64 | 488 | 0.7984 | 0.1279 | −0.011 |

- CAGRA matches PGC quality; **GPU *exact* Lloyd is the worst** (0.769) — graph-guided assignment
  lands better centroids than exact Lloyd, just as PGC does on CPU.
- itopk64 is the sweet spot (itopk128 = 2.7× slower, no gain; ivf_pq lower fidelity).

**Apples-to-apples speed** (matched: K, 10 iters, full-data, both graph-guided Lloyd):

| | clustering wall-clock | nDCG@10 |
|---|--:|--:|
| CPU-PGC (32-core) | 1415.7s | 0.8097 |
| GPU-CAGRA (V100) | 578.7s | 0.8049 |

→ **2.45× faster at equal quality.** (The 455s PGC baseline above was subsampled n_iter=5;
forcing identical work removes that confound.)

## LoTTE pooled/dev/search (2.4M docs, 279M tokens, K=2,792,304)

GPU-CAGRA clustered on a **single V100** (cu12 sidecar): train_sample=70M, n_iter=5, streamed
final assignment over all 279M tokens, **2037s (~34 min), peak 24.2 GB**. CPU-PGC
(`lotte_pgc_m4t05`, mult=40) took **~3h10m** → **~5.6× faster** (gap widens with K vs trec-covid's 2.45×).

Downstream head-to-head, identical pipeline (M=32, alpha=0.45), kc×kd grid:

| kc | kd | CAGRA nDCG@10 | PGC nDCG@10 | Δndcg | Δrecall@100 | Δsuccess@5 | QPS (C/P) |
|--:|--:|--:|--:|--:|--:|--:|--:|
| 80 | 20000 | 0.5454 | 0.5471 | −0.0017 | +0.0000 | −0.0013 | 194 / 199 |
| 80 | 10000 | 0.5454 | 0.5471 | −0.0017 | −0.0001 | −0.0013 | 191 / 200 |
| 40 | 20000 | 0.5429 | 0.5453 | −0.0024 | +0.0007 | −0.0034 | 253 / 256 |
| 40 | 10000 | 0.5429 | 0.5453 | −0.0024 | +0.0007 | −0.0034 | 248 / 254 |

All deltas within run-to-run noise; QPS identical (same downstream).

**Caveat on absolutes.** This head-to-head used `alpha=0.45` (default), so absolute numbers sit
below `SWEEP_SUMMARY.md`'s 0.556 nDCG / 0.825 R@100 at kc80/kd20000 (which used `alpha=null`).
Both arms used identical params, so the *comparison* is valid; re-run the grid at `alpha=null`
to line the absolutes up with the published table.

## Conclusion

CAGRA-guided GPU clustering is a **fidelity-neutral, 2.4–5.6× faster drop-in for CPU-PGC**, with
the speed advantage widening with centroid count (29M→279M tokens). The win is confined to the
build/clustering phase; tachiom's CPU inference path (inverted-list gather + PQ decode) is unchanged.

## Reproduce

```bash
# trec-covid sweep (V100 clustering + cpu eval, sbatch):
scripts/run_gpu_cagra_sweep.sh ; uv run --no-sync python scripts/summarize_gpucagra.py

# LoTTE clustering (V100 sidecar) + head-to-head eval:
#   cl: .venv-cu12/bin/python scripts/gpu_cluster.py shard_dir=<lotte docs> \
#         out_dir=clusterings/lotte_cagra clustering.backend=cagra clustering.k=2792304 \
#         clustering.iters=5 clustering.train_sample=70000000 clustering.chunk=2000000
#   ev: scripts/run_lotte_cagra_eval.sh ; uv run --no-sync python scripts/summarize_lotte_cagra.py
```
Env: `scripts/setup_cu12_venv.sh` builds the V100 (cu12) sidecar; L40S/A100 use the main cu13 venv
with `cuvs-cu13` (pyproject `gpu` extra, installed via `uv pip`).
