# Tachiom: clustering, search params, and PQ fidelity vs PLAID

Study on `pgc-clustering` branch (June 2026). Model: **lightonai/LateOn-regularized**
(identical checkpoint to `lateon-fix-m`; lotte cache symlinked to avoid re-encoding).
Datasets: **trec-covid** (deep relevance pools, lexical) and **fiqa** (shallow pools).
All numbers single-run; treat ±3rd-decimal nDCG/recall differences as ties.

## TL;DR

1. **Coarse clustering method is irrelevant to retrieval quality.** PGC, TAC, and exact
   GPU k-means are a wash at good search params (within noise on both datasets). The
   earlier apparent gaps (e.g. TAC ≫ PGC on trec-covid; PGC ≫ TAC elsewhere) were
   **search-param artifacts**, not clustering. → Choose clustering on **build cost** alone.
2. **`k_docs_to_score` was the dominant quality lever**, not clustering. Default **500
   starved the reranker**; it sets how many candidates reach full-MaxSim Stage-3 scoring.
   Raising it (+ `alpha=null`) recovered the entire tachiom→PLAID gap on trec-covid.
3. **`pq_subspaces` (M) is the residual-precision lever.** Default M=32 = 2 bits/dim is
   ~2× more lossy than PLAID's 4-bit residuals. On precision-bound (shallow-pool) fiqa,
   raising to M=64 (= PLAID bitrate) matched/beat PLAID; on coverage-bound trec-covid it
   did nothing (M=32 already at ceiling).
4. With both levers set, **tachiom is at quality parity with PLAID** — at much lower
   retrieval cost (CPU, no GPU).

## Three-stage retrieval and which param drives which

1. **Coarse probe** — each query token probes `k_centroids` centroids (PLAID `nprobe`
   analog); docs get a cheap **centroid-similarity** score. `ef_search` = HNSW beam for
   that probe. `lambda_` = early-exit (speed).
2. **Candidate selection** — keep top **`k_docs_to_score`** docs by coarse score, then
   `alpha`-prune (drop docs below `score_k − |score_k|·alpha`). Note `alpha<1` caps the
   pool *after* k_docs, so **alpha and deep k_docs are in tension**.
3. **Rerank** — full late-interaction MaxSim over each survivor's PQ-decompressed
   residuals (the accurate score). `beta` = early-exit (speed). `pq_subspaces` (M) sets
   residual fidelity here: bits/dim = M/16, bytes/token = M.

## Key tables (iso, kc=20, ef=40)

**trec-covid `k_docs_to_score` sweep (alpha=null) — the coverage lever:**

| k_docs | nDCG@10 | recall@100 | QPS (cpu-32) |
|--:|--:|--:|--:|
| 500 (old default) | 0.743 | 0.079 | 171 |
| 2000 | 0.815 | 0.130 | 89 |
| 5000 | 0.826 | 0.157 | 51 |
| 10000 | 0.832 | 0.165 | 26 |
| PLAID (GPU) | 0.828 | 0.167 | ~10 |

**Iso-good-search clustering comparison (alpha=null, k_docs=10000):**

| | trec-covid (PGC / TAC / kmeans / PLAID) | fiqa (PGC / TAC / PLAID) |
|---|---|---|
| nDCG@10 | 0.833 / 0.829 / 0.832 / 0.828 | 0.503 / 0.503 / 0.514 |
| recall@100 | 0.165 / 0.166 / 0.165 / 0.167 | 0.808 / 0.804 / 0.824 |

→ PGC ≈ TAC ≈ kmeans everywhere. trec-covid matches PLAID; fiqa trails (precision-bound).

**fiqa PQ-fidelity (M) sweep — the precision lever:**

| M | bits/dim | nDCG@10 | recall@100 | disk |
|--:|--:|--:|--:|--:|
| 32 (default) | 2 | 0.503 | 0.808 | 353 MB |
| 64 (= PLAID) | 4 | 0.518 | 0.819 | 588 MB |
| 128 | 8 | 0.518 | 0.820 | 1057 MB |
| PLAID | 4 | 0.514 | 0.824 | 633 MB |

→ M=64 reaches PLAID parity at iso size; M=128 adds nothing (diminishing returns).
On trec-covid, M=128 = M=32 (coverage-bound, fidelity irrelevant).

## Diagnosis: two failure modes, two levers

- **Deep-pool / coverage-bound (trec-covid):** recall is gated by how many candidates get
  scored → fix with **`k_docs_to_score`** (+ `alpha=null`). PQ fidelity irrelevant.
- **Shallow-pool / precision-bound (fiqa):** recall already saturated; ranking precision is
  gated by residual fidelity → fix with **`pq_subspaces=64`**. More probing/k_docs doesn't help.

## Recommended defaults / knobs

- `k_docs_to_score`: **2000** (was 500). 5000–10000 + `alpha=null` for deep-pool sets.
- `alpha`: 0.45 (fine/faster at moderate k_docs); `null` for max recall at deep k_docs.
- `pq_subspaces`: 32 default (smallest/fastest); **64** when precision-bound. 128 = overkill.
- `k_centroids`: 20 (probe breadth was never the bottleneck).
- Clustering: PGC or TAC by **build cost** — TAC cheap at moderate K but degrades/crawls
  past its sweet spot (TAC@1.46M centroids on trec-covid took ~5h); PGC scales to large K.

## Infrastructure added this study

- **Decoupled clustering from indexing.** `cluster_pgc`/`cluster_tac`/GPU-kmeans produce
  `(centroids, assignments)`; `TachiomIndex(clustering="external", ...)` /
  `build_from_arrays_with_centroids` ingest them → cluster once, build many (skip the
  expensive cluster step when sweeping PQ/HNSW/search params). Scripts:
  `scripts/cluster_only.py`, `scripts/gpu_kmeans_cluster.py`.
- **`pq_subspaces` (M) is a runtime config knob** over compiled variants {4,8,16,32,64,128}
  (M is a const generic; `PyTachiom` dispatches via a `TachiomInner` enum). Validation gates
  only on what's compiled, not quality.
- Per-phase build timers (`TIMING <phase>`), soft top-m PGC assignment (`assign_topm`/
  `assign_temp`), PGC early-exit (`iter_lambda`), parallelized PGC accumulation.

## Caveats

- Single runs; ±3rd-decimal nDCG/recall = noise.
- QPS comparisons are cross-node/unbatched and favor PLAID (it ran on GPU); CPU tachiom is
  still several× faster at matched quality. A batched L40S-64 QPS run is the clean follow-up.
- Findings shown on trec-covid + fiqa (one deep-pool, one shallow-pool); broader BEIR/lotte
  confirmation would strengthen the coverage-vs-precision generalization.
