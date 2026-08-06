#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Search-lever sweeps on the EXISTING PGC M=32 index (retrieve-only, no rebuild).
# Sweep 1: coarse-probe breadth  k_centroids x k_docs  (ef_search=null=auto=1.5*kc)
# Sweep 2: alpha pruning at the parity point (kd=20000, kc=20, ef=40)
# Baselines already on disk (RUN4 + deep probe): kc=20 x kd{...}, and alpha=null@kd20k.
# Two phases run in parallel (separate cpu nodes), like RUN 4. REUSE-SAFE: stages=[retrieve].
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
CL="$PROJ/clusterings/lotte_pgc_m4t05"
M=32
IDXDIR="indexes/bench_lightonai_LateOn-regularized_lotte_pooled_dev_search_tachiom_external_lotte_pgc_m4t05_m${M}"
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"

if [ ! -d "$PROJ/$IDXDIR" ]; then echo "ERROR: index $IDXDIR missing" >&2; exit 1; fi

run_one() {  # $1=tag $2..=extra hydra overrides
  local tag="$1"; shift
  uv run --no-sync python scripts/benchmark_indexes.py \
    model=$MODEL index=tachiom index/clustering=external \
    index.clustering.centroids_path="$CL/centroids.npy" \
    index.clustering.assignments_path="$CL/assignments.npy" \
    index.pq_subspaces=$M \
    stages="[retrieve]" datasets="[$DATASET]" \
    output.index_folder=indexes \
    output.results_file="results/scale_val/lotte_search_pgc_${tag}.jsonl" \
    output.runs_dir=null "$@"
}
export -f run_one
export MODEL CL M DATASET

# --- Phase A: Sweep 1 (k_centroids x k_docs) ---
LOGA="$PROJ/logs/scale_val/lotte_pgc_lever_kc.log"
srun -u -p cpu -t 8:00:00 --cpus-per-task=64 --mem=240G -J lt_pgc_kc \
  bash -c "cd $PROJ && \
    for kc in 40 80; do for kd in 10000 20000; do \
      echo \"=== kc=\$kc kd=\$kd alpha=null ef=auto  \$(date) ===\"; \
      run_one kc\${kc}_kd\${kd} index.k_centroids=\$kc index.ef_search=null \
        index.k_docs_to_score=\$kd index.alpha=null; \
    done; done" \
  > "$LOGA" 2>&1 &
echo "launched Sweep1 (k_centroids) -> $LOGA"
sleep 2

# --- Phase B: Sweep 2 (alpha at parity point kd=20000, kc=20, ef=40) ---
LOGB="$PROJ/logs/scale_val/lotte_pgc_lever_alpha.log"
srun -u -p cpu -t 8:00:00 --cpus-per-task=64 --mem=240G -J lt_pgc_alpha \
  bash -c "cd $PROJ && \
    for a in 0.6 0.45 0.3; do \
      aslug=\$(echo \$a | tr -d .); \
      echo \"=== alpha=\$a kd=20000 kc=20 ef=40  \$(date) ===\"; \
      run_one alpha\${aslug}_kd20000 index.k_centroids=20 index.ef_search=40 \
        index.k_docs_to_score=20000 index.alpha=\$a; \
    done" \
  > "$LOGB" 2>&1 &
echo "launched Sweep2 (alpha) -> $LOGB"
wait
echo "=== PGC search-lever sweeps done $(date) ==="
