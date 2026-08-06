#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Build a tachiom index from the VANILLA TAC clustering (clusterings/lotte_tac,
# iso 1% = 2,792,304 centroids, auto-cap, n_iter=10) via the external-clustering
# path — identical recipe to build_lotte_external_M.sh (PGC) so TAC-vs-PGC retrieval
# is apples-to-apples. M=32 (baseline bitrate, matches the PGC comparison point).
# Index name auto-derives to ..._tachiom_external_lotte_tac_m32.
# REUSE-SAFE: stages=[build_index] only — no delete.
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
CL="$PROJ/clusterings/lotte_tac"
M=32
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"

if [ ! -f "$CL/centroids.npy" ] || [ ! -f "$CL/assignments.npy" ]; then
  echo "ERROR: TAC clustering not found at $CL" >&2; exit 1
fi

LOG="$PROJ/logs/scale_val/lotte_build_tac_m${M}.log"
srun -u -p cpu -t 16:00:00 --cpus-per-task=64 --mem=240G -J "lt_build_tac_m${M}" \
  bash -c "cd $PROJ && uv run --no-sync python scripts/benchmark_indexes.py \
    model=$MODEL index=tachiom index/clustering=external \
    index.clustering.centroids_path='$CL/centroids.npy' \
    index.clustering.assignments_path='$CL/assignments.npy' \
    index.pq_subspaces=$M index.total_centroids=2792304 \
    stages=\"[build_index]\" datasets=\"[$DATASET]\" \
    output.index_folder=indexes \
    output.results_file=results/scale_val/lotte_build_tac_m${M}.jsonl \
    output.runs_dir=null" \
  > "$LOG" 2>&1
echo "=== TAC build M=$M done $(date) -> $LOG ==="
