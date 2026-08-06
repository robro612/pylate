#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# RUN 3 — Build many from the ONE clustering (RUN 2). index/clustering=external
# reuses centroids/assignments (skips the coarse-clustering long pole), doing only
# PQ-train + HNSW per M. M=64 is the precision lever (= PLAID 4-bit); M=32 is the
# baseline bitrate. Index names now carry _m{M} (auto-naming fix), so both land in
# indexes/ as ..._tachiom_external_m32 / _m64 with no collision.
#
# Depends on RUN 2 output. Both builds run in parallel (separate cpu nodes).
# REUSE-SAFE: stages=[build_index] only — no delete stage.
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
CL="$PROJ/clusterings/lotte_pgc_m4t05"
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"

if [ ! -f "$CL/centroids.npy" ] || [ ! -f "$CL/assignments.npy" ]; then
  echo "ERROR: clustering not found at $CL — run cluster_lotte_pgc.sh (RUN 2) first." >&2
  exit 1
fi

for M in 32 64; do
  LOG="$PROJ/logs/scale_val/lotte_build_ext_m${M}.log"
  srun -u -p cpu -t 16:00:00 --cpus-per-task=64 --mem=240G -J "lt_build_m${M}" \
    bash -c "cd $PROJ && uv run --no-sync python scripts/benchmark_indexes.py \
      model=$MODEL index=tachiom index/clustering=external \
      index.clustering.centroids_path='$CL/centroids.npy' \
      index.clustering.assignments_path='$CL/assignments.npy' \
      index.pq_subspaces=$M index.total_centroids=2792304 \
      stages=\"[build_index]\" datasets=\"[$DATASET]\" \
      output.index_folder=indexes \
      output.results_file=results/scale_val/lotte_build_ext_m${M}.jsonl \
      output.runs_dir=null" \
    > "$LOG" 2>&1 &
  echo "launched build M=$M -> $LOG"
  sleep 2
done
wait
echo "=== RUN3 builds done $(date) ==="
