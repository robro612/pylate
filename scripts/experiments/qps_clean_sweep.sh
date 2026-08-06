#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Clean QPS + quality sweep over the two SEARCH-TIME precision/coverage knobs:
#   k_centroids  (nprobe   — coarse probe breadth)
#   k_docs_to_score (kd / rerank depth — recall/coverage lever)
# on the prebuilt M=32 indexes. Runs inside ONE --exclusive allocation on a fast
# EPYC 7713 node with a FIXED thread count, so every QPS number is uncontaminated
# (no co-tenants) and directly comparable across configs and TAC-vs-PGC.
# alpha=null / ef_search=auto throughout. Reload-per-config is wasteful wall-clock
# but does NOT affect QPS (search_time excludes index load).
set -u
PROJ=/exp/rjha/pylate-pgc
cd "$PROJ"
DATASET="lotte/pooled/dev/search"; MODEL=lateon_regularized
TAC="$PROJ/clusterings/lotte_tac"
PGC="$PROJ/clusterings/lotte_pgc_m4t05"

run () {  # method_tag  centroids_dir  kc  kd
  local tag=$1 cl=$2 kc=$3 kd=$4
  uv run --no-sync python scripts/benchmark_indexes.py \
    model=$MODEL index=tachiom index/clustering=external \
    index.clustering.centroids_path="$cl/centroids.npy" \
    index.clustering.assignments_path="$cl/assignments.npy" \
    index.pq_subspaces=32 index.k_centroids=$kc index.k_docs_to_score=$kd \
    index.alpha=null index.ef_search=null \
    stages='[retrieve]' datasets="[$DATASET]" \
    output.index_folder=indexes \
    output.results_file="results/scale_val/qps_${tag}_kc${kc}_kd${kd}.jsonl" \
    output.runs_dir=null
  echo "  [done] $tag kc=$kc kd=$kd"
}

echo "=== QPS sweep start $(date) on $(hostname); RAYON_NUM_THREADS=${RAYON_NUM_THREADS:-unset} ==="
# TAC: full kc x kd frontier
for kc in 20 40 80; do
  for kd in 5000 10000 20000 40000; do run tac "$TAC" $kc $kd; done
done
# PGC: anchor points on the SAME node/threads for a clean TAC-vs-PGC comparison
for kc in 40 80; do
  for kd in 10000 20000; do run pgc "$PGC" $kc $kd; done
done
echo "=== QPS sweep done $(date) ==="
