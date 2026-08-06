#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Retrieve-only eval against the VANILLA TAC lotte index (built by
# build_lotte_tac_external.sh from clusterings/lotte_tac). Mirrors search_one.sh
# (the PGC retrieve harness) exactly, only swapping the clustering dir, so TAC-vs-PGC
# retrieval numbers are directly comparable. Usage:
#   scripts/search_one_tac.sh <results_tag> <hydra-override>...
set -u
PROJ=/exp/rjha/pylate-pgc
CL="$PROJ/clusterings/lotte_tac"
TAG="$1"; shift
LOG="$PROJ/logs/scale_val/lotte_search_tac_${TAG}.log"
srun -u -p cpu -t 4:00:00 --cpus-per-task=24 --mem=48G -J "st_${TAG}" \
  bash -c "cd $PROJ && uv run --no-sync python scripts/benchmark_indexes.py \
    model=lateon_regularized index=tachiom index/clustering=external \
    index.clustering.centroids_path='$CL/centroids.npy' \
    index.clustering.assignments_path='$CL/assignments.npy' \
    index.pq_subspaces=32 \
    stages='[retrieve]' datasets='[lotte/pooled/dev/search]' \
    output.index_folder=indexes \
    output.results_file=results/scale_val/lotte_search_tac_${TAG}.jsonl \
    output.runs_dir=null $*" \
  > "$LOG" 2>&1
