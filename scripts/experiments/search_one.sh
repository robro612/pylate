#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Submit ONE retrieve-only tachiom search run as its own srun job (for parallel fan-out;
# cpu partition allows ~8 concurrent). Usage:
#   scripts/search_one.sh <results_tag> <hydra-override>...
# Fixed: M=32 PGC index, lotte/pooled/dev/search, alpha/kc/kd/ef passed as overrides.
set -u
PROJ=/exp/rjha/pylate-pgc
CL="$PROJ/clusterings/lotte_pgc_m4t05"
TAG="$1"; shift
LOG="$PROJ/logs/scale_val/lotte_search_pgc_${TAG}.log"
# Retrieve-only is light: measured MaxRSS ~18.5GB (M=32) / ~26GB (M=64), so 48G is safe
# (~2.6x M=32). Cores-only would default to a 2G flat allocation here (no DefMemPerCPU
# on the cluster) and OOM, so --mem is required. The binding cap for parallelism is the
# QOS cpu_limit (cpu=240, mem=1.07TB/user): at 24 cores/job, 8 jobs = 192c/384G fit easily.
srun -u -p cpu -t 4:00:00 --cpus-per-task=24 --mem=48G -J "s_${TAG}" \
  bash -c "cd $PROJ && uv run --no-sync python scripts/benchmark_indexes.py \
    model=lateon_regularized index=tachiom index/clustering=external \
    index.clustering.centroids_path='$CL/centroids.npy' \
    index.clustering.assignments_path='$CL/assignments.npy' \
    index.pq_subspaces=32 \
    stages='[retrieve]' datasets='[lotte/pooled/dev/search]' \
    output.index_folder=indexes \
    output.results_file=results/scale_val/lotte_search_pgc_${TAG}.jsonl \
    output.runs_dir=null $*" \
  > "$LOG" 2>&1
