#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Test the shift-invariant (gap-relative) alpha threshold, gated to the imputation
# path. At the production alpha=0.45:
#   - impute=false uses the historical magnitude band (gap_relative=false).
#   - impute=true  uses gap-relative pruning (gap_relative=true), which is invariant
#     to the imputation Sum(m_i) constant -> pruning should be FUNCTIONAL again.
# TACHIOM_PRUNE_DEBUG=1 makes batch_search emit one aggregate [PRUNE_DEBUG] line per
# run (to stderr/log) reporting candidates in/kept/pruned. Goal: confirm impute=true
# now prunes a healthy fraction (vs ~0% when the constant made the magnitude band
# inert) WHILE keeping quality near the alpha=null impute=true numbers
# (msmarco 0.4313/0.8466, lotte 0.5471/0.7828) and recovering some QPS.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs
export TACHIOM_PRUNE_DEBUG=1

echo "[$(date)] Rebuilding tachiom (gap-relative alpha + prune-debug)"
( cd /exp/rjha/tachiom && \
  VIRTUAL_ENV=$PYLATE_VENV_PATH \
  $PYLATE_VENV_PATH/bin/maturin develop --release )

RESULTS="results_impute_xtr_gaprel2.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS}"  # alpha defaults to 0.45

run() {
  local tag="$1"; shift
  echo; echo "---------------------------------------------------------------"
  echo "[$(date)] RUN: ${tag}"; echo "  overrides: $*"
  echo "---------------------------------------------------------------"
  uv run --no-sync python scripts/benchmark_indexes.py "$@"
}

LOTTE="${BASE} datasets=[lotte/pooled/dev/search] index/clustering=external"
LOTTE="${LOTTE} index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy"
LOTTE="${LOTTE} index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy"
run "lotte alpha=0.45 impute=false (magnitude band)" $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "lotte alpha=0.45 impute=true (gap-relative)"    $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=true

MSM="${BASE} datasets=[beir/msmarco/dev] index/clustering=external"
MSM="${MSM} index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy"
MSM="${MSM} index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
run "msmarco alpha=0.45 impute=false (magnitude band)" $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "msmarco alpha=0.45 impute=true (gap-relative)"    $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=true

echo; echo "[$(date)] DONE. Results -> ${RESULTS}"
echo "Prune behavior:"; grep "\[PRUNE_DEBUG\]" logs/impute_xtr_gaprel2.log || true
