#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Re-test the imputation after adding the Sum(m_i) constant back, so it is now
# compatible with alpha-pruning. Runs the DEFAULT alpha=0.45 A/B (the config that
# cratered with the constant dropped). Expectation: impute=true no longer loses
# recall vs impute=false. The impute=false rows should match the earlier
# results_impute_xtr.jsonl alpha=0.45 baselines (constant is a no-op when off) —
# a useful sanity check.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs

echo "[$(date)] Rebuilding tachiom (constant-preserving imputation)"
( cd /exp/rjha/tachiom && \
  VIRTUAL_ENV=$PYLATE_VENV_PATH \
  $PYLATE_VENV_PATH/bin/maturin develop --release )

RESULTS="results_impute_xtr_const.jsonl"
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
run "lotte alpha=0.45 impute=false" $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "lotte alpha=0.45 impute=true"  $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=true

MSM="${BASE} datasets=[beir/msmarco/dev] index/clustering=external"
MSM="${MSM} index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy"
MSM="${MSM} index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
run "msmarco alpha=0.45 impute=false" $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "msmarco alpha=0.45 impute=true"  $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=true

echo; echo "[$(date)] DONE. Results -> ${RESULTS}"
