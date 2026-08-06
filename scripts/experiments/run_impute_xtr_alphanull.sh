#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Isolation run: same impute_missing A/B, but with alpha=null (no candidate
# pruning). This removes the alpha-threshold confound — with alpha active, the
# imputation's dropped Sum(m_i) constant shifts prune_by_alpha's threshold and
# over-prunes, conflating the genuine XTR re-weighting with a pruning artifact.
# With alpha=null the candidate set is exactly the top-k_docs_to_score by the
# (re-weighted) coarse score, fully reranked, so any delta is pure re-weighting.
#
# Extension already rebuilt by run_impute_xtr_test.sh; no recompile here.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs

RESULTS="results_impute_xtr_alphanull.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS} index.alpha=null"

run() {
  local tag="$1"; shift
  echo; echo "----------------------------------------------------------------"
  echo "[$(date)] RUN: ${tag}"; echo "  overrides: $*"
  echo "----------------------------------------------------------------"
  uv run --no-sync python scripts/benchmark_indexes.py "$@"
}

LOTTE="${BASE} datasets=[lotte/pooled/dev/search] index/clustering=external"
LOTTE="${LOTTE} index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy"
LOTTE="${LOTTE} index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy"
run "lotte alpha=null impute=false" $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "lotte alpha=null impute=true"  $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=true

MSM="${BASE} datasets=[beir/msmarco/dev] index/clustering=external"
MSM="${MSM} index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy"
MSM="${MSM} index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
run "msmarco alpha=null impute=false" $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "msmarco alpha=null impute=true"  $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=true

echo; echo "[$(date)] DONE. Results -> ${RESULTS}"
