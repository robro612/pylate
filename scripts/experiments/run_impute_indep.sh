#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Independence demo: does turning imputation ON change how much alpha prunes?
# We hold the pruning FORMULA fixed (via TACHIOM_GAP_RELATIVE) and flip imputation.
#
#   GAP=1 (gap-relative, shift-invariant): impute=false vs true should prune the
#          SAME fraction and hit the SAME QPS  -> pruning is INDEPENDENT of imputation.
#   GAP=0 (magnitude band): impute=true goes ~inert (≈0% pruned) vs impute=false's
#          32-42% -> pruning is NOT independent (the original bug).
#
# Read the ordered [PRUNE_DEBUG] lines + QPS lines in the log; each RUN echoes its tag.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs
export TACHIOM_PRUNE_DEBUG=1

echo "[$(date)] Rebuilding tachiom (gap-relative + env override + prune-debug)"
( cd /exp/rjha/tachiom && \
  VIRTUAL_ENV=$PYLATE_VENV_PATH \
  $PYLATE_VENV_PATH/bin/maturin develop --release )

RESULTS="results_impute_indep.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS}"  # alpha=0.45

# args: <tag> <GAP value> <dataset-overrides...> <impute-override>
run() {
  local tag="$1"; local gap="$2"; shift 2
  echo; echo "==============================================================="
  echo "[$(date)] RUN: ${tag}   (TACHIOM_GAP_RELATIVE=${gap})"
  echo "==============================================================="
  TACHIOM_GAP_RELATIVE="${gap}" uv run --no-sync python scripts/benchmark_indexes.py "$@"
}

L="${BASE} datasets=[lotte/pooled/dev/search] index/clustering=external index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy stages=[encode_queries,retrieve]"
M="${BASE} datasets=[beir/msmarco/dev] index/clustering=external index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy stages=[encode_queries,retrieve]"

# --- gap-relative formula fixed: flip imputation (independence test) ---
run "lotte   GAP impute=false" 1 $L index.impute_missing=false
run "lotte   GAP impute=true"  1 $L index.impute_missing=true
run "msmarco GAP impute=false" 1 $M index.impute_missing=false
run "msmarco GAP impute=true"  1 $M index.impute_missing=true

# --- magnitude band fixed, impute=true: show it goes inert (contrast) ---
run "lotte   MAG impute=true"  0 $L index.impute_missing=true
run "msmarco MAG impute=true"  0 $M index.impute_missing=true

echo; echo "[$(date)] DONE."
echo "=== prune behavior (in run order) ==="
grep -E "RUN:|PRUNE_DEBUG" logs/impute_indep.log || true
