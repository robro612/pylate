#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Pareto frontier: quality vs QPS, for two configs, sweeping each config's own alpha.
#   Baseline   : impute=false + magnitude band   (TACHIOM_GAP_RELATIVE=0)
#   Treatment  : impute=true  + gap-relative prune (TACHIOM_GAP_RELATIVE=1)
# Compare the two frontiers iso-QPS: if treatment gives higher quality at matched
# throughput across the range, imputation is a win independent of the alpha knob.
#
# alpha traces each curve (lower alpha -> prune more -> higher QPS). Ranges chosen so
# the two configs span overlapping QPS. Rows distinguished by index_config
# {impute_missing, alpha}; [PRUNE_DEBUG] lines (in log order) give prune fractions.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs
export TACHIOM_PRUNE_DEBUG=1

echo "[$(date)] Rebuilding tachiom"
( cd /exp/rjha/tachiom && \
  VIRTUAL_ENV=$PYLATE_VENV_PATH \
  $PYLATE_VENV_PATH/bin/maturin develop --release )

RESULTS="results_impute_frontier.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS}"
LDS="datasets=[lotte/pooled/dev/search] index/clustering=external index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy"
MDS="datasets=[beir/msmarco/dev] index/clustering=external index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
STAGES='stages=[encode_queries,retrieve]'

# args: <tag> <GAP> <dataset-overrides> <impute> <alpha>
run() {
  local tag="$1" gap="$2" dsov="$3" imp="$4" al="$5"
  echo; echo "==============================================================="
  echo "[$(date)] RUN: ${tag}  (GAP=${gap} impute=${imp} alpha=${al})"
  echo "==============================================================="
  TACHIOM_GAP_RELATIVE="${gap}" uv run --no-sync python scripts/benchmark_indexes.py \
    $BASE $dsov $STAGES index.impute_missing=${imp} index.alpha=${al}
}

for ds in L M; do
  if [ "$ds" = L ]; then DSOV="$LDS"; NAME=lotte; else DSOV="$MDS"; NAME=msmarco; fi
  # Baseline frontier: no-impute, magnitude band
  for al in null 0.45 0.2 0.1; do
    run "${NAME} BASE" 0 "$DSOV" false "$al"
  done
  # Treatment frontier: impute, gap-relative
  for al in null 0.85 0.6 0.45; do
    run "${NAME} TREAT" 1 "$DSOV" true "$al"
  done
done

echo; echo "[$(date)] DONE -> ${RESULTS}"
echo "=== prune (log order) ==="; grep -E "RUN:|PRUNE_DEBUG" logs/impute_frontier.log || true
