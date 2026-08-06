#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Find the gap-relative operating alpha for impute=true that buys QPS at marginal
# quality loss (the analog of magnitude-band alpha=0.45 for no-impute).
#
# In the gap formula  threshold = s_k - alpha*(s_best - s_k),  HIGHER alpha => lower
# threshold => keep more => prune LESS. The frontier sweep only had {0.85,0.6,0.45}
# (44-77% pruned); the low-prune knee is at alpha >= 0.9 (and likely > 1.0). Sweep it.
#
# No Rust change since the frontier build -> no rebuild; current environment is correct.

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs
export TACHIOM_PRUNE_DEBUG=1
export TACHIOM_GAP_RELATIVE=1   # force gap-relative (impute=true would too; explicit)

RESULTS="results_impute_gap_alpha.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS}"
L="datasets=[lotte/pooled/dev/search] index/clustering=external index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy"
M="datasets=[beir/msmarco/dev] index/clustering=external index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
ST='stages=[encode_queries,retrieve]'

run() {
  local tag="$1"; shift
  echo; echo "==============================================================="
  echo "[$(date)] RUN: ${tag}"
  echo "==============================================================="
  uv run --no-sync python scripts/benchmark_indexes.py "$@"
}

# alpha grid in the low-prune region (impute=true, gap-relative)
for al in 0.9 1.0 1.2 1.5 2.0; do
  run "lotte   gap impute=true alpha=${al}" $BASE $L $ST index.impute_missing=true index.alpha=${al}
  run "msmarco gap impute=true alpha=${al}" $BASE $M $ST index.impute_missing=true index.alpha=${al}
done

echo; echo "[$(date)] DONE -> ${RESULTS}"
echo "=== prune (log order) ==="; grep -E "RUN:|PRUNE_DEBUG" logs/impute_gap_alpha.log || true
