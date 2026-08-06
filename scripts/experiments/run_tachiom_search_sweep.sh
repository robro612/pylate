#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Tachiom SEARCH-param sweep (no rebuild) — chase the PGC/tachiom -> PLAID downstream gap.
# We proved clustering isn't the lever (exact kmeans == PGC through tachiom); the gap is
# the retrieval path. Build ONE index, then re-run retrieve-only with varying search params.
#
# Base index: external exact-kmeans centroids (best-case clustering, removed as a variable).
# Phase 1 (recall-first): widen k_centroids (PLAID nprobe analog), scale ef_search, deep
#   k_docs_to_score, no alpha pruning -> find the quality ceiling vs PLAID (0.8278 / r@100 0.1666).
#
# trec-covid / LateOn-regularized. One srun: build (~9min) then N fast retrieve-only runs.
set -u

PROJ=/exp/rjha/pylate-pgc
CD=$PROJ/indexes/km_treccovid_torch_v100        # external centroids/assignments
IDXF=$PROJ/indexes/sweep_search_tc              # persisted base index
EXT="index/clustering=external index.clustering.centroids_path=$CD/centroids.npy index.clustering.assignments_path=$CD/assignments.npy"
LOGDIR=$PROJ/logs/tachiom_search; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/tachiom_search"

srun -u -p cpu --cpus-per-task=32 --mem=128G -t 8:00:00 -J tsearch_tc bash -c "
  cd $PROJ && source $PYLATE_VENV_PATH/bin/activate
  echo '=== BUILD (persist, no delete) ==='
  python scripts/benchmark_indexes.py model=lateon_regularized index=tachiom $EXT \
    datasets='[beir/trec-covid]' stages='[build_index]' \
    output.index_folder=$IDXF output.runs_dir=null \
    output.results_file=results/tachiom_search/_build.jsonl
  # ── Phase 1: widen k_centroids (recall-first downstream) ──
  for kc in 20 40 80 160; do
    efs=\$((kc*2))
    echo \"=== retrieve k_centroids=\$kc ef_search=\$efs ===\"
    python scripts/benchmark_indexes.py model=lateon_regularized index=tachiom $EXT \
      datasets='[beir/trec-covid]' stages='[retrieve]' \
      index.k_centroids=\$kc index.ef_search=\$efs index.k_docs_to_score=2000 index.alpha=null \
      output.index_folder=$IDXF output.runs_dir=null \
      output.results_file=results/tachiom_search/kc\${kc}.jsonl
  done
  echo '=== SWEEP DONE ==='
" 2>&1 | tee "$LOGDIR/sweep_tc.log"
