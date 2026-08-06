#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# RUN 1 — Coverage lever at LoTTE scale, NO rebuild.
# Retrieve-only sweep of k_docs_to_score over the EXISTING M=32 TAC index
# (bench_lightonai_lateon-fix-m_..._tachiom_tac, reused via the LateOn-regularized
# symlink). Tests whether the trec-covid coverage recovery (0.74->0.83) generalizes
# to lotte, where the prior PGC run (k_docs=500) trailed PLAID badly on recall
# (0.58 vs 0.81). alpha=null for max recall; ef_search=40, k_centroids=20.
#
# Target (PLAID baseline, already on disk): nDCG@10 0.5468 / recall@100 0.8087.
# REUSE-SAFE: stages=[retrieve] only — never builds, never deletes.
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"
LOG="$PROJ/logs/scale_val/lotte_coverage_sweep.log"

srun -u -p cpu -t 8:00:00 --cpus-per-task=64 --mem=240G -J lt_cov \
  bash -c "cd $PROJ && \
  for kd in 2000 5000 10000; do \
    echo \"=== k_docs_to_score=\$kd  \$(date) ===\"; \
    uv run --no-sync python scripts/benchmark_indexes.py \
      model=$MODEL index=tachiom index/clustering=tac \
      stages=\"[retrieve]\" datasets=\"[$DATASET]\" \
      index.pq_subspaces=32 index.k_centroids=20 index.ef_search=40 \
      index.k_docs_to_score=\$kd index.alpha=null \
      output.index_folder=indexes \
      output.results_file=results/scale_val/lotte_cov_tac_m32_kd\${kd}.jsonl \
      output.runs_dir=null; \
  done" \
  > "$LOG" 2>&1
echo "=== RUN1 coverage sweep done $(date) ==="
