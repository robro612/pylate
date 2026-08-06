#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# RUN 4 — Search sweep per built index (RUN 3), NO rebuild. For each M in {32,64},
# retrieve-only over k_docs_to_score in {2000,5000,10000} with alpha=null. Find the
# config that reaches PLAID parity (nDCG@10 0.5468 / recall@100 0.8087) and isolate
# the precision-lever contribution (M=64 vs M=32).
#
# clustering=external + centroids/assignments paths are still passed at retrieve time:
# they satisfy external.yaml's required fields and produce the ..._tachiom_external
# index name that matches the RUN 3 folders. pq_subspaces MUST match the built M.
# Depends on RUN 3. REUSE-SAFE: stages=[retrieve] only.
set -u
PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
CL="$PROJ/clusterings/lotte_pgc_m4t05"
mkdir -p "$PROJ/logs/scale_val" "$PROJ/results/scale_val"

for M in 64 32; do
  IDXDIR="indexes/bench_lightonai_LateOn-regularized_lotte_pooled_dev_search_tachiom_external_lotte_pgc_m4t05_m${M}"
  if [ ! -d "$PROJ/$IDXDIR" ]; then
    echo "skip M=$M: index dir $IDXDIR missing (run build_lotte_external_M.sh first)" >&2
    continue
  fi
  LOG="$PROJ/logs/scale_val/lotte_search_ext_m${M}.log"
  srun -u -p cpu -t 8:00:00 --cpus-per-task=64 --mem=240G -J "lt_srch_m${M}" \
    bash -c "cd $PROJ && \
    for kd in 2000 5000 10000; do \
      echo \"=== M=$M k_docs=\$kd  \$(date) ===\"; \
      uv run --no-sync python scripts/benchmark_indexes.py \
        model=$MODEL index=tachiom index/clustering=external \
        index.clustering.centroids_path='$CL/centroids.npy' \
        index.clustering.assignments_path='$CL/assignments.npy' \
        index.pq_subspaces=$M index.k_centroids=20 index.ef_search=40 \
        index.k_docs_to_score=\$kd index.alpha=null \
        stages=\"[retrieve]\" datasets=\"[$DATASET]\" \
        output.index_folder=indexes \
        output.results_file=results/scale_val/lotte_search_ext_m${M}_kd\${kd}.jsonl \
        output.runs_dir=null; \
    done" \
    > "$LOG" 2>&1 &
  echo "launched search M=$M -> $LOG"
  sleep 2
done
wait
echo "=== RUN4 search sweep done $(date) ==="
