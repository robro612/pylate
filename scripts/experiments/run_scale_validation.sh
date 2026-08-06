#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Scale validation: optimized vs default clustering+PQ config, clean & iso.
#
# 4 runs (build3 phase timers, GTE model, total_centroids=null so all iso-centroid):
#   tac_default : pq_sample=10M, pq_iter=10
#   tac_opt     : pq_sample=4M,  pq_iter=10                          (optimized PQ)
#   pgc_default : sample_mult=null, pq_sample=10M, pq_iter=10
#   pgc_opt     : sample_mult=40,   pq_sample=4M,  pq_iter=10        (optimized PGC + PQ)
#
# Hypothesis: opt matches default quality (nDCG@10/recall@100) at materially
# faster clustering (PGC mult=40) and PQ-train (4M cap, scale-invariant).
#
# Usage:  scripts/run_scale_validation.sh <dataset>   e.g. beir/trec-covid
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:?usage: run_scale_validation.sh <dataset_id>}"
DS_SLUG=$(echo "$DATASET" | tr '/' '_')
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/scale_val"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/scale_val"

COMMON="model=$MODEL index=tachiom index.total_centroids=null datasets=\"[$DATASET]\""
# tag | overrides
CONFIGS=(
  "tac_default|index/clustering=tac index.clustering.n_iter=5 index.pq_sample_size=10000000 index.pq_n_iter=10"
  "tac_opt|index/clustering=tac index.clustering.n_iter=5 index.pq_sample_size=4000000 index.pq_n_iter=10"
  "pgc_default|index/clustering=pgc index.clustering.n_iter=5 index.clustering.sample_multiplier=null index.pq_sample_size=10000000 index.pq_n_iter=10"
  "pgc_opt|index/clustering=pgc index.clustering.n_iter=5 index.clustering.sample_multiplier=40 index.pq_sample_size=4000000 index.pq_n_iter=10"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${DS_SLUG}_${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=128G -t 8:00:00 -J "sv_${DS_SLUG}_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      $COMMON $ov \
      output.index_folder=indexes/sv_${DS_SLUG}_${tag} \
      output.results_file=results/scale_val/${DS_SLUG}_${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== SCALE VALIDATION DONE for $DATASET $(date) ==="
