#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# PQ-training work-reduction sweep, with TAC clustering (cheap + fixed) so the
# build delta is attributable to the PQ encoder-train phase.
#
# PQ train is already nested-parallel (M subspaces x parallel k-means); it's
# compute-bound, so the lever is *less work*: pq_sample_size and pq_n_iter.
# Baseline (sample=10M/all 7.7M, n_iter=10) is the existing tac_timed run
# (~315s PQ-train, nDCG@10 0.390). We sweep reduced settings here.
#
# Fixed: TAC clustering n_iter=5, iso-centroid (total_centroids=null), default
# serving HNSW. fiqa.
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/fiqa/test}"
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/pq_sweep"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/pq_sweep"

BASE="index=tachiom index/clustering=tac index.clustering.n_iter=5"
# tag | pq overrides
CONFIGS=(
  "pq_s4M_i10|index.pq_sample_size=4000000 index.pq_n_iter=10"
  "pq_s2M_i10|index.pq_sample_size=2000000 index.pq_n_iter=10"
  "pq_s1M_i10|index.pq_sample_size=1000000 index.pq_n_iter=10"
  "pq_s2M_i5|index.pq_sample_size=2000000 index.pq_n_iter=5"
  "pq_s10M_i5|index.pq_sample_size=10000000 index.pq_n_iter=5"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=64G -t 2:00:00 -J "pq_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $BASE $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/pq_${tag} \
      output.results_file=results/pq_sweep/${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== ALL PQ SWEEP JOBS DONE $(date) ==="
