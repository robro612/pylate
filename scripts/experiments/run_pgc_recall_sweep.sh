#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# PGC assignment-recall sweep: graph quality (iter_hnsw_m, iter_ef_construction)
# x search depth (iter_ef_search). Tests whether raising approximate-NN recall@1
# of the centroid assignment lifts retrieval quality toward flat-kmeans/PLAID.
#
# Iso: sample_multiplier=40, n_iter=5, assign_topm=1 (hard), no lambda, total_centroids=null.
# Empty-anchor count doubles as a recall proxy (missed nearest -> spurious empties).
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/fiqa/test}"
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/pgc_recall"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/pgc_recall"

PGC="index=tachiom index/clustering=pgc"
CONFIGS=(
  "g_efs50|"
  "g_efs200|index.clustering.iter_ef_search=200"
  "g_efs600|index.clustering.iter_ef_search=600"
  "g_m32e600_efs50|index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600"
  "g_m32e600_efs200|index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600 index.clustering.iter_ef_search=200"
  "g_m32e600_efs600|index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600 index.clustering.iter_ef_search=600"
  "g_m48e600_efs200|index.clustering.iter_hnsw_m=48 index.clustering.iter_ef_construction=600 index.clustering.iter_ef_search=200"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=64G -t 4:00:00 -J "rc_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $PGC $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/rc_${tag} \
      output.results_file=results/pgc_recall/${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== PGC RECALL SWEEP DONE $(date) ==="
