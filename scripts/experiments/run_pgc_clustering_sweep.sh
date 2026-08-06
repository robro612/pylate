#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# PGC *clustering*-config sweep with a FIXED downstream serving HNSW.
#
# Goal: find the best PGC-internal clustering config (sampling + assignment
# fidelity + anchor-graph quality + iterations) for the centroids it produces,
# scored by the resulting index's retrieval quality. The serving HNSW over
# centroids is left at its defaults (hnsw_m=32, ef_construction=1500) and the
# centroid budget is iso (total_centroids=null -> same 65536 for all). QPS is
# not the objective here.
#
# No iter_lambda: the DistanceAdaptive early-exit is an accuracy *relaxation*
# (explores more at fixed ef_search), the wrong lever for k=1 assignment.
#
# Each config is an isolated CPU job (unique index_folder + results_file).
# Merge:  cat results/pgc_clust_sweep/*.jsonl > results/pgc_clust_sweep_all.jsonl
#
# Usage:  scripts/run_pgc_clustering_sweep.sh [dataset]   # default beir/fiqa/test
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/fiqa/test}"
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/pgc_clust_sweep"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/pgc_clust_sweep"

PGC="index=tachiom index/clustering=pgc"
# tag | extra clustering overrides
CONFIGS=(
  "c_null|"                                                                                   # best-quality reference (mult=null, m16, efc200, efs50)
  "c_s20|index.clustering.sample_multiplier=20"
  "c_s40|index.clustering.sample_multiplier=40"
  "c_s20_efs100|index.clustering.sample_multiplier=20 index.clustering.iter_ef_search=100"
  "c_s40_efs100|index.clustering.sample_multiplier=40 index.clustering.iter_ef_search=100"
  "c_s20_m32_efc600_efs100|index.clustering.sample_multiplier=20 index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600 index.clustering.iter_ef_search=100"
  "c_s40_m32_efc600|index.clustering.sample_multiplier=40 index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600"
  "c_s20_niter8|index.clustering.sample_multiplier=20 index.clustering.n_iter=8"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=64G -t 4:00:00 -J "pgcc_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $PGC $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/clust_${tag} \
      output.results_file=results/pgc_clust_sweep/${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== ALL PGC CLUSTERING-SWEEP JOBS DONE $(date) ==="
