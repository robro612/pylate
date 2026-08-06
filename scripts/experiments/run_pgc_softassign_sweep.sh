#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# PGC soft top-m assignment sweep (training-only soft centroid update).
#
# Iso-everything-else: sample_multiplier=40, iter_hnsw_m=16, iter_ef_construction=200,
# iter_ef_search=50 (the codified pgc defaults). Vary ONLY assign_topm x assign_temp.
# m=1 is the hard-top-1 baseline (should reproduce c_s40). Temperatures are kept
# small because top-m cosine neighbours are near-equidistant (large tau -> uniform).
#
# fiqa screen first; scale the winner (if it beats m=1 iso) to trec-covid / lotte.
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/fiqa/test}"
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/pgc_softassign"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/pgc_softassign"

PGC="index=tachiom index/clustering=pgc"   # pgc.yaml: sample_multiplier=40, assign_topm=1 default
CONFIGS=(
  "sa_m1|"                                                                                  # hard top-1 baseline
  "sa_m2_t03|index.clustering.assign_topm=2 index.clustering.assign_temp=0.03"
  "sa_m4_t01|index.clustering.assign_topm=4 index.clustering.assign_temp=0.01"
  "sa_m4_t03|index.clustering.assign_topm=4 index.clustering.assign_temp=0.03"
  "sa_m4_t10|index.clustering.assign_topm=4 index.clustering.assign_temp=0.1"
  "sa_m8_t03|index.clustering.assign_topm=8 index.clustering.assign_temp=0.03"
  "sa_m8_t10|index.clustering.assign_topm=8 index.clustering.assign_temp=0.1"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=64G -t 4:00:00 -J "sa_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $PGC $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/sa_${tag} \
      output.results_file=results/pgc_softassign/${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== PGC SOFT-ASSIGN SWEEP DONE $(date) ==="
