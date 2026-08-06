#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Optimize soft top-m assignment at scale + seed-confirm the m2_t03 win.
# Iso to pgc_opt: sample_multiplier=40, pq_sample_size=4M (codified), m16/efc200/efs50,
# n_iter=5, total_centroids=null. Vary assign_topm x assign_temp (+ one seed repeat).
#
# Baselines (trec-covid, already have): pgc_opt m1 = 0.7123/0.0805/12.9% empty;
#                                       sa_m2_t03 seed42 = 0.7219/0.0897/6.1% empty.
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/trec-covid}"
DS_SLUG=$(echo "$DATASET" | tr '/' '_')
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/scale_val"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/scale_val"

PGC="index=tachiom index/clustering=pgc index.total_centroids=null"
CONFIGS=(
  "sa_m2_t02|index.clustering.assign_topm=2 index.clustering.assign_temp=0.02"
  "sa_m2_t05|index.clustering.assign_topm=2 index.clustering.assign_temp=0.05"
  "sa_m4_t02|index.clustering.assign_topm=4 index.clustering.assign_temp=0.02"
  "sa_m4_t05|index.clustering.assign_topm=4 index.clustering.assign_temp=0.05"
  "sa_m2_t03_s43|index.clustering.assign_topm=2 index.clustering.assign_temp=0.03 index.clustering.seed=43"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${DS_SLUG}_pgc_${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=128G -t 8:00:00 -J "so_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $PGC $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/so_${DS_SLUG}_${tag} \
      output.results_file=results/scale_val/${DS_SLUG}_pgc_${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== SOFT-ASSIGN OPT SWEEP DONE for $DATASET $(date) ==="
