#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# PGC clustering hyperparameter sweep on a single (small) dataset.
#
# One CPU slurm job per config: each builds a tachiom index with the given PGC
# clustering params and runs E2E retrieval, so we get both LATENCY (build_time,
# plus the verbose "PGC complete in ..." line in the log) and ACCURACY
# (ndcg@10, recall@100) for the latency/accuracy tradeoff.
#
# Each job is isolated: unique index_folder + results_file (parallel-safe).
# Merge afterwards with:  cat results/pgc_sweep/*.jsonl > results/pgc_sweep_all.jsonl
#
# Usage:  scripts/run_pgc_sweep.sh [dataset]      # default beir/fiqa/test
#         MODEL=gte_moderncolbert scripts/run_pgc_sweep.sh beir/fiqa/test
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="${1:-beir/fiqa/test}"
MODEL="${MODEL:-gte_moderncolbert}"
LOGDIR="$PROJ/logs/pgc_sweep"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/pgc_sweep"

# tag | clustering overrides
# Axes (author's advice in parens):
#   sample_multiplier: null(all) vs 5/20            -- dominant cost lever
#   iter_hnsw_m 16->32, iter_ef_construction ->600  (build is cheap, go high)
#   iter_ef_search down + iter_lambda early-exit     (search dominates; efs in [k,k+20], lambda [0.01,0.2])
CONFIGS=(
  "tac|index/clustering=tac index.clustering.n_iter=5"
  "pgc_base|index/clustering=pgc"
  "pgc_s5|index/clustering=pgc index.clustering.sample_multiplier=5"
  "pgc_s20|index/clustering=pgc index.clustering.sample_multiplier=20"
  "pgc_s5_m32_efc600|index/clustering=pgc index.clustering.sample_multiplier=5 index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600"
  "pgc_s5_efs20|index/clustering=pgc index.clustering.sample_multiplier=5 index.clustering.iter_ef_search=20"
  "pgc_s5_efs20_l05|index/clustering=pgc index.clustering.sample_multiplier=5 index.clustering.iter_ef_search=20 index.clustering.iter_lambda=0.05"
  "pgc_s5_efs10_l10|index/clustering=pgc index.clustering.sample_multiplier=5 index.clustering.iter_ef_search=10 index.clustering.iter_lambda=0.1"
  "pgc_fast|index/clustering=pgc index.clustering.sample_multiplier=5 index.clustering.iter_hnsw_m=32 index.clustering.iter_ef_construction=600 index.clustering.iter_ef_search=20 index.clustering.iter_lambda=0.05"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/${tag}.log"
  srun -u -p cpu --cpus-per-task=32 --mem=64G -t 4:00:00 -J "pgcsw_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL index=tachiom $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/sweep_${tag} \
      output.results_file=results/pgc_sweep/${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 1
done
wait
echo "=== ALL PGC SWEEP JOBS DONE $(date) ==="
