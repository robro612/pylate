#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH -J msmarco_encode
#SBATCH -o /exp/rjha/pylate-pgc/logs/scale_val/msmarco_encode.log
# Encode the MSMARCO passage corpus + dev queries with LateOn-regularized on an L40S
# (V100 can't run the cu13 torch build). fp16 / no-pool, token_ids on -> cache at
# embeddings_cache/beir_msmarco_dev/lightonai_LateOn-regularized/nopool_fp16/.
set -u
cd /exp/rjha/pylate-pgc
uv run --no-sync python scripts/benchmark_indexes.py \
  model=lateon_regularized index=tachiom \
  datasets='[beir/msmarco/dev]' stages='[encode_docs,encode_queries]' \
  output.runs_dir=null
