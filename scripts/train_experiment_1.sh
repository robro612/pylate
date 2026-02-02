#!/usr/bin/env bash
#SBATCH --job-name=experiment_1
#SBATCH --array=0-1
#SBATCH --output=logs/experiment_1_%a.out
#SBATCH --error=logs/experiment_1_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1

set -euo pipefail

LOSSES=(
  "contrastive_colbert"
  "contrastive_xtr_primeqa"
)

RUN_NAMES=(
  "experiment_1_contrastive_colbert_bs196_50k"
  "experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_primeqa_bugfix"
)

RETRIEVES=(
  "colbert"
  "xtr"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
LOSS=${LOSSES[$IDX]}
RUN_NAME=${RUN_NAMES[$IDX]}
RETRIEVE=${RETRIEVES[$IDX]}
MODEL_PATH="output/${RUN_NAME}/final"



python examples/train/xtr_unified_hydra.py \
  loss="${LOSS}" \
  optimization.batch_size=196 \
  optimization.warmup_ratio=0.01 \
  run.name="${RUN_NAME}"



python eval_model_irds_v2.py \
  retrieve="${RETRIEVE}" \
  "model.name_or_path=[${MODEL_PATH}]" \
  encode.batch_size=2000 \
  