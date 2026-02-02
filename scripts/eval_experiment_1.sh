#!/usr/bin/env bash
#SBATCH --job-name=eval_experiment_1
#SBATCH --array=0-1
#SBATCH --output=logs/eval_experiment_1_%a.out
#SBATCH --error=logs/eval_experiment_1_%a.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=450G

set -euo pipefail

RUN_NAMES=(
  "experiment_1_contrastive_colbert_bs196_50k"
  "experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k"
)

RETRIEVES=(
  "colbert"
  "xtr"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
RUN_NAME=${RUN_NAMES[$IDX]}
RETRIEVE=${RETRIEVES[$IDX]}

MODEL_PATH="output/${RUN_NAME}/checkpoint-10000"

python eval_model_irds_v2.py \
  retrieve="${RETRIEVE}" \
  "model.name_or_path=[${MODEL_PATH}]" \
  encode.batch_size=2000 \
  
