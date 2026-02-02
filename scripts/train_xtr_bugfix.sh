#!/usr/bin/env bash
#SBATCH --job-name=xtr_bugfix
#SBATCH --output=logs/xtr_bugfix.out
#SBATCH --error=logs/xtr_bugfix.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1

set -euo pipefail

RUN_NAME="experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix"
MODEL_PATH="output/${RUN_NAME}/final"

python examples/train/xtr_unified_hydra.py \
  loss="contrastive_xtr_primeqa" \
  optimization.batch_size=196 \
  optimization.warmup_ratio=0.01 \
  run.name="${RUN_NAME}"

python eval_model_irds_v2.py \
  retrieve="xtr" \
  "model.name_or_path=[${MODEL_PATH}]" \
  encode.batch_size=2000
