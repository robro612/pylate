#!/usr/bin/env bash
#SBATCH --job-name=xtr_bugfix
#SBATCH --output=logs/xtr_bugfix_%j.out
#SBATCH --error=logs/xtr_bugfix_%j.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1

set -euo pipefail

K_PRIME=128
RUN_NAME="experiment_1_contrastive_xtr_primeqa_kprime_${K_PRIME}_bs196_50k_bugfix"
MODEL_PATH="output/${RUN_NAME}/checkpoint-50000"


python examples/train/xtr_unified_hydra.py \
  loss="contrastive_xtr_primeqa" \
  loss.XTRPrimeQA.k_prime=${K_PRIME} \
  optimization.batch_size=196 \
  optimization.warmup_ratio=0.01 \
  optimization.num_train_steps=50000 \
  optimization.lr=3e-5 \
  run.name="${RUN_NAME}"

python eval_model_irds_v3.py \
  retrieve="xtr" \
  "model.name_or_path=[${MODEL_PATH}]" \
  encode.batch_size=2000
