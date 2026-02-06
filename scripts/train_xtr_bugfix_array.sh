#!/usr/bin/env bash
#SBATCH --job-name=xtr_bugfix_array
#SBATCH --output=logs/xtr_bugfix_array_%a.out
#SBATCH --error=logs/xtr_bugfix_array_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-1

set -euo pipefail

# K_PRIME values to sweep
K_PRIMES=(64 256)

# Select k_prime based on array task ID
K_PRIME="${K_PRIMES[$SLURM_ARRAY_TASK_ID]}"
RUN_NAME="experiment_1_contrastive_xtr_primeqa_kprime_${K_PRIME}_bs196_50k_bugfix"
MODEL_PATH="output/${RUN_NAME}/checkpoint-50000"

echo "================================================"
echo "Training XTR with PrimeQA Loss (Bugfix)"
echo "================================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "K_PRIME: ${K_PRIME}"
echo "Run Name: ${RUN_NAME}"
echo "================================================"

python examples/train/xtr_unified_hydra.py \
  loss="contrastive_xtr_primeqa" \
  loss.XTRPrimeQA.k_prime=${K_PRIME} \
  optimization.batch_size=196 \
  optimization.warmup_ratio=0.01 \
  optimization.num_train_steps=50000 \
  optimization.lr=3e-5 \
  run.name="${RUN_NAME}"

echo "================================================"
echo "Training complete for k_prime=${K_PRIME}"
echo "Model saved to: ${MODEL_PATH}"
echo "================================================"
