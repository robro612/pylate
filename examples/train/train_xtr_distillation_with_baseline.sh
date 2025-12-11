#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=/dev/null
#SBATCH --error=/dev/null

# Determine job type based on array task ID and set log file names
if [ $SLURM_ARRAY_TASK_ID -eq 0 ]; then
    # XTR job with Z arguments
    LOG_FILE="logs/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_xtr.out"
    python examples/train/xtr_distillation.py \
        --batch_size 12 \
        --grad_acc_steps 1 \
        --kd_minmax_norm \
        --use_normalizer_Z \
        --Z_clamp_value 1.0 \
        --warmup_ratio 0.0 \
        --model_name "Alibaba-NLP/gte-modernbert-base" \
        --lr 0.00001 \
        --save_steps 10000 > "${LOG_FILE}" 2>&1
else
    # ColBERT job
    LOG_FILE="logs/job-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}_colbert.out"
    python examples/train/xtr_distillation.py \
        --batch_size 12 \
        --grad_acc_steps 1 \
        --kd_minmax_norm \
        --use_colbert \
        --warmup_ratio 0.0 \
        --model_name "Alibaba-NLP/gte-modernbert-base" \
        --lr 0.00001 \
        --save_steps 10000 > "${LOG_FILE}" 2>&1
fi