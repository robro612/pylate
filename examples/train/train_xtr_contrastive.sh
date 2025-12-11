#!/bin/bash
#SBATCH --array=0-1
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --output=logs/train_xtr_contrastive_%A_%a.out
#SBATCH --error=logs/train_xtr_contrastive_%A_%a.err

python examples/train/xtr_unified.py \
    --training_method contrastive \
    --model_name "Alibaba-NLP/gte-modernbert-base" \
    --query_length 32 \
    --doc_length 300 \
    --batch_size 196 \
    --grad_acc_steps 2 \
    --k_prime 128 \
    --use_normalizer_Z \
    --Z_clamp_value 1.0 \
    --start_normalizer_Z_at_step 0 \
    --lr 0.00001
    --warmup_ratio 0.0 \
    --use_triplet_evaluator \
    --eval_steps 1000 \
    --save_steps 10000 \
    --run_name "contrastive-train-${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID}"