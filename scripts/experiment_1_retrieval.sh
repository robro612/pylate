#!/usr/bin/env bash
#SBATCH --job-name=exp1_retrieval
#SBATCH --array=0-2%3
#SBATCH --output=logs/exp1_retrieval_%j_%a.out
#SBATCH --error=logs/exp1_retrieval_%j_%a.err
#SBATCH --time=48:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=450G

set -euo pipefail

# Array of "model:retrieval_mode" pairs
# Format: "path/to/model:xtr" or "path/to/model:colbert"
CONFIGS=(
    # Main models
    # "output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/checkpoint-50000:xtr"
    # "output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/checkpoint-50000:colbert"
    # "output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000:xtr"
    # "output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000:colbert"
    # k_train variants XTR only
    "output/experiment_1_contrastive_xtr_primeqa_kprime_64_bs196_50k_bugfix/checkpoint-50000:xtr"
    "output/experiment_1_contrastive_xtr_primeqa_kprime_256_bs196_50k_bugfix/checkpoint-50000:xtr"
    "output/experiment_1_contrastive_xtr_primeqa_kprime_512_bs196_50k_bugfix/checkpoint-50000:xtr"
    # # SOTA ColBERT model both configs
    # "lightonai/GTE-ModernColBERT-v1:colbert"
    # "robro612/xtr-base-en-pylate:colbert"
    # # Google-XTR model
    # "robro612/xtr-base-en-pylate:xtr"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Validate array index
if [ $IDX -ge ${#CONFIGS[@]} ]; then
    echo "ERROR: Array index $IDX out of range (only ${#CONFIGS[@]} configs defined)"
    exit 1
fi

CONFIG=${CONFIGS[$IDX]}

# Parse the tuple
MODEL_NAME=${CONFIG%:*}
RETRIEVAL_MODE=${CONFIG##*:}

echo "Running config $((IDX+1))/${#CONFIGS[@]}"
echo "Model: ${MODEL_NAME}"
echo "Retrieval mode: ${RETRIEVAL_MODE}"
echo "================================================"

python eval_model_irds_v3.py \
  --config-path "conf/eval" \
  --config-name "experiment_1_retrieval_${RETRIEVAL_MODE}" \
  "model.name_or_path=[${MODEL_NAME}]"
