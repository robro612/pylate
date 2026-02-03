#!/usr/bin/env bash
#SBATCH --job-name=experiment_2_imputation
#SBATCH --array=0-39%8
#SBATCH --output=logs/experiment_2_imputation_%a.out
#SBATCH --error=logs/experiment_2_imputation_%a.err
#SBATCH --time=48:00:00
#SBATCH --mem=200G
#SBATCH --gres=gpu:v100:1

set -euo pipefail

# Imputation methods
IMPUTATIONS=(
  "min"
  "mean"
  "percentile"
  "power_law"
  "zero"
)

# Model paths
MODELS=(
  "output/experiment_1_contrastive_colbert_bs196_50k/final"
  "output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/final"
)

# Datasets (ordered roughly by size, smallest first)
DATASETS=(
  "beir/nfcorpus/test"
  "beir/fiqa/test"
  "beir/trec-covid"
  "lotte/lifestyle/dev/search"
)

# Calculate indices from array task ID
# Ordering: imputation (outer) → model → dataset (inner)
# This ensures jobs with same (model, dataset) are 8 apart in the array
# With %8 limit, only one job per (model, dataset) runs at a time
#
# 5 imputations × 2 models × 4 datasets = 40 jobs (0-39)
IDX=${SLURM_ARRAY_TASK_ID:-0}
NUM_MODELS=${#MODELS[@]}
NUM_DATASETS=${#DATASETS[@]}
NUM_MODEL_DATASET=$((NUM_MODELS * NUM_DATASETS))  # 8

IMPUTATION_IDX=$((IDX / NUM_MODEL_DATASET))
REMAINDER=$((IDX % NUM_MODEL_DATASET))
MODEL_IDX=$((REMAINDER / NUM_DATASETS))
DATASET_IDX=$((REMAINDER % NUM_DATASETS))

IMPUTATION=${IMPUTATIONS[$IMPUTATION_IDX]}
MODEL_PATH=${MODELS[$MODEL_IDX]}
DATASET=${DATASETS[$DATASET_IDX]}

echo "=== Job ${IDX} ==="
echo "Imputation: ${IMPUTATION}"
echo "Model: ${MODEL_PATH}"
echo "Dataset: ${DATASET}"
echo ""

for K_TOKEN in 10000 40000; do
  echo "Running k_token=${K_TOKEN}"
  python eval_model_irds_v2.py \
    retrieve="xtr" \
    "retrieve/imputation=${IMPUTATION}" \
    "model.name_or_path=[${MODEL_PATH}]" \
    "retrieve.k_token=${K_TOKEN}" \
    "dataset.names=[${DATASET}]" \
    encode.batch_size=1000
  echo ""
done

echo "=== Job ${IDX} complete ==="
