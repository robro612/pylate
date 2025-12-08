#!/bin/bash

#SBATCH --job-name=li-compression
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/eval_%j.out
#SBATCH --error=work_dirs/slurm/eval_%j.err
#SBATCH --time=24:00:00

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

PROJECT_ROOT="./"
cd "${PROJECT_ROOT}"

DATASET_NAME="$1"
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
MODEL_NAME_SANITIZED=$(echo "${MODEL_NAME}" | tr "/" "_")
INDEX_TYPE="plaid"
OUTPUT_DIR="results/compression_experiments/${MODEL_NAME_SANITIZED}/${DATASET_NAME}/experiment_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"

COMMAND="""python experiments/compression/compression_experiment.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --index_type "${INDEX_TYPE}" \
  --experiment_output_dir "${OUTPUT_DIR}"
"""

echo "${COMMAND}"
eval "${COMMAND}"

echo "Compression experiment completed successfully"

