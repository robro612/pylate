#!/bin/bash

#SBATCH --job-name=li-compression
#SBATCH --gres=gpu:a100:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=li_compression_%j.out
#SBATCH --error=li_compression_%j.err

set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 <dataset_name>"
  exit 1
fi

PROJECT_ROOT="/home/hltcoe/rjha/rjha_exp/pylate"
cd "${PROJECT_ROOT}"

DATASET_NAME="$1"
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
MODEL_NAME_SANITIZED=$(echo "${MODEL_NAME}" | tr "/" "_")
INDEX_TYPE="plaid"
OUTPUT_DIR="results/compression_experiments/${MODEL_NAME_SANITIZED}/${DATASET_NAME}/experiment_${SLURM_JOB_ID}"

mkdir -p "${OUTPUT_DIR}"

COMMAND="""python -u experiments/compression/compression_experiment.py \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --index_type "${INDEX_TYPE}" \
  --experiment_output_dir "${OUTPUT_DIR}"
"""

echo "${COMMAND}"
eval "${COMMAND}"

echo "Compression experiment completed successfully"

