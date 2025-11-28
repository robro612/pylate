#!/bin/bash

#SBATCH --job-name=li-compression
#SBATCH --gres=gpu:l40s:1
#SBATCH --time=48:00:00
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
EMBEDDINGS_DIR="results/compression_experiments/${MODEL_NAME_SANITIZED}/${DATASET_NAME}/embeddings"

mkdir -p "${OUTPUT_DIR}"
mkdir -p "${EMBEDDINGS_DIR}"

COMMAND="""python -u experiments/compression/compression_experiment.py \
  --mode encode \
  --dataset_name "${DATASET_NAME}" \
  --model_name "${MODEL_NAME}" \
  --index_type "${INDEX_TYPE}" \
  --experiment_output_dir "${OUTPUT_DIR}" \
  --encoded_data_base_dir "${EMBEDDINGS_DIR}" \
  --batch_size 500 \
  --save_runfiles \
  --num_workers 1
"""

echo "${COMMAND}"
eval "${COMMAND}"

echo "Compression experiment completed successfully"

sruncpu --pty python experiments/compression/plot_results.py \
  --output_dir results/compression_experiments/plots/ \
  --metric recall@100  \
  --results_path /home/hltcoe/rjha/rjha_exp/pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/scidocs/experiment_113917/results_20251126_203125.jsonl