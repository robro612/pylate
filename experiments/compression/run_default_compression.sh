#!/bin/bash

#SBATCH --job-name=li-compression
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00
#SBATCH --output=li_compression_%j.out
#SBATCH --error=li_compression_%j.err

set -euo pipefail

PROJECT_ROOT="/home/hltcoe/rjha/rjha_exp/pylate"
cd "${PROJECT_ROOT}"

RUN_ID=$(date +"%Y%m%d_%H%M%S")
OUTPUT_DIR="results/compression_experiments/compression_experiment_${RUN_ID}"

mkdir -p "${OUTPUT_DIR}"

python -u experiments/compression/compression_experiment.py \
  --dataset_name fiqa \
  --model_name lightonai/GTE-ModernColBERT-v1 \
  --index_type plaid \
  --experiment_output_dir "${OUTPUT_DIR}"

