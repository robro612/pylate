#!/usr/bin/env bash
#SBATCH --job-name=exp4_index_encode_nq
#SBATCH --output=logs/exp4_index_encode_nq_%a.out
#SBATCH --error=logs/exp4_index_encode_nq_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=450G
#SBATCH --array=0-3

set -euo pipefail

# Models from experiment 4
MODELS=(
  "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/checkpoint-50000"
  "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000"
  "lightonai/GTE-ModernColBERT-v1"
  "robro612/xtr-base-en-pylate"
)

STORE_EMBEDDINGS=(
  true
  true
  true
  false
)

RETRIEVAL=(
  "xtr"
  "colbert"
  "colbert"
  "xtr"
)

# Select model based on array task ID
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"
STORE_EMBEDDINGS="${STORE_EMBEDDINGS[$SLURM_ARRAY_TASK_ID]}"
RETRIEVAL="${RETRIEVAL[$SLURM_ARRAY_TASK_ID]}"

echo "================================================"
echo "Experiment 4: Index & Encode Queries for beir/nq"
echo "================================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Store embeddings: ${STORE_EMBEDDINGS}"
echo "Retrieval: ${RETRIEVAL}"
echo "Dataset: beir/nq"
echo "================================================"

python eval_model_irds_v3.py \
  --config-name=experiment_4_rank_analysis \
  "model.name_or_path=[${MODEL}]" \
  "dataset.names=[beir/nq]" \
  "stages=[encode_queries, build_index]" \
  "retrieve=${RETRIEVAL}" \
  "index.scann.store_embeddings=${STORE_EMBEDDINGS}" \
  "cache.enable_queries=true"

echo "================================================"
echo "Done! Index and queries encoded for model ${MODEL}"
echo "================================================"
