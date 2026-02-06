#!/usr/bin/env bash
#SBATCH --job-name=exp1_index
#SBATCH --array=0-2
#SBATCH --output=logs/exp1_index_%a.out
#SBATCH --error=logs/exp1_index_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=450G

set -euo pipefail

# Array of "model:retrieval_mode:store_embeddings" tuples
# Format: "path/to/model:xtr:true" or "path/to/model:colbert:false"
CONFIGS=(
    # "output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/checkpoint-50000:xtr:true"
    # "output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000:colbert:true"
    # "lightonai/GTE-ModernColBERT-v1:colbert:true"
    # "robro612/xtr-base-en-pylate:xtr:false"
    # k_train variants XTR only
    "output/experiment_1_contrastive_xtr_primeqa_kprime_64_bs196_50k_bugfix/checkpoint-50000:xtr:false"
    "output/experiment_1_contrastive_xtr_primeqa_kprime_256_bs196_50k_bugfix/checkpoint-50000:xtr:false"
    "output/experiment_1_contrastive_xtr_primeqa_kprime_512_bs196_50k_bugfix/checkpoint-50000:xtr:false"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}

# Validate array index
if [ $IDX -ge ${#CONFIGS[@]} ]; then
    echo "ERROR: Array index $IDX out of range (only ${#CONFIGS[@]} configs defined)"
    exit 1
fi

CONFIG=${CONFIGS[$IDX]}

# Parse the tuple (split on :)
IFS=':' read -r MODEL_NAME RETRIEVAL_MODE STORE_EMBEDDINGS <<< "$CONFIG"

echo "================================================"
echo "Experiment 1: Index & Encode Docs"
echo "================================================"
echo "Running config $((IDX+1))/${#CONFIGS[@]}"
echo "Model: ${MODEL_NAME}"
echo "Retrieval mode: ${RETRIEVAL_MODE}"
echo "Store embeddings: ${STORE_EMBEDDINGS}"
echo "Datasets: beir/nfcorpus/test, beir/trec-covid, beir/nq"
echo "================================================"

python eval_model_irds_v3.py \
  --config-path "conf/eval" \
  --config-name "experiment_1_index" \
  "model.name_or_path=[${MODEL_NAME}]" \
  "stages=[encode_queries, encode_docs, build_index]" \
  "index.scann.store_embeddings=${STORE_EMBEDDINGS}"

echo "================================================"
echo "Done! Indexes built for model ${MODEL_NAME}"
echo "================================================"
