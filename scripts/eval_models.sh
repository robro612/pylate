#!/bin/bash
#SBATCH --job-name=mv_eval
#SBATCH --array=0-127
#SBATCH --gres=gpu:h100:1
#SBATCH --time=8:00:00
#SBATCH --output=logs/mv_eval_%A_%a.out

# DISABLED: SBATCH --mail-type=BEGIN,END,FAIL,ARRAY_TASKS
# DISABLED: SBATCH --mail-user=rjha5@jh.edu

# Submit from project root:
#   sbatch scripts/eval_models.sh
#
# task_id = dataset_idx * N_MODELS + model_idx
# dataset_idx: 0=nfcorpus  1=scifact  2=arguana  3=scidocs  4=fiqa
#              5=trec-covid  6=touche2020  7=quora  8=nq  9=msmarco
#              10=lotte/lifestyle  11=lotte/writing  12=lotte/recreation
#              13=lotte/technology  14=lotte/science  15=lotte/pooled
# model_idx:   0=colbert_kd  1=xtr_k512_kd  2=xtr_k256_kd  3=xtr_multi_kd
#              4=colbert_contrastive  5=xtr_k512_contrastive  6=xtr_k256_contrastive
#              7=xtr_multi_contrastive

set -e
mkdir -p logs

MODEL_PATHS=(
  "output/modernbert_colbert_kd/final"
  "output/modernbert_xtr_kd_k512/final"
  "output/modernbert_xtr_kd_k256/final"
  # "output/modernbert_xtr_kd_k128/final" k=128 diverged during training. No final model.
  "output/modernbert_xtr_kd_multik128-256-512/final"
  "output/modernbert_colbert_contrastive/final"
  "output/modernbert_xtr_contrastive_k512/final"
  "output/modernbert_xtr_contrastive_k256/final"
  "output/modernbert_xtr_contrastive_multik128-256-512/final"
)

DATASET_LIST=(
  "beir/nfcorpus/test"       # 3.6K : Done
  "beir/scifact/test"        # 5K : Done
  "beir/scidocs"             # 25K : Done
  "beir/fiqa/test"           # 57K : Done
  "beir/trec-covid"          # 171K : Done
  "beir/arguana"        # 8.7K : Done
  "beir/webis-touche2020/v2" # 382K : Done
  "beir/quora/test"          # 523K : Done
  "beir/nq"                  # 2.68M
  "beir/msmarco/dev"         # 8.84M
  "lotte/lifestyle/test/search"   # 120K # Done
  "lotte/writing/test/search"     # 200K # Done
  "lotte/recreation/test/search"  # 500K # Done
  "lotte/technology/test/search"  # 1.2M # Done
  "lotte/science/test/search"     # 1.7M # Done
  "lotte/pooled/test/search"      # 2.8M # Done
)

N_MODELS=${#MODEL_PATHS[@]}
DATASET_IDX=$(( SLURM_ARRAY_TASK_ID / N_MODELS ))
MODEL_IDX=$(( SLURM_ARRAY_TASK_ID % N_MODELS ))

MODEL_PATH=${MODEL_PATHS[$MODEL_IDX]}
DATASET=${DATASET_LIST[$DATASET_IDX]}

RESULTS=${RESULTS:-"results/ictir_eval.jsonl"}

COMMON_ARGS=(
  "model.name_or_path=$MODEL_PATH"
  "datasets=[$DATASET]"
  "output.results_file=$RESULTS"
  "encode.batch_size=6000" # change if we change gpu type
)

export CUDA_LAUNCH_BLOCKING=${CUDA_LAUNCH_BLOCKING:-0}

echo "=== [task $SLURM_ARRAY_TASK_ID] model: $MODEL_PATH  dataset: $DATASET ==="

# WARP first — encodes and caches documents; delete index but keep embeddings for PLAID
python scripts/benchmark_indexes.py "${COMMON_ARGS[@]}" index=warp \
  'stages=[encode_docs,encode_queries,build_index,retrieve,delete_index]'

# PLAID reuses cached document encodings; delete index and embeddings when done
python scripts/benchmark_indexes.py "${COMMON_ARGS[@]}" index=plaid \
  'stages=[build_index,retrieve,delete_index,delete_embeddings]'
