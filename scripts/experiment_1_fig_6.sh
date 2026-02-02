#!/usr/bin/env bash
#SBATCH --job-name=experiment_1_fig_6
#SBATCH --array=0-1
#SBATCH --output=logs/experiment_1_fig_6_%a.out
#SBATCH --error=logs/experiment_1_fig_6_%a.err
#SBATCH --time=36:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=450G

set -euo pipefail

CONFIGS=(
    repro_fig6_xtr
    repro_fig6_colbert
)

STORE_EMBEDDINGS=(
    false
    true
)

MODEL_NAMES=(
    "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k/final"
    "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_colbert_bs196_50k/final"
)

IDX=${SLURM_ARRAY_TASK_ID:-0}
CONFIG=${CONFIGS[$IDX]}
MODEL_NAME=${MODEL_NAMES[$IDX]}
STORE_EMBEDDINGS_ARG=${STORE_EMBEDDINGS[$IDX]}
echo "Running config: ${CONFIG}"
echo "Running model: ${MODEL_NAME}"
echo "Store embeddings: ${STORE_EMBEDDINGS_ARG}"
echo "================================================"

python eval_model_irds_v2.py \
  --config-path "conf/eval" \
  --config-name "${CONFIG}" \
  "model.name_or_path=[${MODEL_NAME}]" \
  "index.save=true" \
  "index.scann.store_embeddings=${STORE_EMBEDDINGS_ARG}"