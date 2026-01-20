#!/bin/bash
#SBATCH --job-name=colbert-variants
#SBATCH --output=logs/colbert-variants-%A_%a.out
#SBATCH --error=logs/colbert-variants-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --gres=gpu:h100:1
#SBATCH --array=0-8

set -euo pipefail

repo_root="/home/hltcoe/rjha/rjha_exp/pylate"
cd "${repo_root}"

configs=(
  "gte_modern_colbert"
  "gte_modern_colbert model=proxy_attention model.variant_args.num_proxy_tokens=32 model.variant_args.num_select_tokens=32 compile=false"
  "gte_modern_colbert model=constbert model.variant_args.constbert_seq_length=128"
  "gte_modern_colbert model=memory_token model.variant_args.num_memory_tokens=128"
  "gte_modern_colbert model=proxy_attention model.variant_args.num_proxy_tokens=24 model.variant_args.num_select_tokens=24 compile=false"
  "gte_modern_colbert model=constbert model.variant_args.constbert_seq_length=64"
  "gte_modern_colbert model=memory_token model.variant_args.num_memory_tokens=64"
  "gte_modern_colbert model=proxy_attention model.variant_args.num_proxy_tokens=16 model.variant_args.num_select_tokens=16 compile=false"
  "gte_modern_colbert model=constbert model.variant_args.constbert_seq_length=32"
  "gte_modern_colbert model=memory_token model.variant_args.num_memory_tokens=32"
)

config_args="${configs[${SLURM_ARRAY_TASK_ID}]}"
python examples/train/gte_modern_colbert_hydra.py \
  --config-name ${config_args} \
  ${EXTRA_ARGS:-} \
  "$@"
