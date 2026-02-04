#!/usr/bin/env bash
#SBATCH --job-name=exp_4_rank_analysis
#SBATCH --output=logs/exp_4_rank_analysis_%a.out
#SBATCH --error=logs/exp_4_rank_analysis_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --array=0-4

set -euo pipefail

# Models from experiment 1
MODELS=(
  "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k/checkpoint-50000"
  "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k_bugfix/checkpoint-50000"
  "/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000"
  "lightonai/GTE-ModernColBERT-v1"
  "robro612/xtr-base-en-pylate"
)

# Select model based on array task ID
MODEL="${MODELS[$SLURM_ARRAY_TASK_ID]}"

echo "================================================"
echo "Token Rank Analysis - P(Gold | rank k)"
echo "================================================"
echo "Array Task ID: ${SLURM_ARRAY_TASK_ID}"
echo "Model: ${MODEL}"
echo "Dataset: beir/nfcorpus/test, beir/fiqa/test, beir/trec-covid"
echo "================================================"

python analyze_token_rank.py \
  "model.name_or_path=[${MODEL}]" \
  "dataset.names=[beir/nfcorpus/test, beir/fiqa/test, beir/trec-covid]" \
  analysis.k_token=4000 \
  analysis.relevance_threshold=1 \
  analysis.batch_size=32 \
  encode.batch_size=1000 \
  cache.enable=true \
  output.results_dir="results/experiment_4_token_rank_analysis" \
  output.save_data=true \
  output.figure_format=png

echo "================================================"
echo "Done! Results saved to results/token_rank_analysis/"
echo "================================================"
