#!/usr/bin/env bash
#SBATCH --job-name=exp_3_token_score_dist
#SBATCH --output=logs/exp_3_token_score_dist.out
#SBATCH --error=logs/exp_3_token_score_dist.err
#SBATCH --time=4:00:00
#SBATCH --gres=gpu:v100:1
#SBATCH --mem=64G

set -euo pipefail

# Models from experiment 1
XTR_MODEL="/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_xtr_primeqa_kprime_128_bs196_50k/checkpoint-50000"
COLBERT_MODEL="/home/hltcoe/rjha/rjha_exp/pylate-xtr/output/experiment_1_contrastive_colbert_bs196_50k/checkpoint-50000"
GOOGLE_XTR_MODEL="robro612/xtr-base-en-pylate"
SOTA_COLBERT_MODEL="lightonai/GTE-ModernColBERT-v1"

echo "================================================"
echo "Token Score Distribution Analysis"
echo "================================================"
echo "XTR Model: ${XTR_MODEL}"
echo "ColBERT Model: ${COLBERT_MODEL}"
echo "Dataset: beir/nfcorpus/test"
echo "================================================"

python analyze_token_scores.py \
  "model.name_or_path=[${XTR_MODEL},${COLBERT_MODEL}, ${GOOGLE_XTR_MODEL}, ${SOTA_COLBERT_MODEL}]" \
  "dataset.names=[beir/nfcorpus/test]" \
  retrieve.k=100 \
  retrieve.k_token=40000 \
  encode.batch_size=1000 \
  cache.enable=true \
  output.figures_dir="figures/token_score_distribution" \
  output.save_data=true \
  plot.format=png \
  plot.dpi=300

echo "================================================"
echo "Done! Figures saved to figures/token_score_distribution/"
echo "================================================"