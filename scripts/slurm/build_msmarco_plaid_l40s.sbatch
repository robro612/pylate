#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:l40s:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH -J ms_plaid_l40s
#SBATCH -o /exp/rjha/pylate-pgc/logs/scale_val/msmarco_plaid_l40s.log
# PLAID (fast-plaid) MSMARCO baseline at nbits=2 on an L40S (48GB VRAM, sm_89 -> supported by
# the project's cu13 torch). Rerouted off the heavily-contended 80GB A100 queue (job 358435 left
# queued as a fallback). Index ~30GB fits in 48GB; low_memory stays OFF.
# --mem=128G (not 400G): L40S nodes have ~366GB total RAM, and build streams shards via mmap so
# host-RAM peak is just doc-list loading + overhead (tens of GB). 128G is safe headroom and fits.
# If retrieve OOMs on the 48GB card, re-run retrieve-only with a smaller index.search_batch_size
# (the built index persists, no rebuild needed).
set -euo pipefail
cd /exp/rjha/pylate-pgc
uv run --no-sync python scripts/benchmark_indexes.py \
  model=lateon_regularized index=plaid index.nbits=2 \
  device=cuda \
  datasets='[beir/msmarco/dev]' stages='[build_index,retrieve]' \
  output.index_folder=indexes \
  output.results_file=results/scale_val/msmarco_plaid_nbits2.jsonl output.runs_dir=null
echo "=== msmarco PLAID nbits=2 (L40S) build+retrieve done $(date) ==="
