#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:a100:1
#SBATCH --constraint=80gb
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=400G
#SBATCH -J ms_plaid
#SBATCH -o /exp/rjha/pylate-pgc/logs/scale_val/msmarco_plaid.log
# PLAID (fast-plaid) MSMARCO baseline at nbits=2 (iso-bitrate with tachiom M=32, 32 B/tok),
# on an 80GB A100 (sm_80 -> works with the project's cu13 torch; no V100/cu12 overlay needed).
# low_memory stays at its False default (NOT enabled) -> full index on the 80GB card (index
# ~25-30GB, fits). search_batch_size=128 comes from plaid.yaml (the 1<<18 default OOMs on
# MSMARCO's 6980 queries). Build streams shards (mmap + batched add, k-means on a ~520K-doc
# sample), so CPU RAM peak is tens of GB; 400G is ample headroom (no thrash).
#
# OOM-FIX (prior run 358435 died at update-batch 12/21, 2h15m in): fast-plaid's incremental
# update() does a transient `batch @ existing_centroids.T` matmul (~9.8GB for 4 shards). It
# OOM'd not on resident index (only ~10-30GB) but on allocator FRAGMENTATION — 17GB sat in
# reserved-but-unallocated pool, so a contiguous 9.8GB alloc failed with 7.9GB "free". Two fixes:
#   1. PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True -> reclaims the fragmented reserved pool.
#   2. index.add_batch_size=2 (was 4) -> halves the per-update transient matmul to ~5GB.
# A rerun must start clean: the PLAID index is built with override=False, so it would try to
# load/append to the partial 32GB dir -> corrupt. Wipe it first (this dir is a dead build artifact
# from the failed run, created by this same benchmark).
# If retrieve still OOMs on the GPU, re-run retrieve-only with a smaller index.search_batch_size
# (the built index persists to disk, so no rebuild needed).
set -euo pipefail
cd /exp/rjha/pylate-pgc
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
rm -rf indexes/bench_lightonai_LateOn-regularized_beir_msmarco_dev_plaid
uv run --no-sync python scripts/benchmark_indexes.py \
  model=lateon_regularized index=plaid index.nbits=2 index.add_batch_size=2 \
  device=cuda \
  datasets='[beir/msmarco/dev]' stages='[build_index,retrieve]' \
  output.index_folder=indexes \
  output.results_file=results/scale_val/msmarco_plaid_nbits2.jsonl output.runs_dir=null
echo "=== msmarco PLAID nbits=2 build+retrieve done $(date) ==="
