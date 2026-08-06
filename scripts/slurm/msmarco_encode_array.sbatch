#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J ms_encode
#SBATCH --array=0-7
#SBATCH -o /exp/rjha/pylate-pgc/logs/scale_val/msmarco_encode_%a.log
# Data-parallel MSMARCO encode across 8 V100s. Each array task encodes a disjoint
# shard stride (task i -> shards i::8) of the corpus and writes its own .npy/.doclens/
# .token_ids directly to the shared cache dir — disk is the handoff, no IPC of
# embeddings (we deliberately avoid pylate's encode_multi_process, which routes all
# embeddings back to the parent and never persists token_ids). The single-GPU encode
# path is sharded + resumable + saves token_ids, so this reuses the proven code.
#
# V100 (sm_70) can't run the project's torch 2.11+cu130 (CUDA 13 dropped Volta), so we
# overlay a CUDA-12 torch transiently via `uv run --with torch==2.6.0` (cu124 wheel,
# ships sm_70). Verified: matmul runs on a V100 under this overlay.
# batch_size=2000 fits a 32GB V100 for fp16 encoding of (short) MSMARCO passages.
set -u
cd /exp/rjha/pylate-pgc
N=8
# task 0 also encodes the (tiny) query set; the rest do docs only.
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then STAGES='[encode_docs,encode_queries]'; else STAGES='[encode_docs]'; fi
uv run --no-sync --with 'torch==2.6.0' python scripts/benchmark_indexes.py \
  model=lateon_regularized index=tachiom \
  datasets='[beir/msmarco/dev]' stages="$STAGES" \
  encode.shards="${SLURM_ARRAY_TASK_ID}::${N}" \
  encode.batch_size=2000 \
  output.runs_dir=null
