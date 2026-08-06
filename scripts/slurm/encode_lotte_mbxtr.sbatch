#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J lotte_mbxtr_enc
#SBATCH --array=0-7
#SBATCH -o /exp/rjha/pylate-pgc/logs/mbxtr/encode_%a.log
# Step 1/4. Data-parallel LoTTE encode with robro612/ModernBERT-XTR across 8 V100s.
# Each array task i encodes the disjoint shard stride i::8 and writes its own
# .npy/.doclens/.token_ids straight to the shared cache dir (disk = handoff, no IPC).
# Cache lands at embeddings_cache/lotte_pooled_dev_search/robro612_ModernBERT-XTR/nopool_fp16/.
#
# V100 (sm_70) can't run the project's torch 2.11+cu130 (CUDA 13 dropped Volta), so we
# overlay torch 2.6.0 transiently via `uv run --with` — the PyPI 2.6.0 wheel is a cu124
# build that still ships sm_70 (verified on the MSMARCO V100 encode). The rest of the
# env (transformers/sentence-transformers/pylate) is .venv-cu130, unchanged.
# batch_size=2000 fits a 32GB V100 for fp16 docs capped at doc_length=300.
set -u
cd /exp/rjha/pylate-pgc
N=8
# task 0 also encodes the (tiny) query set; the rest do docs only.
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then STAGES='[encode_docs,encode_queries]'; else STAGES='[encode_docs]'; fi
uv run --no-sync --with 'torch==2.6.0' python scripts/benchmark_indexes.py \
  model=modernbert_xtr index=tachiom \
  datasets='[lotte/pooled/dev/search]' stages="$STAGES" \
  encode.shards="${SLURM_ARRAY_TASK_ID}::${N}" \
  encode.batch_size=2000 \
  output.runs_dir=null
