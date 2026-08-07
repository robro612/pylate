#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH -J lotte_enc
#SBATCH --array=0-3
#SBATCH -o /exp/rjha/pylate-pgc/logs/lotte_pooled_encode_%a.log
#
# Data-parallel LoTTE-pooled encode across 4 V100s, encode only — no indexing.
#
# Four single-GPU tasks, not one job with four GPUs. PyLate's multi-GPU path
# (`encode_multi_process`, auto-enabled when torch.cuda.device_count() > 1)
# routes embeddings back to the parent and never persists token IDs —
# benchmark_indexes.py gates `save_token_ids` on `not use_multi_gpu` — and
# Tachiom needs them. Each task instead encodes a disjoint shard stride
# (task i -> shards i::4) through the proven single-GPU path and writes its own
# .npy/.doclens/.token_ids straight into the shared cache dir. Disk is the
# handoff; no IPC. Same approach as msmarco_encode_array.sbatch.
#
# .venv-cu126 because CUDA 13 dropped Volta (sm_70), so V100s cannot run the
# cu130 torch build. Non-profile env on purpose: encoding does not need stage
# timers, and the profiling build CUDA-syncs every stage.
#
# batch_size=2000 is the value already verified to fit a 32GB V100 for fp16
# encoding at this document length; the config default of 3000 is sized for
# L40S/A100.
set -u
cd /exp/rjha/pylate-pgc

N=4
export UV_PROJECT_ENVIRONMENT=.venv-cu126

# Task 0 also encodes the (small) query set; the rest do documents only, so the
# queries are not encoded four times into the same cache path.
if [ "${SLURM_ARRAY_TASK_ID}" -eq 0 ]; then
  STAGES='[encode_docs,encode_queries]'
else
  STAGES='[encode_docs]'
fi

echo "=== task ${SLURM_ARRAY_TASK_ID}/${N} on $(hostname) — shards ${SLURM_ARRAY_TASK_ID}::${N} ==="
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
date

uv run --no-sync python scripts/benchmark_indexes.py \
  --config-name profile_suite \
  index=tachiom \
  datasets='["lotte/pooled/dev/search"]' \
  stages="$STAGES" \
  encode.shards="${SLURM_ARRAY_TASK_ID}::${N}" \
  encode.batch_size=2000 \
  search.profile=false \
  output.results_file=results/lotte_pooled_encode.jsonl \
  output.runs_dir=null

echo "=== task ${SLURM_ARRAY_TASK_ID} exit=$? ==="
date
