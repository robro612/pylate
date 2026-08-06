#!/usr/bin/env bash
#SBATCH -p gpu
#SBATCH --gres=gpu:v100:1
#SBATCH -t 6:00:00
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH -J lotte_mbxtr_cl
#SBATCH -o /exp/rjha/pylate-pgc/logs/mbxtr/cluster.log
# Step 2/4. CAGRA-guided Lloyd (GPU-PGC) over the ModernBERT-XTR LoTTE embeddings, on a
# V100 via the cu12 sidecar (.venv-cu126 — the main venv's cu13 torch can't run Volta).
# Streaming: train on a 70M-token GPU-resident subsample, then stream the final hard
# top-1 assignment over ALL tokens. Config mirrors the proven lotte_cagra build
# (K = 1% of tokens, train_sample=70M, iters=5, nn_descent g64/i128, chunk=2M).
set -euo pipefail
cd /exp/rjha/pylate-pgc
DOCS=embeddings_cache/lotte_pooled_dev_search/robro612_ModernBERT-XTR/nopool_fp16/docs
OUT=clusterings/lotte_mbxtr_cagra

# K = 1% of total doc tokens (the LateOn lotte_cagra convention: 279.2M tokens -> K=2792304).
K=$(.venv-cu126/bin/python - "$DOCS" <<'PY'
import glob, sys, numpy as np
d = sys.argv[1]
tot = sum(int(np.load(p).sum()) for p in sorted(glob.glob(d + "/doc_shard_*.doclens.npy")))
print(round(tot * 0.01))
PY
)
echo "[$(date)] total-token-derived K=$K  (shard_dir=$DOCS)"

.venv-cu126/bin/python scripts/gpu_cluster.py \
  shard_dir="$DOCS" out_dir="$OUT" \
  clustering.backend=cagra clustering.k="$K" \
  clustering.train_sample=70000000 clustering.iters=5 clustering.chunk=2000000
echo "[$(date)] DONE -> $OUT"
