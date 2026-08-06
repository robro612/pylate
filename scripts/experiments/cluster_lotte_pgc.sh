#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# RUN 2 — Cluster ONCE (the one expensive step; the long pole).
# PGC soft-assign (the confirmed scale winner: sample_multiplier=40, top-m=4, tau=0.05)
# over the cached lotte shards -> durable centroids.npy + assignments.npy that RUN 3
# builds many M-variant indexes from (index/clustering=external) without re-clustering.
# total_centroids = 1% of lotte tokens = 2,792,304 (matches prior baselines).
#
# REUSE-SAFE: writes only to clusterings/lotte_pgc_m4t05/; touches no embeddings/indices.
set -u
PROJ=/exp/rjha/pylate-pgc
SHARDS="$PROJ/embeddings_cache/lotte_pooled_dev_search/lightonai_LateOn-regularized/nopool_fp16/docs"
OUT="$PROJ/clusterings/lotte_pgc_m4t05"
mkdir -p "$PROJ/logs/scale_val"
LOG="$PROJ/logs/scale_val/lotte_cluster_pgc_m4t05.log"

srun -u -p cpu -t 16:00:00 --cpus-per-task=64 --mem=240G -J lt_cluster \
  bash -c "cd $PROJ && uv run --no-sync python scripts/cluster_only.py \
    --shard-dir '$SHARDS' --method pgc --total-centroids 2792304 --n-iter 5 \
    --pgc-sample-multiplier 40 --pgc-assign-topm 4 --pgc-assign-temp 0.05 \
    --out-dir '$OUT'" \
  > "$LOG" 2>&1
echo "=== RUN2 clustering done $(date) -> $OUT ==="
