#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# RUN 2b — TAC clustering ONCE, for the PGC-vs-TAC comparison at LoTTE scale.
# Runs in PARALLEL with the PGC clustering (separate cpu node) so it never blocks
# the PGC critical path. Quality is expected to be a wash vs PGC (study finding);
# the real deliverable is TAC's BUILD COST at K=2,792,304. NOTE: TAC crawls at high
# centroid budgets (~5h @ 1.46M on trec-covid) — this may be the long pole / may
# degrade; abort if it's not converging. n_iter=10 matches tac.yaml (normal TAC).
#
# REUSE-SAFE: writes only to clusterings/lotte_tac/; touches no embeddings/indices.
set -u
PROJ=/exp/rjha/pylate-pgc
SHARDS="$PROJ/embeddings_cache/lotte_pooled_dev_search/lightonai_LateOn-regularized/nopool_fp16/docs"
OUT="$PROJ/clusterings/lotte_tac"
mkdir -p "$PROJ/logs/scale_val"
LOG="$PROJ/logs/scale_val/lotte_cluster_tac.log"

# Run on a 500G cpu-partition node (EPYC 7713, Zen 3). The earlier 64-thread/180G run
# crawled for 16h: peak RSS (~178G) sat right at the --mem ceiling on a 185G node, so it
# was almost certainly thrashing on the large transient per-head-token assignment buffers.
# Fix: fewer threads (32 = fewer concurrent head-token buffers) + big RAM (400G on a 500G
# node) so it can't swap. Pinned to rack8n10 (idle, ~500G free) for a clean early read.
# verbose=True (cluster_only.py) now streams live k-means progress, so we can watch it tick.
srun -u -p cpu -w rack8n10 -t 24:00:00 --cpus-per-task=32 --mem=400G -J lt_cluster_tac \
  bash -c "cd $PROJ && uv run --no-sync python scripts/cluster_only.py \
    --shard-dir '$SHARDS' --method tac --total-centroids 2792304 --n-iter 10 \
    --out-dir '$OUT'" \
  > "$LOG" 2>&1
echo "=== RUN2b TAC clustering done $(date) -> $OUT ==="
