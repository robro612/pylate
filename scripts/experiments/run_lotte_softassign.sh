#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Lotte (LateOn-regularized) — does the trec-covid soft-assign PGC win generalize
# to the true scale target? PLAID baseline + PGC hard-top-1 + PGC soft top-4 (tau=0.05).
# TAC skipped (focus: boosting PGC). Snappy assignment (efs50, mult40 — codified
# defaults; NO ef_search cranking). total_centroids=null (resolver; m1 & m4 share it).
#
# Cache: embeddings_cache/.../lightonai_LateOn-regularized is a SYMLINK to the
# lateon-fix-m lotte cache (asserted identical model) — docs+queries reused, no encode.
#
# L40S for the 375GB RAM (lotte build peaks ~150GB; cpu nodes' 185GB is risky).
set -u

PROJ=/exp/rjha/pylate-pgc
DATASET="lotte/pooled/dev/search"
MODEL=lateon_regularized
LOGDIR="$PROJ/logs/scale_val"; mkdir -p "$LOGDIR"
mkdir -p "$PROJ/results/scale_val"

# total_centroids = 1% of lotte tokens (2,792,304) — matches prior lateon-fix-m baseline.
CONFIGS=(
  "plaid|index=plaid"
  "pgc_m1|index=tachiom index/clustering=pgc index.total_centroids=2792304"
  "pgc_m4_t05|index=tachiom index/clustering=pgc index.total_centroids=2792304 index.clustering.assign_topm=4 index.clustering.assign_temp=0.05"
)

for entry in "${CONFIGS[@]}"; do
  tag="${entry%%|*}"; ov="${entry#*|}"
  log="$LOGDIR/lotte_${MODEL}_${tag}.log"
  # cpu partition: PGC/PLAID read cached embeddings (no encoding) -> no GPU needed.
  # 500GB nodes (rack7*) handle lotte's build peak; --mem=240G targets them.
  srun -u -p cpu -t 24:00:00 --cpus-per-task=64 --mem=240G -J "lt_${tag}" \
    bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py \
      model=$MODEL $ov datasets=\"[$DATASET]\" \
      output.index_folder=indexes/lotte_${tag} \
      output.results_file=results/scale_val/lotte_${MODEL}_${tag}.jsonl \
      output.runs_dir=null" \
    > "$log" 2>&1 &
  echo "launched $tag  -> $log"
  sleep 2
done
wait
echo "=== LOTTE SOFT-ASSIGN BATCH DONE $(date) ==="
