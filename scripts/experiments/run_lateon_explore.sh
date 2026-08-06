#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Drive the LateOn vs LateOn-regularized index-config comparison.
#
# One srunl40s job per (model, index config) -- NO hydra multirun, so a failure
# isolates to a single log. plaid runs first per model to populate the shared
# encode cache (embeddings_cache/<dataset>/<model>/...); tachiom reuses it.
#
# Idempotent: a (model, dataset, index, clustering) already present in results.jsonl
# is SKIPPED, so the driver can be killed and relaunched without redoing work.
#
# Resource note: tachiom's build (PGC/TAC k-means, PQ training, HNSW) is CPU-bound.
# srunl40s alone gets the cluster default (4 cores / 2G = CpusPerTres=gpu:4), which
# starves it. We request 32 cores / 64G explicitly.
#
# Usage:  run_lateon_explore.sh <dataset_id>          e.g. beir/scidocs
#         MODELS="lateon_regularized" run_lateon_explore.sh beir/scidocs   # subset
set -u

DATASET="${1:?usage: run_lateon_explore.sh <dataset_id>}"
DS_SLUG=$(echo "$DATASET" | tr '/' '_')
PROJ=/exp/rjha/pylate-pgc
LOGDIR="$PROJ/logs/lateon_explore"
mkdir -p "$LOGDIR"
MODELS="${MODELS:-lateon lateon_regularized}"

SRUN="srun -u -t 24:00:00 --gres=gpu:l40s:1 --cpus-per-task=64 --mem=256G -J latex_${DS_SLUG}"

hf_name() { case "$1" in
  lateon) echo "lightonai/LateOn";;
  lateon_regularized) echo "lightonai/LateOn-regularized";;
esac; }

# exit 0 if a matching row already exists in results.jsonl
already_done() { # model_cfg index_type clustering
  python3 - "$(hf_name "$1")" "$2" "$3" "$DATASET" <<'PY'
import json, sys
name, itype, clust, ds = sys.argv[1:5]
try:
    rows = [json.loads(l) for l in open("results.jsonl") if l.strip()]
except FileNotFoundError:
    sys.exit(1)
for r in rows:
    ic = r.get("index_config", {}) or {}
    c = ((ic.get("clustering") or {}) or {}).get("type") or ""
    if r.get("model") == name and r.get("dataset") == ds \
       and ic.get("type") == itype and c == clust:
        sys.exit(0)
sys.exit(1)
PY
}

run_job() { # model index_args tag itype clustering
  local model="$1" index_args="$2" tag="$3" itype="$4" clust="$5"
  local log="$LOGDIR/${model}_${DS_SLUG}_${tag}.log"
  if already_done "$model" "$itype" "$clust"; then
    echo "=== $(date '+%F %T') SKIP   $model / $tag (already in results.jsonl)"
    return 0
  fi
  echo ">>> $(date '+%F %T') START  $model / $tag / $DATASET  -> $log"
  $SRUN bash -c "cd $PROJ && source $PYLATE_VENV_PATH/bin/activate && python scripts/benchmark_indexes.py model=$model $index_args datasets=\"[$DATASET]\"" \
    >"$log" 2>&1
  echo "<<< $(date '+%F %T') END    $model / $tag  exit=$?"
}

for model in $MODELS; do
  run_job "$model" "index=plaid" "plaid" "plaid" ""
  slug=$( [ "$model" = "lateon" ] && echo "lightonai_LateOn" || echo "lightonai_LateOn-regularized" )
  docs_dir="$PROJ/embeddings_cache/${DS_SLUG}/${slug}/nopool_fp16/docs"
  if [ ! -d "$docs_dir" ]; then
    echo "!!! encode cache missing for $model ($docs_dir) -- skipping tachiom for this model"
    continue
  fi
  run_job "$model" "index=tachiom index/clustering=tac" "tachiom_tac" "tachiom" "tac"
  run_job "$model" "index=tachiom index/clustering=pgc" "tachiom_pgc" "tachiom" "pgc"
done

echo "=== $(date '+%F %T') ALL DONE for $DATASET ==="
