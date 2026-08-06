#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# Isolated A/B test of the XTR-style missing-score imputation (impute_missing).
#
# For each dataset (ascending size: trec-covid -> lotte -> msmarco) we run
# retrieval twice over the SAME index — once with impute_missing=false (the
# original 0-truncated coarse score) and once with impute_missing=true (each hit
# credited best_sim - m_i, rank-equivalent to XTR imputation). Everything else is
# held fixed, so any metric delta is attributable solely to this change.
#
# Retrieval reuses cached query embeddings + the prebuilt on-disk index, so no
# model load / GPU is needed. trec-covid has no prebuilt index, so we build it
# once (default in-process TAC clustering) before its two retrieval runs.
#
# Launch on a CPU-partition node with plenty of RAM (msmarco index ~31GB mmap):
#   srun --partition=cpu --cpus-per-task=32 --mem=480G --job-name=impute_xtr \
#        bash scripts/run_impute_xtr_test.sh 2>&1 | tee logs/impute_xtr.log
# then:  tail -f logs/impute_xtr.log

set -euo pipefail
cd /exp/rjha/pylate-pgc
mkdir -p logs

RESULTS="results_impute_xtr.jsonl"
BASE="model=lateon_regularized index=tachiom output.results_file=${RESULTS}"

echo "================================================================"
echo "[$(date)] Rebuilding tachiom Rust extension (release) into $PYLATE_VENV"
echo "================================================================"
# Rebuild ONLY the tachiom extension via maturin, bypassing uv's full dependency
# resolution (which fails on pylate[gpu] -> cuvs-cu13 under exclude-newer). maturin
# installs into the venv named by VIRTUAL_ENV; uses [tool.maturin] features=python.
( cd /exp/rjha/tachiom && \
  VIRTUAL_ENV=$PYLATE_VENV_PATH \
  $PYLATE_VENV_PATH/bin/maturin develop --release )

run() {
  local tag="$1"; shift
  echo
  echo "----------------------------------------------------------------"
  echo "[$(date)] RUN: ${tag}"
  echo "  overrides: $*"
  echo "----------------------------------------------------------------"
  uv run --no-sync python scripts/benchmark_indexes.py "$@"
}

# trec-covid skipped: no prebuilt index on disk.

# ---------------------------------------------------------------- lotte (PGC, prebuilt)
LOTTE="${BASE} datasets=[lotte/pooled/dev/search] index/clustering=external"
LOTTE="${LOTTE} index.clustering.centroids_path=clusterings/lotte_pgc_m4t05/centroids.npy"
LOTTE="${LOTTE} index.clustering.assignments_path=clusterings/lotte_pgc_m4t05/assignments.npy"
run "lotte impute=false" $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "lotte impute=true"  $LOTTE 'stages=[encode_queries,retrieve]' index.impute_missing=true

# ---------------------------------------------------------------- msmarco (PGC, prebuilt)
MSM="${BASE} datasets=[beir/msmarco/dev] index/clustering=external"
MSM="${MSM} index.clustering.centroids_path=clusterings/msmarco_pgc/centroids.npy"
MSM="${MSM} index.clustering.assignments_path=clusterings/msmarco_pgc/assignments.npy"
run "msmarco impute=false" $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=false
run "msmarco impute=true"  $MSM 'stages=[encode_queries,retrieve]' index.impute_missing=true

echo
echo "================================================================"
echo "[$(date)] DONE. Results appended to ${RESULTS}"
echo "Compare rows by index_config.impute_missing within each dataset:"
echo "  uv run --no-sync python -c \"import json;[print(r['dataset'], r['index_config']['impute_missing'], 'ndcg@10=%.4f'%r.get('ndcg@10',0), 'recall@100=%.4f'%r.get('recall@100',0), 'success@5=%.4f'%r.get('hit_rate@5',0)) for r in map(json.loads, open('${RESULTS}'))]\""
echo "================================================================"
