#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -t 16:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH -J lotte_mbxtr_sweep
#SBATCH -o /exp/rjha/pylate-pgc/logs/mbxtr/sweep.log
# Step 4/4. Finer-grained imputation x alpha sweep on the prebuilt index (retrieve-only,
# CPU). Tests the hypothesis: an XTR-TRAINED model (ModernBERT-XTR) should benefit from
# missing-score imputation where LateOn-regularized only broke even off the no-prune point.
#
# Two frontiers, each swept on its OWN pruning formula (the only iso-QPS-fair comparison):
#   BASE  impute=false + magnitude-band prune     (TACHIOM_GAP_RELATIVE=0)
#   TREAT impute=true  + gap-relative prune        (TACHIOM_GAP_RELATIVE=1, shift-invariant)
# Lower alpha => prune more => higher QPS. null = no prune (pure quality ceiling).
# [PRUNE_DEBUG] lines (TACHIOM_PRUNE_DEBUG=1) report prune fractions per run, in log order.
set -uo pipefail
cd /exp/rjha/pylate-pgc
export TACHIOM_PRUNE_DEBUG=1
CL=clusterings/lotte_mbxtr_cagra
RESULTS=results_mbxtr_impute_sweep.jsonl
BASE="model=modernbert_xtr index=tachiom index/clustering=external \
  index.clustering.centroids_path=$CL/centroids.npy \
  index.clustering.assignments_path=$CL/assignments.npy \
  index.pq_subspaces=32 output.index_folder=indexes output.runs_dir=null \
  output.results_file=$RESULTS datasets=[lotte/pooled/dev/search] \
  stages=[encode_queries,retrieve]"

run() {
  local tag="$1" gap="$2" imp="$3" al="$4"
  echo; echo "==============================================================="
  echo "[$(date)] RUN: ${tag}   (GAP=${gap} impute=${imp} alpha=${al})"
  echo "==============================================================="
  TACHIOM_GAP_RELATIVE="${gap}" uv run --no-sync python scripts/benchmark_indexes.py \
    $BASE index.impute_missing=${imp} index.alpha=${al}
}

# Baseline frontier: impute=false, magnitude-band prune.
for al in null 0.6 0.45 0.3 0.2 0.1 0.05; do
  run "BASE a=${al}" 0 false "${al}"
done
# Treatment frontier: impute=true, gap-relative prune (finer, spanning low-prune -> high-prune).
for al in null 2.0 1.5 1.2 1.0 0.85 0.6 0.45 0.3; do
  run "TREAT a=${al}" 1 true "${al}"
done

echo; echo "[$(date)] DONE -> ${RESULTS}"
echo "=== prune (log order) ==="; grep -E "RUN:|PRUNE_DEBUG" "${SLURM_JOB_NAME:+logs/mbxtr/sweep.log}" 2>/dev/null || true
