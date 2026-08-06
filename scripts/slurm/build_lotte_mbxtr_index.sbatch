#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -t 16:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=320G
#SBATCH -J lotte_mbxtr_build
#SBATCH -o /exp/rjha/pylate-pgc/logs/mbxtr/build.log
# Step 3/4. Build the tachiom index ONCE over the new CAGRA clustering (PQ-train -> HNSW
# over centroids -> IVF). impute_missing/alpha are SEARCH-time params, so the sweep
# (step 4) reuses this single index for every config via load_existing_index.
# index/clustering=external + pq_subspaces=32 fixes the on-disk index identity:
#   bench_robro612_ModernBERT-XTR_lotte_pooled_dev_search_tachiom_external_lotte_mbxtr_cagra_m32
# The sweep MUST pass the identical centroids_path + M to load this index.
set -euo pipefail
cd /exp/rjha/pylate-pgc
CL=clusterings/lotte_mbxtr_cagra

K=$(uv run --no-sync python - "$CL" <<'PY'
import sys, numpy as np
print(int(np.load(sys.argv[1] + "/centroids.npy", mmap_mode="r").shape[0]))
PY
)
echo "[$(date)] building tachiom index, total_centroids=$K"

uv run --no-sync python scripts/benchmark_indexes.py \
  model=modernbert_xtr index=tachiom index/clustering=external \
  index.clustering.centroids_path="$CL/centroids.npy" \
  index.clustering.assignments_path="$CL/assignments.npy" \
  index.total_centroids="$K" index.pq_subspaces=32 \
  stages='[build_index]' datasets='[lotte/pooled/dev/search]' \
  output.index_folder=indexes output.runs_dir=null \
  output.results_file=results_mbxtr_build.jsonl
echo "[$(date)] DONE build"
