#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -t 16:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=460G
#SBATCH -J ms_build
# Build a tachiom M=32 index from an MSMARCO clustering (method via $1: tac|pgc) over the
# encoded shards, external-clustering path — same recipe as the lotte builds. total_centroids
# is read from the clustering's centroids.npy (PGC may differ slightly from the 1% target).
# Index name auto-derives to ..._beir_msmarco_dev_tachiom_external_msmarco_<method>_m32.
# Submit with --dependency=afterok:<clustering_jobid>.
set -euo pipefail   # fail loudly so a failed build can't be masked as COMPLETED.
PROJ=/exp/rjha/pylate-pgc; cd "$PROJ"
METHOD="${1:?usage: sbatch build_msmarco_external.sbatch <tac|pgc>}"
CL="$PROJ/clusterings/msmarco_${METHOD}"
[ -f "$CL/centroids.npy" ] && [ -f "$CL/assignments.npy" ] || { echo "ERROR: clustering missing at $CL"; exit 1; }
TC=$(uv run --no-sync python -c "import numpy as np; print(np.load('$CL/centroids.npy', mmap_mode='r').shape[0])")
echo "=== build msmarco $METHOD index: centroids=$TC ==="
uv run --no-sync python scripts/benchmark_indexes.py \
  model=lateon_regularized index=tachiom index/clustering=external \
  index.clustering.centroids_path="$CL/centroids.npy" \
  index.clustering.assignments_path="$CL/assignments.npy" \
  index.pq_subspaces=32 index.total_centroids="$TC" \
  stages='[build_index]' datasets='[beir/msmarco/dev]' \
  output.index_folder=indexes \
  output.results_file="results/scale_val/msmarco_build_${METHOD}_m32.jsonl" output.runs_dir=null
echo "=== msmarco $METHOD build done $(date) ==="
