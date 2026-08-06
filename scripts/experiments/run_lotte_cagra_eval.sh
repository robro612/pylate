#!/usr/bin/env bash

# Selects the environment (PYLATE_VENV, default .venv-cu130) and exports
# UV_PROJECT_ENVIRONMENT for the uv invocations below. See docs/environments.md.
source "$(dirname "${BASH_SOURCE[0]}")/../lib/env.sh"
# LoTTE head-to-head: GPU-CAGRA clustering vs the best CPU-PGC clustering
# (lotte_pgc_m4t05), through the IDENTICAL tachiom pipeline — same build params
# (M=32, hnsw_m=32, alpha=0.45 defaults) and the same kc×kd retrieval grid. The ONLY
# variable is the clustering source, so any quality delta is attributable to CAGRA-guided
# vs PGC partitioning. Builds the CAGRA index once; the PGC index is already built and reused.
#
# Submitted with --dependency=afterok on the clustering job (cl_lotte_cagra). cpu partition.
set -u
PROJ=/exp/rjha/pylate-pgc; cd "$PROJ"
MODEL=lateon_regularized
DS='lotte/pooled/dev/search'
M=32
RES=results/gpucagra; mkdir -p "$RES"
CAGRA=clusterings/lotte_cagra
PGC=clusterings/lotte_pgc_m4t05
GRID=("40 10000" "40 20000" "80 10000" "80 20000")
COMMON="model=$MODEL index=tachiom index.pq_subspaces=$M datasets=[$DS] \
        output.index_folder=indexes output.runs_dir=null"
UVG="uv run --no-sync python"

echo "=== build CAGRA index once ($(date)) ==="
$UVG scripts/benchmark_indexes.py $COMMON \
  index/clustering=gpu \
  index.clustering.centroids_path=$CAGRA/centroids.npy \
  index.clustering.assignments_path=$CAGRA/assignments.npy \
  index.k_centroids=80 index.k_docs_to_score=20000 stages="[build_index]"

for cell in "${GRID[@]}"; do
  set -- $cell; kc=$1; kd=$2
  echo "=== retrieve kc=$kc kd=$kd ($(date)) ==="
  # GPU-CAGRA (clustering=gpu, reuse just-built index)
  $UVG scripts/benchmark_indexes.py $COMMON \
    index/clustering=gpu \
    index.clustering.centroids_path=$CAGRA/centroids.npy \
    index.clustering.assignments_path=$CAGRA/assignments.npy \
    index.k_centroids=$kc index.k_docs_to_score=$kd stages="[retrieve]" \
    output.results_file=$RES/lotte_cagra_kc${kc}_kd${kd}.jsonl
  # CPU-PGC baseline (clustering=external, reuse existing lotte_pgc_m4t05 index)
  $UVG scripts/benchmark_indexes.py $COMMON \
    index/clustering=external \
    index.clustering.centroids_path=$PGC/centroids.npy \
    index.clustering.assignments_path=$PGC/assignments.npy \
    index.k_centroids=$kc index.k_docs_to_score=$kd stages="[retrieve]" \
    output.results_file=$RES/lotte_pgc_kc${kc}_kd${kd}.jsonl
done
echo "=== LoTTE CAGRA-vs-PGC eval done ($(date)) ==="
