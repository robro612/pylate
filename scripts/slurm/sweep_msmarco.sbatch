#!/usr/bin/env bash
#SBATCH -p cpu
#SBATCH -t 12:00:00
#SBATCH --cpus-per-task=64
#SBATCH --mem=160G
#SBATCH -J ms_sweep
# Clean QPS + quality sweep over the search-time knobs (kc=nprobe, kd=rerank depth) for
# the MSMARCO M=32 index (method via $1). Fixed 64 threads, alpha=null / ef=auto, same grid
# as the lotte sweep so the two scales are directly comparable. Submit with
# --dependency=afterok:<build_jobid>. --mem=160G: the msmarco index (~30GB, 627M tokens)
# is ~2.3x lotte's; 160G covers index load + deep-rerank (kd=40000) working set.
set -u
PROJ=/exp/rjha/pylate-pgc; cd "$PROJ"
export RAYON_NUM_THREADS=64
METHOD="${1:?usage: sbatch sweep_msmarco.sbatch <tac|pgc>}"
CL="$PROJ/clusterings/msmarco_${METHOD}"
DATASET="beir/msmarco/dev"
echo "=== msmarco $METHOD sweep start $(date) on $(hostname); RAYON_NUM_THREADS=$RAYON_NUM_THREADS ==="
for kc in 20 40 80; do
  for kd in 5000 10000 20000 40000; do
    uv run --no-sync python scripts/benchmark_indexes.py \
      model=lateon_regularized index=tachiom index/clustering=external \
      index.clustering.centroids_path="$CL/centroids.npy" \
      index.clustering.assignments_path="$CL/assignments.npy" \
      index.pq_subspaces=32 index.k_centroids=$kc index.k_docs_to_score=$kd \
      index.alpha=null index.ef_search=null \
      stages='[retrieve]' datasets="[$DATASET]" \
      output.index_folder=indexes \
      output.results_file="results/scale_val/msmarco_qps_${METHOD}_kc${kc}_kd${kd}.jsonl" output.runs_dir=null
    echo "  [done] $METHOD kc=$kc kd=$kd"
  done
done
echo "=== msmarco $METHOD sweep done $(date) ==="
