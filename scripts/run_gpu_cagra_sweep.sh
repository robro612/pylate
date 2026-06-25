#!/usr/bin/env bash
# trec-covid GPU-PGC (CAGRA-guided Lloyd) fidelity + speed sweep.
#
# Question: does CAGRA-guided GPU clustering preserve downstream tachiom retrieval
# quality vs CPU-PGC, and how much faster is the clustering step? trec-covid is the
# cheap proxy for the CAGRA parameter sweep; the winning settings carry to LoTTE.
#
# Design: every condition is clustered to the SAME K and ingested through the SAME
# precomputed-centroids path (clustering=gpu/external), so the downstream build (PQ ->
# HNSW -> IVF) and search params are byte-identical — the ONLY variable is the
# clustering method. Clustering jobs run on V100 via the cu12 sidecar (.venv-cu12; build it
# with scripts/setup_cu12_venv.sh) — V100 allocates faster than L40S. Build+eval run on the
# cpu partition with the main cu13 .venv.
#
# Robustness: sbatch (not srun) so jobs survive the launching session; eval jobs use
# --dependency=afterok on their clustering job.
set -u

PROJ=/exp/rjha/pylate-pgc
cd "$PROJ"
SHARDS=embeddings_cache/beir_trec-covid/lightonai_LateOn-regularized/nopool_fp16/docs
K=293130                       # = TAC/auto_build_params total_centroids for trec-covid
MODEL=lateon_regularized
DS='[beir/trec-covid]'
CL=clusterings
RES=results/gpucagra; mkdir -p "$RES" logs/gpucagra
UVG="uv run --no-sync python"  # main cu13 .venv (CPU build+eval); --no-sync keeps pip-installed cuvs
PY_CU12=".venv-cu12/bin/python" # V100 sidecar (Volta sm_70 + cuvs-cu12) for GPU clustering

# CAGRA grid (build_algo|graph_degree|intermediate|itopk). nn_descent baseline +
# graph-degree / itopk sweep + one ivf_pq point.
CAGRA_GRID=(
  "nd_g32_i64|nn_descent|32|128|64"
  "nd_g64_i32|nn_descent|64|128|32"
  "nd_g64_i64|nn_descent|64|128|64"
  "nd_g64_i128|nn_descent|64|128|128"
  "ivfpq_g64_i64|ivf_pq|64|128|64"
)

submit_cluster_gpu() {  # tag, "clustering.* overrides..."
  local tag=$1; shift
  sbatch --parsable -p gpu --gres=gpu:v100:1 -t 3:00:00 -J "cl_$tag" \
    -o "logs/gpucagra/cl_${tag}.log" \
    --wrap "$PY_CU12 scripts/gpu_cluster.py shard_dir=$SHARDS out_dir=$CL/tcv_$tag \
            clustering.k=$K clustering.iters=10 $*"
}

submit_eval() {  # tag, dep_jobid, clustering_type(gpu|external)
  local tag=$1 dep=$2 cltype=$3
  sbatch --parsable -p cpu --cpus-per-task=32 --mem=96G -t 4:00:00 -J "ev_$tag" \
    --dependency=afterok:"$dep" -o "logs/gpucagra/ev_${tag}.log" \
    --wrap "$UVG scripts/benchmark_indexes.py model=$MODEL index=tachiom \
            index/clustering=$cltype \
            index.clustering.centroids_path=$CL/tcv_$tag/centroids.npy \
            index.clustering.assignments_path=$CL/tcv_$tag/assignments.npy \
            datasets='$DS' output.index_folder=indexes/ev_$tag \
            output.results_file=$RES/$tag.jsonl output.runs_dir=null"
}

echo "=== submitting clustering + eval jobs ($(date)) ==="

# --- CPU-PGC baseline (clustering quality reference) ---
JID=$(sbatch --parsable -p cpu --cpus-per-task=32 --mem=96G -t 4:00:00 -J cl_pgc \
        -o logs/gpucagra/cl_pgc.log \
        --wrap "$UVG scripts/cluster_only.py --shard-dir $SHARDS --method pgc \
                --total-centroids $K --n-iter 5 --out-dir $CL/tcv_pgc")
echo "pgc      cluster=$JID  eval=$(submit_eval pgc "$JID" external)"

# --- GPU exact (brute) reference ---
JID=$(submit_cluster_gpu brute "clustering.backend=brute")
echo "brute    cluster=$JID  eval=$(submit_eval brute "$JID" gpu)"

# --- GPU CAGRA grid ---
for entry in "${CAGRA_GRID[@]}"; do
  IFS='|' read -r tag algo gd ig itopk <<<"$entry"
  ov="clustering.backend=cagra clustering.cagra.build_algo=$algo \
      clustering.cagra.graph_degree=$gd clustering.cagra.intermediate_graph_degree=$ig \
      clustering.cagra.itopk_size=$itopk"
  JID=$(submit_cluster_gpu "$tag" "$ov")
  echo "$tag  cluster=$JID  eval=$(submit_eval "$tag" "$JID" gpu)"
done

echo "=== all submitted. monitor: squeue -u $USER ; results in $RES/*.jsonl ==="
