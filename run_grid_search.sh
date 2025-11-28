#!/bin/bash
#SBATCH --job-name=scann_grid
#SBATCH --output=logs/scann_grid_%A_%a.out
#SBATCH --error=logs/scann_grid_%A_%a.err
#SBATCH --time=4:00:00
#SBATCH --cpus-per-task=1
#SBATCH --array=0-19

# Create logs directory if it doesn't exist
mkdir -p logs

# Define parameter grid
# num_neighbors: [10, 50, 100, 200]
# k_token: [1000, 4000, 10000, 20000, 40000]
# Total combinations: 4 * 5 = 20 (array indices 0-19)

NUM_NEIGHBORS_ARRAY=(10 50 100 200)
K_TOKEN_ARRAY=(1000 4000 10000 20000 40000)
NUM_LEAVES=2000
NUM_LEAVES_TO_SEARCH=100

# Calculate which configuration to use based on SLURM_ARRAY_TASK_ID
NUM_NEIGHBORS_OPTIONS=${#NUM_NEIGHBORS_ARRAY[@]}
K_TOKEN_OPTIONS=${#K_TOKEN_ARRAY[@]}

# Get indices for this array task
NUM_NEIGHBORS_IDX=$((SLURM_ARRAY_TASK_ID / K_TOKEN_OPTIONS))
K_TOKEN_IDX=$((SLURM_ARRAY_TASK_ID % K_TOKEN_OPTIONS))

# Get actual parameter values
NUM_NEIGHBORS=${NUM_NEIGHBORS_ARRAY[$NUM_NEIGHBORS_IDX]}
K_TOKEN=${K_TOKEN_ARRAY[$K_TOKEN_IDX]}

echo "=========================================="
echo "SLURM Array Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "Configuration:"
echo "  num_neighbors: ${NUM_NEIGHBORS}"
echo "  k_token: ${K_TOKEN}"
echo "=========================================="

# Load any necessary modules (adjust as needed for your cluster)
# module load python/3.9
# module load cuda/11.8

# Activate virtual environment (adjust path as needed)
# source /path/to/venv/bin/activate

# Run the test with this configuration
python test_index.py \
    --dataset_name nfcorpus \
    --cache_embeddings \
    --cache_dir test_embeddings \
    --num_neighbors ${NUM_NEIGHBORS} \
    --k_token ${K_TOKEN} \
    --num_leaves ${NUM_LEAVES} \
    --num_leaves_to_search ${NUM_LEAVES_TO_SEARCH} \
    --verbose

echo "=========================================="
echo "Job completed: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
echo "=========================================="

