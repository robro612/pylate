#!/bin/bash
#SBATCH --job-name=compression-exp
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --time=24:00:00
#SBATCH --output=pylate/logs/compression-exp_%j.out
#SBATCH --error=pylate/logs/compression-exp_%j.err
#
# Submit a single compression experiment as SLURM job
#
# This script can be used in two ways:
#   1. Submit directly with sbatch: sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh
#   2. Call from submit_multiple_jobs.sh (recommended for multiple datasets)
#
# Environment Variables (set via --export):
#   DATASET                   Dataset name (required)
#   MODEL                     Model name (default: lightonai/GTE-ModernColBERT-v1)
#   INDEX_TYPE                Index type: flat or plaid (default: plaid)
#   SAVE_RUNFILES             Set to "true" to save ranx runfiles
#   SAVE_RETRIEVAL_RESULTS    Set to "true" to save raw retrieval results
#
# Examples:
#   # Submit directly with sbatch
#   sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh
#
#   # Submit with custom options
#   sbatch --export=DATASET=scifact,SAVE_RETRIEVAL_RESULTS=true \
#          --gpus=2 --time=48:00:00 --mem=128G \
#          pylate/scripts/submit_single_job.sh
#
#   # Use submit_multiple_jobs.sh for multiple datasets (recommended)
#   ./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana"
#

set -euo pipefail

# Default experiment values (can be overridden via environment variables)
DATASET="${DATASET:-}"
MODEL="${MODEL:-lightonai/GTE-ModernColBERT-v1}"
INDEX_TYPE="${INDEX_TYPE:-plaid}"
SAVE_RUNFILES="${SAVE_RUNFILES:-false}"
SAVE_RETRIEVAL_RESULTS="${SAVE_RETRIEVAL_RESULTS:-false}"

# Check required environment variables
if [[ -z "${DATASET}" ]]; then
    echo "Error: DATASET environment variable is required"
    echo "Usage: sbatch --export=DATASET=nfcorpus pylate/scripts/submit_single_job.sh"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Create logs directory
mkdir -p "${PROJECT_ROOT}/pylate/logs"

echo "================================================================================"
echo "SLURM JOB INFORMATION"
echo "================================================================================"
echo "Job ID:               ${SLURM_JOB_ID}"
echo "Job Name:             ${SLURM_JOB_NAME}"
echo "Node:                 ${SLURM_NODELIST}"
echo "Dataset:              ${DATASET}"
echo "Model:                ${MODEL}"
echo "Index Type:           ${INDEX_TYPE}"
echo "Save Runfiles:        ${SAVE_RUNFILES}"
echo "Save Retrieval:       ${SAVE_RETRIEVAL_RESULTS}"
echo "Start Time:           $(date)"
echo "================================================================================"
echo ""

cd "${PROJECT_ROOT}"

# Build command
CMD="${SCRIPT_DIR}/run_compression_experiments.sh"
CMD="${CMD} --dataset ${DATASET}"
CMD="${CMD} --model \"${MODEL}\""
CMD="${CMD} --index-type ${INDEX_TYPE}"

if [[ "${SAVE_RUNFILES}" == "true" ]]; then
    CMD="${CMD} --save-runfiles"
fi

if [[ "${SAVE_RETRIEVAL_RESULTS}" == "true" ]]; then
    CMD="${CMD} --save-retrieval-results"
fi

# Run experiment
eval "${CMD}"

echo ""
echo "================================================================================"
echo "SLURM JOB COMPLETED"
echo "================================================================================"
echo "Dataset:              ${DATASET}"
echo "End Time:             $(date)"
echo "================================================================================"

