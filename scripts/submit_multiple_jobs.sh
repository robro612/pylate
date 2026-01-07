#!/bin/bash
#
# Submit multiple compression experiments as SLURM jobs (one job per dataset)
#
# Usage:
#   ./pylate/scripts/submit_multiple_jobs.sh [OPTIONS]
#
# SLURM Options:
#   --job-name NAME           Job name prefix (default: compression-exp)
#                             Each job will be named: {prefix}-{dataset}
#   --partition PARTITION     Partition name (default: h100,a100)
#   --gpus NUM                Number of GPUs (default: 1)
#   --cpus NUM                Number of CPUs (default: 8)
#   --mem MEMORY              Memory limit (default: 80G)
#   --time TIME               Time limit (default: 24:00:00)
#   --exclude NODES           Nodes to exclude (optional, e.g., c007,h02)
#
# Experiment Options:
#   -d, --datasets DATASETS   Space-separated dataset names (required)
#                             Example: "nfcorpus scifact arguana"
#   -m, --model MODEL         Model name (default: lightonai/GTE-ModernColBERT-v1)
#   -i, --index-type TYPE     Index type: flat or plaid (default: plaid)
#   --save-runfiles           Save ranx runfiles
#   --save-retrieval-results  Save raw retrieval results
#   -h, --help                Show this help message
#
# Examples:
#   # Submit jobs for multiple datasets
#   ./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana"
#
#   # Submit jobs with retrieval results saved
#   ./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana" --save-retrieval-results
#
#   # Submit jobs with custom resources
#   ./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact" --gpus 2 --mem 128G --time "48:00:00"
#
#   # Submit jobs for all BEIR datasets
#   ./pylate/scripts/submit_multiple_jobs.sh -d "nfcorpus scifact arguana fiqa trec-covid" --save-retrieval-results
#

set -euo pipefail

# Default SLURM values
JOB_NAME_PREFIX="compression-exp"
PARTITION="h100,a100"
GPUS=1
CPUS=8
MEMORY="80G"
TIME_LIMIT="24:00:00"
EXCLUDE=""

# Default experiment values
DATASETS="nfcorpus scifact arguana"
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
INDEX_TYPE="plaid"
SAVE_RUNFILES=""
SAVE_RETRIEVAL_RESULTS=""

# Function to show help
show_help() {
    sed -n '2,36p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --job-name)
            JOB_NAME_PREFIX="$2"
            shift 2
            ;;
        --partition)
            PARTITION="$2"
            shift 2
            ;;
        --gpus)
            GPUS="$2"
            shift 2
            ;;
        --cpus)
            CPUS="$2"
            shift 2
            ;;
        --mem)
            MEMORY="$2"
            shift 2
            ;;
        --time)
            TIME_LIMIT="$2"
            shift 2
            ;;
        --exclude)
            EXCLUDE="$2"
            shift 2
            ;;
        -d|--datasets)
            DATASETS="$2"
            shift 2
            ;;
        -m|--model)
            MODEL_NAME="$2"
            shift 2
            ;;
        -i|--index-type)
            INDEX_TYPE="$2"
            shift 2
            ;;
        --save-runfiles)
            SAVE_RUNFILES="--save-runfiles"
            shift
            ;;
        --save-retrieval-results)
            SAVE_RETRIEVAL_RESULTS="--save-retrieval-results"
            shift
            ;;
        -h|--help)
            show_help
            ;;
        *)
            echo "Unknown option: $1"
            echo "Use -h or --help for usage information"
            exit 1
            ;;
    esac
done

# Check required arguments
if [[ -z "${DATASETS}" ]]; then
    echo "Error: Datasets are required (-d or --datasets)"
    echo "Example: -d \"nfcorpus scifact arguana\""
    echo "Use -h or --help for usage information"
    exit 1
fi

# Get script directory and project root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Create logs directory
mkdir -p "${PROJECT_ROOT}/pylate/logs"

# Check if submit_single_job.sh exists
SINGLE_JOB_SCRIPT="${SCRIPT_DIR}/submit_single_job.sh"
if [[ ! -f "${SINGLE_JOB_SCRIPT}" ]]; then
    echo "Error: ${SINGLE_JOB_SCRIPT} not found"
    exit 1
fi

# Make sure single job script is executable
chmod +x "${SINGLE_JOB_SCRIPT}"

# Print overall configuration
echo ""
echo "================================================================================"
echo "SUBMITTING MULTIPLE SLURM JOBS"
echo "================================================================================"
echo "Datasets:             ${DATASETS}"
echo "Number of jobs:       $(echo "${DATASETS}" | wc -w)"
echo "Job name prefix:      ${JOB_NAME_PREFIX}"
echo "Model:                ${MODEL_NAME}"
echo "Index Type:           ${INDEX_TYPE}"
echo "Partition:            ${PARTITION}"
echo "GPUs:                 ${GPUS}"
echo "CPUs:                 ${CPUS}"
echo "Memory:               ${MEMORY}"
echo "Time Limit:           ${TIME_LIMIT}"
echo "Exclude Nodes:        ${EXCLUDE:-"(none)"}"
echo "Save Runfiles:        $([ -n "${SAVE_RUNFILES}" ] && echo "Yes" || echo "No")"
echo "Save Retrieval:       $([ -n "${SAVE_RETRIEVAL_RESULTS}" ] && echo "Yes" || echo "No")"
echo "================================================================================"
echo ""

# Submit jobs for each dataset
submitted_jobs=()
failed_datasets=()

for dataset in ${DATASETS}; do
    echo "--------------------------------------------------------------------------------"
    echo "Submitting job for dataset: ${dataset}"
    echo "--------------------------------------------------------------------------------"

    # Build sbatch command
    SBATCH_CMD="sbatch"
    SBATCH_CMD="${SBATCH_CMD} --job-name=${JOB_NAME_PREFIX}-${dataset}"
    SBATCH_CMD="${SBATCH_CMD} --partition=${PARTITION}"
    SBATCH_CMD="${SBATCH_CMD} --gpus=${GPUS}"
    SBATCH_CMD="${SBATCH_CMD} --cpus-per-task=${CPUS}"
    SBATCH_CMD="${SBATCH_CMD} --mem=${MEMORY}"
    SBATCH_CMD="${SBATCH_CMD} --time=${TIME_LIMIT}"
    SBATCH_CMD="${SBATCH_CMD} --output=${PROJECT_ROOT}/pylate/logs/${JOB_NAME_PREFIX}-${dataset}_%j.out"
    SBATCH_CMD="${SBATCH_CMD} --error=${PROJECT_ROOT}/pylate/logs/${JOB_NAME_PREFIX}-${dataset}_%j.err"

    if [[ -n "${EXCLUDE}" ]]; then
        SBATCH_CMD="${SBATCH_CMD} --exclude=${EXCLUDE}"
    fi

    # Build export variables
    EXPORT_VARS="DATASET=${dataset},MODEL=${MODEL_NAME},INDEX_TYPE=${INDEX_TYPE}"

    if [[ -n "${SAVE_RUNFILES}" ]]; then
        EXPORT_VARS="${EXPORT_VARS},SAVE_RUNFILES=true"
    else
        EXPORT_VARS="${EXPORT_VARS},SAVE_RUNFILES=false"
    fi

    if [[ -n "${SAVE_RETRIEVAL_RESULTS}" ]]; then
        EXPORT_VARS="${EXPORT_VARS},SAVE_RETRIEVAL_RESULTS=true"
    else
        EXPORT_VARS="${EXPORT_VARS},SAVE_RETRIEVAL_RESULTS=false"
    fi

    SBATCH_CMD="${SBATCH_CMD} --export=${EXPORT_VARS}"
    SBATCH_CMD="${SBATCH_CMD} ${SINGLE_JOB_SCRIPT}"

    # Submit job
    if job_id=$(eval "${SBATCH_CMD}" 2>&1); then
        submitted_jobs+=("${job_id}")
        echo "✓ Successfully submitted job ${job_id} for ${dataset}"
    else
        echo "✗ Failed to submit job for ${dataset}"
        echo "Error: ${job_id}"
        failed_datasets+=("${dataset}")
    fi

    echo ""
done

# Print summary
echo ""
echo "================================================================================"
echo "SUBMISSION SUMMARY"
echo "================================================================================"
echo "Total datasets:       $(echo "${DATASETS}" | wc -w)"
echo "Jobs submitted:       ${#submitted_jobs[@]}"
echo "Failed submissions:   ${#failed_datasets[@]}"

if [[ ${#submitted_jobs[@]} -gt 0 ]]; then
    echo ""
    echo "Job IDs:              ${submitted_jobs[*]}"
fi

if [[ ${#failed_datasets[@]} -gt 0 ]]; then
    echo ""
    echo "Failed datasets:      ${failed_datasets[*]}"
fi

echo ""
echo "Monitor with:         squeue -u \$USER"
echo "Logs directory:       ${PROJECT_ROOT}/pylate/logs/"
echo "================================================================================"
echo ""

# Exit with error if any submissions failed
if [[ ${#failed_datasets[@]} -gt 0 ]]; then
    exit 1
fi

