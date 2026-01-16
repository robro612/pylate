#!/bin/bash
#
# Comprehensive script for running compression experiments
#
# Usage:
#   ./pylate/scripts/run_compression_experiments.sh [OPTIONS]
#
# Options:
#   -d, --dataset DATASET       Dataset name (default: nfcorpus)
#   -m, --model MODEL          Model name (default: lightonai/GTE-ModernColBERT-v1)
#   -i, --index-type TYPE      Index type: flat or plaid (default: plaid)
#   -o, --output-dir DIR       Output directory (default: auto-generated)
#   -b, --batch-size SIZE      Batch size for encoding (default: 1000)
#   --save-runfiles            Save ranx runfiles
#   --save-retrieval-results   Save raw retrieval results
#   --metrics METRICS          Space-separated metrics (default: map ndcg@10 ndcg@100 recall@10 recall@100 mrr@10 precision@10)
#   --configs-file FILE        Path to custom configs JSONL file
#   --kmeans-gpu               Enable GPU for fastkmeans in spherical pooling (experimental)
#   --multi-gpu                Enable multi-GPU encoding (uses all available GPUs)
#   --num-gpus N               Number of GPUs to use for multi-GPU encoding
#   -h, --help                 Show this help message
#
# Examples:
#   # Run on nfcorpus with default settings
#   ./pylate/scripts/run_compression_experiments.sh
#
#   # Run on scifact with retrieval results saved
#   ./pylate/scripts/run_compression_experiments.sh -d scifact --save-retrieval-results
#
#   # Run on arguana with custom output directory
#   ./pylate/scripts/run_compression_experiments.sh -d arguana -o results/my_experiment
#
#   # Run with custom metrics
#   ./pylate/scripts/run_compression_experiments.sh --metrics "map ndcg@10 precision@10"
#

set -euo pipefail

# Default values
DATASET_NAME="nfcorpus"
MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
INDEX_TYPE="plaid"
OUTPUT_DIR=""
BATCH_SIZE=256
SAVE_RUNFILES=""
SAVE_RETRIEVAL_RESULTS=""
METRICS="map ndcg@10 ndcg@100 recall@10 recall@100 mrr@10 precision@10"
CONFIGS_FILE=""
KMEANS_GPU=""
DOC_LENGTH=""  # Empty means use model's max length
MULTI_GPU=""
NUM_GPUS=""

# Function to show help
show_help() {
    sed -n '2,32p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        -d|--dataset)
            DATASET_NAME="$2"
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
        -o|--output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --save-runfiles)
            SAVE_RUNFILES="--save_runfiles"
            shift
            ;;
        --save-retrieval-results)
            SAVE_RETRIEVAL_RESULTS="--save_retrieval_results"
            shift
            ;;
        --metrics)
            METRICS="$2"
            shift 2
            ;;
        --configs-file)
            CONFIGS_FILE="$2"
            shift 2
            ;;
        --kmeans-gpu)
            KMEANS_GPU="--kmeans_gpu"
            shift
            ;;
        --doc-length)
            DOC_LENGTH="$2"
            shift 2
            ;;
        --multi-gpu)
            MULTI_GPU="--multi_gpu"
            shift
            ;;
        --num-gpus)
            NUM_GPUS="$2"
            shift 2
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

# Get script directory (pylate/scripts) and pylate root
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLATE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"

# Generate output directory if not specified
if [[ -z "${OUTPUT_DIR}" ]]; then
    MODEL_NAME_SANITIZED=$(echo "${MODEL_NAME}" | tr "/" "_")
    # Sanitize dataset name: use basename for paths, replace / with _
    DATASET_NAME_SANITIZED=$(basename "${DATASET_NAME}" | tr "/" "_")
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    OUTPUT_DIR="${PYLATE_ROOT}/results/compression_experiments_new2/${MODEL_NAME_SANITIZED}/${DATASET_NAME_SANITIZED}/experiment_${TIMESTAMP}"
fi

# Create output directory
mkdir -p "${OUTPUT_DIR}"

# Build command
COMMAND="python -u ${PYLATE_ROOT}/experiments/compression/compression_experiment.py"
COMMAND="${COMMAND} --dataset_name \"${DATASET_NAME}\""
COMMAND="${COMMAND} --model_name \"${MODEL_NAME}\""
COMMAND="${COMMAND} --index_type \"${INDEX_TYPE}\""
COMMAND="${COMMAND} --experiment_output_dir \"${OUTPUT_DIR}\""
COMMAND="${COMMAND} --batch_size ${BATCH_SIZE}"

if [[ -n "${SAVE_RUNFILES}" ]]; then
    COMMAND="${COMMAND} ${SAVE_RUNFILES}"
fi

if [[ -n "${SAVE_RETRIEVAL_RESULTS}" ]]; then
    COMMAND="${COMMAND} ${SAVE_RETRIEVAL_RESULTS}"
fi

if [[ -n "${METRICS}" ]]; then
    COMMAND="${COMMAND} --metrics ${METRICS}"
fi

if [[ -n "${CONFIGS_FILE}" ]]; then
    COMMAND="${COMMAND} --configs_file \"${CONFIGS_FILE}\""
fi

if [[ -n "${KMEANS_GPU}" ]]; then
    COMMAND="${COMMAND} ${KMEANS_GPU}"
fi

if [[ -n "${DOC_LENGTH}" ]]; then
    COMMAND="${COMMAND} --document_length ${DOC_LENGTH}"
fi

if [[ -n "${MULTI_GPU}" ]]; then
    COMMAND="${COMMAND} ${MULTI_GPU}"
fi

if [[ -n "${NUM_GPUS}" ]]; then
    COMMAND="${COMMAND} --num_gpus ${NUM_GPUS}"
fi

# Print configuration
echo "================================================================================"
echo "COMPRESSION EXPERIMENT CONFIGURATION"
echo "================================================================================"
echo "Dataset:              ${DATASET_NAME}"
echo "Model:                ${MODEL_NAME}"
echo "Index Type:           ${INDEX_TYPE}"
echo "Output Directory:     ${OUTPUT_DIR}"
echo "Batch Size:           ${BATCH_SIZE}"
echo "Document Length:      ${DOC_LENGTH:-"(model max)"}"
echo "Save Runfiles:        $([ -n "${SAVE_RUNFILES}" ] && echo "Yes" || echo "No")"
echo "Save Retrieval:       $([ -n "${SAVE_RETRIEVAL_RESULTS}" ] && echo "Yes" || echo "No")"
echo "KMeans GPU:           $([ -n "${KMEANS_GPU}" ] && echo "Yes" || echo "No")"
echo "Multi-GPU:            $([ -n "${MULTI_GPU}" ] && echo "Yes" || echo "No")"
echo "Num GPUs:             ${NUM_GPUS:-"(all available)"}"
echo "Metrics:              ${METRICS}"
echo "Configs File:         ${CONFIGS_FILE:-"(default)"}"
echo "================================================================================"
echo ""
echo "Command:"
echo "${COMMAND}"
echo ""
echo "================================================================================"
echo ""

# Run the experiment
eval "${COMMAND}"

echo ""
echo "================================================================================"
echo "EXPERIMENT COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo ""

