#!/bin/bash
#SBATCH --job-name=exp_colbert
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --exclude=c007,h02
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/exp_colbert/exp_colbert_%j.out
#SBATCH --error=work_dirs/slurm/exp_colbert/exp_colbert_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

# Default values
# DATASETS="scifact arguana"
# DATASETS="../amazon_dataset/beir_format"
DATASETS="../amazon_dataset/beir_format_full"

# MODEL_NAME="lightonai/GTE-ModernColBERT-v1"
MODEL_NAME="jinaai/jina-colbert-v2"
# MODEL_NAME="antoinelouis/colbert-xm"

INDEX_TYPE="plaid"
SAVE_RUNFILES=""
SAVE_RETRIEVAL_RESULTS="--save-retrieval-results"
PARALLEL=false
KMEANS_GPU=""
DOC_LENGTH=""  # Empty means use model's max length
BATCH_SIZE=4

# Function to show help
show_help() {
    sed -n '2,25p' "$0" | sed 's/^# //' | sed 's/^#//'
    exit 0
}

# Parse command line arguments
while [[ $# -gt 0 ]]; do
    case $1 in
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
        --parallel)
            PARALLEL=true
            shift
            ;;
        --kmeans-gpu)
            KMEANS_GPU="--kmeans-gpu"
            shift
            ;;
        --doc-length)
            DOC_LENGTH="$2"
            shift 2
            ;;
        -b|--batch-size)
            BATCH_SIZE="$2"
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


RUN_SCRIPT="./scripts/run_compression_experiments.sh"

# Check if run script exists
if [[ ! -f "${RUN_SCRIPT}" ]]; then
    echo "Error: ${RUN_SCRIPT} not found"
    exit 1
fi

# Make sure run script is executable
chmod +x "${RUN_SCRIPT}"

# Build DOC_LENGTH_ARG
if [[ -n "${DOC_LENGTH}" ]]; then
    DOC_LENGTH_ARG="--doc-length ${DOC_LENGTH}"
else
    DOC_LENGTH_ARG=""
fi

# Print configuration
echo "================================================================================"
echo "RUNNING COMPRESSION EXPERIMENTS ON MULTIPLE DATASETS"
echo "================================================================================"
echo "Datasets:             ${DATASETS}"
echo "Model:                ${MODEL_NAME}"
echo "Index Type:           ${INDEX_TYPE}"
echo "Document Length:      ${DOC_LENGTH:-"(model max)"}"
echo "Batch Size:           ${BATCH_SIZE}"
echo "Save Runfiles:        $([ -n "${SAVE_RUNFILES}" ] && echo "Yes" || echo "No")"
echo "Save Retrieval:       $([ -n "${SAVE_RETRIEVAL_RESULTS}" ] && echo "Yes" || echo "No")"
echo "KMeans GPU:           $([ -n "${KMEANS_GPU}" ] && echo "Yes" || echo "No")"
echo "Parallel:             ${PARALLEL}"
echo "================================================================================"
echo ""

# Function to run experiment for a single dataset
run_dataset() {
    local dataset=$1
    echo ""
    echo "################################################################################"
    echo "# Starting experiment for dataset: ${dataset}"
    echo "################################################################################"
    echo ""

    "${RUN_SCRIPT}" \
        --dataset "${dataset}" \
        --model "${MODEL_NAME}" \
        --index-type "${INDEX_TYPE}" \
        --batch-size "${BATCH_SIZE}" \
        ${SAVE_RUNFILES} \
        ${SAVE_RETRIEVAL_RESULTS} \
        ${KMEANS_GPU} \
        ${DOC_LENGTH_ARG}
    
    local exit_code=$?
    
    if [[ ${exit_code} -eq 0 ]]; then
        echo ""
        echo "✓ Successfully completed experiment for ${dataset}"
        echo ""
    else
        echo ""
        echo "✗ Failed experiment for ${dataset} (exit code: ${exit_code})"
        echo ""
        return ${exit_code}
    fi
}

# Run experiments
if [[ "${PARALLEL}" == true ]]; then
    echo "Running experiments in parallel..."
    echo ""
    
    # Run in parallel using background processes
    pids=()
    for dataset in ${DATASETS}; do
        run_dataset "${dataset}" &
        pids+=($!)
    done
    
    # Wait for all background processes
    failed=0
    for pid in "${pids[@]}"; do
        if ! wait "${pid}"; then
            failed=$((failed + 1))
        fi
    done
    
    if [[ ${failed} -gt 0 ]]; then
        echo "⚠ ${failed} experiment(s) failed"
        exit 1
    fi
else
    echo "Running experiments sequentially..."
    echo ""
    
    # Run sequentially
    failed_datasets=()
    for dataset in ${DATASETS}; do
        if ! run_dataset "${dataset}"; then
            failed_datasets+=("${dataset}")
        fi
    done
    
    if [[ ${#failed_datasets[@]} -gt 0 ]]; then
        echo ""
        echo "================================================================================"
        echo "⚠ SOME EXPERIMENTS FAILED"
        echo "================================================================================"
        echo "Failed datasets: ${failed_datasets[*]}"
        exit 1
    fi
fi

echo ""
echo "================================================================================"
echo "✓ ALL EXPERIMENTS COMPLETED SUCCESSFULLY"
echo "================================================================================"
echo ""

