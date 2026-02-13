#!/bin/bash

# These SBATCH lines are for running the whole loop as a single job (local mode)
# In --sbatch mode, we submit one Slurm job per dataset with its own SBATCH args.
#SBATCH --job-name=compression-eval
#SBATCH --partition=h100,a100
#SBATCH --gpus=1
#SBATCH --mem=80G
#SBATCH --output=work_dirs/slurm/compression_eval_%j.out
#SBATCH --error=work_dirs/slurm/compression_eval_%j.err
#SBATCH --time=48:00:00

set -euo pipefail

# ----------------------------------------------------------------------------
# Args / Mode
# ----------------------------------------------------------------------------
SBATCH_MODE=0
DISPATCH_MODE=0
# Parse flags: --sbatch or --dispatch (mutually exclusive)
while [[ ${1:-} =~ ^-- ]]; do
  case "${1}" in
    --sbatch) SBATCH_MODE=1 ;;
    --dispatch) DISPATCH_MODE=1 ;;
    --) shift; break ;;
    *) echo "Unknown option: ${1}" >&2; exit 2 ;;
  esac
  shift || true
done
if [[ $SBATCH_MODE -eq 1 && $DISPATCH_MODE -eq 1 ]]; then
  echo "Error: --sbatch and --dispatch are mutually exclusive." >&2
  exit 2
fi

# Defaults for submitted jobs (only used in --sbatch mode)
SBATCH_PARTITION=${PARTITION:-h100,a100}
SBATCH_GPUS=${GPUS:-1}
SBATCH_MEM=${MEM:-80G}
SBATCH_TIME=${TIME_LIMIT:-72:00:00}

# Compute pylate dir for reliable working directory in jobs
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYLATE_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"

# Ensure slurm log dir exists (relative to submit cwd)
mkdir -p work_dirs/slurm

# Force ir_datasets to use a cache under ./pylate by default
# You can override by exporting IR_DATASETS_HOME before invoking this script.
IR_DATASETS_HOME_DEFAULT="${PYLATE_DIR}/work_dirs/ir_datasets_cache"
export IR_DATASETS_HOME="${IR_DATASETS_HOME:-$IR_DATASETS_HOME_DEFAULT}"
mkdir -p "$IR_DATASETS_HOME"
echo "IR_DATASETS_HOME: $IR_DATASETS_HOME"

# Note: We keep current working directory. Jobs submitted with --sbatch will use
# --chdir to run inside ${PYLATE_DIR} so relative paths resolve.

# ============================================================================
# Configuration
# ============================================================================

MODEL_NAME="${MODEL_NAME:-lightonai/GTE-ModernColBERT-v1}"

# All BEIR datasets (ir_datasets format)
BEIR_DATASETS=(
    # "beir/nfcorpus/test"
    # "beir/fiqa/test"
    # "beir/trec-covid"
    # "beir/scifact/test"
    "beir/scidocs/test"
    "beir/webis-touche2020/v2"
    "beir/quora/test"
    "beir/nq"
    "beir/hotpotqa/test"
    "beir/fever/test"
    "beir/climate-fever"
    "beir/dbpedia-entity/test"
    "beir/arguana"
    "beir/msmarco/dev"
)

# Override with specific datasets if provided
if [[ -n "${DATASETS:-}" ]]; then
    IFS=',' read -ra BEIR_DATASETS <<< "${DATASETS}"
fi

# ============================================================================
# Run compression evaluation on each dataset
# ============================================================================

echo "============================================================================"
if [[ $SBATCH_MODE -eq 1 ]]; then
  echo "Compression Evaluation (sbatch mode: one job per dataset)"
elif [[ $DISPATCH_MODE -eq 1 ]]; then
  echo "Compression Evaluation (dispatch mode: parallel on local GPUs)"
else
  echo "Compression Evaluation (local mode: sequential in one job)"
fi
echo "============================================================================"
echo "Model: ${MODEL_NAME}"
echo "Datasets: ${#BEIR_DATASETS[@]}"
echo "============================================================================"
echo ""

failed_datasets=()
submitted_jobs=()

# ----------------------------------------------------------------------------
# Dispatch mode: assign datasets to local GPUs by size (balanced groups)
# ----------------------------------------------------------------------------
if [[ $DISPATCH_MODE -eq 1 ]]; then
    # Resolve list of GPU device IDs
    IFS=',' read -ra GPU_IDS <<< "${DISPATCH_GPUS:-}"
    if [[ ${#GPU_IDS[@]} -eq 0 ]]; then
        # Detect available GPUs, take up to 4 by default
        mapfile -t DETECTED_GPUS < <(nvidia-smi --query-gpu=index --format=csv,noheader 2>/dev/null | head -n 4 || true)
        if [[ ${#DETECTED_GPUS[@]} -gt 0 ]]; then
            GPU_IDS=("${DETECTED_GPUS[@]}")
        else
            GPU_IDS=(0 1 2 3)
        fi
    fi

    NUM_GROUPS=${#GPU_IDS[@]}
    echo "Using GPUs: ${GPU_IDS[*]} (groups: ${NUM_GROUPS})"

    # Helper: get dataset size (docs_count) via ir_datasets; fall back to 0 on error
    dataset_size() {
        local ds="$1"
        python - "$ds" <<'PY' 2>/dev/null || true
import sys
name = sys.argv[1]
try:
    import ir_datasets as irds
    d = irds.load(name)
    n = None
    if hasattr(d, 'docs_count'):
        try:
            n = d.docs_count()
        except Exception:
            n = None
    print(int(n) if n is not None else 0)
except Exception:
    print(0)
PY
    }

    # Build size list
    declare -a SIZE_LINES=()
    for ds in "${BEIR_DATASETS[@]}"; do
        sz=$(dataset_size "$ds")
        if ! [[ "$sz" =~ ^[0-9]+$ ]]; then sz=0; fi
        SIZE_LINES+=("${sz}	${ds}")
    done

    echo "Dataset sizes (docs_count):"
    printf '%s\n' "${SIZE_LINES[@]}" | sed 's/^/  - /'

    # Initialize groups
    declare -a GROUP_SIZES=()
    declare -a GROUP_LISTS=()
    for ((i=0; i<NUM_GROUPS; i++)); do
        GROUP_SIZES[$i]=0
        GROUP_LISTS[$i]=''
    done

    # Greedy balance: assign largest-first to group with smallest total
    while IFS=$'\t' read -r sz ds; do
        min_idx=0
        min_val=${GROUP_SIZES[0]}
        for ((i=1; i<NUM_GROUPS; i++)); do
            if (( ${GROUP_SIZES[$i]} < min_val )); then
                min_idx=$i
                min_val=${GROUP_SIZES[$i]}
            fi
        done
        GROUP_SIZES[$min_idx]=$(( ${GROUP_SIZES[$min_idx]} + sz ))
        if [[ -z "${GROUP_LISTS[$min_idx]}" ]]; then
            GROUP_LISTS[$min_idx]="$ds"
        else
            GROUP_LISTS[$min_idx]+=$'\n'"$ds"
        fi
    done < <(printf '%s\n' "${SIZE_LINES[@]}" | sort -nr -k1,1)

    echo "Group assignments:"
    for ((i=0; i<NUM_GROUPS; i++)); do
        echo "  GPU ${GPU_IDS[$i]} -> total_size=${GROUP_SIZES[$i]}"
        while IFS= read -r dsi; do
            [[ -z "$dsi" ]] && continue
            echo "    - $dsi"
        done <<< "${GROUP_LISTS[$i]}"
    done

    mkdir -p work_dirs/dispatch
    FAIL_LOG="work_dirs/dispatch/dispatch_failures.log"
    : > "$FAIL_LOG"
    # Per-dataset log directory (timestamped for this dispatch run)
    DISPATCH_RUN_ID=${DISPATCH_RUN_ID:-$(date +%Y%m%d_%H%M%S)}
    DISPATCH_LOG_DIR="work_dirs/dispatch/logs/${DISPATCH_RUN_ID}"
    mkdir -p "$DISPATCH_LOG_DIR"
    echo "Per-dataset logs dir: $DISPATCH_LOG_DIR"

    # Launch one worker per GPU
    pids=()
    for ((i=0; i<NUM_GROUPS; i++)); do
        (
          # In worker: do not exit on first error; track failures per-dataset
          set -uo pipefail
          gpu_id="${GPU_IDS[$i]}"
          echo "[Worker i=$i GPU=$gpu_id] Starting at $(date)"
          cd "$PYLATE_DIR"
          while IFS= read -r dsi; do
            [[ -z "$dsi" ]] && continue
            san_ds=$(echo "$dsi" | sed -e 's#[/ ]#_#g' -e 's#[^A-Za-z0-9_.-]#_#g')
            log_path="$DISPATCH_LOG_DIR/comp_eval_${san_ds}.log"
            echo "[GPU $gpu_id] Dataset: $dsi | log: $log_path"
            echo "==== START $(date '+%F %T') | GPU=$gpu_id | DATASET=$dsi ====" | tee -a "$log_path"
            # Run and tee stdout+stderr to the per-dataset log; capture python exit code via PIPESTATUS
            CUDA_VISIBLE_DEVICES="$gpu_id" python experiments/compression/compression_eval.py \
                model.name_or_path="${MODEL_NAME}" dataset.name="$dsi" \
                2>&1 | tee -a "$log_path"
            status=${PIPESTATUS[0]}
            if [[ $status -eq 0 ]]; then
                echo "==== END   $(date '+%F %T') | OK ====" | tee -a "$log_path"
                echo "[GPU $gpu_id] ✓ Completed: $dsi"
            else
                echo "==== END   $(date '+%F %T') | FAIL (code=$status) ====" | tee -a "$log_path"
                echo "[GPU $gpu_id] ✗ Failed: $dsi (code=$status)" | tee -a "$FAIL_LOG"
            fi
          done <<< "${GROUP_LISTS[$i]}"
          echo "[Worker i=$i GPU=$gpu_id] Done at $(date)"
        ) &
        pids+=("$!")
    done

    # Wait for all workers
    exit_code=0
    for pid in "${pids[@]}"; do
        if ! wait "$pid"; then
            exit_code=1
        fi
    done

    echo "All dispatch workers finished."
    if [[ -s "$FAIL_LOG" ]]; then
        echo "Some datasets failed in dispatch mode:"
        cat "$FAIL_LOG"
        exit 1
    fi

    # Skip the standard loop and go to summary
else

for dataset in "${BEIR_DATASETS[@]}"; do
    echo ""
    echo "============================================================================"
    echo "Dataset: ${dataset}"
    echo "============================================================================"

    if [[ $SBATCH_MODE -eq 1 ]]; then
        # Submit one Slurm job per dataset
        san_ds=$(echo "${dataset}" | sed -e 's#[/ ]#_#g' -e 's#[^A-Za-z0-9_.-]#_#g')
        job_name="comp-eval_${san_ds}"
        out_log="work_dirs/slurm/comp_eval_${san_ds}_%j.out"
        err_log="work_dirs/slurm/comp_eval_${san_ds}_%j.err"

        jid=$(sbatch \
          --parsable \
          --job-name="$job_name" \
          --partition="$SBATCH_PARTITION" \
          --gpus="$SBATCH_GPUS" \
          --mem="$SBATCH_MEM" \
          --time="$SBATCH_TIME" \
          --output="$out_log" \
          --error="$err_log" \
          --export=ALL,IR_DATASETS_HOME="$IR_DATASETS_HOME" \
          --chdir="$PYLATE_DIR" \
          --wrap "python experiments/compression/compression_eval.py model.name_or_path='${MODEL_NAME}' dataset.name='${dataset}'") || {
              echo "✗ Failed to submit: ${dataset}"; failed_datasets+=("${dataset}"); continue; }
        submitted_jobs+=("${dataset} -> ${jid}")
        echo "✓ Submitted: ${dataset} (job ${jid})"
    else
        if python experiments/compression/compression_eval.py \
            model.name_or_path="${MODEL_NAME}" \
            dataset.name="${dataset}"; then
            echo "✓ Completed: ${dataset}"
        else
            echo "✗ Failed: ${dataset}"
            failed_datasets+=("${dataset}")
        fi
    fi
done

fi

echo ""
echo "============================================================================"
echo "Summary"
echo "============================================================================"

if [[ $SBATCH_MODE -eq 1 ]]; then
    if [[ ${#submitted_jobs[@]} -gt 0 ]]; then
        echo "Submitted jobs (${#submitted_jobs[@]}):"
        for item in "${submitted_jobs[@]}"; do
            echo "  - ${item}"
        done
    else
        echo "No jobs submitted."
    fi
    if [[ ${#failed_datasets[@]} -gt 0 ]]; then
        echo "Failed submissions (${#failed_datasets[@]}):"
        for ds in "${failed_datasets[@]}"; do
            echo "  - ${ds}"
        done
        exit 1
    fi
else
    if [[ ${#failed_datasets[@]} -gt 0 ]]; then
        echo "Failed datasets (${#failed_datasets[@]}):"
        for ds in "${failed_datasets[@]}"; do
            echo "  - ${ds}"
        done
        exit 1
    else
        echo "All ${#BEIR_DATASETS[@]} datasets completed successfully!"
    fi
fi