#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=monitor/logs/monitor_%j.log
#SBATCH --error=monitor/logs/monitor_%j.err
#SBATCH --job-name=monitor_training

# Launch script for training monitor (CPU-only job)
# This monitor watches for new checkpoints and launches GPU evaluation jobs
# Usage: sbatch monitor/launch_monitor.sh monitor/configs/xtr_config.yaml [model_dir]
#        If model_dir is provided, it overrides the training_output_dir in the config

# Check if config file is provided
if [ -z "$1" ]; then
    echo "ERROR: Config file not provided"
    echo "Usage: sbatch $0 <config_file> [model_dir]"
    exit 1
fi

CONFIG_FILE="$1"
MODEL_DIR="$2"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting training monitor"
echo "Config file: $CONFIG_FILE"
if [ -n "$MODEL_DIR" ]; then
    echo "Model directory (override): $MODEL_DIR"
fi
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo ""

# Run the monitor script
if [ -n "$MODEL_DIR" ]; then
    python monitor/monitor.py --config "$CONFIG_FILE" --model-dir "$MODEL_DIR"
else
    python monitor/monitor.py --config "$CONFIG_FILE"
fi

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE

