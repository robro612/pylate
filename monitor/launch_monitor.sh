#!/bin/bash
#SBATCH --time=48:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G
#SBATCH --output=monitor/logs/monitor_%j.log
#SBATCH --error=monitor/logs/monitor_%j.err
#SBATCH --job-name=monitor_training

# Launch script for training monitor (CPU-only job)
# This monitor watches for new checkpoints and launches GPU evaluation jobs
# Usage: sbatch monitor/launch_monitor.sh monitor/configs/xtr_config.yaml

# Check if config file is provided
if [ -z "$1" ]; then
    echo "ERROR: Config file not provided"
    echo "Usage: sbatch $0 <config_file>"
    exit 1
fi

CONFIG_FILE="$1"

if [ ! -f "$CONFIG_FILE" ]; then
    echo "ERROR: Config file not found: $CONFIG_FILE"
    exit 1
fi

echo "Starting training monitor"
echo "Config file: $CONFIG_FILE"
echo "Hostname: $(hostname)"
echo "Start time: $(date)"
echo "SLURM Job ID: $SLURM_JOB_ID"
echo ""

# Run the monitor script
python monitor/monitor.py --config "$CONFIG_FILE"

EXIT_CODE=$?

echo ""
echo "End time: $(date)"
echo "Exit code: $EXIT_CODE"

exit $EXIT_CODE

