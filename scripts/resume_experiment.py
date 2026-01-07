#!/usr/bin/env python3
"""
Resume an incomplete compression experiment.

This script reads an existing results JSONL file, identifies which configurations
have already been completed, and calls compression_experiment.py with --skip to
resume from the last completed config.
"""

import argparse
import json
import subprocess
import sys
from pathlib import Path


def get_completed_count(results_jsonl_path: Path) -> int:
    """Get the number of configs that have been completed."""
    completed = 0
    with open(results_jsonl_path, 'r') as f:
        for line in f:
            data = json.loads(line.strip())
            if data.get('type') == 'result':
                completed += 1
    return completed


def get_metadata_from_results(results_jsonl_path: Path) -> dict:
    """Extract metadata from the first line of results file."""
    with open(results_jsonl_path, 'r') as f:
        first_line = f.readline().strip()
        metadata = json.loads(first_line)
        if metadata.get('type') != 'metadata':
            raise ValueError("First line of results file must be metadata")
        return metadata


def main():
    parser = argparse.ArgumentParser(description='Resume incomplete compression experiment')
    parser.add_argument(
        '--results_file',
        type=str,
        required=True,
        help='Path to the incomplete results JSONL file'
    )
    parser.add_argument(
        '--batch_size',
        type=int,
        default=1000,
        help='Batch size for encoding (default: 1000)'
    )
    parser.add_argument(
        '--save_retrieval_results',
        action='store_true',
        help='Save raw retrieval results for each configuration'
    )
    parser.add_argument(
        '--kmeans_gpu',
        action='store_true',
        help='Enable GPU for fastkmeans in spherical pooling'
    )

    args = parser.parse_args()

    results_path = Path(args.results_file).resolve()
    if not results_path.exists():
        print(f"Error: Results file not found: {results_path}")
        sys.exit(1)

    # Get metadata and completed count
    print("=" * 80)
    print("ANALYZING EXPERIMENT FOR RESUME")
    print("=" * 80)

    metadata = get_metadata_from_results(results_path)
    completed_count = get_completed_count(results_path)

    model_name = metadata['model_name']
    dataset_name = metadata['dataset_name']
    total_configs = metadata['num_configs']
    remaining = total_configs - completed_count

    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Total configs: {total_configs}")
    print(f"Completed: {completed_count}")
    print(f"Remaining: {remaining}")

    if remaining <= 0:
        print("\n✓ All configurations already completed!")
        return

    # Build command to call compression_experiment.py
    script_dir = Path(__file__).parent.parent
    experiment_script = script_dir / "experiments" / "compression" / "compression_experiment.py"

    cmd = [
        sys.executable, "-u", str(experiment_script),
        "--model_name", model_name,
        "--dataset_name", dataset_name,
        "--index_type", metadata['args']['index_type'],
        "--batch_size", str(args.batch_size),
        "--metrics", *metadata['args']['metrics'],
        "--skip", str(completed_count),
        "--append_to", str(results_path),  # Append to existing file
    ]

    if args.save_retrieval_results:
        cmd.append("--save_retrieval_results")

    if args.kmeans_gpu:
        cmd.append("--kmeans_gpu")

    if metadata['args'].get('configs_file'):
        cmd.extend(["--configs_file", metadata['args']['configs_file']])

    print("\n" + "=" * 80)
    print(f"RESUMING FROM CONFIG {completed_count}")
    print("=" * 80)
    print(f"\nCommand: {' '.join(cmd)}\n")

    # Run the experiment
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()

