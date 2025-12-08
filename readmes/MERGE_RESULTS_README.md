# Merge Results Script

## Overview

The `merge_results.py` script merges multiple JSONL results files from compression experiments into a single consolidated file. It carefully handles metadata to ensure consistency and tracks the source of each result.

## Features

- **Metadata Validation**: Checks that all input files have compatible metadata (same model, dataset, metrics)
- **Source Tracking**: Each merged result includes information about its source file and original run ID
- **Flexible Merging**: Can merge results from different runs while preserving all information
- **Config Renumbering**: Optional sequential renumbering of config indices across merged results
- **Timing Aggregation**: Combines timing information from all source experiments

## Usage

### Basic Usage

```bash
python merge_results.py <output_file> <input_file1> <input_file2> [...]
```

### Examples

#### Merge multiple result files
```bash
python merge_results.py merged_results.jsonl \
    results/exp1/results_20251123_120000.jsonl \
    results/exp2/results_20251123_130000.jsonl \
    results/exp3/results_20251123_140000.jsonl
```

#### Merge with config renumbering
```bash
python merge_results.py --renumber merged.jsonl results/*/results_*.jsonl
```

#### Merge from a specific directory pattern
```bash
python merge_results.py merged.jsonl results/compression_experiments/*/nfcorpus/*/results_*.jsonl
```

#### Strict mode (fail on incompatibilities)
```bash
python merge_results.py --strict merged.jsonl results1.jsonl results2.jsonl
```

## Command-Line Options

- `output_file`: Path to the output merged JSONL file
- `input_files`: One or more input JSONL files to merge
- `--renumber`: Renumber config_idx sequentially across all merged results
- `--strict`: Fail on metadata incompatibilities (default: warn only)

## Output Format

The output JSONL file has the same structure as input files:

### Line 1: Merged Metadata
```json
{
  "type": "metadata",
  "run_id": "merged_20251123_223936",
  "timestamp": "2025-11-23T22:39:36.513170",
  "model_name": "lightonai/GTE-ModernColBERT-v1",
  "dataset_name": "nfcorpus",
  "merged_from": ["results1.jsonl", "results2.jsonl"],
  "num_source_files": 2,
  "source_run_ids": ["20251123_120000", "20251123_130000"],
  "timing": {
    "total_encoding_time": 37.18,
    "total_query_encoding_time": 0.33,
    "total_time": 1556.99
  },
  "configs": [...],
  "num_configs": 66
}
```

### Lines 2+: Result Entries
```json
{
  "type": "result",
  "run_id": "20251123_120000",
  "config_idx": 0,
  "config_name": "Baseline",
  "config": {...},
  "token_count": 862599,
  "avg_tokens_per_doc": 237.43,
  "compression_time": 0.0,
  "metrics": {...},
  "source_file": "results1.jsonl",
  "original_run_id": "20251123_120000"
}
```

## Metadata Handling

The script performs the following metadata operations:

1. **Validation**: Checks for consistency in:
   - Model names
   - Dataset names
   - Metrics used

2. **Merging**: Combines metadata by:
   - Using the first file's metadata as a base
   - Creating a new merged run_id
   - Tracking all source files and run IDs
   - Aggregating timing information
   - Combining all configs from all runs

3. **Warnings**: Issues warnings (or errors in strict mode) for:
   - Multiple different model names
   - Multiple different dataset names
   - Different metrics across experiments
   - Missing metadata entries

## Use Cases

### 1. Combining Partial Experiments
If you ran experiments in batches and want to combine them:
```bash
python merge_results.py combined.jsonl batch1.jsonl batch2.jsonl batch3.jsonl
```

### 2. Aggregating Results from Different Runs
Combine results from multiple experimental runs:
```bash
python merge_results.py all_results.jsonl results/*/results_*.jsonl
```

### 3. Creating a Master Results File
Merge all results from a specific model/dataset combination:
```bash
python merge_results.py master_nfcorpus.jsonl \
    results/compression_experiments/*/nfcorpus/*/results_*.jsonl
```

## Notes

- The script preserves all original information from each result entry
- Source tracking allows you to trace each result back to its original file
- Config renumbering is useful when you want sequential indices in the merged file
- Timing information is aggregated (summed) across all source files
- The merged file can be used with existing analysis scripts that read JSONL format

