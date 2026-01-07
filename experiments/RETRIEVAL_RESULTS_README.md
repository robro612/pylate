# Saving and Analyzing Retrieval Results

## Overview

The compression experiment script now supports saving **raw retrieval results** in addition to evaluation metrics. This allows you to:

1. **Analyze query-specific performance** - See exactly which documents were retrieved for each query
2. **Compare methods in detail** - Understand how different compression methods affect retrieval
3. **Debug and iterate** - Identify queries where compression hurts/helps performance
4. **Reuse results** - Load saved results for further analysis without re-running experiments

## What Gets Saved

### Runfiles (`--save_runfiles`)

**Format**: ranx Run objects (JSON)  
**Location**: `<experiment_dir>/runfiles/run-<run_id>.config-<config_idx>.json`  
**Contains**: Query-document relevance scores in ranx format (for evaluation)

**Use case**: Re-evaluate with different metrics without re-running retrieval

### Retrieval Results (`--save_retrieval_results`)

**Format**: Custom JSON format  
**Location**: `<experiment_dir>/retrieval_results/retrieval-<run_id>.config-<config_idx>.json`  
**Contains**: Complete retrieval results with metadata

**Structure**:
```json
{
  "run_id": "20231215_143022",
  "config_idx": 5,
  "config_name": "Attention score pooling keep_ratio=0.5",
  "dataset_name": "nfcorpus",
  "model_name": "lightonai/GTE-ModernColBERT-v1",
  "num_queries": 323,
  "k": 20,
  "results": {
    "query_id_1": [
      {"id": "doc_123", "score": 0.8542},
      {"id": "doc_456", "score": 0.7891},
      ...
    ],
    "query_id_2": [...],
    ...
  }
}
```

**Use case**: Detailed analysis of what documents were retrieved and their scores

## Usage

### Saving Results During Experiments

```bash
# Save only runfiles (for re-evaluation)
python pylate/experiments/compression/compression_experiment.py \
    --model_name "lightonai/GTE-ModernColBERT-v1" \
    --dataset_name "nfcorpus" \
    --save_runfiles

# Save only retrieval results (for detailed analysis)
python pylate/experiments/compression/compression_experiment.py \
    --model_name "lightonai/GTE-ModernColBERT-v1" \
    --dataset_name "nfcorpus" \
    --save_retrieval_results

# Save both
python pylate/experiments/compression/compression_experiment.py \
    --model_name "lightonai/GTE-ModernColBERT-v1" \
    --dataset_name "nfcorpus" \
    --save_runfiles \
    --save_retrieval_results
```

### Loading and Analyzing Results

Use the provided helper script:

```bash
# Load all results from a directory
python pylate/experiments/compression/load_retrieval_results.py \
    --directory pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results

# Analyze a specific query
python pylate/experiments/compression/load_retrieval_results.py \
    --directory pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results \
    --query_id "PLAIN-2891"

# Calculate top-10 overlap between methods
python pylate/experiments/compression/load_retrieval_results.py \
    --directory pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results \
    --query_id "PLAIN-2891" \
    --top_k 10

# Export to CSV
python pylate/experiments/compression/load_retrieval_results.py \
    --directory pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results \
    --export_csv results.csv
```

### Programmatic Access

```python
from pylate.experiments.compression.load_retrieval_results import (
    load_retrieval_results,
    load_all_results_from_directory,
    compare_methods,
    get_top_k_overlap,
)

# Load a single result file
results = load_retrieval_results("retrieval-20231215_143022.config-5.json")

# Access retrieval results for a specific query
query_id = "PLAIN-2891"
retrieved_docs = results['results'][query_id]

for rank, doc in enumerate(retrieved_docs, 1):
    print(f"Rank {rank}: {doc['id']} (score: {doc['score']:.4f})")

# Load all results from a directory
all_results = load_all_results_from_directory("path/to/retrieval_results/")

# Compare methods for a specific query
comparison_df = compare_methods(all_results, query_id="PLAIN-2891")
print(comparison_df)

# Calculate top-10 overlap
overlap_matrix = get_top_k_overlap(all_results, query_id="PLAIN-2891", k=10)
print(overlap_matrix)
```

## Example Analysis Workflows

### 1. Find Queries Where Compression Hurts Performance

```python
import json
from pathlib import Path

# Load baseline and compressed results
baseline = load_retrieval_results("retrieval-xxx.config-0.json")  # Baseline
compressed = load_retrieval_results("retrieval-xxx.config-5.json")  # Attention pooling

# Compare top-10 for each query
for query_id in baseline['results'].keys():
    baseline_top10 = {doc['id'] for doc in baseline['results'][query_id][:10]}
    compressed_top10 = {doc['id'] for doc in compressed['results'][query_id][:10]}
    
    overlap = len(baseline_top10 & compressed_top10)
    
    if overlap < 5:  # Less than 50% overlap
        print(f"Query {query_id}: only {overlap}/10 overlap")
```

### 2. Analyze Score Distributions

```python
import numpy as np
import matplotlib.pyplot as plt

results = load_retrieval_results("retrieval-xxx.config-5.json")

# Collect all scores
all_scores = []
for query_results in results['results'].values():
    all_scores.extend([doc['score'] for doc in query_results])

# Plot distribution
plt.hist(all_scores, bins=50)
plt.xlabel('Retrieval Score')
plt.ylabel('Frequency')
plt.title(f"Score Distribution: {results['config_name']}")
plt.show()
```

### 3. Compare Rank Changes

```python
# See how document ranks change between methods
baseline = load_retrieval_results("retrieval-xxx.config-0.json")
compressed = load_retrieval_results("retrieval-xxx.config-5.json")

query_id = "PLAIN-2891"

baseline_ranks = {doc['id']: rank for rank, doc in enumerate(baseline['results'][query_id], 1)}
compressed_ranks = {doc['id']: rank for rank, doc in enumerate(compressed['results'][query_id], 1)}

# Find documents with large rank changes
for doc_id in baseline_ranks:
    if doc_id in compressed_ranks:
        rank_change = compressed_ranks[doc_id] - baseline_ranks[doc_id]
        if abs(rank_change) > 5:
            print(f"Doc {doc_id}: rank {baseline_ranks[doc_id]} → {compressed_ranks[doc_id]} (Δ{rank_change:+d})")
```

## File Naming Convention

- **Runfiles**: `run-{run_id}.config-{config_idx}.json`
- **Retrieval Results**: `retrieval-{run_id}.config-{config_idx}.json`

Where:
- `run_id`: Timestamp-based unique identifier for the experiment run
- `config_idx`: Index of the compression configuration (0 = baseline)

## Storage Considerations

Retrieval results can be large:
- **Per query**: ~20 documents × (doc_id + score) ≈ 1-2 KB
- **Per config**: num_queries × 1-2 KB
- **Full experiment**: num_configs × num_queries × 1-2 KB

Example: 30 configs × 300 queries × 1.5 KB ≈ **13.5 MB** per experiment

💡 **Tip**: Only save retrieval results when you need detailed analysis. For most cases, the evaluation metrics in the main results file are sufficient.

