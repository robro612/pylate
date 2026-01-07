#!/usr/bin/env python3
"""
Example script showing how to analyze saved retrieval results.

This demonstrates various analysis tasks you can perform with saved retrieval results.
"""

import sys
from pathlib import Path

# Add pylate to path if needed
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from load_retrieval_results import (
    load_retrieval_results,
    load_all_results_from_directory,
    compare_methods,
    get_top_k_overlap,
)


def example_1_load_and_inspect():
    """Example 1: Load a single result file and inspect it."""
    print("=" * 80)
    print("Example 1: Load and Inspect Results")
    print("=" * 80)
    
    # Replace with your actual file path
    filepath = "pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results/retrieval-20231215_143022.config-5.json"
    
    if not Path(filepath).exists():
        print(f"File not found: {filepath}")
        print("Please run an experiment with --save_retrieval_results first")
        return
    
    results = load_retrieval_results(filepath)
    
    print(f"\nConfiguration: {results['config_name']}")
    print(f"Dataset: {results['dataset_name']}")
    print(f"Model: {results['model_name']}")
    print(f"Number of queries: {results['num_queries']}")
    print(f"Top-k retrieved: {results['k']}")
    
    # Show results for first query
    first_query_id = list(results['results'].keys())[0]
    print(f"\nResults for query '{first_query_id}':")
    for rank, doc in enumerate(results['results'][first_query_id][:5], 1):
        print(f"  Rank {rank}: {doc['id']} (score: {doc['score']:.4f})")


def example_2_compare_baseline_vs_compressed():
    """Example 2: Compare baseline vs compressed retrieval."""
    print("\n" + "=" * 80)
    print("Example 2: Compare Baseline vs Compressed")
    print("=" * 80)
    
    # Replace with your actual directory
    directory = "pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results"
    
    if not Path(directory).exists():
        print(f"Directory not found: {directory}")
        return
    
    all_results = load_all_results_from_directory(directory)
    
    if len(all_results) < 2:
        print("Need at least 2 result files to compare")
        return
    
    baseline = all_results[0]  # Assuming config_idx=0 is baseline
    compressed = all_results[5]  # Example: config_idx=5
    
    print(f"\nBaseline: {baseline['config_name']}")
    print(f"Compressed: {compressed['config_name']}")
    
    # Compare for a specific query
    query_id = list(baseline['results'].keys())[0]
    print(f"\nComparing query: {query_id}")
    
    baseline_top10 = {doc['id'] for doc in baseline['results'][query_id][:10]}
    compressed_top10 = {doc['id'] for doc in compressed['results'][query_id][:10]}
    
    overlap = len(baseline_top10 & compressed_top10)
    only_baseline = baseline_top10 - compressed_top10
    only_compressed = compressed_top10 - baseline_top10
    
    print(f"\nTop-10 overlap: {overlap}/10 documents")
    print(f"Only in baseline: {len(only_baseline)} documents")
    print(f"Only in compressed: {len(only_compressed)} documents")
    
    if only_baseline:
        print(f"\nDocuments lost from top-10: {list(only_baseline)[:3]}")
    if only_compressed:
        print(f"New documents in top-10: {list(only_compressed)[:3]}")


def example_3_find_problematic_queries():
    """Example 3: Find queries where compression significantly changes results."""
    print("\n" + "=" * 80)
    print("Example 3: Find Problematic Queries")
    print("=" * 80)
    
    directory = "pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results"
    
    if not Path(directory).exists():
        print(f"Directory not found: {directory}")
        return
    
    all_results = load_all_results_from_directory(directory)
    
    if len(all_results) < 2:
        print("Need at least 2 result files to compare")
        return
    
    baseline = all_results[0]
    compressed = all_results[5]
    
    print(f"\nComparing: {baseline['config_name']} vs {compressed['config_name']}")
    print("\nQueries with low top-10 overlap (< 50%):")
    
    problematic_queries = []
    
    for query_id in list(baseline['results'].keys())[:20]:  # Check first 20 queries
        baseline_top10 = {doc['id'] for doc in baseline['results'][query_id][:10]}
        compressed_top10 = {doc['id'] for doc in compressed['results'][query_id][:10]}
        overlap = len(baseline_top10 & compressed_top10)
        
        if overlap < 5:  # Less than 50% overlap
            problematic_queries.append((query_id, overlap))
    
    for query_id, overlap in problematic_queries[:5]:
        print(f"  {query_id}: {overlap}/10 overlap")
    
    print(f"\nTotal problematic queries: {len(problematic_queries)}")


def example_4_score_analysis():
    """Example 4: Analyze score distributions."""
    print("\n" + "=" * 80)
    print("Example 4: Score Distribution Analysis")
    print("=" * 80)
    
    directory = "pylate/results/compression_experiments/lightonai_GTE-ModernColBERT-v1/nfcorpus/retrieval_results"
    
    if not Path(directory).exists():
        print(f"Directory not found: {directory}")
        return
    
    all_results = load_all_results_from_directory(directory)
    
    for result in all_results[:3]:  # Analyze first 3 configs
        all_scores = []
        for query_results in result['results'].values():
            all_scores.extend([doc['score'] for doc in query_results])
        
        import numpy as np
        print(f"\n{result['config_name']}:")
        print(f"  Mean score: {np.mean(all_scores):.4f}")
        print(f"  Std score: {np.std(all_scores):.4f}")
        print(f"  Min score: {np.min(all_scores):.4f}")
        print(f"  Max score: {np.max(all_scores):.4f}")
        print(f"  Median score: {np.median(all_scores):.4f}")


def main():
    """Run all examples."""
    print("\n" + "=" * 80)
    print("RETRIEVAL RESULTS ANALYSIS EXAMPLES")
    print("=" * 80)
    print("\nNote: Update file paths in the examples to match your actual results")
    print()
    
    # Run examples
    example_1_load_and_inspect()
    example_2_compare_baseline_vs_compressed()
    example_3_find_problematic_queries()
    example_4_score_analysis()
    
    print("\n" + "=" * 80)
    print("Examples complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()

