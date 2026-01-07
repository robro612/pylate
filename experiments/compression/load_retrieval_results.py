#!/usr/bin/env python3
"""
Helper script to load and analyze saved retrieval results.

This script provides utilities to:
1. Load retrieval results from JSON files
2. Compare results across different compression methods
3. Analyze query-specific performance
4. Export results to different formats
"""

import json
import argparse
from pathlib import Path
from typing import Dict, List, Any
import pandas as pd


def load_retrieval_results(filepath: str) -> Dict[str, Any]:
    """
    Load retrieval results from a JSON file.
    
    Parameters
    ----------
    filepath : str
        Path to the retrieval results JSON file
        
    Returns
    -------
    dict
        Dictionary containing retrieval results with keys:
        - run_id: Unique run identifier
        - config_idx: Configuration index
        - config_name: Configuration name
        - dataset_name: Dataset name
        - model_name: Model name
        - num_queries: Number of queries
        - k: Number of retrieved documents per query
        - results: Dict mapping query_id -> list of retrieved documents
    """
    with open(filepath, 'r') as f:
        data = json.load(f)
    return data


def load_all_results_from_directory(directory: str) -> List[Dict[str, Any]]:
    """
    Load all retrieval results from a directory.
    
    Parameters
    ----------
    directory : str
        Path to directory containing retrieval result JSON files
        
    Returns
    -------
    list
        List of retrieval result dictionaries
    """
    directory_path = Path(directory)
    results = []
    
    for filepath in sorted(directory_path.glob("retrieval-*.json")):
        data = load_retrieval_results(filepath)
        results.append(data)
    
    return results


def compare_methods(results_list: List[Dict[str, Any]], query_id: str) -> pd.DataFrame:
    """
    Compare retrieval results for a specific query across different methods.
    
    Parameters
    ----------
    results_list : list
        List of retrieval result dictionaries
    query_id : str
        Query ID to compare
        
    Returns
    -------
    pd.DataFrame
        DataFrame with columns: config_name, doc_id, score, rank
    """
    comparison_data = []
    
    for result in results_list:
        config_name = result['config_name']
        if query_id in result['results']:
            query_results = result['results'][query_id]
            for rank, doc_match in enumerate(query_results, 1):
                comparison_data.append({
                    'config_name': config_name,
                    'doc_id': doc_match['id'],
                    'score': doc_match['score'],
                    'rank': rank
                })
    
    return pd.DataFrame(comparison_data)


def get_top_k_overlap(results_list: List[Dict[str, Any]], query_id: str, k: int = 10) -> pd.DataFrame:
    """
    Calculate top-k overlap between different methods for a specific query.
    
    Parameters
    ----------
    results_list : list
        List of retrieval result dictionaries
    query_id : str
        Query ID to analyze
    k : int
        Number of top documents to consider (default: 10)
        
    Returns
    -------
    pd.DataFrame
        Pairwise overlap matrix between methods
    """
    # Get top-k doc IDs for each method
    method_top_k = {}
    for result in results_list:
        config_name = result['config_name']
        if query_id in result['results']:
            query_results = result['results'][query_id]
            top_k_ids = {doc['id'] for doc in query_results[:k]}
            method_top_k[config_name] = top_k_ids
    
    # Calculate pairwise overlap
    methods = list(method_top_k.keys())
    overlap_matrix = []
    
    for method1 in methods:
        row = []
        for method2 in methods:
            overlap = len(method_top_k[method1] & method_top_k[method2])
            row.append(overlap)
        overlap_matrix.append(row)
    
    return pd.DataFrame(overlap_matrix, index=methods, columns=methods)


def export_to_csv(results: Dict[str, Any], output_path: str):
    """
    Export retrieval results to CSV format.
    
    Parameters
    ----------
    results : dict
        Retrieval results dictionary
    output_path : str
        Path to save CSV file
    """
    rows = []
    for query_id, query_results in results['results'].items():
        for rank, doc_match in enumerate(query_results, 1):
            rows.append({
                'query_id': query_id,
                'doc_id': doc_match['id'],
                'score': doc_match['score'],
                'rank': rank
            })
    
    df = pd.DataFrame(rows)
    df.to_csv(output_path, index=False)
    print(f"Exported to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Load and analyze retrieval results")
    parser.add_argument("--directory", type=str, required=True, help="Directory containing retrieval results")
    parser.add_argument("--query_id", type=str, help="Specific query ID to analyze")
    parser.add_argument("--export_csv", type=str, help="Export results to CSV file")
    parser.add_argument("--top_k", type=int, default=10, help="Top-k for overlap analysis")
    
    args = parser.parse_args()
    
    # Load all results
    print(f"Loading results from {args.directory}...")
    results_list = load_all_results_from_directory(args.directory)
    print(f"Loaded {len(results_list)} result files")
    
    # Print summary
    for result in results_list:
        print(f"\nConfig {result['config_idx']}: {result['config_name']}")
        print(f"  Dataset: {result['dataset_name']}")
        print(f"  Model: {result['model_name']}")
        print(f"  Queries: {result['num_queries']}")
        print(f"  Top-k: {result['k']}")
    
    # Query-specific analysis
    if args.query_id:
        print(f"\n{'='*80}")
        print(f"Analysis for query: {args.query_id}")
        print(f"{'='*80}")
        
        # Compare methods
        comparison_df = compare_methods(results_list, args.query_id)
        print("\nTop 10 results per method:")
        for config_name in comparison_df['config_name'].unique():
            method_df = comparison_df[comparison_df['config_name'] == config_name].head(10)
            print(f"\n{config_name}:")
            print(method_df.to_string(index=False))
        
        # Calculate overlap
        overlap_df = get_top_k_overlap(results_list, args.query_id, k=args.top_k)
        print(f"\nTop-{args.top_k} overlap matrix:")
        print(overlap_df)
    
    # Export to CSV if requested
    if args.export_csv and results_list:
        export_to_csv(results_list[0], args.export_csv)


if __name__ == "__main__":
    main()

