#!/usr/bin/env python3
"""Analyze grid search results from JSONL files."""

import json
import os
from pathlib import Path
import pandas as pd

def analyze_grid_results(results_dir="results/nfcorpus"):
    """Parse all ScaNN.jsonl results and create a summary."""
    
    jsonl_file = Path(results_dir) / "ScaNN.jsonl"
    
    if not jsonl_file.exists():
        print(f"No results found at {jsonl_file}")
        return
    
    results = []
    with open(jsonl_file, 'r') as f:
        for line in f:
            data = json.loads(line)
            config = data['index_config']['init_kwargs']
            
            result = {
                'num_neighbors': config['num_neighbors'],
                'k_token': data['k_token'],
                'num_leaves': config.get('num_leaves', 'N/A'),
                'num_leaves_to_search': config.get('num_leaves_to_search', 'N/A'),
                'index_time': data['index_time'],
                'retrieve_time': data['retrieve_time'],
            }
            
            # Add evaluation metrics
            for metric, value in data['evaluation_scores'].items():
                result[metric] = value
            
            results.append(result)
    
    # Create DataFrame
    df = pd.DataFrame(results)
    
    # Sort by ndcg@10 (or your preferred metric)
    df = df.sort_values('ndcg@10', ascending=False)
    
    print("\n" + "="*80)
    print("GRID SEARCH RESULTS SUMMARY")
    print("="*80)
    print(f"\nTotal configurations tested: {len(df)}")
    print(f"\nTop configurations by NDCG@10:")
    print("-"*80)
    
    display_cols = ['num_neighbors', 'k_token', 'ndcg@10', 'recall@100', 
                    'index_time', 'retrieve_time']
    print(df[display_cols].to_string(index=False))
    
    print("\n" + "="*80)
    print("BEST CONFIGURATION:")
    print("="*80)
    best = df.iloc[0]
    for col in df.columns:
        print(f"  {col}: {best[col]}")
    
    # Save full results to CSV
    output_file = Path(results_dir) / "grid_search_summary.csv"
    df.to_csv(output_file, index=False)
    print(f"\nFull results saved to: {output_file}")
    
    # Create pivot tables for easier analysis
    print("\n" + "="*80)
    print("NDCG@10 HEATMAP (num_neighbors vs k_token):")
    print("="*80)
    pivot_ndcg = df.pivot_table(
        values='ndcg@10', 
        index='num_neighbors', 
        columns='k_token',
        aggfunc='first'
    )
    print(pivot_ndcg.to_string())
    
    print("\n" + "="*80)
    print("RETRIEVAL TIME HEATMAP (num_neighbors vs k_token):")
    print("="*80)
    pivot_time = df.pivot_table(
        values='retrieve_time', 
        index='num_neighbors', 
        columns='k_token',
        aggfunc='first'
    )
    print(pivot_time.to_string())
    
    return df

if __name__ == "__main__":
    analyze_grid_results()