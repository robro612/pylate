#!/usr/bin/env python3
"""
Minimal script to plot compression experiment results.
Plots avg tokens/doc vs metric, grouped by method.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path


def extract_method(config_str):
    """Extract method name from Config string."""
    # Handle Baseline case
    if config_str.startswith("Baseline"):
        return "Baseline"
    
    # Extract method name before parameters
    # Examples:
    # "Doc-wise IDF pruning k=2" -> "Doc-wise IDF pruning"
    # "Global IDF pruning k=5" -> "Global IDF pruning"
    # "Hierarchical Pooling f=2 protected tokens=1" -> "Hierarchical Pooling"
    
    # Remove parameter patterns (k=, f=, protected tokens=)
    cleaned = re.sub(r'\s+(k|f|protected tokens)=\d+', '', config_str)
    # Remove parenthetical info like "(no compression)"
    cleaned = re.sub(r'\s+\([^)]+\)', '', cleaned)
    
    return cleaned.strip()


def plot_results(tsv_path, dataset_name, metric, model='GTE-ModernColBERT-v1', output_dir='results/compression_experiments/plots'):
    """Plot avg tokens/doc vs metric, grouped by method."""
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t')
    
    # Extract method from Config column
    df['method'] = df['Config'].apply(extract_method)
    
    # Clean column names (remove spaces, handle special chars)
    df.columns = df.columns.str.strip()
    
    # Verify columns exist
    if 'Avg Tokens/Doc' not in df.columns:
        raise ValueError(f"Column 'Avg Tokens/Doc' not found. Available columns: {df.columns.tolist()}")
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {df.columns.tolist()}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate baseline from other methods
    baseline_data = df[df['method'] == 'Baseline']
    other_methods = df[df['method'] != 'Baseline']
    
    # Plot baseline as horizontal and vertical lines if it exists
    if not baseline_data.empty:
        baseline_metric_value = baseline_data[metric].iloc[0]
        baseline_tokens_value = baseline_data['Avg Tokens/Doc'].iloc[0]
        ax.axhline(y=baseline_metric_value, color='gray', linestyle='--', linewidth=2, label='Baseline')
        ax.axvline(x=baseline_tokens_value, color='gray', linestyle='--', linewidth=2)
    
    # Plot other methods as lines
    for method in other_methods['method'].unique():
        method_data = other_methods[other_methods['method'] == method].sort_values('Avg Tokens/Doc')
        ax.plot(
            method_data['Avg Tokens/Doc'],
            method_data[metric],
            marker='o',
            label=method,
            linewidth=2,
            markersize=6
        )
    
    ax.set_xlabel('Avg Tokens/Doc', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{dataset_name} - {metric} vs Avg Tokens/Doc', fontsize=14)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plot directory if it doesn't exist
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    output_path = plot_dir / f'{model}_{dataset_name}_{metric}_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also show plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot compression experiment results')
    parser.add_argument('--results_path', type=str, help='Path to results tsv file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--dataset_name', type=str, help='Dataset name')
    parser.add_argument('--metric', type=str, help='Metric to plot (e.g., map, ndcg@10)')
    parser.add_argument('--model', type=str, default='GTE-ModernColBERT-v1', help='Model name (default: GTE-ModernColBERT-v1)')
    
    args = parser.parse_args()
    
    plot_results(args.results_path, args.dataset_name, args.metric, args.model, args.output_dir)

