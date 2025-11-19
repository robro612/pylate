#!/usr/bin/env python3
"""
Minimal script to plot compression experiment results.
Plots avg_tokens_per_document vs metric, grouped by method.
"""

import argparse
import pandas as pd
import matplotlib.pyplot as plt
import re
from pathlib import Path

def plot_results(tsv_path, dataset_name, metric, model='GTE-ModernColBERT-v1', output_dir='results/compression_experiments/plots'):
    """Plot avg_tokens_per_document vs metric, grouped by method."""
    # Read TSV file
    df = pd.read_csv(tsv_path, sep='\t', index_col=0)
    
    # Extract method from Config column
    def extract_method_class(name):
        if isinstance(name, str):
            name = name.lower()
            if name == "baseline":
                return "baseline"
            elif name.startswith("idf_pruning_mode-global"):
                return "idf-global"
            elif name.startswith("idf_pruning_mode-document"):
                return "idf-doc"
            elif name.startswith("pooling-hierarchical"):
                return "pooling-hierarchical"
            elif name.startswith("pooling-spherical"):
                return "pooling-spherical"
        return "other"
    df['method'] = df.index
    df['method_class'] = df.index.map(extract_method_class)
    
    # Clean column names (remove spaces, handle special chars)
    df.columns = df.columns.str.strip()
    
    # Verify columns exist
    if 'avg_tokens_per_document' not in df.columns:
        raise ValueError(f"Column 'avg_tokens_per_document' not found. Available columns: {df.columns.tolist()}")
    
    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {df.columns.tolist()}")
    
    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate baseline from other methods
    baseline_data = df[df['method_class'] == 'baseline']
    other_methods = df[df['method_class'] != 'baseline']
    
    # Plot baseline as horizontal and vertical lines if it exists
    if not baseline_data.empty:
        baseline_metric_value = baseline_data[metric].iloc[0]
        baseline_tokens_value = baseline_data['avg_tokens_per_document'].iloc[0]
        ax.axhline(y=baseline_metric_value, color='gray', linestyle='--', linewidth=2, label='Baseline')
        ax.axvline(x=baseline_tokens_value, color='gray', linestyle='--', linewidth=2)
    
    # Plot other methods as lines
    for method in other_methods['method_class'].unique():
        method_data = other_methods[other_methods['method_class'] == method].sort_values('avg_tokens_per_document')
        method_class = method_data['method_class'].iloc[0]
        ax.plot(
            method_data['avg_tokens_per_document'],
            method_data[metric],
            marker='o',
            label=method_class,
            linewidth=2,
            markersize=6
        )
    
    ax.set_xlabel('Average Tokens per Document', fontsize=12)
    ax.set_ylabel(metric, fontsize=12)
    ax.set_title(f'{dataset_name} - {metric} vs Average Tokens per Document', fontsize=14)
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
    parser.add_argument('--metric', type=str, default='ndcg@10', help='Metric to plot (e.g., map, ndcg@10)')
    parser.add_argument('--model', type=str, default='GTE-ModernColBERT-v1', help='Model name (default: GTE-ModernColBERT-v1)')
    
    args = parser.parse_args()
    
    plot_results(args.results_path, args.dataset_name, args.metric, args.model, args.output_dir)

