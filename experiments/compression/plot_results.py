#!/usr/bin/env python3
"""
Minimal script to plot compression experiment results.
Plots avg_tokens_per_document vs metric, grouped by method.
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _classify_method(config: dict) -> str:
    """Return a compact label for the compression strategy."""
    if not config:
        return "other"
    if config.get("type") == "baseline":
        return "baseline"

    strategies = config.get("strategies") or []
    if not strategies:
        return "other"

    strategy = strategies[0]
    strategy_type = (strategy or {}).get("type")
    strategy_cfg = (strategy or {}).get("config", {})

    if strategy_type == "attention_pruning":
        return f"attention-{strategy_cfg.get('head_reduction', "sum")}"
    if strategy_type == "compactor_pruning":
        sketch_str = "" if strategy_cfg.get('sketch_dim') is None else f"-sketch-{strategy_cfg.get('sketch_dim')}"
        return f"compactor-attn-{strategy_cfg.get('attention_head_reduction', "sum")}-lev-{strategy_cfg.get('leverage_head_reduction', "sum")}-lambda-{strategy_cfg.get('lambda_mix')}{sketch_str}"

    if strategy_type == "idf_pruning":
        mode = strategy_cfg.get("mode")
        if mode == "global":
            return "idf-global"
        if mode == "document":
            return "idf-doc"
        return "idf"

    if strategy_type == "pooling":
        clustering = strategy_cfg.get("clustering_method")
        if clustering:
            return f"pooling-{clustering}"
        return "pooling"

    return strategy_type or "other"


def _load_results(jsonl_path: str) -> pd.DataFrame:
    """Read compression experiment results from the jsonl file."""
    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "result":
                continue

            config = entry.get("config", {})
            metrics = entry.get("metrics", {}) or {}
            row = {
                "method": entry.get("config_name"),
                "method_class": _classify_method(config),
                "avg_tokens_per_document": entry.get("avg_tokens_per_doc"),
                "token_count": entry.get("token_count"),
            }
            row.update(metrics)
            rows.append(row)

    if not rows:
        raise ValueError(f"No result entries found in {jsonl_path}")

    df = pd.DataFrame(rows)
    df.columns = df.columns.str.strip()
    return df

def _load_metadata(jsonl_path: str) -> dict:
    """Read compression experiment metadata from the jsonl file."""
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            entry = json.loads(line)
            if entry.get("type") == "metadata":
                return entry
    raise ValueError(f"No metadata entry found in {jsonl_path}")

def plot_results(results_path, metric, output_dir):
    """Plot avg_tokens_per_document vs metric, grouped by method."""
    df = _load_results(results_path)
    metadata = _load_metadata(results_path)

    dataset_name = metadata.get("dataset_name")
    model_name = metadata.get("model_name")
    model_name_sanitized = model_name.replace("/", "_")
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
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, -0.15), ncol=2)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    
    # Create plot directory if it doesn't exist
    plot_dir = Path(output_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)
    
    # Save plot
    output_path = plot_dir / f'{model_name_sanitized}_{dataset_name}_{metric}_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")
    
    # Also show plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot compression experiment results')
    parser.add_argument('--results_path', type=str, help='Path to results jsonl file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--metric', type=str, default='ndcg@10', help='Metric to plot (e.g., map, ndcg@10)')
    
    args = parser.parse_args()
    
    plot_results(args.results_path, args.metric, args.output_dir)

