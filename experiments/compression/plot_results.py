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

    # IGNORE: importance-based strategies, random strategies, and IDF global
    for st in ["importance_pruning", "importance_pooling"]:
        if st in strategy_type:
            return "IGNORE"
    if strategy_type == "hybrid_importance_clustering_pooling" or (
        strategy_type and strategy_type.startswith("hybrid_imp+clust_pooling")
    ):
        return "IGNORE"

    # # IGNORE: random pooling and random pruning
    # if strategy_type in ["random_pooling", "random_pruning"]:
    #     return "IGNORE"

    if strategy_type == "attention_pruning":
        return "attention"

    if strategy_type == "idf_pruning":
        mode = strategy_cfg.get("mode")
        if mode == "global":
            return "IGNORE"  # Ignore IDF global
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

    def _extract_keep_ratio(config: dict) -> float | None:
        """Extract keep_ratio if present; if pool_factor exists, use 1/pool_factor."""
        strategies = (config or {}).get("strategies") or []
        if not strategies:
            return 1.0 if config.get("type") == "baseline" else None
        strat = strategies[0] or {}
        cfg = strat.get("config", {}) or {}
        if "keep_ratio" in cfg and cfg.get("keep_ratio") is not None:
            return float(cfg.get("keep_ratio"))
        if "pool_factor" in cfg and cfg.get("pool_factor"):
            try:
                return 1.0 / float(cfg.get("pool_factor"))
            except Exception:
                return None
        return None

    rows = []
    with open(jsonl_path, "r", encoding="utf-8") as infile:
        for line in infile:
            line = line.strip()
            if not line:
                continue
            entry = json.loads(line)
            if entry.get("type") != "result":
                continue

            config = entry.get("config", {}) or {}
            # Some files store metrics under "metrics", merged ones may use "evaluation"
            metrics = entry.get("metrics") or entry.get("evaluation") or {}
            avg_tokens = entry.get("avg_tokens_per_doc") or entry.get("avg_tokens_per_document")
            avg_tokens_std = entry.get("avg_tokens_per_doc_std") or entry.get("avg_tokens_per_document_std")
            keep_ratio = _extract_keep_ratio(config)

            row = {
                "method": entry.get("config_name"),
                "method_class": _classify_method(config),
                "avg_tokens_per_document": avg_tokens,
                "avg_tokens_per_document_std": avg_tokens_std,
                "token_count": entry.get("token_count"),
                "token_count_std": entry.get("token_count_std"),
                "compression_time": entry.get("compression_time"),
                "compression_time_std": entry.get("compression_time_std"),
                "keep_ratio": keep_ratio,
            }
            # Include all metric values (including *_std) to enable error bars
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
    """Plot avg_tokens_per_document vs metric, and keep_ratio vs compression_time."""
    df = _load_results(results_path)
    metadata = _load_metadata(results_path)

    # Filter out IGNORE methods (importance-based strategies)
    df = df[df['method_class'] != 'IGNORE']
    print(f"Filtered out importance-based strategies. Remaining methods: {df['method_class'].unique().tolist()}")

    dataset_name = metadata.get("dataset_name")
    model_name = metadata.get("model_name")
    model_name_sanitized = model_name.replace("/", "_")
    # Verify columns exist
    if 'avg_tokens_per_document' not in df.columns:
        raise ValueError(f"Column 'avg_tokens_per_document' not found. Available columns: {df.columns.tolist()}")

    if metric not in df.columns:
        raise ValueError(f"Metric '{metric}' not found. Available columns: {df.columns.tolist()}")

    metric_std_col = f"{metric}_std" if f"{metric}_std" in df.columns else None
    tokens_std_col = "avg_tokens_per_document_std" if "avg_tokens_per_document_std" in df.columns else None
    
    # Deterministic color assignment per method_class
    method_classes = sorted(df['method_class'].unique())
    # Use evenly spaced HSV colors to guarantee uniqueness per method class
    n_methods = max(len(method_classes), 1)
    palette = [plt.cm.hsv(i / n_methods) for i in range(n_methods)]
    color_map = {m: palette[i] for i, m in enumerate(method_classes)}

    # -------- Plot 1: metric vs avg tokens --------
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Separate baseline from other methods
    baseline_data = df[df['method_class'] == 'baseline']
    other_methods = df[df['method_class'] != 'baseline']
    
    # Plot baseline as horizontal and vertical lines if it exists
    if not baseline_data.empty:
        baseline_metric_value = baseline_data[metric].iloc[0]
        baseline_tokens_value = baseline_data['avg_tokens_per_document'].iloc[0]
        yerr = None
        xerr = None
        if metric_std_col and not baseline_data[metric_std_col].isna().all():
            yerr_val = baseline_data[metric_std_col].iloc[0]
            if pd.notna(yerr_val):
                yerr = float(yerr_val)
        if tokens_std_col and not baseline_data[tokens_std_col].isna().all():
            xerr_val = baseline_data[tokens_std_col].iloc[0]
            if pd.notna(xerr_val):
                xerr = float(xerr_val)

        err_kwargs = {}
        if xerr is not None:
            err_kwargs["xerr"] = xerr
        if yerr is not None:
            err_kwargs["yerr"] = yerr

        ax.errorbar(
            x=baseline_tokens_value,
            y=baseline_metric_value,
            fmt='s',
            color='gray',
            markersize=8,
            label='Baseline',
            capsize=4,
            **err_kwargs,
        )
        ax.axhline(y=baseline_metric_value, color='gray', linestyle='--', linewidth=1, alpha=0.6)
        ax.axvline(x=baseline_tokens_value, color='gray', linestyle='--', linewidth=1, alpha=0.6)
    
    # Plot other methods as lines with optional error bars
    for method in other_methods['method_class'].unique():
        method_data = other_methods[other_methods['method_class'] == method].sort_values('avg_tokens_per_document')
        x = method_data['avg_tokens_per_document']
        y = method_data[metric]
        xerr = None
        yerr = None
        if tokens_std_col and tokens_std_col in method_data:
            xerr_series = method_data[tokens_std_col].astype(float)
            if not xerr_series.isna().all():
                xerr = xerr_series.to_numpy()
        if metric_std_col and metric_std_col in method_data:
            yerr_series = method_data[metric_std_col].astype(float)
            if not yerr_series.isna().all():
                yerr = yerr_series.to_numpy()

        ax.errorbar(
            x,
            y,
            xerr=xerr,
            yerr=yerr,
            marker='o',
            label=method,
            linewidth=1.8,
            markersize=3,
            capsize=3,
            color=color_map.get(method, None),
        )
    
    # Set y-lower cutoff so roughly the bottom 20% of points are cropped out
    metric_values = df[metric].dropna()
    if not metric_values.empty:
        y_lower = metric_values.quantile(0.05)
        y_upper = metric_values.max()
        ax.set_ylim(bottom=y_lower, top=y_upper * 1.01)

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
    output_path = plot_dir / f'{model_name_sanitized}_{dataset_name}_{metric}_plot.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"Plot saved to {output_path}")

    # -------- Plot 2: compression_time vs keep_ratio --------
    df_keep = df.dropna(subset=["keep_ratio", "compression_time"])
    if not df_keep.empty:
        fig2, ax2 = plt.subplots(figsize=(10, 6))
        # Baseline keep_ratio=1 if exists
        baseline_keep = df_keep[df_keep["method_class"] == "baseline"]
        other_methods_keep = df_keep[df_keep["method_class"] != "baseline"]

        if not baseline_keep.empty:
            base_x = baseline_keep["keep_ratio"].iloc[0]
            base_y = baseline_keep["compression_time"].iloc[0]
            yerr = None
            if "compression_time_std" in baseline_keep.columns and not baseline_keep["compression_time_std"].isna().all():
                yerr_val = baseline_keep["compression_time_std"].iloc[0]
                if pd.notna(yerr_val):
                    yerr = float(yerr_val)
            ax2.errorbar(
                x=base_x,
                y=base_y,
                yerr=yerr,
                fmt='s',
                color='gray',
                markersize=8,
                label='Baseline',
                capsize=4,
            )

        for method in other_methods_keep['method_class'].unique():
            method_data = other_methods_keep[other_methods_keep['method_class'] == method].sort_values('keep_ratio')
            x = method_data['keep_ratio']
            y = method_data['compression_time']
            yerr = None
            if "compression_time_std" in method_data.columns and method_data["compression_time_std"].notna().any():
                yerr = method_data["compression_time_std"].to_numpy()
            ax2.errorbar(
                x,
                y,
                yerr=yerr,
                marker='o',
                label=method,
                linewidth=1.8,
                markersize=3,
                capsize=3,
                color=color_map.get(method, None),
            )

        ax2.set_xlabel('Keep Ratio', fontsize=12)
        ax2.set_ylabel('Compression Time (s)', fontsize=12)
        ax2.set_title(f'{dataset_name} - Compression Time vs Keep Ratio', fontsize=14)
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
        ax2.grid(True, alpha=0.3)
        plt.tight_layout()

        output_path_time = plot_dir / f'{model_name_sanitized}_{dataset_name}_compression_time_plot.png'
        plt.savefig(output_path_time, dpi=300, bbox_inches='tight')
        print(f"Compression-time plot saved to {output_path_time}")

    # Also show plot
    plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plot compression experiment results')
    parser.add_argument('--results_path', type=str, help='Path to results jsonl file')
    parser.add_argument('--output_dir', type=str, help='Path to output directory')
    parser.add_argument('--metric', type=str, default='ndcg@10', help='Metric to plot (e.g., map, ndcg@10)')
    
    args = parser.parse_args()
    
    plot_results(args.results_path, args.metric, args.output_dir)
