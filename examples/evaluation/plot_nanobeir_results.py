"""Plot NanoBEIR evaluation results comparing multiple models."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
# file_name = "nanobeir_model_comparison_new"
# file_name = "nanobeir_model_comparison"
file_name = "model_comparison_scifact_nfcorpus_clean2_10000"


df = pd.read_csv(f"{file_name}.csv", index_col=0)

# Shorten model names for plotting
short_names = {
    "GTE-ModernColBERT-v1-base": "Base",
    "GTE-ModernColBERT-v1-base-HPool32": "Base-HPool32",
    "GTE-ModernColBERT-v1-finetuned-5000": "Finetuned",
    "GTE-ModernColBERT-v1-finetuned-HPool32": "Finetuned-HPool32",
    "ProxyAttention-ColBERT-32tok-5000": "ProxyAttn",
}
df.index = df.index.map(lambda x: short_names.get(x, x))

# Define datasets
datasets = [
    "ClimateFEVER", "DBPedia", "FEVER", "FiQA2018", "HotpotQA", "MSMARCO",
    "NFCorpus", "NQ", "QuoraRetrieval", "SCIDOCS", "ArguAna", "SciFact", "Touche2020"
]

# Determine which dataset columns are actually present (Nano vs full BEIR)
def _has_col(df: pd.DataFrame, ds: str, metric: str) -> tuple[bool, str]:
    nano = f"Nano{ds}_MaxSim_{metric}"
    full = f"{ds}_MaxSim_{metric}"
    if nano in df.columns:
        return True, nano
    if full in df.columns:
        return True, full
    return False, ""

present_datasets = [ds for ds in datasets if _has_col(df, ds, "ndcg@10")[0]]
if not present_datasets:
    # Fallback: infer from any ndcg@10-like columns
    present_datasets = [
        col.split("_MaxSim_")[0].removeprefix("Nano")
        for col in df.columns if col.endswith("_MaxSim_ndcg@10")
    ]

# Extract key metrics for each dataset
def get_metric(dataset, metric):
    ok, col = _has_col(df, dataset, metric)
    return df[col].values if ok else None

# Create figure with subplots - larger figure for better spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
# Dynamic title reflecting dataset type
title_prefix = (
    "NanoBEIR" if any(c.startswith("Nano") for c in df.columns)
    else ("BEIR" if any(c.startswith("BEIR_mean_") for c in df.columns) or any((not c.startswith("Nano")) and c.endswith("_MaxSim_ndcg@10") for c in df.columns) else "")
)
fig.suptitle(f"{title_prefix + ' ' if title_prefix else ''}Model Comparison", fontsize=16, fontweight='bold', y=0.98)

# Calculate bar width based on number of models
n_models = len(df.index)
width = 0.8 / n_models  # Total group width of 0.8, divided by number of models

# Color palette for consistent colors across plots
colors = plt.cm.tab10(np.linspace(0, 1, n_models))

# Helper: add numeric value labels to bar containers
def _label_bars(ax, bars, fmt: str = "{:.3f}", fontsize: int = 8):
    y_min, y_max = ax.get_ylim()
    offset = 0.01 * (y_max - y_min)
    for b in bars:
        h = b.get_height()
        try:
            val = float(h)
        except Exception:
            continue
        ax.text(
            b.get_x() + b.get_width() / 2.0,
            h + offset,
            fmt.format(val),
            ha="center",
            va="bottom",
            fontsize=fontsize,
            rotation=0,
            color="black",
            clip_on=True,
        )

# Plot 1: NDCG@10 across all datasets
ax1 = axes[0, 0]
ndcg_data = {ds: get_metric(ds, "ndcg@10") for ds in present_datasets}
x = np.arange(len(present_datasets))
for i, model in enumerate(df.index):
    values = [ndcg_data[ds][i] if ndcg_data[ds] is not None else 0 for ds in present_datasets]
    offset = (i - n_models/2 + 0.5) * width
    bars = ax1.bar(x + offset, values, width, label=model, color=colors[i])
    _label_bars(ax1, bars)
ax1.set_ylabel("NDCG@10", fontsize=11)
ax1.set_title("NDCG@10 by Dataset", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(present_datasets, rotation=45, ha='right', fontsize=9)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: MRR@10 across all datasets
ax2 = axes[0, 1]
mrr_data = {ds: get_metric(ds, "mrr@10") for ds in present_datasets}
for i, model in enumerate(df.index):
    values = [mrr_data[ds][i] if mrr_data[ds] is not None else 0 for ds in present_datasets]
    offset = (i - n_models/2 + 0.5) * width
    bars = ax2.bar(x + offset, values, width, label=model, color=colors[i])
    _label_bars(ax2, bars)
ax2.set_ylabel("MRR@10", fontsize=11)
ax2.set_title("MRR@10 by Dataset", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(present_datasets, rotation=45, ha='right', fontsize=9)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Mean metrics comparison (bar chart)
ax3 = axes[1, 0]
mean_metrics = ["ndcg@10", "mrr@10", "map@100", "recall@10"]
# Use mean columns if present; otherwise compute on-the-fly across present datasets
if all(f"NanoBEIR_mean_MaxSim_{m}" in df.columns for m in mean_metrics):
    mean_cols = [f"NanoBEIR_mean_MaxSim_{m}" for m in mean_metrics]
    mean_data = df[mean_cols].values.T
elif all(f"BEIR_mean_MaxSim_{m}" in df.columns for m in mean_metrics):
    mean_cols = [f"BEIR_mean_MaxSim_{m}" for m in mean_metrics]
    mean_data = df[mean_cols].values.T
else:
    mean_rows = []
    for m in mean_metrics:
        cols = []
        for ds in present_datasets:
            ok, col = _has_col(df, ds, m)
            if ok:
                cols.append(col)
        if cols:
            mean_rows.append(df[cols].mean(axis=1).values)
        else:
            mean_rows.append(np.zeros(len(df.index)))
    mean_data = np.vstack(mean_rows)
x3 = np.arange(len(mean_metrics))
for i, model in enumerate(df.index):
    offset = (i - n_models/2 + 0.5) * width
    bars = ax3.bar(x3 + offset, mean_data[:, i], width, label=model, color=colors[i])
    _label_bars(ax3, bars)
ax3.set_ylabel("Score", fontsize=11)
ax3.set_title("Mean Metrics Across All Datasets", fontsize=12)
ax3.set_xticks(x3)
ax3.set_xticklabels(["NDCG@10", "MRR@10", "MAP@100", "Recall@10"], fontsize=10)
ax3.legend(fontsize=9, framealpha=0.9)
ax3.set_ylim(0, 1.0)
ax3.grid(axis='y', alpha=0.3)

# Plot 4: Recall@k comparison for mean
ax4 = axes[1, 1]
recall_metrics = ["recall@1", "recall@3", "recall@5", "recall@10"]
if all(f"NanoBEIR_mean_MaxSim_{m}" in df.columns for m in recall_metrics):
    recall_cols = [f"NanoBEIR_mean_MaxSim_{m}" for m in recall_metrics]
    recall_data = df[recall_cols].values.T
elif all(f"BEIR_mean_MaxSim_{m}" in df.columns for m in recall_metrics):
    recall_cols = [f"BEIR_mean_MaxSim_{m}" for m in recall_metrics]
    recall_data = df[recall_cols].values.T
else:
    recall_rows = []
    for m in recall_metrics:
        cols = []
        for ds in present_datasets:
            ok, col = _has_col(df, ds, m)
            if ok:
                cols.append(col)
        if cols:
            recall_rows.append(df[cols].mean(axis=1).values)
        else:
            recall_rows.append(np.zeros(len(df.index)))
    recall_data = np.vstack(recall_rows)
x4 = np.arange(len(recall_metrics))
for i, model in enumerate(df.index):
    offset = (i - n_models/2 + 0.5) * width
    bars = ax4.bar(x4 + offset, recall_data[:, i], width, label=model, color=colors[i])
    _label_bars(ax4, bars)
ax4.set_ylabel("Recall", fontsize=11)
ax4.set_title("Mean Recall at Different K Values", fontsize=12)
ax4.set_xticks(x4)
ax4.set_xticklabels(["@1", "@3", "@5", "@10"], fontsize=10)
ax4.legend(fontsize=9, framealpha=0.9)
ax4.set_ylim(0, 0.85)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
plt.subplots_adjust(hspace=0.35, wspace=0.2)  # Add spacing between subplots
plt.savefig(f"{file_name}.png", dpi=150, bbox_inches='tight')
plt.savefig(f"{file_name}.pdf", bbox_inches='tight')
print(f"Saved plots to {file_name}.png and {file_name}.pdf")

# Print summary table
print("\n" + "="*60)
print("SUMMARY: Mean Metrics Across Selected Datasets")
print("="*60)
summary_options = [
    [
        "NanoBEIR_mean_MaxSim_ndcg@10",
        "NanoBEIR_mean_MaxSim_mrr@10",
        "NanoBEIR_mean_MaxSim_map@100",
        "NanoBEIR_mean_MaxSim_recall@10",
    ],
    [
        "BEIR_mean_MaxSim_ndcg@10",
        "BEIR_mean_MaxSim_mrr@10",
        "BEIR_mean_MaxSim_map@100",
        "BEIR_mean_MaxSim_recall@10",
    ],
]
summary_df = None
for cols in summary_options:
    if all(c in df.columns for c in cols):
        summary_df = df[cols].copy()
        break
if summary_df is None:
    computed = []
    for m in ["ndcg@10", "mrr@10", "map@100", "recall@10"]:
        cols = []
        for ds in present_datasets:
            ok, col = _has_col(df, ds, m)
            if ok:
                cols.append(col)
        series = df[cols].mean(axis=1) if cols else pd.Series([0.0] * len(df), index=df.index)
        computed.append(series)
    summary_df = pd.concat(computed, axis=1)
summary_df.columns = ["NDCG@10", "MRR@10", "MAP@100", "Recall@10"]
print(summary_df.to_string())
summary_df.to_csv(f"{file_name}_summary.csv")

# Show which model wins on each dataset
print("\n" + "="*60)
print("BEST MODEL PER DATASET (NDCG@10)")
print("="*60)
for ds in present_datasets:
    ok, col = _has_col(df, ds, "ndcg@10")
    if ok:
        best_model = df[col].idxmax()
        best_score = df[col].max()
        print(f"{ds:20s}: {best_model} ({best_score:.4f})")

plt.show()

