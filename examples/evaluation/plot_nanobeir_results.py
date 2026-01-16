"""Plot NanoBEIR evaluation results comparing multiple models."""

from __future__ import annotations

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Read the results
df = pd.read_csv("nanobeir_model_comparison.csv", index_col=0)

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

# Extract key metrics for each dataset
def get_metric(dataset, metric):
    col = f"Nano{dataset}_MaxSim_{metric}"
    return df[col].values if col in df.columns else None

# Create figure with subplots - larger figure for better spacing
fig, axes = plt.subplots(2, 2, figsize=(18, 12))
fig.suptitle("NanoBEIR Model Comparison", fontsize=16, fontweight='bold', y=0.98)

# Calculate bar width based on number of models
n_models = len(df.index)
width = 0.8 / n_models  # Total group width of 0.8, divided by number of models

# Color palette for consistent colors across plots
colors = plt.cm.tab10(np.linspace(0, 1, n_models))

# Plot 1: NDCG@10 across all datasets
ax1 = axes[0, 0]
ndcg_data = {ds: get_metric(ds, "ndcg@10") for ds in datasets}
x = np.arange(len(datasets))
for i, model in enumerate(df.index):
    values = [ndcg_data[ds][i] if ndcg_data[ds] is not None else 0 for ds in datasets]
    offset = (i - n_models/2 + 0.5) * width
    ax1.bar(x + offset, values, width, label=model, color=colors[i])
ax1.set_ylabel("NDCG@10", fontsize=11)
ax1.set_title("NDCG@10 by Dataset", fontsize=12)
ax1.set_xticks(x)
ax1.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
ax1.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax1.set_ylim(0, 1.15)
ax1.grid(axis='y', alpha=0.3)

# Plot 2: MRR@10 across all datasets
ax2 = axes[0, 1]
mrr_data = {ds: get_metric(ds, "mrr@10") for ds in datasets}
for i, model in enumerate(df.index):
    values = [mrr_data[ds][i] if mrr_data[ds] is not None else 0 for ds in datasets]
    offset = (i - n_models/2 + 0.5) * width
    ax2.bar(x + offset, values, width, label=model, color=colors[i])
ax2.set_ylabel("MRR@10", fontsize=11)
ax2.set_title("MRR@10 by Dataset", fontsize=12)
ax2.set_xticks(x)
ax2.set_xticklabels(datasets, rotation=45, ha='right', fontsize=9)
ax2.legend(loc='upper right', fontsize=9, framealpha=0.9)
ax2.set_ylim(0, 1.15)
ax2.grid(axis='y', alpha=0.3)

# Plot 3: Mean metrics comparison (bar chart)
ax3 = axes[1, 0]
mean_metrics = ["ndcg@10", "mrr@10", "map@100", "recall@10"]
mean_cols = [f"NanoBEIR_mean_MaxSim_{m}" for m in mean_metrics]
mean_data = df[mean_cols].values.T
x3 = np.arange(len(mean_metrics))
for i, model in enumerate(df.index):
    offset = (i - n_models/2 + 0.5) * width
    ax3.bar(x3 + offset, mean_data[:, i], width, label=model, color=colors[i])
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
recall_cols = [f"NanoBEIR_mean_MaxSim_{m}" for m in recall_metrics]
recall_data = df[recall_cols].values.T
x4 = np.arange(len(recall_metrics))
for i, model in enumerate(df.index):
    offset = (i - n_models/2 + 0.5) * width
    ax4.bar(x4 + offset, recall_data[:, i], width, label=model, color=colors[i])
ax4.set_ylabel("Recall", fontsize=11)
ax4.set_title("Mean Recall at Different K Values", fontsize=12)
ax4.set_xticks(x4)
ax4.set_xticklabels(["@1", "@3", "@5", "@10"], fontsize=10)
ax4.legend(fontsize=9, framealpha=0.9)
ax4.set_ylim(0, 0.85)
ax4.grid(axis='y', alpha=0.3)

plt.tight_layout(rect=[0, 0, 1, 0.96])  # Leave space for suptitle
plt.subplots_adjust(hspace=0.35, wspace=0.2)  # Add spacing between subplots
plt.savefig("nanobeir_comparison.png", dpi=150, bbox_inches='tight')
plt.savefig("nanobeir_comparison.pdf", bbox_inches='tight')
print("Saved plots to nanobeir_comparison.png and nanobeir_comparison.pdf")

# Print summary table
print("\n" + "="*60)
print("SUMMARY: Mean Metrics Across All NanoBEIR Datasets")
print("="*60)
summary_cols = [
    "NanoBEIR_mean_MaxSim_ndcg@10",
    "NanoBEIR_mean_MaxSim_mrr@10", 
    "NanoBEIR_mean_MaxSim_map@100",
    "NanoBEIR_mean_MaxSim_recall@10"
]
summary_df = df[summary_cols].copy()
summary_df.columns = ["NDCG@10", "MRR@10", "MAP@100", "Recall@10"]
print(summary_df.to_string())

# Show which model wins on each dataset
print("\n" + "="*60)
print("BEST MODEL PER DATASET (NDCG@10)")
print("="*60)
for ds in datasets:
    col = f"Nano{ds}_MaxSim_ndcg@10"
    if col in df.columns:
        best_model = df[col].idxmax()
        best_score = df[col].max()
        print(f"{ds:20s}: {best_model} ({best_score:.4f})")

plt.show()

