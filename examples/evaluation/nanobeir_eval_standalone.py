"""Standalone NanoBEIR evaluation script for comparing multiple models.

Adds hierarchical pooling (~32 tokens) variants for base & finetuned models.
"""

from __future__ import annotations

import argparse
import os
import math
import pandas as pd

from pylate import evaluation, models
from pylate.models.compression import PoolingConfig, PoolingStrategy


class DocHierPoolWrapper:
    """Wraps a ColBERT model and applies hierarchical pooling to document encodings.

    Pooling is only applied when is_query=False. Queries are left unchanged.
    """

    def __init__(self, base_model: models.ColBERT, pool_factor: int = 10, protected_tokens: int = 1):
        self.base = base_model
        self.strategy = PoolingStrategy(
            PoolingConfig(
                pool_factor=pool_factor,
                protected_tokens=protected_tokens,
                clustering_method="hierarchical",
                show_progress_bar=False,
            )
        )

    # Pass-through attributes commonly used by evaluators
    def truncate_sentence_embeddings(self, *args, **kwargs):
        return self.base.truncate_sentence_embeddings(*args, **kwargs)

    @property
    def device(self):
        return getattr(self.base, "device", None)

    # Score function passthroughs expected by evaluators
    @property
    def similarity_fn_name(self):
        return getattr(self.base, "similarity_fn_name", "MaxSim")

    def similarity(self, *args, **kwargs):
        return self.base.similarity(*args, **kwargs)

    @property
    def model_card_data(self):
        return getattr(self.base, "model_card_data", None)

    # Fallback passthrough for any other attribute the evaluator might access
    def __getattr__(self, name):
        return getattr(self.base, name)

    def encode(self, sentences, *args, **kwargs):
        is_query = kwargs.get("is_query", False)
        # Ensure tensors are returned for padding downstream
        kwargs.setdefault("convert_to_tensor", True)
        embs = self.base.encode(sentences, *args, **kwargs)

        if is_query:
            return embs

        # Apply hierarchical pooling to document embeddings
        # PoolingStrategy.compress expects (embeddings, artifacts)
        pooled_embs, _ = self.strategy.compress(embs, artifacts={})
        return pooled_embs


# Define the models to evaluate with (path, display_name, model_type, hpool)
# model_type: "colbert" for standard ColBERT, "proxy" for ProxyAttentionColBERT
MODELS = [
    ("lightonai/GTE-ModernColBERT-v1", "GTE-ModernColBERT-v1-base", "colbert", False),
    ("lightonai/GTE-ModernColBERT-v1", "GTE-ModernColBERT-v1-base-HPool32", "colbert", True),
    ("output/lightonai_GTE-ModernColBERT-v1-3e-05-lr-3-epochs-gemma/checkpoint-5000", "ColBERT-V1-300tok-5000", "colbert", False),
    ("output/lightonai_GTE-ModernColBERT-v1-3e-05-lr-3-epochs-gemma/checkpoint-5000", "ColBERT-V1-300tok-5000-HPool32", "colbert", True),
    ("output/Alibaba-GTE-ModernColBERT-3e-05-lr-3-epochs-gemma-bs24-nway16/checkpoint-25000", "ColBERT-Base-300tok-5000", "colbert", False),
    ("output/Alibaba-GTE-ModernColBERT-3e-05-lr-3-epochs-gemma-bs24-nway16/checkpoint-25000", "ColBERT-Base-300tok-5000-HPool32", "colbert", True),

    ("output/ProxyAttention-ColBERT-32tok-3e-05-lr-3-epochs-full/checkpoint-5000", "ProxyAttention-V1-32tok-5000", "proxy", False),
    ("output/ProxyAttention-ColBERT-32tok-3e-05-lr-1-epochs-full/checkpoint-5000", "ProxyAttention-V1-32tok-5000-bs112", "proxy", False),
    ("output/ProxyAttention-Alibaba-ColBERT-32tok-3e-05-lr-3-epochs-full-bs24-nway16/checkpoint-25000", "ProxyAttention-Base-32tok-5000-bs24", "proxy", False),
]
"ProxyAttention-Base-32tok-5000-bs24 ColBERT-Base-300tok-5000 ColBERT-Base-300tok-5000-HPool32"

def make_hpool_factor(target_tokens: int = 32, protected_tokens: int = 1, assumed_doc_len: int = 300) -> int:
    """Compute a pool_factor that approximately yields target_tokens per document.

    pooled_len ≈ protected_tokens + floor((doc_len - protected_tokens) / pool_factor)
    """
    effective = max(assumed_doc_len - protected_tokens, 1)
    clusters = max(target_tokens - protected_tokens, 1)
    # Use round to better hit target on average
    return max(int(round(effective / clusters)), 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Standalone NanoBEIR evaluation for multiple models with optional merge/partial runs."
    )
    parser.add_argument(
        "--only",
        nargs="+",
        default=None,
        help=(
            "Display name(s) of models to run (exact match). If omitted, all models are evaluated.\n"
            "Choices: "
            + ", ".join([m[1] for m in MODELS])
        ),
    )
    parser.add_argument(
        "--merge",
        action="store_true",
        help=(
            "Merge new results into an existing CSV instead of overwriting it. "
            "If the CSV exists, rows for rerun models are replaced; others are kept."
        ),
    )
    parser.add_argument(
        "--output",
        default="nanobeir_model_comparison.csv",
        help="Path to the results CSV (default: nanobeir_model_comparison.csv)",
    )
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model display names and exit.",
    )
    args = parser.parse_args()

    if args.list_models:
        print("Available model display names:")
        for _, name, mtype, use_hpool in MODELS:
            tag = "HPool32" if (mtype == "colbert" and use_hpool) else "no-HPool"
            print(f"- {name}  (type={mtype}, {tag})")
        raise SystemExit(0)

    # Figure out which models to run
    selected = MODELS
    if args.only is not None:
        wanted = set(args.only)
        name_to_entry = {name: (path, name, mtype, use_hpool) for (path, name, mtype, use_hpool) in MODELS}
        not_found = [n for n in wanted if n not in name_to_entry]
        if not_found:
            print("Error: the following --only names were not found:\n  " + "\n  ".join(not_found))
            print("\nChoices are:")
            for _, name, _, _ in MODELS:
                print(f"  - {name}")
            raise SystemExit(2)
        selected = [name_to_entry[n] for n in args.only]

    all_results: dict[str, dict] = {}

    # Choose a pool_factor that targets ~32 tokens per doc when docs are ~300 tokens
    pool_factor_approx_32 = make_hpool_factor(target_tokens=32, protected_tokens=1, assumed_doc_len=300)

    for model_path, model_name, model_type, use_hpool in selected:
        print(f"\n{'='*60}")
        print(f"Evaluating: {model_name}")
        print(f"Path: {model_path}")
        print(f"Type: {model_type} | HPool32: {use_hpool}")
        print(f"{'='*60}\n")

        # Load base model
        if model_type == "proxy":
            model = models.ProxyAttentionColBERT.load(
                path=model_path,
                document_length=300,
            )
        else:
            model = models.ColBERT(
                model_name_or_path=model_path,
                document_length=300,
            )

        # Wrap with hierarchical pooling for documents, if requested (only for ColBERT)
        if use_hpool and model_type == "colbert":
            model = DocHierPoolWrapper(
                base_model=model,
                pool_factor=pool_factor_approx_32,
                protected_tokens=1,
            )

        # Create and run NanoBEIREvaluator
        evaluator = evaluation.NanoBEIREvaluator()
        results = evaluator(model)

        # Store results with the display name
        all_results[model_name] = results

        print(f"\nResults for {model_name}:")
        print(results)

    # New results as DataFrame
    df_new = pd.DataFrame(all_results).T

    # Save or merge results
    output_file = args.output
    if args.merge and os.path.exists(output_file):
        try:
            df_old = pd.read_csv(output_file, index_col=0)
        except Exception as e:
            print(f"Warning: Failed to read existing CSV at {output_file}: {e}. Overwriting.")
            df_old = None

        if df_old is not None:
            # Drop any re-run models from old results and append new
            df_old = df_old.drop(index=list(df_new.index), errors="ignore")
            df_combined = pd.concat([df_old, df_new], axis=0, sort=False)
        else:
            df_combined = df_new

        # Print summary
        print(f"\n{'='*60}")
        print("COMBINED COMPARISON SUMMARY (after merge)")
        print(f"{'='*60}\n")
        print(df_combined.to_string())

        df_combined.to_csv(output_file)
        print(f"\nMerged results saved to: {output_file}")
    else:
        # Print summary
        print(f"\n{'='*60}")
        print("COMPARISON SUMMARY (new runs)")
        print(f"{'='*60}\n")
        print(df_new.to_string())

        df_new.to_csv(output_file)
        print(f"\nResults saved to: {output_file}")

