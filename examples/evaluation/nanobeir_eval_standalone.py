"""Standalone NanoBEIR evaluation script for comparing multiple models.

Adds hierarchical pooling (fixed-size, default 32 tokens) variants for base & finetuned models.
"""

from __future__ import annotations

import argparse
import os
import math
import pandas as pd

from pylate import evaluation, models
from pylate.evaluation.nano_beir_evaluator import (
    MAPPING_DATASET_NAME_TO_ID,
)
from pylate.models.compression import PoolingConfig, PoolingStrategy


class DocHierPoolWrapper:
    """Wraps a ColBERT model and applies hierarchical pooling to document encodings.

    Pooling is only applied when is_query=False. Queries are left unchanged.
    """

    def __init__(
        self,
        base_model: models.ColBERT,
        target_tokens: int = 32,
        protected_tokens: int = 1,
        hierarchical_variant: str = "h2pool",
    ):
        self.base = base_model
        self.strategy = PoolingStrategy(
            PoolingConfig(
                protected_tokens=protected_tokens,
                clustering_method="hierarchical",
                hierarchical_variant=hierarchical_variant,
                show_progress_bar=False,
                fixed_size=True,
                fixed_tokens=target_tokens,
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
    ("output/Alibaba-GTE-ModernColBERT-3e-05-lr-3-epochs-gemma-bs24-nway16/checkpoint-10000", "ColBERT-Base-300tok-10000", "colbert", False),
    ("output/Alibaba-GTE-ModernColBERT-3e-05-lr-3-epochs-gemma-bs24-nway16/checkpoint-10000", "ColBERT-Base-300tok-10000-HPool32", "colbert", True, 'ward_embeddings'),

    # Copy of the above with explicit h2pool variant
    ("output/Alibaba-GTE-ModernColBERT-3e-05-lr-3-epochs-gemma-bs24-nway16/checkpoint-10000", "ColBERT-Base-300tok-10000-HPool32-h2pool", "colbert", True, "h2pool"),

    ("output/ProxyAttention-ColBERT-32tok-3e-05-lr-3-epochs-full/checkpoint-5000", "ProxyAttention-V1-32tok-5000", "proxy", False),
    ("output/ProxyAttention-ColBERT-32tok-3e-05-lr-1-epochs-full/checkpoint-5000", "ProxyAttention-V1-32tok-5000-bs112", "proxy", False),
    ("output/ProxyAttention-Alibaba-ColBERT-32tok-3e-05-lr-3-epochs-full-bs24-nway16/checkpoint-10000", "ProxyAttention-Base-32tok-10000-bs24", "proxy", False),
]
"ProxyAttention-Base-32tok-5000-bs24 ColBERT-Base-300tok-5000 ColBERT-Base-300tok-5000-HPool32"


def compute_avg_doc_tokens_for_datasets(model, dataset_names: list[str]) -> float:
    """Compute the average number of tokens per document used by `model`.

    Encodes all corpus documents for the given NanoBEIR datasets and averages the
    token lengths of the returned document embeddings. Uses batch_size=1 to avoid
    any batch-dependent effects (e.g., ProxyAttention's batch-min selection).
    """
    try:
        from datasets import load_dataset
    except Exception:
        # Fallback if datasets is unavailable
        return float("nan")

    texts: list[str] = []
    for name in dataset_names:
        ds_id = MAPPING_DATASET_NAME_TO_ID.get(name.lower())
        if not ds_id:
            continue
        corpus = load_dataset(ds_id, "corpus", split="train")
        texts.extend([r["text"] for r in corpus if len(r["text"]) > 0])

    if not texts:
        return float("nan")

    embs = model.encode(
        texts,
        is_query=False,
        batch_size=1,
        convert_to_tensor=True,
        show_progress_bar=False,
    )
    # Expect a list of tensors with variable lengths
    try:
        lengths = [e.shape[0] for e in embs]
    except Exception:
        # If encode returns a single stacked tensor (unlikely for variable-length), handle gracefully
        try:
            lengths = [int(embs.shape[1])] * int(embs.shape[0])
        except Exception:
            return float("nan")

    return float(sum(lengths) / len(lengths)) if lengths else float("nan")


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
        default="nanobeir_model_comparison_new.csv",
        help="Path to the results CSV (default: nanobeir_model_comparison.csv)",
    )
    parser.add_argument(
        "--datasets-scifact-nfcorpus",
        action="store_true",
        help="Evaluate on FULL BEIR SciFact and NFCorpus (not Nano versions).",
    )
    # HPool uses fixed-size pooling only (no factor-based options)
    parser.add_argument(
        "--list-models",
        action="store_true",
        help="List available model display names and exit.",
    )
    parser.add_argument(
        "--show-progress",
        action="store_true",
        help="Show tqdm progress bars for query/corpus encoding and corpus chunking.",
    )
    parser.add_argument(
        "--enable-avg-doc-tokens",
        action="store_true",
        help="Compute and include avg_doc_tokens in results (WARNING: slow). Disabled by default.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size to use for both query and corpus encoding (default: 32).",
    )
    args = parser.parse_args()
    # Dataset selection override
    dataset_names = None
    if args.datasets_scifact_nfcorpus:
        # For this flag, we run FULL BEIR SciFact and NFCorpus via load_beir
        dataset_names = ["scifact", "nfcorpus"]

    if args.list_models:
        print("Available model display names:")
        for entry in MODELS:
            if len(entry) == 5:
                _, name, mtype, use_hpool, variant = entry
            else:
                _, name, mtype, use_hpool = entry
                variant = None
            tag = "HPool32" if (mtype == "colbert" and use_hpool) else "no-HPool"
            if variant:
                tag += f" ({variant})"
            print(f"- {name}  (type={mtype}, {tag})")
        raise SystemExit(0)

    # Figure out which models to run
    selected = MODELS
    if args.only is not None:
        wanted = set(args.only)
        # Support 4- or 5-tuple entries
        name_to_entry = {}
        for entry in MODELS:
            if len(entry) == 5:
                path, name, mtype, use_hpool, variant = entry
                name_to_entry[name] = (path, name, mtype, use_hpool, variant)
            else:
                path, name, mtype, use_hpool = entry
                name_to_entry[name] = (path, name, mtype, use_hpool)
        not_found = [n for n in wanted if n not in name_to_entry]
        if not_found:
            print("Error: the following --only names were not found:\n  " + "\n  ".join(not_found))
            print("\nChoices are:")
            for _, name, _, _ in MODELS:
                print(f"  - {name}")
            raise SystemExit(2)
        selected = [name_to_entry[n] for n in args.only]

    all_results: dict[str, dict] = {}

    # Fixed-size pooling target for HPool variants
    HPOOL_TARGET_TOKENS = 32

    for entry in selected:
        if len(entry) == 5:
            model_path, model_name, model_type, use_hpool, pool_variant = entry
        else:
            model_path, model_name, model_type, use_hpool = entry
            pool_variant = None
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
                target_tokens=HPOOL_TARGET_TOKENS,
                protected_tokens=1,
                hierarchical_variant=pool_variant or "h2pool",
            )

        # Create and run evaluator(s)
        if args.datasets_scifact_nfcorpus:
            # Full BEIR path for SciFact and NFCorpus
            full_results: dict[str, float] = {}
            beir_prefix_map = {"scifact": "SciFact", "nfcorpus": "NFCorpus"}
            per_ds_results: dict[str, dict[str, float]] = {}

            for ds in dataset_names:
                # Load full BEIR dataset
                documents, queries, qrels = evaluation.load_beir(
                    dataset_name=ds, split="test"
                )
                corpus_dict = {d["id"]: d["text"] for d in documents}
                # qrels from BEIR is dict[qid] -> dict[doc_id]->score; convert to set of doc_ids
                relevant_docs = {
                    qid: set(doc_scores.keys()) for qid, doc_scores in qrels.items()
                }
                human_name = beir_prefix_map.get(ds, ds.title())
                beir_eval = evaluation.PyLateInformationRetrievalEvaluator(
                    queries=queries,
                    corpus=corpus_dict,
                    relevant_docs=relevant_docs,
                    name=human_name,
                    batch_size=args.batch_size,
                    show_progress_bar=args.show_progress,
                )
                ds_scores = beir_eval(model)
                per_ds_results[ds] = ds_scores
                full_results.update(ds_scores)

            # Compute mean across selected BEIR datasets for each metric suffix
            metrics_by_suffix: dict[str, list[float]] = {}
            for ds, ds_scores in per_ds_results.items():
                prefix = beir_prefix_map.get(ds, ds.title()) + "_"
                for k, v in ds_scores.items():
                    if k.startswith(prefix):
                        suffix = k[len(prefix) :]
                        metrics_by_suffix.setdefault(suffix, []).append(float(v))

            for suffix, values in metrics_by_suffix.items():
                if values:
                    full_results[f"BEIR_mean_{suffix}"] = sum(values) / len(values)

            results = full_results
        else:
            evaluator = (
                evaluation.NanoBEIREvaluator(dataset_names=dataset_names, show_progress_bar=args.show_progress)
                if dataset_names is not None
                else evaluation.NanoBEIREvaluator(show_progress_bar=args.show_progress)
            )
            # Ensure the evaluator uses the requested batch size for encoding
            try:
                setattr(evaluator, "batch_size", int(args.batch_size))
            except Exception:
                pass
            results = evaluator(model)

        # Optionally augment results with average document token count across the evaluated datasets
        if args.enable_avg_doc_tokens:
            if args.datasets_scifact_nfcorpus:
                # Skip heavy avg_doc_tokens in full BEIR mode; leave as NaN
                results["avg_doc_tokens"] = float("nan")
            else:
                ds_for_stats = dataset_names or list(MAPPING_DATASET_NAME_TO_ID.keys())
                avg_tokens = compute_avg_doc_tokens_for_datasets(model, ds_for_stats)
                results["avg_doc_tokens"] = avg_tokens

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

