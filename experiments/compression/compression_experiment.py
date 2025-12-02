"""Evaluation script for BEIR datasets with compression experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
import time
from tqdm.autonotebook import tqdm

import pandas as pd
import torch

from pylate import evaluation, indexes, models, retrieve
from pylate.models import ColBERT
from pylate.models.compression import (
    CompressionConfig,
    IDFPruningConfig,
    IDFPruningStrategy,
    PoolingConfig,
    PoolingStrategy,
    AttentionPruningConfig,
    AttentionPruningStrategy,
    CompactorPruningConfig,
    CompactorPruningStrategy,
)

# Query length mapping for different datasets
QUERY_LEN = {
    "quora": 32,
    "climate-fever": 64,
    "nq": 32,
    "msmarco": 32,
    "hotpotqa": 32,
    "nfcorpus": 32,
    "scifact": 48,
    "trec-covid": 48,
    "fiqa": 32,
    "arguana": 64,
    "scidocs": 48,
    "dbpedia-entity": 32,
    "webis-touche2020": 32,
    "fever": 32,
    "cqadupstack/android": 32,
    "cqadupstack/english": 32,
    "cqadupstack/gaming": 32,
    "cqadupstack/gis": 32,
    "cqadupstack/mathematica": 32,
    "cqadupstack/physics": 32,
    "cqadupstack/programmers": 32,
    "cqadupstack/stats": 32,
    "cqadupstack/tex": 32,
    "cqadupstack/unix": 32,
    "cqadupstack/webmasters": 32,
    "cqadupstack/wordpress": 32,
}


def load_model(model_name: str, dataset_name: str) -> ColBERT:
    """
    Load and initialize the ColBERT model.

    Parameters
    ----------
    model_name : str
        Name/path of the model to load
    dataset_name : str
        Dataset name to determine query length

    Returns
    -------
    ColBERT
        Initialized model
    """
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=QUERY_LEN.get(dataset_name, 32),
    )

    print(f"✓ Loaded model: {model_name}")
    print(f"  Document length: {model.document_length}")
    print(f"  Query length: {model.query_length}")

    return model


def load_dataset(dataset_name: str) -> tuple[list[dict], dict, dict]:
    """
    Load dataset (documents, queries, qrels).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load

    Returns
    -------
    tuple
        (documents, queries, qrels)
    """
    print("\n" + "=" * 80)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 80)

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    print(f"✓ Loaded dataset: {dataset_name}")
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")
    print(f"  Qrels: {len(qrels)}")

    return documents, queries, qrels


def sanitize_name(name: str) -> str:
    """Sanitize dataset/model names so they can be safely used in filesystem paths."""
    return (
        name.replace("/", "_")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("\\", "_")
    )


def get_encoded_data_dir(
    model_name: str,
    dataset_name: str,
    encoded_data_base_dir: Path | str | None = None,
    encode_id: str | None = None,
) -> Path:
    """
    Get the directory path for encoded data storage/loading.
    
    Parameters
    ----------
    model_name : str
        Model name
    dataset_name : str
        Dataset name
    encoded_data_base_dir : Path | str | None, optional
        Base directory for encoded data. If None, defaults to "encoded_data".
        If provided, should already include model/dataset structure (e.g., 
        results/compression_experiments/<model>/<dataset>/embeddings).
    encode_id : str | None, optional
        Encoding run ID. If None, uses timestamp-based ID
    
    Returns
    -------
    Path
        Path to encoded data directory
    """
    if encode_id is None:
        encode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if encoded_data_base_dir is None:
        # Default case: build full path from scratch
        encoded_data_base_dir = Path("encoded_data")
        model_dir = sanitize_name(model_name)
        dataset_dir = sanitize_name(dataset_name)
        return encoded_data_base_dir / model_dir / dataset_dir / encode_id
    else:
        # Base dir provided: assume it already includes model/dataset structure
        # Just append the encode_id
        encoded_data_base_dir = Path(encoded_data_base_dir)
        return encoded_data_base_dir / encode_id


def save_encoded_data(
    encoded_data_dir: Path,
    documents_embeddings: list[torch.Tensor],
    queries_embeddings: list[torch.Tensor],
    artifacts: dict[str, Any],
    metadata: dict[str, Any],
) -> None:
    """
    Save encoded documents, queries, artifacts, and metadata to disk.
    
    Parameters
    ----------
    encoded_data_dir : Path
        Directory to save encoded data
    documents_embeddings : list[torch.Tensor]
        Document embeddings
    queries_embeddings : list[torch.Tensor]
        Query embeddings
    artifacts : dict[str, Any]
        Compression artifacts
    metadata : dict[str, Any]
        Metadata about the encoding (model, dataset, etc.)
    """
    encoded_data_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"\nSaving encoded data to: {encoded_data_dir}")
    
    # Save embeddings
    torch.save(documents_embeddings, encoded_data_dir / "documents_embeddings.pt")
    torch.save(queries_embeddings, encoded_data_dir / "queries_embeddings.pt")
    
    # Save artifacts
    torch.save(artifacts, encoded_data_dir / "artifacts.pt")
    
    # Save metadata as JSON
    with open(encoded_data_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, default=str, indent=2)
    
    print(f"✓ Saved encoded data:")
    print(f"  - Documents embeddings: {len(documents_embeddings):,} documents")
    print(f"  - Queries embeddings: {len(queries_embeddings):,} queries")
    print(f"  - Artifacts: {list(artifacts.keys())}")
    print(f"  - Metadata: {metadata}")


def load_encoded_data(encoded_data_dir: Path, device: torch.device = "cpu") -> tuple[list[torch.Tensor], list[torch.Tensor], dict[str, Any], dict[str, Any]]:
    """
    Load encoded documents, queries, artifacts, and metadata from disk.
    
    Parameters
    ----------
    encoded_data_dir : Path
        Directory containing encoded data
    
    Returns
    -------
    tuple
        (documents_embeddings, queries_embeddings, artifacts, metadata)
    """
    if not encoded_data_dir.exists():
        raise FileNotFoundError(f"Encoded data directory not found: {encoded_data_dir}")
    
    print(f"\nLoading encoded data from: {encoded_data_dir}")
    
    # Load embeddings
    documents_embeddings = torch.load(encoded_data_dir / "documents_embeddings.pt", map_location=device)
    queries_embeddings = torch.load(encoded_data_dir / "queries_embeddings.pt", map_location=device)
    
    # Load artifacts
    artifacts = torch.load(encoded_data_dir / "artifacts.pt", map_location="cpu")
    
    # Load metadata
    with open(encoded_data_dir / "metadata.json", "r") as f:
        metadata = json.load(f)
    
    print(f"✓ Loaded encoded data:")
    print(f"  - Documents embeddings: {len(documents_embeddings):,} documents")
    print(f"  - Queries embeddings: {len(queries_embeddings):,} queries")
    print(f"  - Artifacts: {list(artifacts.keys())}")
    print(f"  - Metadata: {metadata}")
    
    return documents_embeddings, queries_embeddings, artifacts, metadata


def serialize_config_for_storage(config: CompressionConfig | None) -> dict[str, Any]:
    """Serialize a compression config (or baseline) for storage."""
    if config is None:
        return {"type": "baseline", "description": "No compression"}
    return config.serialize()


def save_results_jsonl(
    output_dir: Path,
    run_id: str,
    model_name: str,
    dataset_name: str,
    args: argparse.Namespace,
    configs: list[CompressionConfig | None],
    stats: dict[str, Any],
    evaluation_results: list[dict[str, Any]],
) -> Path:
    """
    Save experiment metadata, compression configs, timing, and per-config results to JSONL.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = output_dir / f"results_{run_id}.jsonl"

    metadata_entry = {
        "type": "metadata",
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "dataset_name": dataset_name,
        "num_documents": stats.get("num_documents"),
        "num_configs": len(configs),
        "args": {
            "index_type": args.index_type,
            "batch_size": args.batch_size,
            "metrics": args.metrics,
            "configs_file": args.configs_file,
        },
        "timing": {
            "encoding_time": stats.get("encoding_time"),
            "query_encoding_time": stats.get("query_encoding_time"),
            "total_time": stats.get("total_time"),
        },
        "configs": [serialize_config_for_storage(config) for config in configs],
    }

    with open(jsonl_path, "w") as f:
        f.write(json.dumps(metadata_entry, default=str) + "\n")

        for result in evaluation_results:
            config_idx = result["config_idx"]
            entry = {
                "type": "result",
                "run_id": run_id,
                "config_idx": config_idx,
                "config_name": result["config_name"],
                "config": serialize_config_for_storage(configs[config_idx]),
                "token_count": result["token_count"],
                "avg_tokens_per_doc": result["avg_tokens_per_doc"],
                "compression_time": stats["compression_times"][config_idx],
                "metrics": result["evaluation"],
                "runfile_path": result.get("runfile_path"),
            }
            f.write(json.dumps(entry, default=str) + "\n")

    print(f"\nSaved JSONL results to: {jsonl_path}")
    return jsonl_path


def load_configs_from_jsonl(jsonl_path: Path, model: ColBERT) -> list[CompressionConfig | None]:
    """
    Load compression configurations from a JSONL file.
    
    Each line should be a JSON object representing a CompressionConfig.
    For baseline (no compression), use either:
    - {"type": "baseline"} or {"description": "baseline"}
    - An empty strategies list: {"strategies": [], "description": "Baseline"}
    
    Parameters
    ----------
    jsonl_path : Path
        Path to JSONL file containing compression configs
    model : ColBERT
        Model instance (needed for tokenizer.all_special_ids)
        
    Returns
    -------
    list[CompressionConfig | None]
        List of compression configs (None for baseline)
    """
    configs = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            # Check if this is a baseline config
            if (data.get("type") == "baseline" or 
                not data.get("strategies") or 
                len(data.get("strategies", [])) == 0):
                configs.append(None)
            else:
                config = CompressionConfig.from_dict(data)
                # Set ignore_token_ids for IDF pruning strategies if not already set
                for strategy in config.strategies:
                    if isinstance(strategy, IDFPruningStrategy) and strategy.config.ignore_token_ids is None:
                        strategy.config.ignore_token_ids = model.tokenizer.all_special_ids
                configs.append(config)
    return configs


def create_default_configs(model: ColBERT, dataset_name: str) -> list[CompressionConfig | None]:
    """
    Create default compression configurations matching beir_dataset.py.
    
    Parameters
    ----------
    model : ColBERT
        Model instance (needed for tokenizer.all_special_ids)
        
    Returns
    -------
    list[CompressionConfig | None]
        List of compression configs (None for baseline)
    """
    configs = []
    # configs.append(None)  # Baseline (no compression)

    ks = {
        "trec-covid" :[10, 20, 40, 80, 120],
        "fiqa" :[5, 10, 15, 25, 40, 60],
        "nfcorpus" : [10, 20, 40, 80, 120, 180],
    }.get(dataset_name, [10, 20, 40, 80])

    # attention score pruning configs
    # for lambda_mix in [0.15, 0.3, 0.45, 0.6, 0.75]:
    #     for k in ks:
    #         compactor_config = CompactorPruningConfig(
    #             top_k=k,
    #             protected_tokens=1,
    #             sketch_dim=None,
    #             attention_head_reduction="sum",
    #             leverage_head_reduction="sum",
    #             lambda_mix=lambda_mix,
    #         )
    #         strategy = CompactorPruningStrategy(compactor_config)
    #         config = CompressionConfig(
    #             strategies=[strategy],
    #             description=f"Compactor pruning k={k} lambda_mix={lambda_mix} head_reductions=sum",
    #         )
    #         configs.append(config)
    # for k in ks:
    #     attention_config = AttentionPruningConfig(
    #         top_k=k,
    #         protected_tokens=1,
    #         head_reduction="max",
    #         track_pruned_tokens=False,
    #     )
    #     strategy = AttentionPruningStrategy(attention_config)
    #     config = CompressionConfig(
    #         strategies=[strategy],
    #         description=f"Attention score pruning k={k} head_reduction=max",
    #     )
    #     configs.append(config)
    
    # # Global IDF pruning configs
    # for k in ks:
    #     pruning_config = IDFPruningConfig(
    #         mode="global",
    #         top_k=k,
    #         protected_tokens=1,
    #         ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
    #         use_tfidf=False,
    #         track_pruned_tokens=False,
    #     )
    #     strategy = IDFPruningStrategy(pruning_config)
    #     config = CompressionConfig(
    #         strategies=[strategy],
    #         description=f"Global IDF pruning k={k}",
    #     )
    #     configs.append(config)
    
    # # Document-wise IDF pruning configs
    # for k in ks:
    #     pruning_config = IDFPruningConfig(
    #         mode="document",
    #         top_k=k,
    #         protected_tokens=1,
    #         ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
    #         use_tfidf=False,
    #         track_pruned_tokens=False,
    #     )
    #     strategy = IDFPruningStrategy(pruning_config)
    #     config = CompressionConfig(
    #         strategies=[strategy],
    #         description=f"Doc-wise IDF pruning k={k}",
    #     )
    #     configs.append(config)
    
    # Pooling configs
    for method in ["window", "random"]: 
        for k in [2, 3, 4, 5]:
            for weight_by in [None]:
                pooling_config = PoolingConfig(pool_factor=k, protected_tokens=1, clustering_method=method, show_progress_bar=True, weight_by=weight_by)
                strategy = PoolingStrategy(pooling_config)
                config = CompressionConfig(strategies=[strategy], description=f"{method[0].upper() + method[1:]} Pooling f={k} protected tokens=1 weight_by={weight_by}")
                configs.append(config)
    
    return configs


def print_experiment_statistics(
    stats: dict[str, Any],
    configs: list[CompressionConfig | None],
) -> None:
    """
    Print experiment statistics.

    Parameters
    ----------
    stats : dict
        Statistics dictionary with keys: num_documents, encoding_time, config_token_counts, avg_tokens_per_doc
    configs : list
        List of compression configs
    """
    print("\n" + "=" * 80)
    print("EXPERIMENT STATISTICS")
    print("=" * 80)
    print(f"Documents encoded: {stats['num_documents']}")
    print(f"Encoding time: {stats['encoding_time']:.3f}s")
    if 'total_time' in stats:
        print(f"Total time: {stats['total_time']:.3f}s")

    # Create DataFrame for token counts
    data = []
    for i, config in enumerate(configs):
        config_name = config.description if config else "Baseline"
        data.append(
            {
                "Config": config_name,
                "Total Tokens": stats["config_token_counts"][i],
                "Avg Tokens/Doc": stats["avg_tokens_per_doc"][i],
            }
        )

    df = pd.DataFrame(data)
    print("\nToken counts per config:")
    print(df.to_string(index=False))
    print()


def evaluate_config(
    config_idx: int,
    config: CompressionConfig | None,
    documents_embeddings: list,
    documents: list[dict],
    queries: dict,
    qrels: dict,
    queries_embeddings: list,
    dataset_name: str,
    model_name: str,
    index_type: str,
    stats: dict[str, Any],
    metrics: list[str] | None = None,
    save_runfile: bool = False,
    runfile_output_dir: Path | None = None,
    run_id: str | None = None,
) -> dict[str, Any]:
    """
    Evaluate a single compression configuration.

    Parameters
    ----------
    config_idx : int
        Index of the configuration
    config : CompressionConfig | None
        Compression configuration (None for baseline)
    documents_embeddings : list
        Document embeddings for this config
    documents : list[dict]
        Original documents
    queries : dict
        Query dictionary
    qrels : dict
        Query relevance judgments
    queries_embeddings : list
        Encoded query embeddings
    dataset_name : str
        Dataset name for index naming
    model_name : str
        Model name for index naming
    index_type : str
        Type of index ("flat" or "plaid")
    stats : dict
        Experiment statistics
    metrics : list[str] | None, optional
        List of evaluation metrics to compute. If None, defaults to
        ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"]
    save_runfile : bool, optional
        Whether to save the runfile. Defaults to False.
    runfile_output_dir : Path | None, optional
        Directory to save runfiles. Required if save_runfile is True.
    run_id : str | None, optional
        Unique run ID for this experiment. Required if save_runfile is True.

    Returns
    -------
    dict
        Evaluation results dictionary
    """
    config_name = config.description if config else "Baseline"
    print(f"\n[{config_idx}] Evaluating: {config_name}")
    print("-" * 80)

    # Create a new index for this config
    config_index_name = (
        f"{dataset_name}_{model_name.split('/')[-1]}_config_{config_idx}"
    )
    match index_type:
        case "flat":
            config_index = indexes.Flat(
                override=True,
                index_name=config_index_name,
            )
        case "plaid":
            config_index = indexes.PLAID(
                override=True,
                index_name=config_index_name,
            )
        case _:
            raise ValueError(f"Invalid index type: {index_type}")

    # Add documents to index
    config_index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
    )

    # Retrieve
    retriever = retrieve.ColBERT(index=config_index)
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    # Evaluate
    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"]
    
    from ranx import Qrels, Run, evaluate as ranx_evaluate
    
    query_list = list(queries.keys())
    
    # Handle duplicate queries (like in beir.py)
    if len(query_list) > len(scores):
        from pylate.evaluation.beir import add_duplicates
        scores = add_duplicates(queries=query_list, scores=scores)
    
    # Create Qrels and Run objects
    qrels_obj = Qrels(qrels=qrels)
    
    run_dict = {
        query: {
            match["id"]: match["score"]
            for rank, match in enumerate(iterable=query_matches)
        }
        for query, query_matches in zip(query_list, scores)
    }
    
    run = Run(run=run_dict)
    
    # Evaluate
    evaluation_scores = ranx_evaluate(
        qrels=qrels_obj,
        run=run,
        metrics=metrics,
        make_comparable=True,
    )
    
    # Save runfile if requested
    runfile_path = None
    if save_runfile and runfile_output_dir is not None and run_id is not None:
        # Use parseable format: run-{run_id}.config-{config_idx}.json
        runfile_path = runfile_output_dir / f"run-{run_id}.config-{config_idx}.json"
        run.save(str(runfile_path), kind="json")
        print(f"Saved runfile to: {runfile_path}")

    # Store results
    result_entry = {
        "config_idx": config_idx,
        "config_name": config_name,
        "token_count": stats["config_token_counts"][config_idx],
        "avg_tokens_per_doc": stats["avg_tokens_per_doc"][config_idx],
        "evaluation": evaluation_scores,
        "runfile_path": str(runfile_path) if runfile_path else None,
    }

    print(f"Token count: {stats['config_token_counts'][config_idx]:,}")
    print(f"Avg tokens/doc: {stats['avg_tokens_per_doc'][config_idx]:.1f}")
    print("Evaluation scores:")
    for metric, value in evaluation_scores.items():
        print(f"  {metric}: {value:.4f}")

    return result_entry


def create_or_update_runfile_manifest(
    runfile_output_dir: Path,
    run_id: str,
    stats: dict[str, Any],
    all_evaluation_results: list[dict[str, Any]],
    configs: list[CompressionConfig | None],
    args: argparse.Namespace,
    dataset_name: str,
) -> None:
    """
    Create or update jsonl runfile manifest with full configuration and experiment metadata.
    
    Each line represents one configuration evaluation with full metadata. 
    Multiple experiments can append to the same manifest file without loading/parsing existing entries.
    
    Parameters
    ----------
    runfile_output_dir : Path
        Directory containing runfiles
    run_id : str
        Unique run ID for this experiment (e.g., timestamp-based)
    stats : dict
        Statistics dictionary
    all_evaluation_results : list[dict]
        List of evaluation result dictionaries
    configs : list[CompressionConfig | None]
        List of compression configs
    args : argparse.Namespace
        Command line arguments
    dataset_name : str
        Dataset name
    """
    manifest_path = runfile_output_dir / "manifest.jsonl"
    
    timestamp = datetime.now().isoformat()
    experiment_args = {
        "model_name": args.model_name,
        "dataset_name": dataset_name,
        "index_type": args.index_type,
        "batch_size": args.batch_size,
        "metrics": args.metrics,
        "configs_file": str(args.configs_file) if args.configs_file else None,
    }
    
    # Append each config entry as a new line (naive append)
    with open(manifest_path, "a") as f:  # 'a' mode for append
        for result in all_evaluation_results:
            config_idx = result["config_idx"]
            config = configs[config_idx]
            
            # Serialize config
            if config is None:
                serialized_config = {"type": "baseline", "description": "No compression"}
            else:
                serialized_config = config.serialize()
            
            # Get timing information for this config
            compression_time = None
            if "compression_times" in stats and config_idx < len(stats["compression_times"]):
                compression_time = stats["compression_times"][config_idx]
            
            # Each line is a complete JSON object with all metadata
            entry = {
                "run_id": run_id,
                "timestamp": timestamp,
                "experiment_args": experiment_args,
                "experiment_stats": {
                    "num_documents": stats.get("num_documents"),
                    "num_configs": stats.get("num_configs", len(configs)),
                    "encoding_time": stats.get("encoding_time"),
                    "total_time": stats.get("total_time"),
                },
                "config_idx": config_idx,
                "config_name": result["config_name"],
                "config": serialized_config,
                "runfile": f"run-{run_id}.config-{config_idx}.json",
                "runfile_path": result.get("runfile_path"),
                "token_count": result["token_count"],
                "avg_tokens_per_doc": result["avg_tokens_per_doc"],
                "compression_time": compression_time,
                "evaluation": result["evaluation"],
            }
            
            # Write as a single line (JSONL format)
            f.write(json.dumps(entry, default=str) + "\n")
    
    print(f"\nAppended {len(all_evaluation_results)} entries to manifest: {manifest_path}")
    print(f"Run ID: {run_id}")


def print_results_table(
    all_evaluation_results: list[dict[str, Any]], metrics: list[str] | None = None
) -> pd.DataFrame:
    """
    Print evaluation results summary table using pandas.

    Parameters
    ----------
    all_evaluation_results : list[dict]
        List of evaluation result dictionaries
    metrics : list[str] | None, optional
        List of evaluation metrics to include in the table. If None, defaults to
        ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"]
    """
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)

    # Determine which metrics to include in the table
    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"]
    
    # Prepare data for DataFrame
    data = []
    for result in all_evaluation_results:
        eval_scores = result["evaluation"]
        row = {
            "Config": result["config_name"],
            "Tokens": result["token_count"],
            "Avg Tokens/Doc": f"{result['avg_tokens_per_doc']:.1f}",
        }
        # Add all metrics dynamically
        for metric in metrics:
            row[metric] = eval_scores.get(metric, 0)
        data.append(row)

    df = pd.DataFrame(data)

    # Format numeric columns (all columns except Config, Tokens, Avg Tokens/Doc)
    numeric_cols = [col for col in df.columns if col not in ["Config", "Tokens", "Avg Tokens/Doc"]]
    for col in numeric_cols:
        df[col] = df[col].apply(lambda x: f"{x:.4f}")

    # Format tokens column
    df["Tokens"] = df["Tokens"].apply(lambda x: f"{x:,}")

    print(df.to_string())
    print()

    return df


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compression experiment evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    
    # Common arguments
    parser.add_argument(
        "--model_name",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Name of the model to use (default: 'lightonai/GTE-ModernColBERT-v1')",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'nfcorpus')",
    )
    
    # Mode selection
    parser.add_argument(
        "--mode",
        type=str,
        default="both",
        choices=["encode", "compress", "both"],
        help="Execution mode: 'encode' (only encode and save), 'compress' (only load and compress), 'both' (default: encode then compress)",
    )
    
    # Encoding group
    encoding_group = parser.add_argument_group(
        "Encoding options",
        "Options for encoding documents and queries (used in 'encode' and 'both' modes)"
    )
    encoding_group.add_argument(
        "--encoded_data_base_dir",
        type=str,
        default=None,
        help="Base directory for storing encoded data. Defaults to results/compression_experiments/<model>/<dataset>/embeddings",
    )
    encoding_group.add_argument(
        "--encode_id",
        type=str,
        default=None,
        help="ID for this encoding run. If not provided, uses timestamp-based ID. Required when loading in 'compress' mode.",
    )
    encoding_group.add_argument(
        "--batch_size",
        type=int,
        default=2400,
        help="Batch size for encoding (default: 2400)",
    )
    
    # Compression group
    compression_group = parser.add_argument_group(
        "Compression options",
        "Options for compression and evaluation (used in 'compress' and 'both' modes)"
    )
    compression_group.add_argument(
        "--index_type",
        type=str,
        default="plaid",
        help="Index type to use (default: 'plaid')",
        choices=["flat", "plaid"],
    )
    compression_group.add_argument(
        "--experiment_output_dir",
        type=str,
        default=None,
        help="Output directory for compression experiment results. Defaults to results/compression_experiments/<model>/<dataset>",
    )
    compression_group.add_argument(
        "--configs_file",
        type=str,
        default=None,
        help="Path to JSONL file containing compression configs. If not provided, uses default configs matching beir_dataset.py",
    )
    compression_group.add_argument(
        "--num_workers",
        type=int,
        default=8,
        help="Number of workers for parallel compression (default: 8)",
    )
    compression_group.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"],
        help="Evaluation metrics to compute (default: ['map', 'ndcg@10', 'ndcg@100', 'recall@10', 'recall@100', 'mrr@10'])",
    )
    compression_group.add_argument(
        "--save_runfiles",
        action="store_true",
        help="Save ranx runfiles for each configuration (default: False)",
    )
    
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    overall_start = time.time()

    # Determine which operations to perform
    do_encode = args.mode in ["encode", "both"]
    do_compress = args.mode in ["compress", "both"]
    
    if not do_encode and not do_compress:
        raise ValueError("Invalid mode: must be 'encode', 'compress', or 'both'")

    # Load model (needed for both encoding and compression)
    model: ColBERT = load_model(args.model_name, args.dataset_name)

    # ENCODING PHASE
    encoded_data_dir = None
    if do_encode:
        # Load dataset
        documents, queries, qrels = load_dataset(args.dataset_name)
        
        # Determine encoded data base directory
        # Default to results/compression_experiments/<model>/<dataset>/embeddings
        if args.encoded_data_base_dir is None:
            model_dir = sanitize_name(args.model_name)
            dataset_dir = sanitize_name(args.dataset_name)
            encoded_data_base_dir = (
                Path("results")
                / "compression_experiments"
                / model_dir
                / dataset_dir
                / "embeddings"
            )
        else:
            encoded_data_base_dir = Path(args.encoded_data_base_dir)
        
        # Determine encoded data directory
        # If encode_id not provided, generate one (will be used for compression phase too)
        encode_id = args.encode_id
        if encode_id is None:
            encode_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        encoded_data_dir = get_encoded_data_dir(
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            encoded_data_base_dir=encoded_data_base_dir,
            encode_id=encode_id,
        )
        
        # Load compression configs to determine artifact requirements
        print("\n" + "=" * 80)
        print("Loading compression configurations...")
        print("=" * 80)
        if args.configs_file:
            configs = load_configs_from_jsonl(Path(args.configs_file), model)
            print(f"✓ Loaded {len(configs)} configs from {args.configs_file}")
        else:
            configs = create_default_configs(model, args.dataset_name)
            print(f"✓ Created {len(configs)} default configs")
        
        # Determine artifact requirements
        compressors = [config.create_compressor() if config else None for config in configs]
        artifacts_with_args = []
        for compressor in compressors:
            if compressor and compressor.strategies:
                artifacts_with_args.append(compressor.strategies[0].get_artifact_requirements())
            else:
                artifacts_with_args.append({})
        
        artifact_flags = {}
        for artifact_with_arg in artifacts_with_args:
            for artifact, arg in artifact_with_arg.items():
                if artifact not in artifact_flags:
                    artifact_flags[artifact] = arg if arg else True
        
        print(f"Artifact requirements: {artifact_flags}")

        # Encode documents once with artifacts (input_ids needed for IDF pruning)
        print("\n" + "=" * 80)
        print("Encoding documents...")
        print("=" * 80)
        encoding_start = time.time()
        model_outputs = model.encode(
            sentences=[document["text"] for document in documents],
            batch_size=args.batch_size,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
            return_extra_artifacts=artifact_flags,
        )
        if len(model_outputs) == 2:
            documents_embeddings, artifacts = model_outputs
        else:
            documents_embeddings = model_outputs
            artifacts = {}

        encoding_time = time.time() - encoding_start
        print(f"✓ Encoded {len(documents_embeddings)} documents in {encoding_time:.3f}s")
        print(f"Artifacts collected: {list(artifacts.keys())}")

        # Encode queries once
        print("\n" + "=" * 80)
        print("Encoding queries...")
        print("=" * 80)
        query_encoding_start = time.time()
        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=args.batch_size,
            convert_to_tensor=True,
        )
        query_encoding_time = time.time() - query_encoding_start
        print(f"✓ Encoded {len(queries_embeddings)} queries in {query_encoding_time:.3f}s")
        
        # Save encoded data
        metadata = {
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
            "encode_id": encoded_data_dir.name,
            "timestamp": datetime.now().isoformat(),
            "num_documents": len(documents),
            "num_queries": len(queries),
            "encoding_time": encoding_time,
            "query_encoding_time": query_encoding_time,
            "document_length": model.document_length,
            "query_length": model.query_length,
            "artifacts_collected": list(artifacts.keys()),
        }
        
        save_encoded_data(
            encoded_data_dir=encoded_data_dir,
            documents_embeddings=documents_embeddings,
            queries_embeddings=queries_embeddings,
            artifacts=artifacts,
            metadata=metadata,
        )
        
        print(f"\n✓ Encoding complete. Encoded data saved to: {encoded_data_dir}")
        print(f"  Use --encode_id={encoded_data_dir.name} to load this encoding in compress mode")
        
        # If only encoding, exit here
        if not do_compress:
            return
    
    # COMPRESSION PHASE
    if do_compress:
        # Load encoded data
        # If we just encoded, use that directory; otherwise require encode_id
        if encoded_data_dir is None:
            if args.encode_id is None:
                raise ValueError("--encode_id is required when using 'compress' mode (without encoding first)")
            
            # Determine encoded data base directory (same logic as encoding phase)
            if args.encoded_data_base_dir is None:
                model_dir = sanitize_name(args.model_name)
                dataset_dir = sanitize_name(args.dataset_name)
                encoded_data_base_dir = (
                    Path("results")
                    / "compression_experiments"
                    / model_dir
                    / dataset_dir
                    / "embeddings"
                )
            else:
                encoded_data_base_dir = Path(args.encoded_data_base_dir)
            
            encoded_data_dir = get_encoded_data_dir(
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                encoded_data_base_dir=encoded_data_base_dir,
                encode_id=args.encode_id,
            )
        
        documents_embeddings, queries_embeddings, artifacts, metadata = load_encoded_data(encoded_data_dir, device="cuda")
        
        # Load dataset (needed for evaluation)
        documents, queries, qrels = load_dataset(args.dataset_name)
        
        # Verify metadata matches
        if metadata["model_name"] != args.model_name:
            print(f"Warning: Model name mismatch. Encoded: {metadata['model_name']}, Current: {args.model_name}")
        if metadata["dataset_name"] != args.dataset_name:
            print(f"Warning: Dataset name mismatch. Encoded: {metadata['dataset_name']}, Current: {args.dataset_name}")
        
        # Set up experiment output directory
        model_dir = sanitize_name(args.model_name)
        dataset_dir = sanitize_name(args.dataset_name)
        if args.experiment_output_dir is None:
            experiment_output_dir = (
                Path("results")
                / "compression_experiments"
                / model_dir
                / dataset_dir
            )
        else:
            experiment_output_dir = Path(args.experiment_output_dir)
        experiment_output_dir.mkdir(parents=True, exist_ok=True)

        # Generate run_id for consistent naming and tracking
        run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load compression configs
        print("\n" + "=" * 80)
        print("Loading compression configurations...")
        print("=" * 80)
        if args.configs_file:
            configs = load_configs_from_jsonl(Path(args.configs_file), model)
            print(f"✓ Loaded {len(configs)} configs from {args.configs_file}")
        else:
            configs = create_default_configs(model, args.dataset_name)
            print(f"✓ Created {len(configs)} default configs")
        
        print("\nConfigurations:")
        for i, config in enumerate(configs):
            if config is None:
                print(f"  [{i}] Baseline (no compression)")
            else:
                print(f"  [{i}] {config.description}")

        # Use encoding time from metadata if available
        encoding_time = metadata.get("encoding_time", 0.0)
        query_encoding_time = metadata.get("query_encoding_time", 0.0)

        # Track statistics
        stats = {
            "num_documents": len(documents),
            "encoding_time": encoding_time,
            "query_encoding_time": query_encoding_time,
            "config_token_counts": [],
            "avg_tokens_per_doc": [],
            "compression_times": [],
            "num_configs": len(configs),
        }

        # Evaluate each compression config
        print("\n" + "=" * 80)
        print("EVALUATING COMPRESSION CONFIGS")
        print("=" * 80)

        all_evaluation_results = []
        
        # Set up runfile output directory if saving runfiles
        runfile_output_dir = None
        if args.save_runfiles:
            runfile_output_dir = experiment_output_dir / "runfiles"
            runfile_output_dir.mkdir(parents=True, exist_ok=True)

        for config_idx, config in enumerate(configs):
            compression_start = time.time()
            
            # Apply compression if config is not None (baseline)
            if config is None:
                compressed_embeddings = documents_embeddings
            else:
                compressor = config.create_compressor()
                compressed_embeddings, _ = compressor.compress_parallel(
                    embeddings=documents_embeddings,
                    artifacts=artifacts,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    show_progress=True,
                )
            
            compression_time = time.time() - compression_start
            
            # Calculate token statistics
            num_tokens = sum(len(emb) for emb in compressed_embeddings)
            avg_tokens_per_doc = num_tokens / len(documents) if documents else 0
            
            stats["config_token_counts"].append(num_tokens)
            stats["avg_tokens_per_doc"].append(avg_tokens_per_doc)
            stats["compression_times"].append(compression_time)
            
            # Evaluate this config
            result = evaluate_config(
                config_idx=config_idx,
                config=config,
                documents_embeddings=compressed_embeddings,
                documents=documents,
                queries=queries,
                qrels=qrels,
                queries_embeddings=queries_embeddings,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                index_type=args.index_type,
                stats=stats,
                metrics=args.metrics,
                save_runfile=args.save_runfiles,
                runfile_output_dir=runfile_output_dir,
                run_id=run_id,
            )
            all_evaluation_results.append(result)

        stats["total_time"] = time.time() - overall_start

        # Print experiment statistics
        print_experiment_statistics(stats, configs)

        # Print summary table
        df = print_results_table(all_evaluation_results, metrics=args.metrics)

        # Save results to TSV
        df.to_csv(experiment_output_dir / f"results_{run_id}.tsv", index=False, sep="\t")
        save_results_jsonl(
            output_dir=experiment_output_dir,
            run_id=run_id,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            args=args,
            configs=configs,
            stats=stats,
            evaluation_results=all_evaluation_results,
        )
        
        # Create or update runfile manifest if saving runfiles
        if args.save_runfiles and runfile_output_dir is not None:
            create_or_update_runfile_manifest(
                runfile_output_dir=runfile_output_dir,
                run_id=run_id,
                stats=stats,
                all_evaluation_results=all_evaluation_results,
                configs=configs,
                args=args,
                dataset_name=args.dataset_name,
            )


if __name__ == "__main__":
    main()
