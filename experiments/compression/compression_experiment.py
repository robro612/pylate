"""Evaluation script for BEIR datasets with compression experiments."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import time
from tqdm.autonotebook import tqdm

import pandas as pd

from pylate import evaluation, indexes, models, retrieve
from pylate.models import ColBERT
from pylate.models.compression import (
    CompressionConfig,
    IDFPruningConfig,
    IDFPruningStrategy,
    IDFPoolingConfig,
    IDFPoolingStrategy,
    PoolingConfig,
    PoolingStrategy,
    AttentionPruningConfig,
    AttentionPruningStrategy,
    AttentionPoolingConfig,
    AttentionPoolingStrategy,
    LeverageScorePruningConfig,
    LeverageScorePruningStrategy,
    ImportancePruningConfig,
    ImportancePruningStrategy,
    ImportancePoolingConfig,
    ImportancePoolingStrategy,
    HybridPoolingConfig,
    HybridImportanceClusteringPoolingStrategy,
    RandomPruningConfig,
    RandomPruningStrategy,
    RandomPoolingConfig,
    RandomPoolingStrategy,
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


def load_model(model_name: str, dataset_name: str, document_length: int | None = None) -> ColBERT:
    """
    Load and initialize the ColBERT model.

    Parameters
    ----------
    model_name : str
        Name/path of the model to load
    document_length : int | None
        Maximum document length. If None, uses model's max length.
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

    # First load model to get its max_length if document_length not specified
    if document_length is None:
        # Load tokenizer to get max_length
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
        max_len = getattr(tokenizer, 'model_max_length', 8192)
        # Cap at reasonable limit (some models report very large values like 1e30)
        document_length = min(max_len, 8192)
        print(f"  Using model's max length: {document_length}")

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=document_length,
        query_length=QUERY_LEN.get(dataset_name, 32),
        trust_remote_code=True,
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
        Name of the dataset to load. Can be:
        - A BEIR dataset name (e.g., "nfcorpus", "scifact")
        - A path to a custom dataset directory (e.g., "amazon_dataset/beir_format")

    Returns
    -------
    tuple
        (documents, queries, qrels)
    """
    print("\n" + "=" * 80)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 80)

    # Check if dataset_name is a path to a local directory
    # Resolve relative paths and check for corpus.jsonl to confirm it's a valid BEIR dataset
    dataset_path = Path(dataset_name).resolve()
    is_local_dataset = (
        dataset_path.exists()
        and dataset_path.is_dir()
        and (dataset_path / "corpus.jsonl").exists()
    )

    if is_local_dataset:
        # Load custom dataset from local directory
        print(f"Loading custom dataset from: {dataset_path}")
        documents, queries, qrels = evaluation.load_custom_dataset(
            str(dataset_path),
            split="test",
        )
    elif "cqadupstack" in dataset_name:
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
        # Load BEIR dataset
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


def create_default_configs(model: ColBERT, kmeans_gpu: bool = False) -> list[CompressionConfig | None]:
    """
    Create default compression configurations matching beir_dataset.py.

    Parameters
    ----------
    model : ColBERT
        Model instance (needed for tokenizer.all_special_ids)
    kmeans_gpu : bool
        Enable GPU for fastkmeans in spherical pooling (experimental)

    Returns
    -------
    list[CompressionConfig | None]
        List of compression configs (None for baseline)
    """
    configs = [None]  # Baseline (no compression)
    # configs = []  # exclude Baseline (no compression)


    # # # Importance-based pruning configs (keep_ratio approach)
    # for keep_ratio in [0.1, 0.2, 0.33, 0.5,]:
    #     importance_config = ImportancePruningConfig(
    #         keep_ratio=keep_ratio,
    #         protected_tokens=1,
    #         min_tokens=8,
    #         use_norm=True,
    #         use_idf=False,  # Set to True if IDF artifacts are available
    #         use_token_weights=False,  # Set to True if token_weights artifacts are available
    #         norm_weight=1.0,
    #     )
    #     strategy = ImportancePruningStrategy(importance_config)
    #     config = CompressionConfig(
    #         strategies=[strategy],
    #         description=f"Importance pruning keep_ratio={keep_ratio}",
    #     )
    #     configs.append(config)

    #     # Importance-based pooling configs (keep_ratio approach)
    #     pooling_config = ImportancePoolingConfig(
    #         keep_ratio=keep_ratio,
    #         protected_tokens=1,
    #         min_tokens=8,
    #         use_norm=True,
    #         use_idf=False,  # Set to True if IDF artifacts are available
    #         use_token_weights=False,  # Set to True if token_weights artifacts are available
    #         norm_weight=1.0,
    #     )
    #     strategy = ImportancePoolingStrategy(pooling_config)
    #     config = CompressionConfig(
    #         strategies=[strategy],
    #         description=f"Importance pooling keep_ratio={keep_ratio}",
    #     )
    #     configs.append(config)

    # # Hybrid importance + clustering pooling (anchor-aware pooling within clusters)
    # for pool_factor, keep_ratio in [
    #     # (2, 0.5),
    #     # (3, 0.33),
    #     # (4, 0.25),
    #     # (5, 0.2),
    #     (2, 0.5*0.8),
    #     (3, 0.33*0.8),
    #     (5, 0.2*0.8),
    #     (10, 0.1*0.8),
    # ]:
    #     hybrid_config = HybridPoolingConfig(
    #         pool_factor=pool_factor,
    #         keep_ratio=keep_ratio,
    #         protected_tokens=1,
    #         min_tokens=8,
    #         clustering_method="hierarchical",
    #         show_progress_bar=True,
    #         use_norm=True,
    #         use_idf=False,
    #         use_token_weights=False,
    #         norm_weight=1.0,
    #         idf_weight=1.0,
    #         token_weights_weight=1.0,
    #     )
    #     hybrid_strategy = HybridImportanceClusteringPoolingStrategy(hybrid_config)
    #     configs.append(
    #         CompressionConfig(
    #             strategies=[hybrid_strategy],
    #             description=f"Hybrid importance+clustering pooling pf={pool_factor} kr={keep_ratio}",
    #         )
    #     )

    # Random pruning baseline
    for keep_ratio in [0.1, 0.2, 0.33, 0.5, ]:
        rand_prune_cfg = RandomPruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        rand_prune_strategy = RandomPruningStrategy(rand_prune_cfg)
        configs.append(
            CompressionConfig(
                strategies=[rand_prune_strategy],
                description=f"Random pruning keep_ratio={keep_ratio}",
            )
        )

        # Random pooling baseline
        rand_pool_cfg = RandomPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        rand_pool_strategy = RandomPoolingStrategy(rand_pool_cfg)
        configs.append(
            CompressionConfig(
                strategies=[rand_pool_strategy],
                description=f"Random pooling keep_ratio={keep_ratio}",
            )
        )


        # attention score pruning configs
        attention_config = AttentionPruningConfig(
            # top_k=k,
            keep_ratio=keep_ratio,
            protected_tokens=1,
            track_pruned_tokens=False,
        )
        strategy = AttentionPruningStrategy(attention_config)
        config = CompressionConfig(
            strategies=[strategy],
            # description=f"Attention score pruning k={k}",
            description=f"Attention score pruning keep_ratio={keep_ratio}",
        )
        configs.append(config)

        # attention score pooling configs
        attention_pool_config = AttentionPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            show_progress_bar=True,
        )
        attention_pool_strategy = AttentionPoolingStrategy(attention_pool_config)
        attention_pool_compression_config = CompressionConfig(
            strategies=[attention_pool_strategy],
            description=f"Attention score pooling keep_ratio={keep_ratio}",
        )
        configs.append(attention_pool_compression_config)

        leverage_config = LeverageScorePruningConfig(
            # top_k=k,
            keep_ratio=keep_ratio,
            protected_tokens=1,
            track_pruned_tokens=False,
        )
        strategy = LeverageScorePruningStrategy(leverage_config)
        config = CompressionConfig(
            strategies=[strategy],
            # description=f"Leverage score pruning k={k}",
            description=f"Leverage score pruning keep_ratio={keep_ratio}",
        )
        configs.append(config)
    
        # # Global IDF pruning configs
        # pruning_config = IDFPruningConfig(
        #     mode="global",
        #     # top_k=k,
        #     keep_ratio=keep_ratio,
        #     protected_tokens=1,
        #     ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
        #     use_tfidf=False,
        #     track_pruned_tokens=False,
        # )
        # strategy = IDFPruningStrategy(pruning_config)
        # config = CompressionConfig(
        #     strategies=[strategy],
        #     # description=f"Global IDF pruning k={k}",
        #     description=f"Global IDF pruning keep_ratio={keep_ratio}",
        # )
        # configs.append(config)
    
        # Document-wise IDF pruning configs
        pruning_config = IDFPruningConfig(
            mode="document",
            # top_k=k,
            keep_ratio=keep_ratio,
            protected_tokens=1,
            ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
            use_tfidf=False,
            track_pruned_tokens=False,
        )
        strategy = IDFPruningStrategy(pruning_config)
        config = CompressionConfig(
            strategies=[strategy],
            # description=f"Doc-wise IDF pruning k={k}",
            description=f"Doc-wise IDF pruning keep_ratio={keep_ratio}",
        )
        configs.append(config)

        # IDF Pooling configs (document mode)
        idf_pooling_config = IDFPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            use_tfidf=False,
            ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
            show_progress_bar=False,
        )
        strategy = IDFPoolingStrategy(idf_pooling_config)
        config = CompressionConfig(
            strategies=[strategy],
            description=f"IDF Pooling keep_ratio={keep_ratio}",
        )
        configs.append(config)

    # Pooling configs
    for method in ["spherical", "hierarchical"]:
        for k in [2, 3, 5, 10]:
            pooling_config = PoolingConfig(
                pool_factor=k,
                protected_tokens=1,
                clustering_method=method,
                show_progress_bar=True,
                kmeans_gpu=kmeans_gpu if method == "spherical" else False,
            )
            strategy = PoolingStrategy(pooling_config)
            config = CompressionConfig(
                strategies=[strategy],
                description=f"{method[0].upper() + method[1:]} Pooling f={k} protected tokens=1",
            )
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
    save_retrieval_results: bool = False,
    retrieval_results_output_dir: Path | None = None,
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
        ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "precision@10"]
    save_runfile : bool, optional
        Whether to save the ranx runfile (for evaluation). Defaults to False.
    runfile_output_dir : Path | None, optional
        Directory to save runfiles. Required if save_runfile is True.
    run_id : str | None, optional
        Unique run ID for this experiment. Required if save_runfile is True.
    save_retrieval_results : bool, optional
        Whether to save the raw retrieval results (all scores). Defaults to False.
    retrieval_results_output_dir : Path | None, optional
        Directory to save retrieval results. Required if save_retrieval_results is True.

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
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "precision@10"]
    
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

    # Save retrieval results if requested
    retrieval_results_path = None
    if save_retrieval_results and retrieval_results_output_dir is not None and run_id is not None:
        # Save the raw retrieval scores (before evaluation)
        retrieval_results_path = retrieval_results_output_dir / f"retrieval-{run_id}.config-{config_idx}.json"

        # Create a structured format with query IDs and their retrieved documents
        retrieval_data = {
            "run_id": run_id,
            "config_idx": config_idx,
            "config_name": config_name,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "num_queries": len(query_list),
            "k": 20,  # Number of retrieved documents per query
            "results": {
                query_id: query_scores
                for query_id, query_scores in zip(query_list, scores)
            }
        }

        with open(retrieval_results_path, "w") as f:
            json.dump(retrieval_data, f, indent=2)

        print(f"Saved retrieval results to: {retrieval_results_path}")

    # Store results
    result_entry = {
        "config_idx": config_idx,
        "config_name": config_name,
        "token_count": stats["config_token_counts"][config_idx],
        "avg_tokens_per_doc": stats["avg_tokens_per_doc"][config_idx],
        "evaluation": evaluation_scores,
        "runfile_path": str(runfile_path) if runfile_path else None,
        "retrieval_results_path": str(retrieval_results_path) if retrieval_results_path else None,
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
    parser = argparse.ArgumentParser(description="Compression experiment evaluation")
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
    parser.add_argument(
        "--index_type",
        type=str,
        default="plaid",
        help="Index type to use (default: 'plaid')",
        choices=["flat", "plaid"],
    )
    parser.add_argument(
        "--experiment_output_dir",
        type=str,
        default=None,
        help="Output directory for compression experiment results. Defaults to results/compression_experiments/<model>/<dataset>",
    )
    parser.add_argument(
        "--configs_file",
        type=str,
        default=None,
        help="Path to JSONL file containing compression configs. If not provided, uses default configs matching beir_dataset.py",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1000,
        help="Batch size for encoding (default: 1000)",
    )
    parser.add_argument(
        "--metrics",
        type=str,
        nargs="+",
        default=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "precision@10"],
        help="Evaluation metrics to compute (default: ['map', 'ndcg@10', 'ndcg@100', 'recall@10', 'recall@100', 'mrr@10', 'precision@10'])",
    )
    parser.add_argument(
        "--save_runfiles",
        action="store_true",
        help="Save ranx runfiles for each configuration (default: False)",
    )
    parser.add_argument(
        "--save_retrieval_results",
        action="store_true",
        help="Save raw retrieval results (all query-document scores) for each configuration (default: False)",
    )
    parser.add_argument(
        "--kmeans_gpu",
        action="store_true",
        help="Enable GPU for fastkmeans in spherical pooling (experimental, will fallback to CPU on error)",
    )
    parser.add_argument(
        "--skip",
        type=int,
        default=0,
        help="Number of configs to skip (for resuming experiments). Skips configs 0 to skip-1.",
    )
    parser.add_argument(
        "--append_to",
        type=str,
        default=None,
        help="Path to existing results JSONL file to append to (for resuming experiments).",
    )
    parser.add_argument(
        "--document_length",
        type=int,
        default=None,
        help="Maximum document length in tokens. If not specified, uses model's max length (capped at 8192).",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    overall_start = time.time()

    # Load model
    model: ColBERT = load_model(args.model_name, args.dataset_name, args.document_length)

    # Load dataset
    documents, queries, qrels = load_dataset(args.dataset_name)

    # Set up experiment output directory and results file path
    if args.append_to:
        # Resume mode: append to existing file
        results_jsonl_path = Path(args.append_to).resolve()
        if not results_jsonl_path.exists():
            print(f"Error: --append_to file not found: {results_jsonl_path}")
            sys.exit(1)
        experiment_output_dir = results_jsonl_path.parent
        # Extract run_id from existing filename (e.g., results_20251221_194501.jsonl)
        run_id = results_jsonl_path.stem.replace("results_", "")
        print(f"\n✓ Appending to existing results file: {results_jsonl_path}")
    else:
        # Normal mode: create new experiment
        model_dir = sanitize_name(args.model_name.split("/")[-1])
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
        results_jsonl_path = experiment_output_dir / f"results_{run_id}.jsonl"

    # Load compression configs
    print("\n" + "=" * 80)
    print("Loading compression configurations...")
    print("=" * 80)
    if args.configs_file:
        configs = load_configs_from_jsonl(Path(args.configs_file), model)
        print(f"✓ Loaded {len(configs)} configs from {args.configs_file}")
    else:
        configs = create_default_configs(model, kmeans_gpu=args.kmeans_gpu)
        print(f"✓ Created {len(configs)} default configs")
        if args.kmeans_gpu:
            print("  (GPU enabled for spherical pooling kmeans)")
    
    print("\nConfigurations:")
    for i, config in enumerate(configs):
        if config is None:
            print(f"  [{i}] Baseline (no compression)")
        else:
            print(f"  [{i}] {config.description}")

    # Encode documents once with artifacts (input_ids needed for IDF pruning)
    # Use normalize_embeddings=False to get unnormalized embeddings for importance scoring
    print("\n" + "=" * 80)
    print("Encoding documents (unnormalized for importance scoring)...")
    print("=" * 80)
    encoding_start = time.time()
    documents_embeddings, artifacts = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
        convert_to_tensor=True,
        normalize_embeddings=False,  # Keep unnormalized for importance scoring
        return_extra_artifacts={"input_ids": True, "attention_scores": True},
    )
    encoding_time = time.time() - encoding_start
    print(f"✓ Encoded {len(documents_embeddings)} documents in {encoding_time:.3f}s")
    print(f"   Embeddings are UNNORMALIZED (for importance-based compression)")

    # Encode queries once
    print("\n" + "=" * 80)
    print("Encoding queries...")
    print("=" * 80)
    query_encoding_start = time.time()
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=512,
        convert_to_tensor=True,
    )
    query_encoding_time = time.time() - query_encoding_start

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

    # Write initial metadata only if not appending to existing file
    if not args.append_to:
        metadata_entry = {
            "type": "metadata",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": args.model_name,
            "dataset_name": args.dataset_name,
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
                "total_time": None,  # filled in after all configs
            },
            "configs": [serialize_config_for_storage(config) for config in configs],
        }
        with open(results_jsonl_path, "w") as f:
            f.write(json.dumps(metadata_entry, default=str) + "\n")

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

    # Set up retrieval results output directory if saving retrieval results
    retrieval_results_output_dir = None
    if args.save_retrieval_results:
        retrieval_results_output_dir = experiment_output_dir / "retrieval_results"
        retrieval_results_output_dir.mkdir(parents=True, exist_ok=True)

    for config_idx, config in enumerate(configs):
        # Skip configs if --skip is specified (for resuming experiments)
        if config_idx < args.skip:
            config_name = "Baseline (no compression)" if config is None else config.description
            print(f"\n[{config_idx}] Skipping: {config_name}")
            # Add placeholder stats for skipped configs
            stats["config_token_counts"].append(0)
            stats["avg_tokens_per_doc"].append(0)
            stats["compression_times"].append(0)
            continue

        compression_start = time.time()

        # Apply compression if config is not None (baseline)
        if config is None:
            # Baseline: normalize the unnormalized embeddings
            import torch.nn.functional as F
            compressed_embeddings = [
                F.normalize(emb, p=2, dim=-1) for emb in documents_embeddings
            ]
        else:
            compressor = config.create_compressor()

            # Compress with unnormalized embeddings (for importance scoring)
            compressed_embeddings, _ = compressor.compress_parallel(
                embeddings=documents_embeddings,
                artifacts=artifacts,
                batch_size=args.batch_size,
                num_workers=8,
                show_progress=True,
            )

            # Normalize embeddings AFTER compression
            import torch.nn.functional as F
            compressed_embeddings = [
                F.normalize(emb, p=2, dim=-1) for emb in compressed_embeddings
            ]

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
            save_retrieval_results=args.save_retrieval_results,
            retrieval_results_output_dir=retrieval_results_output_dir,
        )
        all_evaluation_results.append(result)
        # Stream the result to disk immediately
        with open(results_jsonl_path, "a") as f:
            f.write(json.dumps(
                {
                    "type": "result",
                    "run_id": run_id,
                    "config_idx": result["config_idx"],
                    "config_name": result["config_name"],
                    "config": serialize_config_for_storage(configs[result["config_idx"]]),
                    "token_count": result["token_count"],
                    "avg_tokens_per_doc": result["avg_tokens_per_doc"],
                    "compression_time": stats["compression_times"][config_idx],
                    "metrics": result["evaluation"],
                    "runfile_path": result.get("runfile_path"),
                },
                default=str,
            ) + "\n")

    stats["total_time"] = time.time() - overall_start

    # Print experiment statistics
    print_experiment_statistics(stats, configs)

    # Print summary table
    df = print_results_table(all_evaluation_results, metrics=args.metrics)

    # Save results to TSV
    df.to_csv(experiment_output_dir / f"results_{run_id}.tsv", index=False, sep="\t")
    # Rewrite JSONL with final metadata and all results for consistency
    final_metadata = metadata_entry.copy()
    final_metadata["timing"]["total_time"] = stats["total_time"]
    final_metadata["timing"]["encoding_time"] = stats.get("encoding_time")
    final_metadata["timing"]["query_encoding_time"] = stats.get("query_encoding_time")
    with open(results_jsonl_path, "w") as f:
        f.write(json.dumps(final_metadata, default=str) + "\n")
        for result in all_evaluation_results:
            f.write(json.dumps(
                {
                    "type": "result",
                    "run_id": run_id,
                    "config_idx": result["config_idx"],
                    "config_name": result["config_name"],
                    "config": serialize_config_for_storage(configs[result["config_idx"]]),
                    "token_count": result["token_count"],
                    "avg_tokens_per_doc": result["avg_tokens_per_doc"],
                    "compression_time": stats["compression_times"][result["config_idx"]],
                    "metrics": result["evaluation"],
                    "runfile_path": result.get("runfile_path"),
                },
                default=str,
            ) + "\n")
    
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
