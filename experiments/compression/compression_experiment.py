"""Evaluation script for BEIR datasets with compression experiments."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any
from tqdm.autonotebook import tqdm

import pandas as pd

from pylate import evaluation, indexes, models, retrieve
from pylate.models import (
    ColBERT,
    CompressionConfig,
    CompressionContext,
    CompressionExperimentConfig,
    CompressionExperimentResults,
    IDFPruningConfig,
    PoolingConfig,
)
from pylate.models.utils import TokenTFIDFStats

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


def load_dataset(dataset_name: str) -> tuple[list[dict], dict, dict, str]:
    """
    Load dataset (documents, queries, qrels).

    Parameters
    ----------
    dataset_name : str
        Name of the dataset to load

    Returns
    -------
    tuple
        (documents, queries, qrels, normalized_dataset_name)
    """
    print("\n" + "=" * 80)
    print(f"Loading dataset: {dataset_name}")
    print("=" * 80)

    normalized_name = dataset_name

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
        normalized_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    print(f"✓ Loaded dataset: {normalized_name}")
    print(f"  Documents: {len(documents)}")
    print(f"  Queries: {len(queries)}")

    return documents, queries, qrels, normalized_name


def create_experiment_config(
    idf_stats: TokenTFIDFStats,
    idf_document_top_k: list[int],
    idf_global_top_k: list[int],
    pool_factors: list[int],
    output_dir: Path,
    storage_mode: str,
    run_id: str,
) -> CompressionExperimentConfig:
    """
    Create compression experiment configuration.

    Parameters
    ----------
    idf_stats : TokenTFIDFStats
        IDF statistics for pruning
    idf_document_top_k : list[int]
        Top-k values for per-document IDF pruning
    idf_global_top_k : list[int]
        Top-k values for global IDF pruning
    pool_factors : list[int]
        Pool factors for pooling compression
    output_dir : Path
        Output directory for experiment results
    storage_mode : str
        Storage mode ("cpu", "memory", or "disk")
    run_id : str
        Run ID for consistent file naming and tracking. Shard files use format:
        run-{run_id}.config-{config_idx}.pt (matching runfile format).
        The run_id is saved in experiment_metadata.json for tracking and matching shards to configs.

    Returns
    -------
    CompressionExperimentConfig
        Configured experiment config
    """
    # Create compression experiment configs
    configs = [
        CompressionConfig(description="Baseline (no compression)")
    ]  # Baseline (no compression)

    # Add per-document IDF pruning configs
    for k in idf_document_top_k:
        configs.append(
            CompressionConfig(
                description=f"Doc-wise IDF pruning k={k}",
                pruning=[IDFPruningConfig(mode="document", top_k=k, stats=idf_stats)],
            )
        )

    # Add global IDF pruning configs
    for k in idf_global_top_k:
        configs.append(
            CompressionConfig(
                description=f"Global IDF pruning k={k}",
                pruning=[IDFPruningConfig(mode="global", top_k=k, stats=idf_stats)],
            )
        )

    # Add pooling configs
    for pool_factor in pool_factors:
        configs.append(
            CompressionConfig(
                description=f"Hierarchical Pooling f={pool_factor} protected tokens=1",
                pooling=[
                    PoolingConfig(
                        pool_factor=pool_factor,
                        protected_tokens=1,
                        clustering_method="hierarchical",
                    )
                ],
            )
        )

    # Create compression experiment config
    experiment_config = CompressionExperimentConfig(
        configs=configs,
        output_dir=output_dir,
        storage_mode=storage_mode,
        overwrite=True,
        run_id=run_id,
    )

    print("\n" + "=" * 80)
    print(f"Created compression experiment with {len(configs)} configs")
    print(f"Storage mode: {storage_mode}")
    print(f"Output directory: {output_dir}")
    print("=" * 80)
    print("\nConfigurations:")
    for i, config in enumerate(configs):
        if config is None:
            print(f"  [{i}] Baseline (no compression)")
        else:
            print(f"  [{i}] {config.description}")

    return experiment_config


def print_experiment_statistics(
    experiment_results: CompressionExperimentResults,
    configs: list[CompressionConfig | None],
) -> None:
    """
    Print experiment statistics.

    Parameters
    ----------
    experiment_results : CompressionExperimentResults
        Results from compression experiment
    configs : list
        List of compression configs
    """
    stats = experiment_results.statistics

    print("\n" + "=" * 80)
    print("EXPERIMENT STATISTICS")
    print("=" * 80)
    print(f"Documents encoded: {stats['num_documents']}")
    print(f"Encoding time: {stats['encoding_time']:.3f}s")
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
    experiment_results: CompressionExperimentResults,
    all_evaluation_results: list[dict[str, Any]],
    experiment_config: CompressionExperimentConfig,
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
    experiment_results : CompressionExperimentResults
        Results from compression experiment
    all_evaluation_results : list[dict]
        List of evaluation result dictionaries
    experiment_config : CompressionExperimentConfig
        The experiment configuration used
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
        "storage_mode": args.storage_mode,
        "batch_size": args.batch_size,
        "metrics": args.metrics,
        "idf_document_top_k": args.idf_document_top_k,
        "idf_global_top_k": args.idf_global_top_k,
        "pool_factors": args.pool_factors,
    }
    
    # Get statistics from experiment results
    stats = experiment_results.statistics
    
    # Append each config entry as a new line (naive append)
    with open(manifest_path, "a") as f:  # 'a' mode for append
        for result in all_evaluation_results:
            config_idx = result["config_idx"]
            config = experiment_config.configs[config_idx]
            
            # Serialize config
            if config is None:
                serialized_config = {"type": "none", "description": "No compression"}
            elif isinstance(config, CompressionContext):
                serialized_config = config.serialize()
            elif isinstance(config, CompressionConfig):
                serialized_config = config.serialize()
            else:
                serialized_config = {"type": "unknown", "description": str(config)}
            
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
                    "num_configs": stats.get("num_configs"),
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
        "--storage_mode",
        type=str,
        default="disk",
        choices=["cpu", "memory", "disk"],
        help="Storage mode for experiment results (default: 'disk')",
    )
    parser.add_argument(
        "--experiment_output_dir",
        type=str,
        default="results/compression_experiments",
        help="Output directory for compression experiment results (default: 'results/compression_experiments')",
    )
    parser.add_argument(
        "--clean_output_dir",
        action="store_true",
        help="Clean output directory of shard files after experiment",
    )
    parser.add_argument(
        "--idf_document_top_k",
        type=int,
        nargs="+",
        default=[],
        help="Top-k values for per-document IDF pruning (default: [])",
    )
    parser.add_argument(
        "--idf_global_top_k",
        type=int,
        nargs="+",
        default=[],
        help="Top-k values for global IDF pruning (default: [])",
    )
    parser.add_argument(
        "--pool_factors",
        type=int,
        nargs="+",
        default=[],
        help="Pool factors for pooling compression (default: [])",
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
        default=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10"],
        help="Evaluation metrics to compute (default: ['map', 'ndcg@10', 'ndcg@100', 'recall@10', 'recall@100', 'mrr@10'])",
    )
    parser.add_argument(
        "--save_runfiles",
        action="store_true",
        help="Save ranx runfiles for each configuration (default: False)",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()

    # Load model
    model = load_model(args.model_name, args.dataset_name)

    # Load dataset
    documents, queries, qrels, dataset_name = load_dataset(args.dataset_name)

    # Collect IDF statistics
    print("\n" + "=" * 80)
    print("Collecting IDF statistics from corpus...")
    print("=" * 80)
    document_texts = [doc["text"] for doc in documents]
    idf_stats = TokenTFIDFStats.from_colbert_model(
        model, document_texts, show_progress=True
    )
    print(f"✓ Collected stats for {len(idf_stats.idf_scores)} unique tokens")

    # Set up experiment output directory
    if args.experiment_output_dir is None:
        experiment_output_dir = Path(
            f"./compression_experiments/{dataset_name}_{args.model_name.split('/')[-1]}"
        )
    else:
        experiment_output_dir = Path(args.experiment_output_dir)

    # Generate run_id for consistent naming and tracking (saved in experiment_metadata.json)
    # This allows matching shard files to configs even without runfiles
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create experiment config
    experiment_config = create_experiment_config(
        idf_stats=idf_stats,
        idf_document_top_k=args.idf_document_top_k,
        idf_global_top_k=args.idf_global_top_k,
        pool_factors=args.pool_factors,
        output_dir=experiment_output_dir,
        storage_mode=args.storage_mode,
        run_id=run_id,
    )

    # Run compression experiment
    print("\n" + "=" * 80)
    print("Running compression experiment...")
    print("=" * 80)
    experiment_results = model.encode(
        sentences=[document["text"] for document in documents],
        batch_size=args.batch_size,
        is_query=False,
        show_progress_bar=True,
        compression_config=experiment_config,
    )

    # Print experiment statistics
    print_experiment_statistics(experiment_results, experiment_config.configs)

    # Encode queries once
    print("\n" + "=" * 80)
    print("Encoding queries...")
    print("=" * 80)
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=512,
    )

    # Evaluate each compression config
    print("\n" + "=" * 80)
    print("EVALUATING COMPRESSION CONFIGS")
    print("=" * 80)

    all_evaluation_results = []
    stats = experiment_results.statistics
    
    # Set up runfile output directory if saving runfiles
    runfile_output_dir = None
    if args.save_runfiles:
        runfile_output_dir = experiment_output_dir / "runfiles"
        runfile_output_dir.mkdir(parents=True, exist_ok=True)
        # Use the same run_id that was passed to experiment_config
        run_id = experiment_config.run_id  # Already set, but explicit for clarity

    for config_idx, (config, documents_embeddings) in enumerate(experiment_results):
        result = evaluate_config(
            config_idx=config_idx,
            config=config,
            documents_embeddings=documents_embeddings,
            documents=documents,
            queries=queries,
            qrels=qrels,
            queries_embeddings=queries_embeddings,
            dataset_name=dataset_name,
            model_name=args.model_name,
            index_type=args.index_type,
            stats=stats,
            metrics=args.metrics,
            save_runfile=args.save_runfiles,
            runfile_output_dir=runfile_output_dir,
            run_id=run_id,
        )
        all_evaluation_results.append(result)

    # Print summary table
    df = print_results_table(all_evaluation_results, metrics=args.metrics)

    # Save results to TSV
    df.to_csv(experiment_output_dir / f"results_{run_id}.tsv", index=False, sep="\t")
    
    # Create or update runfile manifest if saving runfiles
    if args.save_runfiles and runfile_output_dir is not None and run_id is not None:
        create_or_update_runfile_manifest(
            runfile_output_dir=runfile_output_dir,
            run_id=run_id,
            experiment_results=experiment_results,
            all_evaluation_results=all_evaluation_results,
            experiment_config=experiment_config,
            args=args,
            dataset_name=dataset_name,
        )
    
    if args.clean_output_dir:
        print("\n" + "=" * 80)
        print("Cleaning output directory of .pt files...")
        print("=" * 80)
        # remove all .pt files in the output directory
        bar = tqdm(experiment_output_dir.glob("*.pt"), desc="Cleaning output directory", leave=False)   
        for file in bar:
            bar.set_description(f"Cleaning {file.name}", refresh=True)
            file.unlink()


if __name__ == "__main__":
    main()
