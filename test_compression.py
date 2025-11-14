#!/usr/bin/env python3
"""
Test script for compression integration in ColBERT using BEIR nfcorpus dataset.
"""

import random
import json
from pathlib import Path
from typing import Literal, Optional
from rich.console import Console
from rich.table import Table
from pylate import evaluation, indexes, models, retrieve
from pylate.models import CompressionConfig, IDFPruningConfig, PoolingConfig
from pylate.models.utils import TokenTFIDFStats
from pylate.models.compression import CompressionContext

console = Console(width=200)


def serialize_compression_config(config: Optional[CompressionConfig]) -> dict:
    """
    Serialize a CompressionConfig to a JSON-compatible dictionary.

    Parameters
    ----------
    config : CompressionConfig or None
        The compression config to serialize

    Returns
    -------
    dict
        JSON-serializable dictionary representation
    """
    if config is None:
        return {"type": "none", "description": "No compression"}

    result = {
        "description": config.description,
        "pruning": [],
        "pooling": [],
    }

    # Serialize pruning configs
    for pruning_cfg in config.pruning:
        pruning_dict = {
            "type": "idf",
            "mode": pruning_cfg.mode,
            "top_k": pruning_cfg.top_k,
            "threshold": pruning_cfg.threshold,
            "apply_to_queries": pruning_cfg.apply_to_queries,
            "protected_tokens": pruning_cfg.protected_tokens,
            "use_tfidf": pruning_cfg.use_tfidf,
            "track_pruned_tokens": pruning_cfg.track_pruned_tokens,
            "ignore_tokens_count": len(pruning_cfg.ignore_tokens)
            if pruning_cfg.ignore_tokens
            else 0,
        }
        result["pruning"].append(pruning_dict)

    # Serialize pooling configs
    for pooling_cfg in config.pooling:
        pooling_dict = {
            "pool_factor": pooling_cfg.pool_factor,
            "protected_tokens": pooling_cfg.protected_tokens,
            "clustering_method": pooling_cfg.clustering_method,
        }
        result["pooling"].append(pooling_dict)

    return result


def collect_idf_stats(model, documents):
    """
    Collect IDF statistics by tokenizing documents and using TokenTFIDFStats.fit().
    """
    print("\nCollecting IDF statistics from corpus...")

    # Tokenize all documents
    tokenized_docs = []
    for doc in documents:
        tokens = model.tokenizer.encode(
            doc["text"],
            add_special_tokens=True,
            truncation=True,
            max_length=model.document_length,
        )
        tokenized_docs.append(tokens)

    # Use TokenTFIDFStats.fit() to compute statistics
    stats = TokenTFIDFStats()
    stats.fit(tokenized_docs, show_progress=True)

    print(f"Collected stats for {len(stats.idf_scores)} unique tokens")
    print(f"Total documents processed: {stats.num_docs}")

    return stats


def test_compression_configs(
    model,
    documents,
    configs,
    queries: Optional[dict] = None,
    qrels: Optional[dict] = None,
    dataset_name: str = "test_dataset",
    index_type: str = "plaid",
    batch_size: int = 1024,
    output_file: str = "compression_eval_results.jsonl",
):
    """
    Test a list of compression configurations on the same corpus.

    Parameters
    ----------
    model : ColBERT
        The model to use for encoding
    documents : list
        List of documents to encode
    configs : list of CompressionConfig or None
        List of compression configs to test. Use None for baseline (no compression).
        Each config should have a 'description' field for display in the results table.
    queries : dict, optional
        Dictionary of query_id -> query_text. If provided with qrels, will perform
        retrieval evaluation.
    qrels : dict, optional
        Dictionary of relevance judgments. If provided with queries, will perform
        retrieval evaluation.
    dataset_name : str
        Name of the dataset (used for index naming). Default: "test_dataset"
    index_type : str
        Type of index to use: "flat" or "plaid". Default: "plaid"
    batch_size : int
        Batch size for encoding. Default: 1024
    output_file : str
        Path to save JSONL results. Default: "compression_eval_results.jsonl"
    """
    print("\n" + "=" * 80)
    print(f"COMPRESSION TESTING ON {dataset_name.upper()}")
    print("=" * 80)

    document_texts = [document["text"] for document in documents]
    document_ids = [document["id"] for document in documents]
    num_docs = len(documents)

    # Determine if we should do retrieval evaluation
    do_evaluation = queries is not None and qrels is not None

    # Store results with evaluation scores if available
    results = []  # Store (description, num_tokens, avg_tokens, reduction_pct, eval_scores)
    baseline_tokens = None

    for idx, config in enumerate(configs, 1):
        # Get description from config, or use "Baseline" for None
        if config is None:
            description = "Baseline (no compression)"
        elif config.description:
            description = config.description
        else:
            description = f"Configuration {idx}"

        print(f"\n[{idx}] {description}")
        print("-" * 80)

        # Encode documents with this configuration
        embeddings = model.encode(
            sentences=document_texts,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
            compression_config=config,
        )

        # Calculate token metrics
        num_tokens = sum(len(emb) for emb in embeddings)
        avg_tokens = num_tokens / num_docs

        # Calculate reduction relative to baseline
        if baseline_tokens is None:
            baseline_tokens = num_tokens
            reduction_pct = None
        else:
            reduction_pct = (1 - num_tokens / baseline_tokens) * 100

        # Print token metrics
        print(f"Total tokens: {num_tokens:,}")
        print(f"Avg tokens/doc: {avg_tokens:.2f}")
        if reduction_pct is not None:
            print(f"Reduction: {reduction_pct:.2f}%")

        # Perform retrieval evaluation if queries and qrels are provided
        eval_scores = None
        if do_evaluation:
            print("\nPerforming retrieval evaluation...")

            # Create index
            index_name = f"{dataset_name}_{idx}_{description[:20].replace(' ', '_')}"
            if index_type == "flat":
                index = indexes.Flat(override=True, index_name=index_name)
            elif index_type == "plaid":
                index = indexes.PLAID(override=True, index_name=index_name)
            else:
                raise ValueError(f"Invalid index type: {index_type}")

            # Add documents to index
            index.add_documents(
                documents_ids=document_ids,
                documents_embeddings=embeddings,
            )

            # Encode queries
            queries_embeddings = model.encode(
                sentences=list(queries.values()),
                is_query=True,
                show_progress_bar=False,
                batch_size=32,
            )

            # Retrieve
            retriever = retrieve.ColBERT(index=index)
            scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)

            # Remove query_id from scores if it appears (needed for some datasets like FiQA)
            for (query_id, query), query_scores in zip(queries.items(), scores):
                for score in query_scores:
                    if score["id"] == query_id:
                        query_scores.remove(score)

            # Evaluate
            eval_scores = evaluation.evaluate(
                scores=scores,
                qrels=qrels,
                queries=list(queries.keys()),
                metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
            )

            print(f"  MAP: {eval_scores.get('map', 0):.4f}")
            print(f"  NDCG@10: {eval_scores.get('ndcg@10', 0):.4f}")
            print(f"  Recall@10: {eval_scores.get('recall@10', 0):.4f}")

        # Store results
        results.append(
            (description, num_tokens, avg_tokens, reduction_pct, eval_scores)
        )

    # Print summary table using rich
    console.print("\n")

    if do_evaluation:
        table = Table(
            title="SUMMARY - Compression & Retrieval Performance",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", justify="right", style="cyan", width=3)
        table.add_column("Configuration", style="white", width=40)
        table.add_column("Total Tokens", justify="right", style="green")
        table.add_column("Avg/Doc", justify="right", style="green")
        table.add_column("Reduction", justify="right", style="yellow")
        table.add_column("MAP", justify="right", style="blue")
        table.add_column("NDCG@10", justify="right", style="blue")
        table.add_column("Recall@10", justify="right", style="blue")

        for idx, (
            description,
            num_tokens,
            avg_tokens,
            reduction_pct,
            eval_scores,
        ) in enumerate(results, 1):
            reduction_str = (
                f"{reduction_pct:.2f}%" if reduction_pct is not None else "-"
            )

            map_score = eval_scores.get("map", 0) if eval_scores else 0
            ndcg10 = eval_scores.get("ndcg@10", 0) if eval_scores else 0
            recall10 = eval_scores.get("recall@10", 0) if eval_scores else 0

            table.add_row(
                str(idx),
                description,
                f"{num_tokens:,}",
                f"{avg_tokens:.2f}",
                reduction_str,
                f"{map_score:.4f}",
                f"{ndcg10:.4f}",
                f"{recall10:.4f}",
            )
    else:
        table = Table(
            title="SUMMARY - Token Compression",
            show_header=True,
            header_style="bold magenta",
        )
        table.add_column("#", justify="right", style="cyan", width=3)
        table.add_column("Configuration", style="white", width=50)
        table.add_column("Total Tokens", justify="right", style="green")
        table.add_column("Avg/Doc", justify="right", style="green")
        table.add_column("Reduction", justify="right", style="yellow")

        for idx, (description, num_tokens, avg_tokens, reduction_pct, _) in enumerate(
            results, 1
        ):
            reduction_str = (
                f"{reduction_pct:.2f}%" if reduction_pct is not None else "-"
            )

            table.add_row(
                str(idx),
                description,
                f"{num_tokens:,}",
                f"{avg_tokens:.2f}",
                reduction_str,
            )

    console.print(table)

    # Save results to JSONL file
    output_path = Path(output_file)
    with open(output_path, "w+") as f:
        for idx, (
            description,
            num_tokens,
            avg_tokens,
            reduction_pct,
            eval_scores,
        ) in enumerate(results):
            config = configs[idx]

            result_dict = {
                "dataset": dataset_name,
                "config": serialize_compression_config(config),
                "compression_results": {
                    "total_tokens": num_tokens,
                    "avg_tokens_per_doc": float(avg_tokens),
                    "reduction_pct": float(reduction_pct)
                    if reduction_pct is not None
                    else None,
                    "num_documents": num_docs,
                },
            }

            # Add evaluation results if available
            if eval_scores:
                result_dict["eval_results"] = eval_scores

            f.write(json.dumps(result_dict) + "\n")

    console.print(f"\n[green]Results saved to {output_path}[/green]")

    return results


def inspect_pruned_tokens(
    model,
    documents,
    config: CompressionConfig,
    sampling: Literal["start", "random"] | list[int] = "start",
    num_docs: int = 10,
    num_tokens_per_doc: int = 10,
    batch_size: int = 1024,
):
    """
    Inspect which tokens were pruned by a compression configuration.

    Parameters
    ----------
    model : ColBERT
        The model to use for encoding
    documents : list
        List of documents to encode
    config : CompressionConfig
        Compression configuration with track_pruned_tokens=True in at least one pruning strategy
    sampling : "start", "random", or list of int
        How to sample documents:
        - "start": Take first num_docs documents
        - "random": Randomly sample num_docs documents
        - list[int]: Use specific document indices
    num_docs : int
        Number of documents to sample (ignored if sampling is a list)
    num_tokens_per_doc : int
        Maximum number of pruned tokens to display per document
    batch_size : int
        Batch size for encoding
    """
    print("\n" + "=" * 80)
    print("PRUNED TOKEN INSPECTION")
    print("=" * 80)

    # Sample documents based on sampling strategy
    if isinstance(sampling, list):
        doc_indices = sampling
        sample_docs = [documents[i] for i in doc_indices]
        print(f"Using specified document indices: {doc_indices}")
    elif sampling == "start":
        doc_indices = list(range(min(num_docs, len(documents))))
        sample_docs = documents[:num_docs]
        print(f"Sampling first {len(sample_docs)} documents")
    elif sampling == "random":
        doc_indices = random.sample(
            range(len(documents)), min(num_docs, len(documents))
        )
        sample_docs = [documents[i] for i in doc_indices]
        print(f"Randomly sampled {len(sample_docs)} documents: {doc_indices}")
    else:
        raise ValueError(
            f"Invalid sampling strategy: {sampling}. Use 'start', 'random', or a list of indices."
        )

    # Create a context from the config to access tracking data
    context = CompressionContext(config)

    # Encode with tracking
    print(f"\nEncoding {len(sample_docs)} documents with tracking enabled...")
    embeddings = model.encode(
        sentences=[doc["text"] for doc in sample_docs],
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=False,
        compression_config=context,
    )

    total_tokens = sum(len(emb) for emb in embeddings)
    avg_tokens = total_tokens / len(sample_docs)

    print(f"Total tokens after compression: {total_tokens:,}")
    print(f"Average tokens per doc: {avg_tokens:.2f}")

    # Get pruned tokens (assuming IDF pruning strategy)
    pruned_tokens = context.get_pruned_tokens("idf")
    stats = context.finalize()["idf"]
    print(f"Stats: {stats}")

    if not pruned_tokens:
        print(
            "\nNo pruned tokens tracked. Ensure track_pruned_tokens=True in your pruning config."
        )
        return

    # Get statistics about pruned tokens
    unique_pruned_tokens = list(
        set(token_id for doc_pruned in pruned_tokens for token_id in doc_pruned.keys())
    )
    total_pruned_occurrences = sum(len(doc_pruned) for doc_pruned in pruned_tokens)

    console.print("\n[bold cyan]PRUNING STATISTICS[/bold cyan]")
    console.print(
        f"Unique token types pruned: [green]{len(unique_pruned_tokens)}[/green]"
    )
    console.print(
        f"Total token occurrences pruned: [green]{total_pruned_occurrences:,}[/green]"
    )

    # Create a table for the most common pruned tokens
    token_table = Table(
        title="Token Types Pruned (sorted by IDF)",
        show_header=True,
        header_style="bold magenta",
    )
    token_table.add_column("Token", style="white")
    token_table.add_column("IDF Score", justify="right", style="yellow")

    # Show top 20 tokens with lowest IDF
    sorted_tokens = sorted(unique_pruned_tokens, key=lambda x: stats.get_idf(x))[:20]
    for token_id in sorted_tokens:
        token_text = model.tokenizer.decode([token_id])
        idf_score = stats.get_idf(token_id)
        token_table.add_row(f"'{token_text}'", f"{idf_score:.8f}")

    console.print(token_table)

    # Display pruned tokens per document
    console.print(
        f"\n[bold cyan]PRUNED TOKENS BY DOCUMENT[/bold cyan] (showing up to {num_tokens_per_doc} per document)"
    )

    for i, (doc_idx, doc_pruned) in enumerate(zip(doc_indices, pruned_tokens)):
        if doc_pruned:
            console.print(
                f"\n[bold yellow]Document {doc_idx}[/bold yellow] (sample index {i}): [green]{len(doc_pruned)} tokens pruned[/green]"
            )
            console.print(
                f"  Original text: [dim]{sample_docs[i]['text'][:100]}...[/dim]"
            )
            console.print(f"  Final token count: [green]{len(embeddings[i])}[/green]")

            # Create a table for pruned tokens in this document
            doc_table = Table(
                show_header=True, header_style="bold", box=None, padding=(0, 1)
            )
            doc_table.add_column("#", justify="right", style="cyan", width=4)
            doc_table.add_column("Token", style="white")
            doc_table.add_column("Position", justify="right", style="yellow")
            doc_table.add_column("IDF Score", justify="right", style="magenta")

            # Sort by IDF score (ascending) to show most common tokens first
            sorted_pruned = sorted(doc_pruned.items(), key=lambda x: x[1][1])

            for j, (token_id, (position, score)) in enumerate(
                sorted_pruned[:num_tokens_per_doc]
            ):
                token_text = model.tokenizer.decode([token_id])
                doc_table.add_row(
                    f"{j + 1}",
                    f"'{token_text}'",
                    str(position),
                    f"{score:.8f}",
                )

            console.print(doc_table)

            if len(doc_pruned) > num_tokens_per_doc:
                console.print(
                    f"  [dim]... and {len(doc_pruned) - num_tokens_per_doc} more tokens[/dim]"
                )
        else:
            console.print(
                f"\n[bold yellow]Document {doc_idx}[/bold yellow] (sample index {i}): [red]No tokens pruned[/red]"
            )

    console.print("")


if __name__ == "__main__":
    print("ColBERT Compression Testing on BEIR nfcorpus")
    print("=" * 80)

    # Load model
    MODEL_NAME = "lightonai/GTE-ModernColBERT-v1"
    print(f"\nLoading model: {MODEL_NAME}")
    model = models.ColBERT(
        model_name_or_path=MODEL_NAME,
        document_length=300,
        query_length=32,
    )

    # Load nfcorpus dataset
    print("\nLoading BEIR nfcorpus dataset...")
    documents, queries, qrels = evaluation.load_beir(
        dataset_name="nfcorpus",
        split="test",
    )
    print(f"Loaded {len(documents)} documents and {len(queries)} queries")

    try:
        idf_stats = collect_idf_stats(model, documents)

        # Get special tokens to demonstrate ignore_tokens feature
        special_token_ids = set(model.tokenizer.all_special_ids)
        print(
            f"\nFound {len(special_token_ids)} special tokens to ignore during pruning"
        )

        document_ks = [5, 10, 15, 20, 25]
        global_ks = [5, 10, 15, 20, 25, 30, 40]
        pooling_ks = [2, 3, 4, 5]
        # Define configurations to test
        baseline_config = CompressionConfig(description="Baseline (no compression)")
        document_idf_pruning_configs = [
            CompressionConfig(
                pruning=[
                    IDFPruningConfig(
                        mode="document",
                        top_k=k,
                        stats=idf_stats,
                        protected_tokens=1,
                        ignore_tokens=special_token_ids,
                    )
                ],
                description=f"IDF Pruning (per-document, prune {k} lowest-IDF tokens per doc)",
            )
            for k in document_ks
        ]
        global_idf_pruning_configs = [
            CompressionConfig(
                pruning=[
                    IDFPruningConfig(
                        mode="document",
                        top_k=10,
                        stats=idf_stats,
                        protected_tokens=1,
                        ignore_tokens=special_token_ids,
                    )
                ],
                description=f"IDF Pruning (per-document, prune {k} lowest-IDF tokens per doc)",
            )
            for k in document_ks
        ]
        pooling_configs = [
            CompressionConfig(
                pooling=[
                    PoolingConfig(
                        pool_factor=k,
                        protected_tokens=1,
                        clustering_method="hierarchical",
                    )
                ],
                description=f"Pooling Only (hierarchical, pool_factor={k})",
            )
            for k in pooling_ks
        ]

        configs = [
            # baseline_config,
            *document_idf_pruning_configs,
            # *global_idf_pruning_configs,
            # *pooling_configs,
        ]
        # Test 1: Test compression configurations with retrieval evaluation
        results = test_compression_configs(
            model=model,
            documents=documents,
            configs=configs,
            queries=queries,
            qrels=qrels,
            dataset_name="nfcorpus",
            index_type="plaid",
        )

        # Test 2: Inspect pruned tokens
        if False:
            # Test 2a: Global pruning - prune 50 token TYPES (all occurrences)
            print("\n" + "=" * 80)
            print("TEST: GLOBAL PRUNING (k=50 means 50 unique token types)")
            print(
                "This should identify the 50 token types with lowest IDF and prune ALL occurrences"
            )
            print("=" * 80)
            inspect_pruned_tokens(
                model=model,
                documents=documents,
                config=CompressionConfig(
                    pruning=[
                        IDFPruningConfig(
                            mode="global",
                            top_k=50,
                            stats=idf_stats,
                            protected_tokens=1,
                            track_pruned_tokens=True,
                            ignore_tokens=special_token_ids,
                        )
                    ]
                ),
                sampling="random",
                num_docs=5,
                num_tokens_per_doc=10,
            )

            # Test 2b: Per-document pruning - prune 20 token OCCURRENCES per doc
            print("\n" + "=" * 80)
            print("TEST: PER-DOCUMENT PRUNING (k=20 means 20 occurrences per doc)")
            print(
                "This should prune 20 lowest-IDF token occurrences from each document"
            )
            print("=" * 80)
            inspect_pruned_tokens(
                model=model,
                documents=documents,
                config=CompressionConfig(
                    pruning=[
                        IDFPruningConfig(
                            mode="document",
                            top_k=20,
                            stats=idf_stats,
                            protected_tokens=1,
                            track_pruned_tokens=True,
                            ignore_tokens=special_token_ids,
                        )
                    ]
                ),
                sampling="random",
                num_docs=10,
                num_tokens_per_doc=10,
            )

        print("\n✓ All tests completed successfully!")

    except Exception as e:
        print(f"\n✗ Test failed with error: {e}")
        import traceback

        traceback.print_exc()
