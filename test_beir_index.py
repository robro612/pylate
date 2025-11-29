"""Evaluation script for BEIR datasets comparing multiple indexes."""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import os
import pickle
import time
from typing import Any
from tqdm.auto import tqdm
import torch
import itertools

import numpy as np

from pylate import evaluation, indexes, models, retrieve

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

if __name__ == "__main__":
    query_len = {
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

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'nfcorpus')",
    )
    parser.add_argument(
        "--cache_embeddings",
        action="store_true",
        help="Save/load document embeddings to avoid re-encoding",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="embedding_cache",
        help="Directory to store cached embeddings (default: 'embedding_cache')",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2048,
        help="Batch size to use for encoding documents and queries (default: 2048)",
    )
    parser.add_argument(
        "--k",
        type=int,
        default=100,
        help="Number of documents to retrieve per query (default: 100)",
    )
    parser.add_argument(
        "--k_token",
        type=int,
        default=4_000,
        help="Number of candidates to retrieve per query token for non-PLAID indexes (default: 4000)",
    )
    parser.add_argument(
        "--num_leaves",
        type=int,
        default=None,
        help="Number of leaves to use for the ScaNN index (default: None)",
    )
    parser.add_argument(
        "--num_leaves_to_search",
        type=int,
        default=None,
        help="Number of leaves to search for the ScaNN index (default: None)",
    )
    parser.add_argument(
        "--num_neighbors",
        type=int,
        default=None,
        help="Number of neighbors to use for the ScaNN index (default: None)",
    )
    parser.add_argument(
        "--index_types",
        type=str,
        default=["Flat"],
        help="Types of indexes to test (default: ['Flat'])",
        nargs="+",
        choices=["Flat", "ScaNN", "Voyager", "PLAID"],
    )

    parser.add_argument(
        "--limit_queries",
        type=int,
        default=None,
        help="Limit the number of queries to test (default: None)",
    )
    parser.add_argument(
        "--limit_documents",
        type=int,
        default=None,
        help="Limit the number of documents to test (default: None)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging of timing and operations",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    # model_name = "lightonai/GTE-ModernColBERT-v1"
    model_name = "robro612/xtr-base-en-pylate"
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )

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
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    # casting to lowercase
    print("Casting documents and queries to lowercase...")
    documents = [
        {
            "id": document["id"],
            "text": document["text"].lower(),
        }
        for document in documents
    ]

    queries = {
        query_id: query.lower() for query_id, query in queries.items()
    }


    # Create cache directory if caching is enabled
    if args.cache_embeddings:
        os.makedirs(args.cache_dir, exist_ok=True)
    
    # Generate cache filenames based on dataset, model, and document content hash
    # Use a hash of document IDs and texts to detect if dataset changed
    doc_hash_input = "".join([doc["id"] + doc["text"] for doc in documents[:100]])  # Sample for hash
    doc_hash = hashlib.md5(doc_hash_input.encode()).hexdigest()[:8]
    cache_key = f"{dataset_name}_{model_name.split('/')[-1]}_{len(documents)}_{doc_hash}"
    doc_embeddings_cache_file = os.path.join(args.cache_dir, f"{cache_key}_doc_embeddings.pkl")
    query_embeddings_cache_file = os.path.join(args.cache_dir, f"{cache_key}_query_embeddings.pkl")
    
    # Load or encode document embeddings
    if args.cache_embeddings and os.path.exists(doc_embeddings_cache_file):
        print(f"Loading cached document embeddings from {doc_embeddings_cache_file}...")
        with open(doc_embeddings_cache_file, "rb") as f:
            documents_embeddings = pickle.load(f)
        print(f"Loaded {len(documents_embeddings)} document embeddings from cache.")
    else:
        print("Encoding documents...")
        documents_embeddings = model.encode(
            sentences=[document["text"] for document in documents],
            batch_size=args.batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        
        # Save document embeddings if caching is enabled
        if args.cache_embeddings:
            print(f"Saving document embeddings to {doc_embeddings_cache_file}...")
            with open(doc_embeddings_cache_file, "wb") as f:
                pickle.dump(documents_embeddings, f)
            print("Document embeddings saved.")
    
    # Load or encode query embeddings
    if args.cache_embeddings and os.path.exists(query_embeddings_cache_file):
        print(f"Loading cached query embeddings from {query_embeddings_cache_file}...")
        with open(query_embeddings_cache_file, "rb") as f:
            queries_embeddings = pickle.load(f)
        print(f"Loaded {len(queries_embeddings)} query embeddings from cache.")
    else:
        print("Encoding queries...")
        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=args.batch_size,
        )
        
        # Save query embeddings if caching is enabled
        if args.cache_embeddings:
            print(f"Saving query embeddings to {query_embeddings_cache_file}...")
            with open(query_embeddings_cache_file, "wb") as f:
                pickle.dump(queries_embeddings, f)
            print("Query embeddings saved.")

    # Get embedding size from the model's final layer
    embedding_size = 128

    # Define index configurations
    base_name = f"{dataset_name}_{model_name.split('/')[-1]}"
    all_index_configs = [
        {
            "name": "ScaNN",
            "index_class": indexes.ScaNN,
            "init_kwargs": {
                "name": f"{base_name}_scann",
                "embedding_size": embedding_size,
                "num_neighbors": args.num_neighbors,
                "num_leaves": args.num_leaves,
                "num_leaves_to_search": args.num_leaves_to_search,
                "verbose": False,
            },
            "add_documents_kwargs": {
                "batch_size": args.batch_size,
            },
        },
        {
            "name": "Flat",
            "index_class": indexes.Flat,
            "init_kwargs": {
                "name": f"{base_name}_flat",
                "embedding_size": embedding_size,
                "device": "cuda",  # Use GPU acceleration
                "search_batch_size": args.batch_size,  # Batch size for search
                "verbose": False,
            },
            "add_documents_kwargs": {
                "batch_size": args.batch_size,
            },
        },
        {
            "name": "Voyager",
            "index_class": indexes.Voyager,
            "init_kwargs": {
                "index_folder": "test_indexes",
                "index_name": f"{base_name}_voyager",
                "override": True,
                "embedding_size": embedding_size,
            },
            "add_documents_kwargs": {},
        },
        {
            "name": "Voyager",
            "index_class": indexes.Voyager,
            "init_kwargs": {
                "index_folder": "test_indexes",
                "index_name": f"{base_name}_voyager",
                "override": True,
                "embedding_size": embedding_size,
            },
            "add_documents_kwargs": {},
        },
        {
            "name": "PLAID",
            "index_class": indexes.PLAID,
            "init_kwargs": {
                "override": True,
                "index_name": f"{base_name}_plaid",
            },
            "add_documents_kwargs": {},
        },
    ]

    index_configs = [config for config in all_index_configs if config["name"] in args.index_types]
    print(f"Testing {len(index_configs)} indexes: {args.index_types}")
    print(f"Index configurations: {index_configs}")

    # convert documents_embeddings and queries_embeddings to lists of tensors on device
    documents_embeddings = [torch.tensor(doc_emb, device="cuda") for doc_emb in tqdm(documents_embeddings, desc="Converting documents embeddings to tensors")]
    queries_embeddings = [torch.tensor(query_emb, device="cuda") for query_emb in tqdm(queries_embeddings, desc="Converting queries embeddings to tensors")]

    if args.limit_queries:
        queries_embeddings = queries_embeddings[:args.limit_queries]

    if args.limit_documents:
        documents_embeddings = documents_embeddings[:args.limit_documents]

    print(f"Embedding size: {embedding_size}")
    print(f"Number of documents: {len(documents)}")
    print(f"Doc 1 embedding shape: {documents_embeddings[0].shape}")

    # Create results directory
    results_dir = f"results/{dataset_name}"
    os.makedirs(results_dir, exist_ok=True)
    
    results = {}

    # Test each index
    for config in index_configs:
        index_name = config["name"]
        print("\n" + "="*80)
        print(f"Testing {index_name} index...")
        print("="*80)
        
        # Initialize index
        index = config["index_class"](**config["init_kwargs"])
        retriever = retrieve.ColBERT(index=index, verbose=args.verbose)
        
        # Add documents
        print(f"Adding documents to {index_name} index...")
        start_time = time.time()
        add_kwargs = {
            "documents_ids": [document["id"] for document in documents],
            "documents_embeddings": documents_embeddings,
            **config["add_documents_kwargs"],
        }
        index.add_documents(**add_kwargs)
        index_time = time.time() - start_time
        print(f"{index_name} indexing time: {index_time:.2f} seconds")
        
        # Retrieve
        print(f"Retrieving with {index_name}...")
        start_time = time.time()
        scores = retriever.retrieve_xtr(queries_embeddings=queries_embeddings, k=args.k, k_token=args.k_token)
        retrieve_time = time.time() - start_time
        print(f"{index_name} retrieval time: {retrieve_time:.2f} seconds")
        
        # Remove query_id from scores, needed for FiQA dataset
        for (query_id, query), query_scores in zip(queries.items(), scores):
            for score in query_scores:
                if score["id"] == query_id:
                    query_scores.remove(score)
        
        # Evaluate
        evaluation_scores = evaluation.evaluate(
            scores=scores,
            qrels=qrels,
            queries=list(queries.keys()),
            metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )
        
        # Store results
        results[index_name] = {
            "scores": evaluation_scores,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
        }
        
        # Save retrieval results to file
        results_file = os.path.join(results_dir, f"{index_name}_results.json")
        with open(results_file, "w") as f:
            # Convert scores to serializable format
            serializable_scores = []
            for query_id, query_scores_list in zip(queries.keys(), scores):
                serializable_scores.append({
                    "query_id": query_id,
                    "scores": [
                        {"id": score["id"], "score": float(score["score"])}
                        for score in query_scores_list
                    ]
                })
            
            json.dump({
                "index_name": index_name,
                "dataset": dataset_name,
                "model": model_name,
                "evaluation_scores": evaluation_scores,
                "index_time": index_time,
                "retrieve_time": retrieve_time,
                "retrieval_results": serializable_scores,
            }, f, indent=2)
        
        # Prepare index config info (excluding non-serializable class)
        index_config_info = {
            "name": config["name"],
            "init_kwargs": config["init_kwargs"],
            **({"add_documents_kwargs": config["add_documents_kwargs"]} if "add_documents_kwargs" in config else {}),
        }
        
        # Create result entry for JSONL
        jsonl_entry = {
            "dataset": dataset_name,
            "model": model_name,
            "evaluation_scores": evaluation_scores,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
            "k": 20,  # Number of documents to retrieve
            "k_token": args.k_token,
            "index_config": index_config_info,
        }
        
        # Append to index-specific JSONL file
        jsonl_file = os.path.join(results_dir, f"{index_name}.jsonl")
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")
        
        print(f"\n{index_name} Results:")
        print(evaluation_scores)
        print(f"\nResults saved to: {results_file}")
        print(f"Results appended to: {jsonl_file}")

    # Print comparison
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\nDataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Number of documents: {len(documents)}")
    print(f"Number of queries: {len(queries)}")
    
    if len(results) < 2:
        print("\nOnly one index tested. Comparison requires at least 2 indexes.")
        print("\n" + "="*80)
        exit(0)
    
    # Get baseline (first index) for comparison
    baseline_name = list(results.keys())[0]
    baseline_results = results[baseline_name]
    
    print("\n" + "-"*80)
    print("METRICS COMPARISON")
    print("-"*80)
    metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]
    
    # Create header with all index names
    index_names = list(results.keys())
    header = f"{'Metric':<20}"
    for idx_name in index_names:
        header += f" {idx_name:<15}"
    if len(index_names) == 2:
        header += f" {'Difference':<15}"
    print(f"\n{header}")
    print("-" * (20 + 15 * len(index_names) + (15 if len(index_names) == 2 else 0)))
    
    for metric in metrics:
        row = f"{metric:<20}"
        baseline_val = baseline_results["scores"].get(metric, 0.0)
        for idx_name in index_names:
            val = results[idx_name]["scores"].get(metric, 0.0)
            row += f" {val:<15.4f}"
        
        if len(index_names) == 2:
            # Show difference between two indexes
            other_name = index_names[1] if index_names[0] == baseline_name else index_names[0]
            other_val = results[other_name]["scores"].get(metric, 0.0)
            diff = other_val - baseline_val
            diff_pct = (diff / baseline_val * 100) if baseline_val > 0 else 0.0
            row += f" {diff:+.4f} ({diff_pct:+.2f}%)"
        
        print(row)
    
    print("\n" + "-"*80)
    print("PERFORMANCE COMPARISON")
    print("-"*80)
    
    # Indexing time comparison
    print(f"\n{'Operation':<20}", end="")
    for idx_name in index_names:
        print(f" {idx_name:<15}", end="")
    if len(index_names) == 2:
        print(f" {'Speedup':<15}", end="")
    print()
    print("-" * (20 + 15 * len(index_names) + (15 if len(index_names) == 2 else 0)))
    
    baseline_index_time = baseline_results["index_time"]
    baseline_retrieve_time = baseline_results["retrieve_time"]
    
    # Indexing times
    row = f"{'Indexing (s)':<20}"
    for idx_name in index_names:
        idx_time = results[idx_name]["index_time"]
        row += f" {idx_time:<15.2f}"
    if len(index_names) == 2:
        other_name = index_names[1] if index_names[0] == baseline_name else index_names[0]
        other_time = results[other_name]["index_time"]
        speedup = baseline_index_time / other_time if other_time > 0 else 0.0
        row += f" {speedup:.2f}x"
    print(row)
    
    # Retrieval times
    row = f"{'Retrieval (s)':<20}"
    for idx_name in index_names:
        ret_time = results[idx_name]["retrieve_time"]
        row += f" {ret_time:<15.2f}"
    if len(index_names) == 2:
        other_name = index_names[1] if index_names[0] == baseline_name else index_names[0]
        other_time = results[other_name]["retrieve_time"]
        speedup = baseline_retrieve_time / other_time if other_time > 0 else 0.0
        row += f" {speedup:.2f}x"
    print(row)
    
    # Save comparison summary
    comparison_file = os.path.join(results_dir, "comparison_summary.json")
    with open(comparison_file, "w") as f:
        json.dump({
            "dataset": dataset_name,
            "model": model_name,
            "num_documents": len(documents),
            "num_queries": len(queries),
            "results": {
                name: {
                    "evaluation_scores": result["scores"],
                    "index_time": result["index_time"],
                    "retrieve_time": result["retrieve_time"],
                }
                for name, result in results.items()
            }
        }, f, indent=2)
    
    print(f"\nComparison summary saved to: {comparison_file}")
    print("\n" + "="*80)
