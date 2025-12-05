"""Evaluation script for BEIR datasets comparing multiple indexes."""

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import time
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
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

# Query length configuration for different datasets
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


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test BEIR index on a dataset")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=["robro612/xtr-base-en-pylate"],
        help="Name or path of the model(s) to use. Can be a single model, multiple models (space-separated), or a glob pattern like 'output/model_name/checkpoint-*' (default: 'robro612/xtr-base-en-pylate')",
        nargs="+",
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default=["nfcorpus"],
        help="Name(s) of the dataset(s) to evaluate on. Can be a single dataset or multiple datasets (space-separated) (default: 'nfcorpus')",
        nargs="+",
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
    parser.add_argument(
        "--shard_size",
        type=int,
        default=10_000,
        help="Number of documents per shard for encoding (default: 10000)",
    )
    parser.add_argument(
        "--embedding_dtype",
        type=str,
        default="fp16",
        choices=["fp32", "fp16", "bf16"],
        help="Datatype to cast embeddings to after encoding (model still runs in fp32) for saving and retrieval. Options: fp32, fp16, bf16 (default: fp16). NOTE: bf16 is not supported by all index types (e.g. ScaNN).",
    )
    parser.add_argument(
        "--lowercase",
        action="store_true",
        help="Convert documents and queries to lowercase before encoding (default: False)",
    )
    return parser.parse_args()


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert dtype string to torch dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    return dtype_map.get(dtype_str)


def cast_embeddings(embeddings: List[torch.Tensor], dtype: torch.dtype) -> List[torch.Tensor]:
    """Cast embeddings to the specified dtype."""
    current_dtype = embeddings[0].dtype
    if dtype == current_dtype:
        return embeddings  # No casting needed
    return [emb.to(dtype) for emb in embeddings]


def pack_embeddings(embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """
    Pack a list of embeddings into a dict with concatenated embeddings and lengths.
    
    Returns:
        Dict with 'embeddings' (concatenated tensor) and 'lengths' (tensor of sequence lengths)
    """
    if not embeddings:
        return {"embeddings": torch.empty(0), "lengths": torch.empty(0, dtype=torch.long)}
    
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
    concatenated = torch.cat(embeddings, dim=0)
    
    return {"embeddings": concatenated, "lengths": lengths}


def unpack_embeddings(packed: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """
    Unpack a dict of concatenated embeddings and lengths back into a list of tensors.
    
    Args:
        packed: Dict with 'embeddings' (concatenated tensor) and 'lengths' (tensor of sequence lengths)
    
    Returns:
        List of embedding tensors
    """
    if packed["lengths"].numel() == 0:
        return []
    
    concatenated = packed["embeddings"]
    lengths = packed["lengths"]
    
    embeddings = []
    start_idx = 0
    for length in lengths:
        end_idx = start_idx + length.item()
        embeddings.append(concatenated[start_idx:end_idx])
        start_idx = end_idx
    
    return embeddings


def expand_model_paths(model_paths: List[str]) -> List[str]:
    """Expand glob patterns in model paths."""
    expanded = []
    for path in model_paths:
        # Check if it's a glob pattern
        if '*' in path or '?' in path or '[' in path:
            matches = sorted(glob.glob(path))
            if not matches:
                print(f"Warning: No matches found for glob pattern: {path}")
            expanded.extend(matches)
        else:
            # Check if path exists (for local paths)
            if os.path.exists(path) or '/' in path or path.startswith('robro612/') or path.startswith('colbert-ir/'):
                expanded.append(path)
            else:
                print(f"Warning: Model path may not exist: {path}")
                expanded.append(path)  # Still add it, might be a HuggingFace model name
    return expanded


def load_dataset(dataset_name: str) -> Tuple[List[Dict[str, str]], Dict[str, str], Any]:
    """
    Load dataset and return documents, queries, and qrels.
    
    Returns:
        Tuple of (documents, queries, qrels), and the potentially modified dataset_name
    """
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
    
    return documents, queries, qrels, dataset_name


def preprocess_texts(documents: List[Dict[str, str]], queries: Dict[str, str], lowercase: bool = False) -> Tuple[List[Dict[str, str]], Dict[str, str]]:
    """Optionally convert documents and queries to lowercase."""
    if lowercase:
        print("Casting documents and queries to lowercase...")
        documents = [
            {
                "id": document["id"],
                "text": document["text"].lower(),
            }
            for document in documents
        ]

        queries = {
            query_id: query.lower() if lowercase else query for query_id, query in queries.items()
        }

    
    return documents, queries


def setup_cache_directory(
    model_name: str,
    dataset_name: str,
    documents: List[Dict[str, str]],
    cache_dir: str,
    cache_embeddings: bool,
    embedding_dtype: str,
) -> Tuple[Path, str, Path, Path]:
    """
    Set up cache directory structure and generate cache keys.
    
    Returns:
        Tuple of (cache_subdir, cache_key, doc_embeddings_cache_file, query_embeddings_cache_file)
    """
    # Generate cache directory structure: cache_dir/sanitized_model_name/dataset/embedding_dtype
    # Include dtype in path to prevent mixing shards from different dtypes
    sanitized_model_name = "_".join(model_name.split("/")[-2:])
    cache_subdir = Path(cache_dir) / sanitized_model_name / dataset_name / embedding_dtype
    
    # Create cache directory if caching is enabled
    if cache_embeddings:
        cache_subdir.mkdir(parents=True, exist_ok=True)
    
    # Generate cache filenames based on dataset, model, and document content hash
    # Use a hash of document IDs and texts to detect if dataset changed
    doc_hash_input = "".join([doc["id"] + doc["text"] for doc in documents[:100]])  # Sample for hash
    doc_hash = hashlib.md5(doc_hash_input.encode()).hexdigest()[:8]
    cache_key = f"{len(documents)}_{doc_hash}"
    doc_embeddings_cache_file = cache_subdir / f"{cache_key}_doc_embeddings.pt"
    query_embeddings_cache_file = cache_subdir / f"{cache_key}_query_embeddings.pt"
    
    return cache_subdir, cache_key, doc_embeddings_cache_file, query_embeddings_cache_file


def encode_documents_with_sharding(
    model: models.ColBERT,
    documents: List[Dict[str, str]],
    cache_subdir: Path,
    cache_key: str,
    doc_embeddings_cache_file: Path,
    cache_embeddings: bool,
    shard_size: int,
    batch_size: int,
    embedding_dtype: torch.dtype,
) -> List[Any]:
    """
    Encode documents with sharding support, saving iteratively.
    
    Returns:
        List of document embeddings
    """
    num_documents = len(documents)
    num_shards = (num_documents + shard_size - 1) // shard_size
    
    # Calculate number of digits needed for zero-padding shard numbers
    num_digits = len(str(num_shards - 1)) if num_shards > 0 else 1
    
    # Check for existing shards if caching is enabled
    cached_shards = {}
    if cache_embeddings:
        for shard_idx in range(num_shards):
            shard_num_str = f"{shard_idx:0{num_digits}d}"
            shard_cache_file = cache_subdir / f"{cache_key}_doc_embeddings_shard_{shard_num_str}.pt"
            if shard_cache_file.exists():
                cached_shards[shard_idx] = shard_cache_file
    
    # Encode documents in shards
    print(f"Encoding documents in shards of size {shard_size}...")
    print(f"Total documents: {num_documents}, Number of shards: {num_shards}")
    if cached_shards:
        print(f"Found {len(cached_shards)} cached shards, will encode {num_shards - len(cached_shards)} missing shards.")
    
    documents_embeddings = []
    
    # Process shards in order (load cached or encode missing)
    for shard_idx in range(num_shards):
        if shard_idx in cached_shards:
            # Load cached shard
            shard_cache_file = cached_shards[shard_idx]
            print(f"Loading cached shard {shard_idx + 1}/{num_shards} from {shard_cache_file}...")
            
            # Load torch format
            packed = torch.load(shard_cache_file, map_location="cpu")
            shard_embeddings = unpack_embeddings(packed)
            # Cast embeddings to specified dtype
            shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
            
            documents_embeddings.extend(shard_embeddings)
            print(f"Loaded shard {shard_idx + 1}/{num_shards} ({len(shard_embeddings)} embeddings)")
        else:
            # Encode missing shard
            start_idx = shard_idx * shard_size
            end_idx = min(start_idx + shard_size, num_documents)
            shard_documents = documents[start_idx:end_idx]
            
            print(f"Encoding shard {shard_idx + 1}/{num_shards} (documents {start_idx} to {end_idx - 1})...")
            shard_embeddings = model.encode(
                sentences=[document["text"] for document in shard_documents],
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
            # Cast embeddings to specified dtype after encoding
            shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
            documents_embeddings.extend(shard_embeddings)
            
            # Save shard immediately after encoding
            if cache_embeddings:
                shard_num_str = f"{shard_idx:0{num_digits}d}"
                shard_cache_file = cache_subdir / f"{cache_key}_doc_embeddings_shard_{shard_num_str}.pt"
                print(f"Saving shard {shard_idx + 1}/{num_shards} to {shard_cache_file}...")
                # Pack embeddings before saving
                packed = pack_embeddings(shard_embeddings)
                torch.save(packed, shard_cache_file)
                print(f"Shard {shard_idx + 1}/{num_shards} saved.")
    
    print(f"Document encoding complete. Total embeddings: {len(documents_embeddings)}")
    
    return documents_embeddings


def encode_queries(
    model: models.ColBERT,
    queries: Dict[str, str],
    query_embeddings_cache_file: Path,
    cache_embeddings: bool,
    batch_size: int,
    embedding_dtype: torch.dtype,
) -> List[Any]:
    """
    Encode queries, loading from cache if available.
    
    Returns:
        List of query embeddings
    """
    if cache_embeddings and query_embeddings_cache_file.exists():
        print(f"Loading cached query embeddings from {query_embeddings_cache_file}...")
        packed = torch.load(query_embeddings_cache_file, map_location="cpu")
        queries_embeddings = unpack_embeddings(packed)
        print(f"Loaded {len(queries_embeddings)} query embeddings from cache.")
        # Cast embeddings to specified dtype
        queries_embeddings = cast_embeddings(queries_embeddings, embedding_dtype)
    else:
        print("Encoding queries...")
        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=batch_size,
            convert_to_tensor=True,
        )
        # Cast embeddings to specified dtype after encoding
        queries_embeddings = cast_embeddings(queries_embeddings, embedding_dtype)
        
        # Save query embeddings if caching is enabled
        if cache_embeddings:
            print(f"Saving query embeddings to {query_embeddings_cache_file}...")
            # Pack embeddings before saving
            packed = pack_embeddings(queries_embeddings)
            torch.save(packed, query_embeddings_cache_file)
            print("Query embeddings saved.")
    
    return queries_embeddings


def get_index_configs(
    dataset_name: str,
    model_name: str,
    embedding_size: int,
    args: argparse.Namespace,
    index_types: List[str],
) -> List[Dict[str, Any]]:
    """
    Get index configurations for the specified index types.
    
    Returns:
        List of index configuration dictionaries
    """
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
                "verbose": True,
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
            "name": "PLAID",
            "index_class": indexes.PLAID,
            "init_kwargs": {
                "override": True,
                "index_name": f"{base_name}_plaid",
            },
            "add_documents_kwargs": {},
        },
    ]

    index_configs = [config for config in all_index_configs if config["name"] in index_types]
    return index_configs


def test_index(
    config: Dict[str, Any],
    documents: List[Dict[str, str]],
    documents_embeddings: List[torch.Tensor],
    queries: Dict[str, str],
    queries_embeddings: List[torch.Tensor],
    qrels: Any,
    dataset_name: str,
    model_name: str,
    k: int,
    k_token: int,
    verbose: bool,
    results_dir: Path,
    embedding_dtype: str,
    lowercase: bool,
) -> None:
    """
    Test a single index: create, add documents, retrieve, evaluate, and save results.
    """
    index_name = config["name"]
    print("\n" + "="*80)
    print(f"Testing {index_name} index...")
    print("="*80)
    
    # Initialize index
    index = config["index_class"](**config["init_kwargs"])
    retriever = retrieve.ColBERT(index=index, verbose=verbose)
    
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
    if isinstance(index, indexes.PLAID):
        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=k)
    else:
        scores = retriever.retrieve_xtr(queries_embeddings=queries_embeddings, k=k, k_token=k_token, batch_size=1)
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
        "embedding_dtype": embedding_dtype,
        "lowercase": lowercase,
        "evaluation_scores": evaluation_scores,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "k": k,
        "k_token": k_token,
        "index_config": index_config_info,
    }
    
    # Append to index-specific JSONL file
    jsonl_file = Path(results_dir) / f"{index_name}.jsonl"
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(jsonl_entry) + "\n")
    
    print(f"\n{index_name} Results:")
    print(evaluation_scores)
    print(f"Results appended to: {jsonl_file}")


def main() -> None:
    """Main function that orchestrates the evaluation process."""
    args = parse_arguments()
    
    # Convert embedding dtype string to torch dtype
    embedding_dtype = get_torch_dtype(args.embedding_dtype)
    print(f"Embedding dtype: {args.embedding_dtype} ({embedding_dtype})")
    
    # Expand glob patterns for model paths
    all_model_paths = expand_model_paths(args.model_name_or_path)
    all_dataset_names = args.dataset_name
    
    print(f"Models to test: {all_model_paths}")
    print(f"Datasets to test: {all_dataset_names}")
    print(f"Total combinations: {len(all_model_paths) * len(all_dataset_names)}")
    
    # Iterate over all combinations of datasets and models
    for dataset_name, model_name in itertools.product(all_dataset_names, all_model_paths):
        print("\n" + "="*80)
        print(f"Processing: Dataset={dataset_name}, Model={model_name}")
        print("="*80)
        
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=300,
            query_length=QUERY_LEN.get(dataset_name),
        )
        model.compile()

        # Load dataset
        documents, queries, qrels, dataset_name = load_dataset(dataset_name)

        # Preprocess texts (optionally lowercase)
        documents, queries = preprocess_texts(documents, queries, lowercase=args.lowercase)

        # Setup cache directory
        cache_subdir, cache_key, doc_embeddings_cache_file, query_embeddings_cache_file = setup_cache_directory(
            model_name=model_name,
            dataset_name=dataset_name,
            documents=documents,
            cache_dir=args.cache_dir,
            cache_embeddings=args.cache_embeddings,
            embedding_dtype=args.embedding_dtype,
        )
        
        # Encode documents with sharding
        documents_embeddings = encode_documents_with_sharding(
            model=model,
            documents=documents,
            cache_subdir=cache_subdir,
            cache_key=cache_key,
            doc_embeddings_cache_file=doc_embeddings_cache_file,
            cache_embeddings=args.cache_embeddings,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            embedding_dtype=embedding_dtype,
        )
        
        # Encode queries
        queries_embeddings = encode_queries(
            model=model,
            queries=queries,
            query_embeddings_cache_file=query_embeddings_cache_file,
            cache_embeddings=args.cache_embeddings,
            batch_size=args.batch_size,
            embedding_dtype=embedding_dtype,
        )

        # Get embedding size from the model's final layer
        embedding_size = 128

        # Get index configurations
        index_configs = get_index_configs(
            dataset_name=dataset_name,
            model_name=model_name,
            embedding_size=embedding_size,
            args=args,
            index_types=args.index_types,
        )
        
        print(f"Testing {len(index_configs)} indexes: {args.index_types}")
        print(f"Index configurations: {index_configs}")

        if args.limit_queries:
            queries_embeddings = queries_embeddings[:args.limit_queries]
        if args.limit_documents:
            documents_embeddings = documents_embeddings[:args.limit_documents]

        print(f"Embedding size: {embedding_size}")
        print(f"Number of documents: {len(documents)}")
        print(f"Doc 1 embedding shape: {documents_embeddings[0].shape}")

        # Create results directory
        results_dir = Path("results") / dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)

        # Test each index
        for config in index_configs:
            test_index(
                config=config,
                documents=documents,
                documents_embeddings=documents_embeddings,
                queries=queries,
                queries_embeddings=queries_embeddings,
                qrels=qrels,
                dataset_name=dataset_name,
                model_name=model_name,
                k=args.k,
                k_token=args.k_token,
                verbose=args.verbose,
                results_dir=results_dir,
                embedding_dtype=args.embedding_dtype,
                lowercase=args.lowercase,
            )


if __name__ == "__main__":
    main()
