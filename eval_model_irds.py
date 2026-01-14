# Evaluation script for IRDS datasets comparing multiple indexes.

from __future__ import annotations

import argparse
import glob
import hashlib
import json
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path
from typing import Any, List, Dict, Tuple, Optional
from tqdm.auto import tqdm
import torch
import itertools
from ranx import Run

import numpy as np

from pylate import evaluation, indexes, models, retrieve
import ir_datasets

# Configure logging to show INFO level messages
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# Query length configuration for different datasets
QUERY_LEN = {
    "beir/nfcorpus/test": 32,
    "beir/fiqa/test": 32,
    "beir/scidocs" : 48,
    "beir/scifact/test" : 48,
    "beir/trec-covid" : 48,
    "beir/webis-touche2020/v2" : 32,
    "beir/quora/test" : 32,
    "beir/nq" : 32,
    "disks45/nocr/trec-robust-2004" : 32,
    "lotte/lifestyle/dev/forum" : 32,
    "lotte/lifestyle/dev/search" : 32,
    "lotte/lifestyle/test/forum" : 32,
    "lotte/lifestyle/test/search" : 32,
    "lotte/pooled/dev/forum" : 32,
    "lotte/pooled/dev/search" : 32,
    "lotte/pooled/test/forum" : 32,
    "lotte/pooled/test/search" : 32,
    "lotte/recreation/dev/forum" : 32,
    "lotte/recreation/dev/search" : 32,
    "lotte/recreation/test/forum" : 32,
    "lotte/recreation/test/search" : 32,
    "lotte/science/dev/forum" : 32,
    "lotte/science/dev/search" : 32,
    "lotte/science/test/forum" : 32,
    "lotte/science/test/search" : 32,
    "lotte/technology/dev/forum" : 32,
    "lotte/technology/dev/search" : 32,
    "lotte/technology/test/forum" : 32,
    "lotte/technology/test/search" : 32,
    "lotte/writing/dev/forum" : 32,
    "lotte/writing/dev/search" : 32,
    "lotte/writing/test/forum" : 32,
    "lotte/writing/test/search" : 32,
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Test index on ir_datasets dataset")
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
        default=["beir/nfcorpus/test"],
        help="Full ir_datasets dataset ID(s) (e.g., 'beir/nfcorpus/test'). Can be a single dataset or multiple datasets (space-separated) (default: 'beir/nfcorpus/test')",
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
        "--encode_batch_size",
        type=int,
        default=2048,
        help="Batch size to use for encoding documents and queries (default: 2048)",
    )
    parser.add_argument(
        "--retrieve_batch_size",
        type=int,
        default=1,
        help="Batch size to use for retrieval operations (default: 1)",
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
        "--use_autopilot",
        action="store_true",
        help="Use ScaNN's autopilot() method for automatic parameter tuning. Overrides num_leaves, num_leaves_to_search, and training_sample_size (default: False)",
    )
    parser.add_argument(
        "--save_index",
        action="store_true",
        help="Save the index to disk after encoding (default: False)",
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
        "--model_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Datatype to cast the model to after encoding (default: fp32).",
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
    parser.add_argument(
        "--query_len",
        type=int,
        default=None,
        help="Query length to use. If not specified, uses QUERY_LEN dict value for the dataset, or 32 as default (default: None)",
    )
    parser.add_argument(
        "--doc_len",
        type=int,
        default=300,
        help="Document length to use (default: 300)",
    )
    parser.add_argument(
        "--move_embeddings_to_cpu",
        dest="move_embeddings_to_cpu",
        action="store_true",
        default=False,
        help="Move embeddings to CPU immediately after encoding to save GPU memory. Useful for large datasets to avoid OOM (default: False)",
    )
    parser.add_argument(
        "--save_runfile",
        action="store_true",
        default=False,
        help="Save ranx Run file for each evaluation run (default: False)",
    )
    parser.add_argument(
        "--retrieval_mode",
        type=str,
        default="XTR",
        choices=["ColBERT", "XTR"],
        help="Retrieval mode: 'ColBERT' for full ColBERT reranking (requires store_embeddings=True for ScaNN), 'XTR' for XTR scoring (default: 'XTR')",
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


def move_embeddings_to_cpu(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Move embeddings to CPU."""
    return [emb.cpu() for emb in embeddings]


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


def sanitize_dataset_name(dataset_name: str) -> str:
    """Sanitize dataset name for use in file paths by replacing / with _."""
    return dataset_name.replace("/", "_")

def sanitize_model_name(model_name: str) -> str:
    # remove everything before output/
    sanitized = model_name.split("output/")[-1].replace("/", "_")
    # Remove checkpoint suffix if present (we extract it separately)
    if "_checkpoint-" in sanitized:
        sanitized = sanitized.rsplit("_checkpoint-", 1)[0]
    return sanitized

def extract_checkpoint_number(model_name: str) -> str | None:
    """Extract checkpoint number from model path if present.
    
    Examples:
        "output/model_name/checkpoint-15000" -> "15000"
        "robro612/xtr-base-en-pylate" -> None
        "output/model/checkpoint-2500" -> "2500"
    """
    if "checkpoint-" in model_name:
        parts = model_name.split("checkpoint-")
        if len(parts) > 1:
            # Get the number part (may have trailing path)
            checkpoint_part = parts[-1].split("/")[0]
            # Try to extract just the number
            match = re.search(r'\d+', checkpoint_part)
            if match:
                return match.group()
    return None

def generate_scann_index_name(dataset_name: str, model_name: str) -> str:
    """Generate a programmatic index name from dataset, model, and checkpoint.
    
    Format: {sanitized_dataset}_{sanitized_model}_{checkpoint if exists}
    
    Args:
        dataset_name: Full dataset name (e.g., "beir/nfcorpus/test")
        model_name: Model path or name (e.g., "output/model/checkpoint-15000" or "robro612/xtr-base-en-pylate")
    
    Returns:
        Sanitized index name suitable for filesystem paths
    """
    sanitized_dataset = sanitize_dataset_name(dataset_name)
    sanitized_model = sanitize_model_name(model_name)
    
    checkpoint = extract_checkpoint_number(model_name)
    if checkpoint:
        return f"{sanitized_dataset}_{sanitized_model}_ckpt{checkpoint}"
    else:
        return f"{sanitized_dataset}_{sanitized_model}"


def expand_model_paths(model_paths: List[str]) -> List[str]:
    """Expand glob patterns in model paths."""
    expanded = []
    for path in model_paths:
        # Check if it's a glob pattern
        if '*' in path:
            matches = sorted(glob.glob(path))
            if not matches:
                print(f"Warning: No matches found for glob pattern: {path}")
            expanded.extend(matches)
        else:
            # Check if path exists (for local paths)
            if Path(path).exists():
                expanded.append(path)
            else:
                print(f"Warning: Model path may not exist: {path}")
                expanded.append(path)  # Still add it, might be a HuggingFace model name
    return expanded


def load_dataset(dataset_id: str, lowercase: bool = False) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """
    Load dataset from ir_datasets and return documents, queries, and qrels.
    
    Args:
        dataset_id: Full ir_datasets dataset ID (e.g., 'beir/nfcorpus/test')
        lowercase: If True, convert documents and queries to lowercase (default: False)
    
    Returns:
        Tuple of (documents, queries, qrels)
        - documents: List[Dict[str, str]] with 'id' and 'text' keys
        - queries: Dict[str, str] mapping query_id to query text
        - qrels: Dict[str, Dict[str, int]] mapping query_id to {doc_id: relevance}
    """
    print(f"Loading dataset: {dataset_id}")
    
    try:
        dataset = ir_datasets.load(dataset_id)
    except Exception as e:
        raise ValueError(f"Failed to load dataset '{dataset_id}': {e}. "
                        f"Make sure the dataset ID is correct and ir_datasets is installed.")
    
    # Check that dataset has required components
    if not dataset.has_docs():
        raise ValueError(f"Dataset '{dataset_id}' does not have documents")
    if not dataset.has_queries():
        raise ValueError(f"Dataset '{dataset_id}' does not have queries")
    
    # Load documents
    print("Loading documents...")
    if lowercase:
        print("Converting documents to lowercase...")
    documents = []
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents", unit="docs"):
        
        # Combine title and text if title exists, otherwise just use text
        if hasattr(doc, 'title') and doc.title:
            text = f"{doc.title}\n\n{doc.text}".strip()
        else:
            text = doc.text.strip()
        
        # Apply lowercase if requested
        if lowercase:
            text = text.lower()
        
        documents.append({
            "id": doc.doc_id,
            "text": text,
        })
    
    # Load queries
    print("Loading queries...")
    if lowercase:
        print("Converting queries to lowercase...")
    queries = {}
    for query in tqdm(dataset.queries_iter(), desc="Loading queries", unit="queries"):
        query_text = query.text.strip()
        
        # Apply lowercase if requested
        if lowercase:
            query_text = query_text.lower()
        
        queries[query.query_id] = query_text
    
    # Load qrels if available
    qrels = {}
    if dataset.has_qrels():
        print("Loading qrels...")
        for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels", unit="qrels"):
            relevance = int(qrel.relevance)
            
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = relevance
    else:
        print(f"Warning: Dataset '{dataset_id}' does not have qrels (relevance judgments)")
    
    print(f"Loaded {len(documents)} documents, {len(queries)} queries, {len(qrels)} queries with qrels")
    
    return documents, queries, qrels




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
    move_to_cpu: bool = False,
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
            
            # Load torch format (explicitly to CPU)
            packed = torch.load(shard_cache_file, map_location="cpu")
            shard_embeddings = unpack_embeddings(packed)
            # Ensure embeddings are on CPU (they should be already from map_location="cpu", but be explicit)
            if move_to_cpu:
                shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
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
            # Move to CPU immediately after encoding to free GPU memory
            if move_to_cpu:
                shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
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
    move_to_cpu: bool = False,
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
        # Ensure embeddings are on CPU (they should be already from map_location="cpu", but be explicit)
        if move_to_cpu:
            queries_embeddings = move_embeddings_to_cpu(queries_embeddings)
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
        # Move to CPU immediately after encoding to free GPU memory
        if move_to_cpu:
            queries_embeddings = move_embeddings_to_cpu(queries_embeddings)
        
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
    base_name = f"{dataset_name}_{model_name.replace('/', '_')}"
    # Generate programmatic index name for ScaNN
    scann_index_name = generate_scann_index_name(dataset_name, model_name)
    
    all_index_configs = [
        {
            "name": "ScaNN",
            "index_class": indexes.ScaNN,
            "init_kwargs": {
                "name": scann_index_name,
                "embedding_size": embedding_size,
                "num_neighbors": args.num_neighbors,
                "num_leaves": args.num_leaves,
                "num_leaves_to_search": args.num_leaves_to_search,
                "verbose": True,
                "use_autopilot": args.use_autopilot,
                "store_embeddings": True,  # Store embeddings for all indexes (so XTR indices can be used subsequently for ColBERT retrieval)
                "index_folder": "indexes" if args.save_index else None,  # Save indices in the indexes directory
                "override": False,  # Don't override existing indices, load them instead
            },
            "add_documents_kwargs": {
                "batch_size": args.encode_batch_size,
            },
        },
        {
            "name": "Flat",
            "index_class": indexes.Flat,
            "init_kwargs": {
                "name": f"{base_name}_flat",
                "embedding_size": embedding_size,
                "device": "cuda",  # Use GPU acceleration
                "search_batch_size": args.retrieve_batch_size,  # Batch size for search
                "verbose": False,
            },
            "add_documents_kwargs": {
                "batch_size": args.encode_batch_size,
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
    documents_ids: List[str],
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
    model_dtype: str,
    embedding_dtype: str,
    query_length: int,
    doc_length: int,
    lowercase: bool,
    retrieval_mode: str = "XTR",
    save_runfile: bool = False,
    encode_batch_size: int = 2000,
    retrieval_batch_size: int = 1,
    limit_queries: Optional[int] = None,
    limit_documents: Optional[int] = None,
    shard_size: Optional[int] = None,
    cache_embeddings: Optional[bool] = None,
    cache_dir: Optional[str] = None,
    move_embeddings_to_cpu_flag: Optional[bool] = None,
    save_index: Optional[bool] = None,
) -> None:
    """
    Test a single index: create, add documents, retrieve, evaluate, and save results.
    """
    index_name = config["name"]
    print("\n" + "="*80)
    print(f"Testing {index_name} index...")
    print("="*80)
    
    # Initialize index
    index : indexes.Base = config["index_class"](**config["init_kwargs"])

    retriever = retrieve.ColBERT(index=index, verbose=verbose)

    # # convert documents_embeddings to numpy array
    print(f"Converting documents_embeddings and queries_embeddings to {embedding_dtype}...")
    match embedding_dtype:
        case "bf16":
            from ml_dtypes import bfloat16
            # documents_embeddings = [emb.detach().cpu().view(torch.uint16).numpy().view(bfloat16) for emb in documents_embeddings]
            # queries_embeddings = [emb.detach().cpu().view(torch.uint16).numpy().view(bfloat16) for emb in queries_embeddings]
        case _:
            # documents_embeddings = [emb.to(dtype=get_torch_dtype(embedding_dtype)).detach().cpu().numpy() for emb in documents_embeddings]
            queries_embeddings = [emb.to(dtype=get_torch_dtype(embedding_dtype)).detach().cpu().numpy() for emb in queries_embeddings]


    # Add documents (skip if index was already loaded from disk)
    if hasattr(index, "_documents_added") and index._documents_added:
        print(f"{index_name} index already loaded from disk, skipping document addition")
        index_time = 0.0  # No time spent indexing since we loaded from disk

    else:
        print(f"Adding documents to {index_name} index...")
        start_time = time.time()
        add_kwargs = {
            "documents_ids": documents_ids,
            "documents_embeddings": documents_embeddings,
            **config["add_documents_kwargs"],
        }
        index.add_documents(**add_kwargs)
        index_time = time.time() - start_time
        print(f"{index_name} indexing time: {index_time:.2f} seconds")
    
    # Retrieve
    print(f"Retrieving with {index_name} using {retrieval_mode} mode...")
    start_time = time.time()
    if retrieval_mode == "ColBERT":
        # ColBERT-style retrieval (works for PLAID, ScaNN with store_embeddings=True, Flat, Voyager)
        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=k, k_token=k_token, batch_size=retrieval_batch_size)
    else:
        # XTR-style retrieval (works for all indexes)
        scores = retriever.retrieve_xtr(queries_embeddings=queries_embeddings, k=k, k_token=k_token, batch_size=retrieval_batch_size)
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
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "hit_rate@5"],
    )

    # Get timestamp for this evaluation run
    timestamp = datetime.now().isoformat()
    
    if save_runfile:
        # Create run_dict: {query_id: {doc_id: score, ...}, ...}
        run_dict = {
            query_id: {
                match["id"]: match["score"]
                for match in query_matches
            }
            for query_id, query_matches in zip(queries.keys(), scores)
        }

        run = Run(run=run_dict)
        checkpoint = extract_checkpoint_number(model_name)
        run.metadata = {
            "index_type": config["name"],
            **{k : v for k, v in config["init_kwargs"].items() if k != "index_class"},
            "dataset": dataset_name,
            "model": model_name,
            "checkpoint": checkpoint,
            "model_dtype": model_dtype,
            "embedding_dtype": embedding_dtype,
            "lowercase": lowercase,
            "query_length": query_length,
            "doc_length": doc_length,
            "k": k,
            "k_token": k_token,
            "retrieval_mode": retrieval_mode,
            "encode_batch_size": encode_batch_size,
            "retrieval_batch_size": retrieval_batch_size,
            "limit_queries": limit_queries,
            "limit_documents": limit_documents,
            "shard_size": shard_size,
            "cache_embeddings": cache_embeddings,
            "cache_dir": cache_dir,
            "move_embeddings_to_cpu": move_embeddings_to_cpu_flag,
            "save_index": save_index,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
            "timestamp": timestamp,
        }
        
        # Create runfiles subdirectory
        runfiles_dir = results_dir / "runs"
        runfiles_dir.mkdir(parents=True, exist_ok=True)
        
        # Generate descriptive filename: model_dataset_index_retrievalmode.json
        # Sanitize model name for filesystem
        sanitized_model_name = sanitize_model_name(model_name)
        if checkpoint:
            sanitized_model_name = f"{sanitized_model_name}_ckpt{checkpoint}"
        sanitized_dataset_name = sanitize_dataset_name(dataset_name)
        run_filename = (
            f"{sanitized_model_name}_{sanitized_dataset_name}_{index_name}_{retrieval_mode}"
            f"_{model_dtype}_{embedding_dtype}_qlen{query_length}_dlen{doc_length}.json"
        )
        run_filepath = runfiles_dir / run_filename
        
        # Save runfile
        run.save(run_filepath.as_posix())
        print(f"Runfile saved to: {run_filepath}")

    
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
        "model_dtype": model_dtype,
        "embedding_dtype": embedding_dtype,
        "query_length": query_length,
        "doc_length": doc_length,
        "lowercase": lowercase,
        "evaluation_scores": evaluation_scores,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "k": k,
        "k_token": k_token,
        "retrieval_mode": retrieval_mode,
        "encode_batch_size": encode_batch_size,
        "retrieval_batch_size": retrieval_batch_size,
        "timestamp": timestamp,
        "index_config": index_config_info,
    }
    
    # Append to index-specific JSONL file
    jsonl_file = Path(results_dir) / f"{index_name}.jsonl"
    with open(jsonl_file, "a") as f:
        f.write(json.dumps(jsonl_entry) + "\n")

    # Append to the dataset-specific results file
    dataset_results_file = Path(results_dir) / f"{sanitize_dataset_name(dataset_name)}.jsonl"
    with open(dataset_results_file, "a+") as f:
        f.write(json.dumps(jsonl_entry) + "\n")

    # also append to the overall results file
    overall_results_file = Path(results_dir).parent / "all_results.jsonl"
    with open(overall_results_file, "a+") as f:
        f.write(json.dumps(jsonl_entry) + "\n")

    print(f"\n{index_name} Results:")
    print(evaluation_scores)
    print(f"Results appended to: {jsonl_file}")
    print(f"Results appended to: {overall_results_file}")


def main() -> None:
    """Main function that orchestrates the evaluation process."""
    args = parse_arguments()
    
    # Convert model dtype and embedding dtype strings to torch dtypes
    model_dtype_torch = get_torch_dtype(args.model_dtype)
    embedding_dtype_torch = get_torch_dtype(args.embedding_dtype)
    print(f"Model dtype: {args.model_dtype} ({model_dtype_torch})")
    print(f"Embedding dtype: {args.embedding_dtype} ({embedding_dtype_torch})")
    
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
        
        # Determine query length: use override, then QUERY_LEN dict, then default 32
        if args.query_len is not None:
            query_length = args.query_len
            print(f"Using query length from --query_len: {query_length}")
        else:
            query_length = QUERY_LEN.get(dataset_name)
            if query_length is None:
                query_length = 32
                print(f"No query length configured for dataset '{dataset_name}' in QUERY_LEN dict, using default: {query_length}")
            else:
                print(f"Using query length from QUERY_LEN dict: {query_length}")
        
        # Use document length from args (default 300)
        doc_length = args.doc_len
        print(f"Using document length: {doc_length}")
        
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=doc_length,
            query_length=query_length,
        )
        model.compile()
        
        # Cast model to specified dtype with clear logging
        print("\n" + "="*80)
        print(f"MODEL CASTING: Checking model dtype before casting...")
        current_model_dtype = next(model.parameters()).dtype
        print(f"MODEL CASTING: Current model dtype: {current_model_dtype}")
        print(f"MODEL CASTING: Target model dtype: {model_dtype_torch} ({args.model_dtype})")
        if current_model_dtype != model_dtype_torch:
            print(f"MODEL CASTING: *** CASTING MODEL FROM {current_model_dtype} TO {model_dtype_torch} ({args.model_dtype}) ***")
            model = model.to(model_dtype_torch)
            # Verify casting occurred
            new_model_dtype = next(model.parameters()).dtype
            print(f"MODEL CASTING: Verification - Model dtype after casting: {new_model_dtype}")
            if new_model_dtype != model_dtype_torch:
                print(f"MODEL CASTING: WARNING - Model dtype mismatch! Expected {model_dtype_torch}, got {new_model_dtype}")
            else:
                print(f"MODEL CASTING: ✓ Model successfully cast to {model_dtype_torch} ({args.model_dtype})")
        else:
            print(f"MODEL CASTING: No casting needed - model already in {model_dtype_torch} ({args.model_dtype})")
        print("="*80 + "\n")

        # Load dataset (keep original dataset_id for loading)
        documents, queries, qrels = load_dataset(dataset_name, lowercase=args.lowercase)

        # Sanitize dataset name for file paths
        sanitized_dataset_name = sanitize_dataset_name(dataset_name)

        # Setup cache directory
        cache_subdir, cache_key, doc_embeddings_cache_file, query_embeddings_cache_file = setup_cache_directory(
            model_name=model_name,
            dataset_name=sanitized_dataset_name,
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
            batch_size=args.encode_batch_size,
            embedding_dtype=embedding_dtype_torch,
            move_to_cpu=args.move_embeddings_to_cpu,
        )

        # from here on in the loop we only need the documents_ids and queries
        documents_ids = [document["id"] for document in documents]
        del documents
        
        # Encode queries
        queries_embeddings = encode_queries(
            model=model,
            queries=queries,
            query_embeddings_cache_file=query_embeddings_cache_file,
            cache_embeddings=args.cache_embeddings,
            batch_size=args.encode_batch_size,
            embedding_dtype=embedding_dtype_torch,
            move_to_cpu=args.move_embeddings_to_cpu,
        )

        # Get embedding size from the model's final layer
        embedding_size = 128

        # Get index configurations (use sanitized name for file paths)
        index_configs = get_index_configs(
            dataset_name=sanitized_dataset_name,
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
        print(f"Number of documents: {len(documents_ids)}")
        print(f"Doc 1 embedding shape: {documents_embeddings[0].shape}")

        # Create results directory (use sanitized name)
        results_dir = Path("results") / sanitized_dataset_name
        results_dir.mkdir(parents=True, exist_ok=True)

        for config in index_configs:
            test_index(
                config=config,
                documents_ids=documents_ids,
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
                model_dtype=args.model_dtype,
                embedding_dtype=args.embedding_dtype,
                query_length=query_length,
                doc_length=doc_length,
                lowercase=args.lowercase,
                retrieval_mode=args.retrieval_mode,
                save_runfile=args.save_runfile,
                encode_batch_size=args.encode_batch_size,
                retrieval_batch_size=args.retrieve_batch_size,
                limit_queries=args.limit_queries,
                limit_documents=args.limit_documents,
                shard_size=args.shard_size,
                cache_embeddings=args.cache_embeddings,
                cache_dir=args.cache_dir,
                move_embeddings_to_cpu_flag=args.move_embeddings_to_cpu,
                save_index=args.save_index,
            )


if __name__ == "__main__":
    main()
