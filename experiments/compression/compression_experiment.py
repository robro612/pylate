"""Evaluation script for BEIR datasets with compression experiments."""

from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any
import time
from tqdm.autonotebook import tqdm
import torch
import torch.multiprocessing as mp

import pandas as pd

from pylate import evaluation, indexes, models, retrieve
from pylate.models import ColBERT
from pylate.models.ProxyAttentionColBERT import ProxyAttentionColBERT
from pylate.models.ConstBERT import ConstBERT
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


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """
    Convert string dtype to torch dtype.

    Parameters
    ----------
    dtype_str : str
        String representation of dtype ('fp32', 'fp16', 'bf16')

    Returns
    -------
    torch.dtype
        Corresponding torch dtype
    """
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Invalid dtype: {dtype_str}. Must be one of {list(dtype_map.keys())}")
    return dtype_map[dtype_str]


def _encode_worker(
    gpu_id: int,
    model_name: str,
    document_length: int,
    query_length: int,
    sentences: list[str],
    batch_size: int,
    result_queue: mp.Queue,
    worker_idx: int,
    model_type: str = "ColBERT",
    model_dtype: str = "fp32",
) -> None:
    """Worker function for multi-GPU encoding. Runs in a separate process."""
    # Set the device for this worker
    device = f"cuda:{gpu_id}"

    # Import and load model in this process
    from pylate import models
    from pylate.models.ConstBERT import ConstBERT

    # Convert dtype string to torch dtype
    torch_dtype = get_torch_dtype(model_dtype)

    if model_type == "ConstBERT":
        model = ConstBERT.load(
            path=model_name,
            document_length=document_length,
            model_kwargs={"torch_dtype": torch_dtype},
        )
        model = model.to(device)
    else:
        model = models.ColBERT(
            model_name_or_path=model_name,
            document_length=document_length,
            query_length=query_length,
            trust_remote_code=True,
            device=device,
            model_kwargs={"torch_dtype": torch_dtype},
        )

    # Encode documents
    # For ConstBERT, don't request attention scores since compression is via learned projection
    if model_type == "ConstBERT":
        extra_artifacts = {"input_ids": True, "attention_scores": False}
    else:
        extra_artifacts = {"input_ids": True, "attention_scores": True}

    embeddings, artifacts = model.encode(
        sentences=sentences,
        batch_size=batch_size,
        is_query=False,
        show_progress_bar=True,
        convert_to_tensor=False,  # Keep as numpy for pickling
        normalize_embeddings=False,
        return_extra_artifacts=extra_artifacts,
    )

    # Convert tensors to CPU numpy for pickling
    def to_numpy(t):
        if hasattr(t, 'cpu'):
            t = t.detach().cpu()
            # Convert bfloat16 to float32 (numpy doesn't support bfloat16)
            if t.dtype == torch.bfloat16:
                t = t.float()
            if hasattr(t, 'numpy'):
                return t.numpy()
        return t

    embeddings_np = [to_numpy(emb) for emb in embeddings]
    artifacts_np = {}
    for key, val in artifacts.items():
        artifacts_np[key] = [to_numpy(v) for v in val]

    result_queue.put((worker_idx, embeddings_np, artifacts_np))


def encode_multi_gpu(
    model_name: str,
    document_length: int,
    query_length: int,
    sentences: list[str],
    batch_size: int,
    num_gpus: int | None = None,
    model_type: str = "ColBERT",
    model_dtype: str = "fp32",
) -> tuple[list, dict]:
    """
    Encode documents using multiple GPUs in parallel.

    Parameters
    ----------
    model_name : str
        Model name/path
    document_length : int
        Maximum document length
    query_length : int
        Query length
    sentences : list[str]
        List of sentences to encode
    batch_size : int
        Batch size for encoding
    num_gpus : int | None
        Number of GPUs to use. If None, uses all available GPUs.
    model_type : str
        Type of model: "ColBERT" or "ConstBERT"
    model_dtype : str
        Model dtype: "fp32", "fp16", or "bf16"

    Returns
    -------
    tuple[list, dict]
        Embeddings and artifacts (same format as model.encode with return_extra_artifacts)
    """
    if num_gpus is None:
        num_gpus = torch.cuda.device_count()

    if num_gpus <= 1:
        # Fall back to single GPU encoding
        from pylate import models
        torch_dtype = get_torch_dtype(model_dtype)
        if model_type == "ConstBERT":
            model = ConstBERT.load(
                path=model_name,
                document_length=document_length,
                model_kwargs={"torch_dtype": torch_dtype},
            )
            extra_artifacts = {"input_ids": True, "attention_scores": False}
        else:
            model = models.ColBERT(
                model_name_or_path=model_name,
                document_length=document_length,
                query_length=query_length,
                trust_remote_code=True,
                model_kwargs={"torch_dtype": torch_dtype},
            )
            extra_artifacts = {"input_ids": True, "attention_scores": True}
        return model.encode(
            sentences=sentences,
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=False,
            return_extra_artifacts=extra_artifacts,
        )

    print(f"  Using {num_gpus} GPUs for parallel encoding")

    # Split sentences into chunks for each GPU
    chunk_size = (len(sentences) + num_gpus - 1) // num_gpus
    chunks = []
    for i in range(num_gpus):
        start_idx = i * chunk_size
        end_idx = min(start_idx + chunk_size, len(sentences))
        if start_idx < len(sentences):
            chunks.append(sentences[start_idx:end_idx])

    # Create result queue and spawn workers
    ctx = mp.get_context("spawn")
    result_queue = ctx.Queue()
    processes = []

    for i, chunk in enumerate(chunks):
        p = ctx.Process(
            target=_encode_worker,
            args=(i, model_name, document_length, query_length, chunk, batch_size, result_queue, i, model_type, model_dtype),
        )
        p.start()
        processes.append(p)

    # Collect results
    results = []
    for _ in range(len(chunks)):
        results.append(result_queue.get())

    # Wait for all processes to finish
    for p in processes:
        p.join()

    # Sort results by worker index and merge
    results.sort(key=lambda x: x[0])

    all_embeddings = []
    all_artifacts = {"input_ids": [], "attention_scores": []}

    for _, embeddings_np, artifacts_np in results:
        # Convert back to tensors
        all_embeddings.extend([torch.from_numpy(emb) for emb in embeddings_np])
        for key in all_artifacts:
            if key in artifacts_np:
                all_artifacts[key].extend([torch.from_numpy(v) for v in artifacts_np[key]])

    return all_embeddings, all_artifacts


def is_proxy_attention_model(model_path: str) -> bool:
    """
    Check if a model path contains a ProxyAttentionColBERT model.

    Checks for config.yaml with model.type == "proxy_attention" or
    config_sentence_transformers.json with proxy_attention_params.

    Parameters
    ----------
    model_path : str
        Path to the model directory

    Returns
    -------
    bool
        True if the model is a ProxyAttentionColBERT
    """
    model_path = Path(model_path)

    # Check config.yaml (training config)
    config_yaml_path = model_path / "config.yaml"
    if config_yaml_path.exists():
        try:
            import yaml
            with open(config_yaml_path, "r") as f:
                config = yaml.safe_load(f)
            if config.get("model", {}).get("type") == "proxy_attention":
                return True
        except Exception:
            pass

    # Check config_sentence_transformers.json (saved model config)
    config_st_path = model_path / "config_sentence_transformers.json"
    if config_st_path.exists():
        try:
            with open(config_st_path, "r") as f:
                config = json.load(f)
            if "proxy_attention_params" in config:
                return True
        except Exception:
            pass

    # Check for proxy_embeddings directory (definitive sign of ProxyAttentionColBERT)
    if (model_path / "proxy_embeddings").exists():
        return True

    return False


def get_proxy_attention_config(model_path: str) -> dict:
    """
    Get ProxyAttentionColBERT configuration from model path.

    Parameters
    ----------
    model_path : str
        Path to the model directory

    Returns
    -------
    dict
        Configuration with num_proxy_tokens, num_select_tokens, etc.
    """
    model_path = Path(model_path)

    # Try config.yaml first (training config)
    config_yaml_path = model_path / "config.yaml"
    if config_yaml_path.exists():
        try:
            import yaml
            with open(config_yaml_path, "r") as f:
                config = yaml.safe_load(f)
            variant_args = config.get("model", {}).get("variant_args", {})
            return {
                "num_proxy_tokens": variant_args.get("num_proxy_tokens", 32),
                "num_select_tokens": variant_args.get("num_select_tokens", 32),
                "use_cluster_pooling": variant_args.get("use_cluster_pooling", True),
                "proxy_tau": variant_args.get("proxy_tau", 1.0),
            }
        except Exception:
            pass

    # Try config_sentence_transformers.json
    config_st_path = model_path / "config_sentence_transformers.json"
    if config_st_path.exists():
        try:
            with open(config_st_path, "r") as f:
                config = json.load(f)
            proxy_params = config.get("proxy_attention_params", {})
            return {
                "num_proxy_tokens": proxy_params.get("num_proxy_tokens", 32),
                "num_select_tokens": proxy_params.get("num_select_tokens", 32),
                "use_cluster_pooling": proxy_params.get("use_cluster_pooling", True),
                "proxy_tau": proxy_params.get("proxy_tau", 1.0),
            }
        except Exception:
            pass

    # Default values
    return {
        "num_proxy_tokens": 32,
        "num_select_tokens": 32,
        "use_cluster_pooling": True,
        "proxy_tau": 1.0,
    }


def is_constbert_model(model_path: str) -> bool:
    """
    Check if a model path contains a ConstBERT model.

    Checks for config_sentence_transformers.json with model_type == "ConstBERT"
    or constbert_variant/constbert_seq_length parameters.

    Parameters
    ----------
    model_path : str
        Path to the model directory

    Returns
    -------
    bool
        True if the model is a ConstBERT model
    """
    model_path = Path(model_path)

    # Check config_sentence_transformers.json
    config_st_path = model_path / "config_sentence_transformers.json"
    if config_st_path.exists():
        try:
            with open(config_st_path, "r") as f:
                config = json.load(f)
            # Check for model_type == "ConstBERT"
            if config.get("model_type") == "ConstBERT":
                return True
            # Check for constbert_variant or constbert_seq_length
            if "constbert_variant" in config or "constbert_seq_length" in config:
                return True
        except Exception:
            pass

    # Check for constbert_projection directory (definitive sign of ConstBERT)
    if (model_path / "constbert_projection").exists():
        return True

    return False


def get_constbert_config(model_path: str) -> dict:
    """
    Get ConstBERT configuration from model path.

    Parameters
    ----------
    model_path : str
        Path to the model directory

    Returns
    -------
    dict
        Configuration with constbert_variant, constbert_seq_length, etc.
    """
    model_path = Path(model_path)

    # Try config_sentence_transformers.json
    config_st_path = model_path / "config_sentence_transformers.json"
    if config_st_path.exists():
        try:
            with open(config_st_path, "r") as f:
                config = json.load(f)
            return {
                "constbert_variant": config.get("constbert_variant", "flatten"),
                "constbert_seq_length": config.get("constbert_seq_length", 32),
                "document_length": config.get("document_length", 300),
            }
        except Exception:
            pass

    # Default values
    return {
        "constbert_variant": "flatten",
        "constbert_seq_length": 32,
        "document_length": 300,
    }


def load_model(
    model_name: str,
    dataset_name: str,
    document_length: int | None = None,
    num_select_tokens: int | None = None,
    model_dtype: str = "fp32",
) -> ColBERT | ProxyAttentionColBERT | ConstBERT:
    """
    Load and initialize the ColBERT, ProxyAttentionColBERT, or ConstBERT model.

    Parameters
    ----------
    model_name : str
        Name/path of the model to load
    dataset_name : str
        Dataset name to determine query length
    document_length : int | None
        Maximum document length. If None, uses model's max length.
    num_select_tokens : int | None
        For ProxyAttentionColBERT, override the number of tokens to select.
        If None, uses the model's default.
    model_dtype : str
        Model dtype: "fp32" (default), "fp16", or "bf16"

    Returns
    -------
    ColBERT | ProxyAttentionColBERT | ConstBERT
        Initialized model
    """
    print("\n" + "=" * 80)
    print("Loading model...")
    print("=" * 80)

    # Convert dtype string to torch dtype
    torch_dtype = get_torch_dtype(model_dtype)
    print(f"  Model dtype: {model_dtype} ({torch_dtype})")

    # Check if this is a ConstBERT model
    if is_constbert_model(model_name):
        print(f"  Detected ConstBERT model")
        constbert_config = get_constbert_config(model_name)
        print(f"  ConstBERT config: {constbert_config}")

        load_kwargs = {
            "document_length": document_length or 300,
            "model_kwargs": {"torch_dtype": torch_dtype},
        }
        model = ConstBERT.load(path=model_name, **load_kwargs)

        print(f"✓ Loaded ConstBERT: {model_name}")
        print(f"  Document length: {model.document_length}")
        print(f"  Query length: {model.query_length}")
        print(f"  Variant: {model.projection_variant}")
        print(f"  Output seq length: {model.constbert_seq_length}")

        return model

    # Check if this is a ProxyAttentionColBERT model
    if is_proxy_attention_model(model_name):
        print(f"  Detected ProxyAttentionColBERT model")
        proxy_config = get_proxy_attention_config(model_name)
        print(f"  Proxy config: {proxy_config}")

        # Override num_select_tokens if specified
        load_kwargs = {
            "document_length": document_length or 300,
            "model_kwargs": {"torch_dtype": torch_dtype},
        }
        if num_select_tokens is not None:
            load_kwargs["num_select_tokens"] = num_select_tokens
            print(f"  Overriding num_select_tokens: {num_select_tokens}")

        model = ProxyAttentionColBERT.load(path=model_name, **load_kwargs)

        print(f"✓ Loaded ProxyAttentionColBERT: {model_name}")
        print(f"  Document length: {model.document_length}")
        print(f"  Query length: {model.query_length}")
        print(f"  Num proxy tokens: {model.num_proxy_tokens}")
        print(f"  Num select tokens: {model.num_select_tokens}")

        return model

    # Standard ColBERT loading
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
        model_kwargs={"torch_dtype": torch_dtype},
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
            show_progress_bar=False,
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
                show_progress_bar=False,
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
    configs: list[CompressionConfig | None] | list[int],
) -> None:
    """
    Print experiment statistics.

    Parameters
    ----------
    stats : dict
        Statistics dictionary with keys: num_documents, encoding_time, config_token_counts, avg_tokens_per_doc
    configs : list
        List of compression configs (or list of int for ProxyAttentionColBERT num_select_tokens)
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
        # Handle both compression configs and proxy num_select_tokens (int)
        if isinstance(config, int):
            config_name = f"ProxyAttention num_select={config}"
        elif config is None:
            config_name = "Baseline"
        else:
            config_name = config.description
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


def evaluate_config_with_shards(
    config_idx: int,
    config: CompressionConfig | None,
    shard_files: list[Path],
    documents: list[dict],
    queries: dict,
    qrels: dict,
    queries_embeddings: list,
    dataset_name: str,
    model_name: str,
    index_type: str,
    stats: dict[str, Any],
    nbits: int = 2,
    compression_batch_size: int = 1000,
    global_idf_stats: tuple[dict[int, float], int] | None = None,
    metrics: list[str] | None = None,
    save_runfile: bool = False,
    runfile_output_dir: Path | None = None,
    run_id: str | None = None,
    save_retrieval_results: bool = False,
    retrieval_results_output_dir: Path | None = None,
) -> dict[str, Any]:
    """
    Evaluate a compression config using sharded inputs.

    Loads shards iteratively, compresses each shard, and builds the index incrementally.

    Parameters
    ----------
    config_idx : int
        Index of the configuration
    config : CompressionConfig | None
        Compression configuration (None for baseline)
    shard_files : list[Path]
        List of shard file paths
    documents : list[dict]
        Original documents (for document IDs)
    queries : dict
        Query dictionary
    qrels : dict
        Query relevance judgments
    queries_embeddings : list
        Encoded query embeddings
    dataset_name : str
        Dataset name
    model_name : str
        Model name
    index_type : str
        Type of index ("flat", "plaid", "scann", "faiss_ivfpq")
    stats : dict
        Experiment statistics
    compression_batch_size : int
        Batch size for compression (default: 1000)
    global_idf_stats : tuple[dict[int, float], int] | None
        Optional global IDF statistics (idf_scores, total_docs) to inject into artifacts
    metrics : list[str] | None
        Evaluation metrics
    save_runfile : bool
        Whether to save runfile
    runfile_output_dir : Path | None
        Directory to save runfiles
    run_id : str | None
        Run ID
    save_retrieval_results : bool
        Whether to save retrieval results
    retrieval_results_output_dir : Path | None
        Directory to save retrieval results

    Returns
    -------
    dict
        Evaluation results
    """
    config_name = config.description if config else "Baseline"
    print(f"\n[{config_idx}] Evaluating: {config_name}")
    print("-" * 80)

    # Check if index type supports incremental addition
    if index_type not in ["faiss_ivfpq", "plaid"]:
        raise ValueError(
            f"Sharded mode only supports 'faiss_ivfpq' and 'plaid' indexes. Got: {index_type}"
        )

    config_index_name = (
        f"{dataset_name}_{model_name.split('/')[-1]}_config_{config_idx}_index_{index_type}_run_id_{run_id}"
    )

    if index_type == "faiss_ivfpq":
        config_index = indexes.FaissIVFPQ(
            name=config_index_name,
            override=True,
            verbose_level="init",
            index_folder="indexes",
        )
    else:  # plaid
        config_index = indexes.PLAID(
            override=True,
            index_name=config_index_name,
            index_folder="indexes",
            nbits=nbits,
            n_samples_kmeans=250_000,
        )

    # Process shards iteratively
    total_tokens = 0
    total_docs = 0
    compression_time_total = 0

    print(f"Processing {len(shard_files)} shards...")
    for shard_idx, (embeddings_path, artifacts_path) in enumerate(tqdm(shard_files, desc="Processing shards")):
        # Load shard (both embeddings and artifacts)
        shard = load_shard(
            embeddings_path=embeddings_path,
            artifacts_path=artifacts_path,
            load_embeddings=True,
            load_artifacts=True,
        )
        shard_embeddings = shard["embeddings"]
        shard_doc_ids = shard["document_ids"]
        shard_artifacts = shard.get("artifacts", {})

        # Inject global IDF stats if provided
        if global_idf_stats is not None:
            idf_scores, total_docs = global_idf_stats
            shard_artifacts["global_idf_scores"] = idf_scores
            shard_artifacts["global_total_docs"] = total_docs

        # Apply compression if needed
        compression_start = time.time()
        if config is None:
            # Baseline: normalize embeddings
            import torch.nn.functional as F
            compressed_embeddings = torch.nn.functional.normalize(shard_embeddings, p=2, dim=-1)
        else:
            # Compress shard
            compressor = config.create_compressor()
            compressed_embeddings, _ = compressor.compress_parallel(
                embeddings=shard_embeddings,
                artifacts=shard_artifacts,
                batch_size=compression_batch_size,
                num_workers=None,
                show_progress=True,
            )
            # Normalize after compression
            import torch.nn.functional as F
            compressed_embeddings = [
                F.normalize(emb, p=2, dim=-1) for emb in compressed_embeddings
            ]

        compression_time_total += time.time() - compression_start

        # Add to index incrementally
        config_index.add_documents(
            documents_ids=shard_doc_ids,
            documents_embeddings=compressed_embeddings,
        )

        # Update statistics
        total_tokens += sum(len(emb) for emb in compressed_embeddings)
        total_docs += len(shard_doc_ids)

        # Free memory
        del shard, shard_embeddings, compressed_embeddings
        torch.cuda.empty_cache()

    # Finalize index (required for FaissIVFPQ)
    if index_type == "faiss_ivfpq":
        print("Finalizing FAISS index...")
        config_index.finalize()

    # Update stats
    avg_tokens_per_doc = total_tokens / total_docs if total_docs > 0 else 0
    stats["config_token_counts"].append(total_tokens)
    stats["avg_tokens_per_doc"].append(avg_tokens_per_doc)
    stats["compression_times"].append(compression_time_total)

    print(f"Total tokens: {total_tokens:,}")
    print(f"Avg tokens/doc: {avg_tokens_per_doc:.1f}")
    print(f"Compression time: {compression_time_total:.3f}s")

    # Retrieve
    print("Retrieving...")
    retriever = retrieve.ColBERT(index=config_index)
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=20)

    # Remove query_id from scores (needed for FiQA dataset)
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    # Evaluate
    if metrics is None:
        metrics = ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100", "mrr@10", "precision@10"]

    from ranx import Qrels, Run, evaluate as ranx_evaluate

    query_list = list(queries.keys())

    # Handle duplicate queries
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
        runfile_path = runfile_output_dir / f"run-{run_id}.config-{config_idx}.json"
        run.save(str(runfile_path), kind="json")
        print(f"Saved runfile to: {runfile_path}")

    # Save retrieval results if requested
    retrieval_results_path = None
    if save_retrieval_results and retrieval_results_output_dir is not None and run_id is not None:
        retrieval_results_path = retrieval_results_output_dir / f"retrieval-{run_id}.config-{config_idx}.json"
        retrieval_data = {
            "run_id": run_id,
            "config_idx": config_idx,
            "config_name": config_name,
            "dataset_name": dataset_name,
            "model_name": model_name,
            "num_queries": len(query_list),
            "k": 20,
            "results": {
                query_id: query_scores
                for query_id, query_scores in zip(query_list, scores)
            }
        }
        with open(retrieval_results_path, "w") as f:
            json.dump(retrieval_data, f, indent=2)
        print(f"Saved retrieval results to: {retrieval_results_path}")

    # Print results
    print("Evaluation scores:")
    for metric, value in evaluation_scores.items():
        print(f"  {metric}: {value:.4f}")

    return {
        "config_idx": config_idx,
        "config_name": config_name,
        "token_count": total_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "evaluation": evaluation_scores,
        "runfile_path": str(runfile_path) if runfile_path else None,
        "retrieval_results_path": str(retrieval_results_path) if retrieval_results_path else None,
    }


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
    plaid_nbits: int = 4,
    plaid_devices: list[str] | None = None,
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
        Type of index ("flat" or "plaid" or "scann" or "faiss_ivfpq")
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
    plaid_nbits : int, optional
        Number of bits for PLAID index quantization (default: 4)
    plaid_devices : list[str] | None, optional
        Devices for PLAID index (default: ["cuda"])

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
        f"{dataset_name}_{model_name.split('/')[-1]}_config_{config_idx}_index_{index_type}"
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
                nbits=plaid_nbits,
                devices=plaid_devices if plaid_devices is not None else ["cuda"],
            )
        case "scann":
            config_index = indexes.ScaNN(
                name=config_index_name,
                override=True,
                verbose_level="init",
                store_embeddings=True,  # Required for reranking
                index_folder="indexes",
            )
        case "faiss_ivfpq":
            config_index = indexes.FaissIVFPQ(
                name=config_index_name,
                override=True,
                verbose_level="all",
                index_folder="indexes",
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
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

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


def discover_shards(sharded_input_dir: Path) -> list[tuple[Path, Path]]:
    """
    Discover all shard files in a directory.

    Looks for pairs of files:
    - shard_NNNNNN_embeddings.pt
    - shard_NNNNNN_artifacts.pt

    Falls back to legacy format (shard_NNNNNN.pt) if new format not found.

    Parameters
    ----------
    sharded_input_dir : Path
        Directory containing shard files

    Returns
    -------
    list[tuple[Path, Path]]
        Sorted list of (embeddings_path, artifacts_path) tuples
    """
    # Try new format first (separate files)
    embeddings_files = sorted(sharded_input_dir.glob("shard_*_embeddings.pt"))

    if embeddings_files:
        # New format: separate embeddings and artifacts
        shard_pairs = []
        for emb_path in embeddings_files:
            # Extract shard index from filename
            shard_idx = emb_path.stem.replace("shard_", "").replace("_embeddings", "")
            artifacts_path = sharded_input_dir / f"shard_{shard_idx}_artifacts.pt"

            if not artifacts_path.exists():
                raise ValueError(f"Missing artifacts file for shard {shard_idx}: {artifacts_path}")

            shard_pairs.append((emb_path, artifacts_path))

        if not shard_pairs:
            raise ValueError(f"No shard files found in {sharded_input_dir}")

        return shard_pairs

    else:
        # Legacy format: single file with both embeddings and artifacts
        legacy_files = sorted(sharded_input_dir.glob("shard_*.pt"))
        if not legacy_files:
            raise ValueError(f"No shard files found in {sharded_input_dir}")

        # Return as (path, None) to indicate legacy format
        return [(f, None) for f in legacy_files]


def load_shard(
    embeddings_path: Path,
    artifacts_path: Path | None = None,
    load_embeddings: bool = True,
    load_artifacts: bool = True,
) -> dict[str, Any]:
    """
    Load a single shard from disk.

    Supports two formats:
    1. New format: Separate embeddings and artifacts files
    2. Legacy format: Single file with both (artifacts_path=None)

    Parameters
    ----------
    embeddings_path : Path
        Path to embeddings file (or legacy combined file)
    artifacts_path : Path | None
        Path to artifacts file (None for legacy format)
    load_embeddings : bool
        Whether to load embeddings (default: True)
    load_artifacts : bool
        Whether to load artifacts (default: True)

    Returns
    -------
    dict
        Shard data with requested components
    """
    result = {}

    if artifacts_path is None:
        # Legacy format: single file with everything
        shard = torch.load(embeddings_path, map_location="cpu")

        if load_embeddings:
            result["embeddings"] = shard.get("embeddings", [])
        if load_artifacts:
            result["artifacts"] = shard.get("artifacts", {})

        result["document_ids"] = shard.get("document_ids", [])
        result["metadata"] = shard.get("metadata", {})

        return result

    # New format: separate files
    if load_embeddings:
        emb_data = torch.load(embeddings_path, map_location="cpu")
        result["embeddings"] = emb_data["embeddings"]
        result["document_ids"] = emb_data["document_ids"]
        result["metadata"] = emb_data.get("metadata", {})

    if load_artifacts:
        art_data = torch.load(artifacts_path, map_location="cpu")
        result["artifacts"] = art_data["artifacts"]

        # If we didn't load embeddings, get metadata from artifacts file
        if not load_embeddings:
            result["document_ids"] = art_data["document_ids"]
            result["metadata"] = art_data.get("metadata", {})

    return result


def get_shard_stats(shard: dict[str, Any]) -> dict[str, int]:
    """
    Get statistics for a shard.

    Parameters
    ----------
    shard : dict
        Shard data

    Returns
    -------
    dict
        Statistics including num_documents and num_tokens
    """
    num_documents = len(shard["document_ids"])
    num_tokens = sum(len(emb) for emb in shard["embeddings"])
    return {
        "num_documents": num_documents,
        "num_tokens": num_tokens,
    }


def check_configs_need_global_idf(configs: list[CompressionConfig | None]) -> bool:
    """
    Check if any compression configs require global IDF statistics.

    Parameters
    ----------
    configs : list[CompressionConfig | None]
        List of compression configs

    Returns
    -------
    bool
        True if any config needs global IDF stats
    """
    for config in configs:
        if config is None:
            continue

        for strategy in config.strategies:
            strategy_type = type(strategy).__name__

            # IDF-based strategies
            if strategy_type == "IDFPruningStrategy":
                return True
            elif strategy_type == "IDFPoolingStrategy":
                return True
            # Importance-based strategies with IDF
            elif strategy_type in ["ImportancePruningStrategy", "ImportancePoolingStrategy"]:
                if hasattr(strategy.config, "use_idf") and strategy.config.use_idf:
                    return True
            # Hybrid strategies with IDF
            elif strategy_type == "HybridImportanceClusteringPoolingStrategy":
                if hasattr(strategy.config, "use_idf") and strategy.config.use_idf:
                    return True

    return False


def aggregate_idf_from_shard_metadata(
    shard_files: list[tuple[Path, Path]],
) -> tuple[dict[int, float], int]:
    """
    Aggregate global IDF statistics from shard metadata (pre-computed during encoding).

    This is much faster than gather_global_idf_stats() because it only loads tiny metadata
    instead of full artifacts.

    Parameters
    ----------
    shard_files : list[tuple[Path, Path]]
        List of (embeddings_path, artifacts_path) tuples

    Returns
    -------
    tuple[dict[int, float], int]
        (idf_scores, total_documents)
        - idf_scores: dict mapping token_id -> IDF score
        - total_documents: total number of documents
    """
    import math

    global_token_doc_freq = {}
    total_docs = 0

    print("\n" + "=" * 80)
    print("AGGREGATING GLOBAL IDF STATISTICS FROM SHARD METADATA")
    print("=" * 80)
    print(f"Processing {len(shard_files)} shards (metadata only)...")

    for embeddings_path, artifacts_path in tqdm(shard_files, desc="Aggregating IDF"):
        # Load only metadata from embeddings file (tiny)
        try:
            emb_data = torch.load(embeddings_path, map_location="cpu")
            metadata = emb_data.get("metadata", {})

            if "token_doc_freq" not in metadata:
                # Fallback: metadata doesn't have pre-computed stats, need to compute from artifacts
                print(f"\nWarning: Shard {embeddings_path.name} missing token_doc_freq in metadata.")
                print("Falling back to computing from artifacts...")
                return gather_global_idf_stats(shard_files)

            # Merge document frequencies
            for token_id, freq in metadata["token_doc_freq"].items():
                global_token_doc_freq[token_id] = global_token_doc_freq.get(token_id, 0) + freq

            total_docs += metadata["num_documents"]

            # Free memory
            del emb_data, metadata

        except Exception as e:
            print(f"\nError loading metadata from {embeddings_path}: {e}")
            print("Falling back to computing from artifacts...")
            return gather_global_idf_stats(shard_files)

    # Compute IDF scores
    idf_scores = {
        token_id: math.log(total_docs / freq)
        for token_id, freq in global_token_doc_freq.items()
    }

    print(f"✓ Aggregated IDF for {len(idf_scores):,} unique tokens")
    print(f"✓ Total documents: {total_docs:,}")

    return idf_scores, total_docs


def gather_global_idf_stats(
    shard_files: list[tuple[Path, Path]],
) -> tuple[dict[int, float], int]:
    """
    Gather global IDF statistics from all shards by loading artifacts.

    This is a fallback for shards that don't have pre-computed token_doc_freq in metadata.
    Use aggregate_idf_from_shard_metadata() instead when available (much faster).

    Parameters
    ----------
    shard_files : list[tuple[Path, Path]]
        List of (embeddings_path, artifacts_path) tuples

    Returns
    -------
    tuple[dict[int, float], int]
        (idf_scores, total_documents)
        - idf_scores: dict mapping token_id -> IDF score
        - total_documents: total number of documents
    """
    import math

    token_doc_freq = {}  # token_id -> number of docs containing it
    total_docs = 0

    print("\n" + "=" * 80)
    print("GATHERING GLOBAL IDF STATISTICS (from artifacts)")
    print("=" * 80)
    print(f"Processing {len(shard_files)} shards (artifacts only)...")

    for embeddings_path, artifacts_path in tqdm(shard_files, desc="Computing IDF"):
        # Load only artifacts (not embeddings)
        shard = load_shard(
            embeddings_path=embeddings_path,
            artifacts_path=artifacts_path,
            load_embeddings=False,
            load_artifacts=True,
        )

        artifacts = shard.get("artifacts", {})
        if "input_ids" not in artifacts:
            raise ValueError(
                f"Shard missing input_ids in artifacts. "
                f"IDF computation requires input_ids."
            )

        # Count unique tokens per document
        for input_ids in artifacts["input_ids"]:
            unique_tokens = torch.unique(input_ids)
            for token_id in unique_tokens.tolist():
                token_doc_freq[token_id] = token_doc_freq.get(token_id, 0) + 1
            total_docs += 1

        # Free memory
        del shard, artifacts
        torch.cuda.empty_cache()

    # Compute IDF scores
    idf_scores = {
        token_id: math.log(total_docs / freq)
        for token_id, freq in token_doc_freq.items()
    }

    print(f"✓ Computed IDF for {len(idf_scores):,} unique tokens")
    print(f"✓ Total documents: {total_docs:,}")

    return idf_scores, total_docs


def encode_and_save_shards(
    model: ColBERT | ProxyAttentionColBERT | ConstBERT,
    documents: list[dict],
    save_shards_dir: Path,
    shard_size: int = 250_000,
    batch_size: int = 1000,
    model_type: str = "ColBERT",
) -> list[Path]:
    """
    Encode documents in shards and save to disk.

    Supports resume: if shard files already exist, they are skipped. This allows
    resuming interrupted encoding jobs without re-encoding already completed shards.

    Parameters
    ----------
    model : ColBERT | ProxyAttentionColBERT | ConstBERT
        Model to use for encoding
    documents : list[dict]
        List of documents to encode
    save_shards_dir : Path
        Directory to save shards
    shard_size : int
        Number of documents per shard (default: 50000)
    batch_size : int
        Batch size for encoding (default: 1000)
    model_type : str
        Type of model: "ColBERT", "ProxyAttentionColBERT", or "ConstBERT"

    Returns
    -------
    list[Path]
        List of saved shard file paths (both newly encoded and pre-existing)
    """
    save_shards_dir.mkdir(parents=True, exist_ok=True)
    shard_files = []

    # Calculate number of shards
    num_shards = (len(documents) + shard_size - 1) // shard_size
    print(f"\nEncoding {len(documents):,} documents into {num_shards} shards")
    print(f"Shard size: {shard_size} documents")
    print(f"Encoding batch size: {batch_size}")

    # Track resume statistics
    num_skipped = 0
    num_encoded = 0

    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, len(documents))
        shard_documents = documents[start_idx:end_idx]

        # Check if shard already exists (for resume capability)
        embeddings_path = save_shards_dir / f"shard_{shard_idx:06d}_embeddings.pt"
        artifacts_path = save_shards_dir / f"shard_{shard_idx:06d}_artifacts.pt"

        embeddings_exists = embeddings_path.exists()
        artifacts_exists = artifacts_path.exists()

        if embeddings_exists and artifacts_exists:
            # Shard already exists - skip encoding
            print(f"\n[Shard {shard_idx + 1}/{num_shards}] ✓ Already exists, skipping (documents {start_idx:,} to {end_idx:,})")
            shard_files.append((embeddings_path, artifacts_path))
            num_skipped += 1
            continue
        elif embeddings_exists or artifacts_exists:
            # Incomplete shard (corrupted state) - warn and re-encode
            print(f"\n[Shard {shard_idx + 1}/{num_shards}] ⚠ Incomplete shard detected, re-encoding...")
            if embeddings_exists:
                embeddings_path.unlink()
            if artifacts_exists:
                artifacts_path.unlink()

        print(f"\n[Shard {shard_idx + 1}/{num_shards}] Encoding documents {start_idx:,} to {end_idx:,}...")

        # Encode documents in this shard
        # For ConstBERT, don't request attention scores
        if model_type == "ConstBERT":
            extra_artifacts = {"input_ids": True, "attention_scores": False}
        else:
            extra_artifacts = {"input_ids": True, "attention_scores": True}

        shard_embeddings, shard_artifacts = model.encode(
            sentences=[doc["text"] for doc in shard_documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
            normalize_embeddings=False,  # Keep unnormalized for compression
            return_extra_artifacts=extra_artifacts,
        )

        # Compute token document frequencies for this shard
        token_doc_freq = {}
        if "input_ids" in shard_artifacts:
            for input_ids in shard_artifacts["input_ids"]:
                unique_tokens = torch.unique(input_ids)
                for token_id in unique_tokens.tolist():
                    token_doc_freq[token_id] = token_doc_freq.get(token_id, 0) + 1

        # Prepare metadata
        document_ids = [doc["id"] for doc in shard_documents]
        metadata = {
            "shard_idx": shard_idx,
            "total_shards": num_shards,
            "start_idx": start_idx,
            "end_idx": end_idx,
            "num_documents": len(shard_documents),
            "num_tokens": sum(len(emb) for emb in shard_embeddings),
            "token_doc_freq": token_doc_freq,  # For IDF aggregation
        }

        # Save embeddings separately
        embeddings_path = save_shards_dir / f"shard_{shard_idx:06d}_embeddings.pt"
        embeddings_data = {
            "embeddings": shard_embeddings,
            "document_ids": document_ids,
            "metadata": metadata,
        }
        torch.save(embeddings_data, embeddings_path)

        # Save artifacts separately
        artifacts_path = save_shards_dir / f"shard_{shard_idx:06d}_artifacts.pt"
        artifacts_data = {
            "artifacts": shard_artifacts,
            "document_ids": document_ids,
            "metadata": metadata,
        }
        torch.save(artifacts_data, artifacts_path)

        shard_files.append((embeddings_path, artifacts_path))
        num_encoded += 1

        print(f"✓ Saved shard {shard_idx:06d}")
        print(f"  Embeddings: {embeddings_path}")
        print(f"  Artifacts: {artifacts_path}")
        print(f"  Documents: {len(shard_documents)}")
        print(f"  Tokens: {metadata['num_tokens']:,}")

        # Free memory
        del shard_embeddings, shard_artifacts, embeddings_data, artifacts_data
        torch.cuda.empty_cache()

    # Print encoding summary
    print(f"\n{'=' * 80}")
    print(f"ENCODING SUMMARY")
    print(f"{'=' * 80}")
    print(f"Total shards: {num_shards}")
    print(f"  Encoded: {num_encoded}")
    print(f"  Skipped (already exist): {num_skipped}")

    # Save manifest
    manifest_path = save_shards_dir / "manifest.json"
    manifest = {
        "format": "split",  # embeddings and artifacts in separate files
        "num_shards": num_shards,
        "total_documents": len(documents),
        "shard_size": shard_size,
        "model_type": model_type,
        "shard_files": [
            {
                "embeddings": str(emb_path.name),
                "artifacts": str(art_path.name),
            }
            for emb_path, art_path in shard_files
        ],
    }
    with open(manifest_path, "w") as f:
        json.dump(manifest, f, indent=2)

    print(f"\n✓ Saved {num_shards} shards to {save_shards_dir}")
    print(f"✓ Saved manifest to {manifest_path}")

    return shard_files


def _query_cache_path(cache_dir: Path) -> Path:
    return cache_dir / "queries_embeddings.pt"


def _idf_cache_path(shard_dir: Path) -> Path:
    return shard_dir / "idf_stats_cache.pt"


def load_idf_stats_cache(shard_dir: Path) -> tuple[dict[int, float], int] | None:
    cache_path = _idf_cache_path(shard_dir)
    if not cache_path.exists():
        return None
    try:
        cache_data = torch.load(cache_path, map_location="cpu")
        idf_scores = cache_data["idf_scores"]
        total_docs = cache_data["total_docs"]
        print(f"✓ Loaded cached IDF stats from {cache_path}")
        print(f"  Unique tokens: {len(idf_scores):,}")
        print(f"  Total documents: {total_docs:,}")
        return idf_scores, total_docs
    except Exception as exc:
        print(f"Warning: Failed to load IDF stats cache from {cache_path}: {exc}")
        return None


def save_idf_stats_cache(
    shard_dir: Path,
    idf_scores: dict[int, float],
    total_docs: int,
) -> None:
    cache_path = _idf_cache_path(shard_dir)
    torch.save({"idf_scores": idf_scores, "total_docs": total_docs}, cache_path)
    print(f"✓ Saved IDF stats cache to {cache_path}")


def _move_embeddings_to_cpu(embeddings: Any) -> Any:
    if torch.is_tensor(embeddings):
        return embeddings.detach().cpu()
    if isinstance(embeddings, list):
        return [emb.detach().cpu() if torch.is_tensor(emb) else emb for emb in embeddings]
    return embeddings


def load_query_embeddings_cache(
    cache_dir: Path,
    queries: dict,
    model_name: str,
    dataset_name: str,
    model_dtype: str,
    query_length: int,
) -> list | None:
    cache_path = _query_cache_path(cache_dir)
    if not cache_path.exists():
        return None

    try:
        cache_data = torch.load(cache_path, map_location="cpu")
    except Exception as exc:
        print(f"Warning: Failed to load cached queries from {cache_path}: {exc}")
        return None

    if not isinstance(cache_data, dict) or "embeddings" not in cache_data or "metadata" not in cache_data:
        print(f"Warning: Invalid query cache format in {cache_path}. Re-encoding queries.")
        return None

    metadata = cache_data.get("metadata", {})
    expected_metadata = {
        "model_name": model_name,
        "dataset_name": dataset_name,
        "model_dtype": model_dtype,
        "query_length": query_length,
        "num_queries": len(queries),
        "query_ids": list(queries.keys()),
    }
    for key, expected_value in expected_metadata.items():
        if metadata.get(key) != expected_value:
            print(f"Warning: Query cache metadata mismatch for '{key}'. Re-encoding queries.")
            return None

    return cache_data["embeddings"]


def save_query_embeddings_cache(
    cache_dir: Path,
    queries_embeddings: list,
    queries: dict,
    model_name: str,
    dataset_name: str,
    model_dtype: str,
    query_length: int,
) -> None:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_path = _query_cache_path(cache_dir)
    cache_data = {
        "embeddings": _move_embeddings_to_cpu(queries_embeddings),
        "metadata": {
            "model_name": model_name,
            "dataset_name": dataset_name,
            "model_dtype": model_dtype,
            "query_length": query_length,
            "num_queries": len(queries),
            "query_ids": list(queries.keys()),
        },
    }
    torch.save(cache_data, cache_path)


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
        choices=["flat", "plaid", "scann", "faiss_ivfpq"],
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
    parser.add_argument(
        "--multi_gpu",
        action="store_true",
        help="Enable multi-GPU encoding. Uses all available GPUs to encode documents in parallel.",
    )
    parser.add_argument(
        "--num_gpus",
        type=int,
        default=None,
        help="Number of GPUs to use for multi-GPU encoding. If not specified, uses all available GPUs.",
    )
    parser.add_argument(
        "--num_select_tokens",
        type=str,
        default=None,
        help="For ProxyAttentionColBERT: comma-separated list of num_select_tokens values to evaluate (e.g., '8,16,24,32'). If not specified, uses default values.",
    )
    parser.add_argument(
        "--model_dtype",
        type=str,
        default="fp32",
        choices=["fp32", "fp16", "bf16"],
        help="Model dtype for loading (default: 'fp32'). Options: 'fp32' (float32), 'fp16' (float16), 'bf16' (bfloat16).",
    )
    parser.add_argument(
        "--plaid_nbits",
        type=int,
        default=4,
        help="Number of bits for PLAID index quantization (default: 4)",
    )
    parser.add_argument(
        "--plaid_devices",
        type=str,
        nargs="+",
        default=["cuda"],
        help="Devices for PLAID index (default: ['cuda']). Can specify multiple devices.",
    )
    parser.add_argument(
        "--sharded_input_dir",
        type=str,
        default=None,
        help="Directory containing pre-encoded document shards (.pt files). If provided, loads embeddings from shards instead of encoding. Shards should be named shard_000000.pt, shard_000001.pt, etc.",
    )
    parser.add_argument(
        "--compression_batch_size",
        type=int,
        default=1000,
        help="Batch size for compression when using sharded input (default: 1000). Controls memory usage during compression.",
    )
    parser.add_argument(
        "--save_shards_dir",
        type=str,
        default=None,
        help="Directory to save encoded document shards. If provided, encodes documents in batches and saves to disk as .pt files (shard_000000.pt, etc.). These can later be loaded with --sharded_input_dir.",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=50000,
        help="Number of documents per shard when saving with --save_shards_dir (default: 50000).",
    )
    parser.add_argument(
        "--config_start",
        type=int,
        default=None,
        help="Start index (inclusive) of configs to evaluate (default: 0). "
             "Use with --config_end to run a slice of configs in parallel jobs.",
    )
    parser.add_argument(
        "--config_end",
        type=int,
        default=None,
        help="End index (exclusive) of configs to evaluate (default: all configs). "
             "Use with --config_start to run a slice of configs in parallel jobs.",
    )
    parser.add_argument(
        "--config_indices",
        type=int,
        nargs="+",
        default=None,
        help="Explicit list of config indices to evaluate (e.g. --config_indices 3 7 25 36). "
             "Cannot be used together with --config_start or --config_end.",
    )
    return parser.parse_args()


def main() -> None:
    """Main execution function."""
    args = parse_args()
    overall_start = time.time()

    # Validate mutual exclusivity of config selection arguments
    if args.config_indices is not None and (args.config_start is not None or args.config_end is not None):
        print("Error: --config_indices cannot be used together with --config_start or --config_end.")
        sys.exit(1)

    # Determine mode: encode-to-shards, load-shards, or in-memory
    save_shards = args.save_shards_dir is not None
    use_sharded_mode = args.sharded_input_dir is not None or save_shards
    shard_files = None

    if save_shards and args.sharded_input_dir is not None:
        print("Error: Cannot specify both --save_shards_dir and --sharded_input_dir")
        sys.exit(1)

    # Check for ProxyAttention/ConstBERT incompatibility with save_shards
    if save_shards:
        if is_proxy_attention_model(args.model_name):
            print("Error: --save_shards_dir is not compatible with ProxyAttentionColBERT models.")
            print("ProxyAttentionColBERT uses learned compression during encoding, not post-hoc compression configs.")
            sys.exit(1)
        if is_constbert_model(args.model_name):
            print("Error: --save_shards_dir is not compatible with ConstBERT models.")
            print("ConstBERT uses learned fixed-length projection during encoding, not post-hoc compression configs.")
            sys.exit(1)

    if use_sharded_mode:
        # Validate index type for sharded mode
        if args.index_type not in ["faiss_ivfpq", "plaid"]:
            print(f"Error: Sharded mode only supports 'faiss_ivfpq' and 'plaid' indexes. Got: {args.index_type}")
            sys.exit(1)

    if args.sharded_input_dir is not None:
        # Load existing shards
        sharded_input_dir = Path(args.sharded_input_dir)
        if not sharded_input_dir.exists():
            print(f"Error: Sharded input directory not found: {sharded_input_dir}")
            sys.exit(1)

        # Discover shard files
        shard_files = discover_shards(sharded_input_dir)
        print(f"\n✓ Found {len(shard_files)} shard files in {sharded_input_dir}")

        print(f"\n{'=' * 80}")
        print("SHARDED MODE: Loading pre-encoded shards")
        print(f"{'=' * 80}")
        print(f"Shard directory: {sharded_input_dir}")
        print(f"Number of shards: {len(shard_files)}")
        print(f"Index type: {args.index_type}")
        print(f"Compression batch size: {args.compression_batch_size}")

    # Check if this is a ProxyAttentionColBERT or ConstBERT model
    # For sharded loading (not saving), we don't need to check model type
    # For saving shards, we need to know the model type to encode properly
    if save_shards:
        # When saving shards, check model type to determine encoding behavior
        is_proxy_model = is_proxy_attention_model(args.model_name)
        is_constbert = is_constbert_model(args.model_name)
    elif use_sharded_mode:
        # When loading shards, model type doesn't matter for the main workflow
        is_proxy_model = False
        is_constbert = False
    else:
        # Normal in-memory mode
        is_proxy_model = is_proxy_attention_model(args.model_name)
        is_constbert = is_constbert_model(args.model_name)

    proxy_config = None
    constbert_config = None
    num_select_values = None

    if is_constbert:
        constbert_config = get_constbert_config(args.model_name)
        print("\n" + "=" * 80)
        print("CONSTBERT EXPERIMENT")
        print("=" * 80)
        print(f"Model: {args.model_name}")
        print(f"Variant: {constbert_config['constbert_variant']}")
        print(f"Fixed output seq length: {constbert_config['constbert_seq_length']}")
    elif is_proxy_model:
        proxy_config = get_proxy_attention_config(args.model_name)
        # Parse num_select_tokens values
        if args.num_select_tokens:
            num_select_values = [int(x.strip()) for x in args.num_select_tokens.split(",")]
        else:
            num_select_values = [4, 8, 12, 16, 20, 24, 28, 32]
        print("\n" + "=" * 80)
        print("PROXY ATTENTION COLBERT EXPERIMENT")
        print("=" * 80)
        print(f"Model: {args.model_name}")
        print(f"Testing num_select_tokens values: {num_select_values}")
        print(f"Num proxy tokens (fixed): {proxy_config['num_proxy_tokens']}")

    # Load dataset first (needed for both model types)
    documents, queries, qrels = load_dataset(args.dataset_name)

    # Load model (for standard ColBERT and ConstBERT, load once; for proxy, we'll reload per config)
    model = None
    if not is_proxy_model:
        model = load_model(args.model_name, args.dataset_name, args.document_length, model_dtype=args.model_dtype)

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
        if is_constbert:
            # For ConstBERT, create a descriptive name
            # Format: ConstBERT-{variant}-C{seq_length}
            variant = constbert_config.get("constbert_variant", "flatten")
            seq_len = constbert_config.get("constbert_seq_length", 32)
            model_dir = sanitize_name(f"ConstBERT-{variant}-C{seq_len}")
        elif is_proxy_model:
            # For ProxyAttentionColBERT, create a more descriptive name
            # Format: ProxyAttention-P{num_proxy}-{base_model_name}
            num_proxy = proxy_config.get("num_proxy_tokens", 32)
            # Try to extract base model name from the training config
            base_model_name = "GTE-ModernColBERT"  # default
            config_yaml_path = Path(args.model_name) / "config.yaml"
            if config_yaml_path.exists():
                try:
                    import yaml
                    with open(config_yaml_path, "r") as f:
                        train_config = yaml.safe_load(f)
                    base_model = train_config.get("model", {}).get("model_name_or_path", "")
                    if base_model:
                        # Extract short name from full path (e.g., "lightonai/GTE-ModernColBERT-v1" -> "GTE-ModernColBERT-v1")
                        base_model_name = base_model.split("/")[-1]
                except Exception:
                    pass
            model_dir = sanitize_name(f"ProxyAttention-P{num_proxy}-{base_model_name}")
        else:
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

    # Load compression configs (or create configs for ProxyAttentionColBERT/ConstBERT)
    print("\n" + "=" * 80)
    print("Loading compression configurations...")
    print("=" * 80)

    if is_constbert:
        # For ConstBERT, there's only one config - the fixed output length
        # No compression configs needed - model outputs fixed length directly
        configs = [None]  # Single baseline config
        print(f"✓ ConstBERT: Single config (fixed output length = {constbert_config['constbert_seq_length']})")
        print("\nConfigurations:")
        print(f"  [0] ConstBERT (fixed {constbert_config['constbert_seq_length']} tokens)")
    elif is_proxy_model:
        # For ProxyAttentionColBERT, configs are num_select_tokens values
        configs = num_select_values  # List of integers
        print(f"✓ Created {len(configs)} ProxyAttention configs (num_select_tokens values)")
        print("\nConfigurations:")
        for i, num_select in enumerate(configs):
            print(f"  [{i}] ProxyAttention num_select={num_select}")
    else:
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

    # Determine effective config selection for this job
    effective_indices = None  # None means use slice logic below
    effective_start = args.config_start if args.config_start is not None else 0
    effective_end = args.config_end
    if effective_end is not None:
        effective_end = min(effective_end, len(configs))

    if args.config_indices is not None:
        effective_indices = set(args.config_indices)
        invalid = effective_indices - set(range(len(configs)))
        if invalid:
            print(f"Error: --config_indices contains out-of-range indices: {sorted(invalid)} "
                  f"(valid range: 0-{len(configs) - 1})")
            sys.exit(1)
        print(f"\n✓ Config indices: {sorted(effective_indices)} ({len(effective_indices)} configs in this job)")
    elif effective_start > 0 or effective_end is not None:
        end_display = effective_end if effective_end is not None else len(configs)
        print(f"\n✓ Config slice: [{effective_start}, {end_display}) of {len(configs)} total configs")
        print(f"  Evaluating {end_display - effective_start} configs in this job")

    # For standard ColBERT and ConstBERT: encode documents once upfront
    # For ProxyAttentionColBERT: encoding happens per-config (inside the loop)
    # For sharded mode: skip encoding, load from shards
    documents_embeddings = None
    artifacts = None
    queries_embeddings = None
    encoding_time = 0
    query_encoding_time = 0

    if save_shards:
        # Encode and save shards mode
        print("\n" + "=" * 80)
        print("SHARDED MODE: Encoding and saving to shards")
        print("=" * 80)

        save_shards_dir = Path(args.save_shards_dir)
        print(f"Output directory: {save_shards_dir}")
        print(f"Shard size: {args.shard_size} documents")

        # Load model if needed
        if model is None:
            model = load_model(args.model_name, args.dataset_name, args.document_length, model_dtype=args.model_dtype)

        # Encode queries (with cache)
        print("\n" + "=" * 80)
        print("Encoding queries...")
        print("=" * 80)
        cache_path = _query_cache_path(save_shards_dir)
        queries_embeddings = load_query_embeddings_cache(
            cache_dir=save_shards_dir,
            queries=queries,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            model_dtype=args.model_dtype,
            query_length=model.query_length,
        )
        if queries_embeddings is not None:
            query_encoding_time = 0
            print(f"✓ Loaded cached queries from {cache_path}")
        else:
            query_encoding_start = time.time()
            queries_embeddings = model.encode(
                sentences=list(queries.values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=512,
                convert_to_tensor=True,
            )
            query_encoding_time = time.time() - query_encoding_start
            save_query_embeddings_cache(
                cache_dir=save_shards_dir,
                queries_embeddings=queries_embeddings,
                queries=queries,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                model_dtype=args.model_dtype,
                query_length=model.query_length,
            )
            print(f"✓ Encoded {len(queries_embeddings)} queries in {query_encoding_time:.3f}s")

        # Determine model type
        if is_constbert:
            encoding_model_type = "ConstBERT"
        elif is_proxy_model:
            encoding_model_type = "ProxyAttentionColBERT"
        else:
            encoding_model_type = "ColBERT"

        # Encode and save shards
        encoding_start = time.time()
        shard_files = encode_and_save_shards(
            model=model,
            documents=documents,
            save_shards_dir=save_shards_dir,
            shard_size=args.shard_size,
            batch_size=args.batch_size,
            model_type=encoding_model_type,
        )
        encoding_time = time.time() - encoding_start
        print(f"\n✓ Total encoding time: {encoding_time:.3f}s")

    elif use_sharded_mode:
        # Load existing shards mode (sharded_input_dir provided)
        print("\n" + "=" * 80)
        print("SHARDED MODE: Skipping document encoding")
        print("=" * 80)
        print("Documents will be loaded from shards during compression/indexing")

        # Still need to load model and encode queries
        if model is None:
            model = load_model(args.model_name, args.dataset_name, args.document_length, model_dtype=args.model_dtype)

        # Encode queries (with cache)
        print("\n" + "=" * 80)
        print("Encoding queries...")
        print("=" * 80)
        cache_dir = Path(args.sharded_input_dir)
        cache_path = _query_cache_path(cache_dir)
        queries_embeddings = load_query_embeddings_cache(
            cache_dir=cache_dir,
            queries=queries,
            model_name=args.model_name,
            dataset_name=args.dataset_name,
            model_dtype=args.model_dtype,
            query_length=model.query_length,
        )
        if queries_embeddings is not None:
            query_encoding_time = 0
            print(f"✓ Loaded cached queries from {cache_path}")
        else:
            query_encoding_start = time.time()
            queries_embeddings = model.encode(
                sentences=list(queries.values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=512,
                convert_to_tensor=True,
            )
            query_encoding_time = time.time() - query_encoding_start
            save_query_embeddings_cache(
                cache_dir=cache_dir,
                queries_embeddings=queries_embeddings,
                queries=queries,
                model_name=args.model_name,
                dataset_name=args.dataset_name,
                model_dtype=args.model_dtype,
                query_length=model.query_length,
            )
            print(f"✓ Encoded {len(queries_embeddings)} queries in {query_encoding_time:.3f}s")

    elif not is_proxy_model:  # This includes ConstBERT (encodes once)
        # Encode documents once with artifacts (input_ids needed for IDF pruning)
        # Use normalize_embeddings=False to get unnormalized embeddings for importance scoring
        print("\n" + "=" * 80)
        print("Encoding documents (unnormalized for importance scoring)...")
        print("=" * 80)
        encoding_start = time.time()

        if args.multi_gpu:
            # Multi-GPU encoding: spawn separate processes for each GPU
            # Determine model type for multi-GPU encoding
            if is_constbert:
                multi_gpu_model_type = "ConstBERT"
            else:
                multi_gpu_model_type = "ColBERT"
            documents_embeddings, artifacts = encode_multi_gpu(
                model_name=args.model_name,
                document_length=model.document_length,
                query_length=model.query_length,
                sentences=[document["text"] for document in documents],
                batch_size=args.batch_size,
                num_gpus=args.num_gpus,
                model_type=multi_gpu_model_type,
                model_dtype=args.model_dtype,
            )
        else:
            # Single GPU encoding
            # For ConstBERT, don't request attention scores since compression is via learned projection
            if is_constbert:
                single_gpu_artifacts = {"input_ids": True, "attention_scores": False}
            else:
                single_gpu_artifacts = {"input_ids": True, "attention_scores": True}
            documents_embeddings, artifacts = model.encode(
                sentences=[document["text"] for document in documents],
                batch_size=args.batch_size,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=False,  # Keep unnormalized for importance scoring
                return_extra_artifacts=single_gpu_artifacts,
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
        # Determine model type
        if is_constbert:
            model_type = "ConstBERT"
        elif is_proxy_model:
            model_type = "ProxyAttentionColBERT"
        else:
            model_type = "ColBERT"

        metadata_entry = {
            "type": "metadata",
            "run_id": run_id,
            "timestamp": datetime.now().isoformat(),
            "model_name": args.model_name,
            "model_type": model_type,
            "dataset_name": args.dataset_name,
            "num_documents": stats.get("num_documents"),
            "num_configs": len(configs),
            "model_dtype": args.model_dtype,
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
        }
        if is_constbert:
            metadata_entry["constbert_config"] = constbert_config
        elif is_proxy_model:
            metadata_entry["proxy_config"] = proxy_config
            metadata_entry["num_select_values"] = num_select_values
        else:
            metadata_entry["configs"] = [serialize_config_for_storage(config) for config in configs]
        with open(results_jsonl_path, "w") as f:
            f.write(json.dumps(metadata_entry, default=str) + "\n")

    # Check if we need to gather global IDF statistics (for sharded mode)
    global_idf_stats = None
    if use_sharded_mode and not is_proxy_model and not is_constbert:
        if check_configs_need_global_idf(configs):
            print("\n" + "=" * 80)
            print("GLOBAL IDF STATISTICS REQUIRED")
            print("=" * 80)
            print("Some compression configs require global IDF statistics.")
            shard_dir = shard_files[0][0].parent
            cached = load_idf_stats_cache(shard_dir)
            if cached is not None:
                global_idf_stats = cached
            else:
                print("Attempting to aggregate from shard metadata...")
                global_idf_stats = aggregate_idf_from_shard_metadata(shard_files)
                save_idf_stats_cache(shard_dir, global_idf_stats[0], global_idf_stats[1])

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
        # Skip configs outside the requested selection
        if effective_indices is not None:
            if config_idx not in effective_indices:
                # Add placeholder stats to maintain array alignment by config_idx
                stats["config_token_counts"].append(0)
                stats["avg_tokens_per_doc"].append(0)
                stats["compression_times"].append(0)
                continue
        else:
            # Slice mode: stop at effective_end, skip before effective_start
            if effective_end is not None and config_idx >= effective_end:
                break

            if config_idx < effective_start:
                if is_constbert:
                    config_name = f"ConstBERT (fixed {constbert_config['constbert_seq_length']} tokens)"
                elif is_proxy_model:
                    config_name = f"ProxyAttention num_select={config}"
                else:
                    config_name = "Baseline (no compression)" if config is None else config.description
                print(f"\n[{config_idx}] Skipping: {config_name}")
                # Add placeholder stats for skipped leading configs (maintains stats array alignment)
                stats["config_token_counts"].append(0)
                stats["avg_tokens_per_doc"].append(0)
                stats["compression_times"].append(0)
                continue

        compression_start = time.time()

        if is_constbert:
            # For ConstBERT: model encodes once with fixed output length
            # No compression - just use the pre-encoded embeddings
            config_name = f"ConstBERT (fixed {constbert_config['constbert_seq_length']} tokens)"
            print(f"\n[{config_idx}] Evaluating: {config_name}")

            # ConstBERT embeddings are already normalized during encoding
            compressed_embeddings = documents_embeddings
            compression_time = 0  # No compression step needed

        elif is_proxy_model:
            # For ProxyAttentionColBERT: config is num_select_tokens (int)
            num_select = config
            config_name = f"ProxyAttention num_select={num_select}"

            print(f"\n[{config_idx}] Evaluating: {config_name}")

            # Load model with this num_select_tokens value
            model = load_model(
                args.model_name,
                args.dataset_name,
                args.document_length,
                num_select_tokens=num_select,
                model_dtype=args.model_dtype,
            )

            # Encode documents (compression happens during encoding)
            print("\nEncoding documents...")
            encoding_start = time.time()
            current_documents_embeddings = model.encode(
                sentences=[document["text"] for document in documents],
                batch_size=args.batch_size,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
            )
            config_encoding_time = time.time() - encoding_start
            print(f"✓ Encoded {len(current_documents_embeddings)} documents in {config_encoding_time:.3f}s")

            # Update stats encoding time for first config
            if config_idx == 0 or stats["encoding_time"] == 0:
                stats["encoding_time"] = config_encoding_time

            # Encode queries (only once, reuse for all configs)
            if queries_embeddings is None:
                print("\nEncoding queries...")
                query_encoding_start = time.time()
                queries_embeddings = model.encode(
                    sentences=list(queries.values()),
                    is_query=True,
                    show_progress_bar=True,
                    batch_size=512,
                    convert_to_tensor=True,
                )
                stats["query_encoding_time"] = time.time() - query_encoding_start

            compressed_embeddings = current_documents_embeddings
            compression_time = config_encoding_time  # For proxy models, encoding IS compression

            # Clean up model to free GPU memory for next iteration
            del model
            torch.cuda.empty_cache()
        else:
            # Standard ColBERT: apply compression config to pre-encoded embeddings
            # Skip if in sharded mode (compression happens in evaluate_config_with_shards)
            if not use_sharded_mode:
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
                        num_workers=None,
                        show_progress=True,
                    )

                    # Normalize embeddings AFTER compression
                    import torch.nn.functional as F
                    compressed_embeddings = [
                        F.normalize(emb, p=2, dim=-1) for emb in compressed_embeddings
                    ]

                compression_time = time.time() - compression_start

                # Calculate token statistics (only for non-sharded mode)
                num_tokens = sum(len(emb) for emb in compressed_embeddings)
                avg_tokens_per_doc = num_tokens / len(documents) if documents else 0

                stats["config_token_counts"].append(num_tokens)
                stats["avg_tokens_per_doc"].append(avg_tokens_per_doc)
                stats["compression_times"].append(compression_time)
            else:
                # Sharded mode: compression and stats computed in evaluate_config_with_shards
                compressed_embeddings = None
                compression_time = 0

        # Evaluate this config
        if use_sharded_mode:
            # Sharded mode: load shards iteratively, compress, and index
            result = evaluate_config_with_shards(
                config_idx=config_idx,
                config=config,
                shard_files=shard_files,
                documents=documents,
                queries=queries,
                qrels=qrels,
                queries_embeddings=queries_embeddings,
                dataset_name=args.dataset_name,
                model_name=args.model_name,
                index_type=args.index_type,
                stats=stats,
                nbits=args.plaid_nbits,
                compression_batch_size=args.compression_batch_size,
                global_idf_stats=global_idf_stats,
                metrics=args.metrics,
                save_runfile=args.save_runfiles,
                runfile_output_dir=runfile_output_dir,
                run_id=run_id,
                save_retrieval_results=args.save_retrieval_results,
                retrieval_results_output_dir=retrieval_results_output_dir,
            )
        else:
            # Normal mode: use pre-loaded/compressed embeddings
            result = evaluate_config(
                config_idx=config_idx,
                config=None if is_proxy_model else config,  # Pass None for proxy models
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
                plaid_nbits=args.plaid_nbits,
                plaid_devices=args.plaid_devices,
            )

        # Override config_name for ConstBERT and proxy models
        if is_constbert or is_proxy_model:
            result["config_name"] = config_name

        all_evaluation_results.append(result)
        # Stream the result to disk immediately
        result_entry = {
            "type": "result",
            "run_id": run_id,
            "config_idx": result["config_idx"],
            "config_name": result["config_name"],
            "token_count": result["token_count"],
            "avg_tokens_per_doc": result["avg_tokens_per_doc"],
            "compression_time": stats["compression_times"][config_idx],
            "metrics": result["evaluation"],
            "runfile_path": result.get("runfile_path"),
        }
        if is_constbert:
            result_entry["constbert_variant"] = constbert_config["constbert_variant"]
            result_entry["constbert_seq_length"] = constbert_config["constbert_seq_length"]
        elif is_proxy_model:
            result_entry["num_select_tokens"] = configs[result["config_idx"]]
            result_entry["num_proxy_tokens"] = proxy_config["num_proxy_tokens"]
        else:
            result_entry["config"] = serialize_config_for_storage(configs[result["config_idx"]])
        with open(results_jsonl_path, "a") as f:
            f.write(json.dumps(result_entry, default=str) + "\n")

    stats["total_time"] = time.time() - overall_start

    # Print experiment statistics (only for configs that were processed)
    print_experiment_statistics(stats, configs[:len(stats["config_token_counts"])])

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
            result_entry = {
                "type": "result",
                "run_id": run_id,
                "config_idx": result["config_idx"],
                "config_name": result["config_name"],
                "token_count": result["token_count"],
                "avg_tokens_per_doc": result["avg_tokens_per_doc"],
                "compression_time": stats["compression_times"][result["config_idx"]],
                "metrics": result["evaluation"],
                "runfile_path": result.get("runfile_path"),
            }
            if is_proxy_model:
                result_entry["num_select_tokens"] = configs[result["config_idx"]]
                result_entry["num_proxy_tokens"] = proxy_config["num_proxy_tokens"]
            else:
                result_entry["config"] = serialize_config_for_storage(configs[result["config_idx"]])
            f.write(json.dumps(result_entry, default=str) + "\n")
    
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
