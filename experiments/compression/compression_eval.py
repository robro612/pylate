"""Compression evaluation script using Hydra configuration.

Adapts compression_experiment.py into the sample_eval.py framework:
- Uses Hydra for configuration management
- Uses ir_datasets for dataset loading
- Supports embedding caching
- Structured output with provenance tracking
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import platform
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import ir_datasets
import torch
import torch.nn.functional as F
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from ranx import Qrels, Run
from ranx import evaluate as ranx_evaluate
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pylate import evaluation, indexes, models, retrieve
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

logger = logging.getLogger(__name__)

# Query length mapping for different datasets (from both files)
QUERY_LEN = {
    # BEIR datasets
    "beir/nfcorpus/test": 32,
    "beir/fiqa/test": 32,
    "beir/scidocs": 48,
    "beir/scifact/test": 48,
    "beir/trec-covid": 48,
    "beir/webis-touche2020/v2": 32,
    "beir/quora/test": 32,
    "beir/nq": 32,
    # TREC
    "disks45/nocr/trec-robust-2004": 32,
    # LoTTE datasets
    "lotte/lifestyle/dev/forum": 32,
    "lotte/lifestyle/dev/search": 32,
    "lotte/lifestyle/test/forum": 32,
    "lotte/lifestyle/test/search": 32,
    "lotte/pooled/dev/forum": 32,
    "lotte/pooled/dev/search": 32,
    "lotte/pooled/test/forum": 32,
    "lotte/pooled/test/search": 32,
    "lotte/recreation/dev/forum": 32,
    "lotte/recreation/dev/search": 32,
    "lotte/recreation/test/forum": 32,
    "lotte/recreation/test/search": 32,
    "lotte/science/dev/forum": 32,
    "lotte/science/dev/search": 32,
    "lotte/science/test/forum": 32,
    "lotte/science/test/search": 32,
    "lotte/technology/dev/forum": 32,
    "lotte/technology/dev/search": 32,
    "lotte/technology/test/forum": 32,
    "lotte/technology/test/search": 32,
    "lotte/writing/dev/forum": 32,
    "lotte/writing/dev/search": 32,
    "lotte/writing/test/forum": 32,
    "lotte/writing/test/search": 32,
    # Short names (for backward compatibility with compression_experiment.py)
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
}


@dataclass(frozen=True)
class CachePaths:
    """Paths for embedding cache files."""
    cache_dir: Path
    doc_meta: Path
    query_meta: Path
    doc_shard_pattern: str
    query_file: Path
    doc_hash: str
    query_hash: str


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    """Convert string dtype to torch.dtype."""
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype_map[dtype_str]


def sanitize_dataset_name(dataset_name: str) -> str:
    """Sanitize dataset name for filesystem paths."""
    return dataset_name.replace("/", "_").replace(" ", "_").replace(":", "_").replace("\\", "_")


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for filesystem paths."""
    sanitized = model_name.split("output/")[-1].replace("/", "_")
    if "_checkpoint-" in sanitized:
        sanitized = sanitized.rsplit("_checkpoint-", 1)[0]
    return sanitized


def short_hash(payload: Dict[str, Any], length: int = 10) -> str:
    """Generate a short hash from a dictionary payload."""
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:length]


def resolve_query_length(dataset_id: str, override: Optional[int]) -> int:
    """Resolve query length from dataset ID or override."""
    if override is not None:
        return override
    return QUERY_LEN.get(dataset_id, 32)


def get_embedding_size(model: models.ColBERT) -> int:
    """Get the embedding dimension from a model."""
    if hasattr(model, "get_sentence_embedding_dimension"):
        return int(model.get_sentence_embedding_dimension())
    try:
        last = model[-1]
        if hasattr(last, "out_features"):
            return int(last.out_features)
    except Exception:
        pass
    return 128


def pack_embeddings(embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Pack variable-length embeddings into a single tensor with lengths."""
    if not embeddings:
        return {"embeddings": torch.empty(0), "lengths": torch.empty(0, dtype=torch.long)}
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
    concatenated = torch.cat(embeddings, dim=0)
    return {"embeddings": concatenated, "lengths": lengths}


def unpack_embeddings(packed: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
    """Unpack embeddings from packed format."""
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


def cast_embeddings(embeddings: List[torch.Tensor], dtype: torch.dtype) -> List[torch.Tensor]:
    """Cast embeddings to a specific dtype."""
    if not embeddings:
        return embeddings
    if embeddings[0].dtype == dtype:
        return embeddings
    return [emb.to(dtype) for emb in embeddings]


def move_embeddings_to_cpu(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    """Move embeddings to CPU."""
    return [emb.cpu() for emb in embeddings]


def load_dataset_irds(
    dataset_id: str, lowercase: bool = False
) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
    """Load dataset using ir_datasets."""
    logger.info("Loading dataset: %s", dataset_id)
    try:
        dataset = ir_datasets.load(dataset_id)
    except Exception as exc:
        raise ValueError(
            f"Failed to load dataset '{dataset_id}': {exc}. "
            "Make sure the dataset ID is correct and ir_datasets is installed."
        ) from exc

    if not dataset.has_docs():
        raise ValueError(f"Dataset '{dataset_id}' does not have documents")
    if not dataset.has_queries():
        raise ValueError(f"Dataset '{dataset_id}' does not have queries")

    documents: List[Dict[str, str]] = []
    logger.info("Loading documents...")
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents", unit="docs"):
        if hasattr(doc, "title") and doc.title:
            text = f"{doc.title}\n\n{doc.text}".strip()
        else:
            text = doc.text.strip()
        if lowercase:
            text = text.lower()
        documents.append({"id": doc.doc_id, "text": text})

    queries: Dict[str, str] = {}
    logger.info("Loading queries...")
    for query in tqdm(dataset.queries_iter(), desc="Loading queries", unit="queries"):
        query_text = query.text.strip()
        if lowercase:
            query_text = query_text.lower()
        queries[query.query_id] = query_text

    qrels: Dict[str, Dict[str, int]] = {}
    if dataset.has_qrels():
        logger.info("Loading qrels...")
        for qrel in tqdm(dataset.qrels_iter(), desc="Loading qrels", unit="qrels"):
            relevance = int(qrel.relevance)
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = relevance
    else:
        logger.warning("Dataset '%s' does not have qrels", dataset_id)

    logger.info(
        "Loaded %d documents, %d queries, %d queries with qrels",
        len(documents),
        len(queries),
        len(qrels),
    )
    return documents, queries, qrels


def build_cache_paths(
    cfg: DictConfig,
    dataset_id: str,
    model_name: str,
    doc_length: int,
    query_length: int,
    lowercase: bool,
) -> CachePaths:
    """Build cache paths for embeddings."""
    dataset_slug = sanitize_dataset_name(dataset_id)
    model_slug = sanitize_model_name(model_name)
    base_dir = (
        Path(cfg.cache.dir)
        / dataset_slug
        / model_slug
        / f"schema_v{cfg.cache.schema_version}"
        / f"doclen{doc_length}"
        / f"qlen{query_length}"
        / f"lower{int(lowercase)}"
        / f"edtype_{cfg.cache.embedding_dtype}"
        / f"mdtype_{cfg.model.dtype}"
    )
    doc_key = {
        "schema_version": cfg.cache.schema_version,
        "dataset": dataset_id,
        "model": model_name,
        "doc_length": doc_length,
        "lowercase": lowercase,
        "embedding_dtype": cfg.cache.embedding_dtype,
        "model_dtype": cfg.model.dtype,
    }
    query_key = {
        "schema_version": cfg.cache.schema_version,
        "dataset": dataset_id,
        "model": model_name,
        "query_length": query_length,
        "lowercase": lowercase,
        "embedding_dtype": cfg.cache.embedding_dtype,
        "model_dtype": cfg.model.dtype,
    }
    doc_hash = short_hash(doc_key, length=12)
    query_hash = short_hash(query_key, length=12)
    doc_meta = base_dir / f"doc_{doc_hash}_meta.json"
    query_meta = base_dir / f"query_{query_hash}_meta.json"
    doc_shard_pattern = f"doc_{doc_hash}_shard_{{:03d}}.pt"
    query_file = base_dir / f"query_{query_hash}.pt"
    return CachePaths(
        cache_dir=base_dir,
        doc_meta=doc_meta,
        query_meta=query_meta,
        doc_shard_pattern=doc_shard_pattern,
        query_file=query_file,
        doc_hash=doc_hash,
        query_hash=query_hash,
    )


def encode_documents_with_cache(
    model: models.ColBERT,
    documents: List[Dict[str, str]],
    batch_size: int,
    shard_size: int,
    embedding_dtype: torch.dtype,
    move_to_cpu: bool,
    cache_paths: CachePaths,
    cache_enabled: bool,
    return_artifacts: bool = False,
) -> Tuple[List[torch.Tensor], Optional[Dict[str, List[torch.Tensor]]]]:
    """Encode documents with caching support. Optionally return artifacts for compression."""
    num_documents = len(documents)
    num_shards = (num_documents + shard_size - 1) // shard_size
    cached_shards: Dict[int, Path] = {}

    if cache_enabled:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(num_shards):
            shard_path = cache_paths.cache_dir / cache_paths.doc_shard_pattern.format(shard_idx)
            if shard_path.exists():
                cached_shards[shard_idx] = shard_path

    if cached_shards:
        logger.info(
            "Found %d cached shards, will encode %d missing shards.",
            len(cached_shards),
            num_shards - len(cached_shards),
        )

    documents_embeddings: List[torch.Tensor] = []
    all_artifacts: Dict[str, List[torch.Tensor]] = {"input_ids": [], "attention_scores": []} if return_artifacts else {}

    for shard_idx in range(num_shards):
        if shard_idx in cached_shards:
            shard_cache_file = cached_shards[shard_idx]
            packed = torch.load(shard_cache_file, map_location="cpu")
            shard_embeddings = unpack_embeddings(packed)
            shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
            if move_to_cpu:
                shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
            documents_embeddings.extend(shard_embeddings)
            # Note: artifacts are not cached, so we can't return them for cached shards
            if return_artifacts:
                logger.warning("Artifacts not available for cached shard %d", shard_idx)
            continue

        start_idx = shard_idx * shard_size
        end_idx = min(start_idx + shard_size, num_documents)
        shard_documents = documents[start_idx:end_idx]
        logger.info(
            "Encoding shard %d/%d (documents %d to %d)",
            shard_idx + 1,
            num_shards,
            start_idx,
            end_idx - 1,
        )

        if return_artifacts:
            shard_embeddings, shard_artifacts = model.encode(
                sentences=[document["text"] for document in shard_documents],
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True,  # Still use normalized embeddings for compression
                return_extra_artifacts={"input_ids": True, "attention_scores": True},
            )
            # Move artifacts to CPU if needed (to match embeddings device)
            if move_to_cpu:
                for key in shard_artifacts:
                    shard_artifacts[key] = [
                        t.cpu() if hasattr(t, 'cpu') else t for t in shard_artifacts[key]
                    ]
            for key in all_artifacts:
                if key in shard_artifacts:
                    all_artifacts[key].extend(shard_artifacts[key])
        else:
            shard_embeddings = model.encode(
                sentences=[document["text"] for document in shard_documents],
                batch_size=batch_size,
                is_query=False,
                show_progress_bar=True,
                convert_to_tensor=True,
                normalize_embeddings=True, # Normalize embeddings for compression
            )

        shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
        if move_to_cpu:
            shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
        documents_embeddings.extend(shard_embeddings)

        if cache_enabled and not return_artifacts:
            # Only cache if not returning artifacts (normalized embeddings)
            shard_cache_file = cache_paths.cache_dir / cache_paths.doc_shard_pattern.format(shard_idx)
            packed = pack_embeddings(shard_embeddings)
            torch.save(packed, shard_cache_file)

    if cache_enabled and not return_artifacts:
        cache_paths.doc_meta.write_text(
            json.dumps(
                {
                    "num_documents": num_documents,
                    "num_shards": num_shards,
                    "shard_size": shard_size,
                    "created_at": datetime.now().isoformat(),
                },
                indent=2,
            )
            + "\n"
        )

    return documents_embeddings, all_artifacts if return_artifacts else None


def encode_queries_with_cache(
    model: models.ColBERT,
    queries: Dict[str, str],
    batch_size: int,
    embedding_dtype: torch.dtype,
    move_to_cpu: bool,
    cache_paths: CachePaths,
    cache_enabled: bool,
) -> List[torch.Tensor]:
    """Encode queries with caching support."""
    if cache_enabled and cache_paths.query_file.exists():
        packed = torch.load(cache_paths.query_file, map_location="cpu")
        query_embeddings = unpack_embeddings(packed)
        query_embeddings = cast_embeddings(query_embeddings, embedding_dtype)
        if move_to_cpu:
            query_embeddings = move_embeddings_to_cpu(query_embeddings)
        return query_embeddings

    logger.info("Encoding queries...")
    query_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=batch_size,
        convert_to_tensor=True,
        normalize_embeddings=True,
    )
    query_embeddings = cast_embeddings(query_embeddings, embedding_dtype)
    if move_to_cpu:
        query_embeddings = move_embeddings_to_cpu(query_embeddings)

    if cache_enabled:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        packed = pack_embeddings(query_embeddings)
        torch.save(packed, cache_paths.query_file)
        cache_paths.query_meta.write_text(
            json.dumps(
                {
                    "num_queries": len(queries),
                    "created_at": datetime.now().isoformat(),
                },
                indent=2,
            )
            + "\n"
        )
    return query_embeddings


def serialize_config_for_storage(config: Optional[CompressionConfig]) -> Dict[str, Any]:
    """Serialize a compression config (or baseline) for storage."""
    if config is None:
        return {"type": "baseline", "description": "No compression"}
    return config.serialize()


def scan_existing_results(results_dir: Path) -> Dict[str, Dict[str, Any]]:
    """Scan an existing results directory and build a map of config_name → result data.

    This enables content-based resume matching: even if config indices changed
    between runs (e.g. new keep_ratio values were added), we can still recognise
    previously-completed experiments by their description string.

    Returns:
        Dict mapping config_name (str) to the full evaluation dict loaded from
        ``config_N/evaluation.json``.
    """
    existing: Dict[str, Dict[str, Any]] = {}
    for config_dir in sorted(results_dir.glob("config_*")):
        eval_file = config_dir / "evaluation.json"
        if not eval_file.exists():
            continue
        try:
            with open(eval_file, "r") as f:
                data = json.load(f)
            name = data.get("config_name")
            if name:
                existing[name] = data
                logger.debug("Found existing result: %s (config_%s)", name, data.get("config_idx"))
        except Exception as exc:
            logger.warning("Could not load %s: %s", eval_file, exc)
    return existing


def load_configs_from_jsonl(jsonl_path: Path, model: models.ColBERT) -> List[Optional[CompressionConfig]]:
    """Load compression configurations from a JSONL file."""
    configs: List[Optional[CompressionConfig]] = []
    with open(jsonl_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if (data.get("type") == "baseline" or
                not data.get("strategies") or
                len(data.get("strategies", [])) == 0):
                configs.append(None)
            else:
                config = CompressionConfig.from_dict(data)
                for strategy in config.strategies:
                    if isinstance(strategy, IDFPruningStrategy) and strategy.config.ignore_token_ids is None:
                        strategy.config.ignore_token_ids = model.tokenizer.all_special_ids
                configs.append(config)
    return configs


def create_default_configs(model: models.ColBERT) -> List[Optional[CompressionConfig]]:
    """Create default compression configurations."""
    configs: List[Optional[CompressionConfig]] = [None]  # Baseline (no compression)

    # Random pruning and pooling baselines
    for keep_ratio in [0.1, 0.2, 0.33, 0.5, 0.75]:
        # Random pruning
        rand_prune_cfg = RandomPruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        configs.append(CompressionConfig(
            strategies=[RandomPruningStrategy(rand_prune_cfg)],
            description=f"Random pruning keep_ratio={keep_ratio}",
        ))

        # Random pooling
        rand_pool_cfg = RandomPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            seed=666,
        )
        configs.append(CompressionConfig(
            strategies=[RandomPoolingStrategy(rand_pool_cfg)],
            description=f"Random pooling keep_ratio={keep_ratio}",
        ))

        # Attention score pruning
        attention_config = AttentionPruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            track_pruned_tokens=False,
        )
        configs.append(CompressionConfig(
            strategies=[AttentionPruningStrategy(attention_config)],
            description=f"Attention score pruning keep_ratio={keep_ratio}",
        ))

        # Attention score pooling
        attention_pool_config = AttentionPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            show_progress_bar=True,
        )
        configs.append(CompressionConfig(
            strategies=[AttentionPoolingStrategy(attention_pool_config)],
            description=f"Attention score pooling keep_ratio={keep_ratio}",
        ))

        # Leverage score pruning
        leverage_config = LeverageScorePruningConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            track_pruned_tokens=False,
        )
        configs.append(CompressionConfig(
            strategies=[LeverageScorePruningStrategy(leverage_config)],
            description=f"Leverage score pruning keep_ratio={keep_ratio}",
        ))

        # Document-wise IDF pruning
        idf_pruning_config = IDFPruningConfig(
            mode="document",
            keep_ratio=keep_ratio,
            protected_tokens=1,
            ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
            use_tfidf=False,
            track_pruned_tokens=False,
        )
        configs.append(CompressionConfig(
            strategies=[IDFPruningStrategy(idf_pruning_config)],
            description=f"Doc-wise IDF pruning keep_ratio={keep_ratio}",
        ))

        # IDF Pooling
        idf_pooling_config = IDFPoolingConfig(
            keep_ratio=keep_ratio,
            protected_tokens=1,
            min_tokens=8,
            use_tfidf=False,
            ignore_token_ids=model.tokenizer.added_tokens_decoder.keys(),
            show_progress_bar=False,
        )
        configs.append(CompressionConfig(
            strategies=[IDFPoolingStrategy(idf_pooling_config)],
            description=f"IDF Pooling keep_ratio={keep_ratio}",
        ))

    # Clustering-based pooling
    for method in ["spherical", "hierarchical"]:
        for k in [1.333, 2, 3, 5, 10]:
            pooling_config = PoolingConfig(
                pool_factor=k,
                protected_tokens=1,
                clustering_method=method,
                show_progress_bar=True,
            )
            configs.append(CompressionConfig(
                strategies=[PoolingStrategy(pooling_config)],
                description=f"{method.capitalize()} Pooling f={k} protected tokens=1",
            ))

    return configs


def get_git_info() -> Dict[str, Any]:
    """Get git commit information."""
    info = {"commit": None, "dirty": None}
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        info["commit"] = commit
        dirty = subprocess.call(["git", "diff", "--quiet"]) != 0
        info["dirty"] = dirty
    except Exception:
        pass
    return info


def build_provenance(
    cfg: DictConfig,
    dataset_id: str,
    model_name: str,
    compression_config: Optional[CompressionConfig],
    run_id: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    """Build provenance metadata for a run."""
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_id,
        "model": model_name,
        "compression_config": serialize_config_for_storage(compression_config),
        "config": OmegaConf.to_container(cfg, resolve=True),
        "environment": {
            "python": sys.version,
            "platform": platform.platform(),
            "torch": torch.__version__,
            "cuda_available": torch.cuda.is_available(),
            "hostname": platform.node(),
        },
        "git": get_git_info(),
        "stats": stats,
    }


def apply_compression(
    embeddings: List[torch.Tensor],
    artifacts: Optional[Dict[str, List[torch.Tensor]]],
    config: Optional[CompressionConfig],
    batch_size: int,
) -> Tuple[List[torch.Tensor], float]:
    """Apply compression to embeddings. Returns compressed embeddings and compression time."""
    start_time = time.time()

    if config is None:
        # Baseline: just normalize the embeddings
        compressed = [F.normalize(emb, p=2, dim=-1) for emb in embeddings]
    else:
        compressor = config.create_compressor()
        compressed, _ = compressor.compress_parallel(
            embeddings=embeddings,
            artifacts=artifacts if artifacts else {},
            batch_size=batch_size,
            num_workers=8,
            show_progress=True,
        )
        # Normalize after compression
        compressed = [F.normalize(emb, p=2, dim=-1) for emb in compressed]

    compression_time = time.time() - start_time
    return compressed, compression_time


def evaluate_compression_config(
    cfg: DictConfig,
    config_idx: int,
    config: Optional[CompressionConfig],
    documents: List[Dict[str, str]],
    documents_embeddings: List[torch.Tensor],
    artifacts: Optional[Dict[str, List[torch.Tensor]]],
    queries: Dict[str, str],
    queries_embeddings: List[torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    dataset_id: str,
    model_name: str,
    results_dir: Path,
    run_id: str,
) -> Dict[str, Any]:
    """Evaluate a single compression configuration."""
    config_name = config.description if config else "Baseline"
    logger.info("")
    logger.info("=" * 80)
    logger.info("[%d] Evaluating: %s", config_idx, config_name)
    logger.info("=" * 80)

    # Apply compression
    compressed_embeddings, compression_time = apply_compression(
        embeddings=documents_embeddings,
        artifacts=artifacts,
        config=config,
        batch_size=cfg.encode.batch_size,
    )

    # Calculate token statistics
    num_tokens = sum(len(emb) for emb in compressed_embeddings)
    avg_tokens_per_doc = num_tokens / len(documents) if documents else 0
    logger.info("Token count: %d, Avg tokens/doc: %.1f", num_tokens, avg_tokens_per_doc)

    # Create index
    dataset_slug = sanitize_dataset_name(dataset_id)
    model_slug = sanitize_model_name(model_name)
    index_name = f"{dataset_slug}_{model_slug}_config_{config_idx}"

    # Wire PLAID parameters from config (with safe memory defaults)
    index_kwargs = dict(
        override=True,
        index_name=index_name,
        use_fast=cfg.index.use_fast,
        nbits=cfg.index.nbits,
        kmeans_niters=cfg.index.kmeans_niters,
        max_points_per_centroid=cfg.index.max_points_per_centroid,
        n_ivf_probe=cfg.index.n_ivf_probe,
        n_full_scores=cfg.index.n_full_scores,
        n_samples_kmeans=cfg.index.n_samples_kmeans,
        batch_size=cfg.index.batch_size,
        show_progress=True,
        device=cfg.index.device,
        use_triton=cfg.index.use_triton,
    )

    index = indexes.PLAID(**index_kwargs)

    # Add documents to index
    logger.info("Building index...")
    index_start = time.time()
    built_on_cpu_due_to_oom = False
    try:
        index.add_documents(
            documents_ids=[doc["id"] for doc in documents],
            documents_embeddings=compressed_embeddings,
        )
    except Exception as e:
        msg = str(e)
        oom_markers = [
            "CUDA out of memory",
            "cuda runtime error",
            "cudaErrorMemoryAllocation",
            "cudaMalloc",
            "out of memory",
        ]
        if any(m.lower() in msg.lower() for m in oom_markers) and getattr(cfg.index, "fallback_on_oom", True):
            logger.warning("Index build OOM on GPU. Retrying on CPU (this will be slower).")
            # Recreate a CPU-backed index and retry
            cpu_kwargs = dict(index_kwargs)
            cpu_kwargs["device"] = "cpu"
            cpu_kwargs["override"] = True
            index = indexes.PLAID(**cpu_kwargs)
            index.add_documents(
                documents_ids=[doc["id"] for doc in documents],
                documents_embeddings=compressed_embeddings,
            )
            built_on_cpu_due_to_oom = True
        else:
            raise
    index_time = time.time() - index_start

    # Retrieve
    # If we had to build on CPU due to OOM but a CUDA device is available, try to reload
    # the existing on-disk index for GPU search with smaller search params.
    search_index = index
    if built_on_cpu_due_to_oom and torch.cuda.is_available() and (
        cfg.index.device is None or (isinstance(cfg.index.device, str) and cfg.index.device.startswith("cuda"))
    ):
        try:
            logger.info("Reloading index on GPU just for retrieval (tighter search params).")
            gpu_kwargs = dict(index_kwargs)
            gpu_kwargs["override"] = False  # do not wipe the CPU-built index
            # Respect explicit device if provided; otherwise auto 'cuda'
            if cfg.index.device is None or cfg.index.device == "cpu":
                gpu_kwargs["device"] = "cuda"
            search_index = indexes.PLAID(**gpu_kwargs)
        except Exception as e:
            logger.warning("Failed to reload index on GPU for retrieval: %s. Continuing on CPU.", e)
            search_index = index

    logger.info("Retrieving...")
    retriever = retrieve.ColBERT(index=search_index)
    retrieve_start = time.time()
    try:
        scores = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=cfg.retrieve.k,
        )
    except Exception as e:
        msg = str(e)
        oom_markers = [
            "CUDA out of memory",
            "cuda runtime error",
            "cudaErrorMemoryAllocation",
            "cudaMalloc",
            "out of memory",
        ]
        if any(m.lower() in msg.lower() for m in oom_markers) and torch.cuda.is_available():
            logger.warning("Retrieval OOM on GPU. Retrying retrieval on CPU.")
            cpu_kwargs = dict(index_kwargs)
            cpu_kwargs["override"] = False
            cpu_kwargs["device"] = "cpu"
            cpu_index = indexes.PLAID(**cpu_kwargs)
            retriever = retrieve.ColBERT(index=cpu_index)
            scores = retriever.retrieve(
                queries_embeddings=queries_embeddings,
                k=cfg.retrieve.k,
            )
        else:
            raise
    retrieve_time = time.time() - retrieve_start

    # Remove query_id from scores (needed for some datasets like FiQA)
    for query_id, query_scores in zip(queries.keys(), scores):
        query_scores[:] = [score for score in query_scores if score["id"] != query_id]

    # Evaluate
    query_list = list(queries.keys())
    qrels_obj = Qrels(qrels=qrels)
    run_dict = {
        query: {match["id"]: match["score"] for match in query_matches}
        for query, query_matches in zip(query_list, scores)
    }
    run = Run(run=run_dict)

    evaluation_scores = ranx_evaluate(
        qrels=qrels_obj,
        run=run,
        metrics=list(cfg.metrics),
        make_comparable=True,
    )

    # Log results
    logger.info("Evaluation scores:")
    for metric, value in evaluation_scores.items():
        logger.info("  %s: %.4f", metric, value)

    # Build result entry
    result = {
        "config_idx": config_idx,
        "config_name": config_name,
        "config": serialize_config_for_storage(config),
        "token_count": num_tokens,
        "avg_tokens_per_doc": avg_tokens_per_doc,
        "compression_time": compression_time,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "evaluation": evaluation_scores,
    }

    # Save individual result
    config_run_dir = results_dir / f"config_{config_idx}"
    config_run_dir.mkdir(parents=True, exist_ok=True)

    provenance = build_provenance(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        compression_config=config,
        run_id=f"{run_id}_config_{config_idx}",
        stats={
            "num_documents": len(documents),
            "num_queries": len(queries),
            "token_count": num_tokens,
            "avg_tokens_per_doc": avg_tokens_per_doc,
            "compression_time": compression_time,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
        },
    )
    (config_run_dir / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    (config_run_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")

    # Save runfile if configured
    if cfg.output.save_runfile:
        run.save((config_run_dir / "runfile.json").as_posix())

    return result


def build_model(cfg: DictConfig, model_name: str, query_length: int, doc_length: int) -> models.ColBERT:
    """Build and configure the ColBERT model."""
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=doc_length,
        query_length=query_length,
        trust_remote_code=True,
    )
    if cfg.model.compile:
        model.compile()
    model_dtype = get_torch_dtype(cfg.model.dtype)
    current_dtype = next(model.parameters()).dtype
    if current_dtype != model_dtype:
        model = model.to(model_dtype)
    return model


@hydra.main(version_base=None, config_path="conf", config_name="compression_eval")
def main(cfg: DictConfig) -> None:
    """Main entry point for compression evaluation."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    overall_start = time.time()

    model_name = cfg.model.name_or_path
    dataset_id = cfg.dataset.name
    embedding_dtype = get_torch_dtype(cfg.cache.embedding_dtype)

    logger.info("Model: %s", model_name)
    logger.info("Dataset: %s", dataset_id)

    # Resolve query and document lengths
    query_length = resolve_query_length(dataset_id, cfg.model.get("query_len"))
    doc_length = cfg.model.doc_len

    # Build model
    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading model...")
    logger.info("=" * 80)
    model = build_model(cfg, model_name, query_length, doc_length)
    logger.info("Model loaded: query_length=%d, doc_length=%d", query_length, doc_length)

    # Load dataset
    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading dataset...")
    logger.info("=" * 80)
    documents, queries, qrels = load_dataset_irds(dataset_id, lowercase=cfg.dataset.lowercase)
    documents_ids = [doc["id"] for doc in documents]

    # Set up output directories
    # When resuming into an existing run directory, write results there instead of
    # the fresh Hydra output directory so everything stays in one place.
    resume_dir = cfg.compression.get("resume_dir", None)
    resume_mode = cfg.compression.get("resume", False)
    if resume_mode and resume_dir:
        results_dir = Path(resume_dir)
        logger.info("Resuming into existing results directory: %s", results_dir)
    else:
        hydra_cfg = HydraConfig.get()
        results_dir = Path(hydra_cfg.runtime.output_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    # Extract run_id from the output directory name (timestamp)
    run_id = results_dir.name
    logger.info("Output directory: %s", results_dir)
    logger.info("Run ID: %s", run_id)

    # Build cache paths
    cache_paths = build_cache_paths(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        doc_length=doc_length,
        query_length=query_length,
        lowercase=cfg.dataset.lowercase,
    )

    # Load or create compression configs
    logger.info("")
    logger.info("=" * 80)
    logger.info("Loading compression configurations...")
    logger.info("=" * 80)
    if cfg.compression.configs_file:
        configs = load_configs_from_jsonl(Path(cfg.compression.configs_file), model)
        logger.info("Loaded %d configs from %s", len(configs), cfg.compression.configs_file)
    else:
        configs = create_default_configs(model)
        logger.info("Created %d default configs", len(configs))

    for i, config in enumerate(configs):
        if config is None:
            logger.info("  [%d] Baseline (no compression)", i)
        else:
            logger.info("  [%d] %s", i, config.description)

    # Encode documents (with artifacts for compression)
    logger.info("")
    logger.info("=" * 80)
    logger.info("Encoding documents...")
    logger.info("=" * 80)
    encode_start = time.time()
    documents_embeddings, artifacts = encode_documents_with_cache(
        model=model,
        documents=documents,
        batch_size=cfg.encode.batch_size,
        shard_size=cfg.encode.shard_size,
        embedding_dtype=embedding_dtype,
        move_to_cpu=cfg.encode.move_embeddings_to_cpu,
        cache_paths=cache_paths,
        cache_enabled=False,  # Don't cache unnormalized embeddings
        return_artifacts=True,
    )
    encode_time = time.time() - encode_start
    logger.info("Encoded %d documents in %.2fs", len(documents_embeddings), encode_time)

    # Encode queries
    logger.info("")
    logger.info("=" * 80)
    logger.info("Encoding queries...")
    logger.info("=" * 80)
    query_encode_start = time.time()
    queries_embeddings = encode_queries_with_cache(
        model=model,
        queries=queries,
        batch_size=cfg.encode.batch_size,
        embedding_dtype=embedding_dtype,
        move_to_cpu=cfg.encode.move_embeddings_to_cpu,
        cache_paths=cache_paths,
        cache_enabled=cfg.cache.enable,
    )
    query_encode_time = time.time() - query_encode_start
    logger.info("Encoded %d queries in %.2fs", len(queries_embeddings), query_encode_time)

    # Evaluate each compression config
    logger.info("")
    logger.info("=" * 80)
    logger.info("EVALUATING COMPRESSION CONFIGS")
    logger.info("=" * 80)

    all_results: List[Dict[str, Any]] = []
    skip_count = cfg.compression.get("skip", 0)

    # Build name-based lookup of already-completed configs so we can match
    # even when config indices have changed between runs (e.g. new keep_ratio
    # values were added to create_default_configs).
    existing_by_name: Dict[str, Dict[str, Any]] = {}
    next_config_idx: Optional[int] = None  # next index for newly-run configs
    if resume_mode:
        existing_by_name = scan_existing_results(results_dir)
        logger.info("Found %d existing results in %s", len(existing_by_name), results_dir)
        # Determine the next available config index so new results are appended
        # after the last existing directory (avoids overwriting old results that
        # may have different index→config mappings).
        existing_indices = [
            int(d.name.split("_", 1)[1])
            for d in results_dir.glob("config_*")
            if d.is_dir() and d.name.split("_", 1)[1].isdigit()
        ]
        next_config_idx = (max(existing_indices) + 1) if existing_indices else 0
        logger.info("New configs will be numbered starting at config_%d", next_config_idx)

    with logging_redirect_tqdm():
        for config_idx, config in enumerate(configs):
            config_name = "Baseline" if config is None else config.description

            # Skip first N configs (simple skip)
            if config_idx < skip_count:
                logger.info("[%d] Skipping (skip=%d): %s", config_idx, skip_count, config_name)
                continue

            # Smart resumption: match by config *name* (content-based) so that
            # old runs with a different numbering scheme can still be resumed.
            if resume_mode and config_name in existing_by_name:
                logger.info("[%d] Skipping (already completed): %s", config_idx, config_name)
                all_results.append(existing_by_name[config_name])
                continue

            # When resuming, assign sequential indices after the last existing
            # directory so we never overwrite old results.
            if resume_mode and next_config_idx is not None:
                run_idx = next_config_idx
                next_config_idx += 1
            else:
                run_idx = config_idx

            result = evaluate_compression_config(
                cfg=cfg,
                config_idx=run_idx,
                config=config,
                documents=documents,
                documents_embeddings=documents_embeddings,
                artifacts=artifacts,
                queries=queries,
                queries_embeddings=queries_embeddings,
                qrels=qrels,
                dataset_id=dataset_id,
                model_name=model_name,
                results_dir=results_dir,
                run_id=run_id,
            )
            all_results.append(result)

            # Stream result to JSONL
            jsonl_path = results_dir / "results.jsonl"
            with open(jsonl_path, "a") as f:
                f.write(json.dumps(result, default=str) + "\n")

    total_time = time.time() - overall_start

    # Write summary
    logger.info("")
    logger.info("=" * 80)
    logger.info("SUMMARY")
    logger.info("=" * 80)
    logger.info("Total time: %.2fs", total_time)
    logger.info("Results saved to: %s", results_dir)

    # Write final metadata
    metadata = {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "model_name": model_name,
        "dataset_id": dataset_id,
        "num_documents": len(documents),
        "num_queries": len(queries),
        "num_configs": len(configs),
        "encode_time": encode_time,
        "query_encode_time": query_encode_time,
        "total_time": total_time,
        "config": OmegaConf.to_container(cfg, resolve=True),
    }
    (results_dir / "metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")

    # Print results table
    logger.info("")
    logger.info("%-50s %12s %12s %12s", "Config", "Tokens", "Avg/Doc", "NDCG@10")
    logger.info("-" * 90)
    for result in all_results:
        ndcg10 = result["evaluation"].get("ndcg@10", 0)
        logger.info(
            "%-50s %12d %12.1f %12.4f",
            result["config_name"][:50],
            result["token_count"],
            result["avg_tokens_per_doc"],
            ndcg10,
        )


if __name__ == "__main__":
    main()

