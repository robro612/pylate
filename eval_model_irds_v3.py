from __future__ import annotations

import hashlib
import inspect
import itertools
import json
import logging
import os
import platform
import glob
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import hydra
import ir_datasets
import torch
from omegaconf import DictConfig, OmegaConf
from ranx import Run
from tqdm.auto import tqdm
from tqdm.contrib.logging import logging_redirect_tqdm

from pylate import evaluation, indexes, models, retrieve

logger = logging.getLogger(__name__)

QUERY_LEN = {
    "beir/nfcorpus/test": 32,
    "beir/fiqa/test": 32,
    "beir/scidocs": 48,
    "beir/scifact/test": 48,
    "beir/trec-covid": 48,
    "beir/webis-touche2020/v2": 32,
    "beir/quora/test": 32,
    "beir/nq": 32,
    "disks45/nocr/trec-robust-2004": 32,
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
}


@dataclass(frozen=True)
class CachePaths:
    cache_dir: Path
    doc_meta: Path
    query_meta: Path
    doc_shard_pattern: str
    query_file: Path


def get_torch_dtype(dtype_str: str) -> torch.dtype:
    dtype_map = {
        "fp32": torch.float32,
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
    }
    if dtype_str not in dtype_map:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    return dtype_map[dtype_str]


def sanitize_dataset_name(dataset_name: str) -> str:
    return dataset_name.replace("/", "_")


def _find_max_checkpoint(training_dir: Path) -> int | None:
    """Find the maximum checkpoint number in a training output directory."""
    max_ckpt = None
    if not training_dir.exists():
        return None
    for item in training_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            try:
                ckpt_num = int(item.name.split("-")[1])
                if max_ckpt is None or ckpt_num > max_ckpt:
                    max_ckpt = ckpt_num
            except (ValueError, IndexError):
                continue
    return max_ckpt


def sanitize_model_name(model_name: str) -> str:
    """Sanitize model name for use in file paths.

    Converts:
      - output/exp_name/checkpoint-N -> exp_name_ckptN
      - output/exp_name/final -> exp_name_ckpt{max} (with warning)
      - huggingface/model-name -> huggingface_model-name (unchanged)
    """
    import re

    sanitized = model_name.split("output/")[-1].replace("/", "_")

    # Handle checkpoint-N -> _ckptN
    if "_checkpoint-" in sanitized:
        match = re.search(r"_checkpoint-(\d+)", sanitized)
        if match:
            ckpt_num = match.group(1)
            sanitized = sanitized.replace(f"_checkpoint-{ckpt_num}", f"_ckpt{ckpt_num}")

    # Handle _final -> _ckpt{max} by looking up the training directory
    elif sanitized.endswith("_final"):
        exp_name = sanitized[:-6]  # Remove "_final"
        # Try to find the training directory and max checkpoint
        for output_base in [Path("output"), Path("/exp/rjha/pylate-xtr/output")]:
            training_dir = output_base / exp_name
            max_ckpt = _find_max_checkpoint(training_dir)
            if max_ckpt:
                logger.warning(
                    "Using /final is deprecated. Resolving %s/final to checkpoint-%d. "
                    "Please use explicit checkpoint paths.",
                    exp_name, max_ckpt
                )
                sanitized = f"{exp_name}_ckpt{max_ckpt}"
                break

    return sanitized


def extract_checkpoint_number(model_name: str) -> str | None:
    import re
    if "checkpoint-" in model_name:
        parts = model_name.split("checkpoint-")
        if len(parts) > 1:
            checkpoint_part = parts[-1].split("/")[0]
            match = re.search(r"\d+", checkpoint_part)
            if match:
                return match.group()
    return None


def short_hash(payload: Dict[str, Any], length: int = 10) -> str:
    raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
    return hashlib.md5(raw).hexdigest()[:length]


def resolve_query_length(dataset_id: str, override: Optional[int]) -> int:
    if override is not None:
        return override
    return QUERY_LEN.get(dataset_id, 32)


def expand_model_paths(model_paths: Iterable[str]) -> List[str]:
    expanded: List[str] = []
    for path in model_paths:
        if any(ch in path for ch in ["*", "?", "["]):
            matches = sorted(glob.glob(path))
            if not matches:
                logger.warning("No matches found for glob pattern: %s", path)
            expanded.extend(matches)
        else:
            expanded.append(path)
    return expanded


def get_embedding_size(model: models.ColBERT) -> int:
    if hasattr(model, "get_sentence_embedding_dimension"):
        emb_dim = model.get_sentence_embedding_dimension()
        if emb_dim is not None:
            return int(emb_dim)
    try:
        last = model[-1]
        if hasattr(last, "out_features"):
            return int(last.out_features)
    except Exception:
        pass
    return 128


def pack_embeddings(embeddings: List[torch.Tensor]) -> Dict[str, torch.Tensor]:
    if not embeddings:
        return {"embeddings": torch.empty(0), "lengths": torch.empty(0, dtype=torch.long)}
    lengths = torch.tensor([emb.shape[0] for emb in embeddings], dtype=torch.long)
    concatenated = torch.cat(embeddings, dim=0)
    return {"embeddings": concatenated, "lengths": lengths}


def unpack_embeddings(packed: Dict[str, torch.Tensor]) -> List[torch.Tensor]:
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
    if not embeddings:
        return embeddings
    if embeddings[0].dtype == dtype:
        return embeddings
    return [emb.to(dtype) for emb in embeddings]


def move_embeddings_to_cpu(embeddings: List[torch.Tensor]) -> List[torch.Tensor]:
    return [emb.cpu() for emb in embeddings]


def load_dataset(
    dataset_id: str, lowercase: bool = False
) -> Tuple[List[Dict[str, str]], Dict[str, str], Dict[str, Dict[str, int]]]:
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
    # Schema v3: use static filenames instead of hashes
    doc_meta = base_dir / "doc_meta.json"
    query_meta = base_dir / "query_meta.json"
    doc_shard_pattern = "doc_shard_{:03d}.pt"
    query_file = base_dir / "query.pt"
    return CachePaths(
        cache_dir=base_dir,
        doc_meta=doc_meta,
        query_meta=query_meta,
        doc_shard_pattern=doc_shard_pattern,
        query_file=query_file,
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
) -> List[torch.Tensor]:
    num_documents = len(documents)
    num_shards = (num_documents + shard_size - 1) // shard_size
    cached_shards: Dict[int, Path] = {}
    if cache_enabled:
        cache_paths.cache_dir.mkdir(parents=True, exist_ok=True)
        for shard_idx in range(num_shards):
            shard_path = cache_paths.cache_dir / cache_paths.doc_shard_pattern.format(
                shard_idx
            )
            if shard_path.exists():
                cached_shards[shard_idx] = shard_path

    if cached_shards:
        logger.info(
            "Found %d cached shards, will encode %d missing shards.",
            len(cached_shards),
            num_shards - len(cached_shards),
        )

    documents_embeddings: List[torch.Tensor] = []
    for shard_idx in range(num_shards):
        if shard_idx in cached_shards:
            shard_cache_file = cached_shards[shard_idx]
            packed = torch.load(shard_cache_file, map_location="cpu")
            shard_embeddings = unpack_embeddings(packed)
            shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
            if move_to_cpu:
                shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
            documents_embeddings.extend(shard_embeddings)
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
        shard_embeddings = model.encode(
            sentences=[document["text"] for document in shard_documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
            convert_to_tensor=True,
        )
        shard_embeddings = cast_embeddings(shard_embeddings, embedding_dtype)
        if move_to_cpu:
            shard_embeddings = move_embeddings_to_cpu(shard_embeddings)
        documents_embeddings.extend(shard_embeddings)

        if cache_enabled:
            shard_cache_file = cache_paths.cache_dir / cache_paths.doc_shard_pattern.format(
                shard_idx
            )
            packed = pack_embeddings(shard_embeddings)
            torch.save(packed, shard_cache_file)

    if cache_enabled:
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

    return documents_embeddings


def encode_queries_with_cache(
    model: models.ColBERT,
    queries: Dict[str, str],
    batch_size: int,
    embedding_dtype: torch.dtype,
    move_to_cpu: bool,
    cache_paths: CachePaths,
    cache_enabled: bool,
) -> List[torch.Tensor]:
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


def build_index_configs(
    cfg: DictConfig,
    dataset_slug: str,
    model_name: str,
    embedding_size: int,
    doc_length: int,
    lowercase: bool,
) -> List[Dict[str, Any]]:
    model_slug = sanitize_model_name(model_name)
    index_configs: List[Dict[str, Any]] = []
    index_types = cfg.index.types
    # Schema v3: use config params in path instead of hash
    index_root = (
        Path(cfg.index.folder)
        / dataset_slug
        / model_slug
        / f"schema_v{cfg.index.schema_version}"
        / f"doclen{doc_length}"
        / f"lower{int(lowercase)}"
        / f"edtype_{cfg.cache.embedding_dtype}"
        / f"mdtype_{cfg.model.dtype}"
    )
    load_dir = Path(cfg.index.load_dir) if cfg.index.load_dir else None

    if "ScaNN" in index_types:
        scann_verbose_level = cfg.index.scann.verbose_level
        # Schema v3: use simple folder name instead of hash
        scann_name = "index"
        scann_folder = (index_root / "scann").as_posix()
        if load_dir is not None:
            scann_name = load_dir.name
            scann_folder = load_dir.parent.as_posix()
        index_configs.append(
            {
                "name": "ScaNN",
                "index_class": indexes.ScaNN,
                "init_kwargs": {
                    "name": scann_name,
                    "embedding_size": embedding_size,
                    "num_neighbors": cfg.index.scann.num_neighbors,
                    "num_leaves": cfg.index.scann.num_leaves,
                    "num_leaves_to_search": cfg.index.scann.num_leaves_to_search,
                    "verbose_level": scann_verbose_level,
                    "use_autopilot": cfg.index.scann.use_autopilot,
                    "store_embeddings": cfg.index.scann.store_embeddings,
                    "index_folder": scann_folder if cfg.index.save or load_dir else None,
                    "override": cfg.index.override,
                },
                "add_documents_kwargs": {"batch_size": cfg.encode.batch_size},
            }
        )

    if "Flat" in index_types:
        # Schema v3: use simple folder name instead of hash
        index_configs.append(
            {
                "name": "Flat",
                "index_class": indexes.Flat,
                "init_kwargs": {
                    "name": "index",
                    "embedding_size": embedding_size,
                    "device": cfg.index.flat.device,
                    "search_batch_size": cfg.index.flat.search_batch_size,
                    "verbose": cfg.index.flat.verbose,
                },
                "add_documents_kwargs": {"batch_size": cfg.encode.batch_size},
            }
        )

    if "Voyager" in index_types:
        # Schema v3: use simple folder name instead of hash
        voyager_folder = (index_root / "voyager").as_posix()
        index_configs.append(
            {
                "name": "Voyager",
                "index_class": indexes.Voyager,
                "init_kwargs": {
                    "index_folder": voyager_folder,
                    "index_name": "index",
                    "override": cfg.index.voyager.override,
                    "embedding_size": embedding_size,
                    "M": cfg.index.voyager.M,
                    "ef_construction": cfg.index.voyager.ef_construction,
                    "ef_search": cfg.index.voyager.ef_search,
                },
                "add_documents_kwargs": {},
            }
        )

    if "PLAID" in index_types:
        # Schema v3: use simple folder name instead of hash
        plaid_folder = (index_root / "plaid").as_posix()
        plaid_name = "index"
        if load_dir is not None:
            plaid_name = load_dir.name
            plaid_folder = load_dir.parent.as_posix()
        index_configs.append(
            {
                "name": "PLAID",
                "index_class": indexes.PLAID,
                "init_kwargs": {
                    "override": cfg.index.plaid.override,
                    "index_folder": plaid_folder if cfg.index.save or load_dir else "indexes",
                    "index_name": plaid_name,
                },
                "add_documents_kwargs": {},
            }
        )

    return index_configs


def get_git_info() -> Dict[str, Any]:
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
    index_name: str,
    retrieval_mode: str,
    run_id: str,
    stats: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "run_id": run_id,
        "timestamp": datetime.now().isoformat(),
        "dataset": dataset_id,
        "model": model_name,
        "index": index_name,
        "retrieval_mode": retrieval_mode,
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


def iter_retrieval_configs(cfg: DictConfig) -> Iterable[DictConfig]:
    if cfg.retrieve.configs:
        for entry in cfg.retrieve.configs:
            merged = OmegaConf.merge(cfg.retrieve, entry)
            yield merged
    else:
        yield cfg.retrieve


def evaluate_index(
    cfg: DictConfig,
    retrieve_cfg: DictConfig,
    index_config: Dict[str, Any],
    index_instance: indexes.Base,
    queries: Dict[str, str],
    queries_embeddings: List[torch.Tensor],
    qrels: Dict[str, Dict[str, int]],
    dataset_id: str,
    model_name: str,
    query_length: int,
    doc_length: int,
    results_dir: Path,
    index_time: float = 0.0,
) -> None:
    """Run retrieval and evaluation on an already-built index."""
    index_name = index_config["name"]
    retriever = retrieve.ColBERT(index=index_instance, verbose=retrieve_cfg.verbose)

    # Verify index is ready
    if not getattr(index_instance, "_documents_added", False):
        raise RuntimeError(
            f"Index {index_name} has no documents. Run with 'build_index' stage first."
        )

    logger.info("")
    logger.info("--- Retrieve (%s) ---", index_name)
    logger.info(
        "Retrieving using mode=%s k=%s k_token=%s",
        retrieve_cfg.mode,
        retrieve_cfg.k,
        retrieve_cfg.k_token,
    )
    start_time = time.time()
    if retrieve_cfg.mode == "ColBERT":
        scores = retriever.retrieve(
            queries_embeddings=queries_embeddings,
            k=retrieve_cfg.k,
            k_token=retrieve_cfg.k_token,
            batch_size=retrieve_cfg.batch_size,
        )
    else:
        imputation_cfg = getattr(retrieve_cfg, "imputation", None)
        imputation_kwargs = {}
        if imputation_cfg is not None:
            imputation_kwargs["imputation"] = getattr(imputation_cfg, "method", "min")
            imputation_kwargs["percentile"] = getattr(imputation_cfg, "percentile", 10.0)
            imputation_kwargs["power_law_multiplier"] = getattr(
                imputation_cfg, "power_law_multiplier", 100.0
            )
        scores = retriever.retrieve_xtr(
            queries_embeddings=queries_embeddings,
            k=retrieve_cfg.k,
            k_token=retrieve_cfg.k_token,
            batch_size=retrieve_cfg.batch_size,
            **imputation_kwargs,
        )
    retrieve_time = time.time() - start_time

    for query_id, query_scores in zip(queries.keys(), scores):
        query_scores[:] = [score for score in query_scores if score["id"] != query_id]

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=cfg.metrics,
    )

    timestamp = datetime.now().isoformat()

    imputation_info = {}
    if hasattr(retrieve_cfg, "imputation") and retrieve_cfg.imputation is not None:
        imputation_info = {
            "method": getattr(retrieve_cfg.imputation, "method", "min"),
            "percentile": getattr(retrieve_cfg.imputation, "percentile", 10.0),
            "power_law_multiplier": getattr(
                retrieve_cfg.imputation, "power_law_multiplier", 100.0
            ),
        }

    run_id_payload = {
        "dataset": dataset_id,
        "model": model_name,
        "index": index_name,
        "index_params": index_config.get("init_kwargs", {}),
        "retrieve": {
            "mode": retrieve_cfg.mode,
            "k": retrieve_cfg.k,
            "k_token": retrieve_cfg.k_token,
            "batch_size": retrieve_cfg.batch_size,
            "imputation": imputation_info,
        },
        "model_dtype": cfg.model.dtype,
        "embedding_dtype": cfg.cache.embedding_dtype,
        "query_length": query_length,
        "doc_length": doc_length,
        "lowercase": cfg.dataset.lowercase,
    }
    run_id = short_hash(run_id_payload, length=10)

    dataset_slug = sanitize_dataset_name(dataset_id)
    model_slug = sanitize_model_name(model_name)
    run_base = (
        f"{model_slug}_{dataset_slug}_{index_name}_{retrieve_cfg.mode}_{run_id}"
    )

    run_dir = Path(cfg.output.run_root) / run_base
    run_dir.mkdir(parents=True, exist_ok=True)
    provenance_path = run_dir / "provenance.json"

    provenance = build_provenance(
        cfg=cfg,
        dataset_id=dataset_id,
        model_name=model_name,
        index_name=index_name,
        retrieval_mode=retrieve_cfg.mode,
        run_id=run_id,
        stats={
            "num_documents": len(getattr(index_instance, "doc_id_to_embedding_range", {})),
            "num_queries": len(queries),
            "index_time": index_time,
            "retrieve_time": retrieve_time,
        },
    )
    provenance_path.write_text(json.dumps(provenance, indent=2) + "\n")

    if cfg.output.save_runfile:
        run_dict = {
            query_id: {match["id"]: match["score"] for match in query_matches}
            for query_id, query_matches in zip(queries.keys(), scores)
        }
        run = Run(run=run_dict)
        run.metadata = {
            "run_id": run_id,
            "dataset": dataset_id,
            "model": model_name,
            "index_type": index_name,
            "index_params": index_config.get("init_kwargs", {}),
            "checkpoint": extract_checkpoint_number(model_name),
            "model_dtype": cfg.model.dtype,
            "embedding_dtype": cfg.cache.embedding_dtype,
            "lowercase": cfg.dataset.lowercase,
            "query_length": query_length,
            "doc_length": doc_length,
            "k": retrieve_cfg.k,
            "k_token": retrieve_cfg.k_token,
            "retrieval_mode": retrieve_cfg.mode,
            "imputation": imputation_info if imputation_info else None,
            "encode_batch_size": cfg.encode.batch_size,
            "retrieval_batch_size": retrieve_cfg.batch_size,
            "shard_size": cfg.encode.shard_size,
            "cache_docs": cfg.cache.enable_docs,
            "cache_queries": cfg.cache.enable_queries,
            "cache_dir": cfg.cache.dir,
            "move_embeddings_to_cpu": cfg.encode.move_embeddings_to_cpu,
            "save_index": cfg.index.save,
            "index_time": index_time,
            "retrieve_time": retrieve_time,
            "timestamp": timestamp,
            "provenance_file": provenance_path.as_posix(),
        }

        run_filepath = run_dir / "runfile.json"
        run.save(run_filepath.as_posix())
        logger.info("Runfile saved to: %s", run_filepath)

    run_summary = {
        "dataset": dataset_id,
        "model": model_name,
        "model_dtype": cfg.model.dtype,
        "embedding_dtype": cfg.cache.embedding_dtype,
        "query_length": query_length,
        "doc_length": doc_length,
        "lowercase": cfg.dataset.lowercase,
        "evaluation_scores": evaluation_scores,
        "index_time": index_time,
        "retrieve_time": retrieve_time,
        "k": retrieve_cfg.k,
        "k_token": retrieve_cfg.k_token,
        "retrieval_mode": retrieve_cfg.mode,
        "imputation": imputation_info if imputation_info else None,
        "encode_batch_size": cfg.encode.batch_size,
        "retrieval_batch_size": retrieve_cfg.batch_size,
        "timestamp": timestamp,
        "run_id": run_id,
        "run_dir": run_dir.as_posix(),
        "provenance_file": provenance_path.as_posix(),
    }
    (run_dir / "evaluation.json").write_text(json.dumps(run_summary, indent=2) + "\n")

    if cfg.output.save_jsonl:
        index_config_info = {
            "name": index_config["name"],
            "init_kwargs": index_config["init_kwargs"],
            **(
                {"add_documents_kwargs": index_config.get("add_documents_kwargs")}
                if "add_documents_kwargs" in index_config
                else {}
            ),
        }
        jsonl_entry = {**run_summary, "index_config": index_config_info}

        jsonl_file = results_dir / f"{index_name}.jsonl"
        with open(jsonl_file, "a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

        dataset_results_file = results_dir / f"{dataset_slug}.jsonl"
        with open(dataset_results_file, "a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

        overall_results_file = Path(cfg.output.results_dir) / cfg.output.overall_results_file
        with open(overall_results_file, "a") as f:
            f.write(json.dumps(jsonl_entry) + "\n")

        logger.info("\n" + "="*80)
        logger.info("Results for index: %s", index_name)
        logger.info("="*80)
        formatted_scores = {
            metric: round(float(value * 100), 1) for metric, value in evaluation_scores.items()
        }
        logger.info("Evaluation scores:\n%s\n%s", json.dumps(formatted_scores, indent=2), "="*80)


def build_model(cfg: DictConfig, model_name: str, query_length: int, doc_length: int) -> models.ColBERT:
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=doc_length,
        query_length=query_length,
    )
    if cfg.model.compile:
        model.compile()
    model_dtype = get_torch_dtype(cfg.model.dtype)
    current_dtype = next(model.parameters()).dtype
    if current_dtype != model_dtype:
        model = model.to(model_dtype)
    return model


ALL_STAGES = {"encode_docs", "build_index", "encode_queries", "retrieve", "evaluate"}


def parse_stages(cfg: DictConfig) -> set:
    """Parse stages config into a set of stage names."""
    stages_list = list(cfg.stages) if hasattr(cfg, "stages") else ["all"]
    if "all" in stages_list:
        return ALL_STAGES.copy()
    stages = set(stages_list)
    unknown = stages - ALL_STAGES
    if unknown:
        raise ValueError(f"Unknown stages: {unknown}. Valid stages: {ALL_STAGES}")
    return stages


def build_index_only(
    index_config: Dict[str, Any],
    index_instance: indexes.Base,
    documents_ids: List[str],
    documents_embeddings: List[torch.Tensor],
) -> float:
    """Build index and return indexing time."""
    index_name = index_config["name"]

    if hasattr(index_instance, "_documents_added") and index_instance._documents_added:
        logger.info("")
        logger.info("--- Index (%s) ---", index_name)
        logger.info("Index loaded from disk, skipping add_documents.")
        return 0.0

    logger.info("")
    logger.info("--- Index (%s) ---", index_name)
    start_time = time.time()
    add_kwargs = {
        "documents_ids": documents_ids,
        "documents_embeddings": documents_embeddings,
        **index_config.get("add_documents_kwargs", {}),
    }
    index_instance.add_documents(**add_kwargs)
    index_time = time.time() - start_time
    logger.info("%s indexing time: %.2fs", index_name, index_time)
    return index_time


@hydra.main(version_base=None, config_path="conf/eval", config_name="eval_model_irds_v3")
def main(cfg: DictConfig) -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(message)s",
    )

    stages = parse_stages(cfg)
    logger.info("Stages to run: %s", sorted(stages))
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    model_paths = expand_model_paths(cfg.model.name_or_path)
    dataset_names = list(cfg.dataset.names)
    embedding_dtype = get_torch_dtype(cfg.cache.embedding_dtype)

    logger.info("Models to test: %s", model_paths)
    logger.info("Datasets to test: %s", dataset_names)

    # Determine what we need based on stages
    need_doc_encoding = "encode_docs" in stages or "build_index" in stages
    need_query_encoding = "encode_queries" in stages or "retrieve" in stages
    need_indexing = "build_index" in stages
    need_retrieval = "retrieve" in stages or "evaluate" in stages

    with logging_redirect_tqdm():
        for dataset_id, model_name in itertools.product(dataset_names, model_paths):
            logger.info("")
            logger.info("=" * 80)
            logger.info("Processing dataset/model: %s | %s", dataset_id, model_name)
            logger.info("=" * 80)

            query_length = resolve_query_length(dataset_id, cfg.model.query_len)
            doc_length = cfg.model.doc_len

            model = build_model(cfg, model_name, query_length, doc_length)
            embedding_size = get_embedding_size(model)

            documents, queries, qrels = load_dataset(
                dataset_id, lowercase=cfg.dataset.lowercase
            )
            documents_ids = [document["id"] for document in documents]

            dataset_slug = sanitize_dataset_name(dataset_id)
            results_dir = Path(cfg.output.results_dir) / dataset_slug
            results_dir.mkdir(parents=True, exist_ok=True)

            cache_paths = build_cache_paths(
                cfg=cfg,
                dataset_id=dataset_id,
                model_name=model_name,
                doc_length=doc_length,
                query_length=query_length,
                lowercase=cfg.dataset.lowercase,
            )

            index_configs = build_index_configs(
                cfg=cfg,
                dataset_slug=dataset_slug,
                model_name=model_name,
                embedding_size=embedding_size,
                doc_length=doc_length,
                lowercase=cfg.dataset.lowercase,
            )

            # Initialize index instances
            index_instances: List[Tuple[Dict[str, Any], indexes.Base]] = []
            any_index_needs_building = False
            for index_config in index_configs:
                index_instance: indexes.Base = index_config["index_class"](
                    **index_config["init_kwargs"]
                )
                index_instances.append((index_config, index_instance))
                if not getattr(index_instance, "_documents_added", False):
                    any_index_needs_building = True

            # --- Stage: encode_docs ---
            documents_embeddings = None
            if need_doc_encoding and any_index_needs_building:
                logger.info("")
                logger.info("--- Encode Documents ---")
                logger.info("Encoding/loading document embeddings...")
                documents_embeddings = encode_documents_with_cache(
                    model=model,
                    documents=documents,
                    batch_size=cfg.encode.batch_size,
                    shard_size=cfg.encode.shard_size,
                    embedding_dtype=embedding_dtype,
                    move_to_cpu=cfg.encode.move_embeddings_to_cpu,
                    cache_paths=cache_paths,
                    cache_enabled=cfg.cache.enable_docs,
                )
            elif need_doc_encoding:
                logger.info("")
                logger.info("--- Encode Documents ---")
                logger.info("All indexes loaded from disk; skipping document embeddings.")

            # --- Stage: build_index ---
            if need_indexing:
                for index_config, index_instance in index_instances:
                    if not getattr(index_instance, "_documents_added", False):
                        if documents_embeddings is None:
                            raise RuntimeError(
                                f"Index {index_config['name']} requires document embeddings, "
                                "but none are available. Include 'encode_docs' stage."
                            )
                        build_index_only(
                            index_config=index_config,
                            index_instance=index_instance,
                            documents_ids=documents_ids,
                            documents_embeddings=documents_embeddings,
                        )

            # Early exit if no retrieval/evaluation needed
            if not need_retrieval:
                logger.info("")
                logger.info("Stages complete for %s | %s (no retrieval requested)", dataset_id, model_name)
                continue

            # --- Stage: encode_queries ---
            queries_embeddings = None
            if need_query_encoding:
                logger.info("")
                logger.info("--- Encode Queries ---")
                queries_embeddings = encode_queries_with_cache(
                    model=model,
                    queries=queries,
                    batch_size=cfg.encode.batch_size,
                    embedding_dtype=embedding_dtype,
                    move_to_cpu=cfg.encode.move_embeddings_to_cpu,
                    cache_paths=cache_paths,
                    cache_enabled=cfg.cache.enable_queries,
                )

            # --- Stages: retrieve + evaluate ---
            for retrieve_cfg in iter_retrieval_configs(cfg):
                logger.info("")
                logger.info("--- Retrieval Config ---")
                logger.info(
                    "Retrieval: mode=%s k=%s k_token=%s batch_size=%s",
                    retrieve_cfg.mode,
                    retrieve_cfg.k,
                    retrieve_cfg.k_token,
                    retrieve_cfg.batch_size,
                )
                cfg_with_retrieve = OmegaConf.merge(cfg, {"retrieve": retrieve_cfg})
                for index_config, index_instance in index_instances:
                    evaluate_index(
                        cfg=cfg_with_retrieve,
                        retrieve_cfg=retrieve_cfg,
                        index_config=index_config,
                        index_instance=index_instance,
                        queries=queries,
                        queries_embeddings=queries_embeddings,
                        qrels=qrels,
                        dataset_id=dataset_id,
                        model_name=model_name,
                        query_length=query_length,
                        doc_length=doc_length,
                        results_dir=results_dir,
                    )


if __name__ == "__main__":
    main()

