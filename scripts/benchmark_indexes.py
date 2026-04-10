"""Benchmark WARP and PLAID indexes on ir_datasets collections.

Collects evaluation metrics (NDCG, recall, MAP), queries per second (QPS),
index build time, index disk usage, and encode time.

Results are appended as JSONL to the output file.

Usage:
    python scripts/benchmark_indexes.py                          # defaults
    python scripts/benchmark_indexes.py model=xtr_finetuned      # override model
    python scripts/benchmark_indexes.py index=plaid               # override index
    python scripts/benchmark_indexes.py datasets=[beir/fiqa/test,beir/scifact/test]
    python scripts/benchmark_indexes.py --multirun index=warp,plaid
"""

from __future__ import annotations

import datetime
import gc
import json
import logging
import os
import shutil
import time
import tracemalloc
from pathlib import Path

import hydra
import ir_datasets
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from pylate import evaluation, indexes, models, retrieve

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


NUMPY_DTYPES = {
    "fp32": np.float32,
    "fp16": np.float16,
}


def get_hardware_info() -> dict:
    """Return GPU, CPU, and RAM info for reproducibility."""
    import multiprocessing
    import platform

    info: dict = {}

    # GPU
    if torch.cuda.is_available():
        count = torch.cuda.device_count()
        info["gpu_count"] = count
        info["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(count)]
    else:
        info["gpu_count"] = 0
        info["gpu_names"] = []

    # CPU
    cpu_model = "unknown"
    try:
        with open("/proc/cpuinfo") as f:
            for line in f:
                if line.startswith("model name"):
                    cpu_model = line.split(":", 1)[1].strip()
                    break
    except OSError:
        cpu_model = platform.processor() or "unknown"
    info["cpu_model"] = cpu_model
    info["cpu_cores"] = multiprocessing.cpu_count()

    # RAM
    try:
        with open("/proc/meminfo") as f:
            for line in f:
                if line.startswith("MemTotal:"):
                    kb = int(line.split()[1])
                    info["ram_gb"] = round(kb / (1024 * 1024), 1)
                    break
    except OSError:
        pass

    return info


def parse_shard_selection(spec, num_shards: int) -> list[int]:
    """Parse a shard selection into a list of shard indices.

    Accepts:
      - A list of ints:  [0, 2, 7, 8]  (Hydra parses this natively)
      - A slice string:  :10, ::2, :16:4, 5:  (no brackets, to avoid Hydra list parsing)
      - None: returns all shard indices
    """
    if spec is None:
        return list(range(num_shards))

    if isinstance(spec, (list, tuple)):
        return [int(i) for i in spec]

    # String — parse as slice
    s = str(spec).strip()
    parts = s.split(":")
    args = [int(p) if p.strip() else None for p in parts]
    sl = slice(*args)
    return list(range(num_shards))[sl]


def sanitize_name(name: str) -> str:
    """Sanitize a model/dataset name for use as a directory name."""
    return name.replace("/", "_").strip("_")


def get_dir_size_mb(path: str) -> float:
    """Return total size of a directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def get_warp_config(index) -> dict:
    """Extract WARP search config from a WARP index, empty dict for others."""
    config = {
        "random_rotation": getattr(index, "random_rotation", False),
        "warp_num_shards": getattr(index, "num_shards", None),
    }
    if hasattr(index, "bound"):
        config.update({
            "warp_bound": index.bound,
            "warp_nprobe": index.nprobe,
            "warp_centroid_score_threshold": index.centroid_score_threshold,
            "warp_max_candidates": index.max_candidates,
            "warp_t_prime": index.t_prime,
            "warp_auto_tune": index.auto_tune,
        })
    return config


def append_jsonl(path: str, row: dict) -> None:
    """Append a single JSON object as a line to the file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def append_stage_marker(
    stage: str,
    event: str,
    dataset: str,
    index_type: str,
    retrieval: str | None = None,
) -> None:
    """Append stage timing markers when BENCH_STAGE_MARKERS_FILE is set."""
    marker_path = os.environ.get("BENCH_STAGE_MARKERS_FILE")
    if not marker_path:
        return
    row = {
        "timestamp": datetime.datetime.now().isoformat(),
        "ts_epoch": time.time(),
        "stage": stage,
        "event": event,
        "dataset": dataset,
        "index_type": index_type,
        "retrieval": retrieval,
    }
    append_jsonl(marker_path, row)


# ---------------------------------------------------------------------------
# Dataset loading via ir_datasets
# ---------------------------------------------------------------------------


def load_dataset(
    dataset_id: str,
) -> tuple[list[dict], dict[str, str], dict[str, dict[str, int]]]:
    """Load documents, queries, and qrels from an ir_datasets collection."""
    logger.info("Loading dataset: %s", dataset_id)
    dataset = ir_datasets.load(dataset_id)

    documents = []
    for doc in tqdm(dataset.docs_iter(), desc="Loading documents", unit="docs"):
        if hasattr(doc, "title") and doc.title:
            text = f"{doc.title}\n\n{doc.text}".strip()
        else:
            text = doc.text.strip()
        documents.append({"id": doc.doc_id, "text": text})

    queries = {}
    for query in tqdm(dataset.queries_iter(), desc="Loading queries", unit="queries"):
        queries[query.query_id] = query.text.strip()

    qrels = {}
    if dataset.has_qrels():
        for qrel in dataset.qrels_iter():
            if qrel.query_id not in qrels:
                qrels[qrel.query_id] = {}
            qrels[qrel.query_id][qrel.doc_id] = int(qrel.relevance)

    logger.info(
        "Loaded %d documents, %d queries, %d queries with qrels",
        len(documents), len(queries), len(qrels),
    )
    return documents, queries, qrels


# ---------------------------------------------------------------------------
# Sharded encoding + caching (.npy + .doclens.npy)
# ---------------------------------------------------------------------------


def _shard_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_shard_{idx:03d}.npy"


def _doclens_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_shard_{idx:03d}.doclens.npy"


def encode_documents_sharded(
    model_name: str,
    documents: list[dict],
    cache_dir: Path,
    cfg: DictConfig,
) -> tuple[Path, float]:
    """Encode documents in shards, saving .npy + .doclens.npy files.

    Returns (shard_directory, encode_time_seconds).
    Skips shards that already exist on disk (checkpointing).
    """
    cache_dir = cache_dir / "docs"
    cache_dir.mkdir(parents=True, exist_ok=True)
    meta_path = cache_dir / "doc_meta.json"

    shard_size = cfg.encode.shard_size
    num_documents = len(documents)
    num_shards = (num_documents + shard_size - 1) // shard_size

    # Determine which shards to encode
    shard_selection = parse_shard_selection(cfg.encode.get("shards", None), num_shards)
    logger.info("Shard selection: %d/%d shards", len(shard_selection), num_shards)

    # Check which selected shards are already cached
    cached = set()
    for idx in shard_selection:
        if _shard_path(cache_dir, idx).exists() and _doclens_path(cache_dir, idx).exists():
            cached.add(idx)

    to_encode = [idx for idx in shard_selection if idx not in cached]

    if cached:
        logger.info(
            "Found %d/%d selected shards cached, encoding %d remaining.",
            len(cached), len(shard_selection), len(to_encode),
        )

    if not to_encode:
        logger.info("All selected shards cached, skipping encoding.")
        return cache_dir, 0.0

    logger.info("Shards to encode: %s", to_encode)

    # Load model only if we have shards to encode
    pool_factor = cfg.encode.get("pool_factor", 1)
    n_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 0
    use_multi_gpu = n_gpus > 1

    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=cfg.doc_length,
        device="cpu" if use_multi_gpu else None,
    )
    if cfg.get("compile", False) and not use_multi_gpu:
        model = torch.compile(model)

    pool = None
    if use_multi_gpu:
        logger.info("Starting multi-GPU encoding pool (%d GPUs).", n_gpus)
        pool = model.start_multi_process_pool()

    encode_start = time.perf_counter()

    for shard_idx in to_encode:
        start = shard_idx * shard_size
        end = min(start + shard_size, num_documents)
        logger.info("Encoding shard %d/%d (docs %d-%d)", shard_idx + 1, num_shards, start, end - 1)

        if use_multi_gpu:
            shard_embeddings = model.encode_multi_process(
                sentences=[doc["text"] for doc in documents[start:end]],
                pool=pool,
                batch_size=cfg.encode.batch_size,
                is_query=False,
                pool_factor=pool_factor,
                protected_tokens=cfg.encode.get("protected_tokens", 1),
            )
        else:
            shard_embeddings = model.encode(
                sentences=[doc["text"] for doc in documents[start:end]],
                batch_size=cfg.encode.batch_size,
                is_query=False,
                show_progress_bar=True,
                pool_factor=pool_factor,
                pool_method=cfg.encode.get("pool_method", "hierarchical"),
                protected_tokens=cfg.encode.get("protected_tokens", 1),
            )

        # Convert to numpy, cast to target dtype, and save
        save_dtype = NUMPY_DTYPES.get(cfg.encode.get("dtype", "fp32"), np.float32)
        doclens = []
        all_tokens = []
        for emb in shard_embeddings:
            if isinstance(emb, torch.Tensor):
                emb = emb.cpu().numpy()
            doclens.append(emb.shape[0])
            all_tokens.append(emb.astype(save_dtype))

        concatenated = np.concatenate(all_tokens, axis=0)
        np.save(_shard_path(cache_dir, shard_idx), concatenated)
        np.save(_doclens_path(cache_dir, shard_idx), np.array(doclens, dtype=np.int32))

    encode_time = time.perf_counter() - encode_start

    if pool is not None:
        model.stop_multi_process_pool(pool)
    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # Write metadata
    meta_path.write_text(json.dumps({
        "num_documents": num_documents,
        "num_shards": num_shards,
        "shard_size": shard_size,
        "pool_factor": pool_factor,
        "created_at": datetime.datetime.now().isoformat(),
    }, indent=2) + "\n")

    return cache_dir, encode_time


def encode_queries(
    model_name: str,
    queries: dict[str, str],
    cache_dir: Path,
    cfg: DictConfig,
) -> tuple[list[torch.Tensor], float]:
    """Encode queries, caching as a single .npy + .doclens.npy file.

    Returns (list_of_query_embeddings, encode_time_seconds).
    """
    cache_dir = cache_dir / "queries"
    cache_dir.mkdir(parents=True, exist_ok=True)
    emb_path = cache_dir / "query_emb.npy"
    doclens_path = cache_dir / "query_emb.doclens.npy"

    if emb_path.exists() and doclens_path.exists():
        logger.info("Loading cached query embeddings.")
        data = np.load(emb_path)
        doclens = np.load(doclens_path)
        embeddings = []
        offset = 0
        for length in doclens:
            emb = torch.from_numpy(data[offset:offset + length].copy()).float()
            embeddings.append(emb)
            offset += length
        return embeddings, 0.0

    dataset_id = None
    for ds in cfg.datasets:
        # Pick query_length from the first dataset that matches this cache_dir
        if sanitize_name(ds) in str(cache_dir):
            dataset_id = ds
            break

    query_length = cfg.query_length.get(dataset_id) if dataset_id else None

    model = models.ColBERT(
        model_name_or_path=model_name,
        query_length=query_length,
    )
    if cfg.get("compile", False):
        model = torch.compile(model)

    encode_start = time.perf_counter()
    query_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=cfg.encode.query_batch_size,
    )
    encode_time = time.perf_counter() - encode_start

    # Save to .npy
    doclens = []
    all_tokens = []
    for emb in query_embeddings:
        if isinstance(emb, torch.Tensor):
            emb_np = emb.cpu().numpy()
        else:
            emb_np = emb
        doclens.append(emb_np.shape[0])
        all_tokens.append(emb_np)

    np.save(emb_path, np.concatenate(all_tokens, axis=0))
    np.save(doclens_path, np.array(doclens, dtype=np.int32))

    del model
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    return query_embeddings, encode_time


# ---------------------------------------------------------------------------
# Shard loading (for PLAID — needs in-memory embeddings)
# ---------------------------------------------------------------------------


def load_shards_to_memory(shard_dir: Path, dtype: torch.dtype = torch.float32) -> list[torch.Tensor]:
    """Load all .npy + .doclens.npy shards into a list of per-document tensors."""
    npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
    npy_files = [f for f in npy_files if not f.name.endswith(".doclens.npy")]

    embeddings = []
    for npy_file in npy_files:
        doclens_file = npy_file.with_suffix(".doclens.npy")
        data = np.load(npy_file)
        doclens = np.load(doclens_file)
        offset = 0
        for length in doclens:
            emb = torch.from_numpy(data[offset:offset + length].copy())
            if emb.dtype != dtype:
                emb = emb.to(dtype)
            embeddings.append(emb)
            offset += length

    return embeddings


def count_doc_tokens_from_shards(shard_dir: Path) -> int:
    """Count total token embeddings across all shards."""
    total = 0
    for npy_file in sorted(shard_dir.glob("doc_shard_*.npy")):
        if npy_file.name.endswith(".doclens.npy"):
            continue
        total += np.load(npy_file, mmap_mode="r").shape[0]
    return total


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------


def build_index(
    index_type: str,
    documents: list[dict],
    shard_dir: Path,
    index_folder: str,
    index_name: str,
    device: str | None,
    index_cfg: DictConfig | None = None,
) -> tuple:
    """Build an index and return (index, build_time, disk_mb)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    doc_ids = [doc["id"] for doc in documents]
    num_docs = len(doc_ids)
    nbits = index_cfg.get("nbits", 4) if index_cfg else 4

    # Compute n_samples_kmeans from full corpus size using xtr-warp's formula:
    # min(1 + 16*sqrt(120*N), N)
    import math
    n_samples_kmeans = min(1 + int(16 * math.sqrt(120 * num_docs)), num_docs)

    random_rotation = index_cfg.get("random_rotation", False) if index_cfg else False
    use_triton = index_cfg.get("use_triton", None) if index_cfg else None
    num_shards = index_cfg.get("num_shards", None) if index_cfg else None
    verbose = index_cfg.get("verbose", False) if index_cfg else False
    plaid_search_batch_size = index_cfg.get("search_batch_size", None) if index_cfg else None

    if index_type == "warp":
        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            device=device,
            nbits=nbits,
            n_samples_kmeans=n_samples_kmeans,
            random_rotation=random_rotation,
            use_triton=use_triton,
            num_shards=num_shards,
            verbose=verbose,
        )
        build_start = time.perf_counter()
        if random_rotation:
            # Must load shards into memory so rotation can be applied
            logger.info("Loading shards into memory for random rotation...")
            documents_embeddings = load_shards_to_memory(shard_dir)
            index.add_documents(
                documents_ids=doc_ids,
                documents_embeddings=documents_embeddings,
            )
            del documents_embeddings
            gc.collect()
        else:
            index.add_documents(
                documents_ids=doc_ids,
                documents_embeddings=shard_dir,
            )
        build_time = time.perf_counter() - build_start

    elif index_type in ("plaid", "fast_plaid"):
        plaid_kwargs = {}
        if plaid_search_batch_size is not None:
            plaid_kwargs["batch_size"] = int(plaid_search_batch_size)
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            nbits=nbits,
            n_samples_kmeans=n_samples_kmeans,
            use_triton=use_triton,
            random_rotation=random_rotation,
            show_progress=verbose,
            **plaid_kwargs,
        )
        # Add documents shard-by-shard to avoid loading all into memory.
        # First call loads enough shards to have >= 5x n_samples_kmeans docs
        # for good centroid quality. Subsequent shards use update().
        npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
        npy_files = [f for f in npy_files if not f.name.endswith(".doclens.npy")]

        # Determine how many shards to load for the initial create().
        # n_samples_kmeans is a vector count. Load enough shards to have
        # >= 5x n_samples_kmeans vectors for representative centroid training.
        min_vectors_for_create = n_samples_kmeans * 5
        initial_shards = 0
        initial_vector_count = 0
        for npy_file in npy_files:
            doclens_file = npy_file.with_suffix(".doclens.npy")
            initial_vector_count += int(np.load(doclens_file).sum())
            initial_shards += 1
            if initial_vector_count >= min_vectors_for_create:
                break

        doc_offset = 0
        build_start = time.perf_counter()

        # Load initial shards together for create() (centroid training)
        initial_embeddings = []
        initial_ids = []
        for shard_file in npy_files[:initial_shards]:
            doclens_file = shard_file.with_suffix(".doclens.npy")
            data = np.load(shard_file, mmap_mode="c")
            doclens = np.load(doclens_file)
            offset = 0
            for length in doclens:
                initial_embeddings.append(torch.from_numpy(data[offset:offset + length]))
                offset += length
            shard_ids = doc_ids[doc_offset:doc_offset + len(doclens)]
            initial_ids.extend(shard_ids)
            doc_offset += len(doclens)
            del data

        logger.info(
            "PLAID: initial create with %d shards (%d docs, n_samples_kmeans=%d)",
            initial_shards, len(initial_ids), n_samples_kmeans,
        )
        index.add_documents(
            documents_ids=initial_ids,
            documents_embeddings=initial_embeddings,
        )
        del initial_embeddings, initial_ids
        gc.collect()

        # Remaining shards via update(), grouped in batches to reduce Python/add overhead.
        # Tune this constant if needed; larger values reduce update-call overhead but increase
        # transient host memory used per add_documents() call.
        update_shard_batch_size = max(
            int(index_cfg.get("add_batch_size", 4)) if index_cfg else 4,
            1,
        )
        remaining_files = npy_files[initial_shards:]
        for batch_start in tqdm(
            range(0, len(remaining_files), update_shard_batch_size),
            desc="PLAID update batches",
            unit="batch",
        ):
            batch_files = remaining_files[batch_start:batch_start + update_shard_batch_size]
            batch_embeddings = []
            batch_ids = []

            for shard_file in batch_files:
                doclens_file = shard_file.with_suffix(".doclens.npy")
                data = np.load(shard_file, mmap_mode="c")
                doclens = np.load(doclens_file)

                offset = 0
                for length in doclens:
                    batch_embeddings.append(torch.from_numpy(data[offset:offset + length]))
                    offset += length

                shard_ids = doc_ids[doc_offset:doc_offset + len(doclens)]
                batch_ids.extend(shard_ids)
                doc_offset += len(doclens)
                del data

            # logger.info(
            #     "PLAID: updating with shards %s (%d docs)",
            #     [f.name for f in batch_files],
            #     len(batch_ids),
            # )
            index.add_documents(
                documents_ids=batch_ids,
                documents_embeddings=batch_embeddings,
            )

            del batch_embeddings, batch_ids
            gc.collect()

        build_time = time.perf_counter() - build_start

    elif index_type == "scann":
        index = indexes.ScaNN(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            store_embeddings=True,
        )
        logger.info("Loading all shards into memory for ScaNN...")
        documents_embeddings = load_shards_to_memory(shard_dir)
        build_start = time.perf_counter()
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=documents_embeddings,
        )
        build_time = time.perf_counter() - build_start
        del documents_embeddings
        gc.collect()

    else:
        raise ValueError(f"Unknown index type: {index_type}")

    index_path = os.path.join(index_folder, index_name)
    disk_mb = get_dir_size_mb(index_path)

    return index, build_time, disk_mb


# ---------------------------------------------------------------------------
# Search + evaluation
# ---------------------------------------------------------------------------


def benchmark_search(
    index,
    index_type: str,
    retrieval: str,
    queries_embeddings: list,
    query_ids: list[str],
    qrels: dict,
    k: int,
    k_token: int = 10000,
    device: str | None = None,
    plaid_outer_batch_size: int | None = None,
    run_save_path: str | None = None,
    metrics: list | None = None,
) -> dict:
    """Run search on a pre-built index and collect metrics."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if index_type == "warp":
        retriever = retrieve.XTR(index=index, verbose=False)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif index_type in ("plaid", "fast_plaid"):
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif index_type == "scann" and retrieval == "xtr":
        retriever = retrieve.XTR(index=index, verbose=False)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu",
        )
    elif index_type == "scann" and retrieval == "colbert":
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu",
        )
    else:
        raise ValueError(f"Unknown index_type/retrieval combo: {index_type}/{retrieval}")

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()

    search_start = time.perf_counter()
    if index_type in ("plaid", "fast_plaid"):
        # Python-level progress for PLAID search in case backend progress is not visible.
        outer_batch_size = max(int(plaid_outer_batch_size), 1) if plaid_outer_batch_size is not None else 256
        scores = []
        for start in tqdm(
            range(0, len(queries_embeddings), outer_batch_size),
            desc="PLAID search batches",
            unit="batch",
        ):
            end = start + outer_batch_size
            scores.extend(
                retriever.retrieve(
                    queries_embeddings=queries_embeddings[start:end],
                    k=k,
                )
            )
    else:
        scores = retriever.retrieve(**retrieve_kwargs)
    search_time = time.perf_counter() - search_start

    _, peak_ram_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    peak_vram_bytes = torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0

    n_queries = len(queries_embeddings)
    qps = n_queries / search_time if search_time > 0 else float("inf")

    if run_save_path is not None:
        from ranx import Run
        run_dict = {
            qid: {d["id"]: float(d["score"]) for d in doc_scores}
            for qid, doc_scores in zip(query_ids, scores)
        }
        os.makedirs(os.path.dirname(run_save_path), exist_ok=True)
        Run(run_dict).save(run_save_path, kind="trec")

    eval_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=metrics,
    )

    return {
        "search_time_s": round(search_time, 4),
        "qps": round(qps, 2),
        "peak_vram_mb": round(peak_vram_bytes / (1024 * 1024), 2),
        "peak_ram_mb": round(peak_ram_bytes / (1024 * 1024), 2),
        "n_queries": n_queries,
        **{k: round(v, 4) for k, v in eval_scores.items()},
    }


# ---------------------------------------------------------------------------
# Load existing index (for retrieve-only)
# ---------------------------------------------------------------------------


def load_existing_index(
    index_type: str,
    index_folder: str,
    index_name: str,
    device: str | None,
    index_cfg: DictConfig | None = None,
):
    """Load a pre-built index from disk."""
    nbits = index_cfg.get("nbits", 4) if index_cfg else 4

    random_rotation = index_cfg.get("random_rotation", False) if index_cfg else False
    num_shards = index_cfg.get("num_shards", None) if index_cfg else None
    verbose = index_cfg.get("verbose", False) if index_cfg else False
    plaid_search_batch_size = index_cfg.get("search_batch_size", None) if index_cfg else None

    if index_type == "warp":
        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            device=device,
            nbits=nbits,
            random_rotation=random_rotation,
            num_shards=num_shards,
            verbose=verbose,
        )
    elif index_type in ("plaid", "fast_plaid"):
        plaid_kwargs = {}
        if plaid_search_batch_size is not None:
            plaid_kwargs["batch_size"] = int(plaid_search_batch_size)
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            nbits=nbits,
            random_rotation=random_rotation,
            show_progress=verbose,
            **plaid_kwargs,
        )
    elif index_type == "scann":
        index = indexes.ScaNN(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    # if not index.is_indexed:
    #     raise FileNotFoundError(
    #         f"No existing {index_type} index found at {index_folder}/{index_name}. "
    #         "Run with build_index stage first."
    #     )
    return index


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


@hydra.main(config_path="../conf/eval", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    logging.basicConfig(level=logging.INFO)

    canonical_stage_order = [
        "encode_docs",
        "encode_queries",
        "build_index",
        "retrieve",
        "delete_index",
        "delete_embeddings",
    ]

    model_name = cfg.model.name_or_path
    model_slug = sanitize_name(model_name)
    index_type = cfg.index.type
    retrieval_modes = list(cfg.index.retrieval)
    device = cfg.get("device")
    stage_set = set(cfg.stages)
    ordered_stages = [s for s in canonical_stage_order if s in stage_set]
    ordered_stages.extend(sorted(s for s in stage_set if s not in canonical_stage_order))

    all_results = []

    for dataset_id in cfg.datasets:
        dataset_slug = sanitize_name(dataset_id)

        print(f"\n{'#' * 60}")
        print(f"# Dataset: {dataset_id}  stages: {ordered_stages}")
        print(f"{'#' * 60}")

        # Always load dataset metadata (needed for doc IDs, qrels, etc.)
        documents, queries, qrels = load_dataset(dataset_id)
        query_ids = list(queries.keys())

        # Cache directory: dataset / model / encoding params
        pool_factor = cfg.encode.get("pool_factor", 1)
        pool_method = cfg.encode.get("pool_method", "none")
        protected_tokens = cfg.encode.get("protected_tokens", 1)
        emb_dtype = cfg.encode.get("dtype", "fp32")
        if pool_factor > 1:
            pool_slug = f"{pool_method}_p{protected_tokens}f{pool_factor}"
        else:
            pool_slug = "nopool"
        cache_dir = Path(cfg.cache.dir) / dataset_slug / model_slug / f"{pool_slug}_{emb_dtype}"
        shard_dir = cache_dir / "docs"
        random_rotation = cfg.index.get("random_rotation", False)
        rotation_slug = "_rotated" if random_rotation else ""
        num_shards = cfg.index.get("num_shards", None)
        if num_shards == 1:
            logger.warning("num_shards is 1, which has a bug in WARP (should be identical to None), setting to None")
            cfg.index.num_shards = None
            num_shards = None
        shard_slug = f"_s{int(num_shards)}" if num_shards and int(num_shards) > 1 else ""
        index_config = OmegaConf.to_container(cfg.index, resolve=True)
        index_name = f"bench_{model_slug}_{dataset_slug}_{index_type}{rotation_slug}{shard_slug}"

        # Track timing for results
        doc_encode_time = 0.0
        query_encode_time = 0.0
        build_time = 0.0
        disk_mb = 0.0
        n_doc_tokens = 0
        index = None

        # --- Stage: encode_docs ---
        if "encode_docs" in stage_set:
            append_stage_marker("encode_docs", "start", dataset_id, index_type)
            print(f"\n  Encoding {len(documents)} documents...")
            shard_dir, doc_encode_time = encode_documents_sharded(
                model_name=model_name,
                documents=documents,
                cache_dir=cache_dir,
                cfg=cfg,
            )
            print(f"  Encode time: {doc_encode_time:.2f}s")
            append_stage_marker("encode_docs", "end", dataset_id, index_type)

        if shard_dir.exists():
            n_doc_tokens = count_doc_tokens_from_shards(shard_dir)
            print(f"  Doc tokens: {n_doc_tokens}")

        # --- Stage: encode_queries ---
        queries_embeddings = None
        if "encode_queries" in stage_set:
            append_stage_marker("encode_queries", "start", dataset_id, index_type)
            print(f"  Encoding {len(queries)} queries...")
            queries_embeddings, query_encode_time = encode_queries(
                model_name=model_name,
                queries=queries,
                cache_dir=cache_dir,
                cfg=cfg,
            )
            print(f"  Query encode time: {query_encode_time:.2f}s")
            append_stage_marker("encode_queries", "end", dataset_id, index_type)

        # --- Stage: build_index ---
        if "build_index" in stage_set:
            append_stage_marker("build_index", "start", dataset_id, index_type)
            if not shard_dir.exists():
                raise FileNotFoundError(
                    f"No doc shards at {shard_dir}. Run with encode_docs stage first."
                )
            print(f"\n  Building {index_type} index...")
            index, build_time, disk_mb = build_index(
                index_type=index_type,
                documents=documents,
                shard_dir=shard_dir,
                index_folder=cfg.output.index_folder,
                index_name=index_name,
                device=device,
                index_cfg=cfg.index,
            )
            print(f"  Build time: {build_time:.2f}s, disk: {disk_mb:.2f} MB")
            append_stage_marker("build_index", "end", dataset_id, index_type)

        # --- Stage: retrieve (loops over retrieval modes) ---
        if "retrieve" in stage_set:
            append_stage_marker("retrieve", "start", dataset_id, index_type)
            # Load queries if not already encoded in this run
            if queries_embeddings is None:
                query_cache = cache_dir / "queries"
                if (query_cache / "query_emb.npy").exists():
                    queries_embeddings, _ = encode_queries(
                        model_name=model_name,
                        queries=queries,
                        cache_dir=cache_dir,
                        cfg=cfg,
                    )
                else:
                    raise FileNotFoundError(
                        f"No cached query embeddings at {query_cache}. "
                        "Run with encode_queries stage first."
                    )

            # Load index if not built in this run
            if index is None:
                print(f"\n  Loading existing {index_type} index...")
                index = load_existing_index(
                    index_type=index_type,
                    index_folder=cfg.output.index_folder,
                    index_name=index_name,
                    device=device,
                    index_cfg=cfg.index,
                )
                index_path = os.path.join(cfg.output.index_folder, index_name)
                disk_mb = get_dir_size_mb(index_path)

            search_repeats = cfg.search.get("repeats", 1)
            for retrieval in retrieval_modes:
                append_stage_marker("retrieve_mode", "start", dataset_id, index_type, retrieval)
                for run_idx in range(search_repeats):
                    run_label = f" (run {run_idx + 1}/{search_repeats})" if search_repeats > 1 else ""
                    print(f"\n  Searching: {index_type} + {retrieval}{run_label}...")
                    runs_dir = cfg.output.get("runs_dir")
                    run_save_path = (
                        os.path.join(runs_dir, model_slug, f"{dataset_slug}_{index_type}_{retrieval}.run")
                        if runs_dir else None
                    )
                    search_result = benchmark_search(
                        index=index,
                        index_type=index_type,
                        retrieval=retrieval,
                        queries_embeddings=queries_embeddings,
                        query_ids=query_ids,
                        qrels=qrels,
                        k=cfg.search.k,
                        k_token=cfg.search.get("k_token", 10000),
                        device=device,
                        plaid_outer_batch_size=cfg.index.get("search_batch_size", None),
                        run_save_path=run_save_path,
                        metrics=list(cfg.search.metrics),
                    )
                    print(f"  QPS: {search_result['qps']}, NDCG@10: {search_result.get('ndcg@10', 'N/A')}")
                    print(f"  Recall@100: {search_result.get('recall@100', 'N/A')}")

                    row = {
                        "timestamp": datetime.datetime.now().isoformat(),
                        "model": model_name,
                        "dataset": dataset_id,
                        "index_type": index_type,
                        "retrieval": retrieval,
                        "stages": ordered_stages,
                        "k": cfg.search.k,
                        "n_documents": len(documents),
                        "n_doc_tokens": n_doc_tokens,
                        "doc_encode_time_s": round(doc_encode_time, 2),
                        "query_encode_time_s": round(query_encode_time, 2),
                        "build_time_s": round(build_time, 2),
                        "disk_mb": round(disk_mb, 2),
                        "index_size_mb": round(disk_mb, 2),
                        "nbits": cfg.index.get("nbits", 4),
                        "index_config": index_config,
                        "pool_factor": pool_factor,
                        "pool_method": pool_method if pool_factor > 1 else "none",
                        "protected_tokens": protected_tokens,
                        "encode_dtype": emb_dtype,
                        "run_idx": run_idx,
                        "search_repeats": search_repeats,
                        **get_hardware_info(),
                        **(get_warp_config(index) if index else {}),
                        **search_result,
                    }
                    all_results.append(row)
                    append_jsonl(cfg.output.results_file, row)
                append_stage_marker("retrieve_mode", "end", dataset_id, index_type, retrieval)
            append_stage_marker("retrieve", "end", dataset_id, index_type)

        # Log build-only results if retrieve not in stages
        if "retrieve" not in stage_set:
            row = {
                "timestamp": datetime.datetime.now().isoformat(),
                "model": model_name,
                "dataset": dataset_id,
                "index_type": index_type,
                "retrieval": None,
                "stages": ordered_stages,
                "k": cfg.search.k,
                "n_documents": len(documents),
                "n_doc_tokens": n_doc_tokens,
                "doc_encode_time_s": round(doc_encode_time, 2),
                "query_encode_time_s": round(query_encode_time, 2),
                "build_time_s": round(build_time, 2),
                "disk_mb": round(disk_mb, 2),
                "nbits": cfg.index.get("nbits", 4),
                "index_config": index_config,
                "pool_factor": pool_factor,
                "pool_method": pool_method if pool_factor > 1 else "none",
                "protected_tokens": protected_tokens,
                **get_hardware_info(),
                **(get_warp_config(index) if index else {}),
            }
            all_results.append(row)
            append_jsonl(cfg.output.results_file, row)

        # Cleanup
        if index is not None:
            del index
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # --- Stage: delete_index ---
        if "delete_index" in stage_set:
            append_stage_marker("delete_index", "start", dataset_id, index_type)
            index_path = os.path.join(cfg.output.index_folder, index_name)
            if os.path.isdir(index_path):
                size_mb = get_dir_size_mb(index_path)
                shutil.rmtree(index_path)
                print(f"  Deleted index at {index_path} (freed {size_mb:.1f} MB)")
            append_stage_marker("delete_index", "end", dataset_id, index_type)

        # --- Stage: delete_embeddings ---
        if "delete_embeddings" in stage_set:
            append_stage_marker("delete_embeddings", "start", dataset_id, index_type)
            if cache_dir.is_dir():
                size_mb = get_dir_size_mb(str(cache_dir))
                shutil.rmtree(cache_dir)
                print(f"  Deleted embeddings cache at {cache_dir} (freed {size_mb:.1f} MB)")
            append_stage_marker("delete_embeddings", "end", dataset_id, index_type)

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    for r in all_results:
        print(
            f"  {r['dataset']:<30} {r['index_type']:<8} {str(r.get('retrieval','')):<8} "
            f"build={r['build_time_s']}s  disk={r['disk_mb']}MB  "
            f"qps={r.get('qps', 'N/A')}  "
            f"ndcg@10={r.get('ndcg@10', 'N/A')}  "
            f"r@100={r.get('recall@100', 'N/A')}"
        )

    print(f"\nResults appended to {cfg.output.results_file}")


if __name__ == "__main__":
    main()
