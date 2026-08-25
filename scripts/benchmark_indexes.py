"""Benchmark indexes on ir_datasets text collections or ViDoRe image collections.

Collects evaluation metrics (NDCG, recall, MAP), queries per second (QPS),
index build time, index disk usage, and encode time.

Results are appended as JSONL to the output file.

Usage:
    python scripts/benchmark_indexes.py                              # defaults
    python scripts/benchmark_indexes.py model=google_xtr             # override model
    python scripts/benchmark_indexes.py index=plaid                  # override index
    python scripts/benchmark_indexes.py index=tachiom                # Tachiom index
    python scripts/benchmark_indexes.py index=chimera                # Chimera index
    python scripts/benchmark_indexes.py index=tachiom 'index/clustering=pgc'
    python scripts/benchmark_indexes.py datasets=[beir/fiqa/test,beir/scifact/test]
    python scripts/benchmark_indexes.py --multirun index=plaid,tachiom,chimera

Multimodal (ViDoRe visual document retrieval): documents are page images
encoded by a ColPali-family VLM. Same pipeline (encode -> cache -> index ->
retrieve), one constraint -- TAC clustering is rejected (image patches have no
vocabulary token types); use PGC or a PLAID/WARP index.
    python scripts/benchmark_indexes.py model=colqwen index=plaid \\
        datasets=[vidore/finance]
    python scripts/benchmark_indexes.py model=colqwen index=tachiom \\
        'index/clustering=pgc' datasets=[vidore/finance]
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
from pylate.profiling import active

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


def get_index_config_dict(index) -> dict:
    """Extract logged config fields from a built index object."""
    config = {}
    for attr in (
        # WARP search params
        "nbits", "n_ivf_probe", "bound", "t_prime", "max_candidates",
        "centroid_score_threshold", "kmeans_niters", "n_samples_kmeans",
        "min_outliers", "max_growth_rate",
        # Chimera search/build params
        "nprobe", "k_refine", "k_full_bit", "cagra_itopk_size", "num_chunks",
        "ex_bits", "tokens_per_cluster", "scores_are_synthetic",
        # Actual centroid count (WARP/PLAID read from disk; Tachiom/Chimera stored at build time)
        "actual_total_centroids",
    ):
        val = getattr(index, attr, None)
        if val is not None:
            config[attr] = val
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


def _doc_payload(doc: dict):
    """Return the model input for a document: a PIL image for multimodal corpora
    (ViDoRe), otherwise the document text."""
    return doc["image"] if "image" in doc else doc["text"]


def is_multimodal_corpus(documents: list[dict]) -> bool:
    """True when documents are images (ViDoRe) rather than text."""
    return bool(documents) and "image" in documents[0]


def load_dataset(
    dataset_id: str,
) -> tuple[list[dict], dict[str, str], dict[str, dict[str, int]]]:
    """Load documents, queries, and qrels.

    ``vidore/<name>`` ids (e.g. ``vidore/finance``) load a ViDoRe visual-document
    dataset whose documents are page images; everything else loads from an
    ir_datasets collection of text documents.
    """
    if dataset_id.startswith("vidore/"):
        logger.info("Loading ViDoRe dataset: %s", dataset_id)
        return evaluation.load_vidore(dataset_id.split("/", 1)[1])

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


def subsample_queries(query_ids, queries_embeddings, qrels, n, seed, cache_path):
    """Deterministically keep ``n`` queries, identical across runs and indexes.

    The subset is a fixed permutation of the *sorted* query ids — sorting first
    makes it independent of dataset load order — and is frozen to ``cache_path``
    on first use, then reused verbatim so it can never drift (even if this code
    or the seed later changes). ``query_ids`` and ``queries_embeddings`` are
    positionally aligned; both are sliced by the same order-preserving mask so
    the alignment survives, and ``qrels`` is filtered to the kept ids so metrics
    are computed over exactly the retained set.
    """
    if not n or n >= len(query_ids):
        return query_ids, queries_embeddings, qrels
    if os.path.exists(cache_path):
        chosen = set(json.load(open(cache_path))["query_ids"])
        missing = chosen - set(query_ids)
        if missing:
            raise ValueError(
                f"subsample {cache_path} has {len(missing)} ids absent from this "
                "dataset — wrong dataset for this frozen subsample."
            )
    else:
        sids = sorted(query_ids)
        perm = np.random.default_rng(seed).permutation(len(sids))
        chosen = {sids[j] for j in perm[:n]}
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "w") as f:
            json.dump(
                {"query_ids": sorted(chosen), "n": n, "seed": seed,
                 "total": len(query_ids)}, f, indent=0,
            )
    keep = [i for i, q in enumerate(query_ids) if q in chosen]
    q_ids = [query_ids[i] for i in keep]
    q_emb = [queries_embeddings[i] for i in keep]
    q_rels = {q: qrels[q] for q in q_ids if q in qrels}
    logger.info(
        "query subsample: %d -> %d (seed=%d) from %s",
        len(query_ids), len(q_ids), seed, cache_path,
    )
    return q_ids, q_emb, q_rels


# ---------------------------------------------------------------------------
# Sharded encoding + caching (.npy + .doclens.npy)
# ---------------------------------------------------------------------------


def _shard_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_shard_{idx:03d}.npy"


def _doclens_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_shard_{idx:03d}.doclens.npy"


def _token_ids_path(cache_dir: Path, idx: int) -> Path:
    return cache_dir / f"doc_shard_{idx:03d}.token_ids.npy"


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

    save_token_ids = cfg.encode.get("save_token_ids", True)

    # Image corpora (ViDoRe) have no vocabulary token IDs -- only visual-patch
    # placeholders -- so caching them is pointless and TAC clustering (which keys
    # on token types) is meaningless. Force them off; build_index rejects TAC.
    multimodal = is_multimodal_corpus(documents)
    if multimodal and save_token_ids:
        logger.warning(
            "Multimodal (image) corpus: forcing save_token_ids=False "
            "(no vocabulary token IDs; TAC clustering is not applicable)."
        )
        save_token_ids = False

    # Check which selected shards are already cached
    cached = set()
    for idx in shard_selection:
        base_cached = _shard_path(cache_dir, idx).exists() and _doclens_path(cache_dir, idx).exists()
        if save_token_ids:
            base_cached = base_cached and _token_ids_path(cache_dir, idx).exists()
        if base_cached:
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

    model_kwargs_cfg = cfg.model.get("model_kwargs", {})
    model_kwargs = {}
    if model_kwargs_cfg:
        dtype_str = model_kwargs_cfg.get("torch_dtype", None)
        if dtype_str is not None:
            model_kwargs["torch_dtype"] = getattr(torch, dtype_str)
    trust_remote_code = bool(cfg.model.get("trust_remote_code", False))

    prof = active()
    n_docs_to_encode = sum(
        min((idx + 1) * shard_size, num_documents) - idx * shard_size
        for idx in to_encode
    )
    with prof.span("encode_docs", count=n_docs_to_encode):
        with prof.span("model_init"):
            model = models.ColBERT(
                model_name_or_path=model_name,
                document_length=cfg.doc_length,
                device="cpu" if use_multi_gpu else None,
                trust_remote_code=trust_remote_code,
                **({"model_kwargs": model_kwargs} if model_kwargs else {}),
            )
            if cfg.get("compile", False) and not use_multi_gpu:
                model = torch.compile(model)

        pool = None
        if use_multi_gpu:
            logger.info("Starting multi-GPU encoding pool (%d GPUs).", n_gpus)
            with prof.span("start_pool"):
                pool = model.start_multi_process_pool()

        encode_start = time.perf_counter()

        for shard_idx in to_encode:
            start = shard_idx * shard_size
            end = min(start + shard_size, num_documents)
            n_shard_docs = end - start
            logger.info("Encoding shard %d/%d (docs %d-%d)", shard_idx + 1, num_shards, start, end - 1)

            shard_token_ids = None
            with prof.span(
                "model_encode",
                device="cuda" if torch.cuda.is_available() and not use_multi_gpu else "cpu",
                count=n_shard_docs,
                shard=shard_idx,
                batch_size=cfg.encode.batch_size,
            ):
                if use_multi_gpu:
                    shard_embeddings = model.encode_multi_process(
                        sentences=[_doc_payload(doc) for doc in documents[start:end]],
                        pool=pool,
                        batch_size=cfg.encode.batch_size,
                        is_query=False,
                        pool_factor=pool_factor,
                        protected_tokens=cfg.encode.get("protected_tokens", 1),
                    )
                else:
                    if save_token_ids:
                        # output_value=None returns unfiltered embeddings + mask + input_ids.
                        encode_result = model.encode(
                            sentences=[_doc_payload(doc) for doc in documents[start:end]],
                            batch_size=cfg.encode.batch_size,
                            is_query=False,
                            show_progress_bar=True,
                            pool_factor=pool_factor,
                            protected_tokens=cfg.encode.get("protected_tokens", 1),
                            output_value=None,
                        )
                    else:
                        shard_embeddings = model.encode(
                            sentences=[_doc_payload(doc) for doc in documents[start:end]],
                            batch_size=cfg.encode.batch_size,
                            is_query=False,
                            show_progress_bar=True,
                            pool_factor=pool_factor,
                            protected_tokens=cfg.encode.get("protected_tokens", 1),
                        )
                        encode_result = None

            if not use_multi_gpu and save_token_ids:
                with prof.span("filter_tokens", count=n_shard_docs, shard=shard_idx):
                    shard_embeddings = []
                    shard_token_ids = []
                    for emb, mask, ids in zip(
                        encode_result["token_embeddings"],
                        encode_result["masks"],
                        encode_result["input_ids"],
                    ):
                        filtered = emb[mask]
                        if hasattr(filtered, "cpu"):
                            filtered = filtered.cpu().numpy()
                        shard_embeddings.append(np.asarray(filtered))
                        filtered_ids = ids[mask]
                        if hasattr(filtered_ids, "cpu"):
                            filtered_ids = filtered_ids.cpu().numpy()
                        shard_token_ids.append(np.asarray(filtered_ids, dtype=np.int64))

            # Convert to numpy, cast to target dtype, and save
            save_dtype = NUMPY_DTYPES.get(cfg.encode.get("dtype", "fp32"), np.float32)
            with prof.span("prepare_shard", count=n_shard_docs, shard=shard_idx):
                doclens = []
                all_tokens = []
                for emb in shard_embeddings:
                    if isinstance(emb, torch.Tensor):
                        emb = emb.cpu().numpy()
                    doclens.append(emb.shape[0])
                    all_tokens.append(emb.astype(save_dtype))
                concatenated = np.concatenate(all_tokens, axis=0)

            with prof.span("save_shard", count=n_shard_docs, shard=shard_idx):
                np.save(_shard_path(cache_dir, shard_idx), concatenated)
                np.save(
                    _doclens_path(cache_dir, shard_idx),
                    np.array(doclens, dtype=np.int32),
                )

                if shard_token_ids is not None:
                    np.save(
                        _token_ids_path(cache_dir, shard_idx),
                        np.concatenate(shard_token_ids),
                    )

        encode_time = time.perf_counter() - encode_start

        with prof.span("cleanup"):
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
        with active().span("load_cached_queries", count=len(queries)):
            with active().span("load_cache_arrays", count=len(queries)):
                data = np.load(emb_path)
                doclens = np.load(doclens_path)
            embeddings = []
            offset = 0
            with active().span("reconstruct_query_tensors", count=len(queries)):
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

    model_kwargs_cfg = cfg.model.get("model_kwargs", {})
    model_kwargs = {}
    if model_kwargs_cfg:
        dtype_str = model_kwargs_cfg.get("torch_dtype", None)
        if dtype_str is not None:
            model_kwargs["torch_dtype"] = getattr(torch, dtype_str)
    trust_remote_code = bool(cfg.model.get("trust_remote_code", False))

    prof = active()
    with prof.span("encode_queries", count=len(queries)):
        with prof.span("model_init"):
            model = models.ColBERT(
                model_name_or_path=model_name,
                query_length=query_length,
                trust_remote_code=trust_remote_code,
                **({"model_kwargs": model_kwargs} if model_kwargs else {}),
            )
            if cfg.get("compile", False):
                model = torch.compile(model)

        encode_start = time.perf_counter()
        if prof.enabled:
            query_embeddings = []
            for query_text in tqdm(
                list(queries.values()),
                desc="Encoding queries (profile bs=1)",
                unit="query",
            ):
                with prof.span(
                    "encode_query",
                    device="cuda" if torch.cuda.is_available() else "cpu",
                    count=1,
                    batch_size=1,
                ):
                    encoded = model.encode(
                        sentences=[query_text],
                        is_query=True,
                        show_progress_bar=False,
                        batch_size=1,
                    )
                if isinstance(encoded, (list, tuple)):
                    query_embeddings.append(encoded[0])
                elif isinstance(encoded, (torch.Tensor, np.ndarray)) and encoded.ndim == 3:
                    query_embeddings.append(encoded[0])
                else:
                    query_embeddings.append(encoded)
        else:
            query_embeddings = model.encode(
                sentences=list(queries.values()),
                is_query=True,
                show_progress_bar=True,
                batch_size=cfg.encode.query_batch_size,
            )
        encode_time = time.perf_counter() - encode_start

        # Save to .npy
        with prof.span("prepare_cache", count=len(queries)):
            doclens = []
            all_tokens = []
            for emb in query_embeddings:
                if isinstance(emb, torch.Tensor):
                    emb_np = emb.cpu().numpy()
                else:
                    emb_np = emb
                doclens.append(emb_np.shape[0])
                all_tokens.append(emb_np)

        with prof.span("save_cache", count=len(queries)):
            np.save(emb_path, np.concatenate(all_tokens, axis=0))
            np.save(doclens_path, np.array(doclens, dtype=np.int32))

        with prof.span("cleanup"):
            del model
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    return query_embeddings, encode_time


# ---------------------------------------------------------------------------
# Shard loading
# ---------------------------------------------------------------------------


def load_shards_to_memory(
    shard_dir: Path,
    dtype: torch.dtype = torch.float32,
    load_token_ids: bool = False,
) -> list[torch.Tensor] | tuple[list[torch.Tensor], list[np.ndarray]]:
    """Load all .npy + .doclens.npy shards into a list of per-document tensors.

    If load_token_ids is True and .token_ids.npy files exist alongside shards,
    also returns a parallel list of per-document int64 numpy arrays with vocabulary
    token IDs for each embedding vector.
    """
    npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
    npy_files = [f for f in npy_files if not f.name.endswith((".doclens.npy", ".token_ids.npy"))]

    embeddings = []
    token_ids_list = [] if load_token_ids else None
    for npy_file in npy_files:
        doclens_file = npy_file.with_suffix(".doclens.npy")
        data = np.load(npy_file)
        doclens = np.load(doclens_file)

        tid_file = npy_file.parent / npy_file.name.replace(".npy", ".token_ids.npy")
        tid_data = None
        if load_token_ids and tid_file.exists():
            tid_data = np.load(tid_file)

        offset = 0
        for length in doclens:
            emb = torch.from_numpy(data[offset:offset + length].copy())
            if emb.dtype != dtype:
                emb = emb.to(dtype)
            embeddings.append(emb)
            if load_token_ids and tid_data is not None:
                token_ids_list.append(tid_data[offset:offset + length].copy())
            offset += length

    if load_token_ids:
        return embeddings, token_ids_list
    return embeddings


def load_shards_flat(
    shard_dir: Path,
    max_workers: int | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    """Optimized drop-in replacement for load_shards_flat.

    Three improvements over the original:
      1. Two-pass pre-allocation: doclens (tiny) are loaded first to compute
         total token count and per-shard offsets, then a single output array is
         allocated at the final dtype — eliminates np.concatenate and the
         intermediate Python list of mmap'd arrays (~2× peak memory in original).
      2. In-place cast: slice-assignment into the pre-allocated fp16 buffer casts
         directly from the source dtype with no intermediate .astype() copy.
      3. Thread-parallel I/O: shards are loaded concurrently via
         ThreadPoolExecutor; each thread writes to a non-overlapping slice so no
         locking is needed.

    max_workers defaults to SLURM_CPUS_PER_TASK if set, otherwise os.cpu_count().
    """
    if max_workers is None:
        slurm_cpus = os.environ.get("SLURM_CPUS_PER_TASK")
        max_workers = int(slurm_cpus) if slurm_cpus else (os.cpu_count() or 16)
    from concurrent.futures import ThreadPoolExecutor, as_completed

    npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
    npy_files = [f for f in npy_files if not f.name.endswith((".doclens.npy", ".token_ids.npy"))]

    if not npy_files:
        return np.empty((0, 0), dtype=np.float16), np.empty(0, dtype=np.int32), None

    # Pass 1: load only doclens (small) to compute sizes and offsets.
    all_doclens = [
        np.load(f.parent / f.name.replace(".npy", ".doclens.npy"))
        for f in npy_files
    ]
    shard_token_counts = [int(d.sum()) for d in all_doclens]
    shard_doc_counts = [len(d) for d in all_doclens]
    total_tokens = sum(shard_token_counts)
    total_docs = sum(shard_doc_counts)

    has_tids = all(
        (f.parent / f.name.replace(".npy", ".token_ids.npy")).exists()
        for f in npy_files
    )

    # Peek at embedding dim from the first shard header (no data read).
    first_mmap = np.load(npy_files[0], mmap_mode="r")
    emb_dim = first_mmap.shape[1]
    del first_mmap

    # Pre-allocate output arrays at final dtype — no intermediate buffers needed.
    flat_embs = np.empty((total_tokens, emb_dim), dtype=np.float16)
    flat_doclens = np.empty(total_docs, dtype=np.int32)
    flat_tids = np.empty(total_tokens, dtype=np.int64) if has_tids else None

    # Compute per-shard token and doc offsets.
    token_offsets = [0] * (len(npy_files) + 1)
    doc_offsets = [0] * (len(npy_files) + 1)
    for i in range(len(npy_files)):
        token_offsets[i + 1] = token_offsets[i] + shard_token_counts[i]
        doc_offsets[i + 1] = doc_offsets[i] + shard_doc_counts[i]

    # Fill doclens immediately — already in memory from pass 1.
    for i, dl in enumerate(all_doclens):
        flat_doclens[doc_offsets[i]:doc_offsets[i + 1]] = dl

    def _load_shard(i: int) -> None:
        t0, t1 = token_offsets[i], token_offsets[i + 1]
        data = np.load(npy_files[i])
        flat_embs[t0:t1] = data  # cast in-place into fp16 destination
        del data
        if has_tids:
            tid_file = npy_files[i].parent / npy_files[i].name.replace(".npy", ".token_ids.npy")
            tid_data = np.load(tid_file)
            flat_tids[t0:t1] = tid_data
            del tid_data

    n_workers = min(max_workers, len(npy_files))
    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_load_shard, i): i for i in range(len(npy_files))}
        with tqdm(total=len(npy_files), desc="Loading shards", unit="shard") as pbar:
            for fut in as_completed(futures):
                fut.result()
                pbar.update(1)

    return flat_embs, flat_doclens, flat_tids


def count_doc_tokens_from_shards(shard_dir: Path) -> int:
    """Count total token embeddings across all shards."""
    total = 0
    for npy_file in sorted(shard_dir.glob("doc_shard_*.npy")):
        if npy_file.name.endswith((".doclens.npy", ".token_ids.npy")):
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

    import math
    n_samples_kmeans = min(1 + int(16 * math.sqrt(120 * num_docs)), num_docs)

    use_triton = index_cfg.get("use_triton", None) if index_cfg else None
    verbose = index_cfg.get("verbose", False) if index_cfg else False

    if index_type == "warp":
        devices_cfg = index_cfg.get("devices", None) if index_cfg else None
        if devices_cfg is not None:
            warp_device: str | dict = OmegaConf.to_container(devices_cfg, resolve=True)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            n = torch.cuda.device_count()
            warp_device = {f"cuda:{i}": 1 / n for i in range(n)}
            logger.info(f"WARP multi-device: {warp_device}")
        else:
            warp_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        warp_batch_size = index_cfg.get("batch_size", 8192) if index_cfg else 8192
        warp_override = index_cfg.get("override", True) if index_cfg else True
        logger.info(f"WARP batch_size={warp_batch_size}, override={warp_override}")

        warp_index_path = os.path.join(index_folder, index_name, "warp_index", "metadata.json")
        if not warp_override and os.path.exists(warp_index_path):
            logger.info("WARP index already exists and override=False — skipping build, loading existing.")
            index = load_existing_index(
                index_type=index_type,
                index_folder=index_folder,
                index_name=index_name,
                device=device,
                index_cfg=index_cfg,
            )
            build_time = 0.0
            disk_mb = get_dir_size_mb(os.path.join(index_folder, index_name))
            return index, build_time, disk_mb

        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=warp_override,
            device=warp_device,
            nbits=nbits,
            n_samples_kmeans=n_samples_kmeans,
            use_triton=use_triton,
            n_ivf_probe=index_cfg.get("n_ivf_probe", 32) if index_cfg else 32,
            bound=index_cfg.get("bound", None) if index_cfg else None,
            t_prime=index_cfg.get("t_prime", 100_000) if index_cfg else 100_000,
            max_candidates=index_cfg.get("max_candidates", None) if index_cfg else None,
            centroid_score_threshold=index_cfg.get("centroid_score_threshold", None) if index_cfg else None,
            batch_size=warp_batch_size,
            show_progress=verbose,
        )
        # Stream shards shard-by-shard: first call creates the index (centroid training),
        # subsequent calls use warp.add.
        npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
        npy_files = [f for f in npy_files if not f.name.endswith((".doclens.npy", ".token_ids.npy"))]

        # The initial batch must have >= n_samples_kmeans *documents* — xtr_warp's
        # kmeans samples passages (documents), not tokens.  Counting tokens here
        # caused crashes on small/long-document corpora where enough tokens
        # accumulated across far fewer documents than n_samples_kmeans.
        initial_shards = 0
        initial_doc_count = 0
        for npy_file in npy_files:
            doclens_file = npy_file.parent / npy_file.name.replace(".npy", ".doclens.npy")
            initial_doc_count += len(np.load(doclens_file))
            initial_shards += 1
            if initial_doc_count >= n_samples_kmeans:
                break

        doc_offset = 0
        build_start = time.perf_counter()

        initial_embeddings = []
        initial_ids = []
        for shard_file in npy_files[:initial_shards]:
            doclens_file = shard_file.parent / shard_file.name.replace(".npy", ".doclens.npy")
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
            "WARP: initial create with %d shards (%d docs, n_samples_kmeans=%d)",
            initial_shards, len(initial_ids), n_samples_kmeans,
        )
        index.add_documents(documents_ids=initial_ids, documents_embeddings=initial_embeddings)
        del initial_embeddings, initial_ids
        gc.collect()

        add_batch_size = max(int(index_cfg.get("add_batch_size", 4)) if index_cfg else 4, 1)
        remaining_files = npy_files[initial_shards:]
        for batch_start in tqdm(
            range(0, len(remaining_files), add_batch_size),
            desc="WARP add batches",
            unit="batch",
        ):
            batch_files = remaining_files[batch_start:batch_start + add_batch_size]
            batch_embeddings = []
            batch_ids = []
            for shard_file in batch_files:
                doclens_file = shard_file.parent / shard_file.name.replace(".npy", ".doclens.npy")
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
            index.add_documents(documents_ids=batch_ids, documents_embeddings=batch_embeddings)
            del batch_embeddings, batch_ids
            gc.collect()

        build_time = time.perf_counter() - build_start

    elif index_type in ("plaid", "fast_plaid"):
        plaid_kwargs = {}
        plaid_search_batch_size = index_cfg.get("search_batch_size", None) if index_cfg else None
        if plaid_search_batch_size is not None:
            plaid_kwargs["batch_size"] = int(plaid_search_batch_size)
        n_full_scores = index_cfg.get("n_full_scores", None) if index_cfg else None
        if n_full_scores is not None:
            plaid_kwargs["n_full_scores"] = int(n_full_scores)
        centroid_score_threshold = index_cfg.get("centroid_score_threshold", None) if index_cfg else None
        if centroid_score_threshold is not None:
            plaid_kwargs["centroid_score_threshold"] = float(centroid_score_threshold)
        use_fast = index_cfg.get("use_fast", True) if index_cfg else True
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            nbits=nbits,
            n_samples_kmeans=n_samples_kmeans,
            use_fast=use_fast,
            use_triton=use_triton,
            show_progress=verbose,
            **plaid_kwargs,
        )
        npy_files = sorted(shard_dir.glob("doc_shard_*.npy"))
        npy_files = [f for f in npy_files if not f.name.endswith((".doclens.npy", ".token_ids.npy"))]

        # Same fix as WARP: count documents, not tokens.
        initial_shards = 0
        initial_doc_count = 0
        for npy_file in npy_files:
            doclens_file = npy_file.parent / npy_file.name.replace(".npy", ".doclens.npy")
            initial_doc_count += len(np.load(doclens_file))
            initial_shards += 1
            if initial_doc_count >= n_samples_kmeans:
                break

        doc_offset = 0
        build_start = time.perf_counter()

        initial_embeddings = []
        initial_ids = []
        for shard_file in npy_files[:initial_shards]:
            doclens_file = shard_file.parent / shard_file.name.replace(".npy", ".doclens.npy")
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
        index.add_documents(documents_ids=initial_ids, documents_embeddings=initial_embeddings)
        del initial_embeddings, initial_ids
        gc.collect()

        update_shard_batch_size = max(int(index_cfg.get("add_batch_size", 4)) if index_cfg else 4, 1)
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
                doclens_file = shard_file.parent / shard_file.name.replace(".npy", ".doclens.npy")
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
            index.add_documents(documents_ids=batch_ids, documents_embeddings=batch_embeddings)
            del batch_embeddings, batch_ids
            gc.collect()

        # FastPlaid keeps both per-shard ({i}.codes/residuals.npy, used for index
        # adds) and merged_* (used at search) files — ~2x on-disk residuals. For a
        # read-only benchmark index the per-shard files are redundant; freeze() drops
        # them so disk_mb below reflects the true searchable footprint (~halved).
        # Requires fast-plaid >= 1.4.7 (freeze() added in 1.4.7); guarded for the
        # Stanford backend, which has no freeze().
        if use_fast:
            try:
                logger.info("Freezing PLAID index (drops redundant per-shard files)...")
                index.freeze()
            except (AttributeError, RuntimeError) as exc:
                # fast-plaid < 1.4.7 has no freeze(); leave the index un-frozen
                # (usable, just ~2x disk) rather than failing a completed build.
                logger.warning(
                    "freeze() unavailable (need fast-plaid >= 1.4.7); "
                    "index left un-frozen, disk_mb will include redundant shards: %s",
                    exc,
                )

        build_time = time.perf_counter() - build_start

    elif index_type == "scann":
        index = indexes.ScaNN(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            store_embeddings=True,
        )
        has_token_ids = any(shard_dir.glob("doc_shard_*.token_ids.npy"))
        logger.info(
            "Loading all shards into memory for ScaNN (token_ids=%s)...",
            has_token_ids,
        )
        load_result = load_shards_to_memory(shard_dir, load_token_ids=has_token_ids)
        if has_token_ids:
            documents_embeddings, documents_token_ids = load_result
        else:
            documents_embeddings = load_result
            documents_token_ids = None
        build_start = time.perf_counter()
        # ScaNN scores on embeddings only; token IDs are a tachiom (TAC) concern,
        # and ScaNN.add_documents does not accept them.
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=documents_embeddings,
        )
        build_time = time.perf_counter() - build_start
        del documents_embeddings, documents_token_ids
        gc.collect()

    elif index_type == "tachiom":
        cl = index_cfg.clustering if index_cfg else {}
        clustering_type = cl.get("type", "tac") if cl else "tac"
        index = indexes.TachiomIndex(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            # Official build params
            center_dataset=index_cfg.get("center_dataset", True) if index_cfg else True,
            total_centroids=index_cfg.get("total_centroids", None) if index_cfg else None,
            tac_n_iter=cl.get("n_iter", 10) if cl else 10,
            tac_micro_threshold=cl.get("micro_threshold", None) if cl else None,
            tac_small_threshold=cl.get("small_threshold", None) if cl else None,
            pq_sample_size=index_cfg.get("pq_sample_size", 10_000_000) if index_cfg else 10_000_000,
            pq_n_iter=index_cfg.get("pq_n_iter", 10) if index_cfg else 10,
            normalize=index_cfg.get("normalize", True) if index_cfg else True,
            pq_seed=index_cfg.get("pq_seed", 42) if index_cfg else 42,
            pq_subspaces=index_cfg.get("pq_subspaces", 32) if index_cfg else 32,
            hnsw_m=index_cfg.get("hnsw_m", 32) if index_cfg else 32,
            ef_construction=index_cfg.get("ef_construction", 1500) if index_cfg else 1500,
            # Official search params
            k_centroids=index_cfg.get("k_centroids", 20) if index_cfg else 20,
            k_docs_to_score=index_cfg.get("k_docs_to_score", 500) if index_cfg else 500,
            ef_search=index_cfg.get("ef_search", None) if index_cfg else None,
            alpha=index_cfg.get("alpha", 0.45) if index_cfg else 0.45,
            beta=index_cfg.get("beta", None) if index_cfg else None,
            lambda_=index_cfg.get("lambda_", None) if index_cfg else None,
            impute_missing=index_cfg.get("impute_missing", False) if index_cfg else False,
            gap_relative=index_cfg.get("gap_relative", False) if index_cfg else False,
            num_threads=index_cfg.get("num_threads", 0) if index_cfg else 0,
            # Local PGC extension
            clustering=clustering_type,
            pgc_n_iter=cl.get("n_iter", 10) if cl else 10,
            pgc_sample_multiplier=cl.get("sample_multiplier", 5) if cl else 5,
            pgc_empty_strategy=cl.get("empty_strategy", "resample") if cl else "resample",
            pgc_iter_hnsw_m=cl.get("iter_hnsw_m", 16) if cl else 16,
            pgc_iter_ef_construction=cl.get("iter_ef_construction", 200) if cl else 200,
            pgc_iter_ef_search=cl.get("iter_ef_search", 50) if cl else 50,
            pgc_iter_lambda=cl.get("iter_lambda", None) if cl else None,
            pgc_assign_topm=cl.get("assign_topm", 1) if cl else 1,
            pgc_assign_temp=cl.get("assign_temp", 0.1) if cl else 0.1,
            pgc_seed=cl.get("seed", 42) if cl else 42,
            external_centroids_path=cl.get("centroids_path", None) if cl else None,
            external_assignments_path=cl.get("assignments_path", None) if cl else None,
        )
        # Tachiom reads shards directly in Rust — no Python flat buffer needed.
        logger.info("Building Tachiom index from shards (Rust native path)...")
        build_start = time.perf_counter()
        index.add_documents_from_shards(
            documents_ids=doc_ids,
            shard_dir=shard_dir,
            glob_pattern="doc_shard_*.npy",
        )
        build_time = time.perf_counter() - build_start
        gc.collect()

    elif index_type == "chimera":
        index = indexes.Chimera(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            # Build params (baked into the artifact)
            n_clusters=index_cfg.get("n_clusters", None) if index_cfg else None,
            tokens_per_cluster=index_cfg.get("tokens_per_cluster", 150) if index_cfg else 150,
            ex_bits=index_cfg.get("ex_bits", 4) if index_cfg else 4,
            # Search params (bound at load time by the C++ index)
            nprobe=index_cfg.get("nprobe", 128) if index_cfg else 128,
            k_refine=index_cfg.get("k_refine", 3000) if index_cfg else 3000,
            k_full_bit=index_cfg.get("k_full_bit", 300) if index_cfg else 300,
            cagra_itopk_size=index_cfg.get("cagra_itopk_size", None) if index_cfg else None,
            num_chunks=index_cfg.get("num_chunks", 5) if index_cfg else 5,
        )
        # Chimera's build() is one shot over the whole corpus and copies the
        # float32 array into a std::vector before clustering, so this peaks at
        # ~2x the flat token buffer. Unlike WARP/PLAID there is no shard-by-shard
        # add to spread that out.
        logger.info("Building Chimera index from shards (single-shot C++/CUDA build)...")
        build_start = time.perf_counter()
        index.add_documents_from_shards(
            documents_ids=doc_ids,
            shard_dir=shard_dir,
            glob_pattern="doc_shard_*.npy",
        )
        build_time = time.perf_counter() - build_start
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
    warp_outer_batch_size: int | None = None,
    run_save_path: str | None = None,
    metrics: list | None = None,
    profile: bool = False,
    maxsim_backend: str | None = None,
    e2e_profile_batch_size: int | None = None,
) -> dict:
    """Run search on a pre-built index and collect metrics.

    ``maxsim_backend`` selects the maxsim scoring kernel on the ColBERT token
    path (scann + colbert): auto | torch | flash | lik; ignored by XTR /
    end-to-end indices. ``profile`` attaches a CUDA-synced
    :class:`pylate.Profiler` and records a per-stage ``profile`` breakdown
    (token path only).
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if index_type == "warp":
        retriever = retrieve.XTR(index=index)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif index_type in ("plaid", "fast_plaid", "tachiom", "chimera"):
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif index_type == "scann" and retrieval == "xtr":
        retriever = retrieve.XTR(index=index)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu",
        )
    elif index_type == "scann" and retrieval == "colbert":
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu", maxsim_backend=maxsim_backend,
        )
    else:
        raise ValueError(f"Unknown index_type/retrieval combo: {index_type}/{retrieval}")

    if profile:
        from pylate.profiling import Profiler

        retriever.profiler = Profiler(cuda_sync=True)
        # Per-query latency mode: bs=1 makes every stage span count=1 (clean
        # latency-grade p50/p90) instead of amortizing index_lookup/gather over
        # a batch. Token path only (retrieve_kwargs carries batch_size there).
        if "k_token" in retrieve_kwargs:
            retrieve_kwargs["batch_size"] = 1

    profile_roots = None

    if torch.cuda.is_available():
        torch.cuda.reset_peak_memory_stats()
    tracemalloc.start()

    search_start = time.perf_counter()
    if (
        profile
        and e2e_profile_batch_size is not None
        and index_type in ("warp", "plaid", "fast_plaid", "tachiom", "chimera")
    ):
        # End-to-end indexes normally profile as one coarse span over the full
        # query set. This opt-in path runs smaller outer batches, commonly bs=1,
        # so the coarse E2E span is latency-grade even before Rust internals are
        # instrumented.
        outer_batch_size = max(int(e2e_profile_batch_size), 1)
        scores = []
        profile_roots = []
        for start in tqdm(
            range(0, len(queries_embeddings), outer_batch_size),
            desc=f"{index_type} profile batches",
            unit="batch",
        ):
            end = start + outer_batch_size
            scores.extend(
                retriever.retrieve(
                    queries_embeddings=queries_embeddings[start:end],
                    k=k,
                )
            )
            last = getattr(retriever, "last_profile", None)
            if last:
                profile_roots.extend(last)
    elif index_type == "warp" and warp_outer_batch_size is not None:
        outer_batch_size = max(int(warp_outer_batch_size), 1)
        scores = []
        for start in tqdm(
            range(0, len(queries_embeddings), outer_batch_size),
            desc="WARP search batches",
            unit="batch",
        ):
            end = start + outer_batch_size
            if torch.cuda.is_available():
                alloc_before = torch.cuda.memory_allocated() / 1024**3
                reserved_before = torch.cuda.memory_reserved() / 1024**3
            scores.extend(
                retriever.retrieve(
                    queries_embeddings=queries_embeddings[start:end],
                    k=k,
                )
            )
            if torch.cuda.is_available():
                alloc_after = torch.cuda.memory_allocated() / 1024**3
                reserved_after = torch.cuda.memory_reserved() / 1024**3
                torch.cuda.empty_cache()
                alloc_cache = torch.cuda.memory_allocated() / 1024**3
                reserved_cache = torch.cuda.memory_reserved() / 1024**3
                logger.info(
                    f"WARP batch q{start}-{end}: "
                    f"alloc {alloc_before:.2f}->{alloc_after:.2f}->{alloc_cache:.2f} GB  "
                    f"reserved {reserved_before:.2f}->{reserved_after:.2f}->{reserved_cache:.2f} GB"
                )
    elif index_type in ("plaid", "fast_plaid"):
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

    profile_summary = None
    if profile:
        from pylate.profiling import reduce_stage_timings

        last = profile_roots if profile_roots is not None else getattr(retriever, "last_profile", None)
        if last:
            profile_summary = reduce_stage_timings(last)

    return {
        "search_time_s": round(search_time, 4),
        "qps": round(qps, 2),
        "peak_vram_mb": round(peak_vram_bytes / (1024 * 1024), 2),
        "peak_ram_mb": round(peak_ram_bytes / (1024 * 1024), 2),
        "n_queries": n_queries,
        "maxsim_backend": maxsim_backend,
        "profile": profile_summary,
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
    use_triton = index_cfg.get("use_triton", None) if index_cfg else None
    verbose = index_cfg.get("verbose", False) if index_cfg else False

    if index_type == "warp":
        devices_cfg = index_cfg.get("devices", None) if index_cfg else None
        if devices_cfg is not None:
            warp_device: str | dict = OmegaConf.to_container(devices_cfg, resolve=True)
        elif torch.cuda.is_available() and torch.cuda.device_count() > 1:
            n = torch.cuda.device_count()
            warp_device = {f"cuda:{i}": 1 / n for i in range(n)}
            logger.info(f"WARP multi-device: {warp_device}")
        else:
            warp_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        warp_batch_size = index_cfg.get("batch_size", 8192) if index_cfg else 8192
        logger.info(f"WARP batch_size={warp_batch_size}")
        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            device=warp_device,
            nbits=nbits,
            use_triton=use_triton,
            n_ivf_probe=index_cfg.get("n_ivf_probe", 32) if index_cfg else 32,
            bound=index_cfg.get("bound", None) if index_cfg else None,
            t_prime=index_cfg.get("t_prime", 100_000) if index_cfg else 100_000,
            max_candidates=index_cfg.get("max_candidates", None) if index_cfg else None,
            centroid_score_threshold=index_cfg.get("centroid_score_threshold", None) if index_cfg else None,
            batch_size=warp_batch_size,
            show_progress=verbose,
        )
    elif index_type in ("plaid", "fast_plaid"):
        plaid_kwargs = {}
        plaid_search_batch_size = index_cfg.get("search_batch_size", None) if index_cfg else None
        if plaid_search_batch_size is not None:
            plaid_kwargs["batch_size"] = int(plaid_search_batch_size)
        n_full_scores = index_cfg.get("n_full_scores", None) if index_cfg else None
        if n_full_scores is not None:
            plaid_kwargs["n_full_scores"] = int(n_full_scores)
        centroid_score_threshold = index_cfg.get("centroid_score_threshold", None) if index_cfg else None
        if centroid_score_threshold is not None:
            plaid_kwargs["centroid_score_threshold"] = float(centroid_score_threshold)
        use_fast = index_cfg.get("use_fast", True) if index_cfg else True
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            nbits=nbits,
            use_fast=use_fast,
            use_triton=use_triton,
            show_progress=verbose,
            **plaid_kwargs,
        )
    elif index_type == "scann":
        index = indexes.ScaNN(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
        )
    elif index_type == "tachiom":
        index = indexes.TachiomIndex(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            # Must match the M the index was built with (on-disk PQ is M-specific).
            pq_subspaces=index_cfg.get("pq_subspaces", 32) if index_cfg else 32,
            k_centroids=index_cfg.get("k_centroids", 20) if index_cfg else 20,
            k_docs_to_score=index_cfg.get("k_docs_to_score", 500) if index_cfg else 500,
            ef_search=index_cfg.get("ef_search", None) if index_cfg else None,
            alpha=index_cfg.get("alpha", 0.45) if index_cfg else 0.45,
            beta=index_cfg.get("beta", None) if index_cfg else None,
            lambda_=index_cfg.get("lambda_", None) if index_cfg else None,
            impute_missing=index_cfg.get("impute_missing", False) if index_cfg else False,
            gap_relative=index_cfg.get("gap_relative", False) if index_cfg else False,
            num_threads=index_cfg.get("num_threads", 0) if index_cfg else 0,
        )
    elif index_type == "chimera":
        # Build params are read back off the index's params.json; only the search
        # params are this instance's to choose, and the C++ index binds them at
        # load(), so they cannot be changed on a live index.
        index = indexes.Chimera(
            index_folder=index_folder,
            index_name=index_name,
            override=False,
            nprobe=index_cfg.get("nprobe", 128) if index_cfg else 128,
            k_refine=index_cfg.get("k_refine", 3000) if index_cfg else 3000,
            k_full_bit=index_cfg.get("k_full_bit", 300) if index_cfg else 300,
            cagra_itopk_size=index_cfg.get("cagra_itopk_size", None) if index_cfg else None,
            num_chunks=index_cfg.get("num_chunks", 5) if index_cfg else 5,
        )

    else:
        raise ValueError(f"Unknown index type: {index_type}")

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
        "cluster",
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

    # Full resolved index config dict — logged into every result row.
    index_config = OmegaConf.to_container(cfg.index, resolve=True)

    all_results = []

    for dataset_id in cfg.datasets:
        dataset_slug = sanitize_name(dataset_id)

        print(f"\n{'#' * 60}")
        print(f"# Dataset: {dataset_id}  stages: {ordered_stages}")
        print(f"{'#' * 60}")

        # Always load dataset metadata (needed for doc IDs, qrels, etc.)
        documents, queries, qrels = load_dataset(dataset_id)
        query_ids = list(queries.keys())

        # Multimodal (image) corpora are modality-agnostic from the index down,
        # with one hard incompatibility: TAC clustering keys on vocabulary token
        # types, which image patches do not have. Reject it loudly rather than
        # silently producing a degenerate clustering. Use clustering=pgc (or a
        # PLAID/WARP index) for multimodal datasets.
        if is_multimodal_corpus(documents) and index_type == "tachiom":
            cl_type = (cfg.index.get("clustering", {}) or {}).get("type", "tac")
            if cl_type == "tac":
                raise ValueError(
                    f"Dataset '{dataset_id}' is a multimodal (image) corpus, but "
                    "index=tachiom uses TAC clustering, which requires vocabulary "
                    "token IDs that image documents do not have. Re-run with "
                    "'index/clustering=pgc' (or index=plaid / index=warp)."
                )

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

        clustering_slug = ""
        if index_type == "tachiom":
            cl_cfg = cfg.index.get("clustering", {})
            clustering_type = cl_cfg.get("type", "tac") if cl_cfg else "tac"
            # pq_subspaces (M) is baked into the on-disk PQ, so it's part of the index
            # identity: different M (or clustering) must not share a name, else builds
            # collide and retrieve loads the wrong codebook. m{M} keeps them distinct.
            m = cfg.index.get("pq_subspaces", 32)
            # If the `cluster` stage will GENERATE the clustering (type=gpu, no explicit
            # centroids_path), pick a canonical out_dir up front — nested under a per-corpus
            # dir, named by backend+params — so the index name is stable and the build +
            # retrieve steps ingest the same files this run produces. The parent dir name
            # (used as the slug `src` below) needs only to disambiguate clustering *method*
            # within a corpus, since model+dataset are already in the index name.
            if clustering_type == "gpu" and "cluster" in stage_set and not cl_cfg.get("centroids_path"):
                bk = cl_cfg.get("backend", "cagra")
                name = bk
                if cl_cfg.get("k"):
                    name += f"_k{int(cl_cfg.get('k'))}"
                if cl_cfg.get("train_sample"):
                    name += f"_s{int(cl_cfg.get('train_sample'))}"
                gen_dir = (Path(cfg.output.get("clusterings_dir", "clusterings"))
                           / f"{dataset_slug}_{model_slug}" / name)
                cl_cfg.centroids_path = str(gen_dir / "centroids.npy")
                cl_cfg.assignments_path = str(gen_dir / "assignments.npy")
            # For external clustering, "external" alone doesn't say which method produced
            # the centroids — PGC and TAC would both land at _external_m{M} and collide,
            # silently overwriting each other's build. Tag with the provenance dir name
            # (parent of centroids_path, e.g. lotte_pgc_m4t05 / lotte_tac) to keep them
            # distinct and self-documenting.
            if clustering_type in ("external", "gpu"):
                cpath = cl_cfg.get("centroids_path") if cl_cfg else None
                src = Path(cpath).parent.name if cpath else "unknown"
                clustering_slug = f"_{clustering_type}_{src}_m{m}"
            else:
                clustering_slug = f"_{clustering_type}_m{m}"
        elif index_type == "chimera":
            # ex_bits and n_clusters are baked into the on-disk codes and the
            # CAGRA graph, so two settings must not share an index name — same
            # reasoning as tachiom's m{M} above.
            nc = cfg.index.get("n_clusters", None)
            nc_slug = f"_nc{int(nc)}" if nc else f"_tpc{cfg.index.get('tokens_per_cluster', 150)}"
            clustering_slug = f"_ex{cfg.index.get('ex_bits', 4)}{nc_slug}"
        index_name = f"bench_{model_slug}_{dataset_slug}_{index_type}{clustering_slug}"
        # Re-snapshot the (possibly path-injected) index config for the result rows.
        index_config = OmegaConf.to_container(cfg.index, resolve=True)

        # Track timing for results
        doc_encode_time = 0.0
        query_encode_time = 0.0
        doc_encode_profile = None
        query_encode_profile = None
        cluster_time = 0.0
        build_time = 0.0
        disk_mb = 0.0
        n_doc_tokens = 0
        index = None

        # --- Stage: encode_docs ---
        if "encode_docs" in stage_set:
            append_stage_marker("encode_docs", "start", dataset_id, index_type)
            print(f"\n  Encoding {len(documents)} documents...")
            if cfg.search.get("profile", False):
                from pylate.profiling import Profiler, reduce_stage_timings, use

                prof = Profiler(cuda_sync=True)
                with use(prof):
                    shard_dir, doc_encode_time = encode_documents_sharded(
                        model_name=model_name,
                        documents=documents,
                        cache_dir=cache_dir,
                        cfg=cfg,
                    )
                if prof.roots:
                    doc_encode_profile = reduce_stage_timings(prof.roots)
            else:
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
            if cfg.search.get("profile", False):
                from pylate.profiling import Profiler, reduce_stage_timings, use

                prof = Profiler(cuda_sync=True)
                with use(prof):
                    queries_embeddings, query_encode_time = encode_queries(
                        model_name=model_name,
                        queries=queries,
                        cache_dir=cache_dir,
                        cfg=cfg,
                    )
                if prof.roots:
                    query_encode_profile = reduce_stage_timings(prof.roots)
            else:
                queries_embeddings, query_encode_time = encode_queries(
                    model_name=model_name,
                    queries=queries,
                    cache_dir=cache_dir,
                    cfg=cfg,
                )
            print(f"  Query encode time: {query_encode_time:.2f}s")
            append_stage_marker("encode_queries", "end", dataset_id, index_type)

        # --- Stage: cluster (precompute GPU coarse clustering -> centroids/assignments) ---
        # Runs scripts/gpu_cluster.py's core in-process and writes the centroids/assignments
        # that build_index then ingests via clustering=gpu. Checkpoint-skips if they already
        # exist, so re-running the pipeline is cheap. Needs a cuvs-capable GPU env in THIS
        # process (cu13 on L40S/A100, cu12 on V100 — both supported).
        if "cluster" in stage_set:
            append_stage_marker("cluster", "start", dataset_id, index_type)
            if index_type != "tachiom":
                raise ValueError("the 'cluster' stage applies only to index=tachiom.")
            cl_cfg = cfg.index.clustering
            if cl_cfg.get("type") != "gpu":
                raise ValueError(
                    "stages=[...,cluster,...] requires index/clustering=gpu (it generates the "
                    "centroids); use clustering=external to ingest a precomputed clustering.")
            if not shard_dir.exists():
                raise FileNotFoundError(
                    f"No doc shards at {shard_dir}; run the encode_docs stage first.")
            cpath, apath = Path(cl_cfg.centroids_path), Path(cl_cfg.assignments_path)
            if cpath.exists() and apath.exists():
                print(f"\n  Clustering already present at {cpath.parent} — skipping (checkpoint).")
            else:
                from gpu_cluster import cluster_tokens
                print(f"\n  Clustering {shard_dir} -> {cpath.parent} ...")
                t_cl = time.perf_counter()
                cluster_tokens(shard_dir, cpath.parent, cl_cfg)
                cluster_time = time.perf_counter() - t_cl
                print(f"  Cluster time: {cluster_time:.2f}s")
            append_stage_marker("cluster", "end", dataset_id, index_type)

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
            if queries_embeddings is None:
                query_cache = cache_dir / "queries"
                if (query_cache / "query_emb.npy").exists():
                    if cfg.search.get("profile", False):
                        from pylate.profiling import Profiler, reduce_stage_timings, use

                        prof = Profiler(cuda_sync=True)
                        with use(prof):
                            queries_embeddings, _ = encode_queries(
                                model_name=model_name,
                                queries=queries,
                                cache_dir=cache_dir,
                                cfg=cfg,
                            )
                        if prof.roots:
                            query_encode_profile = reduce_stage_timings(prof.roots)
                    else:
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

            sub_n = cfg.search.get("query_subsample", None)
            if sub_n:
                sub_path = os.path.join(
                    "results", "subsamples", f"{dataset_slug}_n{int(sub_n)}.json"
                )
                query_ids, queries_embeddings, qrels = subsample_queries(
                    query_ids, queries_embeddings, qrels, int(sub_n),
                    int(cfg.search.get("query_subsample_seed", 42)), sub_path,
                )

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
                        warp_outer_batch_size=cfg.index.get("warp_search_batch_size", None),
                        run_save_path=run_save_path,
                        metrics=list(cfg.search.metrics),
                        profile=bool(cfg.search.get("profile", False)),
                        maxsim_backend=cfg.search.get("maxsim_backend", None),
                        e2e_profile_batch_size=cfg.search.get(
                            "e2e_profile_batch_size", None
                        ),
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
                        "doc_encode_profile": doc_encode_profile,
                        "query_encode_profile": query_encode_profile,
                        "cluster_time_s": round(cluster_time, 2),
                        "build_time_s": round(build_time, 2),
                        "disk_mb": round(disk_mb, 2),
                        "index_config": index_config,
                        "pool_factor": pool_factor,
                        "pool_method": pool_method if pool_factor > 1 else "none",
                        "protected_tokens": protected_tokens,
                        "encode_dtype": emb_dtype,
                        "run_idx": run_idx,
                        "search_repeats": search_repeats,
                        **get_hardware_info(),
                        **get_index_config_dict(index),
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
                "doc_encode_profile": doc_encode_profile,
                "query_encode_profile": query_encode_profile,
                "cluster_time_s": round(cluster_time, 2),
                "build_time_s": round(build_time, 2),
                "disk_mb": round(disk_mb, 2),
                "index_config": index_config,
                "pool_factor": pool_factor,
                "pool_method": pool_method if pool_factor > 1 else "none",
                "protected_tokens": protected_tokens,
                **get_hardware_info(),
                **get_index_config_dict(index),
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
            f"  {r['dataset']:<30} {r['index_type']:<10} {str(r.get('retrieval','')):<8} "
            f"build={r['build_time_s']}s  disk={r['disk_mb']}MB  "
            f"qps={r.get('qps', 'N/A')}  "
            f"ndcg@10={r.get('ndcg@10', 'N/A')}  "
            f"r@100={r.get('recall@100', 'N/A')}"
        )

    print(f"\nResults appended to {cfg.output.results_file}")


if __name__ == "__main__":
    main()
