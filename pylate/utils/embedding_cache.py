from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Generator

import numpy as np

logger = logging.getLogger(__name__)

# Number of documents per shard written by encode_and_cache.
DEFAULT_SHARD_DOCS = 500_000


def model_slug(model_name: str) -> str:
    """Normalise a model name for use as a directory component."""
    return model_name.replace("/", "_")


def get_cache_dir(base: str, dataset_name: str, model_name: str) -> Path:
    return Path(base) / dataset_name / model_slug(model_name)


def cache_exists(cache_dir: str | Path) -> bool:
    """Return True if at least one shard is present in *cache_dir*."""
    d = Path(cache_dir)
    return d.is_dir() and any(d.glob("embeddings_*.npy"))


# ── Writing ───────────────────────────────────────────────────────────────────


def _write_shard(
    cache_dir: Path,
    shard_idx: int,
    doc_ids: list[str],
    embeddings: list[np.ndarray],
    token_ids: list[np.ndarray] | None,
) -> None:
    prefix = cache_dir / f"embeddings_{shard_idx}"
    doclens = np.array([e.shape[0] for e in embeddings], dtype=np.int32)
    vectors = np.concatenate(embeddings, axis=0).astype(np.float32)
    np.save(f"{prefix}.npy", np.ascontiguousarray(vectors))
    np.save(f"{prefix}.doclens.npy", doclens)
    np.save(f"{prefix}.doc_ids.npy", np.array(doc_ids, dtype=object))
    if token_ids is not None:
        tids = np.concatenate(token_ids, axis=0).astype(np.int64)
        np.save(f"{prefix}.token_ids.npy", np.ascontiguousarray(tids))


def encode_and_cache(
    model,
    sentences: list[str],
    doc_ids: list[str],
    cache_dir: str | Path,
    shard_size: int = DEFAULT_SHARD_DOCS,
    return_token_ids: bool = False,
    **encode_kwargs,
) -> tuple[list[np.ndarray], list[np.ndarray] | None]:
    """Encode *sentences* in shards and write each shard to *cache_dir*.

    Parameters
    ----------
    model
        A ``pylate.models.ColBERT`` instance.
    sentences
        Document texts to encode, aligned with *doc_ids*.
    doc_ids
        String document IDs, aligned with *sentences*.
    cache_dir
        Directory to write shards into. Created if absent.
    shard_size
        Maximum number of documents per shard.
    return_token_ids
        When True, also encode and cache vocabulary token IDs alongside
        embeddings (needed for Tachiom TAC).
    **encode_kwargs
        Additional keyword arguments forwarded to ``model.encode`` (e.g.
        ``batch_size``, ``show_progress_bar``).

    Returns
    -------
    (all_embeddings, all_token_ids)
        ``all_token_ids`` is ``None`` when ``return_token_ids=False``.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings: list[np.ndarray] = []
    all_token_ids: list[np.ndarray] | None = [] if return_token_ids else None

    n_shards = (len(sentences) + shard_size - 1) // shard_size
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(sentences))
        shard_sentences = sentences[start:end]
        shard_doc_ids = doc_ids[start:end]

        logger.info(
            "Encoding shard %d/%d (docs %d–%d)", shard_idx + 1, n_shards, start, end - 1
        )

        if return_token_ids:
            embs, tids = model.encode(
                sentences=shard_sentences,
                return_token_ids=True,
                **encode_kwargs,
            )
        else:
            embs = model.encode(sentences=shard_sentences, **encode_kwargs)
            tids = None

        _write_shard(cache_dir, shard_idx, shard_doc_ids, embs, tids)
        all_embeddings.extend(embs)
        if return_token_ids:
            all_token_ids.extend(tids)

    # Write metadata so callers can inspect the cache without loading shards.
    meta = {
        "n_shards": n_shards,
        "total_docs": len(sentences),
        "has_token_ids": return_token_ids,
        "dim": all_embeddings[0].shape[-1] if all_embeddings else None,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return all_embeddings, all_token_ids


# ── Reading ───────────────────────────────────────────────────────────────────


def _shard_paths(cache_dir: Path) -> list[Path]:
    """Return shard base paths in sorted order."""
    paths = sorted(cache_dir.glob("embeddings_*.npy"))
    # Exclude sidecars — keep only the primary vector files.
    return [p for p in paths if not any(p.name.endswith(s) for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy"))]


def iter_cached_shards(
    cache_dir: str | Path,
) -> Generator[tuple[list[str], list[np.ndarray], list[np.ndarray] | None], None, None]:
    """Yield ``(doc_ids, embeddings, token_ids)`` for each cached shard.

    ``token_ids`` is ``None`` for shards that have no ``.token_ids.npy``
    sidecar (e.g. shards written before token-ID caching was added).
    """
    cache_dir = Path(cache_dir)
    for vec_path in _shard_paths(cache_dir):
        prefix = cache_dir / vec_path.stem  # strip .npy

        vectors = np.load(vec_path, mmap_mode="r")
        doclens = np.load(f"{prefix}.doclens.npy", mmap_mode="r")

        # Reconstruct per-document embedding arrays from the flat shard.
        offsets = np.concatenate([[0], np.cumsum(doclens)])
        embs = [np.array(vectors[offsets[i] : offsets[i + 1]]) for i in range(len(doclens))]

        doc_ids_path = Path(f"{prefix}.doc_ids.npy")
        doc_ids = list(np.load(doc_ids_path, allow_pickle=True)) if doc_ids_path.exists() else [str(i) for i in range(len(doclens))]

        tid_path = Path(f"{prefix}.token_ids.npy")
        if tid_path.exists():
            tids_flat = np.load(tid_path, mmap_mode="r")
            tids = [np.array(tids_flat[offsets[i] : offsets[i + 1]]) for i in range(len(doclens))]
        else:
            tids = None

        yield doc_ids, embs, tids


def load_cached(
    cache_dir: str | Path,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray] | None]:
    """Load all cached shards into memory.

    Returns
    -------
    (doc_ids, embeddings, token_ids)
        ``token_ids`` is ``None`` when no shard has a ``.token_ids.npy`` sidecar.
    """
    all_doc_ids: list[str] = []
    all_embs: list[np.ndarray] = []
    all_tids: list[np.ndarray] = []
    has_tids = False

    for doc_ids, embs, tids in iter_cached_shards(cache_dir):
        all_doc_ids.extend(doc_ids)
        all_embs.extend(embs)
        if tids is not None:
            all_tids.extend(tids)
            has_tids = True

    return all_doc_ids, all_embs, (all_tids if has_tids else None)
