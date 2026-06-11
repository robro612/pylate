from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Generator

import numpy as np
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)

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
    vectors = np.concatenate(embeddings, axis=0).astype(np.float16)
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
    save_token_ids: bool = True,
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
    save_token_ids
        When True (default), encode with ``output_value=None`` and cache
        vocabulary token IDs alongside embeddings (needed for TachiomIndex TAC).
    **encode_kwargs
        Additional keyword arguments forwarded to ``model.encode`` (e.g.
        ``batch_size``, ``show_progress_bar``).

    Returns
    -------
    (all_embeddings, all_token_ids)
        ``all_token_ids`` is ``None`` when ``save_token_ids=False``.
        Embeddings are per-document float16 arrays of shape ``(n_tokens, dim)``
        with skiplist/padding tokens already removed.
    """
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    all_embeddings: list[np.ndarray] = []
    all_token_ids: list[np.ndarray] | None = [] if save_token_ids else None

    n_shards = (len(sentences) + shard_size - 1) // shard_size
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(sentences))
        shard_sentences = sentences[start:end]
        shard_doc_ids = doc_ids[start:end]

        logger.info(
            "Encoding shard %d/%d (docs %d–%d)", shard_idx + 1, n_shards, start, end - 1
        )

        if save_token_ids:
            # output_value=None returns a dict with unfiltered embeddings + masks.
            # Apply the mask here so cached shards store only valid tokens.
            result = model.encode(
                sentences=shard_sentences,
                output_value=None,
                **encode_kwargs,
            )
            embs = []
            tids = []
            for emb, mask, ids in zip(
                result["token_embeddings"], result["masks"], result["input_ids"]
            ):
                filtered_emb = emb[mask]
                if hasattr(filtered_emb, "cpu"):
                    filtered_emb = filtered_emb.cpu().numpy()
                embs.append(np.asarray(filtered_emb, dtype=np.float16))

                filtered_ids = ids[mask]
                if hasattr(filtered_ids, "cpu"):
                    filtered_ids = filtered_ids.cpu().numpy()
                tids.append(np.asarray(filtered_ids, dtype=np.int64))
        else:
            raw = model.encode(sentences=shard_sentences, **encode_kwargs)
            embs = [
                (e.cpu().numpy() if hasattr(e, "cpu") else np.asarray(e)).astype(np.float16)
                for e in raw
            ]
            tids = None

        _write_shard(cache_dir, shard_idx, shard_doc_ids, embs, tids)
        all_embeddings.extend(embs)
        if save_token_ids:
            all_token_ids.extend(tids)

    meta = {
        "n_shards": n_shards,
        "total_docs": len(sentences),
        "has_token_ids": save_token_ids,
        "dim": all_embeddings[0].shape[-1] if all_embeddings else None,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))

    return all_embeddings, all_token_ids


def save_to_cache(
    cache_dir: str | Path,
    doc_ids: list[str],
    embeddings: list[np.ndarray],
    token_ids: list[np.ndarray] | None = None,
    shard_size: int = DEFAULT_SHARD_DOCS,
) -> None:
    """Write pre-encoded embeddings to the standard shard format."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    n = len(doc_ids)
    n_shards = (n + shard_size - 1) // shard_size
    for shard_idx in range(n_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, n)
        _write_shard(
            cache_dir,
            shard_idx,
            doc_ids[start:end],
            embeddings[start:end],
            token_ids[start:end] if token_ids is not None else None,
        )

    meta = {
        "n_shards": n_shards,
        "total_docs": n,
        "has_token_ids": token_ids is not None,
        "dim": embeddings[0].shape[-1] if embeddings else None,
    }
    (cache_dir / "meta.json").write_text(json.dumps(meta, indent=2))


# ── Reading ───────────────────────────────────────────────────────────────────


def _shard_paths(cache_dir: Path) -> list[Path]:
    """Return shard base paths in sorted order."""
    paths = sorted(cache_dir.glob("embeddings_*.npy"))
    return [
        p
        for p in paths
        if not any(
            p.name.endswith(s)
            for s in (".doclens.npy", ".doc_ids.npy", ".token_ids.npy")
        )
    ]


def iter_cached_shards(
    cache_dir: str | Path,
) -> Generator[tuple[list[str], list[np.ndarray], list[np.ndarray] | None], None, None]:
    """Yield ``(doc_ids, embeddings, token_ids)`` for each cached shard.

    ``token_ids`` is ``None`` for shards that have no ``.token_ids.npy`` sidecar.
    """
    cache_dir = Path(cache_dir)
    for vec_path in _shard_paths(cache_dir):
        prefix = cache_dir / vec_path.stem

        vectors = np.load(vec_path, mmap_mode="r")
        doclens = np.load(f"{prefix}.doclens.npy", mmap_mode="r")

        offsets = np.empty(len(doclens) + 1, dtype=np.int64)
        offsets[0] = 0
        np.cumsum(doclens, dtype=np.int64, out=offsets[1:])
        split_indices = offsets[1:-1]
        embs = np.split(vectors, split_indices, axis=0)

        doc_ids_path = Path(f"{prefix}.doc_ids.npy")
        doc_ids = (
            list(np.load(doc_ids_path, allow_pickle=True))
            if doc_ids_path.exists()
            else [str(i) for i in range(len(doclens))]
        )

        tid_path = Path(f"{prefix}.token_ids.npy")
        if tid_path.exists():
            tids_flat = np.load(tid_path, mmap_mode="r")
            tids = np.split(tids_flat, split_indices, axis=0)
        else:
            tids = None

        yield doc_ids, embs, tids


def iter_cached_flat_shards(
    cache_dir: str | Path,
) -> Generator[tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None], None, None]:
    """Yield flat shard arrays ``(doc_ids, vectors, doclens, token_ids)``."""
    cache_dir = Path(cache_dir)
    for vec_path in _shard_paths(cache_dir):
        prefix = cache_dir / vec_path.stem

        vectors = np.load(vec_path, mmap_mode="r")
        doclens = np.load(f"{prefix}.doclens.npy", mmap_mode="r")

        doc_ids_path = Path(f"{prefix}.doc_ids.npy")
        doc_ids = (
            list(np.load(doc_ids_path, allow_pickle=True))
            if doc_ids_path.exists()
            else [str(i) for i in range(len(doclens))]
        )

        tid_path = Path(f"{prefix}.token_ids.npy")
        token_ids = np.load(tid_path, mmap_mode="r") if tid_path.exists() else None

        yield doc_ids, vectors, doclens, token_ids


def load_cached(
    cache_dir: str | Path,
) -> tuple[list[str], list[np.ndarray], list[np.ndarray] | None]:
    """Load all cached shards into memory.

    Returns
    -------
    (doc_ids, embeddings, token_ids)
        ``token_ids`` is ``None`` when no shard has a ``.token_ids.npy`` sidecar.
    """
    all_doc_ids = []
    all_embs = []
    all_tids = None

    shard_paths = list(_shard_paths(Path(cache_dir)))
    for doc_ids, embs, tids in tqdm(
        iter_cached_shards(cache_dir),
        total=len(shard_paths),
        desc="Loading cached shards",
    ):
        all_doc_ids.extend(doc_ids)
        all_embs.extend(embs)
        if tids is not None:
            if all_tids is None:
                all_tids = []
            all_tids.extend(tids)

    return all_doc_ids, all_embs, all_tids


def load_cached_flat(
    cache_dir: str | Path,
) -> tuple[list[str], np.ndarray, np.ndarray, np.ndarray | None]:
    """Load all cached shards as flattened arrays.

    Returns
    -------
    (doc_ids, vectors, doclens, token_ids)
        ``vectors`` shape ``[total_tokens, dim]`` dtype float16.
        ``doclens`` shape ``[n_docs]`` dtype int32.
        ``token_ids`` is ``None`` when no shard has a ``.token_ids.npy`` sidecar.
    """
    all_doc_ids = []
    vectors_parts = []
    doclens_parts = []
    token_ids_parts = []
    has_token_ids = False

    shard_paths = list(_shard_paths(Path(cache_dir)))
    for doc_ids, vectors, doclens, token_ids in tqdm(
        iter_cached_flat_shards(cache_dir),
        total=len(shard_paths),
        desc="Loading cached shards (flat)",
    ):
        all_doc_ids.extend(doc_ids)
        vectors_parts.append(np.asarray(vectors))
        doclens_parts.append(np.asarray(doclens, dtype=np.int32))
        if token_ids is not None:
            token_ids_parts.append(np.asarray(token_ids, dtype=np.int64))
            has_token_ids = True

    if not vectors_parts:
        raise ValueError(f"No embedding shards found in cache_dir: {cache_dir}")

    all_vectors = (
        vectors_parts[0]
        if len(vectors_parts) == 1
        else np.concatenate(vectors_parts, axis=0)
    )
    all_doclens = (
        doclens_parts[0]
        if len(doclens_parts) == 1
        else np.concatenate(doclens_parts, axis=0)
    ).astype(np.int32, copy=False)
    all_token_ids = None
    if has_token_ids:
        all_token_ids = (
            token_ids_parts[0]
            if len(token_ids_parts) == 1
            else np.concatenate(token_ids_parts, axis=0)
        ).astype(np.int64, copy=False)

    return all_doc_ids, all_vectors, all_doclens, all_token_ids
