"""Sharded embeddings cache, keyed by directory layout (no hashes).

On-disk layout::

    <root>/<dataset>/<model>/<dtype>/<kind>/
        meta.json
        shard0000.npz
        shard0001.npz
        ...

Where:

- ``<dataset>`` is the BEIR dataset name with ``/`` replaced by ``_`` (e.g. ``trec-covid``,
  ``cqadupstack_android``).
- ``<model>`` is the model name with ``/`` replaced by ``_`` (e.g. ``lightonai_GTE-ModernColBERT-v1``).
- ``<dtype>`` is the numpy dtype string of the on-disk tokens block (e.g. ``float16``).
- ``<kind>`` is ``docs`` or ``queries``.

Each shard ``.npz`` contains:

- ``tokens``  — array of shape ``(sum(doclens), dim)`` in the directory's dtype. Per-token
  vectors concatenated end-to-end across the shard's docs in order.
- ``doclens`` — ``int32`` array of shape ``(n_shard_docs,)``.
- ``ids``     — ``object`` array of shape ``(n_shard_docs,)`` with each doc's id as a string.

``meta.json`` captures the encode config, shard list, and global counts.

Why no hash
-----------
A user-readable directory layout makes the cache greppable / cleanable / sharable; the
config used for encoding is recorded in ``meta.json`` and re-encoding is the right answer
when it changes.
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


DEFAULT_ROOT = Path("embeddings_cache")
DEFAULT_SHARD_SIZE = 100_000
DEFAULT_TOKENS_DTYPE = "float16"

# Single source of truth for the path components we sanitise.
_PATH_SAFE_RE = re.compile(r"[^A-Za-z0-9._-]+")


def _safe_segment(name: str) -> str:
    """Convert an arbitrary identifier to a single safe path segment.

    Slashes (``cqadupstack/android``, ``lightonai/GTE-ModernColBERT-v1``) become
    underscores; other unusual characters are also coalesced to underscores.
    """
    if not name:
        raise ValueError("empty path segment")
    s = _PATH_SAFE_RE.sub("_", name).strip("_")
    if not s:
        raise ValueError(f"path segment {name!r} reduced to empty after sanitisation")
    return s


def kind_dir(
    *,
    root: Path,
    dataset: str,
    model: str,
    dtype: str,
    kind: str,
) -> Path:
    """The directory holding shards + meta.json for one (dataset, model, dtype, kind)."""
    if kind not in ("docs", "queries"):
        raise ValueError(f"kind must be 'docs' or 'queries', got {kind!r}")
    return (
        Path(root)
        / _safe_segment(dataset)
        / _safe_segment(model)
        / _safe_segment(dtype)
        / kind
    )


@dataclass
class CacheManifest:
    dataset: str
    model: str
    dtype: str
    kind: str  # "docs" or "queries"
    num_items: int
    dim: int
    shard_size: int
    shards: list[str]  # filenames relative to the kind_dir
    config: dict        # encode-time config (document_length, normalize_embeddings, etc.)

    def to_dict(self) -> dict:
        return {
            "schema_version": 3,
            "dataset": self.dataset,
            "model": self.model,
            "dtype": self.dtype,
            "kind": self.kind,
            "num_items": self.num_items,
            "dim": self.dim,
            "shard_size": self.shard_size,
            "shards": list(self.shards),
            "config": self.config,
        }

    @classmethod
    def from_dict(cls, m: dict) -> "CacheManifest":
        if m.get("schema_version") != 3:
            raise ValueError(
                f"unsupported manifest schema_version={m.get('schema_version')!r} (expected 3)"
            )
        return cls(
            dataset=m["dataset"],
            model=m["model"],
            dtype=m["dtype"],
            kind=m["kind"],
            num_items=int(m["num_items"]),
            dim=int(m["dim"]),
            shard_size=int(m["shard_size"]),
            shards=list(m.get("shards", [])),
            config=dict(m.get("config", {})),
        )


def _meta_path(d: Path) -> Path:
    return d / "meta.json"


def _shard_path(d: Path, shard_idx: int) -> Path:
    return d / f"shard{shard_idx:04d}.npz"


def manifest_exists(
    *,
    root: Path,
    dataset: str,
    model: str,
    dtype: str,
    kind: str,
) -> bool:
    return _meta_path(kind_dir(
        root=root, dataset=dataset, model=model, dtype=dtype, kind=kind
    )).is_file()


def load_manifest(
    *,
    root: Path,
    dataset: str,
    model: str,
    dtype: str,
    kind: str,
) -> CacheManifest | None:
    p = _meta_path(kind_dir(
        root=root, dataset=dataset, model=model, dtype=dtype, kind=kind
    ))
    if not p.is_file():
        return None
    try:
        return CacheManifest.from_dict(json.loads(p.read_text(encoding="utf-8")))
    except Exception:
        return None


def write_manifest(
    *,
    root: Path,
    dataset: str,
    model: str,
    dtype: str,
    kind: str,
    config: dict,
    shard_filenames: list[str],
    num_items: int,
    dim: int,
    shard_size: int,
) -> Path:
    d = kind_dir(root=root, dataset=dataset, model=model, dtype=dtype, kind=kind)
    d.mkdir(parents=True, exist_ok=True)
    manifest = CacheManifest(
        dataset=dataset,
        model=model,
        dtype=dtype,
        kind=kind,
        num_items=num_items,
        dim=dim,
        shard_size=shard_size,
        shards=list(shard_filenames),
        config=dict(config),
    )
    p = _meta_path(d)
    p.write_text(
        json.dumps(manifest.to_dict(), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return p


def save_shard(
    *,
    root: Path,
    dataset: str,
    model: str,
    dtype: str,
    kind: str,
    shard_idx: int,
    embeddings: list,
    doc_ids: list[str],
) -> tuple[Path, int, int]:
    """Concatenate per-doc embeddings into a single buffer in the directory's dtype and
    write to ``shardNNNN.npz``. Returns ``(shard_path, num_tokens, dim)``.
    """
    if not embeddings:
        raise ValueError(f"shard {shard_idx}: empty embeddings list")
    if len(embeddings) != len(doc_ids):
        raise ValueError(
            f"shard {shard_idx}: embeddings/ids length mismatch ({len(embeddings)} vs {len(doc_ids)})"
        )
    d = kind_dir(root=root, dataset=dataset, model=model, dtype=dtype, kind=kind)
    d.mkdir(parents=True, exist_ok=True)

    np_dtype = np.dtype(dtype)
    first = np.asarray(embeddings[0])
    if first.ndim != 2:
        raise ValueError(
            f"shard {shard_idx}: expected 2D per-doc tensor, got shape {first.shape}"
        )
    dim = int(first.shape[1])
    doclens = np.empty(len(embeddings), dtype=np.int32)
    total = 0
    for i, emb in enumerate(embeddings):
        a = np.asarray(emb)
        if a.ndim != 2 or a.shape[1] != dim:
            raise ValueError(
                f"shard {shard_idx} doc {i}: expected shape (*, {dim}), got {a.shape}"
            )
        doclens[i] = a.shape[0]
        total += int(a.shape[0])
    tokens = np.empty((total, dim), dtype=np_dtype)
    cursor = 0
    for emb, n in zip(embeddings, doclens):
        a = np.asarray(emb, dtype=np_dtype)
        tokens[cursor : cursor + n] = a
        cursor += int(n)
    ids_arr = np.array([str(x) for x in doc_ids], dtype=object)
    out = _shard_path(d, shard_idx)
    np.savez(out, tokens=tokens, doclens=doclens, ids=ids_arr)
    return out, total, dim


def encode_and_save_sharded(
    *,
    root: Path,
    dataset: str,
    model: str,
    kind: str,
    config: dict,
    doc_ids: list[str],
    sentences: list[str],
    encode_fn,                            # callable(list[str]) -> list[np.ndarray]
    shard_size: int = DEFAULT_SHARD_SIZE,
    dtype: str = DEFAULT_TOKENS_DTYPE,
    log_prefix: str = "[shard]",
) -> CacheManifest:
    """Encode in shards of ``shard_size`` and write each shard immediately. The manifest
    is written last, so a mid-encoding crash leaves the cache marked incomplete (no
    ``meta.json``) and re-runs will re-encode cleanly.

    ``encode_fn(batch_sentences)`` must return a list of per-doc 2D arrays.
    """
    if len(doc_ids) != len(sentences):
        raise ValueError(
            f"doc_ids ({len(doc_ids)}) vs sentences ({len(sentences)}) length mismatch"
        )
    n = len(sentences)
    if n == 0:
        raise ValueError("nothing to encode")

    n_shards = math.ceil(n / shard_size)
    print(
        f"{log_prefix} encoding dataset={dataset!r} kind={kind} dtype={dtype} "
        f"n={n} shards={n_shards} shard_size={shard_size}",
        flush=True,
    )
    shard_paths: list[Path] = []
    dim = -1
    for shard_idx in range(n_shards):
        lo = shard_idx * shard_size
        hi = min(n, lo + shard_size)
        slice_sentences = sentences[lo:hi]
        slice_ids = doc_ids[lo:hi]
        print(
            f"{log_prefix} shard {shard_idx + 1}/{n_shards}: encoding docs [{lo}, {hi}) ({hi - lo} docs)",
            flush=True,
        )
        shard_emb = encode_fn(slice_sentences)
        out, tokens, sd = save_shard(
            root=root,
            dataset=dataset,
            model=model,
            dtype=dtype,
            kind=kind,
            shard_idx=shard_idx,
            embeddings=shard_emb,
            doc_ids=slice_ids,
        )
        if dim == -1:
            dim = sd
        elif dim != sd:
            raise RuntimeError(f"shard {shard_idx}: dim {sd} != prior dim {dim}")
        shard_paths.append(out)
        print(
            f"{log_prefix} shard {shard_idx + 1}/{n_shards}: wrote {tokens} tokens to "
            f"{out.parent.name}/{out.name}",
            flush=True,
        )
        del shard_emb
    write_manifest(
        root=root,
        dataset=dataset,
        model=model,
        dtype=dtype,
        kind=kind,
        config=config,
        shard_filenames=[p.name for p in shard_paths],
        num_items=n,
        dim=dim,
        shard_size=shard_size,
    )
    print(f"{log_prefix} done: manifest written to {kind_dir(root=root, dataset=dataset, model=model, dtype=dtype, kind=kind)}/meta.json", flush=True)
    out_manifest = load_manifest(
        root=root, dataset=dataset, model=model, dtype=dtype, kind=kind
    )
    assert out_manifest is not None, "manifest read-back failed"
    return out_manifest


def iter_shards(
    *,
    root: Path,
    manifest: CacheManifest,
    as_float32: bool = False,
) -> Iterator[tuple[list[str], np.ndarray, np.ndarray]]:
    """Yield ``(ids, tokens_block, doclens)`` per shard. ``tokens_block`` is contiguous
    of shape ``(sum(doclens), dim)``. Set ``as_float32=True`` to upcast on read.
    """
    d = kind_dir(
        root=root,
        dataset=manifest.dataset,
        model=manifest.model,
        dtype=manifest.dtype,
        kind=manifest.kind,
    )
    for shard_name in manifest.shards:
        with np.load(d / shard_name, allow_pickle=True) as z:
            tokens = z["tokens"]
            doclens = z["doclens"]
            ids = z["ids"].tolist()
        if as_float32 and tokens.dtype != np.float32:
            tokens = tokens.astype(np.float32, copy=False)
        yield ids, tokens, doclens


def load_per_doc_embeddings(
    *,
    root: Path,
    manifest: CacheManifest,
    as_float32: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """Materialise ``(ids, list_of_per_doc_tensors)``. Per-doc arrays are zero-copy
    views into each shard's ``tokens`` block (one buffer per shard, kept alive via
    the slice's base attribute).
    """
    all_ids: list[str] = []
    all_arrs: list[np.ndarray] = []
    for ids, tokens, doclens in iter_shards(root=root, manifest=manifest, as_float32=as_float32):
        offsets = np.cumsum(doclens)[:-1]
        per_doc = np.split(tokens, offsets, axis=0)
        all_ids.extend(ids)
        all_arrs.extend(per_doc)
    return all_ids, all_arrs


def load_cache(
    *,
    root: Path,
    dataset: str,
    model: str,
    kind: str,
    dtype: str = DEFAULT_TOKENS_DTYPE,
    as_float32: bool = True,
) -> tuple[list[str], list[np.ndarray]]:
    """Load (ids, per-doc tensors) for a (dataset, model, dtype, kind) cell.

    Raises ``FileNotFoundError`` if no manifest is found at that path. Use
    :func:`manifest_exists` first if you want a soft check.
    """
    manifest = load_manifest(
        root=root, dataset=dataset, model=model, dtype=dtype, kind=kind
    )
    if manifest is None:
        d = kind_dir(root=root, dataset=dataset, model=model, dtype=dtype, kind=kind)
        raise FileNotFoundError(
            f"no manifest at {d}/meta.json — encode first via examples/evaluation/_encode_beir.py"
        )
    return load_per_doc_embeddings(root=root, manifest=manifest, as_float32=as_float32)
