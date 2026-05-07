#!/usr/bin/env python3
"""Sweeps of ``maxIVF_cagra`` anchor training on cached BEIR document embeddings.

Default run (single config for inspection):

- ``n_anchors = floor(n_tokens * 0.025)``
- ``n_iter = 5``
- ``normalize_anchors = True`` only (pass ``--sweep-normalization`` for True/False grid)
- Fixed ``token_batch_size`` (default 131072; not swept)

Override anchor fractions with ``--anchor-fractions``; combine with ``--sweep-normalization``
for a full factorial on normalization.

Reads ``*.meta.json`` under ``evaluation_cache/beir_embeddings`` to resolve dataset names.
Requires ``fast-plaid`` with CAGRA/cuVS, CUDA, and ``LD_LIBRARY_PATH`` for libcuvs if needed.

Example::

  uv run python examples/evaluation/maxivf_cagra_anchor_sweep.py \\
    --cache-dir evaluation_cache/beir_embeddings --list-caches-only

  LD_LIBRARY_PATH=... uv run python examples/evaluation/maxivf_cagra_anchor_sweep.py \\
    --dataset fiqa --device cuda:0

  Or opt into random re-seeds for empty clusters::

    FASTPLAID_CAGRA_MAXIVF_RESEED_EMPTY=1 uv run python ... --dataset fiqa

  Or use ``--reseed-empty`` (sets that env var before loading ``fast_plaid_rust``).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np
import torch

DEFAULT_ANCHOR_FRACTIONS = (0.025,)
DEFAULT_N_ITER = 5
DEFAULT_TOKEN_BATCH_SIZE = 131072


def _repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


def meta_to_npz_path(meta_path: Path) -> Path:
    if not meta_path.name.endswith(".meta.json"):
        raise ValueError(f"expected *.meta.json, got {meta_path}")
    key = meta_path.name[: -len(".meta.json")]
    return meta_path.with_name(f"{key}.npz")


def list_document_caches(cache_dir: Path) -> list[tuple[Path, dict[str, Any]]]:
    rows: list[tuple[Path, dict[str, Any]]] = []
    for meta_path in sorted(cache_dir.glob("documents-*.meta.json")):
        try:
            meta: dict[str, Any] = json.loads(meta_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        if meta.get("kind") != "documents":
            continue
        npz = meta_to_npz_path(meta_path)
        cfg = meta.get("config") or {}
        rows.append(
            (
                npz,
                {
                    "meta_path": str(meta_path),
                    "npz_path": str(npz),
                    "npz_exists": npz.is_file(),
                    "dataset_name": cfg.get("dataset_name"),
                    "num_items": meta.get("num_items"),
                    "model_name": cfg.get("model_name"),
                },
            )
        )
    return rows


def pick_documents_npz(cache_dir: Path, dataset: str) -> Path:
    matches: list[tuple[Path, dict[str, Any]]] = []
    for npz, meta in list_document_caches(cache_dir):
        if meta.get("dataset_name") == dataset:
            matches.append((npz, meta))
    if not matches:
        print(
            f"No documents cache for dataset_name={dataset!r} under {cache_dir}",
            file=sys.stderr,
        )
        print("Known document caches:", file=sys.stderr)
        for _, m in list_document_caches(cache_dir):
            print(
                f"  - {m.get('dataset_name')!r}  "
                f"num_items={m.get('num_items')}  "
                f"npz={Path(m['npz_path']).name}",
                file=sys.stderr,
            )
        sys.exit(1)
    if len(matches) > 1:
        print(f"Warning: multiple caches for {dataset!r}; picking the largest by num_items.")
        matches.sort(key=lambda t: int(t[1].get("num_items") or 0), reverse=True)
    npz_path, meta = matches[0]
    if not npz_path.is_file():
        print(f"Missing npz: {npz_path}", file=sys.stderr)
        sys.exit(1)
    return npz_path


def load_document_embeddings(npz_path: Path, max_docs: int | None) -> list[torch.Tensor]:
    with np.load(npz_path, allow_pickle=True) as z:
        raw: list = z["embeddings"].tolist()
    if not isinstance(raw, list) or not raw:
        raise ValueError(f"bad embeddings list in {npz_path}")
    if max_docs is not None:
        raw = raw[: max(0, max_docs)]
    out: list[torch.Tensor] = []
    for i, x in enumerate(raw):
        t = torch.as_tensor(x, dtype=torch.float32)
        if t.dim() != 2:
            raise ValueError(f"doc {i}: expected 2D token tensor, got shape {tuple(t.shape)}")
        out.append(t)
    return out


def centroid_stats(centroids: torch.Tensor) -> dict[str, Any]:
    x = centroids.detach().float().cpu().numpy()
    norms = np.linalg.norm(x, axis=1)
    zero = int((norms == 0.0).sum())
    pos = norms[norms > 0]
    return {
        "shape": tuple(centroids.shape),
        "dtype": str(centroids.dtype),
        "n_rows": int(x.shape[0]),
        "zero_norm_rows": zero,
        "zero_norm_frac": float(zero / max(1, x.shape[0])),
        "norm_min": float(norms.min()),
        "norm_median": float(np.median(norms)),
        "norm_max": float(norms.max()),
        "norm_p01_pos": float(np.quantile(pos, 0.01)) if pos.size else None,
        "norm_p99_pos": float(np.quantile(pos, 0.99)) if pos.size else None,
        "finite": bool(np.isfinite(x).all()),
    }


@dataclass(frozen=True)
class RunCfg:
    anchor_fraction: float
    n_anchors: int
    n_iter: int
    graph_degree: int
    intermediate_graph_degree: int
    itopk_size: int
    token_batch_size: int
    normalize_anchors: bool


def n_anchors_from_fraction(n_tokens: int, fraction: float) -> int:
    """Match fast-plaid Rust ``default_n_anchors`` style: floor and at least 1."""
    n = int(math.floor(float(n_tokens) * float(fraction)))
    return max(1, n)


def build_run_grid(
    *,
    n_tokens: int,
    anchor_fractions: tuple[float, ...],
    n_iter: int,
    token_batch_size: int,
    graph_degree: int,
    intermediate_graph_degree: int,
    itopk_size: int,
    normalize_values: tuple[bool, ...],
) -> list[RunCfg]:
    rows: list[RunCfg] = []
    for frac, norm in product(anchor_fractions, normalize_values):
        na = n_anchors_from_fraction(n_tokens, frac)
        rows.append(
            RunCfg(
                anchor_fraction=frac,
                n_anchors=na,
                n_iter=n_iter,
                graph_degree=graph_degree,
                intermediate_graph_degree=intermediate_graph_degree,
                itopk_size=itopk_size,
                token_batch_size=token_batch_size,
                normalize_anchors=norm,
            )
        )
    return rows


def parse_fractions(s: str) -> tuple[float, ...]:
    out: list[float] = []
    for part in s.replace(",", " ").split():
        if not part:
            continue
        out.append(float(part))
    if not out:
        raise argparse.ArgumentTypeError("need at least one anchor fraction")
    return tuple(out)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cache-dir",
        type=Path,
        default=_repo_root() / "evaluation_cache" / "beir_embeddings",
        help="Directory with documents-*.npz and *.meta.json",
    )
    parser.add_argument(
        "--dataset",
        default="fiqa",
        help="BEIR dataset_name field inside cache meta (e.g. fiqa, nfcorpus)",
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Use only the first N documents after load (for faster sweeps)",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Unless --max-docs is set, use first 500 documents only",
    )
    parser.add_argument(
        "--anchor-fractions",
        type=parse_fractions,
        default=DEFAULT_ANCHOR_FRACTIONS,
        help=(
            "Space- or comma-separated fractions of n_tokens for n_anchors "
            f"(default: {' '.join(str(x) for x in DEFAULT_ANCHOR_FRACTIONS)})"
        ),
    )
    parser.add_argument(
        "--sweep-normalization",
        action="store_true",
        help="Run each fraction with normalize_anchors True and False (default: True only)",
    )
    parser.add_argument(
        "--n-iter",
        type=int,
        default=DEFAULT_N_ITER,
        help=f"maxIVF_cagra outer iterations (default: {DEFAULT_N_ITER})",
    )
    parser.add_argument(
        "--token-batch-size",
        type=int,
        default=DEFAULT_TOKEN_BATCH_SIZE,
        help="Token batch size for streaming assignment (default: fixed; not swept)",
    )
    parser.add_argument("--graph-degree", type=int, default=32)
    parser.add_argument("--intermediate-graph-degree", type=int, default=64)
    parser.add_argument("--itopk-size", type=int, default=128)
    parser.add_argument(
        "--list-caches-only",
        action="store_true",
        help="Print discovered document caches and exit",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--reseed-empty",
        action="store_true",
        help=(
            "Set FASTPLAID_CAGRA_MAXIVF_RESEED_EMPTY=1 before loading Rust (random token "
            "for empty clusters; default is sticky: keep previous anchor)"
        ),
    )
    args = parser.parse_args()

    if args.reseed_empty:
        os.environ["FASTPLAID_CAGRA_MAXIVF_RESEED_EMPTY"] = "1"

    cache_dir = args.cache_dir.resolve()
    if not cache_dir.is_dir():
        print(f"cache dir not found: {cache_dir}", file=sys.stderr)
        sys.exit(1)

    if args.list_caches_only:
        print(f"Document caches under {cache_dir}:\n")
        for _, meta in list_document_caches(cache_dir):
            print(f"  dataset={meta.get('dataset_name')!r}")
            print(f"    items={meta.get('num_items')}  model={meta.get('model_name')!r}")
            print(f"    npz_exists={meta['npz_exists']}  file={Path(meta['npz_path']).name}\n")
        return

    max_docs = args.max_docs
    if args.quick and max_docs is None:
        max_docs = 500

    npz_path = pick_documents_npz(cache_dir, args.dataset)
    print(f"Using embeddings: {npz_path.name} (dataset={args.dataset!r}, max_docs={max_docs})")

    docs = load_document_embeddings(npz_path, max_docs)
    n_tokens = int(sum(t.shape[0] for t in docs))
    dim = int(docs[0].shape[1])
    print(
        f"Loaded {len(docs)} docs, {n_tokens} total tokens, dim={dim} "
        f"(float32 CPU tensors; Rust moves to {args.device})"
    )
    norm_vals: tuple[bool, ...] = (True, False) if args.sweep_normalization else (True,)
    print(
        f"Sweep: n_iter={args.n_iter} anchor_fractions={args.anchor_fractions} "
        f"normalize_anchors={list(norm_vals)} token_batch_size={args.token_batch_size}\n"
    )

    try:
        from fast_plaid import fast_plaid_rust
        from fast_plaid.search.fast_plaid import _load_torch_path
    except ImportError as e:
        print(f"fast_plaid import failed: {e}", file=sys.stderr)
        sys.exit(1)

    fast_plaid_rust.initialize_torch(torch_path=_load_torch_path(args.device))

    runs = build_run_grid(
        n_tokens=n_tokens,
        anchor_fractions=args.anchor_fractions,
        n_iter=max(1, args.n_iter),
        token_batch_size=max(1, args.token_batch_size),
        graph_degree=max(2, args.graph_degree),
        intermediate_graph_degree=max(1, args.intermediate_graph_degree),
        itopk_size=max(1, args.itopk_size),
        normalize_values=norm_vals,
    )
    print(f"Running {len(runs)} maxIVF_cagra configurations...\n")

    for i, rc in enumerate(runs, 1):
        kwargs = dict(
            embeddings=docs,
            device=args.device,
            n_anchors=rc.n_anchors,
            n_iter=rc.n_iter,
            graph_degree=rc.graph_degree,
            intermediate_graph_degree=rc.intermediate_graph_degree,
            itopk_size=rc.itopk_size,
            token_batch_size=rc.token_batch_size,
            seed=args.seed,
            normalize_anchors=rc.normalize_anchors,
        )
        label = (
            f"[{i}/{len(runs)}] anchor_frac={rc.anchor_fraction} n_anchors={rc.n_anchors} "
            f"n_iter={rc.n_iter} normalize_anchors={rc.normalize_anchors} "
            f"token_batch_size={rc.token_batch_size}"
        )
        print(label)
        t0 = time.perf_counter()
        try:
            centroids = fast_plaid_rust.maxivf_cagra_anchors(**kwargs)
        except Exception as e:
            print(f"  FAILED: {e}\n")
            continue
        elapsed = time.perf_counter() - t0
        st = centroid_stats(centroids)
        print(f"  wall_s={elapsed:.2f}")
        for k, v in st.items():
            print(f"  {k}: {v}")
        print()


if __name__ == "__main__":
    main()
