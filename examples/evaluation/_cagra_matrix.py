"""Experiment matrix: clustering x centroid-backend across BEIR datasets.

For each (dataset, anchor_method, centroid_index) cell:
- builds a fresh PLAID index using cached embeddings (no encoder calls)
- times add_documents (anchor training + IVF compression + centroid index build)
- times retrieval over all queries
- evaluates BEIR metrics
- appends a row to a JSONL log and prints a summary table at the end

Run::

  LD_LIBRARY_PATH=... uv run --no-sync python examples/evaluation/_cagra_matrix.py \
    --datasets nfcorpus scifact fiqa scidocs \
    --configs kmeans_dense kmeans_cagra maxivf_dense maxivf_cagra \
    --out /tmp/cagra_matrix.jsonl
"""

from __future__ import annotations

import argparse
import json
import math
import time
import traceback
from pathlib import Path

import numpy as np
import psutil

from pylate import evaluation, indexes, retrieve


def _maxivf_anchor_params(
    *,
    n_iter: int,
    n_anchors: int | None,
    empty_handling: str = "sticky",
) -> dict:
    p = {
        "n_iter": n_iter,
        "graph_degree": 32,
        "intermediate_graph_degree": 64,
        "itopk_size": 128,
        "token_batch_size": 2 ** 17,
        "normalize_anchors": True,
        "empty_handling": empty_handling,  # "sticky" | "reseed" | "prune"
    }
    if n_anchors is not None:
        p["n_anchors"] = n_anchors
    return p


CAGRA_CENTROID_PARAMS = {
    "graph_degree": 32,
    "intermediate_graph_degree": 64,
    "itopk_size": 128,
}


def _build_preset_table() -> dict[str, dict]:
    """Build the (config_name -> kwargs) table.

    The maxIVF cells are parametrised by `(n_iter, anchor_frac)`, where the actual `n_anchors`
    is resolved at run time from the dataset's token count (so cells are comparable across
    datasets). Fraction `None` means "let fast-plaid use its default 0.025".
    """
    presets: dict[str, dict] = {
        # Baselines: classic kmeans anchors.
        "kmeans_dense": {
            "anchor_method": "kmeans",
            "centroid_index": "dense",
        },
        "kmeans_cagra": {
            "anchor_method": "kmeans",
            "centroid_index": "cagra",
            "centroid_index_params": CAGRA_CENTROID_PARAMS,
        },
    }
    # maxIVF sweep: anchor-fraction × {dense, cagra} × empty_handling. `n_iter=3` is
    # locked — earlier ablation showed iter1 hurts and iter5 doesn't pay off.
    # `frac=None` = fast-plaid default 0.025. `empty_handling` is the new axis introduced
    # to test whether dropping degenerate clusters mid-training closes the quality gap.
    frac_sweep = [
        ("default", 3, None),
        ("frac01",  3, 0.01),
        ("frac05",  3, 0.05),
    ]
    empty_sweep = [
        ("",        "sticky"),  # historic default → no suffix on config name
        ("_prune",  "prune"),
        ("_reseed", "reseed"),
    ]
    for tag, n_iter, frac in frac_sweep:
        for cb in ("dense", "cagra"):
            for empty_suffix, empty_mode in empty_sweep:
                name = f"maxivf_{cb}_{tag}{empty_suffix}"
                cfg: dict = {
                    "anchor_method": "maxIVF_cagra",
                    "_n_iter": n_iter,
                    "_anchor_frac": frac,
                    "_empty_handling": empty_mode,
                    "centroid_index": cb,
                }
                if cb == "cagra":
                    cfg["centroid_index_params"] = CAGRA_CENTROID_PARAMS
                presets[name] = cfg

    # n_ivf_probe sweep (roadmap C6). Vary the number of IVF cells probed at retrieval
    # time. Previous sweeps hardcoded probe=1; PLAID default is 8. At ~10x more anchors
    # than kmeans, maxIVF may close the quality gap just by probing more cells.
    # Only add _cagra variants with the best empty-handling modes (prune + sticky).
    for probe in (4, 8, 16):
        for empty_suffix, empty_mode in [("", "sticky"), ("_prune", "prune")]:
            for tag, n_iter, frac in [("default", 3, None), ("frac01", 3, 0.01)]:
                name = f"maxivf_cagra_{tag}{empty_suffix}_probe{probe}"
                presets[name] = {
                    "anchor_method": "maxIVF_cagra",
                    "_n_iter": n_iter,
                    "_anchor_frac": frac,
                    "_empty_handling": empty_mode,
                    "centroid_index": "cagra",
                    "centroid_index_params": CAGRA_CENTROID_PARAMS,
                    "n_ivf_probe": probe,
                }

    return presets


CONFIG_PRESETS: dict[str, dict] = _build_preset_table()


import sys as _sys  # noqa: E402

_sys.path.insert(0, str(Path(__file__).resolve().parent))

from _embeddings_cache import (  # noqa: E402  (local module)
    DEFAULT_ROOT,
    DEFAULT_TOKENS_DTYPE,
    load_cache,
)


def _load_dataset_embeddings(
    *, dataset: str, model: str, root: Path, kind: str, dtype: str
) -> list[np.ndarray]:
    """Load per-doc tensors for one cache cell. Raises if not encoded yet.

    Embeddings are kept at stored dtype (float16 by default). The Rust layer handles
    float16 inputs natively: create.rs avoids the half→half copy, and maxivf_cagra.rs
    converts to float32 per 131K-token batch rather than up-casting the whole corpus.
    This halves peak host RAM vs. as_float32=True, which matters for NQ-scale corpora.
    """
    _ids, arrs = load_cache(
        root=root, dataset=dataset, model=model, dtype=dtype, kind=kind, as_float32=False
    )
    return arrs


def _run_one_with_optional_timeout(*, timeout_s: float | None, **kwargs) -> dict:
    """Run `run_one`, optionally bounded by a wall-clock timeout (POSIX SIGALRM).

    Used so a single bad cell can't stall the whole matrix on multi-hour datasets.
    """
    if not timeout_s:
        return run_one(**kwargs)
    import signal

    class _CellTimeout(Exception):
        pass

    def _handler(signum, frame):
        raise _CellTimeout(f"cell exceeded {timeout_s:.0f}s")

    prev = signal.signal(signal.SIGALRM, _handler)
    signal.setitimer(signal.ITIMER_REAL, float(timeout_s))
    try:
        return run_one(**kwargs)
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, prev)


def _free_gpu_memory() -> None:
    """Release cached GPU allocations between cells so a heavy cell doesn't poison the next."""
    import gc

    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
    except Exception:
        pass


def run_one(
    *,
    dataset: str,
    config_name: str,
    config: dict,
    documents: list,
    queries: dict,
    qrels: dict,
    documents_embeddings: list,
    queries_embeddings: list,
) -> dict:
    document_ids = [d["id"] for d in documents]
    index_kwargs = dict(config)
    # n_tokens_total is the sample-space size that drives every centroid-count formula —
    # always compute it so we can record it for every cell (kmeans and maxIVF alike).
    n_tokens_total = int(sum(int(np.asarray(t).shape[0]) for t in documents_embeddings))
    # Resolve maxIVF placeholders into real anchor_params using this dataset's token count.
    n_iter = index_kwargs.pop("_n_iter", None)
    anchor_frac = index_kwargs.pop("_anchor_frac", None)
    empty_handling = index_kwargs.pop("_empty_handling", "sticky")
    n_anchors = None
    if index_kwargs.get("anchor_method") == "maxIVF_cagra":
        if anchor_frac is not None:
            n_anchors = max(1, int(n_tokens_total * anchor_frac))
        index_kwargs["anchor_params"] = _maxivf_anchor_params(
            n_iter=int(n_iter) if n_iter is not None else 3,
            n_anchors=n_anchors,
            empty_handling=empty_handling,
        )

    # Predict how many centroids each backend will use — useful for the crossover analysis.
    if index_kwargs.get("anchor_method") == "maxIVF_cagra":
        # If no explicit n_anchors was set, fast-plaid's Rust side falls back to floor(0.025 * n_tokens).
        if n_anchors is None:
            n_centroids_planned = max(1, int(0.025 * n_tokens_total))
            anchor_frac_effective = 0.025
        else:
            n_centroids_planned = n_anchors
            anchor_frac_effective = anchor_frac
    else:
        # Classic PLAID kmeans: 2 ** floor(log2(16 * sqrt(n_tokens))).
        # See fast_plaid/search/fast_plaid.py:153 and rust/index/create.rs:294.
        if n_tokens_total > 0:
            sqrt_term = 16.0 * math.sqrt(n_tokens_total)
            n_centroids_planned = 2 ** math.floor(math.log2(max(2.0, sqrt_term)))
        else:
            n_centroids_planned = 0
        anchor_frac_effective = None
    # Distinct on-disk index per cell so override=True doesn't mix state.
    safe_dataset = dataset.replace("/", "_")
    index_kwargs.setdefault("index_name", f"matrix_{safe_dataset}_{config_name}")
    index_kwargs.setdefault("override", True)
    # n_ivf_probe is left to PLAID's default (8) unless explicitly overridden in a
    # config preset. Earlier sweeps forced it to 1 — that suppressed maxIVF's recall
    # because more anchors only help if you probe more cells. Treat n_ivf_probe as a
    # tunable axis going forward, not a fixed constant.
    # Lower nbits for residual quantization (PLAID default is 4). 2-bit cuts on-disk
    # residual size in half and is the published PLAID setting; observed quality drop
    # at ColBERT scale is small.
    index_kwargs.setdefault("nbits", 2)

    print(
        f"\n[matrix] dataset={dataset} config={config_name} "
        f"docs={len(documents)} queries={len(queries)}"
    )

    t_init = time.perf_counter()
    index = indexes.PLAID(**index_kwargs)
    retriever = retrieve.ColBERT(index=index)
    init_s = time.perf_counter() - t_init

    proc = psutil.Process()
    try:
        import torch
        _torch_available = torch.cuda.is_available()
    except Exception:
        _torch_available = False

    rss_before = proc.memory_info().rss
    if _torch_available:
        torch.cuda.reset_peak_memory_stats()

    t_add = time.perf_counter()
    index.add_documents(
        documents_ids=document_ids,
        documents_embeddings=documents_embeddings,
    )
    add_s = time.perf_counter() - t_add
    # Phase decomposition: read from the instrumentation attribute set in fast_plaid.py
    # after add_documents. Path: index._index (pylate FastPlaid) → .fast_plaid (fp FastPlaid).
    try:
        _phase_s = index._index.fast_plaid._last_add_phase_s
    except AttributeError:
        _phase_s = {}

    rss_after = proc.memory_info().rss
    peak_gpu_bytes = torch.cuda.max_memory_allocated() if _torch_available else None

    t_ret = time.perf_counter()
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=1000)
    ret_s = time.perf_counter() - t_ret

    # Drop self-matches as in the reference script.
    for (qid, _q), q_scores in zip(queries.items(), scores):
        for s in list(q_scores):
            if s["id"] == qid:
                q_scores.remove(s)

    eval_metrics = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=[
            "map",
            "ndcg@10", "ndcg@100", "ndcg@1000",
            "recall@10", "recall@50", "recall@100", "recall@500", "recall@1000",
        ],
    )
    eval_metrics = {k: float(v) for k, v in eval_metrics.items()}

    n_q = len(queries)
    # Resolved kwargs minus the on-disk-name fields, so the row is replayable.
    resolved = {
        k: v for k, v in index_kwargs.items()
        if k not in ("index_name", "override")
    }
    return {
        "dataset": dataset,
        "config": config_name,
        "n_docs": len(documents),
        "n_queries": n_q,
        # Corpus shape — needed to compute crossover thresholds post-hoc.
        "n_tokens_total": n_tokens_total,
        # Sweep coordinates (axes of the experiment matrix).
        "n_iter": n_iter,
        "anchor_frac": anchor_frac,
        "anchor_frac_effective": anchor_frac_effective,
        "n_anchors_requested": n_anchors,
        "empty_handling": empty_handling,
        # Predicted centroid count for this cell. For maxIVF this equals n_anchors_requested
        # (or floor(0.025 * n_tokens) when None). For kmeans this is fast-plaid's
        # 2 ** floor(log2(16 * sqrt(n_tokens))) heuristic.
        "n_centroids_planned": n_centroids_planned,
        # Full resolved index kwargs — every parameter that defines this cell.
        "resolved_kwargs": resolved,
        # Phase timings.
        "init_s": init_s,
        "add_documents_s": add_s,
        # Phase decomposition of add_documents: anchor training, IVF compression, centroid build.
        "phase_anchor_train_s": _phase_s.get("anchor_train_s"),
        "phase_ivf_compress_s": _phase_s.get("ivf_compress_s"),
        "phase_centroid_build_s": _phase_s.get("centroid_build_s"),
        "retrieve_s": ret_s,
        "retrieve_ms_per_query": (ret_s / max(1, n_q)) * 1000.0,
        # Peak memory during add_documents (host RSS delta + GPU peak).
        "add_host_rss_delta_mb": (rss_after - rss_before) / 1024 ** 2,
        "add_peak_gpu_mb": (peak_gpu_bytes / 1024 ** 2) if peak_gpu_bytes is not None else None,
        # Metrics.
        "metrics": eval_metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", default=["nfcorpus", "scifact", "scidocs", "fiqa"])
    # The default config set is the "core" 4-cell backend matrix (no n_iter / fraction sweep).
    # Use --suite full to add the CAGRA hyperparameter ablations.
    parser.add_argument(
        "--suite",
        choices=["core", "full", "custom"],
        default="core",
        help="core = 4 backend cells (default); full = adds n_iter / anchor-fraction ablations; "
             "custom = use --configs verbatim",
    )
    parser.add_argument(
        "--configs",
        nargs="+",
        default=None,
        choices=sorted(CONFIG_PRESETS.keys()),
        help="Required when --suite=custom; otherwise overrides the suite's default list.",
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Embeddings cache root (default: {DEFAULT_ROOT}/)",
    )
    parser.add_argument(
        "--model",
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model id used at encode time; selects the cache subdirectory.",
    )
    parser.add_argument(
        "--dtype",
        default=DEFAULT_TOKENS_DTYPE,
        help="On-disk tokens dtype used at encode time (default: float16).",
    )
    parser.add_argument("--out", type=Path, default=Path("/tmp/cagra_matrix.jsonl"))
    parser.add_argument("--split", default="test")
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip (dataset, config) pairs that already have a successful row in --out.",
    )
    parser.add_argument(
        "--fresh",
        action="store_true",
        help="Truncate --out before starting (mutually exclusive with --resume).",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="If set, sample only the first N docs from each dataset (smoke testing).",
    )
    parser.add_argument(
        "--max-queries",
        type=int,
        default=None,
        help="If set, sample only the first N queries from each dataset (smoke testing).",
    )
    parser.add_argument(
        "--per-cell-timeout-s",
        type=float,
        default=None,
        help="Optional wall clock seconds; cells exceeding this raise and continue. "
             "Implemented via SIGALRM on POSIX so subprocess work is also interrupted.",
    )
    args = parser.parse_args()
    if args.resume and args.fresh:
        parser.error("--resume and --fresh are mutually exclusive")

    # Resolve config list from suite + overrides.
    if args.suite == "custom":
        if not args.configs:
            parser.error("--suite=custom requires --configs")
        configs = list(args.configs)
    else:
        core = ["kmeans_dense", "kmeans_cagra", "maxivf_dense_default", "maxivf_cagra_default"]
        if args.suite == "core":
            configs = list(args.configs) if args.configs else core
        else:  # full — adds anchor-fraction + empty_handling sweep
            full = core + [
                "maxivf_cagra_frac01",
                "maxivf_cagra_frac05",
                # empty_handling axis: prune & reseed at default fraction
                "maxivf_cagra_default_prune",
                "maxivf_cagra_default_reseed",
                # and at the more-anchors-than-kmeans operating point
                "maxivf_cagra_frac01_prune",
            ]
            configs = list(args.configs) if args.configs else full

    args.out.parent.mkdir(parents=True, exist_ok=True)
    if args.fresh and args.out.exists():
        args.out.unlink()

    # Resume support: scan existing rows and skip (dataset, config) pairs already done.
    completed: set[tuple[str, str]] = set()
    if args.resume and args.out.exists():
        with args.out.open("r", encoding="utf-8") as fh:
            for line in fh:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("_kind") == "run_header":
                    continue
                if "error" in obj:
                    continue  # don't skip — give it another chance
                key = (obj.get("dataset"), obj.get("config"))
                if all(key):
                    completed.add(key)
        if completed:
            print(f"[matrix] resume: skipping {len(completed)} already-completed cells")

    # Run header — first JSONL line documents the planned suite + env.
    import datetime
    import platform
    header = {
        "_kind": "run_header",
        "started_at": datetime.datetime.now().astimezone().isoformat(),
        "host": platform.node(),
        "suite": args.suite,
        "datasets": args.datasets,
        "configs": configs,
        "config_presets": {k: CONFIG_PRESETS[k] for k in configs},
        "split": args.split,
        "root": str(args.root),
        "model": args.model,
        "dtype": args.dtype,
        "max_docs": args.max_docs,
        "max_queries": args.max_queries,
        "per_cell_timeout_s": args.per_cell_timeout_s,
        "resume": args.resume,
    }
    with args.out.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(header, ensure_ascii=False) + "\n")
    print(f"[matrix] suite={args.suite} datasets={args.datasets} configs={configs}")
    print(f"[matrix] writing rows to {args.out}", flush=True)

    rows: list[dict] = []
    total_cells = len(args.datasets) * len(configs)
    cell_idx = 0
    for dataset in args.datasets:
        split = "dev" if dataset == "msmarco" else args.split
        print(f"[matrix] loading dataset={dataset} split={split} ...", flush=True)
        documents, queries, qrels = evaluation.load_beir(dataset_name=dataset, split=split)
        # On-disk dataset name is sanitised (e.g. cqadupstack/android -> cqadupstack_android).
        on_disk_dataset = dataset.replace("/", "_")
        documents_embeddings = _load_dataset_embeddings(
            dataset=on_disk_dataset, model=args.model, root=args.root,
            kind="docs", dtype=args.dtype,
        )
        queries_embeddings = _load_dataset_embeddings(
            dataset=on_disk_dataset, model=args.model, root=args.root,
            kind="queries", dtype=args.dtype,
        )

        if len(documents) != len(documents_embeddings):
            raise SystemExit(
                f"{dataset}: doc count mismatch ({len(documents)} vs {len(documents_embeddings)})"
            )
        if len(queries) != len(queries_embeddings):
            raise SystemExit(
                f"{dataset}: query count mismatch ({len(queries)} vs {len(queries_embeddings)})"
            )

        # Optional sub-sampling for smoke tests on large datasets.
        if args.max_docs is not None and args.max_docs < len(documents):
            documents = documents[: args.max_docs]
            documents_embeddings = documents_embeddings[: args.max_docs]
            print(f"[matrix] {dataset}: truncated to first {len(documents)} docs", flush=True)
        if args.max_queries is not None and args.max_queries < len(queries):
            qids = list(queries.keys())[: args.max_queries]
            queries = {qid: queries[qid] for qid in qids}
            queries_embeddings = queries_embeddings[: args.max_queries]
            qrels = {qid: qrels.get(qid, {}) for qid in qids}
            print(f"[matrix] {dataset}: truncated to first {len(queries)} queries", flush=True)

        for config_name in configs:
            cell_idx += 1
            if (dataset, config_name) in completed:
                print(f"[matrix] [{cell_idx}/{total_cells}] SKIP {dataset}/{config_name} (resume)", flush=True)
                continue
            config = CONFIG_PRESETS[config_name]
            print(f"[matrix] [{cell_idx}/{total_cells}] START {dataset}/{config_name}", flush=True)
            cell_t0 = time.perf_counter()
            try:
                row = _run_one_with_optional_timeout(
                    dataset=dataset,
                    config_name=config_name,
                    config=config,
                    documents=documents,
                    queries=queries,
                    qrels=qrels,
                    documents_embeddings=documents_embeddings,
                    queries_embeddings=queries_embeddings,
                    timeout_s=args.per_cell_timeout_s,
                )
            except Exception as e:
                row = {
                    "dataset": dataset,
                    "config": config_name,
                    "n_docs": len(documents),
                    "n_queries": len(queries),
                    "error": f"{type(e).__name__}: {e}",
                    "traceback": traceback.format_exc(),
                }
                print(f"[matrix] [{cell_idx}/{total_cells}] FAIL {dataset}/{config_name}: {row['error']}", flush=True)
            cell_wall = time.perf_counter() - cell_t0
            row["cell_wall_s"] = cell_wall
            rows.append(row)
            with args.out.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(f"[matrix] [{cell_idx}/{total_cells}] DONE {dataset}/{config_name} in {cell_wall:.1f}s", flush=True)
            _free_gpu_memory()

    # Summary table.
    print("\n\n========= MATRIX SUMMARY =========")
    headers = [
        "dataset", "config",
        "n_tokens", "n_cents",
        "frac",
        "add_s", "ret_s", "ms/q",
        "ndcg@10", "ndcg@100", "rec@10", "rec@100", "rec@1000", "map",
        "gpu_mb",
    ]
    fmt = "{:<10} {:<32} {:>10} {:>9} {:>5} {:>8} {:>7} {:>7} {:>8} {:>9} {:>8} {:>9} {:>9} {:>7} {:>8}"
    print(fmt.format(*headers))
    print("-" * 175)
    def _fmt_num(x, digits=4):
        if x is None:
            return "-"
        if isinstance(x, float) and x != x:
            return "-"
        return f"{x:.{digits}f}"
    def _fmt_int(x):
        if x is None:
            return "-"
        return f"{int(x):,}"
    for row in rows:
        if "error" in row:
            print(f"{row['dataset']:<10} {row['config']:<32} ERROR: {row['error']}")
            continue
        m = row["metrics"]
        frac = row.get("anchor_frac_effective", row.get("anchor_frac"))
        print(fmt.format(
            row["dataset"], row["config"],
            _fmt_int(row.get("n_tokens_total")),
            _fmt_int(row.get("n_centroids_planned")),
            "-" if frac is None else f"{frac:.3f}",
            f"{row['add_documents_s']:.1f}", f"{row['retrieve_s']:.2f}",
            f"{row['retrieve_ms_per_query']:.2f}",
            _fmt_num(m.get("ndcg@10")),
            _fmt_num(m.get("ndcg@100")),
            _fmt_num(m.get("recall@10")),
            _fmt_num(m.get("recall@100")),
            _fmt_num(m.get("recall@1000")),
            _fmt_num(m.get("map")),
            _fmt_num(row.get("add_peak_gpu_mb"), digits=0) if row.get("add_peak_gpu_mb") else "-",
        ))
    print(f"\nresults appended to {args.out}")


if __name__ == "__main__":
    main()
