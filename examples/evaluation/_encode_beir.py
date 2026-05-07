"""Encode a BEIR dataset's documents and queries to the v3 sharded cache layout.

Layout::

  <root>/<dataset>/<model>/<dtype>/{docs,queries}/
      meta.json
      shard0000.npz
      shard0001.npz
      ...

Usage::

  LD_LIBRARY_PATH=... uv run --no-sync python examples/evaluation/_encode_beir.py \
    --dataset trec-covid

The cache is invalidated by simply pointing at a different ``--root`` (or a different
model). Re-encoding kicks in automatically when no ``meta.json`` exists at the target
path; pass ``--force`` to overwrite an existing one.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# Make the local ``_embeddings_cache`` importable when run as a script.
sys.path.insert(0, str(Path(__file__).resolve().parent))

from _embeddings_cache import (  # noqa: E402  (local module)
    DEFAULT_ROOT,
    DEFAULT_SHARD_SIZE,
    DEFAULT_TOKENS_DTYPE,
    encode_and_save_sharded,
    load_manifest,
)

from pylate import evaluation, models  # noqa: E402


# Shared with beir_dataset.py — keep in sync if you add datasets there.
QUERY_LEN: dict[str, int] = {
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


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True, help="BEIR dataset name (e.g. trec-covid)")
    parser.add_argument(
        "--root",
        type=Path,
        default=DEFAULT_ROOT,
        help=f"Cache root directory (default: {DEFAULT_ROOT}/)",
    )
    parser.add_argument("--model", default="lightonai/GTE-ModernColBERT-v1")
    parser.add_argument("--document-length", type=int, default=300)
    parser.add_argument("--shard-size", type=int, default=DEFAULT_SHARD_SIZE)
    parser.add_argument("--doc-batch-size", type=int, default=6000)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument(
        "--dtype",
        default=DEFAULT_TOKENS_DTYPE,
        choices=["float16", "float32"],
        help=(
            "Compute + on-disk dtype. Drives both the encoder model's torch_dtype and the "
            "stored tokens dtype. bf16 is intentionally not supported (numpy compatibility)."
        ),
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Re-encode even if a manifest already exists at the target path.",
    )
    parser.add_argument(
        "--skip",
        choices=["none", "documents", "queries"],
        default="none",
        help="Skip one phase (e.g. when only queries changed).",
    )
    args = parser.parse_args()

    dataset = args.dataset
    if dataset not in QUERY_LEN:
        print(
            f"WARN: dataset {dataset!r} not in QUERY_LEN map; using query_length=32",
            flush=True,
        )
    query_length = QUERY_LEN.get(dataset, 32)

    args.root.mkdir(parents=True, exist_ok=True)

    print(
        f"[encode] loading model {args.model} "
        f"(document_length={args.document_length}, query_length={query_length}, dtype={args.dtype})",
        flush=True,
    )
    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=args.document_length,
        query_length=query_length,
        # Drive the encoder's compute dtype so the produced embeddings already match the
        # storage dtype — no narrowing/widening surprise at save time.
        model_kwargs={"torch_dtype": args.dtype},
    )
    model.compile()

    if dataset.startswith("cqadupstack/"):
        from beir import util

        util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset}",
            split="test",
        )
        on_disk_dataset = dataset.replace("/", "_")
        split = "test"
    else:
        split = "dev" if "msmarco" in dataset else "test"
        documents, queries, qrels = evaluation.load_beir(dataset_name=dataset, split=split)
        on_disk_dataset = dataset

    print(
        f"[encode] dataset={dataset!r} split={split} docs={len(documents)} queries={len(queries)}",
        flush=True,
    )

    common_cfg = {
        "split": split,
        "document_length": args.document_length,
        "query_length": query_length,
        "compute_dtype": args.dtype,
        "model_do_query_expansion": getattr(model, "do_query_expansion", None),
        "model_attend_to_expansion_tokens": getattr(model, "attend_to_expansion_tokens", None),
    }

    if args.skip != "documents":
        existing = load_manifest(
            root=args.root,
            dataset=on_disk_dataset,
            model=args.model,
            dtype=args.dtype,
            kind="docs",
        )
        if existing is not None and existing.num_items == len(documents) and not args.force:
            print(
                f"[encode] docs already cached: {existing.num_items} items "
                f"across {len(existing.shards)} shards; skipping (use --force to re-encode)",
                flush=True,
            )
        else:
            t0 = time.perf_counter()

            def _enc_docs(batch_sentences: list[str]) -> list:
                return model.encode(
                    sentences=batch_sentences,
                    batch_size=args.doc_batch_size,
                    is_query=False,
                    show_progress_bar=True,
                )

            encode_and_save_sharded(
                root=args.root,
                dataset=on_disk_dataset,
                model=args.model,
                kind="docs",
                config={
                    **common_cfg,
                    "is_query": False,
                    "batch_size": args.doc_batch_size,
                    "normalize_embeddings": True,
                    "pool_factor": 1,
                    "protected_tokens": 1,
                },
                doc_ids=[d["id"] for d in documents],
                sentences=[d["text"] for d in documents],
                encode_fn=_enc_docs,
                shard_size=args.shard_size,
                dtype=args.dtype,
                log_prefix="[encode/docs]",
            )
            print(f"[encode] docs wall {time.perf_counter() - t0:.1f}s", flush=True)

    if args.skip != "queries":
        existing = load_manifest(
            root=args.root,
            dataset=on_disk_dataset,
            model=args.model,
            dtype=args.dtype,
            kind="queries",
        )
        if existing is not None and existing.num_items == len(queries) and not args.force:
            print(
                f"[encode] queries already cached: {existing.num_items} items "
                f"across {len(existing.shards)} shards; skipping (use --force to re-encode)",
                flush=True,
            )
        else:
            t0 = time.perf_counter()

            def _enc_queries(batch_sentences: list[str]) -> list:
                return model.encode(
                    sentences=batch_sentences,
                    is_query=True,
                    show_progress_bar=True,
                    batch_size=args.query_batch_size,
                )

            encode_and_save_sharded(
                root=args.root,
                dataset=on_disk_dataset,
                model=args.model,
                kind="queries",
                config={
                    **common_cfg,
                    "is_query": True,
                    "batch_size": args.query_batch_size,
                    "normalize_embeddings": True,
                    "pool_factor": 1,
                    "protected_tokens": 1,
                },
                doc_ids=list(queries.keys()),
                sentences=list(queries.values()),
                encode_fn=_enc_queries,
                shard_size=args.shard_size,
                dtype=args.dtype,
                log_prefix="[encode/queries]",
            )
            print(f"[encode] queries wall {time.perf_counter() - t0:.1f}s", flush=True)


if __name__ == "__main__":
    main()
