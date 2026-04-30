"""Dynamically batch-encode an HF corpus with ColBERT.

This script is standalone and focused on long-document corpora where fixed
batch size is inefficient or unstable.

Key features:
  - Length-aware sorting (`--order smallest|biggest`)
  - Cost-budget batching based on sum(length^2)
  - OOM recovery via automatic budget backoff
  - Optional shard saving to .npy + .doclens + .docids

Example:
  srun -u -t 24:00:00 --gres=gpu:h100:1 python scripts/dynamic_encode_hf_corpus.py \
    --dataset Tevatron/browsecomp-plus-corpus \
    --model lightonai/GTE-ModernColBERT-v1 \
    --document-length 100000 \
    --dtype bf16 \
    --order biggest \
    --target-cost 6e8 \
    --output-dir outputs/browsecomp_dynamic_enc
"""

from __future__ import annotations

import argparse
import json
import re
import time
from pathlib import Path

import numpy as np
import torch
from datasets import load_dataset
from tqdm.auto import tqdm

from pylate import models


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        type=str,
        default="Tevatron/browsecomp-plus-corpus",
        help="HF dataset name.",
    )
    parser.add_argument("--split", type=str, default="train", help="HF split.")
    parser.add_argument(
        "--text-column",
        type=str,
        default="text",
        help="Document text column in dataset.",
    )
    parser.add_argument(
        "--id-column",
        type=str,
        default="docid",
        help="Document id column in dataset.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="ColBERT model name/path.",
    )
    parser.add_argument(
        "--document-length",
        type=int,
        default=16384,
        help="Max document token length used by model tokenization.",
    )
    parser.add_argument(
        "--dtype",
        choices=("fp32", "bf16", "fp16"),
        default="bf16",
        help="Model dtype.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device for encoding.",
    )
    parser.add_argument(
        "--order",
        choices=("smallest", "biggest"),
        default="smallest",
        help="Length sort direction before batching.",
    )
    parser.add_argument(
        "--target-cost",
        type=float,
        default=1.0e8,
        help="Initial batch cost budget, where cost=sum(length^2).",
    )
    parser.add_argument(
        "--min-target-cost",
        type=float,
        default=1.0e6,
        help="Lower bound for target cost after backoff.",
    )
    parser.add_argument(
        "--backoff-factor",
        type=float,
        default=0.7,
        help="Multiply target cost by this value on OOM.",
    )
    parser.add_argument(
        "--growth-factor",
        type=float,
        default=1.03,
        help="Multiply target cost by this value after successful batches.",
    )
    parser.add_argument(
        "--max-batch-docs",
        type=int,
        default=2048,
        help="Hard cap on docs per batch.",
    )
    parser.add_argument(
        "--length-batch-size",
        type=int,
        default=512,
        help="Batch size used for tokenizer length pre-pass.",
    )
    parser.add_argument(
        "--max-docs",
        type=int,
        default=None,
        help="Optional subset size for quick tests.",
    )
    parser.add_argument(
        "--sample",
        type=str,
        default=None,
        help=(
            "Optional random sample size. "
            "If <1, treated as fraction (e.g. 0.1). "
            "If >=1, treated as document count (e.g. 5000)."
        ),
    )
    parser.add_argument(
        "--sample-seed",
        type=int,
        default=42,
        help="Random seed used with --sample.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Optional output dir for shard files.",
    )
    parser.add_argument(
        "--shard-max-docs",
        type=int,
        default=10000,
        help="Flush a shard when buffered docs reach this count.",
    )
    parser.add_argument(
        "--shard-max-vectors",
        type=int,
        default=2_000_000,
        help="Flush a shard when buffered token vectors reach this count.",
    )
    parser.add_argument(
        "--compile",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable torch.compile for potential encode speedups (default: true).",
    )
    return parser.parse_args()


def dtype_from_name(name: str) -> torch.dtype:
    if name == "bf16":
        return torch.bfloat16
    if name == "fp16":
        return torch.float16
    return torch.float32


def estimate_capped_lengths_from_chars(
    texts: list[str],
    document_length: int,
) -> list[int]:
    """Estimate token lengths from character length with a cheap heuristic.

    Heuristic:
      estimated_tokens = min(document_length, max(1, len(text) // 3))
    """
    lengths: list[int] = []
    for text in tqdm(texts, desc="Length pre-pass (char//3)", unit="doc"):
        est = max(1, len(text) // 3)
        lengths.append(min(document_length, est))
    return lengths


def save_shard(
    out_dir: Path,
    shard_idx: int,
    batch_doc_ids: list[str],
    batch_embeddings: list[np.ndarray | torch.Tensor],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    batch_embeddings_np: list[np.ndarray] = []
    for emb in batch_embeddings:
        if isinstance(emb, torch.Tensor):
            # NumPy does not consistently support bfloat16 tensors across
            # environments; cast to fp16 on CPU before conversion.
            emb = emb.detach().to(device="cpu", dtype=torch.float16).numpy()
        batch_embeddings_np.append(emb)

    doclens = np.asarray([emb.shape[0] for emb in batch_embeddings_np], dtype=np.int32)
    merged = np.concatenate(batch_embeddings_np, axis=0).astype(np.float16, copy=False)

    np.save(out_dir / f"doc_shard_{shard_idx:05d}.npy", merged)
    np.save(out_dir / f"doc_shard_{shard_idx:05d}.doclens.npy", doclens)
    with open(out_dir / f"doc_shard_{shard_idx:05d}.docids.json", "w") as f:
        json.dump(batch_doc_ids, f)


def flush_buffered_shard(
    out_dir: Path | None,
    shard_idx: int,
    buffered_doc_ids: list[str],
    buffered_embeddings: list[np.ndarray | torch.Tensor],
) -> int:
    if out_dir is None or not buffered_doc_ids:
        return shard_idx
    save_shard(
        out_dir=out_dir,
        shard_idx=shard_idx,
        batch_doc_ids=buffered_doc_ids,
        batch_embeddings=buffered_embeddings,
    )
    return shard_idx + 1


def inspect_existing_shards(out_dir: Path) -> tuple[int, int, int]:
    """Inspect existing shards and return (num_docs, num_vectors, next_shard_idx)."""
    pattern = re.compile(r"doc_shard_(\d+)\.docids\.json$")
    shard_indices: list[int] = []
    docs_total = 0
    vectors_total = 0

    for p in out_dir.glob("doc_shard_*.docids.json"):
        m = pattern.match(p.name)
        if not m:
            continue
        idx = int(m.group(1))
        shard_indices.append(idx)

        with p.open() as f:
            doc_ids = json.load(f)
        docs_total += len(doc_ids)

        doclens_path = out_dir / f"doc_shard_{idx:05d}.doclens.npy"
        if doclens_path.exists():
            vectors_total += int(np.load(doclens_path).sum())

    if not shard_indices:
        return 0, 0, 0
    return docs_total, vectors_total, max(shard_indices) + 1


def main() -> None:
    args = parse_args()
    torch_dtype = dtype_from_name(args.dtype)

    print(f"Loading dataset: {args.dataset} [{args.split}]")
    ds = load_dataset(args.dataset, split=args.split)
    if args.max_docs is not None:
        ds = ds.select(range(min(args.max_docs, len(ds))))
    if args.sample is not None:
        total = len(ds)
        sample_value = float(args.sample)
        if sample_value <= 0:
            raise ValueError("--sample must be > 0.")
        if sample_value < 1:
            sample_n = int(total * sample_value)
            sample_n = max(sample_n, 1)
        else:
            sample_n = int(sample_value)
        sample_n = min(sample_n, total)
        if sample_n < total:
            ds = ds.shuffle(seed=args.sample_seed).select(range(sample_n))
            print(
                f"Applied random sample: sample_n={sample_n:,} "
                f"(from {total:,}, seed={args.sample_seed})"
            )
    print(f"rows={len(ds):,} columns={ds.column_names}")

    if args.text_column not in ds.column_names:
        raise ValueError(f"text column '{args.text_column}' not in {ds.column_names}")
    if args.id_column not in ds.column_names:
        raise ValueError(f"id column '{args.id_column}' not in {ds.column_names}")

    doc_ids = [str(x) for x in ds[args.id_column]]
    texts = [str(x) for x in ds[args.text_column]]

    print(
        f"Loading model: {args.model} | dtype={args.dtype} | document_length={args.document_length}"
    )
    model = models.ColBERT(
        model_name_or_path=args.model,
        document_length=args.document_length,
        device=args.device,
        model_kwargs={"dtype": torch_dtype},
    )
    if args.compile:
        try:
            model = torch.compile(model)
            print("torch.compile: enabled")
        except Exception as exc:
            print(f"torch.compile: unavailable/fallback ({type(exc).__name__}: {exc})")
    print(f"model.device={model.device} model.dtype={model.dtype}")

    lengths = estimate_capped_lengths_from_chars(
        texts=texts,
        document_length=args.document_length,
    )
    lengths_np = np.asarray(lengths, dtype=np.int32)
    print(
        "capped length stats: "
        f"p50={np.percentile(lengths_np, 50):.0f} "
        f"p90={np.percentile(lengths_np, 90):.0f} "
        f"p99={np.percentile(lengths_np, 99):.0f} "
        f"max={lengths_np.max():.0f}"
    )

    order_indices = np.argsort(lengths_np)
    if args.order == "biggest":
        order_indices = order_indices[::-1]
    order_indices = order_indices.tolist()

    out_dir = Path(args.output_dir) if args.output_dir else None
    if out_dir is not None:
        out_dir.mkdir(parents=True, exist_ok=True)

    target_cost = float(args.target_cost)
    shard_idx = 0
    i = 0
    resumed_docs = 0
    resumed_vectors = 0
    if out_dir is not None:
        resumed_docs, resumed_vectors, shard_idx = inspect_existing_shards(out_dir)
        if resumed_docs > 0:
            i = min(resumed_docs, len(order_indices))
            print(
                "Resume detected: "
                f"existing_docs={resumed_docs:,} "
                f"existing_vectors={resumed_vectors:,} "
                f"next_shard_idx={shard_idx:,} "
                f"skip_first_docs={i:,}"
            )
    oom_count = 0
    skipped = 0
    total_encoded_docs = resumed_docs
    total_vectors = resumed_vectors
    total_encode_s = 0.0
    buffered_doc_ids: list[str] = []
    buffered_embeddings: list[np.ndarray | torch.Tensor] = []
    buffered_vectors = 0

    progress = tqdm(total=len(order_indices), desc="Encoding docs", unit="doc")
    if i > 0:
        progress.update(i)

    def set_progress_postfix(
        status: str,
        batch_docs: int,
        batch_cost: float,
        target: float,
        oom: int,
        buffer_docs: int,
        buffer_vecs: int,
    ) -> None:
        progress.set_postfix_str(
            (
                f"status={status} "
                f"bs={batch_docs} "
                f"bc={batch_cost:.2e} "
                f"tgt={target:.2e} "
                f"oom={oom} "
                f"bufd={buffer_docs} "
                f"bufv={buffer_vecs}"
            ),
            refresh=True,
        )

    while i < len(order_indices):
        start_i = i
        batch_indices: list[int] = []
        batch_cost = 0.0

        while i < len(order_indices) and len(batch_indices) < args.max_batch_docs:
            idx = order_indices[i]
            c = float(lengths_np[idx]) ** 2
            if batch_indices and (batch_cost + c > target_cost):
                break
            batch_indices.append(idx)
            batch_cost += c
            i += 1

        # Always include at least one doc
        if not batch_indices:
            idx = order_indices[i]
            batch_indices = [idx]
            batch_cost = float(lengths_np[idx]) ** 2
            i += 1

        batch_texts = [texts[idx] for idx in batch_indices]
        batch_doc_ids = [doc_ids[idx] for idx in batch_indices]
        set_progress_postfix(
            status="run",
            batch_docs=len(batch_indices),
            batch_cost=batch_cost,
            target=target_cost,
            oom=oom_count,
            buffer_docs=len(buffered_doc_ids),
            buffer_vecs=buffered_vectors,
        )

        try:
            t0 = time.time()
            batch_embeddings = model.encode(
                sentences=batch_texts,
                is_query=False,
                batch_size=len(batch_texts),
                show_progress_bar=False,
                convert_to_numpy=False,
            )
            elapsed = time.time() - t0
            total_encode_s += elapsed

            batch_vectors = int(sum(emb.shape[0] for emb in batch_embeddings))
            total_vectors += batch_vectors
            total_encoded_docs += len(batch_indices)
            if out_dir is not None:
                buffered_doc_ids.extend(batch_doc_ids)
                buffered_embeddings.extend(batch_embeddings)
                buffered_vectors += batch_vectors
                should_flush = (
                    len(buffered_doc_ids) >= args.shard_max_docs
                    or buffered_vectors >= args.shard_max_vectors
                )
                if should_flush:
                    shard_idx = flush_buffered_shard(
                        out_dir=out_dir,
                        shard_idx=shard_idx,
                        buffered_doc_ids=buffered_doc_ids,
                        buffered_embeddings=buffered_embeddings,
                    )
                    buffered_doc_ids = []
                    buffered_embeddings = []
                    buffered_vectors = 0

            progress.update(len(batch_indices))
            set_progress_postfix(
                status="ok",
                batch_docs=len(batch_indices),
                batch_cost=batch_cost,
                target=target_cost,
                oom=oom_count,
                buffer_docs=len(buffered_doc_ids),
                buffer_vecs=buffered_vectors,
            )

            target_cost *= args.growth_factor
        except torch.OutOfMemoryError:
            oom_count += 1
            torch.cuda.empty_cache()
            set_progress_postfix(
                status="oom",
                batch_docs=len(batch_indices),
                batch_cost=batch_cost,
                target=target_cost,
                oom=oom_count,
                buffer_docs=len(buffered_doc_ids),
                buffer_vecs=buffered_vectors,
            )

            if len(batch_indices) == 1:
                skipped += 1
                progress.update(1)
                target_cost = max(args.min_target_cost, target_cost * args.backoff_factor)
                continue

            # Retry same region with lower budget
            i = start_i
            target_cost = max(args.min_target_cost, target_cost * args.backoff_factor)

    progress.close()

    # Flush remaining buffered embeddings.
    if out_dir is not None and buffered_doc_ids:
        shard_idx = flush_buffered_shard(
            out_dir=out_dir,
            shard_idx=shard_idx,
            buffered_doc_ids=buffered_doc_ids,
            buffered_embeddings=buffered_embeddings,
        )

    docs_per_s = total_encoded_docs / total_encode_s if total_encode_s > 0 else 0.0
    vecs_per_s = total_vectors / total_encode_s if total_encode_s > 0 else 0.0
    print("\nDone")
    print(
        f"encoded_docs={total_encoded_docs:,} skipped_docs={skipped:,} "
        f"oom_events={oom_count:,} resumed_docs={resumed_docs:,}"
    )
    print(f"total_vectors={total_vectors:,} encode_time_s={total_encode_s:.2f}")
    print(f"throughput: docs/s={docs_per_s:.2f} vectors/s={vecs_per_s:.2f}")
    print(f"final_target_cost={target_cost:.2e}")
    if out_dir is not None:
        print(f"shards_written={shard_idx:,} output_dir={out_dir}")


if __name__ == "__main__":
    main()
