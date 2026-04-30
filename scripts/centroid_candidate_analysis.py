"""Centroid candidate analysis for ColBERT-style retrieval.

Mirrors the candidate-generation stage that PLAID/fast_plaid run before
MaxSim scoring: cluster doc tokens with k-means, then for each query
score query tokens against centroids, take top-nprobe per token, and
union the centroid -> doc postings to count unique candidate docs.

Stage A (cached): build centroids + CSR posting list from the encoded
corpus shards produced by `dynamic_encode_hf_corpus.py`.

Stage B (per run): encode queries from a JSONL produced by
`parse_agentir_queries.py` and emit per-(query, num_probes) candidate
counts as JSONL.

Usage:
    srunv100 python scripts/centroid_candidate_analysis.py \\
        query_jsonl=outputs/agentir_queries_..._plain.jsonl \\
        max_queries=20
"""

from __future__ import annotations

import json
import logging
import math
import re
import time
from pathlib import Path

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from tqdm.auto import tqdm

from pylate import models

logger = logging.getLogger(__name__)


DTYPE_TORCH = {"fp32": torch.float32, "bf16": torch.bfloat16, "fp16": torch.float16}


def slugify(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_")


# ---------- shard discovery / metadata ----------


def discover_shards(shard_dir: Path) -> list[Path]:
    files = sorted(shard_dir.glob("doc_shard_*.npy"))
    return [
        f for f in files
        if "doclens" not in f.name and "docids" not in f.name and "token_ids" not in f.name
    ]


def shard_aux_paths(shard: Path) -> tuple[Path, Path]:
    stem = shard.with_suffix("").name  # "doc_shard_00000"
    return (
        shard.parent / f"{stem}.doclens.npy",
        shard.parent / f"{stem}.docids.json",
    )


def collect_corpus_metadata(shard_files: list[Path]) -> dict:
    """Concatenate per-shard doclens/docids without loading any embeddings."""
    all_doclens: list[np.ndarray] = []
    all_docids: list[str] = []
    shard_doc_starts = [0]
    shard_token_starts = [0]
    for shard in shard_files:
        doclens_path, docids_path = shard_aux_paths(shard)
        dl = np.load(doclens_path)
        with docids_path.open() as f:
            ids = json.load(f)
        if len(ids) != len(dl):
            raise ValueError(
                f"Shard {shard} has mismatched docids ({len(ids)}) vs doclens ({len(dl)})."
            )
        all_doclens.append(dl)
        all_docids.extend(str(x) for x in ids)
        shard_doc_starts.append(shard_doc_starts[-1] + len(dl))
        shard_token_starts.append(shard_token_starts[-1] + int(dl.sum()))

    return {
        "doclens": np.concatenate(all_doclens) if all_doclens else np.array([], dtype=np.int64),
        "docids": all_docids,
        "shard_doc_starts": shard_doc_starts,
        "shard_token_starts": shard_token_starts,
    }


# ---------- centroid build ----------


def plaid_num_partitions(n_tokens: int) -> int:
    """fast_plaid's centroid count: 2 ** floor(log2(16 * sqrt(num_partitions)))."""
    if n_tokens <= 0:
        raise ValueError("Need positive token count for k.")
    return int(2 ** math.floor(math.log2(16 * math.sqrt(n_tokens))))


def sample_tokens_for_kmeans(
    shard_files: list[Path],
    meta: dict,
    n_sample_docs: int,
    seed: int,
) -> np.ndarray:
    """Sample tokens from a random subset of documents and concatenate."""
    n_docs = len(meta["docids"])
    rng = np.random.default_rng(seed)
    n_sample_docs = min(n_sample_docs, n_docs)
    sampled = np.sort(rng.choice(n_docs, size=n_sample_docs, replace=False))

    doc_starts = np.array(meta["shard_doc_starts"], dtype=np.int64)
    # shard idx for each sampled doc
    shard_idx_for_doc = np.searchsorted(doc_starts, sampled, side="right") - 1

    chunks: list[np.ndarray] = []
    for s_idx in tqdm(range(len(shard_files)), desc="Sampling tokens for k-means", unit="shard"):
        mask = shard_idx_for_doc == s_idx
        if not mask.any():
            continue
        local_doc_idxs = sampled[mask] - doc_starts[s_idx]
        doclens_path, _ = shard_aux_paths(shard_files[s_idx])
        dl = np.load(doclens_path)
        token_offsets = np.concatenate([[0], np.cumsum(dl)])
        embs = np.load(shard_files[s_idx], mmap_mode="r")
        for d in local_doc_idxs:
            start = int(token_offsets[d])
            end = int(token_offsets[d + 1])
            if end > start:
                chunks.append(np.asarray(embs[start:end], dtype=np.float32))
    if not chunks:
        raise RuntimeError("No tokens sampled for k-means; corpus is empty?")
    return np.concatenate(chunks, axis=0)


def train_kmeans(
    sampled_tokens: np.ndarray,
    k: int,
    niters: int,
    max_points_per_centroid: int,
    seed: int,
    device: str,
) -> np.ndarray:
    """Run fastkmeans, return L2-normalized centroids in fp16 numpy."""
    from fastkmeans.kmeans import FastKMeans

    dim = sampled_tokens.shape[1]
    k = min(k, sampled_tokens.shape[0])
    logger.info(
        "Training fastkmeans: k=%d niter=%d max_pts_per_centroid=%d on %d tokens",
        k, niters, max_points_per_centroid, sampled_tokens.shape[0],
    )
    kmeans = FastKMeans(
        d=dim,
        k=k,
        niter=niters,
        gpu=device.startswith("cuda"),
        seed=seed,
        max_points_per_centroid=max_points_per_centroid,
        verbose=False,
    )
    kmeans.train(data=sampled_tokens)
    centroids = torch.from_numpy(kmeans.centroids).float()
    centroids = torch.nn.functional.normalize(centroids, dim=-1).half()
    return centroids.numpy()


def assign_tokens_streaming(
    shard_files: list[Path],
    meta: dict,
    centroids: np.ndarray,
    device: str,
    chunk_tokens: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Assign every token in the corpus to its nearest centroid.

    Returns (labels_int32 [n_tokens], token_doc_int32 [n_tokens]).
    """
    n_tokens = int(meta["doclens"].sum())
    k, dim = centroids.shape
    centroids_t = torch.from_numpy(centroids).to(device).half()
    centroid_tile = max(1, min(k, 16384))

    labels = np.empty(n_tokens, dtype=np.int32)
    token_doc = np.empty(n_tokens, dtype=np.int32)

    cursor = 0
    doc_cursor = 0
    for shard in tqdm(shard_files, desc="Assigning tokens", unit="shard"):
        doclens_path, _ = shard_aux_paths(shard)
        dl = np.load(doclens_path)
        T_s = int(dl.sum())
        # token -> global doc id within this shard
        global_docs = (doc_cursor + np.arange(len(dl))).astype(np.int32)
        token_doc[cursor:cursor + T_s] = np.repeat(global_docs, dl)

        embs = np.load(shard, mmap_mode="r")
        if embs.shape[0] != T_s:
            raise ValueError(
                f"Shard {shard.name}: emb rows {embs.shape[0]} != sum(doclens) {T_s}"
            )

        for start in range(0, T_s, chunk_tokens):
            end = min(start + chunk_tokens, T_s)
            batch = torch.from_numpy(np.ascontiguousarray(embs[start:end])).to(device).half()
            best_score = None
            best_idx = None
            for cs in range(0, k, centroid_tile):
                ce = min(cs + centroid_tile, k)
                tile = centroids_t[cs:ce]  # (Kc, D)
                scores = batch @ tile.T  # (n, Kc)
                tile_max, tile_arg = scores.max(dim=1)
                if best_score is None:
                    best_score = tile_max
                    best_idx = tile_arg + cs
                else:
                    update = tile_max > best_score
                    best_score = torch.where(update, tile_max, best_score)
                    best_idx = torch.where(update, tile_arg + cs, best_idx)
            labels[cursor + start:cursor + end] = best_idx.to(torch.int32).cpu().numpy()

        cursor += T_s
        doc_cursor += len(dl)

    if cursor != n_tokens:
        raise RuntimeError(f"Expected {n_tokens} tokens, assigned {cursor}.")

    del centroids_t
    if device.startswith("cuda"):
        torch.cuda.empty_cache()

    return labels, token_doc


def build_csr_postings(
    labels: np.ndarray,
    token_doc: np.ndarray,
    n_centroids: int,
) -> tuple[np.ndarray, np.ndarray]:
    """From per-token (centroid, doc) pairs, build a CSR posting list.

    Returns (offsets [K+1] int64, postings [n_pairs] int32) with each
    centroid's docs deduped and sorted ascending.
    """
    logger.info("Sorting %d (centroid, doc) pairs for CSR build", len(labels))
    order = np.lexsort((token_doc, labels))
    sorted_labels = labels[order]
    sorted_docs = token_doc[order]
    del order

    if len(sorted_labels) == 0:
        return np.zeros(n_centroids + 1, dtype=np.int64), np.array([], dtype=np.int32)

    keep = np.empty(len(sorted_labels), dtype=bool)
    keep[0] = True
    keep[1:] = (sorted_labels[1:] != sorted_labels[:-1]) | (sorted_docs[1:] != sorted_docs[:-1])
    posting_labels = sorted_labels[keep]
    posting_docs = sorted_docs[keep]
    del sorted_labels, sorted_docs, keep

    counts = np.bincount(posting_labels, minlength=n_centroids).astype(np.int64)
    offsets = np.zeros(n_centroids + 1, dtype=np.int64)
    offsets[1:] = np.cumsum(counts)
    return offsets, posting_docs.astype(np.int32, copy=False)


def cache_complete(cache_dir: Path) -> bool:
    needed = ["centroids.npy", "centroid_offsets.npy", "centroid_postings.npy",
              "docids.json", "meta.json"]
    return all((cache_dir / n).exists() for n in needed)


def build_cache(cfg: DictConfig, cache_dir: Path) -> None:
    shard_dir = Path(cfg.shard_dir)
    shard_files = discover_shards(shard_dir)
    if not shard_files:
        raise FileNotFoundError(f"No doc_shard_*.npy files in {shard_dir}")
    logger.info("Found %d shards in %s", len(shard_files), shard_dir)

    meta = collect_corpus_metadata(shard_files)
    n_docs = len(meta["docids"])
    n_tokens = int(meta["doclens"].sum())
    logger.info(
        "Corpus: docs=%d, tokens=%d (mean %.1f tok/doc)",
        n_docs, n_tokens, n_tokens / max(1, n_docs),
    )

    if cfg.kmeans.num_partitions is not None:
        k = int(cfg.kmeans.num_partitions)
    else:
        k = plaid_num_partitions(n_tokens)
    logger.info("Centroid count k=%d", k)

    n_sample_docs = min(1 + int(16 * math.sqrt(120 * n_docs)), n_docs)
    sampled = sample_tokens_for_kmeans(
        shard_files, meta, n_sample_docs=n_sample_docs, seed=int(cfg.kmeans.seed),
    )
    logger.info("Sampled %d tokens from %d docs", sampled.shape[0], n_sample_docs)

    centroids = train_kmeans(
        sampled_tokens=sampled,
        k=k,
        niters=int(cfg.kmeans.niters),
        max_points_per_centroid=int(cfg.kmeans.max_points_per_centroid),
        seed=int(cfg.kmeans.seed),
        device=str(cfg.device),
    )
    del sampled

    labels, token_doc = assign_tokens_streaming(
        shard_files=shard_files,
        meta=meta,
        centroids=centroids,
        device=str(cfg.device),
        chunk_tokens=int(cfg.kmeans.assign_chunk_tokens),
    )
    offsets, postings = build_csr_postings(labels, token_doc, n_centroids=centroids.shape[0])
    del labels, token_doc

    cache_dir.mkdir(parents=True, exist_ok=True)
    np.save(cache_dir / "centroids.npy", centroids)
    np.save(cache_dir / "centroid_offsets.npy", offsets)
    np.save(cache_dir / "centroid_postings.npy", postings)
    with (cache_dir / "docids.json").open("w") as f:
        json.dump(meta["docids"], f)
    cache_meta = {
        "model": str(cfg.model),
        "shard_dir": str(shard_dir),
        "corpus_slug": str(cfg.corpus_slug),
        "n_docs": n_docs,
        "n_tokens": n_tokens,
        "k": int(centroids.shape[0]),
        "dim": int(centroids.shape[1]),
        "kmeans": OmegaConf.to_container(cfg.kmeans, resolve=True),
        "n_postings": int(postings.shape[0]),
    }
    with (cache_dir / "meta.json").open("w") as f:
        json.dump(cache_meta, f, indent=2)
    logger.info("Cache written to %s", cache_dir)


# ---------- query analysis ----------


def load_query_jsonl(path: Path, max_queries: int | None) -> list[dict]:
    rows: list[dict] = []
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            rows.append({"id": str(row["id"]), "query": str(row.get("query", ""))})
            if max_queries is not None and len(rows) >= int(max_queries):
                break
    return rows


def real_query_lengths(model: models.ColBERT, queries: list[str]) -> list[int]:
    """Number of non-[MASK] tokens per query.

    PyLate query encoding pads each query to ``query_length`` with [MASK]
    tokens (deliberate MASK-expansion). For this analysis we want centroid
    lookups driven only by *real* query tokens, so we count non-MASK
    positions and slice the encoded embedding to that length downstream.
    """
    mask_id = model.tokenizer.mask_token_id
    if mask_id is None:
        raise RuntimeError("Tokenizer has no mask_token_id; cannot strip MASK expansion.")
    features = model.tokenize(texts=queries, is_query=True)
    input_ids = features["input_ids"]
    if isinstance(input_ids, np.ndarray):
        return (input_ids != mask_id).sum(axis=1).tolist()
    return (input_ids != mask_id).sum(dim=1).tolist()


def encode_queries(
    model: models.ColBERT,
    queries: list[str],
    batch_size: int,
    device: str,
    dtype: str,
) -> list[np.ndarray]:
    torch_dtype = DTYPE_TORCH[dtype]
    with torch.autocast(device_type=device.split(":")[0], dtype=torch_dtype, enabled=dtype != "fp32"):
        embs = model.encode(
            queries,
            batch_size=batch_size,
            is_query=True,
            convert_to_numpy=True,
            show_progress_bar=True,
        )
    # `encode` returns list[ndarray] for variable-length token outputs.
    return [np.asarray(e, dtype=np.float32) for e in embs]


def encode_and_strip_masks(
    model: models.ColBERT,
    queries: list[str],
    batch_size: int,
    device: str,
    dtype: str,
) -> list[np.ndarray]:
    """Encode queries and slice each embedding to its non-MASK prefix."""
    real_lens = real_query_lengths(model, queries)
    full_embs = encode_queries(
        model=model, queries=queries, batch_size=batch_size, device=device, dtype=dtype,
    )
    sliced: list[np.ndarray] = []
    for emb, n_real in zip(full_embs, real_lens):
        n = min(int(n_real), emb.shape[0])
        sliced.append(emb[:n] if n < emb.shape[0] else emb)
    return sliced


def analyse_queries(
    centroids_t: torch.Tensor,
    offsets: np.ndarray,
    postings: np.ndarray,
    n_docs: int,
    queries: list[dict],
    query_embs: list[np.ndarray],
    nprobes: list[int],
    device: str,
    expansion_slug: str,
) -> list[dict]:
    max_nprobe = max(nprobes)
    visited_docs = np.zeros(n_docs, dtype=bool)

    out: list[dict] = []
    for q, q_emb in tqdm(
        list(zip(queries, query_embs)),
        desc=f"Scoring [{expansion_slug}]",
        unit="query",
    ):
        q_len = int(q_emb.shape[0])
        if q_len == 0:
            for nprobe in nprobes:
                out.append({
                    "id": q["id"],
                    "expansion": expansion_slug,
                    "query_len": 0,
                    "num_probes": int(nprobe),
                    "n_unique_centroids": 0,
                    "n_unique_docs": 0,
                })
            continue
        q_t = torch.from_numpy(q_emb).to(device).float()
        scores = q_t @ centroids_t.T
        k_top = min(max_nprobe, scores.shape[1])
        top_ids = torch.topk(scores, k=k_top, dim=1).indices.cpu().numpy()

        for nprobe in nprobes:
            np_eff = min(int(nprobe), top_ids.shape[1])
            chosen = top_ids[:, :np_eff].reshape(-1)
            unique_centroids = np.unique(chosen)
            visited_docs.fill(False)
            for cid in unique_centroids:
                start = offsets[cid]
                end = offsets[cid + 1]
                if end > start:
                    visited_docs[postings[start:end]] = True
            n_unique_docs = int(visited_docs.sum())
            out.append({
                "id": q["id"],
                "expansion": expansion_slug,
                "query_len": q_len,
                "num_probes": int(nprobe),
                "n_unique_centroids": int(unique_centroids.shape[0]),
                "n_unique_docs": n_unique_docs,
            })

    return out


def resolve_query_jsonls(cfg: DictConfig) -> list[Path]:
    """Accept either ``query_jsonls: [list]`` or legacy ``query_jsonl: str``."""
    files: list[Path] = []
    if "query_jsonls" in cfg and cfg.query_jsonls is not None:
        items = cfg.query_jsonls
        if isinstance(items, str):
            items = [items]
        for item in items:
            files.append(Path(str(item)))
    elif "query_jsonl" in cfg and cfg.query_jsonl is not None:
        files.append(Path(str(cfg.query_jsonl)))
    if not files:
        raise ValueError("Provide query_jsonls (list) or query_jsonl (single path).")
    return files


def expansion_slug_for(path: Path) -> str:
    """Strip the boilerplate AgentIR prefix from the filename for cleaner labels."""
    stem = path.stem
    for prefix in (
        "agentir_queries_Tevatron_AgentIR-data_train_",
        "agentir_queries_",
    ):
        if stem.startswith(prefix):
            return stem[len(prefix):]
    return stem


# ---------- entrypoint ----------


@hydra.main(
    config_path="../conf/centroid_analysis",
    config_name="config",
    version_base=None,
)
def main(cfg: DictConfig) -> None:
    logger.info("Config:\n%s", OmegaConf.to_yaml(cfg))

    cache_dir = Path(cfg.cache_root) / f"{slugify(cfg.model)}__{cfg.corpus_slug}"
    if cfg.rebuild_cache or not cache_complete(cache_dir):
        logger.info("Building centroid+postings cache at %s", cache_dir)
        t0 = time.perf_counter()
        build_cache(cfg, cache_dir)
        logger.info("Cache build took %.1fs", time.perf_counter() - t0)
    else:
        logger.info("Using existing cache at %s", cache_dir)

    centroids = np.load(cache_dir / "centroids.npy")
    offsets = np.load(cache_dir / "centroid_offsets.npy")
    postings = np.load(cache_dir / "centroid_postings.npy")
    with (cache_dir / "docids.json").open() as f:
        docids = json.load(f)
    with (cache_dir / "meta.json").open() as f:
        cache_meta = json.load(f)
    n_docs = len(docids)
    logger.info(
        "Cache: k=%d dim=%d n_docs=%d n_postings=%d",
        centroids.shape[0], centroids.shape[1], n_docs, postings.shape[0],
    )

    query_paths = resolve_query_jsonls(cfg)
    logger.info("Will analyse %d query file(s):", len(query_paths))
    for p in query_paths:
        logger.info("  %s", p)

    logger.info("Loading model %s on %s", cfg.model, cfg.device)
    model = models.ColBERT(
        model_name_or_path=str(cfg.model),
        device=str(cfg.device),
        query_length=int(cfg.query_length),
    )
    model.eval()

    centroids_t = torch.from_numpy(centroids).to(str(cfg.device)).float()
    nprobes = [int(x) for x in cfg.nprobes]

    out_root = Path(cfg.output_root) / cache_dir.name
    out_root.mkdir(parents=True, exist_ok=True)

    overall_summary: dict[str, dict[int, np.ndarray]] = {}

    for query_path in query_paths:
        exp_slug = expansion_slug_for(query_path)
        queries = load_query_jsonl(query_path, cfg.max_queries)
        if not queries:
            logger.warning("Skipping empty query file: %s", query_path)
            continue
        logger.info("Loaded %d queries from %s (expansion=%s)",
                    len(queries), query_path, exp_slug)

        t_enc = time.perf_counter()
        query_embs = encode_and_strip_masks(
            model=model,
            queries=[q["query"] for q in queries],
            batch_size=int(cfg.query_batch_size),
            device=str(cfg.device),
            dtype=str(cfg.query_dtype),
        )
        logger.info("Encoded %d queries in %.2fs",
                    len(queries), time.perf_counter() - t_enc)
        real_lens = np.asarray([e.shape[0] for e in query_embs])
        logger.info(
            "Real query lengths (post MASK strip): min=%d median=%.0f mean=%.1f p95=%.0f max=%d",
            int(real_lens.min()), float(np.median(real_lens)),
            float(real_lens.mean()), float(np.percentile(real_lens, 95)),
            int(real_lens.max()),
        )

        rows = analyse_queries(
            centroids_t=centroids_t,
            offsets=offsets,
            postings=postings,
            n_docs=n_docs,
            queries=queries,
            query_embs=query_embs,
            nprobes=nprobes,
            device=str(cfg.device),
            expansion_slug=exp_slug,
        )

        out_path = out_root / f"{exp_slug}.jsonl"
        with out_path.open("w") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        logger.info("Wrote %d rows to %s", len(rows), out_path)

        by_np: dict[int, np.ndarray] = {}
        for r in rows:
            by_np.setdefault(r["num_probes"], []).append(r["n_unique_docs"])
        overall_summary[exp_slug] = {
            np_v: np.asarray(vals) for np_v, vals in by_np.items()
        }

        del query_embs
        if str(cfg.device).startswith("cuda"):
            torch.cuda.empty_cache()

    del centroids_t
    if str(cfg.device).startswith("cuda"):
        torch.cuda.empty_cache()

    # Console summary across all expansions
    print(f"\nCorpus n_docs={n_docs:,}  k={cache_meta['k']:,}")
    for exp_slug, by_np in overall_summary.items():
        n_q = len(next(iter(by_np.values()))) if by_np else 0
        print(f"\n=== {exp_slug}  (queries={n_q:,}) ===")
        print(f"  {'nprobe':>6} {'mean':>10} {'median':>10} {'p95':>10} {'max':>10} {'%docs':>8}")
        for np_v in sorted(by_np):
            arr = by_np[np_v]
            pct = arr.mean() / max(1, n_docs) * 100
            print(
                f"  {np_v:>6d} {arr.mean():>10.0f} {np.median(arr):>10.0f} "
                f"{np.percentile(arr, 95):>10.0f} {arr.max():>10d} {pct:>7.1f}%"
            )


if __name__ == "__main__":
    main()
