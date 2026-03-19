"""Ablation: compare index creation times with fastkmeans vs flash-kmeans.

Uses cached embeddings from benchmark_cache/ (run benchmark_indexes.py first
to populate the cache, or this script will encode on first run).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import pickle
import time
import unittest.mock as mock

import numpy as np
import torch

from pylate import evaluation, indexes, models

QUERY_LEN_MAP = {
    "quora": 32, "climate-fever": 64, "nq": 32, "msmarco": 32,
    "hotpotqa": 32, "nfcorpus": 32, "scifact": 48, "trec-covid": 48,
    "fiqa": 32, "arguana": 64, "scidocs": 48, "dbpedia-entity": 32,
    "webis-touche2020": 32, "fever": 32,
}


def cache_path(cache_dir: str, dataset: str, model_short: str, name: str) -> str:
    return os.path.join(cache_dir, f"{dataset}_{model_short}_{name}.pkl")


def load_or_encode(
    dataset: str,
    model_name: str,
    batch_size: int,
    document_length: int,
    cache_dir: str,
):
    os.makedirs(cache_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]

    data_cp = cache_path(cache_dir, dataset, model_short, "data")
    if os.path.exists(data_cp):
        print(f"  Loading cached dataset...")
        with open(data_cp, "rb") as f:
            documents, queries, qrels = pickle.load(f)
    else:
        print(f"  Loading dataset from BEIR ({dataset})...")
        split = "dev" if "msmarco" in dataset else "test"
        documents, queries, qrels = evaluation.load_beir(dataset, split=split)
        with open(data_cp, "wb") as f:
            pickle.dump((documents, queries, qrels), f)

    doc_cp = cache_path(cache_dir, dataset, model_short, "doc_emb")
    if os.path.exists(doc_cp):
        print(f"  Loading cached document embeddings...")
        with open(doc_cp, "rb") as f:
            documents_embeddings = pickle.load(f)
    else:
        print(f"  Encoding {len(documents)} documents...")
        model = models.ColBERT(
            model_name_or_path=model_name, document_length=document_length,
        )
        documents_embeddings = model.encode(
            sentences=[doc["text"] for doc in documents],
            batch_size=batch_size,
            is_query=False,
            show_progress_bar=True,
        )
        with open(doc_cp, "wb") as f:
            pickle.dump(documents_embeddings, f)
        del model
        gc.collect()

    return documents, documents_embeddings


class FlashKMeansAdapter:
    """Adapter that wraps flash_kmeans.FlashKMeans to match fastkmeans.FastKMeans API.

    FastKMeans.train() takes numpy arrays; FlashKMeans.train() takes torch tensors.
    This adapter handles the conversion and maps constructor args.
    """

    def __init__(
        self,
        d: int,
        k: int,
        niter: int = 25,
        tol: float = 1e-8,
        gpu: bool = True,
        seed: int = 0,
        max_points_per_centroid: int = 1_000_000_000,
        chunk_size_data: int = 51_200,
        chunk_size_centroids: int = 10_240,
        device=None,
        dtype=None,
        pin_gpu_memory: bool = True,
        verbose: bool = False,
        nredo: int = 1,
        use_triton: bool | None = None,
    ):
        from flash_kmeans import FlashKMeans

        flash_device = None
        if device is not None:
            flash_device = torch.device(device) if isinstance(device, str) else device
        elif gpu and torch.cuda.is_available():
            flash_device = torch.device("cuda")

        self._flash = FlashKMeans(
            d=d,
            k=k,
            niter=niter,
            tol=tol,
            seed=seed,
            chunk_size_data=chunk_size_data,
            chunk_size_centroids=chunk_size_centroids,
            device=flash_device,
            dtype=dtype,
            verbose=verbose,
            use_triton=use_triton if use_triton is not None else True,
        )
        self.max_points_per_centroid = max_points_per_centroid
        self.k = k
        self.centroids = None

    def train(self, data: np.ndarray):
        """Train with numpy input (FastKMeans compat), delegates to FlashKMeans."""
        data_torch = torch.from_numpy(data)

        # Subsample if needed (FastKMeans does this internally, FlashKMeans may not)
        n_samples = data_torch.shape[0]
        if self.max_points_per_centroid is not None and n_samples > self.k * self.max_points_per_centroid:
            target = self.k * self.max_points_per_centroid
            perm = torch.randperm(n_samples)[:target]
            data_torch = data_torch[perm]

        self._flash.train(data_torch)
        # centroids_b has shape (batch=1, k, d) — squeeze the batch dim
        self.centroids = self._flash.centroids_b.squeeze(0).cpu().numpy()


def patch_kmeans_for_module(module_path: str):
    """Return a mock.patch that replaces FastKMeans with FlashKMeansAdapter in the given module."""
    return mock.patch(f"{module_path}.FastKMeans", FlashKMeansAdapter)


def build_index_timed(
    index_type: str,
    documents: list[dict],
    documents_embeddings: list,
    index_folder: str,
    index_name: str,
    device: str | None,
) -> tuple[float, int]:
    """Build an index, return (build_time_s, num_partitions_estimate)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    doc_ids = [doc["id"] for doc in documents]

    if index_type == "plaid":
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            device=device,
        )
    elif index_type == "warp":
        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            device=device,
        )
    else:
        raise ValueError(f"Unsupported index type for kmeans ablation: {index_type}")

    start = time.perf_counter()
    index.add_documents(
        documents_ids=doc_ids,
        documents_embeddings=documents_embeddings,
    )
    build_time = time.perf_counter() - start
    return build_time


def append_jsonl(path: str, row: dict) -> None:
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Ablation: compare fastkmeans vs flash-kmeans index creation times."
    )
    parser.add_argument(
        "--datasets", nargs="+", default=["nfcorpus"],
        help="BEIR dataset names",
    )
    parser.add_argument(
        "--model", type=str, default="lightonai/GTE-ModernColBERT-v1",
    )
    parser.add_argument(
        "--indexes", nargs="+", default=["plaid", "warp"],
        choices=["plaid", "warp"],
        help="Index types to benchmark (only plaid and warp use kmeans)",
    )
    parser.add_argument(
        "--kmeans", nargs="+", default=["fastkmeans", "flash-kmeans"],
        choices=["fastkmeans", "flash-kmeans"],
        help="KMeans implementations to compare",
    )
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--batch_size", type=int, default=2000)
    parser.add_argument("--document_length", type=int, default=300)
    parser.add_argument("--index_folder", type=str, default="benchmark_indexes")
    parser.add_argument("--cache_dir", type=str, default="benchmark_cache")
    parser.add_argument(
        "--output", type=str, default="benchmark_kmeans.jsonl",
        help="JSONL file to append results to",
    )

    args = parser.parse_args()

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")

        documents, documents_embeddings = load_or_encode(
            dataset=dataset,
            model_name=args.model,
            batch_size=args.batch_size,
            document_length=args.document_length,
            cache_dir=args.cache_dir,
        )

        n_doc_tokens = sum(emb.shape[0] for emb in documents_embeddings)
        print(f"  {len(documents)} docs, {n_doc_tokens} tokens")

        for idx_type in args.indexes:
            for kmeans_impl in args.kmeans:
                index_name = f"kmeans_ablation_{dataset}_{idx_type}_{kmeans_impl}"

                print(f"\n  Building: {idx_type} + {kmeans_impl}...")

                if kmeans_impl == "flash-kmeans":
                    # Patch both fast_plaid and xtr_warp modules
                    patches = [
                        patch_kmeans_for_module("fast_plaid.search.fast_plaid"),
                        patch_kmeans_for_module("xtr_warp.search"),
                    ]
                    for p in patches:
                        p.start()

                try:
                    build_time = build_index_timed(
                        index_type=idx_type,
                        documents=documents,
                        documents_embeddings=documents_embeddings,
                        index_folder=args.index_folder,
                        index_name=index_name,
                        device=args.device,
                    )
                finally:
                    if kmeans_impl == "flash-kmeans":
                        for p in patches:
                            p.stop()

                row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": args.model,
                    "dataset": dataset,
                    "index_type": idx_type,
                    "kmeans_impl": kmeans_impl,
                    "n_documents": len(documents),
                    "n_doc_tokens": n_doc_tokens,
                    "build_time_s": round(build_time, 2),
                }
                all_results.append(row)
                append_jsonl(args.output, row)

                print(f"    Build time: {build_time:.2f}s")

    # Summary
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Dataset':<15} {'Index':<8} {'KMeans':<15} {'#Docs':<8} {'#Tokens':<10} {'Build(s)':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['dataset']:<15} "
            f"{r['index_type']:<8} "
            f"{r['kmeans_impl']:<15} "
            f"{r['n_documents']:<8} "
            f"{r['n_doc_tokens']:<10} "
            f"{r['build_time_s']:<10}"
        )

    print(f"\nResults appended to {args.output}")


if __name__ == "__main__":
    main()
