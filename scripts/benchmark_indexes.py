"""Benchmark script comparing ScaNN, PLAID, and WARP indexes on BEIR datasets.

Collects evaluation metrics (NDCG, recall, MAP), queries per second (QPS),
index build time, index disk usage, and peak memory during search.

Results are appended as JSONL to the output file (one row per index+retrieval+dataset combo).
"""

from __future__ import annotations

import argparse
import datetime
import gc
import json
import os
import pickle
import time
import tracemalloc

import torch

from pylate import evaluation, indexes, models, retrieve

QUERY_LEN_MAP = {
    "quora": 32, "climate-fever": 64, "nq": 32, "msmarco": 32,
    "hotpotqa": 32, "nfcorpus": 32, "scifact": 48, "trec-covid": 48,
    "fiqa": 32, "arguana": 64, "scidocs": 48, "dbpedia-entity": 32,
    "webis-touche2020": 32, "fever": 32,
}


def get_dir_size_mb(path: str) -> float:
    """Return total size of a directory in MB."""
    total = 0
    for dirpath, _, filenames in os.walk(path):
        for f in filenames:
            fp = os.path.join(dirpath, f)
            if os.path.isfile(fp):
                total += os.path.getsize(fp)
    return total / (1024 * 1024)


def cache_path(cache_dir: str, dataset: str, model_short: str, name: str) -> str:
    return os.path.join(cache_dir, f"{dataset}_{model_short}_{name}.pkl")


def load_or_encode(
    dataset: str,
    model_name: str,
    batch_size: int,
    query_batch_size: int,
    document_length: int,
    cache_dir: str,
) -> tuple:
    """Load dataset and encode (or load cached) embeddings."""
    os.makedirs(cache_dir, exist_ok=True)
    model_short = model_name.split("/")[-1]

    # Dataset
    data_cp = cache_path(cache_dir, dataset, model_short, "data")
    if os.path.exists(data_cp):
        print(f"  Loading cached dataset ({dataset})...")
        with open(data_cp, "rb") as f:
            documents, queries, qrels = pickle.load(f)
    else:
        print(f"  Loading dataset from BEIR ({dataset})...")
        split = "dev" if "msmarco" in dataset else "test"
        documents, queries, qrels = evaluation.load_beir(dataset, split=split)
        with open(data_cp, "wb") as f:
            pickle.dump((documents, queries, qrels), f)

    # Document embeddings
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

    # Query embeddings
    query_cp = cache_path(cache_dir, dataset, model_short, "query_emb")
    if os.path.exists(query_cp):
        print(f"  Loading cached query embeddings...")
        with open(query_cp, "rb") as f:
            queries_embeddings = pickle.load(f)
    else:
        print(f"  Encoding {len(queries)} queries...")
        model = models.ColBERT(
            model_name_or_path=model_name,
            query_length=QUERY_LEN_MAP.get(dataset),
        )
        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=query_batch_size,
        )
        with open(query_cp, "wb") as f:
            pickle.dump(queries_embeddings, f)
        del model
        gc.collect()

    return documents, queries, qrels, documents_embeddings, queries_embeddings


def build_index(
    index_type: str,
    documents: list[dict],
    documents_embeddings: list,
    index_folder: str,
    index_name: str,
    device: str | None,
    store_embeddings: bool = True,
) -> tuple:
    """Build an index and return (index, build_time, disk_mb)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if index_type == "plaid":
        index = indexes.PLAID(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
        )
    elif index_type == "scann":
        index = indexes.ScaNN(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            store_embeddings=store_embeddings,
        )
    elif index_type == "warp":
        index = indexes.WARP(
            index_folder=index_folder,
            index_name=index_name,
            override=True,
            device=device,
        )
    else:
        raise ValueError(f"Unknown index type: {index_type}")

    doc_ids = [doc["id"] for doc in documents]
    build_start = time.perf_counter()
    index.add_documents(
        documents_ids=doc_ids,
        documents_embeddings=documents_embeddings,
    )
    build_time = time.perf_counter() - build_start

    index_path = os.path.join(index_folder, index_name)
    disk_mb = get_dir_size_mb(index_path)

    # For ScaNN, compute disk size without flattened_embeddings.npy
    # (only needed for ColBERT reranking, not XTR)
    flat_emb_path = os.path.join(index_path, "flattened_embeddings.npy")
    if os.path.isfile(flat_emb_path):
        flat_emb_mb = os.path.getsize(flat_emb_path) / (1024 * 1024)
    else:
        flat_emb_mb = 0.0
    disk_mb_index_only = disk_mb - flat_emb_mb

    return index, build_time, disk_mb, disk_mb_index_only


def benchmark_search(
    index,
    index_type: str,
    retrieval: str,
    queries_embeddings: list,
    query_ids: list[str],
    qrels: dict,
    k: int,
    k_token: int,
    device: str | None,
) -> dict:
    """Run search on a pre-built index and collect metrics."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    if index_type == "plaid":
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif index_type == "warp":
        retriever = retrieve.XTR(index=index, verbose=False)
        retrieve_kwargs = dict(queries_embeddings=queries_embeddings, k=k)
    elif retrieval == "xtr":
        retriever = retrieve.XTR(index=index, verbose=False)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu",
        )
    else:
        retriever = retrieve.ColBERT(index=index)
        retrieve_kwargs = dict(
            queries_embeddings=queries_embeddings, k=k,
            k_token=k_token, device=device or "cpu",
        )

    # Warm-up run
    _ = retriever.retrieve(**retrieve_kwargs)

    tracemalloc.start()
    search_start = time.perf_counter()
    scores = retriever.retrieve(**retrieve_kwargs)
    search_time = time.perf_counter() - search_start
    _, peak_mem_bytes = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    n_queries = len(queries_embeddings)
    qps = n_queries / search_time if search_time > 0 else float("inf")
    peak_mem_mb = peak_mem_bytes / (1024 * 1024)

    eval_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=query_ids,
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    return {
        "search_time_s": round(search_time, 4),
        "qps": round(qps, 2),
        "peak_search_mem_mb": round(peak_mem_mb, 2),
        "n_queries": n_queries,
        **{k: round(v, 4) for k, v in eval_scores.items()},
    }


def append_jsonl(path: str, row: dict) -> None:
    """Append a single JSON object as a line to the file."""
    with open(path, "a") as f:
        f.write(json.dumps(row) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Benchmark ScaNN, PLAID, and WARP indexes on BEIR datasets."
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=["nfcorpus"],
        help="BEIR dataset names (default: nfcorpus)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="lightonai/GTE-ModernColBERT-v1",
        help="Model name or path",
    )
    parser.add_argument(
        "--indexes",
        nargs="+",
        default=["warp", "plaid", "scann"],
        choices=["warp", "plaid", "scann"],
        help="Index types to benchmark",
    )
    parser.add_argument("--k", type=int, default=100, help="Top-k results to retrieve")
    parser.add_argument(
        "--k_token_colbert",
        type=int,
        default=4_000,
        help="Token-level candidates for ColBERT retrieval (default: 4000)",
    )
    parser.add_argument(
        "--k_token_xtr",
        type=int,
        default=10_000,
        help="Token-level candidates for XTR retrieval (default: 10000)",
    )
    parser.add_argument(
        "--batch_size", type=int, default=2000, help="Document encoding batch size"
    )
    parser.add_argument(
        "--query_batch_size", type=int, default=32, help="Query encoding batch size"
    )
    parser.add_argument("--device", type=str, default=None, help="Device (e.g. cuda, cpu)")
    parser.add_argument(
        "--document_length", type=int, default=300, help="Max document token length"
    )
    parser.add_argument(
        "--index_folder",
        type=str,
        default="benchmark_indexes",
        help="Folder for index storage",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="benchmark_cache",
        help="Folder for cached embeddings",
    )
    parser.add_argument(
        "--output",
        type=str,
        default="benchmark_indexes.jsonl",
        help="JSONL file to append results to (default: benchmark_indexes.jsonl)",
    )
    parser.add_argument(
        "--skip-retrieval",
        action="store_true",
        help="Only benchmark index building, skip search and evaluation.",
    )

    args = parser.parse_args()

    all_results = []

    for dataset in args.datasets:
        print(f"\n{'#'*60}")
        print(f"# Dataset: {dataset}")
        print(f"{'#'*60}")

        documents, queries, qrels, documents_embeddings, queries_embeddings = (
            load_or_encode(
                dataset=dataset,
                model_name=args.model,
                batch_size=args.batch_size,
                query_batch_size=args.query_batch_size,
                document_length=args.document_length,
                cache_dir=args.cache_dir,
            )
        )
        query_ids = list(queries.keys())
        n_doc_tokens = sum(emb.shape[0] for emb in documents_embeddings)

        for idx_type in args.indexes:
            if idx_type == "scann":
                retrieval_modes = ["colbert", "xtr"]
            elif idx_type == "plaid":
                retrieval_modes = ["colbert"]
            elif idx_type == "warp":
                retrieval_modes = ["xtr"]
            else:
                retrieval_modes = ["colbert"]

            index_name = f"bench_{dataset}_{idx_type}"

            print(f"\n{'='*60}")
            print(f"Building index: {idx_type} ({dataset})")
            print(f"{'='*60}")

            index, build_time, disk_mb, disk_mb_index_only = build_index(
                index_type=idx_type,
                documents=documents,
                documents_embeddings=documents_embeddings,
                index_folder=args.index_folder,
                index_name=index_name,
                device=args.device,
                store_embeddings=True,
            )

            print(f"  Build time: {build_time:.2f}s")
            print(f"  Disk usage: {disk_mb:.2f} MB (index only: {disk_mb_index_only:.2f} MB)")

            if args.skip_retrieval:
                row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": args.model,
                    "dataset": dataset,
                    "index_type": idx_type,
                    "retrieval": None,
                    "k": args.k,
                    "k_token": None,
                    "build_time_s": round(build_time, 2),
                    "disk_mb": round(disk_mb, 2),
                    "disk_mb_index_only": round(disk_mb_index_only, 2),
                    "n_documents": len(documents),
                    "n_doc_tokens": n_doc_tokens,
                }
                all_results.append(row)
                append_jsonl(args.output, row)
                continue

            for retrieval in retrieval_modes:
                print(f"\n  --- Searching: {idx_type} + {retrieval} retrieval ---")

                search_result = benchmark_search(
                    index=index,
                    index_type=idx_type,
                    retrieval=retrieval,
                    queries_embeddings=queries_embeddings,
                    query_ids=query_ids,
                    qrels=qrels,
                    k=args.k,
                    k_token=args.k_token_xtr if retrieval == "xtr" else args.k_token_colbert,
                    device=args.device,
                )

                k_token_used = args.k_token_xtr if retrieval == "xtr" else args.k_token_colbert
                row = {
                    "timestamp": datetime.datetime.now().isoformat(),
                    "model": args.model,
                    "dataset": dataset,
                    "index_type": idx_type,
                    "retrieval": retrieval,
                    "k": args.k,
                    "k_token": k_token_used,
                    "build_time_s": round(build_time, 2),
                    "disk_mb": round(disk_mb, 2),
                    "disk_mb_index_only": round(disk_mb_index_only, 2),
                    "n_documents": len(documents),
                    "n_doc_tokens": n_doc_tokens,
                    **search_result,
                }
                all_results.append(row)
                append_jsonl(args.output, row)

                print(f"  Search time: {row['search_time_s']}s ({row['qps']} QPS)")
                print(f"  Peak mem:    {row['peak_search_mem_mb']} MB")
                for metric in ["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"]:
                    print(f"  {metric}: {row.get(metric, 'N/A')}")

    # Summary table
    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    header = f"{'Dataset':<15} {'Index':<10} {'Retrieval':<10} {'Build(s)':<10} {'Disk(MB)':<10} {'QPS':<10} {'Mem(MB)':<10} {'NDCG@10':<10} {'R@100':<10}"
    print(header)
    print("-" * len(header))
    for r in all_results:
        print(
            f"{r['dataset']:<15} "
            f"{r['index_type']:<10} "
            f"{r['retrieval']:<10} "
            f"{r['build_time_s']:<10} "
            f"{r['disk_mb']:<10} "
            f"{r['qps']:<10} "
            f"{r['peak_search_mem_mb']:<10} "
            f"{r.get('ndcg@10', 'N/A'):<10} "
            f"{r.get('recall@100', 'N/A'):<10}"
        )

    print(f"\nResults appended to {args.output}")


if __name__ == "__main__":
    main()
