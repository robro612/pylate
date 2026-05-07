"""Evaluation script for BEIR datasets with a PLAID index."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from pylate import evaluation, indexes, models, retrieve


def _stable_sha256_json(data: dict) -> str:
    payload = json.dumps(
        data, sort_keys=True, separators=(",", ":"), ensure_ascii=False
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _stable_sha256_lines(lines: list[str]) -> str:
    h = hashlib.sha256()
    for line in lines:
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()


def _load_embeddings_from_cache(
    *,
    cache_dir: Path,
    kind: str,
    config: dict,
    expected_ids: list[str],
):
    # Note: mkdir here so the directory exists even if cache misses.
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _stable_sha256_json(config)
    data_path = cache_dir / f"{kind}-{key}.npz"
    meta_path = cache_dir / f"{kind}-{key}.meta.json"

    if not (data_path.exists() and meta_path.exists()):
        return None

    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    except Exception:
        return None

    if meta.get("ids_sha256") != _stable_sha256_lines(expected_ids):
        return None

    try:
        with np.load(data_path, allow_pickle=True) as z:
            embeddings = z["embeddings"].tolist()
    except Exception:
        return None

    if not isinstance(embeddings, list) or len(embeddings) != len(expected_ids):
        return None

    return embeddings


def _save_embeddings_to_cache(
    *,
    cache_dir: Path,
    kind: str,
    config: dict,
    ids: list[str],
    embeddings,
):
    cache_dir.mkdir(parents=True, exist_ok=True)
    key = _stable_sha256_json(config)
    data_path = cache_dir / f"{kind}-{key}.npz"
    meta_path = cache_dir / f"{kind}-{key}.meta.json"

    meta = {
        "kind": kind,
        "config": config,
        "num_items": len(ids),
        "ids_sha256": _stable_sha256_lines(ids),
    }

    np.savez_compressed(data_path, embeddings=np.array(embeddings, dtype=object))
    meta_path.write_text(
        json.dumps(meta, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )


if __name__ == "__main__":
    import numpy as np

    query_len = {
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
        "cqadupstack/android": 32,
        "cqadupstack/english": 32,
        "cqadupstack/gaming": 32,
        "cqadupstack/gis": 32,
        "cqadupstack/mathematica": 32,
        "cqadupstack/physics": 32,
        "cqadupstack/programmers": 32,
        "cqadupstack/stats": 32,
        "cqadupstack/tex": 32,
        "cqadupstack/unix": 32,
        "cqadupstack/webmasters": 32,
        "cqadupstack/wordpress": 32,
    }

    # Parse dataset_name from command line arguments
    parser = argparse.ArgumentParser(description="Dataset name")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="Name of the dataset to evaluate on (default: 'fiqa')",
    )
    parser.add_argument(
        "--sweep_index_configs",
        action="store_true",
        help="If set, encode once and evaluate multiple index configs (dense + several cagra variants).",
    )
    parser.add_argument(
        "--embeddings_cache_dir",
        type=str,
        default="./evaluation_cache/beir_embeddings",
        help="Directory to cache computed query/document embeddings.",
    )
    parser.add_argument(
        "--no_embeddings_cache",
        action="store_true",
        help="If set, disable embedding caching (always recompute).",
    )
    parser.add_argument(
        "--overwrite_embeddings_cache",
        action="store_true",
        help="If set, recompute and overwrite any existing cached embeddings.",
    )
    args = parser.parse_args()
    dataset_name = args.dataset_name
    model_name = "lightonai/GTE-ModernColBERT-v1"
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )
    model.compile()

    if "cqadupstack" in dataset_name:
        # Download dataset if not already downloaded
        from beir import util

        data_path = util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        split = "test"
        dataset_name = dataset_name.replace("/", "_")
    else:
        split = "dev" if "msmarco" in dataset_name else "test"
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split=split,
        )

    cache_dir = Path(args.embeddings_cache_dir)
    use_cache = (not args.no_embeddings_cache) and (not args.overwrite_embeddings_cache)

    document_ids = [document["id"] for document in documents]
    query_ids = list(queries.keys())

    common_cfg = {
        "dataset_name": dataset_name,
        "split": split,
        "model_name": model_name,
        "model_query_length": getattr(model, "query_length", None),
        "model_document_length": getattr(model, "document_length", None),
        "model_do_query_expansion": getattr(model, "do_query_expansion", None),
        "model_attend_to_expansion_tokens": getattr(
            model, "attend_to_expansion_tokens", None
        ),
    }

    documents_cfg = {
        **common_cfg,
        "kind": "documents",
        "batch_size": 6000,
        "is_query": False,
        "normalize_embeddings": True,
        "pool_factor": 1,
        "protected_tokens": 1,
    }
    queries_cfg = {
        **common_cfg,
        "kind": "queries",
        "batch_size": 32,
        "is_query": True,
        "normalize_embeddings": True,
        "pool_factor": 1,
        "protected_tokens": 1,
    }

    documents_embeddings = (
        _load_embeddings_from_cache(
            cache_dir=cache_dir,
            kind="documents",
            config=documents_cfg,
            expected_ids=document_ids,
        )
        if use_cache
        else None
    )
    if documents_embeddings is None:
        documents_embeddings = model.encode(
            sentences=[document["text"] for document in documents],
            batch_size=6000,
            is_query=False,
            show_progress_bar=True,
        )
        if not args.no_embeddings_cache:
            _save_embeddings_to_cache(
                cache_dir=cache_dir,
                kind="documents",
                config=documents_cfg,
                ids=document_ids,
                embeddings=documents_embeddings,
            )

    queries_embeddings = (
        _load_embeddings_from_cache(
            cache_dir=cache_dir,
            kind="queries",
            config=queries_cfg,
            expected_ids=query_ids,
        )
        if use_cache
        else None
    )
    if queries_embeddings is None:
        queries_embeddings = model.encode(
            sentences=list(queries.values()),
            is_query=True,
            show_progress_bar=True,
            batch_size=32,
        )
        if not args.no_embeddings_cache:
            _save_embeddings_to_cache(
                cache_dir=cache_dir,
                kind="queries",
                config=queries_cfg,
                ids=query_ids,
                embeddings=queries_embeddings,
            )

    def _run_one(label: str, backend: str, anchor_params=None, centroid_index_params=None):
        # Backward-compatible wrapper for older combined config.
        index = indexes.PLAID(
            override=False,
            index_name=f"{dataset_name}_{model_name.split('/')[-1]}_{label}",
            index_anchor_backend=backend,
            index_anchor_params=anchor_params,
            index_centroid_index_params=centroid_index_params,
        )
        
        retriever = retrieve.ColBERT(index=index)
        
        scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

        # Remove query_id from scores, needed for FiQA dataset
        for (query_id, _query), query_scores in zip(queries.items(), scores):
            for score in list(query_scores):
                if score["id"] == query_id:
                    query_scores.remove(score)

        evaluation_scores = evaluation.evaluate(
            scores=scores,
            qrels=qrels,
            queries=list(queries.keys()),
            metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
        )
        print(f"\n===== {label} =====")
        print(evaluation_scores)

    if args.sweep_index_configs:
        # Baseline: K-means anchors + dense probing
        print(f"Skipping the baseline")
        # _run_one(label="dense_default", backend="dense")

        # maxIVF_cagra anchors + CAGRA probing
        # NOTE: cuVS CAGRA single-CTA search used during maxIVF anchor assignment
        # enforces itopk_size <= 512.
        for itopk in (128, 256): #, 512):
            index = indexes.PLAID(
                override=True,
                index_name=f"{dataset_name}_{model_name.split('/')[-1]}_maxivf_cagra_itopk{itopk}",
                anchor_method="maxIVF_cagra",
                n_ivf_probe=1,
                anchor_params={
                    "n_iter": 3,
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                    "itopk_size": itopk,
                    "token_batch_size": 2 ** 17,
                    "normalize_anchors": True,
                },
                centroid_index="cagra",
                centroid_index_params={
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                    "itopk_size": itopk,
                },
            )
            retriever = retrieve.ColBERT(index=index)
            index.add_documents(
                documents_ids=[document["id"] for document in documents],
                documents_embeddings=documents_embeddings,
            )
            scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

            for (query_id, _query), query_scores in zip(queries.items(), scores):
                for score in list(query_scores):
                    if score["id"] == query_id:
                        query_scores.remove(score)

            evaluation_scores = evaluation.evaluate(
                scores=scores,
                qrels=qrels,
                queries=list(queries.keys()),
                metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
            )
            print(f"\n===== maxivf_cagra_itopk{itopk} =====")
            print(evaluation_scores)


        # K-means anchors + CAGRA probing (ablation: routing only)
        for itopk in (128, 256): #, 512, 1024):
            label = f"kmeans_cagra_itopk{itopk}"
            index = indexes.PLAID(
                override=True,
                index_name=f"{dataset_name}_{model_name.split('/')[-1]}_{label}",
                anchor_method="kmeans",
                centroid_index="cagra",
                centroid_index_params={
                    "graph_degree": 32,
                    "intermediate_graph_degree": 64,
                    "itopk_size": itopk,
                },
            )
            retriever = retrieve.ColBERT(index=index)
            index.add_documents(
                documents_ids=[document["id"] for document in documents],
                documents_embeddings=documents_embeddings,
            )
            scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

            for (query_id, _query), query_scores in zip(queries.items(), scores):
                for score in list(query_scores):
                    if score["id"] == query_id:
                        query_scores.remove(score)

            evaluation_scores = evaluation.evaluate(
                scores=scores,
                qrels=qrels,
                queries=list(queries.keys()),
                metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
            )
            print(f"\n===== maxivf_cagra_itopk{itopk} =====")
            print(evaluation_scores)
    else:
        # Default single-run behavior (baseline dense)
        _run_one(label="dense_default", backend="dense")
