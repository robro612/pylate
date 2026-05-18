"""Evaluation script for BEIR datasets using PLAID, WARP, or Tachiom index,
paired with an XTR-trained model.

All three are end-to-end indexes, so the `retrieve.XTR` wrapper short-circuits
to the index's own scoring rather than running a separate XTR scoring pass on
top of token-level hits.

Document embeddings are cached to disk under --cache_dir so subsequent runs
skip encoding entirely. Pass --no_cache to force re-encoding.

For the ColBERT + PLAID pipeline with a standard ColBERT model, see
`beir_dataset.py`.
"""

from __future__ import annotations

import argparse

from pylate import evaluation, indexes, models, retrieve
from pylate.utils import cache_exists, encode_and_cache, get_cache_dir, load_cached

if __name__ == "__main__":
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

    parser = argparse.ArgumentParser(
        description="BEIR evaluation with XTR model and a choice of index"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="BEIR dataset to evaluate on (default: nfcorpus)",
    )
    parser.add_argument(
        "--index",
        type=str,
        default="warp",
        choices=["plaid", "warp", "tachiom"],
        help="Index backend to use (default: warp)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="emb_cache",
        help="Root directory for embedding shards (default: emb_cache)",
    )
    parser.add_argument(
        "--no_cache",
        action="store_true",
        help="Re-encode documents even if a cache exists",
    )
    parser.add_argument(
        "--shard_size",
        type=int,
        default=500_000,
        help="Max documents per embedding shard (default: 500000)",
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    model_name = "robro612/ModernBERT-XTR"
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
    )

    if "cqadupstack" in dataset_name:
        from beir import util

        util.download_and_unzip(
            url="https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/cqadupstack.zip",
            out_dir="./evaluation_datasets/",
        )
        documents, queries, qrels = evaluation.load_custom_dataset(
            f"evaluation_datasets/{dataset_name}",
            split="test",
        )
        dataset_name = dataset_name.replace("/", "_")
    else:
        documents, queries, qrels = evaluation.load_beir(
            dataset_name=dataset_name,
            split="dev" if "msmarco" in dataset_name else "test",
        )

    index_name = f"{dataset_name}_{model_name.split('/')[-1]}"
    if args.index == "plaid":
        index = indexes.PLAID(override=True, index_name=index_name)
    elif args.index == "warp":
        index = indexes.WARP(override=True, index_name=index_name)
    else:
        index = indexes.Tachiom(override=True, index_name=index_name)

    retriever = retrieve.XTR(index=index)

    # ── Document embeddings (cached) ─────────────────────────────────────────
    # Always cache token IDs — negligible extra cost, avoids re-encoding when
    # switching between index types.
    doc_cache_dir = get_cache_dir(args.cache_dir, dataset_name, model_name)

    if not args.no_cache and cache_exists(doc_cache_dir):
        print(f"Loading document embeddings from cache: {doc_cache_dir}")
        _, documents_embeddings, documents_token_ids = load_cached(doc_cache_dir)
    else:
        print(f"Encoding documents and writing cache to: {doc_cache_dir}")
        documents_embeddings, documents_token_ids = encode_and_cache(
            model=model,
            sentences=[d["text"] for d in documents],
            doc_ids=[d["id"] for d in documents],
            cache_dir=doc_cache_dir,
            shard_size=args.shard_size,
            return_token_ids=True,
            batch_size=2000,
            is_query=False,
            show_progress_bar=True,
        )

    index.add_documents(
        documents_ids=[document["id"] for document in documents],
        documents_embeddings=documents_embeddings,
        **({"documents_token_ids": documents_token_ids} if documents_token_ids is not None else {}),
    )

    # ── Query embeddings (not cached — queries are fast) ──────────────────────
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(evaluation_scores)
