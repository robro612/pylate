"""Evaluation script for BEIR datasets using a WARP index, paired with an
XTR-trained model.

WARP is an end-to-end multi-vector retrieval engine (like PLAID), so the
`retrieve.XTR` wrapper short-circuits to the index's own scoring rather than
running a separate XTR scoring pass on top of token-level hits.

For the ColBERT + PLAID pipeline, see `beir_dataset.py`.

Multi-GPU usage (encode across 4 GPUs, shard index across all 4):
    python beir_dataset_xtr.py --dataset_name trec-covid --n_gpu 4

Build-from-disk (encode shard-by-shard, only one shard in RAM at a time):
    python beir_dataset_xtr.py \\
        --dataset_name trec-covid --n_gpu 4 --embeddings_dir /tmp/embs/trec-covid
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from pylate import evaluation, indexes, models, retrieve

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

    parser = argparse.ArgumentParser(description="BEIR evaluation with WARP + XTR")
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="nfcorpus",
        help="BEIR dataset name (default: nfcorpus)",
    )
    parser.add_argument(
        "--n_gpu",
        type=int,
        default=1,
        help="Number of GPUs to use for encoding and sharded index loading (default: 1)",
    )
    parser.add_argument(
        "--embeddings_dir",
        type=str,
        default=None,
        help=(
            "If set, save document embeddings to this directory and build the "
            "index from disk rather than from in-memory tensors.  Useful for "
            "large corpora where keeping all embeddings in RAM is impractical."
        ),
    )
    args = parser.parse_args()

    dataset_name = args.dataset_name
    n_gpu = args.n_gpu
    embeddings_dir = args.embeddings_dir

    model_name = "robro612/ModernBERT-XTR"
    # When using multi-GPU encoding, load on CPU so the main process never
    # creates a CUDA context before workers are spawned.  If the main process
    # holds a CUDA context on e.g. cuda:0, the spawned worker assigned to
    # cuda:0 hits "device busy or unavailable" on exclusive-mode GPUs.
    model_device = "cpu" if n_gpu > 1 else None
    model = models.ColBERT(
        model_name_or_path=model_name,
        document_length=300,
        query_length=query_len.get(dataset_name),
        device=model_device,
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

    # Resolve device list for sharded index loading
    if n_gpu > 1:
        device_list = [f"cuda:{i}" for i in range(n_gpu)]
    else:
        import torch

        device_list = "cuda" if torch.cuda.is_available() else "cpu"

    index = indexes.WARP(
        override=True,
        index_name=f"{dataset_name}_{model_name.split('/')[-1]}",
        device=device_list,
    )

    retriever = retrieve.XTR(index=index)

    doc_sentences = [document["text"] for document in documents]
    doc_ids = [document["id"] for document in documents]
    print(f"[1/4] Loaded {len(doc_sentences)} documents, {len(queries)} queries")

    if embeddings_dir is not None:
        os.makedirs(embeddings_dir, exist_ok=True)
        existing_shards = sorted(
            f
            for f in os.listdir(embeddings_dir)
            if f.endswith(".npy") and not f.endswith(".doclens.npy")
        )

        if existing_shards:
            print(
                f"[2/4] Found {len(existing_shards)} existing shard(s) in {embeddings_dir}, skipping encode"
            )
        else:
            n_shards = n_gpu
            n_docs = len(doc_sentences)
            docs_per_shard = max(1, (n_docs + n_shards - 1) // n_shards)

            if n_gpu > 1:
                os.environ["TORCHDYNAMO_DISABLE"] = "1"
                print(f"[2/4] Starting {n_gpu}-GPU encode pool")
                pool = model.start_multi_process_pool(
                    target_devices=[f"cuda:{i}" for i in range(n_gpu)]
                )
            try:
                for shard_idx in range(n_shards):
                    start = shard_idx * docs_per_shard
                    end = min(start + docs_per_shard, n_docs)
                    if start >= n_docs:
                        break
                    print(
                        f"[2/4] Encoding shard {shard_idx + 1}/{n_shards} (docs {start}–{end - 1})"
                    )
                    if n_gpu > 1:
                        shard_embeddings = model.encode_multi_process(
                            sentences=doc_sentences[start:end],
                            pool=pool,
                            batch_size=2000,
                            is_query=False,
                        )
                    else:
                        shard_embeddings = model.encode(
                            sentences=doc_sentences[start:end],
                            batch_size=2000,
                            is_query=False,
                            show_progress_bar=True,
                        )
                    shard_path = os.path.join(embeddings_dir, f"embeddings_{shard_idx}")
                    print(
                        f"[2/4] Saving shard {shard_idx + 1}/{n_shards} → {shard_path}.npy"
                    )
                    np.save(
                        f"{shard_path}.npy", np.concatenate(shard_embeddings, axis=0)
                    )
                    np.save(
                        f"{shard_path}.doclens.npy",
                        np.array(
                            [e.shape[0] for e in shard_embeddings], dtype=np.int32
                        ),
                    )
                    del shard_embeddings
            finally:
                if n_gpu > 1:
                    model.stop_multi_process_pool(pool)

        print(f"[3/4] Building WARP index from {embeddings_dir}")
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=embeddings_dir,
        )
    else:
        if n_gpu > 1:
            os.environ["TORCHDYNAMO_DISABLE"] = "1"
            print(f"[2/4] Encoding {len(doc_sentences)} documents across {n_gpu} GPUs")
            pool = model.start_multi_process_pool(
                target_devices=[f"cuda:{i}" for i in range(n_gpu)]
            )
            documents_embeddings = model.encode_multi_process(
                sentences=doc_sentences,
                pool=pool,
                batch_size=2000,
                is_query=False,
            )
            model.stop_multi_process_pool(pool)
        else:
            print(f"[2/4] Encoding {len(doc_sentences)} documents on 1 GPU")
            documents_embeddings = model.encode(
                sentences=doc_sentences,
                batch_size=2000,
                is_query=False,
                show_progress_bar=True,
            )
        print("[3/4] Building WARP index (in-memory)")
        index.add_documents(
            documents_ids=doc_ids,
            documents_embeddings=documents_embeddings,
        )

    if n_gpu > 1:
        model.to("cuda:0")

    print(f"[4/4] Encoding {len(queries)} queries and searching (k=100)")
    queries_embeddings = model.encode(
        sentences=list(queries.values()),
        is_query=True,
        show_progress_bar=True,
        batch_size=32,
    )

    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)

    # Remove query_id from scores, needed for FiQA dataset
    for (query_id, _query), query_scores in zip(queries.items(), scores):
        for score in query_scores:
            if score["id"] == query_id:
                query_scores.remove(score)

    evaluation_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print(f"Results ({dataset_name}):", evaluation_scores)
