"""Evaluate ColBERT checkpoints on full BEIR datasets *under their compression*.

For each model, the document/query token embeddings are passed through the same
``Compressor`` the model was trained for (read from ``<model>/compression.json``;
absent -> identity / full precision), then scored with **exact brute-force MaxSim**
over the whole corpus -- so the baseline is true full precision and each STE
variant is measured exactly under its own quantization/pooling, with no ANN or
PLAID approximation in between.

Usage
-----
    python examples/evaluation/eval_compressed_beir.py \
        --models /mnt3/.../output/mbb-baseline/final,/mnt3/.../output/mbb-shbq/final \
        --datasets nfcorpus,scifact,fiqa
"""

from __future__ import annotations

import argparse
import json
import os

import torch

from pylate import evaluation, models
from pylate.models import build_compressor
from pylate.scores import colbert_scores

QUERY_LEN = {"nfcorpus": 32, "scifact": 48, "fiqa": 32}


def load_beir_hf(dataset: str, split: str = "test"):
    """Load a BEIR dataset from the HF hub (BeIR/<name> + BeIR/<name>-qrels).

    Mirrors pylate.evaluation.load_beir's output — documents [{id,text}],
    queries {id:text}, qrels {qid:{docid:rel}} — but reads the cached HF copies
    instead of the (here-unreachable) BEIR download server.
    """
    from datasets import load_dataset

    corpus = load_dataset(f"BeIR/{dataset}", "corpus")["corpus"]
    queries_ds = load_dataset(f"BeIR/{dataset}", "queries")["queries"]
    qrels_ds = load_dataset(f"BeIR/{dataset}-qrels")[split]

    qid_key = "query-id" if "query-id" in qrels_ds.column_names else "query_id"
    did_key = "corpus-id" if "corpus-id" in qrels_ds.column_names else "corpus_id"
    qrels: dict[str, dict[str, int]] = {}
    for r in qrels_ds:
        if int(r["score"]) <= 0:
            continue
        qrels.setdefault(str(r[qid_key]), {})[str(r[did_key])] = int(r["score"])

    queries = {
        str(q["_id"]): q["text"] for q in queries_ds if str(q["_id"]) in qrels
    }
    documents = [
        {"id": str(d["_id"]), "text": (f"{d.get('title', '')} {d['text']}").strip()}
        for d in corpus
    ]
    return documents, queries, qrels


def load_compressors(spec_path: str | None):
    """Return (query_compressor, document_compressor) from a compression.json path.

    ``None`` / missing -> identity (full precision). Passing a *different* model's
    spec applies that compression to this model -- e.g. compressing a
    full-precision baseline at eval time, to test whether STE training was needed.
    """
    if spec_path is None or not os.path.exists(spec_path):
        return build_compressor(None), build_compressor(None)
    spec = json.load(open(spec_path))
    return build_compressor(spec.get("query")), build_compressor(spec.get("document"))


def pad(embeddings: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
    lengths = [e.shape[0] for e in embeddings]
    max_len, hidden = max(lengths), embeddings[0].shape[1]
    out = embeddings[0].new_zeros(len(embeddings), max_len, hidden)
    mask = torch.zeros(len(embeddings), max_len, dtype=torch.bool, device=out.device)
    for i, e in enumerate(embeddings):
        out[i, : e.shape[0]] = e
        mask[i, : e.shape[0]] = True
    return out, mask


@torch.no_grad()
def brute_force_scores(
    query_embeddings, document_embeddings, k, device, q_chunk=64, doc_chunk=1024
):
    """Exact MaxSim (Q, D) scores; return top-k values+indices.

    Double-chunked over queries and documents so the torch backend's transient
    ``(q_chunk, doc_chunk, Qt, Dt)`` tensor stays bounded, with the (small) score
    matrix accumulated on CPU. Documents live on CPU; chunks move to GPU on demand.
    """
    n_q, n_d = len(query_embeddings), len(document_embeddings)
    scores = torch.zeros(n_q, n_d, dtype=torch.float32)  # CPU
    for d0 in range(0, n_d, doc_chunk):
        d_pad, d_mask = pad([d.to(device) for d in document_embeddings[d0 : d0 + doc_chunk]])
        for q0 in range(0, n_q, q_chunk):
            q_pad, q_mask = pad([q.to(device) for q in query_embeddings[q0 : q0 + q_chunk]])
            s = colbert_scores(q_pad, d_pad, queries_mask=q_mask, documents_mask=d_mask)
            scores[q0 : q0 + q_pad.shape[0], d0 : d0 + d_pad.shape[0]] = s.cpu()
    top_values, top_indices = scores.topk(min(k, n_d), dim=1)
    return top_values, top_indices


def evaluate_model_on_dataset(model_dir, dataset, k, device, compression_spec=None):
    spec_path = compression_spec or os.path.join(model_dir, "compression.json")
    query_compressor, document_compressor = load_compressors(spec_path)
    model = models.ColBERT(
        model_name_or_path=model_dir,
        document_length=300,
        query_length=QUERY_LEN.get(dataset, 32),
    ).to(device)
    model.eval()

    documents, queries, qrels = load_beir_hf(dataset, split="test")
    doc_ids = [d["id"] for d in documents]
    query_ids = list(queries.keys())

    doc_embeddings = model.encode(
        [d["text"] for d in documents], is_query=False, batch_size=256,
        convert_to_tensor=True, show_progress_bar=True,
    )
    query_embeddings = model.encode(
        list(queries.values()), is_query=True, batch_size=64,
        convert_to_tensor=True, show_progress_bar=True,
    )
    # Apply the model's own compression (identity for the full-precision baseline),
    # then move the corpus to CPU so large collections don't sit on the GPU.
    doc_embeddings = [d.cpu() for d in document_compressor.transform(doc_embeddings)]
    query_embeddings = [q.cpu() for q in query_compressor.transform(query_embeddings)]

    top_values, top_indices = brute_force_scores(query_embeddings, doc_embeddings, k, device)

    scores = []
    for qi, query_id in enumerate(query_ids):
        row = [
            {"id": doc_ids[j], "score": float(v)}
            for v, j in zip(top_values[qi].tolist(), top_indices[qi].tolist())
            if doc_ids[j] != query_id  # drop self-hit (needed for fiqa)
        ]
        scores.append(row)

    return evaluation.evaluate(
        scores=scores, qrels=qrels, queries=query_ids,
        metrics=["ndcg@10", "recall@10", "recall@100"],
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--models", required=True, help="Comma-separated model dirs.")
    parser.add_argument("--datasets", default="nfcorpus,scifact,fiqa")
    parser.add_argument("--k", type=int, default=100)
    parser.add_argument(
        "--compression_spec",
        default=None,
        help="Override the compression applied at eval with this compression.json "
        "(e.g. apply an STE variant's compressor to the full-precision baseline).",
    )
    parser.add_argument("--label", default=None, help="Override the printed model name.")
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    results = {}
    for model_dir in args.models.split(","):
        name = args.label or os.path.basename(os.path.dirname(model_dir.rstrip("/")))
        results[name] = {}
        for dataset in args.datasets.split(","):
            metrics = evaluate_model_on_dataset(
                model_dir, dataset, args.k, device, compression_spec=args.compression_spec
            )
            results[name][dataset] = metrics
            print(f"[{name}] {dataset}: {metrics}", flush=True)

    print("\n=== SUMMARY: ndcg@10 (retrieval under each model's compression) ===")
    datasets = args.datasets.split(",")
    header = f"{'model':<24} " + " ".join(f"{d:>10}" for d in datasets)
    print(header)
    for name, per in results.items():
        row = f"{name:<24} " + " ".join(f"{per[d]['ndcg@10']:>10.4f}" for d in datasets)
        print(row)


if __name__ == "__main__":
    main()
