"""Single-config FIQA driver to A/B the centroid CAGRA fix toggles.

Reads cached FIQA embeddings (expects them already present), builds an
``maxIVF_cagra`` + ``cagra`` centroid index with itopk=256, and times each
phase. Toggles are read from env vars by the Rust layer:

- FASTPLAID_CAGRA_CENTROID_POSTBUILD_SYNC: ``0`` disables sync_stream after Index::build
- FASTPLAID_CAGRA_CENTROID_RETAIN_DATASET: ``0`` drops the host Array2 right after build
- FASTPLAID_CAGRA_CENTROID_WARMUP: ``0`` skips the in-build diagnostic warmup search

Usage::

  LD_LIBRARY_PATH=... uv run --no-sync python \
    examples/evaluation/_fiqa_cagra_ablation.py --label sync_only

The script doesn't set the env vars itself — set them in the launching shell.
"""

from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np

from pylate import evaluation, indexes, models, retrieve


def _load_npz_embeddings(npz_path: Path) -> list:
    with np.load(npz_path, allow_pickle=True) as z:
        raw: list = z["embeddings"].tolist()
    if not isinstance(raw, list):
        raise ValueError(f"bad embeddings list in {npz_path}")
    return raw


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--label", required=True, help="Tag for this run (printed in summary)")
    parser.add_argument("--itopk", type=int, default=256)
    parser.add_argument("--dataset", default="fiqa")
    args = parser.parse_args()

    cache_dir = Path("evaluation_cache/beir_embeddings")
    # Resolve cache files for fiqa.
    docs_meta = next(
        p for p in cache_dir.glob("documents-*.meta.json")
        if (
            __import__("json").loads(p.read_text(encoding="utf-8"))
            .get("config", {}).get("dataset_name") == args.dataset
        )
    )
    queries_meta = next(
        p for p in cache_dir.glob("queries-*.meta.json")
        if (
            __import__("json").loads(p.read_text(encoding="utf-8"))
            .get("config", {}).get("dataset_name") == args.dataset
        )
    )
    docs_npz = cache_dir / docs_meta.name.replace(".meta.json", ".npz")
    queries_npz = cache_dir / queries_meta.name.replace(".meta.json", ".npz")
    print(f"[ablation:{args.label}] doc embeddings: {docs_npz.name}")
    print(f"[ablation:{args.label}] query embeddings: {queries_npz.name}")

    documents_embeddings = _load_npz_embeddings(docs_npz)
    queries_embeddings = _load_npz_embeddings(queries_npz)

    # Need the original BEIR text to recover document_ids and qrels.
    documents, queries, qrels = evaluation.load_beir(
        dataset_name=args.dataset, split="test"
    )
    document_ids = [d["id"] for d in documents]
    assert len(documents) == len(documents_embeddings), \
        f"doc id/embedding mismatch: {len(documents)} vs {len(documents_embeddings)}"
    assert len(queries) == len(queries_embeddings), \
        f"query id/embedding mismatch: {len(queries)} vs {len(queries_embeddings)}"

    model_short = "GTE-ModernColBERT-v1"
    label = args.label
    itopk = args.itopk
    print(
        f"[ablation:{label}] env: POSTBUILD_SYNC={os.environ.get('FASTPLAID_CAGRA_CENTROID_POSTBUILD_SYNC', '<unset/default 1>')}"
        f" RETAIN_DATASET={os.environ.get('FASTPLAID_CAGRA_CENTROID_RETAIN_DATASET', '<unset/default 1>')}"
        f" WARMUP={os.environ.get('FASTPLAID_CAGRA_CENTROID_WARMUP', '<unset/default 1>')}"
    )

    t_init = time.perf_counter()
    index = indexes.PLAID(
        override=True,
        index_name=f"{args.dataset}_{model_short}_ablation_{label}",
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
    t_init_done = time.perf_counter()

    # ----- clustering / index build phase (maxIVF anchors + PLAID compression + centroid CAGRA) -----
    t_add0 = time.perf_counter()
    index.add_documents(
        documents_ids=document_ids,
        documents_embeddings=documents_embeddings,
    )
    t_add1 = time.perf_counter()
    add_wall_s = t_add1 - t_add0

    # ----- search phase -----
    t_ret0 = time.perf_counter()
    scores = retriever.retrieve(queries_embeddings=queries_embeddings, k=100)
    t_ret1 = time.perf_counter()
    ret_wall_s = t_ret1 - t_ret0

    for (qid, _q), q_scores in zip(queries.items(), scores):
        for s in list(q_scores):
            if s["id"] == qid:
                q_scores.remove(s)
    eval_scores = evaluation.evaluate(
        scores=scores,
        qrels=qrels,
        queries=list(queries.keys()),
        metrics=["map", "ndcg@10", "ndcg@100", "recall@10", "recall@100"],
    )

    print()
    print(f"========== ABLATION SUMMARY [{label}] ==========")
    print(f"  itopk_size                 : {itopk}")
    print(f"  index init wall            : {t_init_done - t_init:8.2f} s")
    print(f"  add_documents wall (cluster+compress+centroid_build): {add_wall_s:8.2f} s")
    print(f"  retrieve wall (search)     : {ret_wall_s:8.2f} s")
    print(f"  num_queries                : {len(queries)}")
    print(f"  retrieve avg per query     : {(ret_wall_s / max(1, len(queries))) * 1000:8.2f} ms")
    print(f"  eval                       : {eval_scores}")
    print(f"=================================================")


if __name__ == "__main__":
    main()
