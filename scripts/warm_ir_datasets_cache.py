"""Warm the ir_datasets cache for all eval datasets.

Downloads and iterates through docs, queries, and qrels sequentially to prevent
concurrent download races when the SLURM array jobs start.

Run once before submitting eval_models.sh:
    python scripts/warm_ir_datasets_cache.py
"""

import ir_datasets
from transformers import AutoTokenizer
from tqdm import tqdm

TOKENIZER = AutoTokenizer.from_pretrained("answerdotai/ModernBERT-base")

DATASETS = [
    # BEIR (sorted by corpus size)
    "beir/nfcorpus/test",        # 3.6K
    "beir/scifact/test",         # 5K
    "beir/arguana",         # 8.7K
    "beir/scidocs",              # 25K
    "beir/fiqa/test",            # 57K
    "beir/trec-covid",           # 171K
    "beir/webis-touche2020/v2",  # 382K
    "beir/quora/test",           # 523K
    "beir/nq",                   # 2.68M
    "beir/msmarco/dev",          # 8.84M
    # LoTTE test/search (sorted by corpus size)
    "lotte/lifestyle/test/search",   # 120K
    "lotte/writing/test/search",     # 200K
    "lotte/recreation/test/search",  # 500K
    "lotte/technology/test/search",  # 1.2M
    "lotte/science/test/search",     # 1.7M
    "lotte/pooled/test/search",      # 2.8M
]


def warm(dataset_id: str) -> None:
    print(f"\n=== {dataset_id} ===")
    dataset = ir_datasets.load(dataset_id)

    BATCH_SIZE = 2048
    doc_count = 0
    total_tokens = 0
    batch = []
    for doc in tqdm(dataset.docs_iter(), desc="  docs", unit="doc"):
        text = f"{doc.title}\n\n{doc.text}".strip() if hasattr(doc, "title") and doc.title else doc.text.strip()
        batch.append(text)
        doc_count += 1
        if len(batch) == BATCH_SIZE:
            enc = TOKENIZER(batch, add_special_tokens=False)
            total_tokens += sum(len(ids) for ids in enc["input_ids"])
            batch = []
    if batch:
        enc = TOKENIZER(batch, add_special_tokens=False)
        total_tokens += sum(len(ids) for ids in enc["input_ids"])
    avg_len = total_tokens / doc_count if doc_count else 0

    query_count = sum(1 for _ in tqdm(dataset.queries_iter(), desc="  queries", unit="query"))

    qrel_count = 0
    if dataset.has_qrels():
        qrel_count = sum(1 for _ in tqdm(dataset.qrels_iter(), desc="  qrels", unit="qrel"))

    print(f"  -> {doc_count} docs, {query_count} queries, {qrel_count} qrels, avg doc len {avg_len:.0f} tokens")


if __name__ == "__main__":
    for dataset_id in DATASETS:
        warm(dataset_id)
    print("\nDone. ir_datasets cache is warm.")
