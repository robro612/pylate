"""Estimate total token cost per ViDoRe v3 corpus (vs a reference like LoTTE).

Streams a sample of images per corpus, measures real tokens/image via
model.preprocess (no forward), and scales by the known corpus size."""

from __future__ import annotations

from datasets import load_dataset

from pylate import models
from pylate.evaluation.vidore_evaluator import VIDORE_V3_DATASETS

SAMPLE = 48
LOTTE_POOLED_TOKENS = 279_230_440

# corpus sizes from the HF size API (images per v3 set)
SIZES = {
    "finance": 2942, "financefr": 2384, "hr": 1110, "industrial": 5244,
    "pharmaceuticals": 2313, "computerscience": 1360, "energy": 2225, "physics": 1674,
}

model = models.ColBERT(model_name_or_path="vidore/colqwen2.5-v0.2", device="cuda")

rows = []
for short, repo in VIDORE_V3_DATASETS.items():
    ds = load_dataset(repo, "corpus", split="test", streaming=True)
    imgs = []
    for row in ds:
        imgs.append(row["image"].convert("RGB"))
        if len(imgs) >= SAMPLE:
            break
    feats = model.preprocess(inputs=imgs, is_query=False)
    tpi = float(feats["attention_mask"].sum().item()) / len(imgs)
    n = SIZES[short]
    total = tpi * n
    rows.append((short, n, tpi, total))
    print(f"{short:16s} images={n:>5d} tok/img={tpi:7.1f} est_tokens={total:>12,.0f}", flush=True)

print(f"\n{'corpus':16s} {'images':>6} {'tok/img':>8} {'est_tokens':>13} {'vs LoTTE pooled':>16}")
for short, n, tpi, total in sorted(rows, key=lambda r: -r[3]):
    print(f"{short:16s} {n:>6d} {tpi:>8.1f} {total:>13,.0f} {total / LOTTE_POOLED_TOKENS:>15.1%}")
print(f"\nLoTTE pooled reference: {LOTTE_POOLED_TOKENS:,} tokens")
