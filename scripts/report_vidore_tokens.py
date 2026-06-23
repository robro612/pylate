"""Report exact corpus token counts for encoded ViDoRe datasets.

Token count == the cached ``.doclens.npy`` the index stores, so this reads the
embeddings cache produced by ``benchmark_indexes.py`` (encode_docs stage) — no
re-encoding. Run after encoding the corpora into --cache_dir.

    python scripts/report_vidore_tokens.py --cache_dir /exp/rjha/pylate-pgc/embeddings_cache
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--cache_dir", default="/exp/rjha/pylate-pgc/embeddings_cache")
parser.add_argument("--model_slug", default="vidore_colqwen2.5-v0.2")
args = parser.parse_args()

cache = Path(args.cache_dir)
rows = []
for ds_dir in sorted(cache.glob("vidore_*")):
    docs_dir = ds_dir / args.model_slug
    doclens_files = sorted(docs_dir.glob("*/docs/*.doclens.npy"))
    if not doclens_files:
        continue
    doclens = np.concatenate([np.load(p) for p in doclens_files])
    n_docs = int(doclens.size)
    n_tokens = int(doclens.sum())
    rows.append((ds_dir.name.replace("vidore_", ""), n_docs, n_tokens, n_tokens / n_docs))

if not rows:
    print(f"No encoded ViDoRe datasets found under {cache} for model {args.model_slug}.")
    raise SystemExit(0)

w = max(len(r[0]) for r in rows)
print(f"{'dataset':<{w}}  {'images':>7}  {'tokens':>12}  {'tok/img':>8}")
for name, n, tok, tpi in sorted(rows, key=lambda r: -r[2]):
    print(f"{name:<{w}}  {n:>7,}  {tok:>12,}  {tpi:>8.1f}")

tot_n = sum(r[1] for r in rows)
tot_t = sum(r[2] for r in rows)
print(f"\n{'TOTAL':<{w}}  {tot_n:>7,}  {tot_t:>12,}  {tot_t / tot_n:>8.1f}  ({len(rows)} datasets)")
