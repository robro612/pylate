#!/usr/bin/env python
"""Postings-per-centroid distribution for the built LoTTE tachiom indexes.

Reads each centroid's inverted-list length (TachiomIndex._index.inverted_list_lengths,
added as a Rust getter) and reports the distribution that drives coarse coverage:
finer clustering => more, shorter lists => fewer docs touched at a fixed k_centroids.

    UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync \
        python scripts/analysis/postings_per_centroid.py
"""

from __future__ import annotations

import numpy as np

from pylate.indexes import TachiomIndex

CASES = [
    ("TAC", "bench_lightonai_LateOn_lotte_pooled_dev_search_tachiom_tac_m32", "tac"),
    ("PGC", "bench_lightonai_LateOn_lotte_pooled_dev_search_tachiom_pgc_m32", "pgc"),
]
PCTS = [50, 75, 90, 99]


def main() -> None:
    rows = []
    for label, name, clustering in CASES:
        ix = TachiomIndex(
            index_folder="indexes", index_name=name, clustering=clustering,
        )
        lengths = np.asarray(ix._index.inverted_list_lengths, dtype=np.int64)
        n = len(lengths)
        total = int(lengths.sum())
        pct = {p: float(np.percentile(lengths, p)) for p in PCTS}
        rows.append((label, n, total, lengths, pct))
        print(
            f"{label}: n_centroids={n:,}  total_postings={total:,}  "
            f"mean={lengths.mean():.1f}  "
            + "  ".join(f"p{p}={pct[p]:.0f}" for p in PCTS)
            + f"  max={lengths.max():,}"
        )
        print(
            f"      empty={100 * (lengths == 0).mean():.2f}%  "
            f"singleton={100 * (lengths == 1).mean():.2f}%  "
            f"postings/doc(avg)={total / ix._index.len:.2f}"
        )

    if len(rows) == 2:
        (_, n0, t0, _, p0), (_, n1, t1, _, p1) = rows
        print(
            f"\nPGC/TAC ratios: centroids={n1 / n0:.2f}x  total_postings={t1 / t0:.2f}x  "
            f"mean_list_len={ (t1 / n1) / (t0 / n0):.2f}x  p50={p1[50] / max(p0[50],1):.2f}x"
        )


if __name__ == "__main__":
    main()
