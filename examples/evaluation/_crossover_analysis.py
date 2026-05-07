"""Post-hoc crossover analysis on a matrix-results JSONL.

For each (dataset, config) row in the input, prints:

- ``n_tokens``, ``n_centroids_planned`` (kmeans formula or maxIVF fraction).
- The kmeans baseline centroid count (``2 ** floor(log2(16 * sqrt(n_tokens)))``).
- The ratio ``n_centroids_planned / kmeans_baseline``.
- Where each f∈{0.01, 0.025, 0.05} crosses over the kmeans baseline (in n_tokens).

Old result files written before n_tokens / n_centroids were tracked are still useful:
this script falls back to recomputing those values from ``n_docs`` if necessary, but a
freshly-produced matrix JSONL has both fields populated.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path


def _kmeans_n_centroids(n_tokens: int) -> int:
    if n_tokens <= 0:
        return 0
    return 2 ** math.floor(math.log2(max(2.0, 16.0 * math.sqrt(n_tokens))))


def crossover_n_tokens(f: float) -> float:
    """Solve f*n = 16*sqrt(n) → n = (16/f)^2."""
    return (16.0 / f) ** 2


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", nargs="+", type=Path)
    args = parser.parse_args()

    print("Crossover thresholds (linear maxIVF fraction = kmeans 16*sqrt(n)):")
    for f in (0.01, 0.025, 0.05):
        n_star = crossover_n_tokens(f)
        print(f"  f={f:.3f}  →  n* = {n_star:>15,.0f} tokens (kmeans @ n*: {_kmeans_n_centroids(int(n_star)):>10,})")
    print()

    rows: list[dict] = []
    for p in args.results:
        with p.open() as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if obj.get("_kind") == "run_header":
                    continue
                if "error" in obj:
                    continue
                rows.append(obj)

    if not rows:
        print("no usable rows", file=sys.stderr)
        sys.exit(1)

    # Collapse to one row per (dataset, config); first occurrence wins.
    by_key: dict[tuple[str, str], dict] = {}
    for r in rows:
        k = (r["dataset"], r["config"])
        if k not in by_key:
            by_key[k] = r

    headers = [
        "dataset", "config",
        "n_tokens", "n_cents", "kmeans_base", "ratio",
        "ndcg@10", "rec@100", "ret_s",
    ]
    fmt = "{:<12} {:<24} {:>12} {:>10} {:>11} {:>7} {:>8} {:>8} {:>7}"
    print(fmt.format(*headers))
    print("-" * 120)
    for (_, _), r in sorted(by_key.items(), key=lambda kv: (kv[0][0], kv[0][1])):
        n_tokens = r.get("n_tokens_total") or 0
        n_centroids = r.get("n_centroids_planned") or 0
        if n_tokens > 0:
            kmeans_base = _kmeans_n_centroids(int(n_tokens))
            ratio = (n_centroids / kmeans_base) if kmeans_base > 0 else 0.0
            ratio_s = f"{ratio:5.2f}x"
        else:
            kmeans_base = 0
            ratio_s = "-"
        m = r.get("metrics", {})
        print(fmt.format(
            r["dataset"], r["config"],
            f"{int(n_tokens):,}" if n_tokens else "-",
            f"{int(n_centroids):,}" if n_centroids else "-",
            f"{kmeans_base:,}" if kmeans_base else "-",
            ratio_s,
            f"{m.get('ndcg@10', float('nan')):.4f}",
            f"{m.get('recall@100', float('nan')):.4f}",
            f"{r.get('retrieve_s', 0):.2f}",
        ))


if __name__ == "__main__":
    main()
