#!/usr/bin/env python3
"""LoTTE head-to-head: GPU-CAGRA vs CPU-PGC clustering, identical pipeline.

Reads results/gpucagra/lotte_{cagra,pgc}_kc*_kd*.jsonl (retrieve rows) + the CAGRA
clustering meta.json, prints a cell-by-cell table. Safe to run anytime.
  uv run --no-sync python scripts/summarize_lotte_cagra.py
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

ROOT = Path("/exp/rjha/pylate-pgc")
RES = ROOT / "results/gpucagra"
GRID = [(40, 10000), (40, 20000), (80, 10000), (80, 20000)]
MET = ("ndcg@10", "recall@100", "hit_rate@5", "mrr@10", "qps")


def row(method, kc, kd):
    f = RES / f"lotte_{method}_kc{kc}_kd{kd}.jsonl"
    if not f.exists():
        return None
    rr = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    rr = [r for r in rr if "ndcg@10" in r or "qps" in r]
    return rr[-1] if rr else None


def g(r, k):
    return r.get(k) if r else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    print("\n=== LoTTE: GPU-CAGRA vs CPU-PGC clustering (identical pipeline, M=32) ===")
    # clustering wall-clock
    meta = ROOT / "clusterings/lotte_cagra/meta.json"
    if meta.exists():
        m = json.loads(meta.read_text())
        print(f"clustering: CAGRA {m['clustering_seconds']:.0f}s on {m['gpu']} "
              f"(train_sample={m.get('train_sample')}, iters={m['iters']}, peak {m['peak_gpu_gb']}GB)")
    print("            CPU-PGC lotte_pgc_m4t05 ~3h10m (mult=40, prior scale study)\n")
    hdr = f"{'kc':>4} {'kd':>7} | " + " ".join(f"{m:>11}" for m in MET)
    for method in ("cagra", "pgc"):
        print(f"--- {method.upper()} ---")
        print(hdr)
        for kc, kd in GRID:
            r = row(method, kc, kd)
            print(f"{kc:>4} {kd:>7} | " + " ".join(f"{fmt(g(r, m), 1 if m=='qps' else 4):>11}" for m in MET))
    print("\n--- Δ (CAGRA − PGC) ---")
    print(f"{'kc':>4} {'kd':>7} | {'Δndcg@10':>11} {'Δrecall@100':>12} {'Δhit_rate@5':>12}")
    for kc, kd in GRID:
        c, p = row("cagra", kc, kd), row("pgc", kc, kd)
        def d(k):
            a, b = g(c, k), g(p, k)
            return f"{a-b:+.4f}" if isinstance(a, (int, float)) and isinstance(b, (int, float)) else "-"
        print(f"{kc:>4} {kd:>7} | {d('ndcg@10'):>11} {d('recall@100'):>12} {d('hit_rate@5'):>12}")
    print()


if __name__ == "__main__":
    main()
