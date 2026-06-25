#!/usr/bin/env python3
"""Aggregate the trec-covid GPU-PGC/CAGRA sweep (scripts/run_gpu_cagra_sweep.sh).

Joins, per condition: clustering wall-clock (gpu_cluster.py meta.json, or the cluster_only
PGC log) with downstream tachiom retrieval (results/gpucagra/<tag>.jsonl retrieve rows).
Safe to run repeatedly while jobs are still in flight — missing pieces show as '-'.

  uv run --no-sync python scripts/summarize_gpucagra.py
"""
from __future__ import annotations
import glob
import json
import re
from pathlib import Path

ROOT = Path("/exp/rjha/pylate-pgc")
RES = ROOT / "results/gpucagra"
CL = ROOT / "clusterings"
LOGS = ROOT / "logs/gpucagra"

# tag -> human label / method
TAGS = [
    ("pgc", "CPU-PGC (baseline)"),
    ("brute", "GPU exact (pytorch)"),
    ("nd_g64_i64", "CAGRA nn_descent g64 itopk64 (default)"),
    ("nd_g32_i64", "CAGRA nn_descent g32 itopk64"),
    ("nd_g64_i32", "CAGRA nn_descent g64 itopk32"),
    ("nd_g64_i128", "CAGRA nn_descent g64 itopk128"),
    ("ivfpq_g64_i64", "CAGRA ivf_pq g64 itopk64"),
]


def clustering_seconds(tag: str):
    meta = CL / f"tcv_{tag}" / "meta.json"
    if meta.exists():
        try:
            return json.loads(meta.read_text()).get("clustering_seconds")
        except Exception:
            pass
    # PGC (cluster_only) has no meta.json — parse its log line "... clustering done in Xs"
    for log in (LOGS / f"cl_{tag}.log",):
        if log.exists():
            m = re.search(r"clustering done in ([\d.]+)s", log.read_text())
            if m:
                return float(m.group(1))
    return None


def retrieve_row(tag: str):
    f = RES / f"{tag}.jsonl"
    if not f.exists():
        return None
    rows = [json.loads(l) for l in f.read_text().splitlines() if l.strip()]
    # retrieve rows carry top-level ndcg@10 / qps; build rows don't
    rr = [r for r in rows if "ndcg@10" in r or "qps" in r]
    return rr[-1] if rr else None


def fmt(v, nd=4):
    return f"{v:.{nd}f}" if isinstance(v, (int, float)) else "-"


def main():
    base_ndcg = None
    print(f"\n=== trec-covid GPU-PGC/CAGRA sweep (K=293130) ===")
    print(f"{'condition':<40} {'cluster_s':>10} {'ndcg@10':>9} {'recall@100':>11} "
          f"{'qps':>8} {'build_s':>8} {'Δndcg':>7}")
    print("-" * 96)
    rows = {}
    for tag, label in TAGS:
        cs = clustering_seconds(tag)
        rr = retrieve_row(tag)
        rows[tag] = (cs, rr)
        if tag == "pgc" and rr and isinstance(rr.get("ndcg@10"), (int, float)):
            base_ndcg = rr["ndcg@10"]
    for tag, label in TAGS:
        cs, rr = rows[tag]
        ndcg = rr.get("ndcg@10") if rr else None
        rec = rr.get("recall@100") if rr else None
        qps = rr.get("qps") if rr else None
        bt = rr.get("build_time_s") if rr else None
        dn = (ndcg - base_ndcg) if (isinstance(ndcg, (int, float)) and base_ndcg is not None) else None
        print(f"{label:<40} {fmt(cs,1):>10} {fmt(ndcg):>9} {fmt(rec):>11} "
              f"{fmt(qps,1):>8} {fmt(bt,1):>8} {('%+.4f'%dn) if dn is not None else '-':>7}")
    print("-" * 96)
    print("cluster_s = clustering wall-clock only; build_s = PQ+HNSW+IVF (downstream, identical across rows).")
    print("Δndcg = ndcg@10 minus CPU-PGC baseline. '-' = job still pending/failed.\n")


if __name__ == "__main__":
    main()
