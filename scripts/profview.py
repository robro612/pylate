#!/usr/bin/env python
"""profview — inspect retrieval timing/profile results in the terminal.

Reads a benchmark results JSONL (rows written by scripts/benchmark_indexes.py)
and renders, with rich:

  * a summary table (method, backend, qps, quality), and
  * stacked stage-timing bars — the per-query token-path breakdown
    (index_lookup / gather / maxsim / topk) from each row's ``profile`` key when
    present, plus the build pipeline (encode / cluster / build) from the coarse
    ``*_time_s`` fields.

Run:
  uv run python scripts/profview.py results_maxsim_nfcorpus.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from rich.box import ROUNDED, SIMPLE
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# Stage palette — shared with the design artifact. "cool→warm", maxsim is the
# warm stage under scrutiny.
QUERY_STAGES = [
    ("index_lookup", "#00af87"),
    ("candidate_dedup", "#87d75f"),
    ("gather", "#ffd75f"),
    ("reshape_inputs", "#5fafff"),
    ("prepare_tensors", "#ffaf00"),
    ("device_transfer", "#ff5fd7"),
    ("maxsim", "#ff5f5f"),
    ("topk", "#875fff"),
    ("result_materialize", "#00d7ff"),
    ("unaccounted", "#8a8a8a"),
]
BUILD_STAGES = [
    ("doc_encode_time_s", "#00af87", "encode_docs"),
    ("query_encode_time_s", "#00d7ff", "encode_q"),
    ("cluster_time_s", "#ffaf00", "cluster"),
    ("build_time_s", "#ff5f5f", "build"),
]
ENCODE_STAGES = [
    ("model_init", "#5fafff"),
    ("start_pool", "#00af87"),
    ("model_encode", "#ff5f5f"),
    ("encode_query", "#ff5f5f"),
    ("load_cache_arrays", "#00af87"),
    ("reconstruct_query_tensors", "#00d7ff"),
    ("filter_tokens", "#87d75f"),
    ("prepare_shard", "#ffaf00"),
    ("prepare_cache", "#ffaf00"),
    ("save_shard", "#875fff"),
    ("save_cache", "#875fff"),
    ("cleanup", "#00d7ff"),
    ("unaccounted", "#8a8a8a"),
]
BAR_W = 46


def load_rows(path: Path) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def short_model(name: str) -> str:
    return (name or "?").split("/")[-1]


def display_retrieval(row: dict) -> str:
    """Human-facing retrieval label.

    Some older Tachiom rows were written with retrieval="xtr" even though the
    benchmark dispatches end-to-end Tachiom through the ColBERT retriever. Keep
    old files readable without rewriting their raw JSON.
    """
    index = str(row.get("index_type") or "")
    retrieval = str(row.get("retrieval") or "")
    if index == "tachiom" and retrieval == "xtr":
        return "colbert"
    return retrieval


def run_label(row: dict) -> str:
    index = str(row.get("index_type") or "?")
    retrieval = display_retrieval(row) or "?"
    backend = row.get("maxsim_backend")
    label = f"{index}/{retrieval}"
    if backend:
        label += f"/{backend}"
    note = row.get("comparison_note")
    if note == "tachiom_full_batch_amortized":
        label += "/amortized"
    elif note == "tachiom_bs1_profile":
        label += "/bs1"
    return label


def stacked_bar(segments: list[tuple[str, float, str]], max_total: float) -> Text:
    """segments: (name, value, color). Width ∝ value, scaled to max_total."""
    bar = Text()
    if max_total <= 0:
        return bar
    for _name, val, color in segments:
        if val and val > 0:
            n = max(1, round(BAR_W * val / max_total))
            bar.append("█" * n, style=color)
    return bar


def fmt_ms(v: float) -> str:
    return f"{v:7.2f}" if v == v else "    -  "  # nan check


def query_profile_segments(row: dict) -> tuple[list, float, str]:
    """Return (segments, total_ms, annotations) from a row's `profile` dict."""
    profile = row["profile"]
    # End-to-end indexes: the rust search isn't broken down yet, so render the
    # coarse `search` total as a single segment rather than per-stage. Older E2E
    # rows have one full-query-set span, so they are shown as amortized ms/query.
    # Newer bs=1 E2E profile runs have one root per query; use their measured
    # p50 directly.
    if profile.get("_root") == "search":
        n_queries = int(row.get("n_queries") or 0)
        tot = profile.get("_total", {})
        n_spans = int(tot.get("n") or 0)
        if n_spans > 1:
            ms = tot.get("p50_ms", 0.0)
            note = f"rust search p50 over {n_spans} query spans; internal stages not yet instrumented"
        elif n_queries > 0 and row.get("search_time_s") is not None:
            ms = float(row["search_time_s"]) * 1000.0 / n_queries
            note = f"rust search — amortized over {n_queries}; internal stages not yet instrumented"
        else:
            ms = tot.get("p50_ms", 0.0)
            note = "rust search — internal stages not yet instrumented"
        return [("search", ms, "#2f6690")], ms, note
    segs, total, notes = [], 0.0, []
    for name, color in QUERY_STAGES:
        st = profile.get(name)
        if not st:
            continue
        ms = st.get("p50_ms", 0.0)
        segs.append((name, ms, color))
        total += ms
        if st.get("amortized"):
            notes.append(f"{name}=amortized/{st.get('count')}")
    backend = profile.get("_maxsim_backend")
    label = f"maxsim:{','.join(backend)}" if backend else ""
    if notes:
        label += "  ⚠ " + ", ".join(notes)
    known = {name for name, _ in QUERY_STAGES}
    extras = [
        name
        for name, value in profile.items()
        if not name.startswith("_") and name not in known and isinstance(value, dict)
    ]
    if extras:
        label += f"  extra:{','.join(extras)}"
    return segs, total, label


def stage_profile_segments(profile: dict, stage_order: list[tuple[str, str]]) -> tuple[list, float, str]:
    segs, total, notes = [], 0.0, []
    known = {name for name, _ in stage_order}
    for name, color in stage_order:
        st = profile.get(name)
        if not st:
            continue
        ms = st.get("p50_ms", 0.0)
        segs.append((name, ms, color))
        total += ms
        if st.get("amortized"):
            notes.append(f"{name}=amortized/{st.get('count')}")
    extras = [
        name
        for name, value in profile.items()
        if not name.startswith("_") and name not in known and isinstance(value, dict)
    ]
    label = ""
    if notes:
        label += "⚠ " + ", ".join(notes)
    if extras:
        label += f"  extra:{','.join(extras)}"
    return segs, total, label


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("file", nargs="?", default="results.jsonl")
    ap.add_argument("--dataset", help="substring filter on dataset")
    ap.add_argument("--index", help="substring filter on index_type")
    ap.add_argument("--backend", help="substring filter on maxsim_backend")
    ap.add_argument("--last", type=int, default=0, help="show only the last N rows")
    args = ap.parse_args()

    console = Console()
    path = Path(args.file)
    if not path.exists():
        console.print(f"[red]no such file:[/red] {path}")
        return
    rows = load_rows(path)
    if args.dataset:
        rows = [r for r in rows if args.dataset in str(r.get("dataset", ""))]
    if args.index:
        rows = [r for r in rows if args.index in str(r.get("index_type", ""))]
    if args.backend:
        rows = [r for r in rows if args.backend in str(r.get("maxsim_backend") or "")]
    if args.last:
        rows = rows[-args.last :]
    if not rows:
        console.print("[yellow]no rows match.[/yellow]")
        return

    console.print(
        Panel(
            f"[bold]{path}[/bold]   [dim]{len(rows)} run(s)[/dim]",
            title="profview", title_align="left", border_style="#1f5fd6", box=ROUNDED,
        )
    )

    # ---- summary table ----
    t = Table(box=ROUNDED, border_style="grey37", header_style="bold", expand=False)
    for col, just in [
        ("model", "left"), ("dataset", "left"), ("index", "left"), ("retr", "left"),
        ("backend", "left"), ("nq", "right"), ("qps", "right"),
        ("ndcg@10", "right"), ("r@100", "right"), ("build_s", "right"),
        ("search_s", "right"), ("prof", "center"),
    ]:
        t.add_column(col, justify=just, no_wrap=True)
    for r in rows:
        t.add_row(
            short_model(r.get("model")),
            str(r.get("dataset", "")).replace("beir/", "").replace("/test", ""),
            str(r.get("index_type", "")),
            display_retrieval(r),
            str(r.get("maxsim_backend") or "[dim]—[/dim]"),
            str(r.get("n_queries", "")),
            f"{r.get('qps', float('nan')):.1f}",
            f"{r.get('ndcg@10', float('nan')):.4f}",
            f"{r.get('recall@100', float('nan')):.4f}",
            f"{r.get('build_time_s', 0):.1f}",
            f"{r.get('search_time_s', 0):.3f}",
            "[#cf5630]●[/#cf5630]" if r.get("profile") else "[dim]·[/dim]",
        )
    console.print(t)

    # ---- query-time stage breakdown (only rows with a profile) ----
    prof_rows = [r for r in rows if r.get("profile")]
    if prof_rows:
        max_total = 0.0
        prepared = []
        for r in prof_rows:
            segs, total, note = query_profile_segments(r)
            max_total = max(max_total, total)
            prepared.append((r, segs, total, note))
        bars = Table.grid(padding=(0, 1))
        bars.add_column(justify="right", no_wrap=True)  # label
        bars.add_column(no_wrap=True)                    # bar
        bars.add_column(justify="right", no_wrap=True)   # total ms
        bars.add_column(no_wrap=True)                    # note
        for r, segs, total, note in prepared:
            label = f"{short_model(r.get('model'))} · {run_label(r)}"
            bars.add_row(
                Text(label, style="bold"),
                stacked_bar(segs, max_total),
                Text(f"{total:7.2f} ms", style="bold"),
                Text(note, style="dim"),
            )
        console.print(
            Panel(bars, title="query-time stage breakdown · p50 per query",
                  title_align="left", border_style="grey37", box=SIMPLE)
        )
        legend = Text("  ")
        for name, color in QUERY_STAGES:
            legend.append("■ ", style=color)
            legend.append(f"{name}   ", style="dim")
        console.print(legend)

    # ---- encode profile breakdowns (only present when encode stages ran with search.profile=true) ----
    for key, title in [
        ("query_encode_profile", "query encode profile · p50"),
        ("doc_encode_profile", "document encode profile · p50"),
    ]:
        enc_rows = [r for r in rows if r.get(key)]
        if not enc_rows:
            continue
        max_total = 0.0
        prepared = []
        for r in enc_rows:
            segs, total, note = stage_profile_segments(r[key], ENCODE_STAGES)
            max_total = max(max_total, total)
            prepared.append((r, segs, total, note))
        bars = Table.grid(padding=(0, 1))
        bars.add_column(justify="right", no_wrap=True)
        bars.add_column(no_wrap=True)
        bars.add_column(justify="right", no_wrap=True)
        bars.add_column(no_wrap=True)
        for r, segs, total, note in prepared:
            label = f"{short_model(r.get('model'))} · {r.get('index_type')}"
            bars.add_row(
                Text(label, style="bold"),
                stacked_bar(segs, max_total),
                Text(f"{total:7.2f} ms", style="bold"),
                Text(note, style="dim"),
            )
        console.print(
            Panel(
                bars,
                title=title,
                title_align="left",
                border_style="grey37",
                box=SIMPLE,
            )
        )
        legend = Text("  ")
        for name, color in ENCODE_STAGES:
            legend.append("■ ", style=color)
            legend.append(f"{name}   ", style="dim")
        console.print(legend)

    # ---- build pipeline (coarse *_time_s) ----
    build_rows = [r for r in rows if any(r.get(k, 0) for k, _, _ in BUILD_STAGES)]
    if build_rows:
        max_total = max(sum(r.get(k, 0) for k, _, _ in BUILD_STAGES) for r in build_rows)
        bars = Table.grid(padding=(0, 1))
        bars.add_column(justify="right", no_wrap=True)
        bars.add_column(no_wrap=True)
        bars.add_column(justify="right", no_wrap=True)
        for r in build_rows:
            segs = [(lbl, r.get(k, 0), color) for k, color, lbl in BUILD_STAGES]
            total = sum(v for _, v, _ in segs)
            label = f"{short_model(r.get('model'))} · {r.get('index_type')}"
            bars.add_row(
                Text(label, style="bold"),
                stacked_bar(segs, max_total),
                Text(f"{total:7.1f} s", style="bold"),
            )
        console.print(
            Panel(bars, title="build pipeline · seconds",
                  title_align="left", border_style="grey37", box=SIMPLE)
        )
        legend = Text("  ")
        for _k, color, lbl in BUILD_STAGES:
            legend.append("■ ", style=color)
            legend.append(f"{lbl}   ", style="dim")
        console.print(legend)


if __name__ == "__main__":
    main()
