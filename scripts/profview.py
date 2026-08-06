#!/usr/bin/env python
"""profview — inspect retrieval timing/profile results in the terminal.

Reads a benchmark results JSONL (rows written by scripts/benchmark_indexes.py)
and renders, with rich:

  * a summary table (method, backend, qps, quality), and
  * grouped stacked timing bars from the row's query, encode, and index profiles.

Run:
  uv run python scripts/profview.py results_maxsim_nfcorpus.jsonl
  uv run python scripts/profview.py results.jsonl --section query,index
  uv run python scripts/profview.py results.jsonl --section query --detail stage
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

# Stage palette — shared with the design artifact. Stage-detail colors are
# intentionally richer, while the default grouped view uses fewer semantic hues.
QUERY_STAGES = [
    ("index_lookup", "#00af87"),
    ("candidate_dedup", "#87d75f"),
    ("gather", "#ffd75f"),
    ("reshape_inputs", "#5fafff"),
    ("prepare_tensors", "#ffaf00"),
    ("device_transfer", "#ff5fd7"),
    ("maxsim", "#ff5f5f"),
    ("coarse_score", "#d75f00"),
    ("coarse_hnsw_lookup", "#d75f00"),
    ("coarse_accumulate", "#af8700"),
    ("candidate_select", "#5fd7ff"),
    ("candidate_topk", "#5fd7ff"),
    ("candidate_sort", "#00afd7"),
    ("alpha_prune", "#87d7ff"),
    ("rerank", "#d70087"),
    ("rerank_prepare", "#ff87af"),
    ("rerank_score", "#d70087"),
    ("rerank_select", "#af005f"),
    ("rerank_sort", "#ff5faf"),
    ("query_prepare", "#5fafff"),
    ("centroid_score", "#d75f00"),
    ("ivf_select", "#ffaf00"),
    ("ivf_lookup", "#ffd75f"),
    ("candidate_lengths", "#afd75f"),
    ("rerank_lengths", "#87af5f"),
    ("approx_lookup", "#5fd7ff"),
    ("approx_score", "#00afd7"),
    ("approx_topk", "#875fff"),
    ("decompress_topk", "#af5fff"),
    ("exact_lookup", "#ff87af"),
    ("residual_decompress", "#d70087"),
    ("exact_score", "#ff5f5f"),
    ("final_topk", "#875f00"),
    ("xtr_validate", "#5fafff"),
    ("xtr_flatten", "#ffd75f"),
    ("xtr_prepare_tensors", "#ffaf00"),
    ("xtr_scatter_max", "#d70087"),
    ("xtr_impute_sum", "#ff5f5f"),
    ("xtr_topk", "#875fff"),
    ("topk", "#875fff"),
    ("result_materialize", "#00d7ff"),
    ("unaccounted", "#8a8a8a"),
]
QUERY_GROUPS = [
    (
        "lookup",
        "#00af87",
        {
            "index_lookup",
            "coarse_hnsw_lookup",
            "centroid_score",
            "ivf_lookup",
            "approx_lookup",
            "exact_lookup",
        },
    ),
    (
        "coarse_score",
        "#d75f00",
        {"coarse_score", "coarse_accumulate", "approx_score"},
    ),
    (
        "candidate_select",
        "#5fd7ff",
        {
            "candidate_select",
            "candidate_dedup",
            "candidate_topk",
            "candidate_sort",
            "alpha_prune",
            "ivf_select",
            "topk",
            "approx_topk",
            "decompress_topk",
        },
    ),
    (
        "prepare",
        "#ffaf00",
        {
            "gather",
            "reshape_inputs",
            "prepare_tensors",
            "device_transfer",
            "query_prepare",
            "rerank_prepare",
            "candidate_lengths",
            "rerank_lengths",
            "xtr_validate",
            "xtr_flatten",
            "xtr_prepare_tensors",
        },
    ),
    (
        "score",
        "#ff5f5f",
        {
            "maxsim",
            "rerank",
            "rerank_score",
            "exact_score",
            "residual_decompress",
            "xtr_scatter_max",
            "xtr_impute_sum",
        },
    ),
    (
        "finalize",
        "#875fff",
        {"rerank_select", "rerank_sort", "final_topk", "xtr_topk", "result_materialize"},
    ),
    ("unaccounted", "#8a8a8a", {"unaccounted"}),
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
ENCODE_GROUPS = [
    ("setup", "#5fafff", {"model_init", "start_pool"}),
    ("cache_load", "#00af87", {"load_cache_arrays", "reconstruct_query_tensors"}),
    ("model_encode", "#ff5f5f", {"model_encode", "encode_query"}),
    ("token_filter", "#87d75f", {"filter_tokens"}),
    ("prepare", "#ffaf00", {"prepare_shard", "prepare_cache"}),
    ("save", "#875fff", {"save_shard", "save_cache"}),
    ("cleanup", "#00d7ff", {"cleanup"}),
    ("unaccounted", "#8a8a8a", {"unaccounted"}),
]
EXTRA_COLORS = [
    "#5fd787",
    "#d7af00",
    "#af5fff",
    "#00afd7",
    "#ff8700",
    "#87afff",
    "#d75f87",
    "#87d7af",
]
BAR_W = 46
SECTIONS = ("query", "query_encode", "doc_encode", "index")
SECTION_ALIASES = {
    "all": set(SECTIONS),
    "query": {"query"},
    "search": {"query"},
    "retrieval": {"query"},
    "query_encode": {"query_encode"},
    "query-encode": {"query_encode"},
    "qencode": {"query_encode"},
    "doc_encode": {"doc_encode"},
    "doc-encode": {"doc_encode"},
    "docencode": {"doc_encode"},
    "document_encode": {"doc_encode"},
    "index": {"index"},
    "build": {"index"},
    "index_build": {"index"},
}


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


def parse_sections(values: list[str] | None) -> set[str]:
    if not values:
        return set(SECTIONS)
    selected: set[str] = set()
    for raw in values:
        for part in raw.split(","):
            key = part.strip().lower().replace(" ", "_")
            if not key:
                continue
            if key not in SECTION_ALIASES:
                choices = ", ".join(SECTIONS)
                raise argparse.ArgumentTypeError(f"unknown section {part!r}; choose from {choices}, all")
            selected.update(SECTION_ALIASES[key])
    return selected


def group_segments(
    segments: list[tuple[str, float, str]],
    groups: list[tuple[str, str, set[str]]],
) -> list[tuple[str, float, str]]:
    totals = {name: 0.0 for name, _color, _members in groups}
    colors = {name: color for name, color, _members in groups}
    stage_to_group = {
        stage: name
        for name, _color, members in groups
        for stage in members
    }
    extras: list[tuple[str, float, str]] = []
    for name, value, color in segments:
        group = stage_to_group.get(name)
        if group:
            totals[group] += value
        else:
            extras.append((name, value, color))
    grouped = [
        (name, totals[name], colors[name])
        for name, _color, _members in groups
        if totals[name] > 0
    ]
    return grouped + extras


def maybe_group_segments(
    segments: list[tuple[str, float, str]],
    groups: list[tuple[str, str, set[str]]],
    detail: str,
) -> list[tuple[str, float, str]]:
    if detail == "stage":
        return segments
    return group_segments(segments, groups)


def query_profile_segments(row: dict, detail: str) -> tuple[list, float, str]:
    """Return (segments, total_ms, annotations) from a row's `profile` dict."""
    profile = row["profile"]
    # End-to-end indexes report a `search` root. If Rust has emitted child
    # spans, render them like any other stage profile; otherwise show the root
    # as one opaque segment. Older E2E rows have one full-query-set span, so
    # they are shown as amortized ms/query. Newer bs=1 E2E profile runs have one
    # root per query; use their measured p50 directly.
    if profile.get("_root") == "search":
        n_queries = int(row.get("n_queries") or 0)
        tot = profile.get("_total", {})
        n_spans = int(tot.get("n") or 0)
        internal = [
            key
            for key, value in profile.items()
            if not key.startswith("_") and key != "unaccounted" and isinstance(value, dict)
        ]
        if internal:
            raw_segs, total, note = stage_profile_segments(profile, QUERY_STAGES, "stage")
            segs = maybe_group_segments(raw_segs, QUERY_GROUPS, detail)
            prefix = f"rust search p50 over {n_spans} query spans" if n_spans > 1 else "rust search"
            return segs, total, (prefix + (f"; {note}" if note else ""))
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
    raw_segs = segs
    segs = maybe_group_segments(raw_segs, QUERY_GROUPS, detail)
    return segs, total, label


def stage_profile_segments(
    profile: dict,
    stage_order: list[tuple[str, str]],
    detail: str,
    groups: list[tuple[str, str, set[str]]] | None = None,
) -> tuple[list, float, str]:
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
    for i, name in enumerate(extras):
        st = profile[name]
        ms = st.get("p50_ms", 0.0)
        segs.append((name, ms, EXTRA_COLORS[i % len(EXTRA_COLORS)]))
        total += ms
        if st.get("amortized"):
            notes.append(f"{name}=amortized/{st.get('count')}")
    label = ""
    if notes:
        label += "⚠ " + ", ".join(notes)
    if extras:
        label += f"  extra:{','.join(extras)}"
    if detail == "group" and groups:
        raw_segs = segs
        segs = group_segments(raw_segs, groups)
    return segs, total, label


def legend_for_segments(
    stage_order: list[tuple[str, str]], prepared: list[tuple[dict, list, float, str]]
) -> Text:
    used = []
    for _row, segs, _total, _note in prepared:
        for name, _value, color in segs:
            if name not in {n for n, _c in used}:
                used.append((name, color))
    ordered = []
    for name, color in stage_order:
        if name in {n for n, _c in used}:
            ordered.append((name, color))
    for name, color in used:
        if name not in {n for n, _c in ordered}:
            ordered.append((name, color))
    legend = Text("  ")
    for name, color in ordered:
        legend.append("■ ", style=color)
        legend.append(f"{name}   ", style="dim")
    return legend


def group_order(groups: list[tuple[str, str, set[str]]]) -> list[tuple[str, str]]:
    return [(name, color) for name, color, _members in groups]


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", nargs="?", default="results.jsonl")
    ap.add_argument("--dataset", help="substring filter on dataset")
    ap.add_argument("--index", help="substring filter on index_type")
    ap.add_argument("--backend", help="substring filter on maxsim_backend")
    ap.add_argument("--last", type=int, default=0, help="show only the last N rows")
    ap.add_argument(
        "--section",
        "--profile",
        action="append",
        help=(
            "profile section(s) to show: query, query_encode, doc_encode, index, all. "
            "May be comma-separated or repeated. Default: all."
        ),
    )
    ap.add_argument(
        "--detail",
        choices=("group", "stage"),
        default="group",
        help="group semantic stages by default; use stage for the full low-level breakdown",
    )
    args = ap.parse_args()
    try:
        sections = parse_sections(args.section)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))

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
    prof_rows = [r for r in rows if "query" in sections and r.get("profile")]
    if prof_rows:
        max_total = 0.0
        prepared = []
        for r in prof_rows:
            segs, total, note = query_profile_segments(r, args.detail)
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
            Panel(bars, title=f"query-time {args.detail} breakdown · p50 per query",
                  title_align="left", border_style="grey37", box=SIMPLE)
        )
        legend_order = group_order(QUERY_GROUPS) if args.detail == "group" else QUERY_STAGES
        console.print(legend_for_segments(legend_order, prepared))

    # ---- encode profile breakdowns (only present when encode stages ran with search.profile=true) ----
    for key, title in [
        ("query_encode_profile", "query encode"),
        ("doc_encode_profile", "document encode"),
    ]:
        section = "query_encode" if key == "query_encode_profile" else "doc_encode"
        enc_rows = [r for r in rows if section in sections and r.get(key)]
        if not enc_rows:
            continue
        max_total = 0.0
        prepared = []
        for r in enc_rows:
            segs, total, note = stage_profile_segments(
                r[key],
                ENCODE_STAGES,
                args.detail,
                ENCODE_GROUPS,
            )
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
                title=f"{title} {args.detail} breakdown · p50",
                title_align="left",
                border_style="grey37",
                box=SIMPLE,
            )
        )
        legend_order = group_order(ENCODE_GROUPS) if args.detail == "group" else ENCODE_STAGES
        console.print(legend_for_segments(legend_order, prepared))

    # ---- build pipeline (coarse *_time_s) ----
    build_rows = [
        r
        for r in rows
        if "index" in sections and any(r.get(k, 0) for k, _, _ in BUILD_STAGES)
    ]
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
            Panel(bars, title="index pipeline · seconds",
                  title_align="left", border_style="grey37", box=SIMPLE)
        )
        legend = Text("  ")
        for _k, color, lbl in BUILD_STAGES:
            legend.append("■ ", style=color)
            legend.append(f"{lbl}   ", style="dim")
        console.print(legend)


if __name__ == "__main__":
    main()
