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
import sys
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
    ("rerank_score_full", "#d70087"),
    ("rerank_seed", "#ff5f87"),
    ("rerank_stream", "#d7005f"),
    ("rerank_select", "#af005f"),
    ("rerank_sort", "#ff5faf"),
    ("query_prepare", "#5fafff"),
    ("query_layout", "#87afff"),
    ("centroid_score", "#d75f00"),
    ("ivf_select", "#ffaf00"),
    ("ivf_lookup", "#ffd75f"),
    ("candidate_lengths", "#afd75f"),
    ("rerank_lengths", "#87af5f"),
    ("approx_lookup", "#5fd7ff"),
    ("approx_gather", "#00d7af"),
    ("approx_pad", "#5fafaf"),
    ("approx_reduce", "#00afd7"),
    ("approx_score", "#00afd7"),
    ("approx_plan", "#87afaf"),
    ("approx_merge", "#5f8787"),
    ("approx_topk", "#875fff"),
    ("decompress_topk", "#af5fff"),
    ("exact_lookup", "#ff87af"),
    ("residual_decompress", "#d70087"),
    ("exact_pad", "#af5f87"),
    ("exact_matmul", "#ff5f5f"),
    ("exact_reduce", "#d75f5f"),
    ("exact_score", "#ff5f5f"),
    ("exact_plan", "#af8787"),
    ("exact_merge", "#875f5f"),
    ("exact_lengths", "#87af5f"),
    ("exact_residual_decompress", "#d70087"),
    ("rerank_encoder", "#ff87d7"),
    ("id_map_load", "#af875f"),
    ("query_convert", "#d7af5f"),
    ("query_pack", "#d7af5f"),
    ("final_topk", "#875f00"),
    ("xtr_validate", "#5fafff"),
    ("xtr_flatten", "#ffd75f"),
    ("xtr_prepare_tensors", "#ffaf00"),
    ("xtr_scatter_max", "#d70087"),
    ("xtr_impute_sum", "#ff5f5f"),
    ("xtr_topk", "#875fff"),
    ("topk", "#875fff"),
    ("result_materialize", "#00d7ff"),
    ("result_convert", "#5fd7ff"),
    # Measurement / dispatch cost, not retrieval work. Coloured close to
    # unaccounted so it reads as overhead at a glance in the bars.
    ("search_dispatch", "#5f5faf"),
    ("profile_drain", "#5f5f87"),
    ("overhead_dispatch", "#5f5faf"),
    ("overhead_drain", "#5f5f87"),
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
        {
            "coarse_score",
            "coarse_accumulate",
            "approx_score",
            "approx_gather",
            "approx_pad",
            "approx_reduce",
            "approx_plan",
            "approx_merge",
        },
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
            "query_layout",
            "rerank_prepare",
            "candidate_lengths",
            "rerank_lengths",
            "exact_lengths",
            "rerank_encoder",
            "id_map_load",
            "query_convert",
            "query_pack",
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
            "rerank_score_full",
            "rerank_seed",
            "rerank_stream",
            "exact_score",
            "residual_decompress",
            "exact_residual_decompress",
            "exact_pad",
            "exact_matmul",
            "exact_reduce",
            "exact_plan",
            "exact_merge",
            "xtr_scatter_max",
            "xtr_impute_sum",
        },
    ),
    (
        "finalize",
        "#875fff",
        {
            "rerank_select",
            "rerank_sort",
            "final_topk",
            "xtr_topk",
            "result_materialize",
            "result_convert",
        },
    ),
    (
        "overhead",
        "#5f5faf",
        {"search_dispatch", "profile_drain", "overhead_dispatch", "overhead_drain"},
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
# Phase (aggregate) rows in the stage table. Tuned for a light terminal: grey85
# is a visible band on white while still letting the stage colours read on top.
# On a dark background swap to something like "on grey19".
PHASE_ROW_STYLE = "on grey85"
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
        group = stage_to_group.get(canonical(name))
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
            # Only anomalies get words. "p50 per query over N spans" was on every
            # row, identical, and pushed the label and total columns off-screen;
            # the panel title already says what the numbers are. One span per
            # query is the expected shape, so say something only when it is not.
            parts = []
            if n_spans and n_queries and n_spans != n_queries:
                parts.append(f"⚠ {n_spans} spans / {n_queries} queries")
            if note:
                parts.append(note)
            return segs, total, "  ".join(parts)
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
    # Match on the underscore spelling so `parent/leaf` stages resolve to their
    # palette entry. Without this every renamed stage falls through to the
    # "extra" branch: wrong colour, and an `extra:` list long enough to squeeze
    # the bars themselves off the panel.
    present = {
        canonical(name): name
        for name, value in profile.items()
        if not name.startswith("_") and isinstance(value, dict)
    }
    matched: set[str] = set()
    for name, color in stage_order:
        key = present.get(canonical(name))
        if key is None:
            continue
        matched.add(key)
        st = profile[key]
        ms = st.get("p50_ms", 0.0)
        segs.append((key, ms, color))
        total += ms
        if st.get("amortized"):
            notes.append(f"{key}=amortized/{st.get('count')}")
    extras = [key for key in present.values() if key not in matched]
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
        segs = group_segments(segs, groups)
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


_STAGE_COLOR = dict(QUERY_STAGES)


def canonical(name: str) -> str:
    """Underscore spelling of a stage name.

    `parent/leaf` and the older `parent_leaf` are the same stage under two
    conventions, and both appear in results on disk either side of the rename.
    Normalising here means the palette and `QUERY_GROUPS` keep one entry per
    stage instead of two, and artifacts from before the rename keep rendering.
    """
    return name.replace("/", "_")


class _StageColor(dict):
    """Palette that resolves `/` names through their underscore spelling."""

    def get(self, name, default=None):  # noqa: A003 - dict interface
        return super().get(name) or super().get(canonical(name), default)


STAGE_COLOR = _StageColor(_STAGE_COLOR)

# ---------------------------------------------------------------------------
# Stage hierarchy.
#
# Two levels, owned in two different places on purpose:
#
#   * **Structural** (this section) — does `rerank_stream` sit inside `rerank`?
#     That is containment, and only the code knows it. The forward-looking form
#     is a `parent/leaf` stage name emitted by the span marker itself, which
#     `reduce_stage_timings` picks up into `_parents` with correct percentiles.
#     `STRUCTURAL_PREFIXES` below is the bridge for the names that predate the
#     convention: they already carry the parent as an underscore prefix, so the
#     pivot works on artifacts already on disk.
#   * **Semantic** (`QUERY_GROUPS`, above) — is PLAID's `exact_matmul` the same
#     *kind* of work as tachiom's `rerank_stream`? Neither fork can answer that
#     about the other, and we want to re-cut it over archived results without
#     re-running, so it stays post-hoc.
#
# Keying the semantic map on parents rather than leaves is what stops it going
# stale: a new `rerank/*` leaf inherits its bucket instead of silently falling
# out as an unclassified extra.
STRUCTURAL_PREFIXES = {
    "rerank", "approx", "exact", "candidate", "query", "result", "ivf", "xtr", "coarse",
}
# Names whose parent is not their underscore prefix.
STAGE_PARENT_OVERRIDE = {
    # Measurement cost, not pipeline work — see the `overhead` semantic group.
    "profile_drain": "overhead",
    "search_dispatch": "overhead",
    # PLAID's exact-scoring phase, spelled without the prefix.
    "residual_decompress": "exact",
}


def parent_of(name: str) -> str | None:
    """Structural parent of a stage, or None if it stands alone.

    Prefers the explicit ``parent/leaf`` form; falls back to the underscore
    prefix for the pre-convention names. Returns None rather than guessing when
    the prefix is not a known phase, so an unrecognised stage renders as its own
    flat row instead of inventing a bogus parent.
    """
    if "/" in name:
        return name.split("/", 1)[0]
    if name in STAGE_PARENT_OVERRIDE:
        return STAGE_PARENT_OVERRIDE[name]
    head, sep, _rest = name.partition("_")
    return head if sep and head in STRUCTURAL_PREFIXES else None


def leaf_of(name: str) -> str:
    """Stage name with its parent prefix stripped, for the nested column."""
    parent = parent_of(name)
    if not parent:
        return name
    for sep in ("/", "_"):
        prefix = parent + sep
        if name.startswith(prefix):
            return name[len(prefix) :]
    return name


def pivot_blocks(names: list[str]) -> list[tuple[str | None, list[str]]]:
    """Group pipeline-ordered stages into ``(parent, [stages])`` blocks.

    Only *contiguous* runs are grouped. A phase that reappears later in the
    pipeline yields a second block rather than being hoisted next to the first:
    reordering would destroy the pipeline order the table exists to show, and a
    phase that genuinely runs twice is worth seeing twice. Singletons are
    demoted to flat rows — a subtotal over one stage is noise.
    """
    blocks: list[tuple[str | None, list[str]]] = []
    for name in names:
        parent = parent_of(name)
        if parent is not None and blocks and blocks[-1][0] == parent:
            blocks[-1][1].append(name)
        else:
            blocks.append((parent, [name]))
    return [(p if p and len(ns) > 1 else None, ns) for p, ns in blocks]


def pipeline_stages(profile: dict) -> list[str]:
    """Stage names in pipeline order.

    ``reduce_stage_timings`` aggregates a root's children in the order they were
    recorded, and dicts preserve insertion order, so the profile already carries
    call order — as long as nothing re-sorts it. ``unaccounted`` is self-time
    rather than a step, so it goes last regardless of where it was inserted.
    """
    names = [
        name
        for name, value in profile.items()
        if not name.startswith("_") and isinstance(value, dict) and name != "unaccounted"
    ]
    if isinstance(profile.get("unaccounted"), dict):
        names.append("unaccounted")
    return names


# Counters (n_candidates, n_docs, early_terminated, ...) are still recorded by
# the reducer into each stage's "meta" and into "_parents", and remain in the
# results JSONL. They are simply not rendered: the key set differed enough
# between stages that a single shared column was more noise than signal.
def stage_table(row: dict, pivot: str = "code") -> Table:
    """Per-stage latency table for one run, as a nested pivot in pipeline order.

    ``share`` is computed from *means*, not from the p50 column beside it. Stage
    means are additive — the stages are disjoint and sum to the root — so mean
    shares sum to 100%. Percentiles are not additive, so a p50-derived share
    would not, and the column would quietly fail to add up.

    ``tail`` is p90/p50: how much worse a slow query gets, which is the number
    that tells you whether a stage is a steady cost or an occasional stall.

    The same non-additivity governs the phase rows. A phase's mean and share are
    summed from its leaves and are exact; its p50/p90/tail are only shown when
    the reducer emitted a ``_parents`` entry, i.e. when the *per-query* sums were
    available at reduce time. On older artifacts, and on the underscore-prefix
    names that predate ``parent/leaf``, those cells are ``·`` rather than a
    plausible-looking number that is not the median of anything.
    """
    profile = row.get("profile") or {}
    total = profile.get("_total") or {}
    parents = profile.get("_parents") or {}
    total_mean = float(total.get("mean_ms") or 0.0) or float("nan")

    table = Table(
        box=ROUNDED, border_style="grey37", header_style="bold", expand=False,
        title=f"{short_model(row.get('model'))} · {run_label(row)} · "
              f"{str(row.get('dataset', '')).replace('beir/', '')} · "
              f"n={total.get('n', '?')} queries",
        title_style="bold", title_justify="left",
    )
    for col, just in [
        ("#", "right"), ("phase", "left"), ("stage", "left"),
        ("p50 ms", "right"), ("share", "right"),
        ("p90 ms", "right"), ("tail", "right"), ("mean ms", "right"),
    ]:
        table.add_column(col, justify=just, no_wrap=True)

    def stat_cells(stat: dict) -> list:
        p50 = float(stat.get("p50_ms") or 0.0)
        p90 = float(stat.get("p90_ms") or 0.0)
        mean = float(stat.get("mean_ms") or 0.0)
        share = mean / total_mean * 100.0 if total_mean == total_mean else float("nan")
        tail = p90 / p50 if p50 > 0 else float("nan")
        tail_style = "bold #ff5f5f" if tail == tail and tail >= 2.0 else "dim"
        share_style = "bold" if share == share and share >= 10.0 else ""
        return [
            f"{p50:.3f}",
            Text(f"{share:5.1f}%" if share == share else "    -", style=share_style),
            f"{p90:.3f}",
            Text(f"{tail:.1f}x" if tail == tail else "  -", style=tail_style),
            f"{mean:.3f}",
        ]

    def leaf_row(index: int, name: str, display: str) -> None:
        colour = "dim" if name == "unaccounted" else STAGE_COLOR.get(name, "white")
        table.add_row(
            Text(str(index), style="dim"),
            "",
            Text(display, style=colour),
            *stat_cells(profile[name]),
        )

    names = pipeline_stages(profile)
    numbering = {name: i for i, name in enumerate(names, start=1)}

    if pivot == "semantic":
        blocks = semantic_blocks(profile, names)
    elif pivot == "code":
        blocks = pivot_blocks(names)
    else:
        blocks = [(None, [name]) for name in names]

    for phase, members in blocks:
        if phase is None:
            for name in members:
                leaf_row(numbering[name], name, name)
            continue
        mean = sum(float(profile[n].get("mean_ms") or 0.0) for n in members)
        share = mean / total_mean * 100.0 if total_mean == total_mean else float("nan")
        exact = parents.get(phase)
        share_style = "bold" if share == share and share >= 10.0 else "bold dim"
        if exact:
            cells = stat_cells(exact)
            cells[1] = Text(f"{share:5.1f}%" if share == share else "    -", style=share_style)
            # Bold the latencies too — same band, but it keeps an aggregate from
            # being mistaken for just another stage when skimming a column.
            for i in (0, 2, 4):
                cells[i] = Text(str(cells[i]), style="bold")
        else:
            cells = [
                Text("     ·", style="dim"),
                Text(f"{share:5.1f}%" if share == share else "    -", style=share_style),
                Text("     ·", style="dim"),
                Text("   ·", style="dim"),
                f"{mean:.3f}",
            ]
        # A background band, not a glyph: the row is an aggregate over the rows
        # beneath it, and shading the whole width says that without spending a
        # column on a marker or making the reader decode one.
        table.add_row(
            "",
            Text(phase, style=f"bold {STAGE_COLOR.get(members[0], 'white')}"),
            "",
            *cells,
            style=PHASE_ROW_STYLE,
        )
        for name in members:
            # Strip the parent prefix only under the structural pivot, where the
            # phase column supplies it. A semantic bucket draws from several
            # phases at once, so stripping would render approx_lookup, ivf_lookup
            # and exact_lookup as three indistinguishable rows called "lookup".
            display = leaf_of(name) if pivot == "code" else name
            leaf_row(numbering[name], name, display)

    table.add_section()
    table.add_row(
        "", Text("TOTAL", style="bold"), "",
        f"{float(total.get('p50_ms') or 0.0):.3f}",
        Text("100.0%", style="bold"),
        f"{float(total.get('p90_ms') or 0.0):.3f}",
        "", f"{total_mean:.3f}",
    )
    return table


def semantic_blocks(
    profile: dict, names: list[str]
) -> list[tuple[str | None, list[str]]]:
    """Group stages by the post-hoc ``QUERY_GROUPS`` taxonomy.

    Unlike the structural pivot this *does* reorder: a semantic bucket collects
    stages from anywhere in the pipeline, so the result is comparable across
    backends but is no longer a timeline. That trade is the reason it lives
    behind an explicit ``--pivot semantic`` rather than being the default.
    """
    stage_to_group = {
        stage: name for name, _color, members in QUERY_GROUPS for stage in members
    }
    # Parent-keyed fallback: an unclassified leaf inherits its phase's bucket,
    # so adding `rerank/foo` never silently drops out of the taxonomy.
    parent_group = {
        parent_of(stage): group
        for stage, group in stage_to_group.items()
        if parent_of(stage)
    }
    blocks: list[tuple[str | None, list[str]]] = []
    order = [name for name, _color, _members in QUERY_GROUPS]
    buckets: dict[str, list[str]] = {name: [] for name in order}
    loose: list[str] = []
    for name in names:
        group = stage_to_group.get(canonical(name)) or parent_group.get(parent_of(name))
        if group:
            buckets[group].append(name)
        else:
            loose.append(name)
    for group in order:
        members = buckets[group]
        if len(members) > 1:
            blocks.append((group, members))
        elif members:
            blocks.append((None, members))
    for name in loose:
        blocks.append((None, [name]))
    return blocks


def write_stage_chart(rows: list[dict], out_path: Path) -> Path:
    """Stacked bar of per-stage *mean* latency, one bar per run.

    Means for the same reason the table's share column uses them: disjoint
    stages sum to the root, so a stacked bar of means is the whole query and
    segment heights are directly comparable. Stacking p50s would draw a bar
    that adds up to nothing in particular.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # Union of stages across runs, ordered by the longest pipeline so backends
    # with different stage sets still stack in a consistent order.
    order: list[str] = []
    for row in sorted(rows, key=lambda r: -len(r.get("profile") or {})):
        for name in pipeline_stages(row.get("profile") or {}):
            if name not in order:
                order.append(name)

    labels = [f"{run_label(r)}\n{str(r.get('dataset','')).replace('beir/','')}" for r in rows]
    fig, ax = plt.subplots(figsize=(max(6.0, 2.2 * len(rows) + 3.0), 7.0))
    bottoms = [0.0] * len(rows)
    for name in order:
        heights = [
            float((r.get("profile") or {}).get(name, {}).get("mean_ms") or 0.0) for r in rows
        ]
        if not any(heights):
            continue
        ax.bar(
            labels, heights, bottom=bottoms, label=name,
            color=STAGE_COLOR.get(name, "#8a8a8a"),
            edgecolor="white", linewidth=0.4,
        )
        bottoms = [b + h for b, h in zip(bottoms, heights)]

    for x, total in enumerate(bottoms):
        ax.text(x, total, f"{total:.1f} ms", ha="center", va="bottom", fontsize=9)

    ax.set_ylabel("mean latency per query (ms)")
    ax.set_title("Query-time stage breakdown")
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(
        bbox_to_anchor=(1.01, 1.0), loc="upper left", fontsize=8,
        frameon=False, ncol=1 if len(order) <= 24 else 2,
    )
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path


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
    ap.add_argument(
        "--table",
        action="store_true",
        help=(
            "per-run stage table in pipeline order: p50 with share of total, "
            "p90 with tail multiple, nested under their phase"
        ),
    )
    ap.add_argument(
        "--pivot",
        choices=("code", "semantic", "flat"),
        default="code",
        help=(
            "how --table nests stages: code = structural phase from the stage "
            "name, pipeline-ordered (default); semantic = the cross-backend "
            "taxonomy, comparable but reordered; flat = no nesting"
        ),
    )
    ap.add_argument(
        "--chart",
        metavar="PATH",
        help="also write a stacked bar chart of mean stage latencies (PNG/SVG/PDF)",
    )
    ap.add_argument(
        "--width",
        type=int,
        default=None,
        help=(
            "console width. Defaults to the terminal's, or 200 when redirected — "
            "rich falls back to 80 columns off a tty, which silently truncates "
            "the stage table's numbers to '1…'."
        ),
    )
    args = ap.parse_args()
    try:
        sections = parse_sections(args.section)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))

    width = args.width
    if width is None and not sys.stdout.isatty():
        width = 200
    console = Console(width=width)
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

        if args.table:
            for r in prof_rows:
                console.print(stage_table(r, args.pivot))
            notes = [
                "  share = stage mean / total mean (means are additive across "
                "disjoint stages; percentiles are not).  tail = p90 / p50.",
                "  shaded rows aggregate the stages beneath them: mean and share are "
                "exact sums; p50/p90 show · unless the reducer recorded per-query "
                "phase totals.",
            ]
            if args.pivot == "semantic":
                notes.append(
                    "  --pivot semantic groups across the pipeline, so rows are no "
                    "longer in execution order."
                )
            for note in notes:
                console.print(Text(note, style="dim"))

        if args.chart:
            out = write_stage_chart(prof_rows, Path(args.chart))
            console.print(Text(f"  chart written to {out}", style="dim"))

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
