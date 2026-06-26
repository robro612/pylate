#!/usr/bin/env python
"""profflame - export profiling JSONL rows to Speedscope.

Benchmark artifacts currently store reduced per-stage timing summaries, not raw
span trees. This exporter maps those summaries to synthetic Speedscope evented
profiles: each selected run/profile section becomes one profile whose frames are
the selected percentile/mean stage buckets.

Future result rows also carry compact histogram bins for `_total` and every
stage. Use `--histogram-out` to write a companion Vega-Lite HTML report for
whole-query and per-stage distributions.

Run:
  uv run python scripts/profflame.py results.jsonl --out profile.speedscope.json
  uv run python scripts/profflame.py results.jsonl --profile query --detail stage
  uv run python scripts/profflame.py results.jsonl --histogram-out profile_hist.html
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from profview import (
    BUILD_STAGES,
    ENCODE_GROUPS,
    ENCODE_STAGES,
    QUERY_GROUPS,
    display_retrieval,
    group_segments,
    load_rows,
    parse_sections,
    query_profile_segments,
    run_label,
    short_model,
    stage_profile_segments,
)


def row_title(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).replace("beir/", "").replace("/test", "")
    parts = [short_model(row.get("model")), dataset, run_label(row)]
    if row.get("qps") is not None:
        parts.append(f"{float(row['qps']):.2f} qps")
    if row.get("ndcg@10") is not None:
        parts.append(f"nDCG@10 {float(row['ndcg@10']):.4f}")
    return " / ".join(parts)


def stage_stat(profile: dict[str, Any], name: str, stat: str) -> float:
    stage = profile.get(name)
    if not isinstance(stage, dict):
        return 0.0
    return float(stage.get(stat, 0.0) or 0.0)


def stage_segments(
    profile: dict[str, Any],
    stage_order: list[tuple[str, str]],
    stat: str,
) -> tuple[list[tuple[str, float, str]], float]:
    known = {name for name, _color in stage_order}
    segments = [
        (name, stage_stat(profile, name, stat), color)
        for name, color in stage_order
        if stage_stat(profile, name, stat) > 0
    ]
    extras = [
        name
        for name, value in profile.items()
        if not name.startswith("_") and name not in known and isinstance(value, dict)
    ]
    extra_colors = [
        "#5fd787",
        "#d7af00",
        "#af5fff",
        "#00afd7",
        "#ff8700",
        "#87afff",
        "#d75f87",
        "#87d7af",
    ]
    for i, name in enumerate(extras):
        value = stage_stat(profile, name, stat)
        if value > 0:
            segments.append((name, value, extra_colors[i % len(extra_colors)]))
    return segments, sum(value for _name, value, _color in segments)


def query_segments(
    row: dict[str, Any],
    detail: str,
    stat: str,
) -> tuple[list[tuple[str, float, str]], float, str]:
    if stat == "p50_ms":
        return query_profile_segments(row, detail)

    profile = row["profile"]
    if profile.get("_root") == "search":
        internal = [
            key
            for key, value in profile.items()
            if not key.startswith("_") and key != "unaccounted" and isinstance(value, dict)
        ]
        if not internal:
            total = float(profile.get("_total", {}).get(stat, 0.0) or 0.0)
            return [("search", total, "#2f6690")], total, "opaque rust search"

    from profview import QUERY_STAGES

    segments, total = stage_segments(profile, QUERY_STAGES, stat)
    if detail == "group":
        segments = group_segments(segments, QUERY_GROUPS)
    backend = profile.get("_maxsim_backend")
    note = f"maxsim:{','.join(backend)}" if backend else ""
    return segments, total, note


def encode_segments(
    profile: dict[str, Any],
    detail: str,
    stat: str,
) -> tuple[list[tuple[str, float, str]], float, str]:
    if stat == "p50_ms":
        return stage_profile_segments(profile, ENCODE_STAGES, detail, ENCODE_GROUPS)
    segments, total = stage_segments(profile, ENCODE_STAGES, stat)
    if detail == "group":
        segments = group_segments(segments, ENCODE_GROUPS)
    return segments, total, ""


def index_segments(row: dict[str, Any]) -> tuple[list[tuple[str, float, str]], float, str]:
    segments = [
        (label, float(row.get(key, 0.0) or 0.0) * 1000.0, color)
        for key, color, label in BUILD_STAGES
        if float(row.get(key, 0.0) or 0.0) > 0
    ]
    return segments, sum(value for _name, value, _color in segments), ""


def collect_blocks(
    rows: list[dict[str, Any]],
    sections: set[str],
    detail: str,
    stat: str,
) -> list[dict[str, Any]]:
    blocks: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        run = row_title(row)
        candidates: list[tuple[str, list[tuple[str, float, str]], float, str]] = []
        if "query" in sections and row.get("profile"):
            segments, total, note = query_segments(row, detail, stat)
            candidates.append(("query", segments, total, note))
        if "query_encode" in sections and row.get("query_encode_profile"):
            segments, total, note = encode_segments(row["query_encode_profile"], detail, stat)
            candidates.append(("query encode", segments, total, note))
        if "doc_encode" in sections and row.get("doc_encode_profile"):
            segments, total, note = encode_segments(row["doc_encode_profile"], detail, stat)
            candidates.append(("document encode", segments, total, note))
        if "index" in sections:
            segments, total, note = index_segments(row)
            if total > 0:
                candidates.append(("index", segments, total, note))
        for section, segments, total, note in candidates:
            if total <= 0:
                continue
            blocks.append(
                {
                    "idx": idx,
                    "run": run,
                    "section": section,
                    "segments": segments,
                    "total": total,
                    "note": note,
                    "index_type": row.get("index_type"),
                    "retrieval": display_retrieval(row),
                }
            )
    return blocks


def group_for_stage(stage: str) -> str:
    for group, _color, members in QUERY_GROUPS:
        if stage in members:
            return group
    for group, _color, members in ENCODE_GROUPS:
        if stage in members:
            return group
    return "stage"


class FrameTable:
    def __init__(self) -> None:
        self.frames: list[dict[str, str]] = []
        self.by_name: dict[str, int] = {}

    def frame(self, name: str) -> int:
        if name not in self.by_name:
            self.by_name[name] = len(self.frames)
            self.frames.append({"name": name})
        return self.by_name[name]


def append_interval(
    events: list[dict[str, Any]],
    frame: int,
    start: float,
    end: float,
) -> None:
    if end <= start:
        return
    events.append({"type": "O", "at": round(start, 6), "frame": frame})
    events.append({"type": "C", "at": round(end, 6), "frame": frame})


def block_to_profile(block: dict[str, Any], frames: FrameTable, detail: str) -> dict[str, Any]:
    events: list[dict[str, Any]] = []
    root_name = f"{block['run']} / {block['section']}"
    root_frame = frames.frame(root_name)
    section_frame = frames.frame(str(block["section"]))
    total = float(block["total"])

    events.append({"type": "O", "at": 0.0, "frame": root_frame})
    events.append({"type": "O", "at": 0.0, "frame": section_frame})
    cursor = 0.0
    open_group: str | None = None
    group_start = 0.0
    group_frame: int | None = None
    for stage, value, _color in block["segments"]:
        duration = float(value)
        if duration <= 0:
            continue
        start = cursor
        end = cursor + duration
        if detail == "stage":
            group = group_for_stage(stage)
            if open_group != group:
                if group_frame is not None:
                    events.append({"type": "C", "at": round(start, 6), "frame": group_frame})
                open_group = group
                group_start = start
                group_frame = frames.frame(group)
                events.append({"type": "O", "at": round(group_start, 6), "frame": group_frame})
            append_interval(events, frames.frame(stage), start, end)
        else:
            append_interval(events, frames.frame(stage), start, end)
        cursor = end
    if group_frame is not None:
        events.append({"type": "C", "at": round(cursor, 6), "frame": group_frame})
    events.append({"type": "C", "at": round(total, 6), "frame": section_frame})
    events.append({"type": "C", "at": round(total, 6), "frame": root_frame})

    return {
        "type": "evented",
        "name": root_name,
        "unit": "milliseconds",
        "startValue": 0,
        "endValue": round(total, 6),
        "events": events,
    }


def write_speedscope(
    path: Path,
    blocks: list[dict[str, Any]],
    *,
    source: Path,
    detail: str,
    stat: str,
) -> None:
    frames = FrameTable()
    profiles = [block_to_profile(block, frames, detail) for block in blocks]
    payload = {
        "$schema": "https://www.speedscope.app/file-format-schema.json",
        "exporter": "pylate scripts/profflame.py",
        "name": f"{source} ({stat}, {detail})",
        "activeProfileIndex": 0,
        "shared": {"frames": frames.frames},
        "profiles": profiles,
    }
    path.write_text(json.dumps(payload, indent=2) + "\n")


def iter_profile_sections(
    row: dict[str, Any],
    sections: set[str],
) -> list[tuple[str, dict[str, Any]]]:
    out: list[tuple[str, dict[str, Any]]] = []
    if "query" in sections and isinstance(row.get("profile"), dict):
        out.append(("query", row["profile"]))
    if "query_encode" in sections and isinstance(row.get("query_encode_profile"), dict):
        out.append(("query encode", row["query_encode_profile"]))
    if "doc_encode" in sections and isinstance(row.get("doc_encode_profile"), dict):
        out.append(("document encode", row["doc_encode_profile"]))
    return out


def histogram_records(rows: list[dict[str, Any]], sections: set[str]) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for idx, row in enumerate(rows, start=1):
        run = row_title(row)
        for section, profile in iter_profile_sections(row, sections):
            for stage, stats in profile.items():
                if stage.startswith("_") and stage != "_total":
                    continue
                if not isinstance(stats, dict):
                    continue
                hist = stats.get("histogram")
                if not isinstance(hist, dict):
                    continue
                edges = hist.get("edges_ms") or []
                counts = hist.get("counts") or []
                if len(edges) != len(counts) + 1:
                    continue
                stage_name = "total" if stage == "_total" else stage
                for left, right, count in zip(edges, edges[1:], counts):
                    records.append(
                        {
                            "row": idx,
                            "run": run,
                            "section": section,
                            "stage": stage_name,
                            "bin_start_ms": left,
                            "bin_end_ms": right,
                            "bin_mid_ms": (float(left) + float(right)) / 2.0,
                            "count": int(count),
                            "p50_ms": stats.get("p50_ms"),
                            "p90_ms": stats.get("p90_ms"),
                            "p95_ms": stats.get("p95_ms"),
                            "p99_ms": stats.get("p99_ms"),
                        }
                    )
    return records


def write_histogram_html(path: Path, records: list[dict[str, Any]], source: Path) -> None:
    data = json.dumps(records)
    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>PyLate profile histograms</title>
<script src="https://cdn.jsdelivr.net/npm/vega@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-lite@5"></script>
<script src="https://cdn.jsdelivr.net/npm/vega-embed@6"></script>
<style>
body {{
  margin: 0;
  padding: 20px;
  font: 13px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
  color: #1f2328;
}}
h1 {{ margin: 0 0 4px; font-size: 22px; }}
.subhead {{ color: #687076; margin-bottom: 16px; }}
</style>
</head>
<body>
<h1>PyLate profile histograms</h1>
<div class="subhead">source: {source}</div>
<div id="vis"></div>
<script>
const values = {data};
const spec = {{
  "$schema": "https://vega.github.io/schema/vega-lite/v5.json",
  "data": {{"values": values}},
  "resolve": {{"scale": {{"x": "independent", "y": "independent"}}}},
  "facet": {{
    "row": {{"field": "run", "type": "nominal", "title": null, "header": {{"labelLimit": 900}}}},
    "column": {{"field": "section", "type": "nominal", "title": null}}
  }},
  "spec": {{
    "width": 260,
    "height": 110,
    "mark": {{"type": "bar", "tooltip": true}},
    "encoding": {{
      "x": {{"field": "bin_mid_ms", "type": "quantitative", "title": "ms"}},
      "x2": {{"field": "bin_end_ms"}},
      "y": {{"field": "count", "type": "quantitative", "title": "count"}},
      "color": {{"field": "stage", "type": "nominal", "legend": {{"columns": 2}}}},
      "tooltip": [
        {{"field": "stage", "type": "nominal"}},
        {{"field": "bin_start_ms", "type": "quantitative", "format": ".3f"}},
        {{"field": "bin_end_ms", "type": "quantitative", "format": ".3f"}},
        {{"field": "count", "type": "quantitative"}},
        {{"field": "p50_ms", "type": "quantitative", "format": ".3f"}},
        {{"field": "p90_ms", "type": "quantitative", "format": ".3f"}},
        {{"field": "p95_ms", "type": "quantitative", "format": ".3f"}},
        {{"field": "p99_ms", "type": "quantitative", "format": ".3f"}}
      ]
    }}
  }}
}};
vegaEmbed("#vis", spec, {{"actions": true}});
</script>
</body>
</html>
"""
    path.write_text(doc)


def filtered_rows(args: argparse.Namespace) -> list[dict[str, Any]]:
    rows = load_rows(Path(args.file))
    if args.dataset:
        rows = [row for row in rows if args.dataset in str(row.get("dataset", ""))]
    if args.index:
        rows = [row for row in rows if args.index in str(row.get("index_type", ""))]
    if args.backend:
        rows = [row for row in rows if args.backend in str(row.get("maxsim_backend") or "")]
    if args.last:
        rows = rows[-args.last :]
    return rows


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", nargs="?", default="results.jsonl")
    ap.add_argument("--out", default="profile.speedscope.json")
    ap.add_argument("--histogram-out", help="write a companion Vega-Lite HTML histogram report")
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
        help="render semantic groups by default; use stage for raw low-level spans",
    )
    ap.add_argument(
        "--stat",
        choices=(
            "min_ms",
            "p10_ms",
            "p25_ms",
            "p50_ms",
            "p75_ms",
            "p90_ms",
            "p95_ms",
            "p99_ms",
            "max_ms",
            "mean_ms",
        ),
        default="p50_ms",
        help="profile statistic to render in Speedscope",
    )
    args = ap.parse_args()
    try:
        sections = parse_sections(args.section)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))

    rows = filtered_rows(args)
    blocks = collect_blocks(rows, sections, args.detail, args.stat)
    if not blocks:
        raise SystemExit("no profile blocks matched")

    out = Path(args.out)
    write_speedscope(
        out,
        blocks,
        source=Path(args.file),
        detail=args.detail,
        stat=args.stat,
    )
    print(f"wrote {out} ({len(blocks)} Speedscope profile(s))")

    if args.histogram_out:
        records = histogram_records(rows, sections)
        if not records:
            print("no histogram bins found; rerun benchmarks with the updated reducer")
        else:
            hist_out = Path(args.histogram_out)
            write_histogram_html(hist_out, records, Path(args.file))
            print(f"wrote {hist_out} ({len(records)} histogram bin(s))")


if __name__ == "__main__":
    main()
