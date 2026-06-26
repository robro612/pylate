#!/usr/bin/env python
"""profflame - write an HTML flamegraph-style view of profiling JSONL rows.

The benchmark artifacts currently store reduced per-stage timing summaries, not
raw span trees. This exporter therefore renders an aggregate icicle/flamegraph:
each selected run/profile section gets a total bar and its p50 stage segments.

Run:
  uv run python scripts/profflame.py results.jsonl --out profile.html
  uv run python scripts/profflame.py results.jsonl --profile query --detail stage
"""

from __future__ import annotations

import argparse
import html
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


def esc(value: object) -> str:
    return html.escape(str(value), quote=True)


def fmt_duration(ms: float) -> str:
    if ms >= 1000:
        return f"{ms / 1000:.3f} s"
    return f"{ms:.2f} ms"


def row_title(row: dict[str, Any]) -> str:
    dataset = str(row.get("dataset", "")).replace("beir/", "").replace("/test", "")
    quality = row.get("ndcg@10")
    qps = row.get("qps")
    parts = [
        short_model(row.get("model")),
        dataset,
        run_label(row),
    ]
    if qps is not None:
        parts.append(f"{float(qps):.2f} qps")
    if quality is not None:
        parts.append(f"nDCG@10 {float(quality):.4f}")
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


def render_segment(name: str, value: float, color: str, total: float, left: float) -> str:
    width = 100.0 * value / total if total > 0 else 0.0
    title = f"{name}: {fmt_duration(value)} ({width:.1f}%)"
    label = name if width >= 6.0 else ""
    return (
        f'<div class="seg" style="left:{left:.5f}%;width:{width:.5f}%;'
        f'background:{esc(color)}" title="{esc(title)}">'
        f'<span>{esc(label)}</span></div>'
    )


def render_html(
    path: Path,
    blocks: list[dict[str, Any]],
    *,
    source: Path,
    detail: str,
    stat: str,
) -> None:
    max_total = max((block["total"] for block in blocks), default=1.0)
    body: list[str] = []
    for block in blocks:
        scale_width = 100.0 * block["total"] / max_total if max_total > 0 else 0.0
        body.append('<section class="block">')
        body.append(
            '<div class="meta">'
            f'<div><strong>{esc(block["section"])}</strong> '
            f'<span class="muted">row {block["idx"]}</span></div>'
            f'<div>{esc(block["run"])}</div>'
            f'<div class="muted">{esc(block["note"])}</div>'
            "</div>"
        )
        body.append(
            f'<div class="root" title="total: {esc(fmt_duration(block["total"]))}">'
            f'<div class="root-scale" style="width:{scale_width:.5f}%"></div>'
            f'<span>{esc(fmt_duration(block["total"]))}</span>'
            "</div>"
        )
        body.append('<div class="flame">')
        left = 0.0
        for name, value, color in block["segments"]:
            pct = 100.0 * value / block["total"] if block["total"] > 0 else 0.0
            body.append(render_segment(name, value, color, block["total"], left))
            left += pct
        body.append("</div>")
        body.append("</section>")

    doc = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>profile flamegraph</title>
<style>
:root {{
  color-scheme: light;
  --bg: #f7f7f4;
  --ink: #1f2328;
  --muted: #687076;
  --line: #d8d8d0;
  --panel: #ffffff;
}}
* {{ box-sizing: border-box; }}
body {{
  margin: 0;
  padding: 24px;
  background: var(--bg);
  color: var(--ink);
  font: 13px/1.35 ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
}}
h1 {{
  margin: 0 0 4px;
  font-size: 22px;
  font-weight: 700;
}}
.subhead {{
  color: var(--muted);
  margin-bottom: 20px;
}}
.block {{
  background: var(--panel);
  border: 1px solid var(--line);
  border-radius: 8px;
  padding: 12px;
  margin: 0 0 12px;
  box-shadow: 0 1px 2px rgb(31 35 40 / 0.04);
}}
.meta {{
  display: grid;
  grid-template-columns: minmax(120px, 180px) minmax(280px, 1fr) minmax(120px, 0.8fr);
  gap: 12px;
  align-items: baseline;
  margin-bottom: 8px;
}}
.muted {{ color: var(--muted); }}
.root {{
  position: relative;
  height: 18px;
  border: 1px solid var(--line);
  background: #efefea;
  border-radius: 4px;
  margin-bottom: 4px;
  overflow: hidden;
}}
.root-scale {{
  position: absolute;
  inset: 0 auto 0 0;
  background: #d8dee4;
}}
.root span {{
  position: relative;
  display: inline-block;
  padding: 1px 6px;
  font-weight: 650;
}}
.flame {{
  position: relative;
  height: 34px;
  border-radius: 5px;
  overflow: hidden;
  background: #ecece7;
  border: 1px solid var(--line);
}}
.seg {{
  position: absolute;
  top: 0;
  bottom: 0;
  min-width: 1px;
  border-right: 1px solid rgb(255 255 255 / 0.7);
  color: #111;
  overflow: hidden;
  white-space: nowrap;
}}
.seg span {{
  display: block;
  padding: 8px 6px;
  overflow: hidden;
  text-overflow: ellipsis;
  font-weight: 650;
  text-shadow: 0 1px 0 rgb(255 255 255 / 0.45);
}}
@media (max-width: 850px) {{
  body {{ padding: 12px; }}
  .meta {{ grid-template-columns: 1fr; gap: 3px; }}
}}
</style>
</head>
<body>
<h1>profile flamegraph</h1>
<div class="subhead">source: {esc(source)} / detail: {esc(detail)} / statistic: {esc(stat)}</div>
{''.join(body)}
</body>
</html>
"""
    path.write_text(doc)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("file", nargs="?", default="results.jsonl")
    ap.add_argument("--out", default="profile_flamegraph.html")
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
        choices=("p50_ms", "p90_ms", "mean_ms"),
        default="p50_ms",
        help="profile statistic to render",
    )
    args = ap.parse_args()
    try:
        sections = parse_sections(args.section)
    except argparse.ArgumentTypeError as exc:
        ap.error(str(exc))

    source = Path(args.file)
    rows = load_rows(source)
    if args.dataset:
        rows = [row for row in rows if args.dataset in str(row.get("dataset", ""))]
    if args.index:
        rows = [row for row in rows if args.index in str(row.get("index_type", ""))]
    if args.backend:
        rows = [row for row in rows if args.backend in str(row.get("maxsim_backend") or "")]
    if args.last:
        rows = rows[-args.last :]

    blocks = collect_blocks(rows, sections, args.detail, args.stat)
    if not blocks:
        raise SystemExit("no profile blocks matched")

    out = Path(args.out)
    render_html(out, blocks, source=source, detail=args.detail, stat=args.stat)
    print(f"wrote {out} ({len(blocks)} block(s))")


if __name__ == "__main__":
    main()
