"""Stage-level timing for retrieval pipelines.

A single span-tree abstraction is the lingua franca for timing across every
retrieval path — the Python token-index + maxsim path here, and the rust
end-to-end indices (tachiom / fastplaid / warp), whose own timing subtrees graft
in under a parent span as foreign children.

Design goals:

- **One schema.** A :class:`Span` is ``{name, dur_ns, device, count, meta,
  children}``. It serialises to JSON, aggregates by stage-path, and maps onto
  stacked-bar segments (top-level spans = bars, children = drill-down).
- **CUDA-correct.** A ``device="cuda"`` span is bracketed with
  ``torch.cuda.synchronize()`` (gated by ``cuda_sync``); without it a GPU span
  times kernel *launch*, not execution, and any kernel comparison is noise.
- **Granularity is encoded, not assumed.** ``Span.count`` records how many
  queries a span covers: ``1`` is latency-grade (eligible for p50/p90), ``>1``
  is batch-level and amortised-only (``dur_ns / count``). Reducers must refuse
  to put latency whiskers on a ``count > 1`` span.
- **Zero-cost when off.** A disabled profiler's :meth:`Profiler.span` is a no-op
  that neither times nor synchronises, so instrumentation can stay in hot paths.

The profiler threads no payload through call signatures: a backend wraps its
stages in :meth:`Profiler.span` and stashes the resulting tree on its own
``_last_profile`` attribute, which the harness reads after the call.
"""

from __future__ import annotations

import contextvars
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

try:  # torch is a hard dep of pylate, but keep the module importable without it.
    import torch

    _HAS_TORCH = True
except ImportError:  # pragma: no cover - torch is always present in practice
    _HAS_TORCH = False


@dataclass
class Span:
    """One timed stage and its nested sub-stages.

    Parameters
    ----------
    name
        Stage label, e.g. ``"encode"``, ``"maxsim"``, ``"rust_search"``.
    device
        ``"cpu"`` or ``"cuda"``. Determines whether a CUDA sync brackets the
        span when timed by a :class:`Profiler` with ``cuda_sync=True``.
    count
        Number of queries this span covers. ``1`` ⇒ latency-grade; ``>1`` ⇒
        batch-level, amortised-only (``dur_ns / count``).
    dur_ns
        Wall-clock duration in nanoseconds, filled in when the span closes.
    meta
        Free-form annotations (``{"backend": "flash", "n_candidates": 512}``).
    children
        Nested sub-stage spans, in start order.
    """

    name: str
    device: str = "cpu"
    count: int = 1
    dur_ns: int = 0
    meta: dict[str, Any] = field(default_factory=dict)
    children: list["Span"] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        """Recursively serialise to plain JSON-able dicts."""
        return {
            "name": self.name,
            "dur_ns": self.dur_ns,
            "device": self.device,
            "count": self.count,
            "meta": self.meta,
            "children": [c.to_dict() for c in self.children],
        }

    def walk(self, _prefix: str = "") -> Iterator[tuple[str, "Span"]]:
        """Yield ``(stage_path, span)`` for this span and all descendants.

        ``stage_path`` is ``/``-joined names (``"retrieve/maxsim"``), the key a
        reducer aggregates on.
        """
        path = f"{_prefix}/{self.name}" if _prefix else self.name
        yield path, self
        for child in self.children:
            yield from child.walk(path)


class Profiler:
    """Builds a :class:`Span` tree from nested :meth:`span` context managers.

    The current parent span is held in a :class:`contextvars.ContextVar`, so
    nesting is correct without threading state through call signatures and
    without leaking across threads/async tasks.

    Parameters
    ----------
    enabled
        When ``False``, :meth:`span` is a no-op (no timing, no sync, no tree).
    cuda_sync
        When ``True`` (default), bracket ``device="cuda"`` spans with
        ``torch.cuda.synchronize()`` so durations reflect kernel execution, not
        async launch. Has a real cost; turn off in production.
    """

    def __init__(self, *, enabled: bool = True, cuda_sync: bool = True) -> None:
        self.enabled = enabled
        self.cuda_sync = cuda_sync
        self.roots: list[Span] = []
        self._current: contextvars.ContextVar[Span | None] = contextvars.ContextVar(
            "pylate_profiler_current", default=None
        )

    def _should_sync(self, device: str) -> bool:
        return (
            self.cuda_sync
            and device == "cuda"
            and _HAS_TORCH
            and torch.cuda.is_available()
        )

    @contextmanager
    def span(
        self,
        name: str,
        device: str = "cpu",
        count: int = 1,
        **meta: Any,
    ) -> Iterator[Span | None]:
        """Time a stage. Yields the live :class:`Span` (or ``None`` if disabled).

        The yielded span is mutable mid-context — set ``span.meta[...]`` once a
        value is known (e.g. the resolved scoring backend), or stash the node on
        ``self._last_profile`` after a top-level span closes.
        """
        if not self.enabled:
            yield None
            return

        sync = self._should_sync(device)
        node = Span(name=name, device=device, count=count, meta=dict(meta))
        parent = self._current.get()
        if parent is None:
            self.roots.append(node)
        else:
            parent.children.append(node)
        token = self._current.set(node)

        if sync:
            torch.cuda.synchronize()
        start = time.perf_counter_ns()
        try:
            yield node
        finally:
            if sync:
                torch.cuda.synchronize()
            node.dur_ns = time.perf_counter_ns() - start
            self._current.reset(token)

    @property
    def root(self) -> Span | None:
        """The most recently opened top-level span tree, or ``None``."""
        return self.roots[-1] if self.roots else None

    def reset(self) -> None:
        """Drop all collected spans (start a fresh profiling session)."""
        self.roots = []
        self._current.set(None)


def _percentile(sorted_ms: list[float], q: float) -> float:
    """Nearest-rank percentile (q in [0,1]) of an already-sorted list."""
    if not sorted_ms:
        return float("nan")
    idx = min(len(sorted_ms) - 1, max(0, int(round(q * (len(sorted_ms) - 1)))))
    return sorted_ms[idx]


def _stats(ns: list[int]) -> dict[str, Any]:
    ms = sorted(x / 1e6 for x in ns)
    return {
        "p50_ms": round(_percentile(ms, 0.50), 4),
        "p90_ms": round(_percentile(ms, 0.90), 4),
        "mean_ms": round(sum(ms) / len(ms), 4) if ms else float("nan"),
        "n": len(ms),
    }


def reduce_stage_timings(roots: list[Span]) -> dict[str, Any]:
    """Reduce per-call span trees to per-stage timing stats (ms).

    ``roots`` is a list of top-level spans (one per query/batch — see a
    retriever's ``last_profile``). Aggregates the **direct children** of each
    root by name (one level → no double counting), plus the root total and the
    ``unaccounted`` self-time (root − Σchildren). Honours :class:`Span.count`:
    a stage is marked ``amortized`` (and gets no latency reading) when any of
    its spans cover more than one query.

    Returns ``{stage: {p50_ms, p90_ms, mean_ms, n, [amortized, count]}, ...}``
    plus ``_total`` and ``_maxsim_backend`` keys.
    """
    if not roots:
        return {}
    totals: list[int] = []
    unaccounted: list[int] = []
    by_stage: dict[str, list[int]] = {}
    stage_counts: dict[str, int] = {}
    backends: set[str] = set()
    for r in roots:
        totals.append(r.dur_ns)
        child_sum = 0
        for child in r.children:
            by_stage.setdefault(child.name, []).append(child.dur_ns)
            stage_counts[child.name] = max(stage_counts.get(child.name, 1), child.count)
            child_sum += child.dur_ns
            if child.name == "maxsim" and "backend" in child.meta:
                backends.add(str(child.meta["backend"]))
        unaccounted.append(max(r.dur_ns - child_sum, 0))

    out: dict[str, Any] = {
        "_root": roots[0].name,  # "retrieve" (token path) | "search" (E2E)
        "_total": _stats(totals),
        "unaccounted": _stats(unaccounted),
    }
    for name, ns in by_stage.items():
        s = _stats(ns)
        if stage_counts.get(name, 1) > 1:
            s["amortized"] = True
            s["count"] = stage_counts[name]
        out[name] = s
    if backends:
        out["_maxsim_backend"] = sorted(backends)
    return out


# A shared disabled profiler so instrumentation can do
# ``(self.profiler or NULL_PROFILER).span(...)`` without per-call None checks.
NULL_PROFILER = Profiler(enabled=False)


# ---------------------------------------------------------------------------
# Ambient profiler
#
# The orchestrator (a retriever) installs a profiler with ``use(...)``; deep
# helpers it calls (``rerank``, ``colbert_scores``) reach it with ``active()``
# and emit spans without the profiler being threaded through their signatures.
# Outside any ``use(...)`` block, ``active()`` returns the disabled
# NULL_PROFILER, so the instrumentation is a no-op.
# ---------------------------------------------------------------------------

_ACTIVE: contextvars.ContextVar[Profiler] = contextvars.ContextVar(
    "pylate_active_profiler", default=NULL_PROFILER
)


def active() -> Profiler:
    """The profiler installed by the nearest enclosing :func:`use`, else the
    disabled NULL_PROFILER."""
    return _ACTIVE.get()


@contextmanager
def use(profiler: Profiler | None) -> Iterator[Profiler]:
    """Install ``profiler`` as the ambient profiler for the duration of the
    block. ``None`` installs NULL_PROFILER (instrumentation stays a no-op)."""
    token = _ACTIVE.set(profiler or NULL_PROFILER)
    try:
        yield _ACTIVE.get()
    finally:
        _ACTIVE.reset(token)
