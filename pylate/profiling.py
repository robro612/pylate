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
import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Iterator

logger = logging.getLogger(__name__)

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

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Span":
        """Reconstruct a span tree from the JSON-able dict representation."""
        return cls(
            name=str(data["name"]),
            device=str(data.get("device", "cpu")),
            count=int(data.get("count", 1)),
            dur_ns=int(data.get("dur_ns", 0)),
            meta=dict(data.get("meta") or {}),
            children=[cls.from_dict(c) for c in data.get("children", [])],
        )

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


def _histogram(ms: list[float], bins: int = 20) -> dict[str, Any]:
    if not ms:
        return {"edges_ms": [], "counts": []}
    lo = ms[0]
    hi = ms[-1]
    if lo == hi:
        return {
            "edges_ms": [round(lo, 4), round(hi, 4)],
            "counts": [len(ms)],
        }
    width = (hi - lo) / bins
    counts = [0] * bins
    for value in ms:
        idx = min(bins - 1, int((value - lo) / width))
        counts[idx] += 1
    edges = [round(lo + i * width, 4) for i in range(bins + 1)]
    return {"edges_ms": edges, "counts": counts}


def _stats(ns: list[int]) -> dict[str, Any]:
    ms = sorted(x / 1e6 for x in ns)
    return {
        "min_ms": round(ms[0], 4) if ms else float("nan"),
        "p10_ms": round(_percentile(ms, 0.10), 4),
        "p25_ms": round(_percentile(ms, 0.25), 4),
        "p50_ms": round(_percentile(ms, 0.50), 4),
        "p75_ms": round(_percentile(ms, 0.75), 4),
        "p90_ms": round(_percentile(ms, 0.90), 4),
        "p95_ms": round(_percentile(ms, 0.95), 4),
        "p99_ms": round(_percentile(ms, 0.99), 4),
        "max_ms": round(ms[-1], 4) if ms else float("nan"),
        "mean_ms": round(sum(ms) / len(ms), 4) if ms else float("nan"),
        "n": len(ms),
        "histogram": _histogram(ms),
    }


def reduce_stage_timings(roots: list[Span]) -> dict[str, Any]:
    """Reduce per-call span trees to per-stage timing stats (ms).

    ``roots`` is a list of top-level spans (one per query/batch — see a
    retriever's ``last_profile``). Aggregates the **direct children** of each
    root by name (one level → no double counting), plus the root total and the
    ``unaccounted`` self-time (root − Σchildren). Honours :class:`Span.count`:
    a stage is marked ``amortized`` (and gets no latency reading) when any of
    its spans cover more than one query.

    Returns per-stage stats including percentiles and compact histogram bins,
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


class RustProfileUnavailableError(RuntimeError):
    """Stage profiling was requested but the backend cannot record it.

    Raised rather than silently degrading: a benchmark that runs to completion
    and reports only ``unaccounted`` looks like a *result*, and the cost of the
    run is already paid by the time anyone notices.
    """


#: Set to ``1`` to downgrade :func:`require_rust_profile` to a warning.
ALLOW_MISSING_ENV = "PYLATE_PROFILE_ALLOW_MISSING"


def require_rust_profile(backend: Any, *, name: str, build_hint: str) -> bool:
    """Check that ``backend`` can record Rust stage timings, or fail loudly.

    Call only when profiling was explicitly requested. Three cases are wrong in
    ways ``hasattr`` alone cannot separate:

    - the object exports no profiling hooks at all (a stock PyPI wheel);
    - it exports them but ``profile_supported()`` is ``False`` (built without
      ``--features profile``, so the hooks are compiled-out no-ops);
    - it exports the hooks but no ``profile_supported`` (a build from before
      that probe existed, which cannot be told apart from the case above).

    All three collect nothing, so all three raise
    :class:`RustProfileUnavailableError` — unless ``PYLATE_PROFILE_ALLOW_MISSING=1``
    is set, which logs a warning and returns ``False`` so a sweep can proceed
    with Python-side spans only.

    Parameters
    ----------
    backend
        Object exposing the hooks — a module (fast-plaid) or index (tachiom).
    name
        Backend name for the message, e.g. ``"fast-plaid"``.
    build_hint
        Shell command that produces a profiling build.

    Returns
    -------
    ``True`` when stages can be collected, ``False`` when the escape hatch is set.
    """
    has_hooks = hasattr(backend, "begin_profile") and hasattr(backend, "take_profile")
    probe = getattr(backend, "profile_supported", None)
    if has_hooks and probe is not None and probe():
        return True

    if not has_hooks:
        reason = f"the installed {name} exports no profiling hooks (stock wheel?)"
    elif probe is None:
        reason = (
            f"the installed {name} predates the profile_supported() probe, so a "
            "feature-off build cannot be ruled out"
        )
    else:
        reason = f"{name} was built without the `profile` Cargo feature"

    message = (
        f"Stage profiling requested but {reason}; no Rust stages would be "
        f"recorded and the run would report only `unaccounted`.\n"
        f"Rebuild with:\n    {build_hint}\n"
        f"Note that a later `uv sync` / `uv run` rebuilds editable installs with "
        f"default features and silently reverts this; see docs/profiling.md.\n"
        f"Set {ALLOW_MISSING_ENV}=1 to downgrade this to a warning."
    )
    if os.environ.get(ALLOW_MISSING_ENV) == "1":
        logger.warning("%s", message)
        return False
    raise RustProfileUnavailableError(message)


def spans_from_rust(raw: list[Any], count: int = 1) -> list[Span]:
    """Normalise a rust backend's ``take_profile()`` payload into flat spans.

    The shared ``stage-profile`` crate returns a **flat** list of disjoint stage
    dicts; a legacy backend may instead return a single nested ``search`` root,
    which is peeled so both shapes reduce identically.

    Backends that loop over queries inside one profiled call (fast-plaid's
    serial ``search_many``) emit one sample per stage *per query*. Those repeats
    are coalesced by name — durations and numeric metadata summed, ``n_calls``
    recorded — and ``count`` marks the result as covering more than one query so
    :func:`reduce_stage_timings` treats it as amortised rather than
    latency-grade. With one query per call (``search.e2e_profile_batch_size=1``)
    coalescing is a no-op.

    Parameters
    ----------
    raw
        Sequence of Span-shaped dicts from the backend's ``take_profile()``.
    count
        Number of queries the drained samples cover.
    """
    if (
        len(raw) == 1
        and isinstance(raw[0], dict)
        and raw[0].get("name") == "search"
        and raw[0].get("children")
    ):
        spans = [Span.from_dict(dict(c)) for c in raw[0]["children"]]
    else:
        spans = [Span.from_dict(dict(s)) for s in raw]

    merged: dict[str, Span] = {}
    for span in spans:
        span.count = count
        existing = merged.get(span.name)
        if existing is None:
            merged[span.name] = span
            continue
        existing.dur_ns += span.dur_ns
        existing.meta["n_calls"] = existing.meta.get("n_calls", 1) + 1
        for key, value in span.meta.items():
            if isinstance(value, (int, float)) and isinstance(
                existing.meta.get(key), (int, float)
            ):
                existing.meta[key] += value
    return list(merged.values())


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
