"""Tests for the pure-Python stage profiler (pylate.profiling)."""

import logging
import time

import pytest

from pylate.profiling import (
    ALLOW_MISSING_ENV,
    NULL_PROFILER,
    Profiler,
    RustProfileUnavailableError,
    Span,
    require_rust_profile,
    spans_from_rust,
)


def test_span_to_dict_roundtrip():
    s = Span(
        name="maxsim", device="cuda", count=4, dur_ns=123, meta={"backend": "flash"}
    )
    s.children.append(Span(name="gather", dur_ns=10))
    d = s.to_dict()
    assert d["name"] == "maxsim"
    assert d["device"] == "cuda"
    assert d["count"] == 4
    assert d["dur_ns"] == 123
    assert d["meta"] == {"backend": "flash"}
    assert d["children"][0]["name"] == "gather"


def test_nesting_builds_tree():
    prof = Profiler()
    with prof.span("retrieve") as root:
        with prof.span("index_lookup"):
            pass
        with prof.span("maxsim", device="cuda", backend="torch"):
            pass
    assert len(prof.roots) == 1
    assert prof.root is root
    names = [c.name for c in root.children]
    assert names == ["index_lookup", "maxsim"]
    assert root.children[1].meta["backend"] == "torch"


def test_count_default_and_override():
    prof = Profiler()
    with prof.span("encode"):
        pass
    with prof.span("centroid_score", count=64):
        pass
    assert prof.roots[0].count == 1
    assert prof.roots[1].count == 64


def test_timing_is_positive():
    prof = Profiler()
    with prof.span("work"):
        time.sleep(0.005)
    assert prof.root.dur_ns >= 5_000_000  # >= 5ms


def test_walk_yields_stage_paths():
    prof = Profiler()
    with prof.span("retrieve"):
        with prof.span("maxsim"):
            pass
    paths = [p for p, _ in prof.root.walk()]
    assert paths == ["retrieve", "retrieve/maxsim"]


def test_disabled_profiler_is_noop():
    prof = Profiler(enabled=False)
    with prof.span("work") as node:
        assert node is None
    assert prof.roots == []


def test_null_profiler_shared_and_silent():
    with NULL_PROFILER.span("anything", device="cuda") as node:
        assert node is None
    assert NULL_PROFILER.roots == []


def test_cuda_span_without_gpu_does_not_error():
    # device="cuda" on a CPU-only host: no sync attempted, still times.
    prof = Profiler(cuda_sync=True)
    with prof.span("maxsim", device="cuda"):
        pass
    assert prof.root.device == "cuda"
    assert prof.root.dur_ns >= 0


class _Backend:
    """Stand-in for a rust extension module / index object."""

    def __init__(self, hooks=True, supported=None):
        if hooks:
            self.begin_profile = lambda: None
            self.take_profile = lambda: []
        if supported is not None:
            self.profile_supported = lambda: supported


def test_require_rust_profile_accepts_a_profiling_build():
    assert require_rust_profile(
        _Backend(supported=True), name="fast-plaid", build_hint="maturin develop ..."
    )


@pytest.mark.parametrize(
    ("backend", "expected"),
    [
        (_Backend(hooks=False), "exports no profiling hooks"),
        (_Backend(supported=False), "without the `profile` Cargo feature"),
        (_Backend(), "predates the profile_supported() probe"),
    ],
)
def test_require_rust_profile_raises_instead_of_silently_collecting_nothing(
    backend, expected
):
    with pytest.raises(RustProfileUnavailableError) as excinfo:
        require_rust_profile(backend, name="fast-plaid", build_hint="maturin develop X")
    message = str(excinfo.value)
    assert expected in message
    # The message has to be actionable: how to fix, and the uv-sync footgun.
    assert "maturin develop X" in message
    assert "uv sync" in message


def test_require_rust_profile_escape_hatch_warns_instead(monkeypatch, caplog):
    monkeypatch.setenv(ALLOW_MISSING_ENV, "1")
    with caplog.at_level(logging.WARNING, logger="pylate.profiling"):
        assert (
            require_rust_profile(
                _Backend(supported=False),
                name="tachiom",
                build_hint="maturin develop X",
            )
            is False
        )
    assert "without the `profile` Cargo feature" in caplog.text


def _rust_stage(name, dur_ns, device="cpu", **meta):
    """A take_profile() entry as the stage-profile PyO3 bridge emits it."""
    return {
        "name": name,
        "dur_ns": dur_ns,
        "device": device,
        "count": 1,
        "meta": meta,
        "children": [],
    }


def test_spans_from_rust_flat_payload():
    spans = spans_from_rust(
        [
            _rust_stage("centroid_score", 100, device="cuda"),
            _rust_stage("exact_score", 300, device="cuda", n_rerank=64.0),
        ]
    )
    assert [s.name for s in spans] == ["centroid_score", "exact_score"]
    assert [s.dur_ns for s in spans] == [100, 300]
    assert spans[1].device == "cuda"
    assert spans[1].meta["n_rerank"] == 64.0
    assert all(s.count == 1 for s in spans)


def test_spans_from_rust_peels_legacy_nested_root():
    nested = {
        "name": "search",
        "dur_ns": 500,
        "device": "cpu",
        "count": 1,
        "meta": {},
        "children": [_rust_stage("rerank", 400)],
    }
    spans = spans_from_rust([nested])
    assert [s.name for s in spans] == ["rerank"]
    assert spans[0].dur_ns == 400


def test_spans_from_rust_coalesces_repeated_stages():
    # Two queries in one profiled call: each stage appears once per query.
    spans = spans_from_rust(
        [
            _rust_stage("exact_score", 100, n_rerank=10.0),
            _rust_stage("final_topk", 20),
            _rust_stage("exact_score", 300, n_rerank=30.0),
            _rust_stage("final_topk", 40),
        ],
        count=2,
    )
    by_name = {s.name: s for s in spans}
    assert set(by_name) == {"exact_score", "final_topk"}
    # Durations and numeric metadata sum; count marks the result amortised.
    assert by_name["exact_score"].dur_ns == 400
    assert by_name["exact_score"].meta["n_rerank"] == 40.0
    assert by_name["exact_score"].meta["n_calls"] == 2
    assert by_name["final_topk"].dur_ns == 60
    assert all(s.count == 2 for s in spans)


def test_reduce_marks_coalesced_stages_amortized():
    from pylate.profiling import reduce_stage_timings

    root = Span(name="search", dur_ns=1000, count=2)
    root.children = spans_from_rust(
        [_rust_stage("exact_score", 300), _rust_stage("exact_score", 500)], count=2
    )
    out = reduce_stage_timings([root])
    assert out["exact_score"]["amortized"] is True
    assert out["exact_score"]["count"] == 2
    assert out["exact_score"]["mean_ms"] == round(800 / 1e6, 4)


def test_reset_clears_session():
    prof = Profiler()
    with prof.span("a"):
        pass
    prof.reset()
    assert prof.roots == []
    with prof.span("b"):
        pass
    assert [r.name for r in prof.roots] == ["b"]
