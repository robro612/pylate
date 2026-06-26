"""Tests for the pure-Python stage profiler (pylate.profiling)."""

import time

from pylate.profiling import NULL_PROFILER, Profiler, Span


def test_span_to_dict_roundtrip():
    s = Span(name="maxsim", device="cuda", count=4, dur_ns=123, meta={"backend": "flash"})
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


def test_reset_clears_session():
    prof = Profiler()
    with prof.span("a"):
        pass
    prof.reset()
    assert prof.roots == []
    with prof.span("b"):
        pass
    assert [r.name for r in prof.roots] == ["b"]
