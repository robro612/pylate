# stage-profile

Flat RAII stage timers for PyLate’s Rust index backends (optional path dependency
behind a Cargo `profile` feature).

```rust
stage_profile::begin();
{
    let mut _stage = stage_profile::StageGuard::new("rerank");
    _stage.set("n_candidates", n as f64);
    // ... work ...
}
let samples = stage_profile::take(); // Vec<StageSample>
```

`StageGuard::new_on("exact_score", "cuda")` labels the sample's device. The
crate only carries the label — it has no accelerator dependency and never
synchronises, so a GPU consumer must bracket its guards with a device sync of
its own (see fast-plaid's `stage_shim`), or the sample times kernel *launch*
rather than execution.

## Threading

The session flag and sample buffer are **process-global**, so `begin()` on the
caller covers stages recorded on worker threads and `take()` drains all of them.
This is required in practice: fast-plaid dispatches the Rust search through a
`ThreadPoolExecutor` even for a single device, and Tachiom’s `batch_search` fans
out over Rayon. An earlier thread-local buffer silently returned empty or
partial profiles in both cases.

| Mode | OK? |
|---|---|
| One query per begin→search→take | Yes — latency-grade |
| Search dispatched to a pool / Rayon worker | Yes |
| `batch_size > 1` | Yes, but amortised only (see below) |
| Two profiled searches concurrently | **No** — one global session |

## Repeated samples

Nothing deduplicates: a stage that runs once per query in a multi-query call
pushes one sample per query, in completion order. Consumers coalesce by name —
PyLate’s `pylate.profiling.spans_from_rust` sums durations and numeric metadata,
records `n_calls`, and sets `count` so the reducer reports the stage as
amortised instead of putting latency whiskers on it.

For p50/p90 per stage, profile one query per call
(PyLate: `search.e2e_profile_batch_size=1`).

## Nesting

The collector is flat and cannot represent a span tree, so stages must be
recorded as **disjoint** segments. A guard opened inside another guard’s scope
overlaps it and the consumer double counts. Splitting a stage into finer ones
means dropping the outer guard, not nesting inside it.
