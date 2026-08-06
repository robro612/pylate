# Stage-span profiling

Experiment-facing stage timing for retrieval (Python + optional Rust). Toggle with
`search.profile=true` in Hydra / `benchmark_indexes.py`. View with:

```bash
uv run python scripts/profview.py results.jsonl --section query --detail stage
```

## Design

- Python: `pylate.profiling.Profiler` + ambient `use` / `active().span(...)`.
- Rust (local forks): optional Cargo feature `profile` depending on
  [`crates/stage-profile`](../crates/stage-profile). `stage!("name")` is a
  compile-time no-op when the feature is off.
- No profile payloads in search signatures: `begin_profile()` → normal search →
  `take_profile()`.

## Enable Rust stages

Both local Rust backends expose `profile_supported()` / `begin_profile()` /
`take_profile()`. The hooks are exported unconditionally; only
`profile_supported()` reports whether the build can actually record anything.
With `search.profile=true`, a build that cannot raises
`RustProfileUnavailableError` before the run starts, rather than producing a
breakdown of pure `unaccounted`. Set `PYLATE_PROFILE_ALLOW_MISSING=1` to
downgrade that to a warning and keep Python-side spans only.

### Environments

Profiling has its own environments — `.venv-cu130-profile` and
`.venv-cu126-profile` — because a profiling build CUDA-syncs every stage and is
therefore the wrong artifact for a throughput number.
[docs/environments.md](environments.md) is the authority on all four; build with:

```bash
srunl40s --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu130-profile
srunv100 --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu126-profile
```

and run with:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cu130-profile uv run --no-sync \
    python scripts/benchmark_indexes.py ... search.profile=true
```

### Why not just a flag on `uv sync`

`profile` is deliberately **not** in either checkout's `[tool.maturin] features`.
That list applies to every build of the tree, including published wheels, so
adding it there would make CUDA-synced profiling builds the default artifact.

Passing the feature per-sync is possible but not something to rely on by hand:

```bash
uv sync --config-settings-package fast-plaid:maturin.build-args='--features pyo3/extension-module,profile' \
        --config-settings-package tachiom:maturin.build-args='--features python,profile'
```

It has to be per-package, because the two forks need different feature sets and
`MATURIN_PEP517_ARGS` is one global string; and the full list must be repeated,
because maturin's CLI `--features` replaces rather than merges
`[tool.maturin] features`. `scripts/sync_env.sh profile` encapsulates all of it.

The reason not to run bare `uv sync`/`uv run` against a profiling environment is
that uv re-syncs *and prunes*: anything outside the default set gets uninstalled,
and the forks get rebuilt with default features. Two settings blunt this —
`[tool.uv] default-groups` keeps the working set installed, and `--no-sync`
skips the whole step:

```bash
UV_PROJECT_ENVIRONMENT=.venv-profile uv run --no-sync \
    python scripts/benchmark_indexes.py ... search.profile=true search.e2e_profile_batch_size=1
```

If a profiled run starts raising `RustProfileUnavailableError` out of nowhere, a
bare `uv run` rebuilt the forks without the feature. That is the error doing its
job; re-run `scripts/sync_env.sh profile`.

### Manual rebuilds

`maturin develop` into a specific environment, with `LIBTORCH` pointed at that
environment's torch (see the Environments note on why `LIBTORCH_USE_PYTORCH`
doesn't work here):

```bash
export VIRTUAL_ENV=/exp/rjha/pylate-pgc/.venv-profile
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LIBTORCH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch"
export LIBTORCH_BYPASS_VERSION_CHECK=1

cd /exp/rjha/tachiom && TMPDIR=/exp/rjha/tachiom/.cargo-cache \
    maturin develop --features python,profile --release
cd /exp/rjha/fast-plaid && maturin develop --features profile --release
```

`xtr-warp-rs` has no `uv.sources` entry yet and is installed by hand:
`uv pip install -e third_party/xtr-warp-rs`.

### Stages

| Backend | Stages |
|---|---|
| tachiom | `coarse_accumulate`, `candidate_select`, `rerank` |
| fastplaid | `query_prepare`, `centroid_score`, `ivf_select`, `ivf_lookup`, `candidate_dedup`, `candidate_lengths`, `approx_score`, `approx_topk`, `rerank_lengths`, `exact_score`, `final_topk`, `result_materialize` |

Stages are recorded as **disjoint** segments, so durations sum without double
counting and `unaccounted` is real self-time. The shared collector is flat: it
cannot express nesting, so a sub-stage inside another stage (e.g. breaking
`exact_score` into `exact_lookup` / `residual_decompress`) would overlap its
parent and inflate the total. That detail is deferred until the crate grows a
span tree.

### CUDA correctness

fast-plaid queues Torch ops asynchronously, so its guards synchronise the CUDA
device on entry and again before recording — otherwise a stage times kernel
*launch* and all the time piles onto whichever stage happens to force a sync.
Like `Profiler(cuda_sync=True)` on the Python side, this makes a profiled run
slower than a production one; compare stage shares, not absolute QPS. Tachiom is
CPU-only and does not sync.

### Threading caveat

The sample buffer is **process-global**, so `take_profile()` drains stages
recorded on worker threads too. That is what makes this work at all: fast-plaid
hands the Rust search to a `ThreadPoolExecutor` worker even for a single device
(and to joblib with `prefer="threads"` for CPU bulk search), and tachiom's
`batch_search` fans out over Rayon — a thread-local buffer drained empty or
partial in both cases.

What a global buffer does *not* recover is per-query attribution. Stages arrive
one copy per query, interleaved across workers, so:

- `spans_from_rust` coalesces repeats by name — summing durations and numeric
  metadata, recording `n_calls` — and marks them amortised via `count`.
- A multi-query profiled call therefore yields per-batch totals, not
  latency-grade percentiles.
- Concurrent profiled searches from different threads would land in one buffer;
  don't run two at once.

Either way, `search.e2e_profile_batch_size=1` is what you want for p50/p90. See
[`crates/stage-profile/README.md`](../crates/stage-profile/README.md).

## Rust metadata

Attach arbitrary numerical fields on a live guard (no-op when profiling is off):

```rust
let mut _stage = stage!("rerank");                 // tachiom (CPU)
let mut _stage = stage!("exact_score", device);    // fast-plaid (syncs CUDA)
_stage.set("n_candidates", candidates.len() as f64);
// ... work ...
```

Values land in the Span `meta` dict drained by `take_profile()`.