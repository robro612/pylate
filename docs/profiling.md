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

| Backend | Where | Stages |
|---|---|---|
| tachiom | Rust | `coarse_accumulate`, `candidate_select`, `rerank/{encoder,seed,stream,select}` (`rerank/score_full` on the non-early-exit path) |
| tachiom | Python | `query_pack`, `result/convert`, `overhead/{drain,dispatch}` |
| fastplaid | Rust | `query/{layout,prepare}`, `centroid_score`, `ivf/{select,lookup}`, `candidate/{dedup,lengths}`, `approx/{plan,lookup,gather,pad,reduce,merge,topk}`, `exact/{lengths,plan,lookup,residual_decompress,pad,matmul,reduce,merge}`, `final_topk`, `result/materialize` |
| fastplaid | Python | `id_map_load`, `query/convert`, `result/convert`, `overhead/{drain,dispatch}` |

Stages named `parent/leaf` roll up into a phase — see [Hierarchy](#hierarchy).
Singletons (`centroid_score`, `final_topk`, `query_pack`) stay flat.

#### Counters

Every stage that touches a candidate set records its size, so the funnel is
readable end to end without re-deriving it from parameters:

| backend | funnel |
|---|---|
| tachiom | `coarse_accumulate.n_docs_touched` → `candidate_select.n_candidates` → `rerank/seed.n_scored` + `rerank/stream.n_scored` → `rerank/select.k` |
| fastplaid | `ivf/select.n_cells` → `ivf/lookup.n_ids` → `candidate/dedup.n_candidates` → `approx/topk.n_rerank` → `exact/*.n_docs` → `final_topk.top_k` |

Two pairs are worth reading together. `ivf/lookup.n_ids` against
`candidate/dedup.n_candidates` is the duplication factor of probing wide, which
`n_candidates` alone hides. And `exact/pad.n_embeddings` against
`exact/matmul.n_padded_tokens × n_docs` is the padding waste — `exact/matmul`
scales in the padded rectangle, not in the real token count.

`rerank/stream` additionally carries `n_admitted` / `n_skipped` /
`early_terminated`, which is what explains its tail. No guard sits inside the
scoring loop: the *sample machinery* (String allocation plus the global mutex,
~21 µs per drained sample) would cost more than the stage — 1274 samples ≈ 26 ms
against a 37 ms stage. The clock read itself is negligible.

Counters roll up to a phase by **max, not sum**: a phase's leaves mostly operate
on the same set, so summing `n_docs` over `exact/lookup`, `exact/pad` and
`exact/matmul` would report three times the documents actually reranked.

`overhead/dispatch` is derived by subtraction (wrapper wall-clock minus the sum
of Rust stages), not measured directly. A span wrapped around the call would
enclose every Rust stage and double count.

The `approx/*` / `exact/*` stages inside the chunk closures fire once per chunk;
`spans_from_rust` coalesces them by name and records `n_calls`. The Python stages
are emitted by the PyLate index wrapper and sit alongside the Rust ones as
siblings, so `unaccounted` is now genuinely unmeasured time rather than the whole
Python layer.

Stages are recorded as **disjoint** segments, so durations sum without double
counting and `unaccounted` is real self-time. The shared collector is flat and
stays that way: a guard nested inside another guard would overlap its parent and
inflate the total.

### Hierarchy

Nesting is expressed in the *name*, not in the collector — a stage called
`parent/leaf` declares its phase while remaining a disjoint sibling of every
other stage. `reduce_stage_timings` then emits a `_parents` entry per phase.

The parent's distribution is accumulated **per query and only then reduced**,
because percentiles do not add: a phase's p50 is the median of the per-query
sums of its leaves, which cannot be recovered from the leaves' own p50s. Means
do add, so a phase's mean and share are exact either way — but p90 and the tail
multiple are most of the reason to have the row at all.

Two levels, owned in two different places on purpose:

| level | owner | why there |
|---|---|---|
| structural (`rerank/seed` ⊂ `rerank`) | the span marker, in code | containment is a fact only the code knows, and a post-hoc name map silently drops every stage added after it was written |
| semantic (`exact_matmul` ≈ `rerank_stream` ≈ "scoring") | `QUERY_GROUPS` in `scripts/profview.py` | no single fork can classify another backend's stages, and the taxonomy must be re-cuttable over archived results without re-running |

The semantic map keys on *parents* where it can, so a new `rerank/*` leaf
inherits its bucket instead of falling out unclassified.

Stage names predating the convention carry their phase as an underscore prefix
(`rerank_seed`, `approx_lookup`); profview reconstructs the pivot from those so
existing artifacts render, but such phases show `·` for p50/p90 since the
per-query sums were never recorded. Prefer `parent/leaf` for new stages.

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