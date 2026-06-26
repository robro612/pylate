# Unified retrieval profiling

Status: **Python/token-path implemented; rust internals pending**.
Goal: high-quality, stage-level timing across every retrieval path so we can
build stacked-bar comparisons across methods and find real bottlenecks.

The constraint that shapes everything: timing must be **principled and
extensible**, not ad-hoc `perf_counter()` calls plus a new payload threaded up
through every layer. We get there with **one schema, one seam, adapters at the
edges**.

---

## Current status (2026-06-26)

Implemented:

- `pylate/profiling.py`: a pure-Python `Span` / `Profiler` tree, CUDA-synced
  stage timing, a reducer, and an ambient context manager so deep helpers can
  emit spans without profile arguments or return payloads.
- Token-path retrieval profiling for ScaNN/Voyager + ColBERT:
  `index_lookup`, `candidate_dedup`, `gather`, `reshape_inputs`,
  `prepare_tensors`, `device_transfer`, `maxsim`, `topk`,
  `result_materialize`, and `unaccounted`.
- MaxSim backend selection via `search.maxsim_backend` / `maxsim_backend=`
  (`torch`, `flash`, `lik`, `auto`) and backend metadata in result rows.
- Query/document encode profiling when the encode stages are explicitly run
  with `search.profile=true`. Query encode profiling runs at batch size 1 for
  latency-grade per-query stats. Cached query loading is also attributed
  (`load_cache_arrays`, `reconstruct_query_tensors`) when retrieve reuses cache.
- Coarse end-to-end index profiling via `search.e2e_profile_batch_size`. Setting
  it to `1` records one outer search span per query for Tachiom/WARP/PLAID-like
  paths, but rust-internal stages are not broken out yet.
- Terminal visualization in `scripts/profview.py` with separate high-contrast
  palettes for query-time, encode, and build sections.
- Tachiom benchmark config now uses the ColBERT retriever label
  (`conf/eval/index/tachiom.yaml`) so future Tachiom rows are not displayed as
  XTR. The visualizer normalizes older Tachiom rows that were produced before
  this config cleanup.

Verified early profiling runs:

- Short nfcorpus ScaNN -> torch MaxSim runs with varied `k_token` completed and
  show increasing query-time latency as the candidate pool grows.
- Tachiom comparison rows exist both as the older amortized full-batch profile
  and as a newer `search.e2e_profile_batch_size=1` latency-style profile.
- Local combined artifact used for inspection:
  `results_profiling_scann_k_sweep_plus_both_tachiom.jsonl`. This is a run
  artifact and should stay out of git.

Still to do:

- Rust/Tachiom internal stages (`coarse_score`, `candidate_select`, `rerank`)
  need real binding/crate instrumentation. Current Tachiom profiles are coarse
  outer search spans only.
- WARP/FastPlaid rust-internal spans are still pending.
- Deeper build/index-construction profiling inside rust code is still pending;
  the harness currently records coarse build pipeline timing and Python encode
  spans.

---

## How to view profile results

Use the CLI visualizer with `uv run`:

```bash
uv run python scripts/profview.py results_profiling_scann_k_sweep_plus_both_tachiom.jsonl
```

Useful filters:

```bash
uv run python scripts/profview.py results.jsonl --dataset nfcorpus
uv run python scripts/profview.py results.jsonl --index scann --backend torch
uv run python scripts/profview.py results.jsonl --last 10
```

The visualizer reads normal benchmark JSONL rows. It renders:

- a summary table with model/index/retriever/backend, QPS, quality, and whether
  a query profile is present;
- query-time stacked bars from the row's `profile`;
- query/document encode stacked bars from `query_encode_profile` and
  `doc_encode_profile` when those stages were run with profiling enabled;
- build pipeline bars from the coarse `*_time_s` fields.

Do not commit generated result JSONL files, cluster directories, notebooks, or
Slurm logs. They are inspection artifacts, not source.

---

## 1. The pipeline component space (what we must time)

Every run is a composition of four slots. The encoder is shared by all paths;
the back half splits into two fundamentally different shapes.

### Slot A — Encoder (shared by all paths)
- `ColBERT.encode()` — `pylate/models/colbert.py:783`. Query forward pass,
  padding, optional hierarchical pooling. The one stage common to *every*
  method ⇒ the shared baseline segment in every chart.

### Slot B — Candidate generation / index backend

| Backend | Shape | Python entry | Internal stages | Timing today |
|---|---|---|---|---|
| ScaNN | token-path | `pylate/indexes/scann.py` | per-token ANN lookup | none |
| Voyager | token-path | `pylate/indexes/voyager.py` | per-token ANN lookup | none |
| PLAID (native) | E2E | `pylate/indexes/plaid.py` | torch | none |
| FastPlaid | E2E (rust) | `fast_plaid.py:393` → `fast-plaid/rust/search/search.rs:471` | IVF score → gather+dedup → approx score → prune → decompress+exact → topk (**5**) | **none** |
| WARP / xtr-warp-rs | E2E (rust) | `warp.py:415` → `sharded_scorer.rs:1420` | centroid score → centroid select → residual decompress → merge/rerank (**4**) | `profile.rs` spans exist, **not wired** into `rank()`, env-gated `XTR_WARP_PROFILE` |
| Tachiom | E2E (rust, local fork) | `tachiom.py:709` → `tachiom/src/tachiom.rs` | coarse-score accumulate (`:564`) → candidate select (`:628`) → rerank (`:659`) (**3**) | `SearchTimings{stage1_ns,stage2_ns,stage3_ns}` + `search_with_timings()` at `:852` **exist but the pyo3 binding doesn't call it** |

Tachiom also has a **build-time clustering sub-stage** (TAC / PGC / GPU-CAGRA /
external), already represented as the `"cluster"` stage in the harness.

### Slot C — Scoring backend (token-path ONLY; E2E fuses scoring into Slot B)
- flash-maxsim — `pylate/scores/_flash_backend.py` (GPU triton) ✅ present on branch
- LIK — `pylate/scores/_lik_backend.py` (GPU triton / MPS) ✅ present on branch
- torch einsum fallback — `pylate/scores/colbert.py:189` ✅ always
- dispatch / auto-select — `colbert_scores()` `pylate/scores/colbert.py:86`

### Slot D — Retriever wrapper
- ColBERT retriever — `retrieve/colbert.py:93` (dedup ids → `get_documents_embeddings` → `rerank`)
- XTR retriever — `retrieve/xtr.py:99` (score from token hits, no full gather)
- top-k sort — `pylate/rank/rank.py:145`

### The two canonical pipelines, as stage sequences

```
TOKEN-PATH:  encode → token-index-lookup → dedup-ids → gather-embeddings → maxsim(flash|lik|torch) → topk-sort
E2E (rust):  encode → [ rust __call__:  candidate-gen → gather → score → topk ]   (3–5 inner stages, fused)
```

This asymmetry is the whole problem: token-path stages live in Python and are
trivial to time; E2E stages live deep inside three different Rust crates with
three different (non-)conventions.

---

## 2. Core abstraction — the Span tree

A span is the lingua franca, identical in Python and Rust:

```
Span {
  name:    str            # "encode", "maxsim", "rust_search", "stage1_coarse"
  dur_ns:  int            # wall-clock nanoseconds for this span
  device:  "cpu"|"cuda"   # determines whether a CUDA sync brackets the span
  count:   int            # queries this span covers — 1 ⇒ latency-grade,
                          #   >1 ⇒ batch-level, amortized-only (see §2a)
  meta:    dict           # {backend:"flash", n_candidates:512, q_tokens:32, ...}
  children: [Span]        # nested sub-stages
}
```

Why a *tree*, not a flat list: `maxsim` contains `gather`+`einsum`; the rust
`__call__` contains its 3–5 inner stages. Trees serialize to JSON, aggregate by
stage-path, and map directly onto stacked bars (top-level spans = bar segments,
children = drill-down). Every backend, Python or Rust, emits this one shape.

### 2a. Granularity: latency vs throughput, and the `count` field

A span's `count` makes every measurement self-describing, which keeps the two
fundamentally different question-types from being conflated:

- **`count == 1` — latency-grade.** The span timed exactly one query's work in
  isolation. Eligible for per-query distributions (p50/p90 whiskers).
- **`count > 1` — batch-level, amortized only.** The stage processed N queries
  as one block (e.g. a cross-query einsum). The only per-query number derivable
  is `dur_ns / count`, which is a *throughput-derived amortized cost*, **not a
  latency and not a distribution**. The harness renders these as amortized bars
  and refuses to attach latency whiskers to them.

"Time the block then divide" is therefore valid for throughput/QPS comparisons
and a category error for latency — `count` encodes which one you're holding.

**Per-crate reality (verified in the rust sources):**

| Crate | Per-query available? | Notes |
|---|---|---|
| Tachiom | ✅ yes | `batch_search` (`tachiom.rs:784`) loops `self.search(...)` per query; `num_threads=1` is the documented single-thread latency path (`:775`). |
| FastPlaid | ✅ yes | `search_many` (`search.rs:219`) maps a per-query closure over `0..num_queries` with a serial `.map()` mode (`:284`). |
| WARP | ⚠️ partial | Phase 0 centroid scoring is a genuinely cross-query einsum `"btd,cd->btc"` over `[B,T,C]` (`sharded_scorer.rs:617`); WARP is *architected* for batched GPU launches (`:386`). That stage is `count==B`, amortized-only; downstream phases (`:127`, indexed by `b`) are per-query. |

**Canonical configurations:**

- **Latency mode (default for the stacked-bar distributions): `batch_size=1`,
  and `num_threads=1`** for the rust crates that expose it. Production-
  representative (one query at a time), yields real per-query span trees with
  zero division. Avoids a subtle trap: per-query times taken *under* rayon
  parallelism (`num_threads>1`) are "service time under contention" — they
  overlap across threads and sum to more than wall-clock, so they are not clean
  latencies.
- **Throughput mode (separate, opt-in): batched.** Reports amortized
  `dur_ns/count` per stage, labeled as amortized, with variance taken across
  *batches* not queries.
- **WARP caveat:** at `batch_size=1` WARP runs its einsum at `B=1` — valid, but
  its *worst* operating point, since it's built for batched throughput. Report
  both its bs=1 latency and its amortized batched number, and say which is which
  on the chart, or its design strengths are under-sold.

---

## 3. Python layer

### 3a. `pylate/profiling.py` (new, ~120 lines, pure Python)

A `Profiler` holding a contextvar stack and a `@contextmanager span(...)`:

```python
@contextmanager
def span(self, name, device="cpu", count=1, **meta):
    if device == "cuda" and self.cuda_sync and torch.cuda.is_available():
        torch.cuda.synchronize()
    t0 = time.perf_counter_ns()
    node = _push(name, device, count, meta)  # link into parent on the stack
    try:
        yield node
    finally:
        if device == "cuda" and self.cuda_sync and torch.cuda.is_available():
            torch.cuda.synchronize()
        node.dur_ns = time.perf_counter_ns() - t0
        _pop()
```

**CUDA-sync is the correctness crux.** Without `torch.cuda.synchronize()` on
enter/exit, every GPU number (flash/LIK kernels, fastplaid/warp CUDA paths) is a
meaningless async-launch time — you'd measure kernel *launch*, not *execution*,
and the maxsim-backend comparison would be noise. Sync has a cost, so it's a
profiler flag (default on when profiling, off in production). xtr-warp already
concedes this point with its `XTR_WARP_PROFILE_SYNC` env var.

`Profiler.disabled` is a no-op fast path: when off, `span()` does nothing and
costs ~nothing, so instrumentation can stay in hot code permanently.

### 3b. The seam — `self._last_profile`, no signature churn

The thing that kills the spaghetti: **we do not thread a timing payload through
`retrieve → _score_batch → rerank → colbert_scores`.** Instead:

- The index/retriever wraps its own Python-side stages in `profiler.span(...)`.
- After the rust `__call__`, it **grafts the rust timing subtree** (Section 4)
  under the call span.
- It stores the finished tree on **`self._last_profile`** (an attribute on the
  index/retriever object).
- The harness reads `retriever._last_profile` (or `index._last_profile`) after
  the call.

No positional contract anywhere changes. Adding a new backend = wrap its stages
+ write one adapter. Nothing else in the call chain is touched. This is the
direct answer to "principled and extensible rather than passing a new payload
upwards all the time."

### 3c. `Base` ABC opt-in (`pylate/indexes/base.py`, `pylate/retrieve/base.py`)

Formalize the seam as an *optional* protocol so it's non-breaking:

```python
class Base(ABC):
    profiler: Profiler | None = None     # injected by the harness; None ⇒ off
    last_profile: Span | None = None     # populated per call when profiler set
```

Implementations consult `self.profiler` if present and emit spans; if `None`,
the no-op path means zero overhead and zero behavior change for existing callers
(library users, tests, examples).

---

## 4. Rust layer — `tracing` instrumentation, out-of-tree collection

The decision: **standardize on `tracing`**, the Rust ecosystem's de-facto
instrumentation crate, rather than a bespoke per-crate timing module. The whole
point is that timing is **a tag on top of the logic, never intermingled with
it**. The current state proves why this matters — tachiom's
`search_with_timings()` (`tachiom.rs:852`) is a *forked copy* of `search` with
`Instant::now()` sprinkled through it. That duplicate drifts every time `search`
changes. `tracing` eliminates the fork.

### Instrumentation lives in the crate (one line per stage)

- `#[tracing::instrument(skip(...))]` on a function = a span for its whole
  duration. This is the decorator: one attribute, no body changes.
- For sub-function stages: `let _s = tracing::info_span!("centroid_score").entered();`
  — a single line whose RAII guard closes the span at scope exit. (This is
  exactly what xtr-warp's `profile.rs` hand-rolls; `tracing` makes it standard.)

These lines are attached to signatures and stage boundaries, not woven into the
algorithm, so they **survive refactors and rebase cleanly** as a small patch.

### Collection lives in our pyo3 wrapper (out of crate tree)

The crate only *emits* spans and has no idea anything is listening. The timing
tree is built by a custom **`tracing-subscriber` Layer (~60 lines) in the pyo3
binding**, not in the crate. On span enter/close it records an `Instant`, builds
the `Span` tree, and on the per-query root span's close ships it out via a
thread-local the binding drains. **No public API change; no return-payload
threading.** This is the "one schema, one seam" boundary realized on the rust
side: the Layer emits the same `Span {name, dur_ns, device, count, meta,
children}` regardless of crate.

- **Per-query association under a parallel batch:** tag the per-query root span
  with a `query_id` field; `tracing`'s span nesting gives the tree for free, and
  the Layer collects one tree per root. In a rayon `par_iter`, each query's work
  roots on its worker thread, so trees never cross-contaminate.
- **CUDA sync still required.** `tracing` measures CPU wall-clock, so a GPU-stage
  span must call the rust sync (`tch::Cuda::synchronize` / candle
  `Device::synchronize`) inside its boundary, mirroring the Python `cuda_sync`.
- **Zero-cost when off.** With no subscriber attached (or a compile-time
  `release_max_level_*` filter), spans collapse to a cheap level check, so
  instrumentation stays in hot paths permanently.

### Transport: return sidecar (Tier 1), env JSONL (Tier 2)

The Layer-built tree is returned **alongside results, gated by a `with_timings`
flag** (Tier 1 — the benchmarking default): exact per-query attribution,
thread-safe, grafts straight into the Python tree as a subtree. A `tracing`
file/flamegraph subscriber (`tracing-chrome`, `tracing-flame`) is the Tier-2
deep-debug option, replacing xtr-warp's env-gated JSON-lines mode.

### Upstream story

Because the in-crate footprint is just idiomatic `tracing` tags, "add
`#[instrument]` to your search stages" is a realistic upstream PR for
fastplaid/warp — `tracing` is the ecosystem standard and many maintainers
already use or welcome it. And the **collection Layer never needs upstreaming**:
even if a maintainer declines the tags, we carry an attribute-only patch that
rebases trivially. This is the answer to "maintainable alongside upstream
advances."

### Per-crate wiring

- **Tachiom** — replace the forked `search_with_timings()` (`tachiom.rs:852`)
  with `info_span!` tags on the three stages of `search` itself
  (`coarse_accumulate` `:564` / `candidate_select` `:628` / `rerank` `:659`), so
  the timed and untimed paths are the same code. Binding (`python.rs:1138`/`:1214`)
  attaches the Layer when `with_timings=True`.
- **WARP** — retire `profile.rs` in favor of `tracing` tags on the phases in
  `rank()` (`sharded_scorer.rs:1420`): `centroid_score` (`:617`, `count=B`,
  amortized), `centroid_select` / `residual_decompress` / `merge_rerank`
  (per-query).
- **FastPlaid** — fresh `tracing` tags at the 5 stage boundaries in
  `search.rs` (≈ `:491` IVF score, `:534` gather+dedup, `:553` approx score,
  `:625` decompress+exact, `:658` topk), inside the per-query `search()` so
  `count=1`.

Each crate needs a `maturin develop` rebuild after instrumentation (uv sync
fails on cpu nodes for the local fork — rebuild maturin-direct, per project
notes).

---

## 5. GPU maxsim as a first-class comparison point

This is its own section because the kernels are currently **present but not
exercised, not selectable, and not recorded** in `benchmark_indexes.py`.

### Current reality (verified)
- `rerank()` calls `colbert_scores(...)` with **no `backend`** (`rank.py:140`).
  The only selector is env `PYLATE_SCORES_BACKEND` / the `auto` default
  (`colbert_scores` docstring `colbert.py:112`).
- The benchmark **never sets** a backend (grep: no `PYLATE_SCORES_BACKEND`).
- The ScaNN+ColBERT token path passes `device=device or "cpu"`
  (`benchmark_indexes.py:1062`). **On CPU, flash and LIK are unavailable**, so
  `auto` silently runs torch. The GPU kernels never fire unless `device="cuda"`
  is forced.
- The **rust E2E indices never call `colbert_scores`** — GPU maxsim is a lever
  *only* on the token-path (ScaNN/Voyager). Framing for the chart: "GPU maxsim"
  is a property of the *scoring stage of the token-path*, not of PLAID/WARP/
  tachiom (which have their own internal scoring).
- Backend choice is **not recorded** in `results.jsonl`.

### Plan to make it usable, timed, and comparable

1. **Selectability — DECIDED: thread `backend=` through the API.** Add a
   `backend` param threaded
   `retrieve(..., backend=) → _score_batch → rerank → colbert_scores(backend=)`.
   `colbert_scores` already accepts `backend` (`colbert.py:91`); the missing
   links are `retrieve`/`_score_batch`/`rerank`, which currently drop it. This
   gives per-call selection (no global env mutation), records cleanly per row,
   and lets a single benchmark process sweep `torch|flash|lik|auto`
   back-to-back. The env var `PYLATE_SCORES_BACKEND` stays as the fallback
   default when the param is `None`. Touches the library + the three call sites
   + their tests; that cost is accepted.
2. **Device.** Default the token-path scoring `device` to `cuda` when available
   (or expose it in config), so the kernels actually fire. Right now `"cpu"` is
   the silent default.
3. **Stage isolation.** Wrap the `colbert_scores` call inside `rerank` in a
   `span("maxsim", device=scoring_device, backend=resolved_backend,
   q_tokens=…, n_candidates=…)`. Also wrap `get_documents_embeddings` as
   `span("gather")` and the encode as `span("encode")`. Without isolating the
   maxsim sub-step, encode+lookup+gather dominate and the kernel difference is
   invisible.
4. **Recording.** Add to each `results.jsonl` row: the resolved backend,
   `flash_available` / `lik_available` (from `_flash_backend.is_available()` /
   `_lik_backend.is_available()`), and the per-stage maxsim timing from the span
   tree.
5. **Optional focused kernel sweep.** Because GPU maxsim only affects the
   scoring stage on a candidate set, the cleanest *isolated* comparison is a
   small harness that calls `colbert_scores` directly across backends on
   controlled (n_queries × q_tokens × n_docs × d_tokens) shapes — independent of
   index choice. Complements the in-situ stage timing; gives clean
   kernel-vs-kernel bars.

Pre-req checklist: confirm the extras are actually installed in the venv
(`uv pip list | grep -E 'flash-maxsim|late-interaction-kernels'`); the runtime
`is_available()` recording (step 4) makes this explicit per run regardless.

---

## 6. `results.jsonl` schema extension

Extend, don't replace. The harness already has `canonical_stage_order`,
`append_stage_marker()` (env `BENCH_STAGE_MARKERS_FILE`), and `*_time_s` fields.
Add one key per row:

```jsonc
"profile": {
  "mode": "latency",                // "latency" (bs=1, count==1) | "throughput" (amortized)
  "stages": {                       // reduced per-stage, ns
    "encode":       {"p50": ..., "p90": ..., "mean": ..., "n": ..., "count": 1},
    "index_lookup": {...},
    "gather":       {...},
    "maxsim":       {..., "backend": "flash"},
    "topk":         {...}
    // count > 1 stages (e.g. WARP centroid_score) carry only "amortized" + "count",
    // never p50/p90 — the reducer drops percentiles when count != 1.
  },
  "tree_sample": { ... }            // one full per-query span tree, for drill-down
}
```

Build-time stages (`encode_docs`, `cluster`, `build_index`) already flow through
the existing `*_time_s` fields; query-time stages become the same shape one
level down.

---

## 7. Visualization

`scripts/profview.py` reads benchmark `results.jsonl` files and renders stacked
bars in the terminal. It uses the same stage names produced by
`reduce_stage_timings()`, orders query-time stages by the canonical retrieval
pipeline, and labels end-to-end index rows as either amortized batch spans or
bs=1 latency spans depending on the captured profile shape.

Example:

```bash
uv run python scripts/profview.py results_profiling_scann_k_sweep_plus_both_tachiom.jsonl
```

Filters:

```bash
uv run python scripts/profview.py results.jsonl --dataset nfcorpus
uv run python scripts/profview.py results.jsonl --index scann --backend torch
uv run python scripts/profview.py results.jsonl --last 10
```

A future saved-figure script can reuse the same reduced profile schema, but the
current committed visualization surface is the terminal CLI.

---

## 8. Build phases / progress

1. **Done:** `pylate/profiling.py` — Span + Profiler (CUDA-sync), reducer, and
   ambient context manager.
2. **Done:** instrument the **token-path in Python** (encode / index_lookup /
   gather / tensor preparation / maxsim / topk / result materialization) +
   `last_profile` seam + `BaseRetriever` opt-in. This also lands the
   GPU-MaxSim backend lever and result metadata.
3. **Done, coarse only:** end-to-end index profiling can force outer bs=1 spans
   via `search.e2e_profile_batch_size=1`, enough to compare Tachiom coarse
   search latency against token-path rows. Internal rust stages are still opaque.
4. **Pending:** **Tachiom** — `tracing` tags on `search`'s three stages + the collection
   Layer in the binding; delete the forked `search_with_timings`. Smallest rust
   change, and it removes existing debt.
5. **Pending:** **WARP** — `tracing` tags on `rank()`'s phases; retire
   `profile.rs`.
6. **Pending:** **FastPlaid** — fresh `tracing` tags at the 5 boundaries.
7. **Pending:** saved plot/export script if terminal visualization is not enough.

Phases 1–3 deliver the current comparison without touching any rust crate; 4–6
progressively light up the E2E internals. The collection Layer (≈60 lines) is
written once in phase 4 and reused by 5–6 — factor it into a shared internal
helper rather than copying.

---

## 9. Open questions / risks

- **`tracing` Layer overhead.** Attaching the subscriber adds per-span tree
  bookkeeping. Gate behind `with_timings`; with no subscriber, tags are a cheap
  level check, so production batched search is unaffected.
- **CUDA sync cost in aggregate.** Syncing around every span in a tight per-query
  loop serializes the GPU (both the Python `cuda_sync` and the rust
  `tch::Cuda::synchronize` inside GPU-stage spans). Accepted while profiling; off
  in production. Consider a "sync only top-level spans" mode if per-query sync
  proves too costly.
- **Schema consistency across three bindings.** Each crate's pyo3 wrapper hosts a
  Layer, so the `Span` field shape could drift. Mitigate by factoring the Layer
  into one shared internal helper (phase 3) and validating the schema Python-side
  when ingesting a rust sidecar.
- **WARP at `batch_size=1`.** Its Phase-0 einsum runs at `B=1`, its worst point.
  Not a bug, but the chart must label WARP's latency-mode and throughput-mode
  numbers distinctly (see §2a) or it reads as slower than it is in deployment.
- **XTR path.** `score_xtr` (`rank.py:160`) is a different scoring routine
  (min-imputation, no full gather) and never calls `colbert_scores` — it needs
  its own span set; it is *not* part of the GPU-maxsim comparison.
