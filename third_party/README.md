# Local Rust index checkouts

Symlinks (gitignored contents) used for editable / maturin builds:

```text
third_party/tachiom      -> /exp/rjha/tachiom
third_party/fast-plaid   -> /exp/rjha/fast-plaid
third_party/xtr-warp-rs  -> /exp/rjha/xtr-warp-rs
third_party/Chimera      -> /exp/rjha/Chimera
```

`pyproject.toml` `[tool.uv.sources]` points both `tachiom` and `fast-plaid` at these checkouts (`../tachiom`, `../fast-plaid` — the same trees as `/exp/rjha/*` on this machine). `xtr-warp-rs` is installed editable by hand.

`fast-plaid` is installed **non-editable**: a maturin editable install is a `.pth` into the source tree, so every environment would share the single compiled `.so` sitting there. With one venv per CUDA variant plus a profiling venv, the last build would win — a V100 env importing a cu130-linked extension, or a QPS run importing a CUDA-synced profiling build. Build environments with [`scripts/sync_env.sh`](../scripts/sync_env.sh) (`cu130` / `cu126` / `profile`), which also points `LIBTORCH` at the target venv; see [`docs/profiling.md`](../docs/profiling.md).

## Shared stage profiler crate

[`crates/stage-profile`](../crates/stage-profile) provides flat RAII `StageGuard` / `begin` / `take` used by local forks behind a Cargo feature:

```toml
# in tachiom / fast-plaid Cargo.toml
stage-profile = { path = "../pylate-pgc/crates/stage-profile", optional = true }
profile = ["dep:stage-profile"]
```

Each fork owns a small `stage_shim` that gates the `stage!` macro on the feature, so default builds compile the timers out entirely and never link the crate. fast-plaid's shim additionally synchronises CUDA around each guard; tachiom's is CPU-only. See [`docs/profiling.md`](../docs/profiling.md) for build commands and the stage list.

| fork | adopted | notes |
|---|---|---|
| tachiom | yes | CPU; 3 stages |
| fast-plaid | yes | CUDA-synced; 12 stages, on upstream 1.6.0 |
| Chimera | n/a | C++/CUDA, not Rust; no stage timers |
| xtr-warp-rs | **no** | has an orphaned `rust/profile.rs` (env-var JSONL writer, never wired into `lib.rs`); delete it when porting to the shared crate |

Push profiling-related Rust changes to user forks (`robro612/...`), not upstream, until ready.

## Chimera

Local checkout on branch `pylate-packaging`, carrying five commits upstream does
not have. Push to `robro612/Chimera`; all five are worth upstreaming.

| commit | what |
|---|---|
| packaging | a `pyproject.toml` — upstream is a bare CMake project you were meant to build in a conda env and put on `PYTHONPATH` |
| CUDA arch | sets `CMAKE_CUDA_ARCHITECTURES` before `enable_language(CUDA)`; without it the module ships PTX the driver cannot JIT |
| query shape | `PADDED_DIM` / `Q_DOCLEN` as CMake options, exported to Python |
| borrowing `build()` | takes `const float*` instead of copying the whole corpus into a `std::vector` — 169 GiB of duplication on lotte |
| dead 1-bit array | stops retaining and persisting doc-major 1-bit codes no search path reads |

With those, it installs like any other backend — `pyproject.toml` declares it as
the `chimera` extra with `[tool.uv.sources] chimera-retrieval = { path =
"../Chimera" }` — and `scripts/sync_env.sh cu130` builds it. Non-editable, and
**built with build isolation off** (`[tool.uv] no-build-isolation-package`): the
extension resolves cuVS/RAFT/RMM out of the target venv's site-packages and
RPATHs those paths, so an isolated build environment would leave a dangling
RPATH. It is also `-march=native`, making the artifact specific to both the venv
and the node class.

`cu126` deliberately skips it: `compute_full_bit_scores` uses AVX-512
intrinsics unconditionally and the V100 nodes are Broadwell Xeons without them.

See [docs/chimera.md](../docs/chimera.md) for the two upstream *behaviours* the
PyLate wrapper works around, as distinct from the two build fixes above.
