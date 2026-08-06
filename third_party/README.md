# Local Rust index checkouts

Symlinks (gitignored contents) used for editable / maturin builds:

```text
third_party/tachiom      -> /exp/rjha/tachiom
third_party/fast-plaid   -> /exp/rjha/fast-plaid
third_party/xtr-warp-rs  -> /exp/rjha/xtr-warp-rs
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
| xtr-warp-rs | **no** | has an orphaned `rust/profile.rs` (env-var JSONL writer, never wired into `lib.rs`); delete it when porting to the shared crate |

Push profiling-related Rust changes to user forks (`robro612/...`), not upstream, until ready.
