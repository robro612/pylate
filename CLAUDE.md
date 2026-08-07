# PyLate-PGC — working notes

Research fork of PyLate for proximity-graph clustering, tachiom, and stage-level
retrieval profiling.

## Environments — read this before running anything

There are **four** venvs and **no `.venv`**. All four come from the single
`uv.lock`; see [docs/environments.md](docs/environments.md) for the full story.

| venv | nodes | Rust forks |
|---|---|---|
| `.venv-cu130` | L40S / A100 / H100 | plain — use for QPS |
| `.venv-cu126` | V100 (CUDA 13 dropped Volta sm_70) | plain — use for QPS |
| `.venv-cu130-profile` | L40S / A100 / H100 | stage timing enabled |
| `.venv-cu126-profile` | V100 | stage timing enabled |

Every Python invocation must select one, because uv has no pyproject setting for
it:

```bash
UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync python scripts/benchmark_indexes.py ...
```

Rules that matter:

- **Always `--no-sync`** on experiment runs. A bare `uv run` re-syncs and rebuilds
  the Rust forks with default features, silently converting a profiling
  environment into a plain one mid-experiment.
- **Never `source .venv*/bin/activate` then plain `python`** unless you also pass
  `uv run --active`; uv otherwise ignores the active venv and warns.
- **Never add `profile` to `[tool.maturin] features`** in either fork. That list
  applies to every build of the tree, including published wheels.
- Rebuild an environment with `./scripts/sync_env.sh {cu130|cu126|cu130-profile|cu126-profile}`,
  under slurm with `--mem=64G --cpus-per-task=8` (torch-sys's generated C++ gets
  OOM-killed at the default allocation).

## Compute

- No compute on the login node — not even an import check. Prefix with `srunv100`
  (V100) or `srunl40s` (L40S); CPU-only work can use `-p cpu`.
- Give every job `--job-name=<short>` and redirect to a persistent log.
- Long benchmarks and sweeps are the user's to launch, not the agent's.
- Never route caches, results, or indexes to `/tmp` — it gets wiped. Use
  `embeddings_cache/`, `results/`, `indexes/`.

## Layout

```
pylate/                 library; pylate/profiling.py is the span/timing layer
crates/stage-profile/   shared Rust stage timer, consumed by the local forks
third_party/            symlinks to the fork checkouts (see its README)
scripts/                durable tools: benchmark_indexes.py, profview.py,
                        gpu_cluster.py, cluster_only.py, sync_env.sh
scripts/lib/env.sh      environment selection, sourced by experiment scripts
scripts/experiments/    one-off sweep drivers (read PYLATE_VENV) — currently empty
scripts/slurm/          .sbatch job files
scripts/analysis/       one-off measurement scripts
docs/environments.md    environment reference — the authority on venv handling
docs/profiling.md       stage-span profiling: stages, CUDA correctness, caveats
```

## Local forks

`tachiom` and `fast-plaid` are built from checkouts next to this repo
(`../tachiom`, `../fast-plaid`), wired via `[tool.uv.sources]`. Both consume
`crates/stage-profile` behind an optional Cargo `profile` feature.

`fast-plaid` is installed **non-editable** deliberately — an editable maturin
install is a `.pth` into the source tree, so every environment would share the
one compiled `.so` there. Rust changes need a re-sync or a targeted
`maturin develop` with `LIBTORCH` set; see docs/environments.md.

Push fork changes to the user's forks (`robro612/...`), not upstream.

## Profiling

`search.profile=true` collects a per-stage breakdown; add
`search.e2e_profile_batch_size=1` for latency-grade percentiles. If the backend
was not built with the feature, PyLate raises `RustProfileUnavailableError`
up front rather than reporting a breakdown of pure `unaccounted`. View results
with `scripts/profview.py`. Details in [docs/profiling.md](docs/profiling.md).
