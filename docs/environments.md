# Environments

Four virtual environments, all resolved from the single `uv.lock`, so they cannot
drift in Python-package versions. Two axes: which CUDA runtime, and whether the
Rust index backends carry stage-timing instrumentation.

| venv | nodes | torch | Rust forks |
|---|---|---|---|
| `.venv-cu130` | L40S / A100 / H100 | `2.11.0+cu130` | plain — use for QPS |
| `.venv-cu126` | V100 | `2.11.0+cu126` | plain — use for QPS |
| `.venv-cu130-profile` | L40S / A100 / H100 | `2.11.0+cu130` | `--features profile` |
| `.venv-cu126-profile` | V100 | `2.11.0+cu126` | `--features profile` |

## Why four

**CUDA 13 dropped Volta (sm_70).** V100 nodes cannot run the default torch 2.11
wheel, so they take the cu126 build from the PyTorch index. The torch *version*
is identical across all four, which matters because the Rust forks link libtorch
directly — same API, so timings are comparable across node types.

**A profiling build is a different artifact.** With `--features profile`,
fast-plaid synchronises the CUDA device around every stage so a span measures
kernel execution rather than launch. That is correct for attribution and wrong
for a throughput number, so it lives in its own environment instead of being a
flag you have to remember not to leave on.

## Building

Under slurm — this compiles `torch-sys`, whose generated C++ translation unit
gets OOM-killed at slurm's default memory allocation:

```bash
srunl40s --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu130
srunv100 --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu126
srunl40s --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu130-profile
srunv100 --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu126-profile
```

Each run prints the torch version and each fork's `profile_supported()` so you
can see what you got.

## Using

uv has no pyproject setting for *which* environment to use — only the
`UV_PROJECT_ENVIRONMENT` variable — so every invocation has to say.

```bash
# Explicit, and what the sbatch scripts should do
UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync python scripts/benchmark_indexes.py ...

# Interactive
source .venv-cu130/bin/activate && uv run --active python ...

# Anything under scripts/experiments/ reads PYLATE_VENV (default .venv-cu130)
PYLATE_VENV=.venv-cu126 ./scripts/experiments/run_pgc_sweep.sh
```

**Always pass `--no-sync` for experiment runs.** A bare `uv run` re-resolves and
re-syncs first, which rebuilds the Rust forks with default features — silently
turning a profiling environment into a plain one mid-experiment.

There is deliberately no `.venv`. If you run `uv run` without selecting an
environment, uv creates one from scratch rather than using the wrong one.

## Maintaining

**Adding a Python dependency.** Edit `[project.optional-dependencies]` (the
published install surface) or `[dependency-groups]`, run `uv lock`, then re-sync
whichever environments you need. The `research` group self-references the extras
(`pylate[dev,eval,api,scann,voyager,tachiom]`) and is in
`[tool.uv] default-groups`, which is what stops `uv run` from pruning tachiom /
hydra / ranx out from under you.

**Changing Rust in a fork.** `fast-plaid` is installed **non-editable** on
purpose: a maturin editable install is a `.pth` into the source tree, so all four
environments would import the one compiled `.so` sitting there and the last build
would win. Either re-run `sync_env.sh` for the environments you care about, or
`maturin develop` into one directly:

```bash
export VIRTUAL_ENV=/exp/rjha/pylate-pgc/.venv-cu130-profile
export PATH="$VIRTUAL_ENV/bin:$PATH"
export LIBTORCH="$VIRTUAL_ENV/lib/python3.12/site-packages/torch"
export LIBTORCH_BYPASS_VERSION_CHECK=1
cd /exp/rjha/fast-plaid && maturin develop --features profile --release
```

`LIBTORCH` has to be explicit. `LIBTORCH_USE_PYTORCH=1` resolves against uv's
*isolated build env*, whose floating `torch >= 2.7.0` picks up a release newer
than 2.11 that no longer compiles against torch-sys 0.20 (`at::Tensor has no
member named align_as`). `scripts/sync_env.sh` handles this by syncing in two
phases: torch first, then the forks with `LIBTORCH` pointed at the target venv.

**Adding a CUDA variant.** Add an extra alongside `cu126`/`cu130`, list it in
`[tool.uv] conflicts` so one environment cannot hold two, and point `torch` at
the matching index in `[tool.uv.sources]`.

## GPU clustering

`scripts/gpu_cluster.py` (CAGRA/brute) needs `cuvs`, which is CUDA-major-version
specific: `.venv-cu130` carries `cuvs-cu13`, `.venv-cu126` carries `cuvs-cu12`.
V100 clustering therefore runs from `.venv-cu126` — which replaced the old
hand-built `.venv-cu12` sidecar, since it now provides both Volta-compatible
torch and `cuvs-cu12` from the shared lockfile.

`cuvs` wheels on `pypi.nvidia.com` carry no upload date, so any `exclude-newer`
cutoff filters out every compatible wheel. The global `~/.config/uv/uv.toml` has
that setting disabled for this project; use a per-command `--exclude-newer`
rather than turning it back on.
