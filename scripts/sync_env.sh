#!/usr/bin/env bash
# Build one of PyLate's environments from the single shared uv.lock.
#
#   ./scripts/sync_env.sh cu130           # L40S/A100/H100 -> .venv-cu130
#   ./scripts/sync_env.sh cu126           # V100           -> .venv-cu126
#   ./scripts/sync_env.sh cu130-profile   # + stage timing -> .venv-cu130-profile
#   ./scripts/sync_env.sh cu126-profile   # + stage timing -> .venv-cu126-profile
#
# Two axes, four environments. CUDA 13 dropped Volta (sm_70), so V100 needs the
# cu126 torch wheel; and a profiling build of the Rust forks CUDA-syncs every
# stage, which is the wrong artifact for a QPS number — so profiling gets its own
# environment per architecture rather than contaminating the fast one. All four
# resolve from the same lockfile and cannot drift in Python-package versions.
#
# Run under slurm — these compile the Rust forks, including torch-sys, whose
# generated C++ translation unit needs several GB per cc job. Ask for memory or
# the build gets OOM-killed:
#   srunl40s --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu130
#   srunv100 --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh cu126
#   srunl40s --mem=64G --cpus-per-task=8 ./scripts/sync_env.sh profile
#
# Then use an environment without letting uv re-sync (and re-prune) it:
#   UV_PROJECT_ENVIRONMENT=.venv-cu130 uv run --no-sync python ...
#   # or: source .venv-cu130/bin/activate && uv run --active python ...
set -euo pipefail

TARGET="${1:-}"
PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$PROJECT_DIR"

# maturin's PEP 517 hook takes extra args per package. The two forks need
# different feature sets, so this cannot be the single global
# MATURIN_PEP517_ARGS string; and the full list must be repeated because
# maturin's CLI --features replaces rather than merges [tool.maturin] features.
PROFILE_ARGS=(
    --config-settings-package
    "fast-plaid:maturin.build-args=--features pyo3/extension-module,profile"
    --config-settings-package
    "tachiom:maturin.build-args=--features python,profile"
)

# Always rebuild the two Rust forks. Their compiled extension is what actually
# differs between these environments — CUDA variant, and profile feature on/off —
# and relying on uv's build cache to distinguish them is how you end up importing
# a profiling extension in a QPS environment.
REBUILD=(--reinstall-package fast-plaid --reinstall-package tachiom)

case "$TARGET" in
cu130) CUDA_EXTRA="cu130" PROFILED=0 ;;
cu126) CUDA_EXTRA="cu126" PROFILED=0 ;;
cu130-profile) CUDA_EXTRA="cu130" PROFILED=1 ;;
cu126-profile) CUDA_EXTRA="cu126" PROFILED=1 ;;
*)
    echo "usage: $0 {cu130|cu126|cu130-profile|cu126-profile}" >&2
    exit 2
    ;;
esac

VENV=".venv-$TARGET"
ARGS=(--extra "$CUDA_EXTRA" "${REBUILD[@]}")
if [[ "$PROFILED" == 1 ]]; then
    ARGS+=("${PROFILE_ARGS[@]}")
fi

export UV_PROJECT_ENVIRONMENT="$VENV"

# Bound compile parallelism: torch-sys's generated C++ unit peaks at several GB,
# and one cc job per core is what turns a big node into an OOM kill.
export CARGO_BUILD_JOBS="${CARGO_BUILD_JOBS:-4}"

# Two phases, because the Rust forks link libtorch from the *target* venv and
# that venv's torch does not exist yet on a fresh sync. Leaving it to
# LIBTORCH_USE_PYTORCH is not an option: under uv's PEP 517 build that resolves
# to the isolated build env's torch, which floats to the newest release and no
# longer compiles against torch-sys 0.20.
TORCH_DIR="$PROJECT_DIR/$VENV/lib/python3.12/site-packages/torch"

echo "==> [1/2] syncing $VENV ($TARGET) without the Rust forks"
uv sync --locked "${ARGS[@]}" \
    --no-install-package fast-plaid --no-install-package tachiom

if [[ ! -d "$TORCH_DIR" ]]; then
    echo "error: expected torch at $TORCH_DIR after phase 1" >&2
    exit 1
fi

echo "==> [2/2] building the Rust forks against $VENV torch"
export LIBTORCH="$TORCH_DIR"
export LIBTORCH_BYPASS_VERSION_CHECK=1
uv sync --locked "${ARGS[@]}"

echo "==> $VENV ready"
uv run --no-sync python - <<'PY'
import torch

print("torch:", torch.__version__, "| cuda:", torch.version.cuda)
try:
    from fast_plaid import fast_plaid_rust as fp

    print("fast-plaid profile:", fp.profile_supported())
except Exception as exc:  # pragma: no cover - diagnostic only
    print("fast-plaid: unavailable —", exc)
try:
    import tachiom

    # A static method, so the build is probeable without loading an index.
    print("tachiom profile:", tachiom.Tachiom.profile_supported())
except Exception as exc:  # pragma: no cover - diagnostic only
    print("tachiom: unavailable —", exc)
PY
