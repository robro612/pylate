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
# The cu130 targets additionally build Chimera (C++/CUDA, cuVS). That one has to
# run on an L40S/A100/H100 node and nowhere else: CMakeLists.txt uses
# -march=native with unconditional AVX-512 intrinsics, and CUDA_ARCHITECTURES
# defaults to `native`, which needs a visible GPU. cu126 skips it — its V100
# nodes are Broadwell Xeons with no AVX-512, so Chimera cannot run there at all.
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
cu130) CUDA_EXTRA="cu130" PROFILED=0 CHIMERA=1 ;;
cu126) CUDA_EXTRA="cu126" PROFILED=0 CHIMERA=0 ;;
cu130-profile) CUDA_EXTRA="cu130" PROFILED=1 CHIMERA=1 ;;
cu126-profile) CUDA_EXTRA="cu126" PROFILED=1 CHIMERA=0 ;;
# A second profiling environment on the same axes. It exists so a Rust change
# can be built and measured while a long sweep still runs against
# .venv-cu130-profile: each sweep point is a fresh `uv run` process, so
# rebuilding the environment under a running sweep would split its result file
# across two different binaries. Retire it once the sweep finishes and the
# change lands in the four standard environments.
cu130-profile-b) CUDA_EXTRA="cu130" PROFILED=1 CHIMERA=1 ;;
*)
    echo "usage: $0 {cu130|cu126|cu130-profile|cu126-profile|cu130-profile-b}" >&2
    exit 2
    ;;
esac

VENV=".venv-$TARGET"
ARGS=(--extra "$CUDA_EXTRA" "${REBUILD[@]}")
if [[ "$PROFILED" == 1 ]]; then
    ARGS+=("${PROFILE_ARGS[@]}")
fi
if [[ "$CHIMERA" == 1 ]]; then
    # Same reasoning as REBUILD above: the compiled artifact is what differs
    # between environments, and uv's build cache cannot tell a -march=native
    # build on one node class from another.
    ARGS+=(--extra chimera --reinstall-package chimera-retrieval)
    HOLD_CHIMERA=(--no-install-package chimera-retrieval)
else
    HOLD_CHIMERA=()
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

echo "==> [1/2] syncing $VENV ($TARGET) without the compiled backends"
uv sync --locked "${ARGS[@]}" "${HOLD_CHIMERA[@]}" \
    --no-install-package fast-plaid --no-install-package tachiom

if [[ ! -d "$TORCH_DIR" ]]; then
    echo "error: expected torch at $TORCH_DIR after phase 1" >&2
    exit 1
fi

echo "==> [2/2] building the compiled backends against $VENV"
export LIBTORCH="$TORCH_DIR"
export LIBTORCH_BYPASS_VERSION_CHECK=1

if [[ "$CHIMERA" == 1 ]]; then
    # nvcc is not on PATH on these nodes, so CMake's enable_language(CUDA) has
    # nothing to find; CUDACXX is the variable it looks at. Match the CUDA major
    # version to the RAPIDS wheels the cu130 extra installed (libcuvs_cu13), not
    # to whatever /usr/local/cuda happens to point at.
    CUDA_MAJOR="$(ls -d "$PROJECT_DIR/$VENV"/lib/python3.12/site-packages/libcuvs_cu*.dist-info 2>/dev/null \
        | head -1 | sed -E 's/.*libcuvs_cu([0-9]+).*/\1/')"
    CHIMERA_CUDA_HOME="${CHIMERA_CUDA_HOME:-$(ls -d /usr/local/cuda-"${CUDA_MAJOR:-13}".* 2>/dev/null | sort -V | tail -1)}"
    if [[ ! -x "$CHIMERA_CUDA_HOME/bin/nvcc" ]]; then
        echo "error: no nvcc under $CHIMERA_CUDA_HOME (needed to build chimera)" >&2
        exit 1
    fi
    export CUDACXX="$CHIMERA_CUDA_HOME/bin/nvcc"
    # scikit-build-core forwards CMAKE_ARGS. Pass the architecture explicitly
    # rather than relying on CMakeLists' default: getting it wrong does not fail
    # the build, it ships PTX the driver cannot JIT and dies at the first kernel
    # launch. `native` reads it off this node's GPU; set CHIMERA_CUDA_ARCH
    # (89 = L40S, 80 = A100, 90 = H100) to build for a different node class.
    export CMAKE_ARGS="-DCUDAToolkit_ROOT=$CHIMERA_CUDA_HOME -DCMAKE_CUDA_ARCHITECTURES=${CHIMERA_CUDA_ARCH:-native}"
    echo "    chimera: nvcc=$CUDACXX arch=${CHIMERA_CUDA_ARCH:-native}"
fi

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
try:
    import chimera

    print("chimera:", chimera.ChimeraIndex)
except Exception as exc:  # pragma: no cover - diagnostic only
    print("chimera: unavailable —", exc)
PY
