#!/usr/bin/env bash
# Build a minimal cu12 sidecar venv (.venv-cu12) so scripts/gpu_cluster.py can run on
# V100 (Volta sm_70). The main .venv is torch 2.11+cu130, and CUDA 13 dropped Volta, so
# V100 can't run the default env's GPU torch. Only the *clustering* step needs a GPU; the
# tachiom index build + eval stays CPU on the main .venv. So this sidecar is intentionally
# minimal (no full pylate stack, no editable tachiom Rust build).
#
# Run gpu_cluster.py on V100 with:
#   srunv100 uv run --no-sync --python .venv-cu12/bin/python python scripts/gpu_cluster.py \
#     clustering.k=<K> ...           # pass explicit k (sidecar has no tachiom auto-resolver)
#
# --no-config drops the global ~/.config/uv/uv.toml `exclude-newer = "P3D"` rolling cutoff,
# which otherwise filters the dated nvidia/pytorch wheels and the undated pypi.nvidia.com ones.
# Rebuild any time with: bash scripts/setup_cu12_venv.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
PY="$ROOT/.venv-cu12/bin/python"
TORCH_VER="${TORCH_VER:-2.6.0}"   # latest cu124 cp312 wheel; still ships Volta sm_70 for V100

uv venv "$ROOT/.venv-cu12" --python 3.12

uv pip install --python "$PY" --no-config "torch==${TORCH_VER}" \
  --extra-index-url https://download.pytorch.org/whl/cu124 --index-strategy unsafe-best-match
uv pip install --python "$PY" --no-config numpy hydra-core omegaconf tqdm
uv pip install --python "$PY" --no-config "cuvs-cu12>=25.10" \
  --extra-index-url https://pypi.nvidia.com --index-strategy unsafe-best-match

echo "=== arch check (must include sm_70 for V100) ==="
"$PY" -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda); al=torch.cuda.get_arch_list(); print('archs', al); print('SM_70 PRESENT:', any('70' in a for a in al))"
