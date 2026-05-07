#!/usr/bin/env bash
# A/B which fix matters: post-build sync_stream vs. retained host dataset.
# Pass label as first arg, the rest are env-var overrides.
set -uo pipefail

cd "$(dirname "$0")"

LD_LIBRARY_PATH="$(conda info --base 2>/dev/null)/envs/fastplaid-cagra/lib:$(conda info --base 2>/dev/null)/envs/fastplaid-cagra/targets/sbsa-linux/lib:$(uv run --no-sync python -c 'import torch, os; print(os.path.join(os.path.dirname(torch.__file__), "lib"))'):${LD_LIBRARY_PATH:-}" \
  uv run --no-sync python examples/evaluation/_fiqa_cagra_ablation.py "$@"
