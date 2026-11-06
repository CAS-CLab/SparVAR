#!/usr/bin/env bash
set -euo pipefail

cd -- "$(dirname -- "${BASH_SOURCE[0]}")"
# The existing HART setup.py detects the visible GPU directly.
unset TORCH_CUDA_ARCH_LIST
python -m pip install --no-build-isolation --no-deps -e .
