# SparVAR CS4A CUDA kernels

This package builds the `chipmunk` Python extension used by the released image
and video attention paths (`torch.ops.chipmunk.*`). Its native attention kernels
require a Hopper GPU (SM90, such as H100/H800), a matching CUDA-enabled PyTorch
installation, CUDA Toolkit 12.8 and a C++20-capable compiler compatible with that
CUDA toolkit. The setup script detects the visible GPU; this is a source build
on the target Linux GPU machine, not a portable prebuilt binary.

The initial release uses the original Chipmunk backend. The independent TK2
comparison package is not required by these model entrypoints. Other hardware
and kernel/backend combinations require separate validation.

## 1. Fetch the pinned ThunderKittens headers

From the SparVAR repository root:

```bash
bash scripts/download_thunderkittens.sh
```

The downloader uses `https://github.com/HazyResearch/ThunderKittens.git` and locks
revision `0c44d2c7262fdd94273a18420a5861bc96a335e3`, the revision specified by the
original kernel build instructions. It creates a separate Git checkout at
`kernels/chipmunk/submodules/ThunderKittens`, verifies its revision and required
headers, and rejects an existing plain directory or modified checkout. It does
not use the surrounding SparVAR Git revision as evidence of the dependency's
version. Keep the downloaded dependency's license and notices with its checkout.

No dependency source or machine-specific `.so` is copied into the release.
The pinned build still needs a fresh GPU build and numerical validation; local
copies of headers from a development machine do not establish reproducibility.

## 2. Build the extension

Install the selected CUDA-enabled PyTorch and FlashAttention environment from the
root installation instructions first, then run from the repository root:

```bash
python -m pip install -r kernels/chipmunk/requirements.txt
python -m pip install --no-build-isolation --no-deps -e kernels/chipmunk
python -c "import chipmunk; import torch; print(torch.ops.chipmunk.dense_attn)"
```

Set `CUDA_HOME`, `PATH` and the runtime library path to the CUDA toolkit used for
the build if it is not the system default. Check `nvcc --version` before building.
CUDA 12.8 requires a compatible host compiler; when using a Conda GCC/G++ 14
compiler, select a toolkit-compatible compiler (for example GCC/G++ 13) rather
than bypassing the compiler version check. Do not replace an already selected
PyTorch/CUDA stack while resolving unrelated research-package dependencies.

The Python extension retains its `chipmunk` import name and all current operator
signatures. The current setup compiles its sparse-MLP sources as well, although
the published attention inference configurations disable sparse MLP.
Rebuild after changing CUDA source or the CUDA/PyTorch/compiler environment.

## 3. Validate on the GPU machine

From the repository root, start with the small correctness pipeline:

```bash
python kernels/tests/test_cross_attn_pipeline.py --help
python kernels/tests/test_cross_attn_pipeline.py
python kernels/tests/test_triton_cross_attn_pipeline.py
python kernels/tests/test_mask_to_indices_width.py
```

These checks cover dense/column-sum/sparse attention, including cross-attention
shapes and grouped KV where implemented. Run the selected model-level tests and
fixed-seed image/video checks from the root reproduction guide afterward.
Some GPU test files allocate CUDA tensors at import; do not collect the whole
kernel test directory in a CPU-only pytest job.

## HART CUDA extension

HART also uses its own normalization/RoPE extension. From the repository root:

```bash
bash models/hart/kernels/install.sh
```

That script builds with the active Python environment and visible GPU. It does
not download weights or third-party headers.
