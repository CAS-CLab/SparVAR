# Installation

This checkout is a local migration draft. The instructions describe the intended
standalone environment; a fresh GPU installation and inference validation are
still pending. Use a separate environment rather than an existing editable Lvar
installation, and run from this repository root.

## Environment

- Linux with an NVIDIA Hopper GPU (SM90a, such as H100/H800).
- Python 3.12 or newer: copied model sources use Python 3.12 f-string syntax.
- A matching CUDA-enabled PyTorch/TorchVision installation and FlashAttention.
- CUDA Toolkit 12.8 with `nvcc` and a compatible C++20 host compiler for the native
  Chipmunk build. See the kernel build guide for compiler selection.

Other hardware, including A100, RTX 4090 and Apple GPU inference, has not been
validated for the current production kernel stack. Broad Python package bounds
are not a validated Torch/CUDA/FlashAttention version matrix. That matrix must
be recorded after clean GPU testing; do not interpret these instructions as a
new numerical or performance validation.

Create an isolated environment, then install a compatible CUDA-enabled
PyTorch/TorchVision pair and FlashAttention using their build instructions.
Install those GPU packages before the editable project and native extensions.

```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip setuptools wheel
# Install the selected matching CUDA PyTorch/TorchVision and FlashAttention here.
python -m pip install -e '.[dev]'
```

The project installation does not automatically build the CUDA extensions or
download model weights. Benchmark scoring dependencies are separate from the
core inference installation.

## Native extensions

The first release uses the original `chipmunk` operator namespace. Follow the
[complete build instructions](../kernels/chipmunk/readme.md), including the
pinned ThunderKittens download:

```bash
bash scripts/download_thunderkittens.sh
python -m pip install -r kernels/chipmunk/requirements.txt
python -m pip install --no-build-isolation --no-deps -e kernels/chipmunk
bash models/hart/kernels/install.sh
```

The HART extension is required before importing HART blocks, even for the dense
HART baseline. FA3 and Quack are optional paths, not silently required by the
initial default presets. The provided D-CSLA presets select the Triton
FlexAttention backend; FA4 and TileLang are outside the initial validated claim.

Check the native modules in the GPU environment:

```bash
python -c "import torch, chipmunk; print(torch.cuda.get_device_name()); print(torch.ops.chipmunk.dense_attn)"
python -c "import hart_backend.fused_kernels"
```

A successful import is only an installation smoke check. Run numerical kernel
checks and fixed-seed model validation as described in
[reproduction](reproduction.md) before using a build for paper comparisons.

## Check CLI resolution without GPU execution

```bash
python examples/infer_image.py --help
python examples/infer_video.py --help
python examples/infer_hart.py --help
```

Add `--dry-run` to a complete example command to inspect its resolved paths,
model type and sparse configuration without loading Torch or checkpoints.
Dry runs do not establish that a checkpoint, compiled kernel or benchmark
installation is usable.

Prepare [model assets](model_weights.md), and keep credentials in environment
variables. Do not add personal tokens or local cache contents to the repository.
