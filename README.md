# SparVAR (CVPR'2026) and SparVAR++

## Introduction

**SparVAR** is a training-free acceleration framework for visual autoregressive generation. It exploits attention sinks, cross-scale self-similarity, and spatial locality in next-scale prediction to reduce attention computation while preserving generation quality.

**SparVAR++** extends cross-scale sparse attention to spatiotemporal autoregressive video generation. This repository provides image and video inference, sparse attention kernels, latency profiling, and benchmark evaluation.

The attention modules include:

- **Cross-Scale Self-Similar Sparse Attention (CS4A):** selects important KV tokens at a decision scale and reuses the sparse pattern at subsequent scales.
- **Cross-Scale Local Sparse Attention (CSLA):** uses static block masks to exploit spatial locality in image generation.
- **Dynamic Cross-Scale Local Sparse Attention (D-CSLA):** transfers decision-scale attention statistics to construct content-adaptive block masks for video generation.

## Supported Models

| Task | Backbone | Dense model | SparVAR model |
|---|---|---|---|
| Text-to-image | [Infinity](https://github.com/FoundationVision/Infinity), 2B / 8B | `infinity_2b`, `infinity_8b` | `sparvar_infinity_2b`, `sparvar_infinity_8b` |
| Text-to-image | [HART](https://github.com/mit-han-lab/hart), 0.7B | `hart` | `sparvar_hart` |
| Text-to-video | [InfinityStar](https://github.com/FoundationVision/InfinityStar), 480p / 720p | `infinitystar_qwen8b` | `sparvar_infinitystar` |

SparVAR uses the pretrained backbone weights directly and does not require additional fine-tuning.

## Installation

### 1. Environment

- Linux with an NVIDIA Hopper GPU, such as H100 or H800, for the native CS4A kernels.
- Python **3.12+** and PyTorch **2.7+**.
- CUDA Toolkit **12.8** with `nvcc` and a compatible C++20 compiler.
- FlashAttention and the native extensions built below.

An example environment using the [official PyTorch CUDA 12.8 wheels](https://pytorch.org/get-started/previous-versions/#v2110):

```bash
git clone https://github.com/CAS-CLab/SparVAR.git
cd SparVAR

conda create -n sparvar python=3.12 -y
conda activate sparvar
python -m pip install --upgrade pip setuptools wheel

python -m pip install torch==2.11.0 torchvision==0.26.0 \
  --index-url https://download.pytorch.org/whl/cu128
python -m pip install -e .
MAX_JOBS=16 python -m pip install flash-attn --no-build-isolation
```

The CUDA Toolkit and host compiler are needed in addition to the PyTorch wheel. Check `nvcc --version` and set `CUDA_HOME` to the toolkit used for compilation if necessary.

### 2. Sparse attention kernels

From the repository root, download the pinned ThunderKittens dependency and build the CS4A extension:

```bash
bash scripts/download_thunderkittens.sh
python -m pip install -r kernels/chipmunk/requirements.txt
python -m pip install --no-build-isolation --no-deps -e kernels/chipmunk
```

The extension is imported as `chipmunk`. CSLA and D-CSLA use PyTorch FlexAttention; the provided D-CSLA configurations select its Triton backend. D-CSLA also uses the CS4A decision-stage operators, so it requires the Chipmunk extension.

See the [kernel build guide](kernels/chipmunk/readme.md) for compiler settings and operator checks.

### 3. HART extension

For HART inference, build its normalization and RoPE operators:

```bash
bash models/hart/kernels/install.sh
```

This extension is required for both dense HART and SparVAR HART.

## Model Zoo

Prepare the transformer checkpoint, visual tokenizer, and text encoder for the selected backbone.

| Backbone | Model assets |
|---|---|
| Infinity | [Transformer and image VAE weights](https://huggingface.co/FoundationVision/Infinity), [FLAN-T5-XL](https://huggingface.co/google/flan-t5-xl) |
| HART | [HART model release](https://huggingface.co/mit-han-lab/hart-0.7b-1024px) and its matching text encoder |
| InfinityStar | Matching 480p or 720p transformer, video VAE, and text encoder from the [InfinityStar project](https://github.com/FoundationVision/InfinityStar) |

For example, download the Infinity 2B assets with the [Hugging Face CLI](https://huggingface.co/docs/huggingface_hub/guides/cli):

```bash
hf download FoundationVision/Infinity \
  infinity_2b_reg.pth infinity_vae_d32reg.pth \
  --local-dir pretrained_models/infinity/Infinity

hf download google/flan-t5-xl \
  --local-dir pretrained_models/infinity/flan-t5-xl
```

A suggested local layout is:

```text
pretrained_models/
├── infinity/
│   ├── Infinity/
│   │   ├── infinity_2b_reg.pth
│   │   ├── infinity_vae_d32reg.pth
│   │   ├── infinity_8b_weights/
│   │   └── infinity_vae_d56_f8_14_patchify.pth
│   └── flan-t5-xl/
├── hart/
│   ├── hart-0.7b-1024px/
│   │   ├── llm/
│   │   └── tokenizer/
│   └── Qwen2-VL-1.5B-Instruct/
└── infinitystar/
    ├── infinitystar_8b_480p_weights/
    ├── infinitystar_8b_720p_weights/
    ├── infinitystar_videovae.pth
    └── text_encoder/flan-t5-xl-official/
```

The inference commands accept explicit paths, so the assets may also be stored elsewhere. Preserve all weight shards and their index files. HART uses EMA by default and expects `ema_model.bin` in the `llm` directory; use `--no-use-ema` to select non-EMA weights.

## Inference

Run the following commands from the repository root. Replace the checkpoint paths with your local paths.

### Infinity image generation

Generate a 1024 x 1024 image with SparVAR on Infinity 2B:

```bash
python examples/infer_image.py \
  --model sparvar_infinity_2b \
  --model-path pretrained_models/infinity/Infinity/infinity_2b_reg.pth \
  --vae-path pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth \
  --text-encoder pretrained_models/infinity/flan-t5-xl \
  --prompt "A red fox resting beside a lake" \
  --seed 42 --cfg 4 --tau 1 \
  --output output/infinity_2b.png
```

For Infinity 8B, use the sharded transformer weights and the matching patchified VAE:

```bash
python examples/infer_image.py \
  --model sparvar_infinity_8b \
  --model-path pretrained_models/infinity/Infinity/infinity_8b_weights \
  --vae-path pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth \
  --text-encoder pretrained_models/infinity/flan-t5-xl \
  --prompt "A red fox resting beside a lake" \
  --seed 42 --cfg 4 --tau 1 \
  --output output/infinity_8b.png
```

Select `--model infinity_2b` or `--model infinity_8b` for the corresponding dense baseline. The accelerated image models drop the unconditional branch at the last three scales by default; set `--drop-uncond 0` to retain it.

### HART image generation

```bash
python examples/infer_hart.py \
  --model sparvar_hart \
  --model-path pretrained_models/hart/hart-0.7b-1024px/llm \
  --vae-path pretrained_models/hart/hart-0.7b-1024px/tokenizer \
  --text-encoder pretrained_models/hart/Qwen2-VL-1.5B-Instruct \
  --options configs/sparvar/hart_csla.json \
  --prompt "A red fox resting beside a lake" \
  --seed 42 --cfg 4.5 \
  --output output/hart.png
```

Use `--model hart` for the dense baseline. Keep EMA selection consistent when comparing dense and accelerated outputs.

### InfinityStar video generation

Generate an 81-frame video at 16 FPS with SparVAR++ on the 480p backbone:

```bash
python examples/infer_video.py \
  --model sparvar_infinitystar --resolution 480p --attention cs4a \
  --model-path pretrained_models/infinitystar/infinitystar_8b_480p_weights \
  --vae-path pretrained_models/infinitystar/infinitystar_videovae.pth \
  --text-encoder pretrained_models/infinitystar/text_encoder/flan-t5-xl-official \
  --prompt "A red fox walking beside a lake" \
  --seed 42 --drop-uncond auto \
  --output output/infinitystar_480p.mp4
```

- For 720p, set `--resolution 720p` and provide the matching `infinitystar_8b_720p_weights` directory.
- Use `--attention dcsla` to select dynamic block-sparse attention.
- Use `--model infinitystar_qwen8b` for the dense baseline, omitting `--drop-uncond auto`.
- `--drop-uncond auto` starts unconditional-branch removal at the sparse decision step. Set `--drop-uncond 0` to keep both branches.
- Add `--low-vram` to offload the text encoder to CPU during the autoregressive loop.

The video CLI uses APG by default. Adjust guidance and temperature with `--cfg`, `--tau-image`, and `--tau-video`.

### Sparse attention configurations

| Model / attention | Configuration |
|---|---|
| Infinity 2B | [infinity_2b.yml](configs/sparvar/infinity_2b.yml) |
| Infinity 8B | [infinity_8b.yml](configs/sparvar/infinity_8b.yml) |
| HART CSLA | [hart_csla.json](configs/sparvar/hart_csla.json) |
| InfinityStar 480p CS4A | [infinitystar_480p_cs4a.yml](configs/sparvar/infinitystar_480p_cs4a.yml) |
| InfinityStar 720p CS4A | [infinitystar_720p_cs4a.yml](configs/sparvar/infinitystar_720p_cs4a.yml) |
| InfinityStar 480p D-CSLA | [infinitystar_480p_dcsla.yml](configs/sparvar/infinitystar_480p_dcsla.yml) |
| InfinityStar 720p D-CSLA | [infinitystar_720p_dcsla.yml](configs/sparvar/infinitystar_720p_dcsla.yml) |

The image and video CLIs select a configuration automatically from the model, resolution, and attention options. Pass `--sparse-config /path/to/config.yml` to override it. HART accepts its option file through `--options`.

Add `--dry-run` to an inference command to inspect its configuration without loading weights. Each generation also saves a `.run.json` file alongside the output with the selected arguments and configuration. Use `--help` to list all command-line options.

## Evaluation

The repository provides benchmark generation and scoring tools for image and video generation:

| Benchmark | Guide |
|---|---|
| GenEval | [Generation and object-based evaluation](docs/evaluation/geneval.md) |
| DPG-Bench | [Dense prompt evaluation](docs/evaluation/dpg_bench.md) |
| HPS v2.1 | [Human preference evaluation](docs/evaluation/hpsv2.md) |
| ImageReward / CLIP | [Preference and text-image scoring](docs/evaluation/image_reward.md) |
| VBench | [Video quality, semantic alignment, and paired metrics](docs/evaluation/vbench.md) |

Generation scripts support `torchrun` for distributing prompts across GPUs. Install the benchmark-specific scorer dependencies and checkpoints described in each guide.

For low-level metrics such as PSNR, SSIM, and LPIPS:

```bash
python -m pip install -e '.[metrics]'
python -m evaluation.vbench.eval_low_level_metrics --help
```

## Latency Profiling

```bash
python -m tools.latency_profile_infinity --help
python -m tools.latency_profile_hart --help
python -m tools.latency_profile_infistar --help
```

The profiling tools expose model paths, batch size, warmup, and sample-count options. Keep prompts, seeds, guidance settings, EMA selection, and attention backends consistent between dense and sparse comparisons. InfinityStar scale-level latency includes all refinement repeats of the scale.

## Acknowledgements and License

This project builds on [Infinity](https://github.com/FoundationVision/Infinity), [HART](https://github.com/mit-han-lab/hart), [InfinityStar](https://github.com/FoundationVision/InfinityStar), and [ThunderKittens](https://github.com/HazyResearch/ThunderKittens). We thank their authors for making their work available.

See [LICENSE](LICENSE) and [third-party notices](THIRD_PARTY_NOTICES.md) for the license terms and acknowledgements of the included components.
