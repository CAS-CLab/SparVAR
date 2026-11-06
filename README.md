# SparVAR (CVPR'2026) and SparVAR++

Training-free sparse attention for next-scale visual autoregressive generation.
This repository contains the SparVAR image path and the SparVAR++ InfinityStar
video path migrated from the Lvar research codebase.

**Status: local migration draft.** GPU inference, kernel compilation and paper
metrics have not yet been independently reproduced from this standalone checkout.
The supplied presets are derived from source defaults; they are starting recipes,
not newly verified or frozen paper configurations. No paper speedup or quality
number is claimed by the local migration checks.

## Included model paths

| Task | Dense baseline | Accelerated model |
|---|---|---|
| Infinity images | `infinity_2b`, `infinity_8b` | `sparvar_infinity_2b`, `sparvar_infinity_8b` |
| HART images | `hart` | `sparvar_hart` |
| InfinityStar video | `infinitystar_qwen8b` | `sparvar_infinitystar` |

CS4A performs decision-scale token selection and cross-scale sparse index
mapping. Static CSLA and dynamic D-CSLA provide block-sparse execution paths.
Other research backbones, competing acceleration methods and training pipelines
are outside the first-release entrypoints.

The current native production stack targets **NVIDIA Hopper (SM90a; H100/H800)**.
Other GPU architectures and backend combinations have not been validated for
this release. HART additionally requires its own CUDA extension.

## Installation and weights

Follow [installation](docs/installation.md), then prepare the matching backbone,
visual tokenizer and text encoder described in [model weights](docs/model_weights.md).
No model weights, personal caches, API credentials or private research documents
are included. Download locations and checkpoint hashes still need release-time
verification; the examples below use explicit local paths.

## Quick start

Run commands from the repository root. The example CLIs accept `--help` and
`--dry-run` without loading Torch or model weights; dry runs only print the
resolved execution plan.

### Infinity image

```bash
python examples/infer_image.py \
  --model sparvar_infinity_2b \
  --model-path /path/to/infinity_2b_checkpoint \
  --vae-path /path/to/infinity_vae_d32reg.pth \
  --text-encoder /path/to/flan-t5-xl \
  --prompt "A red fox resting beside a lake" \
  --output output/image.png
```

The image CLI also accepts `infinity_2b`, `infinity_8b` and
`sparvar_infinity_8b`. Use `--sparse-config /path/to/config.yml` to select an
explicit sparse recipe. Match the checkpoint size to the model type.

### InfinityStar video

```bash
python examples/infer_video.py \
  --model sparvar_infinitystar --resolution 480p --attention cs4a \
  --model-path /path/to/infinitystar_480p_checkpoint \
  --vae-path /path/to/infinitystar_videovae.pth \
  --text-encoder /path/to/matching-video-text-encoder \
  --prompt "A red fox walking beside a lake" \
  --output output/video.mp4 --drop-uncond auto
```

Use `--resolution 720p` with its matching checkpoint, or `--attention dcsla`
for the dynamic block-sparse recipe. `--model infinitystar_qwen8b` selects the
dense video baseline. `--drop-uncond auto` is anchored to the sparse decision
step for the accelerated model; it is not equivalent to a full-CFG dense run.
See [reproduction](docs/reproduction.md) before comparing timings or fidelity.

### HART image

```bash
python examples/infer_hart.py \
  --model sparvar_hart \
  --model-path /path/to/hart/llm \
  --vae-path /path/to/hart/tokenizer \
  --text-encoder /path/to/matching-hart-text-encoder \
  --options configs/sparvar/hart_csla.json \
  --prompt "A red fox resting beside a lake" \
  --output output/hart.png
```

The HART loader uses EMA weights by default and therefore expects the matching
`ema_model.bin`. Select `--no-use-ema` only when intentionally evaluating the
non-EMA model, and record that choice. `--model hart` selects the dense baseline.

## Evaluation and reproducibility

- [Reproduction status, configurations and timing](docs/reproduction.md)
- [GenEval](docs/evaluation/geneval.md)
- [DPG-Bench](docs/evaluation/dpg_bench.md)
- [HPS v2.1](docs/evaluation/hpsv2.md)
- [ImageReward / CLIP](docs/evaluation/image_reward.md)
- [VBench and paired video metrics](docs/evaluation/vbench.md)
- [Native kernel build](kernels/chipmunk/readme.md)

Benchmark generation and scoring have separate dependencies. Prepare their
prompt data and scorer checkpoints before starting a full run. An example
image or a successful CLI dry run does not establish a benchmark result.

## Attribution and licensing

The pre-existing [MIT license](LICENSE) is retained. The migrated Lvar source
license is recorded separately in [LICENSES/Lvar-Apache-2.0.txt](LICENSES/Lvar-Apache-2.0.txt).
This checkout includes components carrying their own notices; the root license
does not relicense them. See [third-party notices](THIRD_PARTY_NOTICES.md).
The source attribution and license inventory require final review before public
publication. Paper citation metadata will be added after its final form is verified.
