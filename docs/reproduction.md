# Reproduction and validation status

This is a **local migration draft**, not a record of independent reproduction.
The standalone repository has not yet completed a clean Hopper kernel build,
fixed-seed GPU inference, full benchmark scoring or paper latency verification.
Local syntax, CLI and configuration checks cannot establish those results.

## Recipes

Seven initial sparse presets are derived from the source working defaults:
two Infinity image sizes, four video combinations (480p/720p x CS4A/D-CSLA),
and one HART CSLA option file. They are not frozen paper configurations.

| Path | Initial recipe choices | Validation still required |
|---|---|---|
| Infinity image 2B/8B | Source-default image CS4A geometry and cache settings | Confirm the intended paper mapping, layer split, CFG and generated output |
| Video CS4A 480p | Decision scale 25; explicit 480p recipe | Verify schedule, repeats, actual sparse steps and expected quality |
| Video CS4A 720p | Decision scale 27; explicit 720p recipe | Verify matching 720p checkpoint and late-scale execution |
| Video D-CSLA 480p/720p | D-CSLA enabled, output cache disabled, FlexAttention Triton backend | Verify block selection, geometry, numerical behavior and scorer results |
| HART CSLA | JSON options derived from the source HART CSLA defaults | Confirm layer/mask preset, EMA checkpoint and matched dense reference |

The original image default uses identity/center-union/cumulative index behavior;
its existence must not be presented as evidence that the paper's footprint
recipe has already been reproduced. Changes in YAML or environment overrides
must be recorded explicitly. `SPARVAR_CONFIG_FILE` selects an explicit sparse
YAML for legacy generation/evaluation entrypoints. The public example CLIs
resolve their chosen recipe in the execution plan.

## Validation order

1. Build native extensions in a clean environment using only this checkout.
2. Run the small kernel correctness checks from the kernel build guide. Check
   grouped KV, variable counts and partial KV tails used by the video path.
3. Run one fixed-seed dense and accelerated sample for each supported model,
   resolution and sparse recipe. Preserve outputs and resolved configurations.
4. Run benchmark generation and scoring, recording all external asset versions.
5. Repeat latency measurements after warmup with a stated measurement boundary.

Keep the original research environment out of `PYTHONPATH`. Ensure imported
`models`, `tools`, `utils` and native extensions belong to the intended release
environment. A working editable Lvar install can otherwise hide missing files.

## Timing and fidelity protocol

Report GPU model, model size, resolution, batch size, precision, backend,
checkpoint, seed, warmup/sample counts and whether model loading, text encoding,
VAE decoding and file encoding are included. For InfinityStar, a scale-level
measurement includes **all repeats of that visual scale**; text `t0` is separate.

State CFG/APG, uncond removal and skipped scales. `--drop-uncond auto` starts a
single-branch suffix at the sparse decision step; this is a changed generation
setting, not merely a kernel replacement. Compare it to the intended reference.
Keep HART EMA selection identical across paired runs.

For paired PSNR/SSIM-style comparisons, use identical prompts, seeds, model
weights and matched baseline kernel/rounding settings. Set
`LVAR_ATTN_BACKEND` explicitly when comparing Infinity dense paths. Changing a
kernel or fused RoPE path may alter autoregressive samples even with the same
seed. Do not infer output equivalence from an equal CLI plan or a microbenchmark.

## Evaluation

The benchmark adapters are migrated code paths, not new verified score tables.
They use legacy underscore options such as `--model_type`, `--model_path`,
`--vae_path` and `--text_encoder_ckpt`; the public example CLIs use their own
hyphenated options. Run DDP generation with `torchrun`, even for a one-GPU smoke
run. Choose benchmark-specific scorer environments where dependencies conflict.

- [GenEval](evaluation/geneval.md)
- [DPG-Bench](evaluation/dpg_bench.md)
- [HPS v2.1](evaluation/hpsv2.md)
- [ImageReward / CLIP](evaluation/image_reward.md)
- [VBench](evaluation/vbench.md)

Before a public release, replace this status with dated evidence: source
revision, exact configurations, asset checksums, tested software/hardware,
passed checks and result files. Preserve unresolved limitations explicitly.
