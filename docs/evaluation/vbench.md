# VBench and paired video fidelity

Status: migrated video generation and aggregation code; no standalone VBench
score or video latency result has been revalidated. The first-release inference
path is text-to-video with the InfinityStar elegant schedule. Other inherited
adapter branches are not additional validated model claims.

## Generate the benchmark videos

Prepare a VBench-compatible prompt JSON containing the intended prompt/seed
protocol. The source adapter supports a preprocessed prompt suite; verify its
provenance and checksum rather than assuming the default file is bundled.

```bash
SPARVAR_CONFIG_FILE=/path/to/explicit-480p-video-recipe.yml \
torchrun --standalone --nproc_per_node=1 evaluation/vbench/infer4vbench_ddp.py \
  --model_type sparvar_infinitystar \
  --model_path /path/to/infinitystar_480p_checkpoint \
  --vae_path /path/to/infinitystar_videovae.pth \
  --text_encoder_ckpt /path/to/matching-video-text-encoder \
  --pn 0.40M --prompt_json /path/to/vbench-prompts-with-seeds.json \
  --output_root work_dir/vbench/sparvar_480p
```

For 720p, use `--pn 0.90M`, the matching 720p checkpoint and the corresponding
recipe. Record frames, FPS, duration, repeats, CFG/APG and uncond removal. The
adapter performs per-prompt DDP work and creates benchmark-oriented output
links/layout; inspect the saved layout before launching the external scorer.

## Score and aggregate

Install the VBench scoring package and its model assets separately. Run its
scorer against the generated videos using the version-matched upstream
interface; this repository's generation adapter does not replace that scorer.
Record the scorer revision and dimension-level JSON results.

After dimension scores are available:

```bash
python evaluation/vbench/cal_vbench_final_score.py \
  --json_files_path /path/to/dimension-result-jsons --mode t2v
```

The aggregate script is not a substitute for computing each required dimension.
Keep the raw dimension results and check that none are missing.

## Paired fidelity

For PSNR/SSIM-style comparisons, generate dense and sparse videos from identical
prompts, seeds, checkpoints and reference settings. Keep CFG/uncond decisions
and backend differences explicit. Saved raw tensors, where supported, avoid
conflating video codec error with model error.

```bash
python evaluation/vbench/eval_low_level_metrics.py \
  --baseline-root /path/to/dense-results \
  --candidate-root /path/to/sparse-results --help
```

Inspect its input-format and metric options in the scoring environment before
running a full comparison. No metric output is implied by `--help`. A reported
scale-level latency must include every repeat of that scale, with text `t0`
reported separately.
