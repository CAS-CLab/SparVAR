# HPS v2.1

Status: migrated adapter; generation and HPS v2.1 scores have not been
independently reproduced from the standalone checkout. The adapter obtains its
prompt sets through the installed `hpsv2` package, which also needs its scorer
checkpoint/cache. Prepare and record that package version and those assets.

```bash
SPARVAR_CONFIG_FILE=/path/to/explicit-image-recipe.yml \
LVAR_ATTN_BACKEND=chipmunk \
torchrun --standalone --nproc_per_node=1 evaluation/hpsv2/infer4hpsv2_ddp.py \
  --model_type sparvar_infinity_2b \
  --model_path /path/to/infinity_2b_checkpoint \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --outdir work_dir/hpsv2/images --n_samples 1
```

The adapter generates the style-specific image layout and then invokes
`hpsv2.evaluate(..., hps_version="v2.1")`. Use `--skip_metrics` to generate without
running the scorer. This flag does not install or validate the scoring assets;
the adapter still imports `hpsv2` to obtain prompts.

Preserve prompt ordering, seeds and sample count across methods. Record the
HPS package/checkpoint version and style-level outputs; scores from different
HPS versions are not the same evaluation protocol. If scorer dependency
constraints conflict with generation, score the saved image layout in a
separate compatible environment using that package's documented interface.
