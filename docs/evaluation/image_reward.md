# ImageReward and CLIP

Status: migrated adapters; standalone generation and scoring are pending.
Generation and scoring are separate steps. Prepare the benchmark prompt JSON
and scorer cache before running them.

## Generate

```bash
SPARVAR_CONFIG_FILE=/path/to/explicit-image-recipe.yml \
LVAR_ATTN_BACKEND=chipmunk \
torchrun --standalone --nproc_per_node=1 evaluation/image_reward/infer4reward_ddp.py \
  --model_type sparvar_infinity_2b \
  --model_path /path/to/infinity_2b_checkpoint \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --metadata_file /path/to/benchmark-prompts.json \
  --outdir work_dir/image_reward/images --n_samples 10
```

The adapter writes `metadata.jsonl` as a **JSON array**, despite the filename.
The scorer uses `json.load`; do not convert that file to line-delimited JSON.
Each entry records a prompt and its `gen_image_paths`.

## Score from a local cache

Install the ImageReward scoring package in a compatible environment. Its cache
must contain nonempty `ImageReward.pt`, `med_config.json` and `ViT-L-14.pt` files.
Required tokenizer/Hugging Face assets must already be cached as well.

```bash
python evaluation/image_reward/cal_imagereward.py \
  --meta_file work_dir/image_reward/images/metadata.jsonl \
  --cache_dir /path/to/ImageReward-cache
```

Cache selection is `--cache_dir`, then `IMAGEREWARD_CACHE`, then
`~/.cache/ImageReward`. The helper sets `HF_HUB_OFFLINE=1` and
`TRANSFORMERS_OFFLINE=1` before importing ImageReward and fails early if one of
the three explicit assets is missing. It does not download or validate the
identity of those assets.

The scorer saves `image_reward_res.json` beside the metadata file and reports
ImageReward and CLIP averages. Record checkpoint identities and the scoring
package version before comparing results; matched filenames alone are not a
numerical validation.
