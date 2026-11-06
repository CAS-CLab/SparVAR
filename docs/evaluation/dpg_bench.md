# DPG-Bench

Status: migrated generation/scoring adapters; standalone GPU and scorer results
are pending. Prepare the official-compatible prompt dataset/CSV and VQA scorer
assets separately, and record their revisions before reporting results.

## Generate

```bash
SPARVAR_CONFIG_FILE=/path/to/explicit-image-recipe.yml \
LVAR_ATTN_BACKEND=chipmunk \
torchrun --standalone --nproc_per_node=1 evaluation/dpg_bench/infer4dpg_ddp.py \
  --model_type sparvar_infinity_2b \
  --model_path /path/to/infinity_2b_checkpoint \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --outdir work_dir/dpg_bench/images --n_samples 4
```

The adapter expects its DPG prompt assets under the benchmark's dataset layout;
verify those files are present before a full run. Its output combines four
samples into the layout consumed by the scorer. Do not silently change the
sample count when comparing runs.

## Score

The mPLUG route imports ModelScope at scoring time. Install a compatible VQA
scorer environment separately from generation if its dependency requirements
differ from the current native CUDA stack.

```bash
python evaluation/dpg_bench/compute_dpg_bench.py \
  --image-root-path work_dir/dpg_bench/images --resolution 1024 \
  --csv /path/to/dpg_bench.csv --pic-num 4 --vqa-model mplug \
  --res-path work_dir/dpg_bench/scores.txt
```

The scorer also supports independent `--num-shards` / `--shard-index` runs,
`--result-json`, and `--merge-result-jsons`. Preserve raw shard results and use
the same CSV and resolution for the merge. Exact download URLs, scorer model
revisions and resulting numerical agreement still need validation in this
standalone checkout.
