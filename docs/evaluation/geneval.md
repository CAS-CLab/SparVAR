# GenEval

Status: migrated adapter; standalone GPU generation and GenEval scoring have not
yet been revalidated. Prompt data, detector and CLIP scoring assets are external
inputs. Their upstream revisions and checksums still need release verification.

## Generate images

Run from the repository root after core inference installation. The metadata
file must use GenEval JSONL entries, including each prompt's evaluation metadata.

```bash
SPARVAR_CONFIG_FILE=/path/to/explicit-image-recipe.yml \
LVAR_ATTN_BACKEND=chipmunk \
torchrun --standalone --nproc_per_node=1 evaluation/gen_eval/infer4eval_ddp.py \
  --model_type sparvar_infinity_2b \
  --model_path /path/to/infinity_2b_checkpoint \
  --vae_path /path/to/infinity_vae_d32reg.pth \
  --text_encoder_ckpt /path/to/flan-t5-xl \
  --metadata_file /path/to/evaluation_metadata.jsonl \
  --outdir work_dir/geneval/images --n_samples 4
```

Keep the same seeds, sample count, generation parameters and prompt-processing
policy for compared methods. Prompt rewriting is not required by the public
examples; record any preprocessed prompt dataset used for paper reproduction.

## Score

Prepare the upstream-compatible MMDetection/MMCV detector environment and
scoring checkpoints separately. The project core installation does not promise
to resolve those packages with every recent PyTorch build.

```bash
python evaluation/gen_eval/evaluate_images.py work_dir/geneval/images \
  --model-config /path/to/mask2former-config.py \
  --model-path /path/to/detector-checkpoint \
  --outfile work_dir/geneval/results.jsonl
```

The generation adapter creates numbered prompt directories containing
`metadata.jsonl` and `samples`. Here `--model-path` is the **detector**, not the
Infinity checkpoint. Retain the detector/CLIP versions, evaluation thresholds
and per-image results alongside any aggregate score. Consult the scorer's
actual `--help` in its installed environment for additional options; no newly
verified paper score is bundled with this draft.
