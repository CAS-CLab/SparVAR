# Model assets

SparVAR and SparVAR++ use pretrained backbone assets; no new fine-tuned
acceleration weights are introduced by the default inference paths. The
repository contains code and configurations, not the model weights or tokenizer
caches.

**Release status:** exact upstream download URLs, checkpoint revisions and SHA256
hashes have not yet been verified for this standalone migration. Prepare the
assets from their original model releases and verify their identity before
reporting results. The paths below describe loader expectations, not new
download endpoints.

| Model | `--model-path` | `--vae-path` | `--text-encoder` |
|---|---|---|---|
| Infinity 2B | Matching 2B autoregressive checkpoint | Matching BSQ image VAE, such as the source recipe's d32 regular tokenizer | Matching FLAN-T5-XL model/tokenizer directory |
| Infinity 8B | Matching 8B checkpoint or weight-shard directory supported by the loader | Matching image VAE for that recipe | Matching FLAN-T5-XL directory |
| InfinityStar 480p | Matching 480p autoregressive weight directory | Matching InfinityStar video VAE checkpoint | Matching video text encoder/tokenizer directory |
| InfinityStar 720p | Matching 720p autoregressive weight directory | Matching InfinityStar video VAE checkpoint | Matching video text encoder/tokenizer directory |
| HART | HART `llm` directory with config and model weights | HART tokenizer directory with its config and weights | The text encoder/tokenizer required by that HART checkpoint |

For HART, keep the original checkpoint layout. The model configuration and the
explicit tokenizer path must agree. The loader instantiates the original
VAE-with-discriminator checkpoint wrapper and then extracts its VAE; do not
strip arbitrary checkpoint keys or omit source wrapper code to save space.

HART uses EMA by default. Supply the matching `ema_model.bin` in the model
directory, or intentionally select `--no-use-ema` and report a non-EMA run.
For sharded models, preserve the original index and all referenced shard files.
A directory name alone is not evidence of model identity.

Before recording a release result, retain this manifest with the run:

- Model family, size and resolution; original release/source identifier.
- Config and tokenizer revisions; all weight and index filenames.
- SHA256 of every downloaded weight/config file used by the run.
- EMA selection, precision, sparse recipe and text-encoder/tokenizer versions.
- Public repository revision and the installed Torch/CUDA/native-kernel versions.

For example, generate checksums over selected local files with `sha256sum` in
the GPU environment. Do not publish a made-up checksum or substitute an
unverified mirror URL for an unresolved original asset.

Benchmark scorers have additional checkpoints. See the individual
[evaluation guides](reproduction.md#evaluation); these are not interchangeable
with the generation backbones listed above.
