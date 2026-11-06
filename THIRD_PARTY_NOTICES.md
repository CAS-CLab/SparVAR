# Third-party source and license inventory

This is an initial inventory for the local migration draft, based on source
headers, references and build metadata present in this checkout. It is not a
completed provenance audit or a blanket relicensing statement. Preserve the
original per-file notices. Complete the copyright, license-text and source
revision review before publishing a release.

## Repository-level records

- The pre-existing `LICENSE` contains the MIT license and CAS CLab attribution.
- `LICENSES/Lvar-Apache-2.0.txt` records the Apache-2.0 license supplied with the
  migrated Lvar codebase.
- These records do not replace component-specific licenses or the licenses of
  separately downloaded models, datasets and benchmark assets.

## Source components

| Component | Files or areas | Evidence retained in the source |
|---|---|---|
| Lvar research code | Sparse model implementation, loaders and evaluation integration | Migrated implementation and separately retained Lvar Apache-2.0 text |
| FoundationVision Infinity / InfinityStar | `models/infinity`, `models/infinitystar`, image/video tokenizer and schedules | FoundationVision adaptation lineage; many `models/videovae` files explicitly carry `Copyright (c) 2025 FoundationVision` and `SPDX-License-Identifier: MIT` |
| HART | `models/hart` and hybrid tokenizer integration | HART implementation lineage; check each copied file and its upstream notices rather than assigning one inferred license to the directory |
| Chipmunk / ThunderKittens | `kernels/chipmunk` | Chipmunk-derived sparse attention implementation; ThunderKittens headers are downloaded separately at the revision recorded in the downloader |
| MAR | `models/diffusion/diffloss.py` | File explicitly states adaptation from `LTH14/mar` |
| OpenAI GLIDE / guided-diffusion / improved-diffusion | `models/diffusion/{gaussian_diffusion,diffusion_utils,respace}.py` | Source links and adaptation references retained in file headers |
| CompVis taming-transformers / stable-diffusion | `models/vae/basic_vae.py`, `models/hart/hart_autoencoder.py`, discriminator support | Source URLs and adaptation comments retained in the implementation |
| Kakao Brain discriminator support | `losses/vqgan` | `losses/vqgan/__init__.py` carries a Kakao Brain copyright notice and Apache-2.0 notice |
| NVIDIA TransformerEngine | HART fused RoPE CUDA sources | Original NVIDIA copyright and modification notices retained in the CUDA files |
| PyTorch / NVIDIA Apex / Ray / Hugging Face Transformers | HART dispatch/build utilities and rotary helpers | Specific adaptation/source URLs retained in comments; HART kernel package metadata declares Apache 2.0 |
| Black Forest Labs FLUX autoencoder reference | `models/vae/flux_vqgan.py` | Implementation contains a specific upstream source-reference link; review the relevant upstream scope before publication |
| GenEval, DPG-Bench, HPS, ImageReward and VBench | `evaluation` adapters and separately installed scorers | Preserve benchmark-specific notices and prepare externally distributed assets under their own terms |

The HART discriminator support is included because the inference checkpoint
loader instantiates `HARTAutoEncoderWithDisc` before extracting its VAE. Its
presence is not a promise to expose a training pipeline.

## Separately installed dependencies

PyTorch, TorchVision, FlashAttention, Transformers, timm, Triton, FlexAttention
backends and benchmark scoring packages retain their own licenses and notices.
Installing this repository does not grant additional rights to downloaded
weights or data. Keep the ThunderKittens checkout's license and notices with the
headers used for a native build.

## Review still required before publication

Verify the copied-source revisions, exact applicable license text and required
attributions for every source component; include missing license/notice files
without overwriting existing headers. Verify model and benchmark distribution
terms independently. Record the final dependency versions and file-level
provenance in the published release artifact. No source-license conflict has
been resolved by changing the root license or package name in this migration.
