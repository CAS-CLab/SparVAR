# Local migration status

This is an uncommitted release draft on `codex/tpami-code-release`. No source
repository files were modified, and no remote execution or push was performed.
The target repository retains its original history and MIT LICENSE; the copied
Lvar license is preserved in `LICENSES/Lvar-Apache-2.0.txt`. See
`THIRD_PARTY_NOTICES.md` before deciding the final public licensing policy.

## Source snapshot and scope

- Source snapshot: `a583911b9b7e0c3994c34cdf0272e8519513a166`, plus the explicitly
  selected working-tree versions recorded in `release_manifest.json`.
- 227 whitelisted existing files: 202 retained byte-for-byte, 25 adapted.
- Added public image/HART/video CLIs, seven labelled migration presets,
  environment-only HF configuration, fixed-revision dependency retrieval,
  package resource declarations and public usage guides.
- No source Git history, private research/paper docs, credentials, weights,
  generated samples, nested checkout, competitor implementations or GPU binaries
  were copied. The approved small benchmark inputs are included.

## Local validation

- Three CLI help commands and ten recipe dry-runs succeed without GPU imports.
- All 198 Python source files pass AST parsing under Python 3.12.13.
- Selected CPU geometry/mask/HART tests: **58 passed, 1 skipped**. The warning
  about unfused FlexAttention is from the tiny CPU reference test, not an
  execution path advertised for large-scale generation.
- Profiling tests: **8 passed**.
- Pure-Python wheel builds without downloading dependencies. Fifteen required
  config/data resources are present. An isolated wheel installation resolves
  its own packages and runs all three CLI dry-runs.
- Twelve numerical model/attention files remain byte-identical to the source;
  image/video SparVAR subclass inference methods have identical ASTs.
- Native downloader control-flow mocks and shell syntax checks pass. CUDA
  source math is unchanged. No native extension was actually built.

## Still required before a public release

1. Build the pinned ThunderKittens and HART extensions from a fresh H800 setup.
2. Test actual checkpoints and fixed-seed dense/sparse image, HART and video
   generation from this independent repository.
3. Validate scorers, latency scope and paper results, then freeze accepted
   presets. Current defaults-derived presets are not certified paper recipes.
4. Verify exact weight sources/revisions/checksums, supported environment
   versions and final attribution/license policy.

No CPU or packaging check substitutes for these GPU validations. Remote runs
require a separate user instruction. All present changes remain unstaged.
