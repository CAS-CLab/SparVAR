"""CUDA-graph replay of the dense transformer block stack at the small image scales.

At the first ten scales of Infinity-8B (1 .. 1024 tokens) every (layer, scale) costs
about 1.4 ms of CPU dispatch for roughly 0.2 ms of GPU work, so the GPU idles and the
whole stack is launch-bound. Capturing the 40-block stack of each such scale into one
CUDA graph and replaying it removes that dispatch floor. The graph replays exactly the
kernels the eager path launches, on the same static inputs, so outputs are bit-identical.

Scope (checked per call; anything else runs the eager loop):
  * scale index <= max_si: all layers take the dense-attention path (no CS4A decision /
    sparse scale, no CSLA scale), and the uncond branch has not been dropped yet
  * no per-layer CFG insertion, no flex attn_fn, no padding, batch/dtype/shape as captured

Mechanics:
  * one capture image (after SPARVAR_CUDA_GRAPH_WARM_IMAGES eager images) captures the
    graphs for si = 0..max_si in order, sharing one memory pool, so the KV cache that
    graph s appends flows into graph s+1 exactly as in eager mode; replays must keep
    the same order, which the autoregressive loop does by construction
  * static inputs: x per scale, the AdaLN condition, and the cross-attention text kv.
    The text kv projection (per layer) is computed eagerly on the exact-length text and
    copied into a static buffer that the graph reads through the CrossAttention
    projection cache; padding rows after the real rows are never read because the
    varlen kernel follows cu_seqlens (max_seqlen_k only bounds the loop)
  * after the last graph of an image the python-side KV cache references of every layer
    are pointed at the graph-owned tensors so the eager large scales continue from them

Enable with SPARVAR_CUDA_GRAPH=1 (default off). SPARVAR_CUDA_GRAPH_MAX_SI overrides the
largest captured scale index; SPARVAR_CUDA_GRAPH_MAX_ROWS (default 4 = batch 2 with CFG) is the
largest batch that takes the graph path.
"""
import os
from typing import List, Optional, Tuple

import torch
import torch.nn.functional as F

from models.infinity import basic_infinity as _bi

_ENABLED = os.environ.get('SPARVAR_CUDA_GRAPH', '0').lower() in ('1', 'true', 'yes', 'on')
_MAX_SI_ENV = os.environ.get('SPARVAR_CUDA_GRAPH_MAX_SI', '')
_WARM_IMAGES = int(os.environ.get('SPARVAR_CUDA_GRAPH_WARM_IMAGES', '2'))
# largest batch (rows incl. the CFG branch, i.e. 2B) that uses graphs; bigger batches are already
# GPU-bound per scale and their graph pools (which cannot be shared with the eager allocator)
# push Infinity-8B batch 4 over 80 GB
_MAX_ROWS = int(os.environ.get('SPARVAR_CUDA_GRAPH_MAX_ROWS', '4'))
_TEXT_MAXLEN = 512   # T5 tokenizer max_length used by encode_prompts


def cuda_graph_enabled() -> bool:
    return _ENABLED


class BlockStackGraphRunner:
    def __init__(self, model, attn_config: dict):
        self.model = model
        decision_scale = int(attn_config.get('decision_scale', 10))
        csla_scales = [int(s) for s in (attn_config.get('csla_scales') or [])]
        # every captured scale must run the dense path in every layer
        max_si = decision_scale - 1
        if csla_scales:
            max_si = min(max_si, min(csla_scales) - 1)
        if _MAX_SI_ENV:
            max_si = min(max_si, int(_MAX_SI_ENV))
        self.max_si = max_si
        self.blocks = list(model.unregistered_blocks)

        self.images_seen = 0
        self.capture_image = False      # True while the graphs are being captured
        self.captured = False
        self.failed = False
        self.pool = None
        self.graphs = {}                # si -> torch.cuda.CUDAGraph
        self.x_static = {}              # si -> static input tensor
        self.out_static = {}            # si -> graph output tensor (pool memory)
        self.cond_static = None
        self.text_static = None
        self.cu_static = None
        self.max_seqlen_static = None
        self.proj_static: List[torch.Tensor] = []
        self.kv_refs = None             # [(cached_k, cached_v) per layer] after the last captured graph
        self.rope_refs = None
        self.batch = None
        self.graph_sis_this_image: List[int] = []

    # ------------------------------------------------------------------ eligibility
    def eligible(self, si, x, ca_kv, cfg_layers, drop_uncond_from, attn_fn, need_to_pad, lvl_only_first_block) -> bool:
        if self.failed or self.model.training or si > self.max_si or x.shape[0] > _MAX_ROWS:
            return False
        if cfg_layers or attn_fn is not None or need_to_pad or not lvl_only_first_block:
            return False
        if drop_uncond_from is not None and si >= drop_uncond_from:
            return False
        kv_compact = ca_kv[0]
        if self.captured:
            if x.shape[0] != self.batch or kv_compact.shape[0] > self.text_static.shape[0]:
                return False
            if si in self.x_static and (x.shape != self.x_static[si].shape or x.dtype != self.x_static[si].dtype):
                return False
        return True

    # ------------------------------------------------------------------ helpers
    def _blocks(self, si, x, cond, ca_kv, scale_schedule, rope_grid, spattn_fn):
        """The plain block loop of autoregressive_infer_cfg for a scale without CFG insertion."""
        layer_idx = 0
        model = self.model
        chunks = model.block_chunks if model.num_block_chunks > 1 else [model]
        for b in chunks:
            mods = b.module if model.num_block_chunks > 1 else model.blocks
            for m in mods:
                x = m(x=x, cond_BD=cond, ca_kv=ca_kv, attn_bias_or_two_vector=None,
                      attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_grid,
                      scale_ind=si, layer_ind=layer_idx, spattn_fn=spattn_fn)
                layer_idx += 1
        return x

    def _begin_image(self, cond, ca_kv):
        """Stage the per-image inputs into the static buffers (allocating them on the capture image)."""
        kv_compact, cu_seqlens_k, _ = ca_kv
        n_rows = kv_compact.shape[0]
        if self.cond_static is None:
            self.cond_static = cond.clone()
            cap = max(n_rows, self.batch * _TEXT_MAXLEN)
            self.text_static = torch.zeros((cap,) + tuple(kv_compact.shape[1:]), dtype=kv_compact.dtype, device=kv_compact.device)
            self.cu_static = cu_seqlens_k.clone()
            self.max_seqlen_static = _TEXT_MAXLEN
        else:
            self.cond_static.copy_(cond)
        self.text_static[:n_rows].copy_(kv_compact)
        self.cu_static.copy_(cu_seqlens_k)
        # text kv projection per layer, eager and on the exact rows, written into the
        # static buffer the captured cross-attention reads (bit-identical to the eager GEMM)
        for li, blk in enumerate(self.blocks):
            ca = blk.ca
            proj = F.linear(kv_compact, weight=ca.mat_kv.weight, bias=ca._kv_bias()).view(n_rows, 2, ca.num_heads, ca.head_dim)
            if len(self.proj_static) <= li:
                buf = torch.zeros((self.text_static.shape[0],) + tuple(proj.shape[1:]), dtype=proj.dtype, device=proj.device)
                self.proj_static.append(buf)
            self.proj_static[li][:n_rows].copy_(proj)
            ca._kvproj_cache = (self.text_static, self.proj_static[li])
        self.graph_sis_this_image = []

    def _static_ca_kv(self):
        return (self.text_static, self.cu_static, self.max_seqlen_static)

    def ensure_eager_ready(self):
        """Called before an eager scale: if graphs ran earlier in this image, point the
        python-side KV cache of every layer at the tensors the last replayed graph produced."""
        if not self.graph_sis_this_image:
            return
        if self.graph_sis_this_image[-1] != self.max_si:
            # eligibility is uniform over si 0..max_si within one image, so this cannot happen
            # unless the caller changed shapes mid-image; the intermediate caches are not kept
            raise RuntimeError(f'[sparvar-cuda-graph] graph chain stopped at si={self.graph_sis_this_image[-1]} '
                               f'before max_si={self.max_si}; cannot hand the KV cache to the eager path')
        for blk, (k, v) in zip(self.blocks, self.kv_refs):
            blk.sa.cached_k = k
            blk.sa.cached_v = v
        self.graph_sis_this_image = []

    # ------------------------------------------------------------------ main entry
    def run(self, si, x, cond, ca_kv, scale_schedule, rope_grid, spattn_fn):
        if si == 0:
            if self.captured or self.capture_image:
                pass
            elif self.images_seen >= _WARM_IMAGES:
                self.capture_image = True
            self.images_seen += 1
            if self.capture_image or self.captured:
                if self.batch is None:
                    self.batch = x.shape[0]
                self._begin_image(cond, ca_kv)

        if not (self.captured or self.capture_image):
            return self._blocks(si, x, cond, ca_kv, scale_schedule, rope_grid, spattn_fn)

        if self.capture_image:
            out = self._capture_and_run(si, x, scale_schedule, rope_grid, spattn_fn)
        else:
            self.x_static[si].copy_(x)
            self.graphs[si].replay()
            out = self.out_static[si]
        self.graph_sis_this_image.append(si)
        if self.capture_image and si == self.max_si:
            # python-side cache attributes now point at the last graph's pool tensors; keep
            # them so replay images (where no python runs) can hand them to the eager scales.
            # Intermediate caches are deliberately not referenced: their pool blocks are then
            # reused by later captures, keeping the pool footprint close to the eager peak.
            self.kv_refs = [(blk.sa.cached_k, blk.sa.cached_v) for blk in self.blocks]
            self.capture_image = False
            self.captured = True
            print(f'[sparvar-cuda-graph] captured {len(self.graphs)} block-stack graphs (si 0..{self.max_si}, batch {self.batch})')
        return out

    def _capture_and_run(self, si, x, scale_schedule, rope_grid, spattn_fn):
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        if si == 0:
            # force the AdaLN split to be recomputed inside graph 0 from the static
            # condition (a cached split would pin stale per-image tensors into the graph)
            for blk in self.blocks:
                blk._adaln_cache = None
            key, _ = _bi._rope_scale_meta(scale_schedule, need_starts=False)
            self.rope_refs = (rope_grid[key], _bi._rope_polar(rope_grid[key]))
        self.x_static[si] = x.clone()
        g = torch.cuda.CUDAGraph()
        try:
            with torch.cuda.graph(g, pool=self.pool):
                out = self._blocks(si, self.x_static[si], self.cond_static, self._static_ca_kv(),
                                   scale_schedule, rope_grid, spattn_fn)
        except Exception as e:     # capture failed: the KV cache state of this image is unusable
            self.failed = True
            raise RuntimeError(f'[sparvar-cuda-graph] capture failed at si={si}: {e}') from e
        self.graphs[si] = g
        self.out_static[si] = out
        g.replay()      # capture records without executing: run it to produce this image's activations
        return out
