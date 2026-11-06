"""CUDA-graph replay of the dense block stack at the launch-bound small scales of
InfinityStar (SparVAR video path).

At the small scales of InfinityStar-8B 720p (image pyramid si 0..9: 1 .. 144 visual
tokens; first video-clip scales si 15..17: 20 .. 160 tokens; batch 2 = cond + uncond)
every step costs ~35 ms of wall time for well under 10 ms of GPU work: 36 layers x ~30
kernel launches are dispatched eagerly from Python, so the GPU idles. Capturing the whole
36-block stack of each such step into one CUDA graph removes that dispatch floor. The
graph replays exactly the kernels the eager path launches on the same values, so the
outputs are bit-identical to eager mode provided the attention kernel used inside the
graph is bit-identical to the eager one (see below).

KV layout of a step (eager: torch.cat([cached_k[ref] for ref in ref_sids] + [cached_k['t0']]
+ [k_step])): [reference-scale KV (R rows, image scales: R = 0; video scales: the last
image scale, R = 3600) | text KV (t rows, prompt dependent) | this step's own K/V (L rows)].
Only the text length is prompt dependent. It is made static by keeping, per layer and per
scale group, one packed KV buffer of B segments of S = R + TEXT_MAXLEN + L_max rows:
reference KV at rows [0, R) (staged once per video from the eager cache), text at
[R, R + t) (staged once per video), the step's own K/V at [R + t, R + t + L) (written
through index_copy_ with an index computed inside the graph from a per-video scalar t),
and FA3 varlen attention with seqused_k = R + t + L so the padding rows are never read.
This keeps the baseline's KV order; FA3 varlen(seqused_k) vs the eager dense FA3 call is
checked bit-identical by kernels/tests/test_fa3_varlen_bitexact.py.

Scope (checked per step; anything else runs the eager loop): si in the configured scale
set, si < decision scale (dense attention in every layer), the uncond branch not dropped,
batch/dtype/shape as captured, text length <= TEXT_MAXLEN. Graphed scales must reference
at most the last image scale (video group) or nothing (image group).

Enable with SPARVAR_VIDEO_CUDA_GRAPH=1 (default off). SPARVAR_VIDEO_CUDA_GRAPH_SCALES
(default "0-9,15-17") selects the scales, SPARVAR_VIDEO_CUDA_GRAPH_WARM_VIDEOS (default 1)
is the number of eager videos before the capture video.
"""
import os
from typing import Dict, List, Optional, Tuple

import torch

from models.infinitystar.fused_elementwise_utils import fused_rope_inplace

try:
    from flash_attn_interface import flash_attn_varlen_func as _fa3_varlen
except Exception:  # noqa: BLE001
    _fa3_varlen = None

_ENABLED = os.environ.get('SPARVAR_VIDEO_CUDA_GRAPH', '0').lower() in ('1', 'true', 'yes', 'on')
_SCALES_ENV = os.environ.get('SPARVAR_VIDEO_CUDA_GRAPH_SCALES', '0-9,15-17')
_WARM_VIDEOS = int(os.environ.get('SPARVAR_VIDEO_CUDA_GRAPH_WARM_VIDEOS', '1'))
_TEXT_MAXLEN = 512   # T5 tokenizer max_length in run_infinity.encode_video_prompt


def video_cuda_graph_enabled() -> bool:
    return _ENABLED and _fa3_varlen is not None


def _parse_scales(spec: str) -> List[int]:
    out = []
    for part in spec.split(','):
        part = part.strip()
        if not part:
            continue
        if '-' in part:
            a, b = part.split('-')
            out += list(range(int(a), int(b) + 1))
        else:
            out.append(int(part))
    return sorted(set(out))


class _Group:
    """Static packed KV buffers for one set of scales sharing the same reference KV."""

    def __init__(self, name: str, ref_sids: List[int], R: int, L_max: int):
        self.name = name
        self.ref_sids = ref_sids
        self.R = int(R)
        self.S = self.R + _TEXT_MAXLEN + int(L_max)
        self.kbuf: List[torch.Tensor] = []
        self.vbuf: List[torch.Tensor] = []
        self.cu_k: Optional[torch.Tensor] = None
        self.cu_q: Dict[int, torch.Tensor] = {}
        self.row_base: Dict[int, torch.Tensor] = {}   # per L: b*S + R + j (text offset t added in-graph)
        self.ref_staged = False


class VideoBlockStackGraphRunner:
    def __init__(self, model, attn_config: dict, scale_schedule, context_info, first_full_idx: int):
        self.model = model
        self.blocks = list(model.unregistered_blocks)
        self.scale_schedule = [tuple(int(v) for v in s) for s in scale_schedule]
        tok = lambda s: s[0] * s[1] * s[2]  # noqa: E731
        decision_scale = int(attn_config.get('decision_scale', 27))
        wanted = [s for s in _parse_scales(_SCALES_ENV) if 0 <= s < min(decision_scale, len(self.scale_schedule))]
        self.groups: Dict[int, _Group] = {}          # si -> group
        img_scales = [s for s in wanted if not context_info[s]['ref_sids']]
        vid_scales = [s for s in wanted if list(context_info[s]['ref_sids']) == [int(first_full_idx)]]
        if img_scales:
            g = _Group('img', [], 0, max(tok(self.scale_schedule[s]) for s in img_scales))
            for s in img_scales:
                self.groups[s] = g
        if vid_scales:
            g = _Group('vid', [int(first_full_idx)], tok(self.scale_schedule[int(first_full_idx)]),
                       max(tok(self.scale_schedule[s]) for s in vid_scales))
            for s in vid_scales:
                self.groups[s] = g
        self.scales = sorted(self.groups)
        self.n_steps_expected = sum(self._repeats_of(s) for s in self.scales)

        self.videos_seen = 0
        self.capture_video = False
        self.captured = False
        self.failed = False
        self.pool = None
        self.batch: Optional[int] = None
        self.graphs: Dict[Tuple[int, int], torch.cuda.CUDAGraph] = {}
        self.x_static: Dict[Tuple[int, int], torch.Tensor] = {}
        self.out_static: Dict[Tuple[int, int], torch.Tensor] = {}
        self.rope_static: Dict[Tuple[int, int], torch.Tensor] = {}
        self.t_static: Optional[torch.Tensor] = None   # 0-dim int64: text length of the current video
        self.text_len: Optional[int] = None
        self.n_replayed = 0

    def _repeats_of(self, si: int) -> int:
        img = getattr(self.model, 'image_scale_repetition', None)
        vid = getattr(self.model, 'video_scale_repetition', None)
        if img is None or vid is None:
            return 3
        n = len(img)
        return int(img[si]) if si < n else int(vid[si % n])

    # ------------------------------------------------------------------ per video
    def begin_video(self, text_len: int, batch: int):
        """Called once per video after the t0 (text) forward: stage the per-layer text KV."""
        self.text_len = int(text_len)
        for g in set(self.groups.values()):
            g.ref_staged = False
        if self.capture_video and self.graphs:
            # the capture video is over: replay whatever was captured, eager for the rest
            self.capture_video = False
            self.captured = True
        if not (self.captured or self.capture_video):
            if self.videos_seen >= _WARM_VIDEOS and not self.failed and self.scales:
                self.capture_video = True
        self.videos_seen += 1
        if not (self.captured or self.capture_video):
            return
        if self.text_len > _TEXT_MAXLEN:
            return                              # eligible() rejects every step of this video
        if self.batch is None:
            self.batch = int(batch)
        if int(batch) != self.batch:
            return
        B = self.batch
        ref_attn = self.blocks[0].attn
        k0, v0 = ref_attn.cached_k['t0'], ref_attn.cached_v['t0']      # [B, Hkv, t, D]
        if self.t_static is None:
            self.t_static = torch.zeros((), dtype=torch.int64, device=v0.device)
            for g in set(self.groups.values()):
                for _ in self.blocks:
                    g.kbuf.append(torch.zeros((B * g.S, k0.shape[1], k0.shape[3]), dtype=v0.dtype, device=v0.device))
                    g.vbuf.append(torch.zeros((B * g.S, v0.shape[1], v0.shape[3]), dtype=v0.dtype, device=v0.device))
                g.cu_k = torch.arange(0, (B + 1) * g.S, g.S, device=v0.device, dtype=torch.int32)
        self.t_static.fill_(self.text_len)
        t = self.text_len
        for li, blk in enumerate(self.blocks):
            ck, cv = blk.attn.cached_k['t0'], blk.attn.cached_v['t0']
            # same cast the eager path applies right before flash attention (.to(value dtype))
            kt = ck.to(cv.dtype).transpose(1, 2)        # [B, t, Hkv, D]
            vt = cv.transpose(1, 2)
            for g in set(self.groups.values()):
                for b in range(B):
                    g.kbuf[li][b * g.S + g.R: b * g.S + g.R + t].copy_(kt[b])
                    g.vbuf[li][b * g.S + g.R: b * g.S + g.R + t].copy_(vt[b])

    def _stage_refs(self, g: _Group):
        """Copy the reference-scale KV (eager cache) into rows [0, R) of every segment."""
        if g.ref_staged or g.R == 0:
            g.ref_staged = True
            return
        B = self.batch
        for li, blk in enumerate(self.blocks):
            off = 0
            for ref in g.ref_sids:
                ck, cv = blk.attn.cached_k[ref], blk.attn.cached_v[ref]      # [B, Hkv, n, D]
                n = ck.shape[2]
                kt = ck.to(cv.dtype).transpose(1, 2)
                vt = cv.transpose(1, 2)
                for b in range(B):
                    g.kbuf[li][b * g.S + off: b * g.S + off + n].copy_(kt[b])
                    g.vbuf[li][b * g.S + off: b * g.S + off + n].copy_(vt[b])
                off += n
            assert off == g.R, f'reference KV rows {off} != R {g.R}'
        g.ref_staged = True

    # ------------------------------------------------------------------ eligibility
    def eligible(self, si: int, x: torch.Tensor, active_branch_repeat: int, batch_B: int) -> bool:
        if self.failed or not (self.captured or self.capture_video):
            return False
        if not isinstance(si, int) or si not in self.groups or self.text_len is None or self.text_len > _TEXT_MAXLEN:
            return False
        if x.shape[0] != self.batch or active_branch_repeat * batch_B != self.batch:
            return False
        return True

    # ------------------------------------------------------------------ the block stack
    def _attn_layer(self, li: int, blk, h: torch.Tensor, rope, L: int, g: _Group):
        """Mirror of SelfAttention.forward (qwen arch, dense FA3 branch) with the KV
        assembled in the static packed buffer and FA3 varlen(seqused_k) attention."""
        attn = blk.attn
        B, Lq, C = h.shape
        hidden_shape = (B, Lq, -1, attn.head_dim)
        query_states = attn.q_norm(attn.q_proj(h).view(hidden_shape)).transpose(1, 2)
        key_states = attn.k_norm(attn.k_proj(h).view(hidden_shape)).transpose(1, 2)
        value_states = attn.v_proj(h).view(hidden_shape).transpose(1, 2)
        query_states, key_states = fused_rope_inplace(query_states, key_states, rope)   # eager RoPE unless INFINITYSTAR_FUSED_ROPE=1
        vd = value_states.dtype
        # rows of this step's tokens inside each batch segment: b*S + R + t + j
        idx = g.row_base[L] + self.t_static
        kb, vb = g.kbuf[li], g.vbuf[li]
        kb.index_copy_(0, idx, key_states.to(vd).transpose(1, 2).reshape(B * L, -1, attn.head_dim))
        vb.index_copy_(0, idx, value_states.transpose(1, 2).reshape(B * L, -1, attn.head_dim))
        seqused_k = (self.t_static + (g.R + L)).to(torch.int32).expand(B).contiguous()
        q = query_states.to(vd).transpose(1, 2).reshape(B * L, -1, attn.head_dim)
        ret = _fa3_varlen(q, kb, vb, g.cu_q[L], g.cu_k, L, g.S,
                          seqused_k=seqused_k, softmax_scale=attn.head_dim ** -0.5)
        o = (ret[0] if isinstance(ret, tuple) else ret).reshape(B, L, C)
        return attn.o_proj(o)

    def _stack(self, x: torch.Tensor, rope, g: _Group):
        L = x.shape[1]
        for li, blk in enumerate(self.blocks):
            residual = x
            h = blk.input_layernorm(x)
            h = self._attn_layer(li, blk, h, rope, L, g)
            h = residual + h
            residual = h
            h = blk.post_attention_layernorm(h)
            h = blk.mlp(h)
            x = residual + h
        return x

    # ------------------------------------------------------------------ main entry
    def run(self, si: int, repeat_idx: int, x: torch.Tensor, rope_cache, last_repetition_step: bool):
        key = (int(si), int(repeat_idx))
        g = self.groups[si]
        L = x.shape[1]
        B = self.batch
        if L not in g.row_base:
            g.row_base[L] = (torch.arange(B, device=x.device, dtype=torch.int64).view(B, 1) * g.S + g.R
                             + torch.arange(L, device=x.device, dtype=torch.int64).view(1, L)).reshape(-1)
            g.cu_q[L] = torch.arange(0, (B + 1) * L, L, device=x.device, dtype=torch.int32)
        self._stage_refs(g)

        if self.captured or key in self.graphs:
            xs = self.x_static.get(key)
            if xs is None or xs.shape != x.shape or xs.dtype != x.dtype:
                return None                       # unseen shape: eager
            xs.copy_(x)
            self.graphs[key].replay()
            self.n_replayed += 1
            out = self.out_static[key]
        else:
            out = self._capture_and_run(key, x, rope_cache, g)
            if out is None:
                return None
        # python-side KV bookkeeping the eager path would have done: the entry must exist
        # (later scales index it when releasing dead caches) but its value is never read
        # again (graphed scales are not referenced by any later scale).
        if last_repetition_step:
            for blk in self.blocks:
                blk.attn.cached_k[si] = None
                blk.attn.cached_v[si] = None
        return out

    def _capture_and_run(self, key, x, rope_cache, g: _Group):
        if self.pool is None:
            self.pool = torch.cuda.graph_pool_handle()
        self.x_static[key] = x.clone()
        self.rope_static[key] = rope_cache.clone()
        graph = torch.cuda.CUDAGraph()
        try:
            torch.cuda.synchronize()
            with torch.cuda.graph(graph, pool=self.pool):
                out = self._stack(self.x_static[key], self.rope_static[key], g)
        except Exception as e:  # noqa: BLE001
            self.failed = True
            self.capture_video = False
            print(f'[sparvar-video-cuda-graph] capture failed at si={key[0]} r={key[1]}: {str(e)[:300]}; eager from now on')
            return None
        self.graphs[key] = graph
        self.out_static[key] = out
        graph.replay()
        if len(self.graphs) == self.n_steps_expected:
            self.capture_video = False
            self.captured = True
            print(f'[sparvar-video-cuda-graph] captured {len(self.graphs)} block-stack graphs '
                  f'(scales {self.scales}, batch {self.batch})')
        return out
