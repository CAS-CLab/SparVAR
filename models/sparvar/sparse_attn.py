import copy
import math
import os
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import triton
import yaml
from einops import rearrange
from torch import Tensor, nn
from torch.nn import functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from kernels.sparse_attn.sparse_attn_config import (GLOBAL_CONFIG,
                                                    get_kernel_config_attn)
from kernels.sparse_attn.sparse_attn_ops import (bitpack, bitunpack, csp_attn,
                                                 dense_attn,
                                                 dense_colsum_attn_q,
                                                 dense_colsum_attn_s,
                                                 mask_to_indices)
from models.sparvar.index_map_utils import (build_footprint_union_indices,
                                            build_frame_band_union_indices,
                                            build_image_local_union_indices,
                                            build_infinity_footprint_union_indices,
                                            expand_mapped_indices_to_budget,
                                            map_inds_counts_across_scales,
                                            map_infinity_sparse_kv_inds_lut,
                                            map_infinitystar_sparse_kv_inds_lut,
                                            map_sparse_kv_inds,
                                            save_attention_selection_visuals)
from models.sparvar.sparse_attn_layer_counter import LayerCounter
from models.sparvar.csp_attn_guard_utils import check_csp_attn_inputs
from models.sparvar.dcsla_utils import (compute_decision_block_mass,
                                        compute_decision_block_mass_image,
                                        dcsla_sparse_attention,
                                        dcsla_warn_once)
from models.sparvar.attn_pattern_stats_utils import (
    maybe_record_infinitystar_attention_stats,
    maybe_record_infinitystar_decision_colsum,
    pattern_stats_enabled_for_layer,
)
from models.sparvar.decision_qkv_dump_utils import maybe_dump_decision_qkv
from models.sparvar.kbench_pattern_dump_utils import maybe_dump_csp_pattern
from models.sparvar.index_width_utils import resolve_footprint_out_width
from models.sparvar.image_identity_utils import (
    CumulativeImageIdentityIndices, validate_identity_cumulative_config,
)
from models.sparvar.sparse_index_viz_utils import (dump_decision_colsum_viz,
                                                   dump_selected_index_heatmap,
                                                   image_kv_scale_boundaries,
                                                   viz_enabled_for)
from models.sparvar.storage import AttnStorage, MaybeOffloadedTensor
from models.sparvar.storage_prefetch_utils import (prefetch_next_layer_storage,
                                                   register_layer_storage)
from utils import lprof


# def upsample_Ocache(
#     O_cache: torch.Tensor,      # [B, H, L_S, Dh]
#     L_src: int, L_tgt: int,
#     # hS: int, wS: int,
#     # hK: int, wK: int,
#     mode: str = "2d",          # "1d" 或 "2d"
#     upsample_mode: str = 'bilinear'      # 'nearest' | 'linear' | 'bilinear' | 'bicubic' | 'trilinear' | 'area' | 'nearest-exact'
# ) -> torch.Tensor:

#     # assert L_S == hS * wS and L_K == hK * wK, "mismatch"

#     B, H, _, Dh = O_cache.shape

#     if mode == '1d':
#         # 最近邻 1D 重采样（长度 → 新长度）
#         idx = torch.round(torch.arange(L_tgt, device=O_cache.device, dtype=torch.float32) * (L_src - 1) / max(L_tgt - 1, 1)).to(torch.long)
#         return O_cache.index_select(dim=2, index=idx)
#     elif mode == "2d":
#         # x = O_cache.permute(0, 1, 3, 2).contiguous()            # [B,H,L,D] -> [B,H,D,L]
#         # x = x.view(B, H, Dh, int(L_S**0.5), int(L_S**0.5))                            # [B,H,Dh,hS,wS]
#         x = O_cache.view(B, H, int(L_src**0.5), int(L_src**0.5), Dh).permute(0, 1, 4, 2, 3)
#         x = x.reshape(B * H, Dh, int(L_src**0.5), int(L_src**0.5))

#         if upsample_mode == 'bilinear':
#             x = F.interpolate(
#                 x, size=(int(L_tgt**0.5), int(L_tgt**0.5)),
#                 mode=upsample_mode,
#                 align_corners=False)
#         else:
#             x = F.interpolate(
#                 x, size=(int(L_tgt**0.5), int(L_tgt**0.5)),
#                 mode=upsample_mode)

#         # x = x.view(B, H, Dh, L_K).permute(0, 1, 3, 2).contiguous()  # [B,H,L_K,Dh]
#         x = x.view(B, H, Dh, int(L_tgt**0.5), int(L_tgt**0.5))
#         x = x.permute(0, 1, 3, 4, 2)
#         return x.reshape(B, H, -1, Dh)      # view also True
#     else:
#         raise ValueError(f"unknown mode={mode}")


def upsample_Ocache(
    O_cache: torch.Tensor,      # [B, H, T_S*H_S*W_S, Dh]
    src_shape: Tuple[int, int, int],
    tgt_shape: Tuple[int, int, int],
    upsample_mode: str = 'trilinear',
) -> torch.Tensor:
    src_t, src_h, src_w = src_shape
    tgt_t, tgt_h, tgt_w = tgt_shape
    B, H, L_src, Dh = O_cache.shape
    src_len = src_t * src_h * src_w
    tgt_len = tgt_t * tgt_h * tgt_w
    if L_src != src_len:
        raise ValueError(f"O_cache length {L_src} does not match source shape {src_shape} ({src_len}).")
    if L_src == tgt_len:
        return O_cache

    if upsample_mode == 'trilinear' and src_t == 1 and tgt_t == 1:
        # t == 1 on both sides (image scales): trilinear degenerates to bilinear
        # on the (h, w) plane -- the source t-coord is exactly 0 under
        # align_corners=False -- and on H800 the two are bit-identical in bf16.
        # Dispatch bilinear2d because the trilinear3d CUDA kernel is
        # pathologically slow for this shape (~8 ms vs ~0.08 ms per call for
        # [28, 128, 1, 40, 40] -> 48x48, see kernels/bench_ocache_upsample.py).
        x = O_cache.reshape(B, H, src_h, src_w, Dh).permute(0, 1, 4, 2, 3)
        x = x.reshape(B * H, Dh, src_h, src_w)
        x = F.interpolate(x, size=(tgt_h, tgt_w), mode='bilinear', align_corners=False)
        x = x.reshape(B, H, Dh, tgt_h, tgt_w).permute(0, 1, 3, 4, 2)
        return x.reshape(B, H, tgt_len, Dh).contiguous()

    x = O_cache.reshape(B, H, src_t, src_h, src_w, Dh).permute(0, 1, 5, 2, 3, 4)
    x = x.reshape(B * H, Dh, src_t, src_h, src_w)
    if upsample_mode in ('linear', 'bilinear', 'bicubic', 'trilinear'):
        x = F.interpolate(x, size=(tgt_t, tgt_h, tgt_w), mode=upsample_mode, align_corners=False)
    else:
        x = F.interpolate(x, size=(tgt_t, tgt_h, tgt_w), mode=upsample_mode)

    x = x.reshape(B, H, Dh, tgt_t, tgt_h, tgt_w).permute(0, 1, 3, 4, 5, 2)
    return x.reshape(B, H, tgt_len, Dh).contiguous()


def _as_scale_shape(scale_schedule, scale_ind):
    if not isinstance(scale_ind, int) or scale_schedule is None:
        return None
    if scale_ind < 0 or scale_ind >= len(scale_schedule):
        return None
    shape = scale_schedule[scale_ind]
    if len(shape) != 3:
        return None
    return tuple(int(x) for x in shape)


def _target_indices_count(kv_len: int, top_keys: float, multiple_of: int) -> int:
    indices_count = int(multiple_of * round((top_keys * kv_len) / multiple_of))
    indices_count = max(multiple_of, indices_count)
    return min(indices_count, kv_len)


# def _ceil_to_multiple(x: int, m: int) -> int:
#     if m is None or m <= 1:
#         return x
#     return ((x + m - 1) // m) * m


def _match_batch(x, batch_size: int):
    if x is None or x.shape[0] == batch_size:
        return x
    return x[:batch_size]


def _require_video_scale_shapes(decision_shape, current_shape):
    if decision_shape is None or current_shape is None:
        raise RuntimeError(
            "CS4A InfinityStar video sparse attention requires both decision and "
            "current scale shapes. Pass scale_schedule into SparseDiffAttn.forward."
        )


def _uniform_counts_like(inds: Tensor, count: int) -> Tensor:
    """``[B, H, G]`` int32 counts tensor filled with one uniform per-group count."""
    return torch.full(
        (inds.shape[0], inds.shape[1], inds.shape[2]),
        count, device=inds.device, dtype=torch.int32
    )


def _pad_uniform_indices(inds: Tensor, indices_count: int, q: Tensor, k: Tensor,
                         bm: int, indices_pad_to: int) -> Tuple[Tensor, Tensor]:
    """Uniform-count ``(inds, counts)`` in the k_len-width layout csp_attn consumes.

    Pads the stride, but not the shape, of ``inds`` so that the TMA stride gets
    aligned to 16 bytes: 对索引进行填充是为了确保内存访问满足 NVIDIA Hopper 架构中
    TMA（张量内存加速器）的对齐要求（通常是128位/16字节），从而最大化数据传输效率
    （Triton 路径 indices_pad_to==1 时为 no-op）。The pad region is ``torch.empty``
    (uninitialized) — downstream kernels only read ``counts[g]`` entries per group.
    Cross-attn note: use k_len (not q_len) when calculating padding_amount.
    """
    counts = torch.full(
        (q.shape[0], q.shape[1], triton.cdiv(q.shape[-2], bm)),
        indices_count, device=q.device, dtype=torch.int32
    )
    padding_amount = (k.shape[-2] - indices_count + indices_pad_to - 1) // indices_pad_to * indices_pad_to
    inds = torch.cat(
        [inds, torch.empty((*counts.shape, padding_amount), device=q.device, dtype=torch.int32)],
        dim=-1
    ).to(torch.int32)
    inds = inds[:, :, :, :k.shape[-2]]
    return inds, counts


# Initialized based on sequence shape
singleton_static_mask = None
singleton_video_query_groups = None


class SparseDiffAttn(nn.Module):
    def __init__(
        self, layer_num: int, layer_counter: LayerCounter,
        decision_scale: int = 10,
        model_family: Optional[str] = None,
    ):
        super().__init__()
        self.layer_num = layer_num          # record number of model attn layer 记录这是模型中的第几层Attention
        print(f'\nCurrent init layer is {self.layer_num}')
        self.layer_counter = layer_counter  # todo: diffusion step --> var scale

        attn_config = GLOBAL_CONFIG['attn']

        # Which model family this module accelerates, bound once at init:
        #   'image' (Infinity)     : KV = stacked pyramid Concat(scale_0..scale_k)
        #                            -> Decompose-Align-Project LUT + local-window union
        #   'video' (InfinityStar) : KV = [ctx prefix] + [current scale]
        #                            -> 3D coord LUT + frame_band / footprint / D-CSLA
        self.model_family = str(model_family or attn_config.get('model_family', 'video')).lower()
        assert self.model_family in ('image', 'video'), f"unknown model_family: {self.model_family}"
        print(f'[Sparse-Attn] model_family = {self.model_family}')

        self.use_o_cache = attn_config['use_o_cache']
        if self.use_o_cache:
            print(f'[Sparse-Attn] Use O_cache is {self.use_o_cache}, O_t = O_cache + ΔO_new')
            self.storage = AttnStorage(layer_num, init_names=['indices', 'out_cache'])              # 该层的专属 cache：indices: 用于存储在稠密步骤中计算出的稀疏模式（即Top-K Key的索引）；out_cache: 用于存储论文公式中的 O_cache，即 O_dense - ΔO_old
        elif attn_config['should_compress_indices']:
            print(f'[Sparse-Attn] Use O_cache is {self.use_o_cache}, O_t = ΔO_new')
            self.storage = AttnStorage(layer_num, init_names=['indices'])
        else:
            # storage opt: non-compress + no O_cache keeps the decision scale's COMPACT
            # top-k as a GPU-resident module attr (self._decision_inds) instead of the
            # offloaded [B,H,G,k_len] indices/counts. No offload buffers are needed, so
            # skip pre-allocating them -- this is what removes the multi-GB pinned-CPU
            # 'attn.counts' (LARGE_BUF) / 'attn.indices' (MEDIUM_BUF) buffers per layer.
            print(f'[Sparse-Attn] Use O_cache is {self.use_o_cache}, O_t = ΔO_new')
            self.storage = AttnStorage(layer_num, init_names=[])
        # Enables "load layer i+1 while layer i computes" on the sparse scales
        # (see models/sparvar/storage_prefetch_utils.py).
        register_layer_storage(layer_num, self.storage)
        self.mask_shape = [None] * GLOBAL_CONFIG['num_model_invocations_per_inference_step']    # 如果启用了索引压缩（bitpacking），这个变量用来存储原始mask的形状，以便后续解压

        self.decision_scale = decision_scale
        print(f'    Sparse decision scale is scale-{decision_scale}')
        _decision_repeat_idx = attn_config.get('decision_repeat_idx', None)
        if _decision_repeat_idx is None:
            print(f'        repeat-last')
        else:
            print(f'        repeat-{_decision_repeat_idx}')

        if not attn_config.get('dcsla_enabled', False):
            # .get: the video-only band keys are absent from the image config
            print(f'[CS4A] {attn_config.get('cs4a_index_expand_mode')=}, {attn_config.get('cs4a_footprint_union_band')=}, {attn_config.get('cs4a_band_neighbor_frames')=}')
        else:
            print(f'[DCSLA] {attn_config['flex_backend']=} {attn_config['flex_mode']=} {attn_config['flex_dynamic']=}')
            print(f'        {attn_config['dcsla_select_mode']=} {attn_config['dcsla_tau']=} {attn_config['dcsla_topk_ratio']=}')
            print(f'        {attn_config['dcsla_local_diag_blocks']=} {attn_config['dcsla_temporal_neighbor_frames']=} {attn_config['dcsla_temporal_diag_blocks']=}')
            print(f'        {attn_config['dcsla_keep_sink']=} {attn_config['dcsla_sink_select_tau']=}')
            print(f'        {attn_config['dcsla_mass_normalize']=} {attn_config['dcsla_colsum_group_rows']=}')
            print(f'        {attn_config['dcsla_min_blocks']=} {attn_config['dcsla_max_blocks_ratio']=} {attn_config['dcsla_image_win_size']=}')

        self.decision_sink_len = 0
        self.decision_scale_shape = None
        self.decision_q_len = 0
        self.decision_k_len = 0

        # storage opt (non-compress + no O_cache): the decision scale's COMPACT top-k
        # indices (no k_len padding) + its constant top-k count, kept GPU-resident here
        # instead of the offloaded [B,H,G,k_len] indices/counts. Replayed by the later
        # video sparse scales (which re-pad downstream as before). None == not computed.
        self._decision_inds = None
        self._decision_indices_count = None

        self._image_identity_indices = CumulativeImageIdentityIndices()

        # D-CSLA: per-target-scale footprint-transferred block mass, computed once per
        # decision step ({scale_ind: [B,H,G_S,V] bf16}) and consumed by the block-sparse
        # FlexAttention path. GPU-resident module attrs (same pattern as _decision_inds).
        self._dcsla_mass = None
        self._dcsla_expected_k_len = None

        # Which version dense_colsum_attn should we use?
        speedup = attn_config['speedup']
        if not speedup:
            print(f'Use ** quality ** dense_colsum_attn')
            self.dense_colsum_attn = dense_colsum_attn_q
        else:
            print(f'Use ** speedup ** dense_colsum_attn')
            self.dense_colsum_attn = dense_colsum_attn_s

    @torch.compile(dynamic=True)            # `dynamic=False` is for cs.shape and topk fixed
    def random_and_topk(self, cs, topk, random_keys: float = 0.0):
        # Decision-scale selection = pure top-k (no random fill). Any random/local
        # padding is governed by the sparse-scale frame_band strategy, not by an
        # uncontrolled 1% here. `random_keys > 0` can re-enable random fill if needed.
        if random_keys > 0:
            mask = torch.rand(cs.shape, device=cs.device) < random_keys
        else:
            mask = torch.zeros(cs.shape, device=cs.device, dtype=torch.bool)
        mask.scatter_(-1, cs.topk(k=topk, dim=-1).indices, True)

        qg = cs.shape[-2]
        n = cs.shape[-1]
        if singleton_video_query_groups is not None:
            mask = mask * singleton_video_query_groups[..., :qg, :n]
        if singleton_static_mask is not None:
            mask = mask | singleton_static_mask[..., :qg, :n]

        return mask

    # @torch.compile(dynamic=True)
    # def random_and_topk(self, cs, topk):
    #     mask = torch.randint(0, 100, cs.shape, device=cs.device, dtype=torch.uint8) == 0
    #     mask.scatter_(-1, cs.topk(k=topk, dim=-1).indices, True)

    #     qg = cs.shape[-2]
    #     n = cs.shape[-1]
    #     if singleton_video_query_groups is not None:
    #         mask = mask * singleton_video_query_groups[..., :qg, :n]
    #     if singleton_static_mask is not None:
    #         mask = mask | singleton_static_mask[..., :qg, :n]

    #     return mask

    # ──────────────────────────────────────────────────────────────────────
    # Shared step helpers (used by both the video and image paths)
    # ──────────────────────────────────────────────────────────────────────

    def _load_and_prefetch_storage(self):
        """Reload this layer's offloaded decision pattern, then prefetch the next layer's.

        The order is stream-correctness critical — do NOT reorder:
          load_async : no-op if the previous layer's prefetch already issued it
          wait       : current stream waits on the load/offload streams
          prefetch   : issued AFTER our own wait (so that wait doesn't cover the new
                       copy) — the next layer's H2D reload runs on the side stream,
                       overlapped with this layer's index mapping + csp_attn.
        """
        self.storage.load_async()
        self.storage.load_async_wait()
        prefetch_next_layer_storage(self.layer_num)

    def _load_upsampled_o_cache(self, provider: str, batch_size: int,
                                current_scale_shape, v_dtype,
                                scale_ind=None, layer_ind=None) -> Tensor:
        """O_cache reload for a sparse step: fetch → (cuda: clone) → batch-match →
        trilinear upsample decision-shape → current-shape → cast to the value dtype.

        (1, h, w) image scale shapes: trilinear with t == 1 degenerates to the 2D
        bilinear the old image implementation used.
        """
        # segment bisect for the ocache latency deep-dive: rows are
        # attn.ocache.{get,clone,up,cast}
        _t = lprof.timer('attn.ocache', si=scale_ind, layer=layer_ind)
        o = self.storage.get_out_cache()
        _t.mark('get')
        if provider == 'cuda': o = o.clone()
        _t.mark('clone')

        o = _match_batch(o, batch_size)     # for skip last 2 scales / drop-uncond
        # --- upscale o_cache ---
        o = upsample_Ocache(
            o, src_shape=self.decision_scale_shape,
            tgt_shape=current_scale_shape, upsample_mode='trilinear'
        )
        _t.mark('up')
        o = o.to(v_dtype)
        _t.stop('cast')
        return o

    # ──────────────────────────────────────────────────────────────────────
    # InfinityStar (video) CS4A path
    # ──────────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _fast_attention(
        self, q: Tensor, k: Tensor, v: Tensor, scale,
        inference_step: int, do_full_step: bool,
        scale_ind: int,
        layer_ind: int,
        scale_schedule=None,
        #* [Index Mapping Exps] Profiling metadata only; it does not affect sparse selection.
        repeat_idx: Optional[int] = None,
    ) -> Tensor:
        """InfinityStar (video) CS4A dispatcher.

        根据 ``do_full_step`` 决定执行稠密决策路径还是稀疏路径；本函数只做调度与公共
        记录，真正的性能来自两个 step 方法调用的预编译 chipmunk.ops CUDA/Triton
        kernels（因此 ``@torch.compiler.disable`` 不参与编译）：

          decision step (do_full_step=True)  → ``_video_decision_step``
              dense colsum → top-k → 存储 decision 稀疏模式（+可选 O_cache）
          sparse step   (do_full_step=False) → ``_video_sparse_step``
              载入 decision 模式 → 跨尺度 index mapping（footprint / legacy 链）→ csp_attn
        """
        attn_config = GLOBAL_CONFIG['attn']
        attn_kernel_config = get_kernel_config_attn()

        stats_sink_len = max(int(k.shape[-2]) - int(q.shape[-2]), 0)
        stats_scale_shape = _as_scale_shape(scale_schedule, scale_ind)
        # env-gated (SPARVAR_PATTERN_STATS)
        maybe_record_infinitystar_attention_stats(
            q, k, scale=scale, scale_ind=scale_ind, layer_ind=layer_ind,
            scale_shape=stats_scale_shape, sink_len=stats_sink_len,
            do_full_step=do_full_step, attn_config=attn_config)

        if layer_ind < attn_config['first_n_dense_layers']:
            # print(f'first_{layer_ind}_dense_layer')
            o, _ = dense_attn(q, k, v, scale=scale)
            # --- torch flash-attn ---
            # with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
            #     o = F.scaled_dot_product_attention(q, k, v, scale=scale)
            return o

        if do_full_step:
            return self._video_decision_step(
                q, k, v, scale, scale_ind, layer_ind, scale_schedule,
                attn_config, attn_kernel_config)
        return self._video_sparse_step(
            q, k, v, scale, scale_ind, layer_ind, scale_schedule,
            attn_config, attn_kernel_config, repeat_idx=repeat_idx)

    def _video_decision_step(
        self, q: Tensor, k: Tensor, v: Tensor, scale,
        scale_ind: int, layer_ind: int, scale_schedule,
        attn_config, attn_kernel_config,
    ) -> Tensor:
        """FULL STEP (sparse decision scale).

        colsum → (D-CSLA: block mass | legacy: top-k) → 存储 decision 稀疏模式
        （compressed: bitpacked mask / non-compressed: compact top-k 或 padded+counts）
        → 可选 O_cache = O_dense + (-1)ΔO → 返回 dense O。
        """
        bm = attn_kernel_config['bm']
        multiple_of = attn_kernel_config['counts_multiple_of']
        indices_pad_to = attn_kernel_config['indices_pad_to']
        provider = attn_config['provider']

        # print(f'FULL STEP: {do_full_step=}')
        # env-gated (SPARVAR_DUMP_DECISION_QKV) raw-input dump for kernel diagnosis
        maybe_dump_decision_qkv(q, k, v, scale, scale_ind=scale_ind, layer_ind=layer_ind)

        #* ----------- 1. compute colsum ----------
        with lprof.region('attn.colsum', si=scale_ind, layer=layer_ind, k_len=int(k.shape[-2])):
            o, bs = self.dense_colsum_attn(q, k, v, scale)
        # The prefix length is dynamic because prompt token count varies per sample.
        decision_sink_len = max(int(k.shape[-2]) - int(q.shape[-2]), 0)
        decision_scale_shape = _as_scale_shape(scale_schedule, scale_ind)
        indices_count = _target_indices_count(int(k.shape[-2]), attn_config['top_keys'], multiple_of)

        self.decision_sink_len = decision_sink_len
        self.decision_scale_shape = decision_scale_shape
        self.decision_q_len = int(q.shape[-2])
        self.decision_k_len = int(k.shape[-2])

        maybe_record_infinitystar_decision_colsum(
            bs, scale_schedule=scale_schedule, decision_scale_ind=scale_ind,
            layer_ind=layer_ind, decision_sink_len=decision_sink_len,
            decision_scale_shape=decision_scale_shape, decision_k_len=int(k.shape[-2]))

        # ------ D-CSLA decision hook: transfer block mass, skip legacy topk/storage ------
        if attn_config.get('dcsla_enabled', False):
            assert not self.use_o_cache and attn_config.get('update_o_cache_on_sparse_scale', 'none') == 'none', \
                'D-CSLA v1: O_cache paths unsupported (set use_o_cache: false)'
            if attn_config['should_compress_indices']:
                dcsla_warn_once('[D-CSLA] should_compress_indices=true pre-allocates per-layer pinned '
                                "'attn.indices' buffers that D-CSLA never reads; set it to false in the config.")
            # Rebuilt wholesale each decision step -> no stale entries across videos.
            self._dcsla_mass, self._dcsla_expected_k_len = compute_decision_block_mass(
                bs, scale_schedule=scale_schedule, decision_scale_ind=scale_ind,
                decision_sink_len=decision_sink_len, decision_scale_shape=decision_scale_shape,
                decision_k_len=int(k.shape[-2]),
                block_size=int(attn_config.get('flex_block_size', 128)),
                normalize=bool(attn_config.get('dcsla_mass_normalize', True)),
                # decision_repeat_idx set: the decision scale's later repeats replay the
                # block-sparse forward on their own geometry, so emit its mass entry too.
                include_decision_scale=attn_config.get('decision_repeat_idx', None) is not None)
            # Raw-evidence viz: token-wise colsum at the decision scale
            # (SPARVAR_VIZ_DECISION_COLSUM=0 disables just this renderer).
            if viz_enabled_for(scale_ind, layer_ind):
                dump_decision_colsum_viz(
                    bs, scale_ind=scale_ind, layer_ind=layer_ind,
                    q_len=int(q.shape[-2]), k_len=int(k.shape[-2]),
                    block_size=int(attn_config.get('flex_block_size', 128)),
                    sink_len=decision_sink_len, scale_shape=decision_scale_shape,
                    tag="dcsla")
            return o

        #* ---------- 2. TopK and get sparse KV index ----------
        # for InfinityStar
        if attn_config['should_compress_indices']:
            # random_keys = float(attn_config.get('random_keys', 0.0))
            # mask = self.random_and_topk(bs, indices_count, random_keys)          # [bs, num_heads, q_groups, k_len] torch.bool
            mask = self.random_and_topk(bs, indices_count)
            packed, mask_shape = bitpack(mask)          # packed: 1D Tensor uint8, mask_shape: torch.Size([bs, num_heads, q_groups, k_len])
            self.mask_shape[self.layer_counter.cur_model_invocation_per_step] = mask_shape
            self.storage.set_indices(packed)
            # Host-known decision top-k count: the compact-width mask_to_indices bound
            # on sparse steps needs it (resolve_footprint_out_width); constant per
            # decision, consumed nowhere else on the compressed path.
            self._decision_indices_count = int(indices_count)
            # inds/counts here are consumed ONLY by the O_cache delta (use_o_cache)
            # and the env-gated viz dump — both off in the speed config — so skip the
            # otherwise-wasted decision-scale mask_to_indices.
            if self.use_o_cache or viz_enabled_for(scale_ind, layer_ind):
                inds, counts = mask_to_indices(mask, multiple_of, indices_pad_to)
                inds = inds[:,:,:,:k.shape[-2]]

        else:
            inds = torch.topk(bs, k=indices_count, dim=-1).indices      # e.g. [bs, num_heads, q_groups, k_len*topk]
            self._decision_indices_count = int(indices_count)

            if not self.use_o_cache:
                # O_cache off => csp_attn is NOT run at the decision scale, so the
                # k_len-padded indices below would never be consumed, and the sparse
                # scales re-pad anyway (map only reads inds[..., :indices_count]). Keep
                # just the compact top-k (GPU module attr) + the constant count; skip
                # the big offloaded indices/counts tensors. Also avoids truncating
                # int32 indices through the uint8 'attn.indices' offload buffer.
                self._decision_inds = inds.to(torch.int32)
            else:
                # TMA-stride padding (see _pad_uniform_indices; no-op stride pad for
                # the Triton provider, indices_pad_to == 1).
                inds, counts = _pad_uniform_indices(inds, indices_count, q, k, bm, indices_pad_to)

                self.storage.set_indices(inds)
                self.storage.set_counts(counts)

        if self.use_o_cache:
            if provider == 'cuda': o_cache = o.clone()
            else: o_cache = o

            #* ---------- 3. sparse-attn and cache O_cache ----------
                # O_cache = O_dense + (-1)ΔO
            o_cache = csp_attn(q, k, v, scale, inds, counts, o_cache, -1)

            self.storage.set_out_cache(o_cache)

        # Optional: dump decision-scale selected KV indices (env-gated).
        if viz_enabled_for(scale_ind, layer_ind):
            if self._decision_inds is not None: inds = self._decision_inds
            if self._decision_indices_count is not None:
                counts = _uniform_counts_like(inds, self._decision_indices_count)
            dump_selected_index_heatmap(
                inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
                q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]), bm=bm,
                sink_len=int(self.decision_sink_len), tag="decision")

        return o

    def _video_sparse_step(
        self, q: Tensor, k: Tensor, v: Tensor, scale,
        scale_ind: int, layer_ind: int, scale_schedule,
        attn_config, attn_kernel_config,
        repeat_idx: Optional[int] = None,
    ) -> Tensor:
        """SPARSE STEP: replay decision 稀疏模式 → 跨尺度映射 → csp_attn.

        O_t = O_cache + ΔO_new（use_o_cache）或 O_t = ΔO_new（o=None 直接返回 Δ）。
        """
        # print(f'SPARSE STEP: {do_full_step=}')
        bm = attn_kernel_config['bm']
        multiple_of = attn_kernel_config['counts_multiple_of']
        indices_pad_to = attn_kernel_config['indices_pad_to']
        provider = attn_config['provider']
        batch_size = int(q.shape[0])

        current_sink_len = max(int(k.shape[-2]) - int(q.shape[-2]), 0)
        assert current_sink_len == self.decision_sink_len, f'InfinityStar per-scale has same sink length'
        current_scale_shape = _as_scale_shape(scale_schedule, scale_ind)

        # ------ D-CSLA sparse hook: block-sparse FlexAttention, skip the csp chain ------
        if attn_config.get('dcsla_enabled', False):
            assert self._dcsla_mass is not None and scale_ind in self._dcsla_mass, \
                f'D-CSLA: no block mass for scale {scale_ind}; the decision step must run with dcsla_enabled'
            assert self._dcsla_expected_k_len[scale_ind] == int(k.shape[-2]), \
                f'D-CSLA: k_len drift vs decision-time geometry ({self._dcsla_expected_k_len[scale_ind]} vs {int(k.shape[-2])})'
            mass = _match_batch(self._dcsla_mass[scale_ind], batch_size)
            return dcsla_sparse_attention(
                q, k, v, scale, mass, sink_len=current_sink_len,
                scale_ind=scale_ind, layer_ind=layer_ind, attn_config=attn_config,
                scale_shape=current_scale_shape)

        expand_mode = str(attn_config.get('cs4a_index_expand_mode', 'footprint')).lower()

        # ------ 1. load sparse patterns ------
        self._load_and_prefetch_storage()

        # ------ 2. decision pattern -> current-scale (inds, counts) ------
        with lprof.region('attn.map', si=scale_ind, repeat=repeat_idx, layer=layer_ind,
                          strategy=expand_mode,
                          band=bool(attn_config.get('cs4a_footprint_union_band', False)),
                          q_len=int(q.shape[-2]), k_len=int(k.shape[-2]),
                          compressed=bool(attn_config['should_compress_indices']),
                          use_o_cache=self.use_o_cache):
            inds, counts = self._build_sparse_indices(
                q, k, batch_size, expand_mode, current_sink_len, current_scale_shape,
                attn_config, bm, multiple_of, indices_pad_to)

        # Optional: dump cross-scale mapped selected KV indices (env-gated).
        if viz_enabled_for(scale_ind, layer_ind):
            dump_selected_index_heatmap(
                inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
                q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]), bm=bm,
                sink_len=int(current_sink_len), tag="mapped")

        # Env-gated (SPARVAR_KBENCH_DUMP): real csp pattern for the kernel-speed bench.
        maybe_dump_csp_pattern(inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
                               q=q, k=k, scale=scale, sink_len=current_sink_len,
                               bm=bm, multiple_of=multiple_of, tag='cs4a_video')

        # ------ 3. load O_cache ------
        if self.use_o_cache:
            o = self._load_upsampled_o_cache(provider, batch_size, current_scale_shape, v.dtype,
                                             scale_ind=scale_ind, layer_ind=layer_ind)
        else:
            # No O_cache => O_t = ΔO_new. Pass o=None so csp_attn returns ΔO directly,
            # skipping the zeros init and the full-size accumulate passes (Triton path).
            o = None

        # ------ 4. sparse-attn Δ and update o ------
            # O_t = O_cache + (1)ΔO_new
        # print(f'%%% before csp_attn %%% {q.shape=}, {k.shape=}, {scale=}, {inds.shape=}')
        check_csp_attn_inputs(q, k, v, inds, counts, o, multiple_of=multiple_of, where="sparse")
        # counts_sum resolves at lprof.flush (post-sync): keeps the density
        # analysis of the old [sparse-profile] report without a hot-path sync.
        with lprof.region('attn.csp', si=scale_ind, repeat=repeat_idx, layer=layer_ind,
                          k_len=int(k.shape[-2]),
                          counts_sum=lprof.lazy(
                              lambda c=counts.detach(): int(c.to('cpu', torch.int64).sum().item()))):
            o = csp_attn(q, k, v, scale, inds, counts, o, 1)

        # ------ 5. optional O_cache rebase for the NEXT scale ------
        update_o_cache_mode = attn_config.get('update_o_cache_on_sparse_scale', 'none')
        has_next_scale = isinstance(scale_ind, int) and scale_schedule is not None and scale_ind < len(scale_schedule) - 1
        if (
            self.use_o_cache
            and update_o_cache_mode == 'dense_sparse_scale'
            and self.decision_scale_shape is not None
            and current_scale_shape is not None
            and has_next_scale
        ):
            dense_o, _ = self.dense_colsum_attn(q, k, v, scale)
            if provider == 'cuda':
                next_o_cache = dense_o.clone()
            else:
                next_o_cache = dense_o
            next_o_cache = csp_attn(q, k, v, scale, inds, counts, next_o_cache, -1)

            self.storage.set_indices(inds)
            self.storage.set_counts(counts)
            self.storage.set_out_cache(next_o_cache)
            self.decision_sink_len = current_sink_len
            self.decision_scale_shape = current_scale_shape
            self.decision_q_len = int(q.shape[-2])
            self.decision_k_len = int(k.shape[-2])

        return o

    def _build_sparse_indices(
        self, q: Tensor, k: Tensor, batch_size: int, expand_mode: str,
        current_sink_len: int, current_scale_shape,
        attn_config, bm: int, multiple_of: int, indices_pad_to: int,
    ) -> Tuple[Tensor, Tensor]:
        """Replay the stored decision pattern and map it to the current scale.

        Two decision sources × two mapping families:
          source  : compressed（bitpacked decision MASK）| non-compressed（compact
                    top-k module attr / offloaded k_len-padded indices+counts）
          mapping : footprint（mask 级 inverse parent-LUT 扩展，variable counts）
                    | legacy chain（q-group remap → coord LUT → none/1d/3d/frame_band）
        Returns ``(inds, counts)`` in the layout ``csp_attn`` consumes.
        """
        footprint_ready = False   # footprint mode builds (inds, counts) here -> skip the map chain
        #* [Index Mapping Exps] Carry a host-known count to avoid timing a counts_S.item() sync.
        source_indices_count = None

        if attn_config['should_compress_indices']:
            packed         = self.storage.get_indices()
            mask           = bitunpack(packed, self.mask_shape[self.layer_counter.cur_model_invocation_per_step])
            # FOOTPRINT mode: carry the decision sparse pattern to the current scale by its
            # geometric footprint (inverse parent-LUT mask gather), straight from the decision
            # MASK. Produces the final (inds, variable counts), so it skips the whole
            # mask_to_indices -> map_inds_counts -> coord-map -> budget-expand chain below.
            # Video path only; any other mode/scale falls through to the legacy chain.
            if expand_mode == 'footprint':
                mask = _match_batch(mask, batch_size)
                inds, counts = self._footprint_indices(
                    mask, q, k, current_sink_len, current_scale_shape,
                    attn_config, bm, multiple_of, indices_pad_to)
                footprint_ready = True
            else:
                inds, counts   = mask_to_indices(mask, multiple_of, indices_pad_to)
                inds = inds[:,:,:,:self.decision_k_len]

        else:
            if not self.use_o_cache:
                # storage opt: replay the decision scale's COMPACT top-k (module attr) and
                # rebuild the constant counts here. Downstream map/pad is unchanged --
                # map_inds_counts only reads inds[..., :indices_count] before re-padding.
                inds = self._decision_inds
                counts = _uniform_counts_like(inds, self._decision_indices_count)
            else:
                inds   = self.storage.get_indices()     # after scale10, [bs, num_heads, q_groups, k_len]
                counts = self.storage.get_counts()      # after scale10, [bs, num_heads, q_groups], value=topk(k_len)
            #* [Index Mapping Exps] Reuse the decision top-k count without synchronizing counts.
            source_indices_count = self._decision_indices_count

            # FOOTPRINT mode for the non-compressed path. The compressed path footprints
            # straight from the decision bool MASK, but here only the compact top-k was
            # kept -- so rebuild that selection mask (scatter the top-k into a
            # [B,H,Gs,decision_k_len] grid) and run the SAME inverse parent-LUT footprint
            # expansion. Produces the final (inds, variable counts) -> skip the map chain.
            if expand_mode == 'footprint':
                inds = _match_batch(inds, batch_size)
                decision_mask = torch.zeros(
                    inds.shape[0], inds.shape[1], inds.shape[2], self.decision_k_len,
                    dtype=torch.bool, device=inds.device,
                )
                decision_mask.scatter_(-1, inds.to(torch.int64).clamp_(0, self.decision_k_len - 1), True)
                inds, counts = self._footprint_indices(
                    decision_mask, q, k, current_sink_len, current_scale_shape,
                    attn_config, bm, multiple_of, indices_pad_to)
                footprint_ready = True

        # Legacy cross-scale map chain (none / 1d / 3d / frame_band). FOOTPRINT already
        # produced the final (inds, variable counts) from the decision mask above, so skip it.
        if not footprint_ready:
            inds, counts = self._legacy_video_map_chain(
                inds, counts, source_indices_count, q, k, batch_size, expand_mode,
                current_sink_len, current_scale_shape,
                attn_config, bm, multiple_of, indices_pad_to)

        return inds, counts

    def _footprint_indices(
        self, decision_mask: Tensor, q: Tensor, k: Tensor,
        current_sink_len: int, current_scale_shape,
        attn_config, bm: int, multiple_of: int, indices_pad_to: int,
    ) -> Tuple[Tensor, Tensor]:
        """Footprint expansion: decision MASK → current-scale (inds, variable counts)."""
        use_band = bool(attn_config.get('cs4a_footprint_union_band', False))
        neighbor_frames = int(attn_config.get('cs4a_band_neighbor_frames', 1))
        # Compact mask_to_indices output width ('cs4a_compact_index_width: auto';
        # -1 = legacy full k_len-width allocation). The analytic bound is only valid
        # when every decision-mask row provably holds <= _decision_indices_count keys:
        # pure top-k rows (random fill stays off at the random_and_topk call site,
        # no singleton static-mask union) and the compact-top-k scatter rebuild. The
        # non-compressed O_cache path scatters k_len-padded indices with an
        # UNINITIALIZED tail -> row popcount unbounded -> stays full-width.
        out_width = resolve_footprint_out_width(
            mode=attn_config.get('cs4a_compact_index_width', 'full'),
            provider=attn_config['provider'],
            decision_scale_shape=self.decision_scale_shape,
            current_scale_shape=current_scale_shape,
            current_k_len=int(k.shape[-2]),
            decision_indices_count=self._decision_indices_count,
            decision_rows_bounded=(
                (attn_config['should_compress_indices'] or not self.use_o_cache)
                and singleton_static_mask is None
            ),
            use_band=use_band,
            neighbor_frames=neighbor_frames,
            multiple_of=multiple_of,
            indices_pad_to=indices_pad_to,
        )
        return build_footprint_union_indices(
            decision_mask=decision_mask,
            decision_sink_len=self.decision_sink_len,
            decision_scale_shape=self.decision_scale_shape,
            decision_k_len=self.decision_k_len,
            current_sink_len=current_sink_len,
            current_k_len=int(k.shape[-2]),
            current_scale_shape=current_scale_shape,
            current_q_len=int(q.shape[-2]),
            bm=bm,
            multiple_of=multiple_of,
            indices_pad_to=indices_pad_to,
            neighbor_frames=neighbor_frames,
            use_band=use_band,
            out_width=out_width,
        )

    def _legacy_video_map_chain(
        self, inds: Tensor, counts: Tensor, source_indices_count: Optional[int],
        q: Tensor, k: Tensor, batch_size: int, expand_mode: str,
        current_sink_len: int, current_scale_shape,
        attn_config, bm: int, multiple_of: int, indices_pad_to: int,
    ) -> Tuple[Tensor, Tensor]:
        """Legacy cross-scale map chain（expand_mode: none / 1d / 3d / frame_band）:
        q-group remap → point2point coord LUT → optional budget/band expand → uniform pad.
        """
        if source_indices_count is None:
            s_indices_count = int(counts[0][0][0])
        else:
            s_indices_count = int(source_indices_count)
        inds = _match_batch(inds, batch_size)
        counts = _match_batch(counts, batch_size)

        #* --- q_groups dim map ---
        inds, counts = map_inds_counts_across_scales(
            inds, counts, cur_q_len=q.shape[-2], cur_k_len=k.shape[-2],
            bm=bm, indices_pad_to=indices_pad_to, device=inds.device,
            qgroup_map_mode=attn_config.get('cs4a_qgroup_map_mode', 'nearest'),
            qgroup_interp_align=attn_config.get('cs4a_qgroup_interp_align', 'center'),
            sink_len=int(self.decision_sink_len),
            source_k_len=int(self.decision_k_len),
            #* [Index Mapping Exps] Pass the host-known count into the synchronization-free path.
            source_indices_count=source_indices_count)

        # if viz_enabled_for(scale_ind, layer_ind):
        #     # Stage 1: group-dim remap only — decision-scale top-k regrouped to the
        #     # current query groups, values still in DECISION key coordinates.
        #     dump_selected_index_heatmap(
        #         inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
        #         q_len=int(q.shape[-2]), kv_len=int(self.decision_k_len), bm=bm,
        #         sink_len=int(self.decision_sink_len), tag="map_group")

        frame_band_ready = False   # frame_band returns (inds, counts) directly -> skip the pad block
        # InfinityStar video KV is [image/text sink prefix] + [current video scale].
        if expand_mode == 'none':
            indices_count = int(inds.shape[-1])
        elif not attn_config['use_current_scale_topklen']:
            indices_count = s_indices_count
        else:
            indices_count = _target_indices_count(int(k.shape[-2]), attn_config['top_keys'], multiple_of)

        #* --- point2point KVs mapping ---
        inds = map_infinitystar_sparse_kv_inds_lut(
            inds_S=inds,
            decision_sink_len=self.decision_sink_len,
            current_sink_len=current_sink_len,
            decision_scale_shape=self.decision_scale_shape,
            current_scale_shape=current_scale_shape,
            current_k_len=int(k.shape[-2]),
            decision_k_len=int(self.decision_k_len),
        )

        # if viz_enabled_for(scale_ind, layer_ind):
        #     # Stage 2: coordinate translation — mapped top-k centers now in CURRENT
        #     # key coordinates (one per center, before budget fill / band union).
        #     dump_selected_index_heatmap(
        #         inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
        #         q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]), bm=bm,
        #         sink_len=int(current_sink_len), tag="map_coord")

        if expand_mode == 'frame_band':
            # Chipmunk-style mask union (no sort): full-frame band_mask | mapped
            # top-k centers -> mask_to_indices -> (inds, variable counts) consumed
            # directly by csp_attn. Skips the uniform-count pad block below.
            inds, counts = build_frame_band_union_indices(
                inds,
                current_sink_len=current_sink_len,
                current_k_len=int(k.shape[-2]),
                current_scale_shape=current_scale_shape,
                bm=bm,
                q_len=int(q.shape[-2]),
                neighbor_frames=int(attn_config.get('cs4a_band_neighbor_frames', 1)),
                multiple_of=multiple_of,
                indices_pad_to=indices_pad_to,
            )
            frame_band_ready = True
        elif expand_mode == '1d' or expand_mode == '3d':
            inds = expand_mapped_indices_to_budget(
                inds,
                target_count=indices_count,
                current_sink_len=current_sink_len,
                current_k_len=int(k.shape[-2]),
                current_scale_shape=current_scale_shape,
                expand_mode=expand_mode,
            )

        # inds change, need pad again. frame_band already produced (inds, counts) via
        # mask_to_indices (variable counts), so it skips this uniform-count padding.
        if not frame_band_ready:
            inds, counts = _pad_uniform_indices(inds, indices_count, q, k, bm, indices_pad_to)

        return inds, counts

    # ──────────────────────────────────────────────────────────────────────
    # Infinity (image) CS4A path
    # ──────────────────────────────────────────────────────────────────────

    @torch.compiler.disable
    def _fast_attention_image(
        self, q: Tensor, k: Tensor, v: Tensor, scale,
        inference_step: int, do_full_step: bool,
        scale_ind: int,
        layer_ind: int,
        scale_schedule=None,
    ) -> Tensor:
        """Infinity (image) CS4A path — same skeleton as the InfinityStar video path
        (``_fast_attention``), specialized for the image KV layout
        KV = Concat(scale_0..scale_k):

          decision step : dense_colsum -> top-k -> compact / O_cache storage (shared code shape)
          sparse step   : Query Block Homography (shared ``map_inds_counts_across_scales``)
                          -> Decompose-Align-Project coordinate LUT
                             (``cs4a_image_kv_map: relative``; ``identity`` keeps decision
                              coordinates — valid because the decision KV is a strict prefix
                              of every later scale's KV)
                          -> union(sink scales | current-scale local window | mapped centers)
                             with variable per-group counts -> csp_attn (+ optional O_cache).

        With ``cs4a_identity_cumulative``, identity mapping replays the previous
        sparse scale's valid selection instead, retaining its added local band.
        O_cache still comes from the decision scale.

        With ``dcsla_enabled`` both steps divert to the D-CSLA path instead: the
        decision step transfers the colsum MASS to every later scale
        (``compute_decision_block_mass_image``) and the sparse step runs the
        block-sparse FlexAttention forward. Same two hook points as the video
        path in ``_video_decision_step`` / ``_video_sparse_step``.
        """
        attn_config = GLOBAL_CONFIG['attn']
        if layer_ind < attn_config['first_n_dense_layers']:
            # Protect image CS4A layers at both decision and sparse scales.
            o, _ = dense_attn(q, k, v, scale=scale)
            return o
        attn_kernel_config = get_kernel_config_attn()
        bm = attn_kernel_config['bm']
        multiple_of = attn_kernel_config['counts_multiple_of']
        indices_pad_to = attn_kernel_config['indices_pad_to']
        provider = attn_config['provider']
        batch_size = int(q.shape[0])

        if scale_schedule is None:
            raise RuntimeError(
                "CS4A image sparse attention requires scale_schedule; pass it into "
                "SparseDiffAttn.forward from the SelfAttention block."
            )
        # Image KV layout bookkeeping: cum[i] = token count of scales [0, i).
        n_list = [int(t) * int(h) * int(w) for (t, h, w) in scale_schedule[:scale_ind + 1]]
        cum = [0]
        for n in n_list:
            cum.append(cum[-1] + n)
        assert cum[scale_ind + 1] == int(k.shape[-2]), (
            f"image KV length {int(k.shape[-2])} != scale_schedule cumulative "
            f"{cum[scale_ind + 1]} at scale {scale_ind}"
        )
        current_scale_shape = _as_scale_shape(scale_schedule, scale_ind)

        cumulative_identity = bool(attn_config.get('cs4a_identity_cumulative', False))
        if cumulative_identity:
            validate_identity_cumulative_config(attn_config)

        # ─────────── FULL STEP (SPARSE DECISION SCALE) ───────────
        if do_full_step:
            # New image or SkipVAR re-decision: never inherit the previous selection.
            self._image_identity_indices.clear()
            #* ----------- 1. compute colsum ----------
            with lprof.region('attn.colsum', si=scale_ind, layer=layer_ind, k_len=int(k.shape[-2])):
                o, bs = self.dense_colsum_attn(q, k, v, scale)
            self.decision_sink_len = 0          # image: no shared ctx prefix
            self.decision_scale_shape = current_scale_shape
            self.decision_q_len = int(q.shape[-2])
            self.decision_k_len = int(k.shape[-2])

            # ------ D-CSLA decision hook: transfer block mass, skip legacy topk/storage ------
            if attn_config.get('dcsla_enabled', False):
                assert not self.use_o_cache and attn_config.get('update_o_cache_on_sparse_scale', 'none') == 'none', \
                    'D-CSLA v1: O_cache paths unsupported (set use_o_cache: false)'
                # Rebuilt wholesale each decision step -> no stale entries across images.
                self._dcsla_mass, self._dcsla_expected_k_len = compute_decision_block_mass_image(
                    bs, scale_schedule=scale_schedule, decision_scale_ind=scale_ind,
                    decision_k_len=int(k.shape[-2]),
                    block_size=int(attn_config.get('flex_block_size', 128)),
                    normalize=bool(attn_config.get('dcsla_mass_normalize', True)))
                # Raw-evidence viz: token-wise colsum at the decision scale.
                # Image: no ctx prefix (sink_len 0), scale boundaries instead
                # of frame lines. SPARVAR_VIZ_DECISION_COLSUM=0 disables.
                if viz_enabled_for(scale_ind, layer_ind):
                    dump_decision_colsum_viz(
                        bs, scale_ind=scale_ind, layer_ind=layer_ind,
                        q_len=int(q.shape[-2]), k_len=int(k.shape[-2]),
                        block_size=int(attn_config.get('flex_block_size', 128)),
                        sink_len=0, scale_shape=None,
                        kv_scale_boundaries=image_kv_scale_boundaries(
                            scale_schedule, int(k.shape[-2]), 0),
                        tag="dcsla")
                return o

            indices_count = _target_indices_count(int(k.shape[-2]), attn_config['top_keys'], multiple_of)

            #* ---------- 2. TopK and get sparse KV index ----------
            with lprof.region('attn.topk', si=scale_ind, layer=layer_ind,
                              use_o_cache=self.use_o_cache):
                inds = torch.topk(bs, k=indices_count, dim=-1).indices

                if not self.use_o_cache:
                    # storage opt (same as the video path): keep the compact top-k GPU-resident.
                    self._decision_inds = inds.to(torch.int32)
                    self._decision_indices_count = int(indices_count)
                else:
                    # TMA-stride padding (no-op for the Triton provider, indices_pad_to == 1).
                    inds, counts = _pad_uniform_indices(inds, indices_count, q, k, bm, indices_pad_to)
                    self.storage.set_indices(inds)
                    self.storage.set_counts(counts)

            if self.use_o_cache:
                #* ---------- 3. sparse-attn and cache O_cache ----------
                    # O_cache = O_dense + (-1)ΔO
                with lprof.region('attn.csp', si=scale_ind, layer=layer_ind,
                                  k_len=int(k.shape[-2]), role='decision_ocache'):
                    if provider == 'cuda': o_cache = o.clone()
                    else: o_cache = o
                    o_cache = csp_attn(q, k, v, scale, inds, counts, o_cache, -1)
                    self.storage.set_out_cache(o_cache)

            # Optional: dump decision-scale selected KV indices (env-gated).
            if cumulative_identity:
                self._image_identity_indices.reset(
                    inds[..., :indices_count] if self.use_o_cache else self._decision_inds,
                    scale_ind=scale_ind,
                    kv_len=int(k.shape[-2]))
            if viz_enabled_for(scale_ind, layer_ind):
                dump_inds = self._decision_inds if self._decision_inds is not None else inds
                dump_counts = _uniform_counts_like(dump_inds, indices_count)
                dump_selected_index_heatmap(
                    dump_inds, dump_counts, scale_ind=scale_ind, layer_ind=layer_ind,
                    q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]), bm=bm,
                    sink_len=0, tag="decision")

            return o

        # ─────────── SPARSE STEP ───────────
        # sink / local-window geometry, shared by the D-CSLA hook and both CS4A
        # expansion modes
        sink_scales = int(attn_config.get('cs4a_sink_scales', 5))
        sink_len = cum[min(max(sink_scales, 0), scale_ind)]         # first 5 scales: 121
        ws_list = attn_config.get('cs4a_ws') or []
        if scale_ind < len(ws_list):
            win_size = int(ws_list[scale_ind])
        else:
            win_size = int(ws_list[-1]) if ws_list else 1

        # ------ D-CSLA sparse hook: block-sparse FlexAttention, skip the csp chain ------
        if attn_config.get('dcsla_enabled', False):
            assert self._dcsla_mass is not None and scale_ind in self._dcsla_mass, \
                f'D-CSLA: no block mass for scale {scale_ind}; the decision step must run with dcsla_enabled'
            assert self._dcsla_expected_k_len[scale_ind] == int(k.shape[-2]), \
                f'D-CSLA: k_len drift vs decision-time geometry ({self._dcsla_expected_k_len[scale_ind]} vs {int(k.shape[-2])})'
            mass = _match_batch(self._dcsla_mass[scale_ind], batch_size)
            return dcsla_sparse_attention(
                q, k, v, scale, mass, sink_len=sink_len,
                scale_ind=scale_ind, layer_ind=layer_ind, attn_config=attn_config,
                scale_shape=current_scale_shape, image_scale_schedule=scale_schedule)

        # ------ 1. load the decision-scale sparse pattern ------
        with lprof.region('attn.load', si=scale_ind, layer=layer_ind,
                          use_o_cache=self.use_o_cache):
            if not cumulative_identity or self.use_o_cache:
                self.storage.load_async()       # no-op when offloading is disabled / nothing stored
                self.storage.load_async_wait()
            if cumulative_identity:
                inds, counts = None, None  # The previous sparse selection is GPU-resident.
            elif not self.use_o_cache:
                # replay the decision scale's COMPACT top-k (module attr), rebuild constant counts.
                inds = self._decision_inds
                counts = _uniform_counts_like(inds, self._decision_indices_count)
            else:
                inds = self.storage.get_indices()
                counts = self.storage.get_counts()
            inds = _match_batch(inds, batch_size)       # for drop-uncond / skip-scales batch shrink
            counts = _match_batch(counts, batch_size)

        expand_mode = str(attn_config.get('cs4a_image_expand_mode', 'center_union')).lower()
        _t = lprof.timer('attn.map', si=scale_ind, layer=layer_ind, strategy=expand_mode)
        if cumulative_identity:
            inds, counts = self._image_identity_indices.advance(
                scale_ind=scale_ind, batch_size=batch_size,
                q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]),
                scale_shape=current_scale_shape, sink_len=sink_len, bm=bm,
                win_size=win_size, multiple_of=multiple_of,
                indices_pad_to=indices_pad_to, provider=provider)
            _t.stop('identity_cumulative')
        elif expand_mode == 'footprint':
            #* --- FOOTPRINT: inverse parent-LUT mask expansion ---
            # Rebuild the decision selection MASK and carry it to the current scale by
            # its geometric footprint. Relative alignment is built into the parent LUT
            # (cs4a_image_kv_map is not consulted on this path). Only the first
            # counts[0,0,0] entries of inds are real top-k — the O_cache path stores
            # k_len-padded indices whose TAIL IS UNINITIALIZED memory, so it must not
            # be scattered.
            s_indices_count = int(counts[0][0][0])
            decision_mask = torch.zeros(
                inds.shape[0], inds.shape[1], inds.shape[2], self.decision_k_len,
                dtype=torch.bool, device=inds.device,
            )
            decision_mask.scatter_(
                -1,
                inds[..., :s_indices_count].to(torch.int64).clamp_(0, self.decision_k_len - 1),
                True,
            )
            # footprint has no separate group/coord kernels; keep the 3-segment
            # structure stable ('group' = decision-mask rebuild above, 'coord' ~ 0)
            _t.mark('group')
            _t.mark('coord')
            inds, counts = build_infinity_footprint_union_indices(
                decision_mask=decision_mask,
                decision_scale_ind=self.decision_scale,
                current_scale_ind=scale_ind,
                scale_schedule=scale_schedule,
                decision_k_len=int(self.decision_k_len),
                current_k_len=int(k.shape[-2]),
                current_q_len=int(q.shape[-2]),
                sink_len=sink_len,
                bm=bm,
                multiple_of=multiple_of,
                indices_pad_to=indices_pad_to,
                union_local=bool(attn_config.get('cs4a_footprint_union_local', False)),
                current_block_start=cum[scale_ind],
                # full per-historical-scale list (cross-scale band semantics);
                # the center_union path below still consumes the scalar win_size
                win_size=ws_list,
            )
            _t.stop('band')
        else:
            #* --- CENTER_UNION: q-group map -> coordinate LUT -> mask union ---
            #* --- q_groups dim map (Query Block Homography; shared with the video path) ---
            inds, counts = map_inds_counts_across_scales(
                inds, counts, cur_q_len=q.shape[-2], cur_k_len=k.shape[-2],
                bm=bm, indices_pad_to=indices_pad_to, device=inds.device,
                qgroup_map_mode=attn_config.get('cs4a_qgroup_map_mode', 'nearest'),
                qgroup_interp_align=attn_config.get('cs4a_qgroup_interp_align', 'center'),
                sink_len=0,
                source_k_len=int(self.decision_k_len))
            _t.mark('group')

            #* --- point2point KVs mapping: Decompose-Align-Project LUT ---
            if str(attn_config.get('cs4a_image_kv_map', 'relative')).lower() == 'relative':
                inds = map_infinity_sparse_kv_inds_lut(
                    inds_S=inds,
                    decision_scale_ind=self.decision_scale,
                    current_scale_ind=scale_ind,
                    scale_schedule=scale_schedule,
                    current_k_len=int(k.shape[-2]),
                    decision_k_len=int(self.decision_k_len),
                )
            _t.mark('coord')

            #* --- union: sink scales | current-scale local window | mapped centers ---
            inds, counts = build_image_local_union_indices(
                inds,
                sink_len=sink_len,
                current_k_len=int(k.shape[-2]),
                current_scale_shape=current_scale_shape,
                current_block_start=cum[scale_ind],
                bm=bm,
                q_len=int(q.shape[-2]),
                win_size=win_size,
                multiple_of=multiple_of,
                indices_pad_to=indices_pad_to,
            )
            _t.stop('band')

        # Optional: dump cross-scale mapped selected KV indices (env-gated).
        if viz_enabled_for(scale_ind, layer_ind):
            dump_selected_index_heatmap(
                inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
                q_len=int(q.shape[-2]), kv_len=int(k.shape[-2]), bm=bm,
                sink_len=int(sink_len), tag="mapped")

        # Env-gated (SPARVAR_KBENCH_DUMP): real csp pattern for the kernel-speed bench.
        maybe_dump_csp_pattern(inds, counts, scale_ind=scale_ind, layer_ind=layer_ind,
                               q=q, k=k, scale=scale, sink_len=sink_len,
                               bm=bm, multiple_of=multiple_of, tag='cs4a_image')

        # ------ 2. load O_cache ------
        if self.use_o_cache:
            with lprof.region('attn.ocache', si=scale_ind, layer=layer_ind,
                              q_len=int(q.shape[-2])):
                o = self._load_upsampled_o_cache(provider, batch_size, current_scale_shape, v.dtype,
                                                 scale_ind=scale_ind, layer_ind=layer_ind)
        else:
            # No O_cache => O_t = ΔO_new (csp_attn returns ΔO directly for o=None).
            o = None

        # ------ 3. sparse-attn Δ and update o ------
            # O_t = O_cache + (1)ΔO_new
        check_csp_attn_inputs(q, k, v, inds, counts, o, multiple_of=multiple_of, where="image-sparse")
        # counts_sum resolves at lprof.flush (post-sync): per-scale density without a hot-path sync
        with lprof.region('attn.csp', si=scale_ind, layer=layer_ind, k_len=int(k.shape[-2]),
                          counts_sum=lprof.lazy(
                              lambda c=counts.detach(): int(c.to('cpu', torch.int64).sum().item()))):
            o = csp_attn(q, k, v, scale, inds, counts, o, 1)

        return o

    def forward(self, q: Tensor, k: Tensor, v: Tensor, scale, do_full_step: bool = False, inference_step: int = 0,
                scale_ind: int = 0, layer_ind: int = 0, scale_schedule=None,
                #* [Index Mapping Exps] Profiling metadata forwarded to the video sparse path.
                repeat_idx: Optional[int] = None,
                # pattern-stats only: None = unknown (callers that do not pass it keep
                # the old record-every-call behavior); the video caller passes the real
                # flag so repeated post-decision scales record only their final repeat
                last_repetition_step: Optional[bool] = None) -> Tensor:
        # check if Chipmunk is enabled
        if not GLOBAL_CONFIG['attn']['is_enabled']:
            attn_config = GLOBAL_CONFIG['attn']
            # [GQA] This collection-only dense path (pattern-stats hooks, dense_colsum
            # probe, SDPA fallback) assumes q/k/v share the head count, but the caller
            # skips the repeat_kv materialization since the GQA head-pack change;
            # re-expand grouped KV here (repeat_interleave(dim=1) == the removed
            # repeat_kv). The production sparse paths below are unaffected.
            if k.shape[1] != q.shape[1]:
                n_rep = q.shape[1] // k.shape[1]
                k = k.repeat_interleave(n_rep, dim=1)
                v = v.repeat_interleave(n_rep, dim=1)
            stats_sink_len = max(int(k.shape[-2]) - int(q.shape[-2]), 0)
            stats_scale_shape = _as_scale_shape(scale_schedule, scale_ind)
            # Record once per scale: the decision step plus the final repetition.
            # Without this gate every repetition of a repeated post-decision scale
            # (e.g. 480p scale 26, repeats=2) writes a duplicate stats row.
            # The caller derives the flag from a numpy comparison, so it arrives as
            # np.bool_ — coerce with bool(); an identity test against False never fires.
            skip_stats = (last_repetition_step is not None
                          and not bool(last_repetition_step) and not do_full_step)
            if not skip_stats:
                maybe_record_infinitystar_attention_stats(
                    q, k, scale=scale, scale_ind=scale_ind, layer_ind=layer_ind,
                    scale_shape=stats_scale_shape, sink_len=stats_sink_len,
                    do_full_step=do_full_step, attn_config=attn_config)
            if do_full_step and pattern_stats_enabled_for_layer(layer_ind):
                _, bs = self.dense_colsum_attn(q, k, v, scale)
                maybe_record_infinitystar_decision_colsum(
                    bs, scale_schedule=scale_schedule, decision_scale_ind=scale_ind,
                    layer_ind=layer_ind, decision_sink_len=stats_sink_len,
                    decision_scale_shape=stats_scale_shape, decision_k_len=int(k.shape[-2]))
            out = F.scaled_dot_product_attention(q, k, v)
            self.layer_counter.increment()
            return out

        # Determine whether the current step is dense or sparse
        # do_full_step = self.layer_counter.should_do_full_attn_step()
        inference_step = self.layer_counter.cur_inference_step

        # model_family is bound at init: 'image' -> Infinity path, 'video' -> InfinityStar path
        if self.model_family == 'image':
            out = self._fast_attention_image(q, k, v, scale, inference_step, do_full_step,
                                             scale_ind, layer_ind, scale_schedule)
        else:
            #* [Index Mapping Exps] Forward repeat metadata to the deferred profiler.
            out = self._fast_attention(q, k, v, scale, inference_step, do_full_step,
                                       scale_ind, layer_ind, scale_schedule,
                                       repeat_idx=repeat_idx)

        # Update the global counter
        self.layer_counter.increment()
        self.storage.complete_cur_layer()
        # print(f'### update layer_counter, {self.layer_counter.cur_inference_step=} ###')

        return out

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


if __name__ == '__main__':
    pass
