import math
import os
from collections import OrderedDict
from typing import Dict, List, Literal, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import Tensor, nn
from torch.nn import functional as F

try:
    from kernels.sparse_attn.sparse_attn_ops import mask_to_indices
except ImportError:  # no triton (e.g. CUDA-less dev box): pure-torch fallback below
    mask_to_indices = None


# ------------------------------------------------------------------
# mask -> (indices, counts): pure-PyTorch fallback for the Triton path
# ------------------------------------------------------------------
@torch.no_grad()
def mask_to_indices_torch(
    mask: torch.Tensor,          # [B, H, M, N] bool
    multiple_of: int,
    pad_to_multiple_of: int,
    chunk_elems: int = 1 << 24,  # ~16M-element row chunks -> bounded transient buffers
    out_width: int = -1,         # >0: narrow the returned row width (parity with the CUDA op)
):
    """Pure-PyTorch drop-in for ``chipmunk.mask_to_indices`` (used when the chipmunk
    CUDA op is unavailable, e.g. on A100 with the Triton attention provider).

    Per row ``[b, h, m]`` (mirrors the CUDA contract):
      ``count``                    = ``ceil(popcount / multiple_of) * multiple_of``
      ``indices[0 : popcount]``    = the selected columns (ascending)
      ``indices[popcount : count]``= the first ``count - popcount`` UNSELECTED columns
                                     (ascending) -> fills the rounding slack
      ``indices[count : pad_n]``   = remaining unselected columns (never read by csp_attn)
    with ``pad_n = ceil(N / pad_to_multiple_of) * pad_to_multiple_of``.

    Each row is partitioned in O(N) with a single int32 ``cumsum`` + ``scatter``: selected
    columns to the front (ascending), then unselected (ascending). Equivalent layout to a
    stable argsort but WITHOUT the int64 permutation and stable-sort workspace argsort
    needs (~3-4x the int32 result, tens of GB at the largest video scales -> 40GB OOM, and
    it can also trip CUDA sort limits near 2^31 elements). Rows are processed in chunks so
    the transient buffers stay bounded regardless of ``B*H*Gk*k_len``; only the int32 index
    result is persistent (the same size the CUDA kernel writes).

    Consistency: the selected SET, ``count`` and the rounding-slack fill are identical to
    the CUDA op, so csp_attn / frame_band / footprint produce element-identical results
    (softmax over the gathered keys is order-invariant). Only the WITHIN-ROW order of the
    selected block differs from the CUDA thread-strided order -- irrelevant to those
    paths; the position-priority 1d/3d budget expansion is the lone order-sensitive
    consumer and is not on the frame_band/footprint path.
    """
    assert mask.dim() == 4 and mask.dtype == torch.bool, f"{mask.shape=}, {mask.dtype=}"
    B, H, M, N = mask.shape
    pad_n = ((N + pad_to_multiple_of - 1) // pad_to_multiple_of) * pad_to_multiple_of
    device = mask.device

    mask2d = mask.reshape(-1, N)                                             # [R, N]
    R = mask2d.shape[0]
    indices = torch.zeros((R, pad_n), device=device, dtype=torch.int32)      # persistent result
    counts = torch.empty((R,), device=device, dtype=torch.int32)
    col_idx = torch.arange(N, device=device, dtype=torch.int32).view(1, N)   # [1, N]

    rows_per_chunk = max(1, int(chunk_elems) // max(N, 1))
    for start in range(0, R, rows_per_chunk):
        end = min(start + rows_per_chunk, R)
        m = mask2d[start:end]                                                # [c, N] bool
        total = m.sum(dim=-1, keepdim=True).to(torch.int32)                  # [c, 1] popcount
        sel_rank = torch.cumsum(m, dim=-1, dtype=torch.int32)                # [c, N] 1..popcount over selected
        # write slot per column: selected -> rank-1 (front, ascending); unselected col i ->
        # total + (#unselected up to i) - 1 = total + i - sel_rank[i] (after the selected
        # block, ascending). One cumsum + arange; no int64 sort buffer.
        pos = torch.where(m, sel_rank - 1, total + col_idx - sel_rank)       # [c, N] int32 in [0, N)
        indices[start:end].scatter_(-1, pos.to(torch.int64), col_idx.expand(end - start, N))
        cnt = (total.squeeze(-1) + multiple_of - 1) // multiple_of * multiple_of
        # Never let counts exceed the real column count: an all-selected group would
        # otherwise round past N and csp_attn would read past the row.
        counts[start:end] = torch.clamp(cnt, max=N)

    if out_width is not None and 0 < out_width < pad_n:
        # Correctness-parity narrow output only (the memory win belongs to the CUDA
        # op): the scatter above addresses positions in [0, N), so the row is built at
        # full width and sliced down. Caller guarantees every row's padded count fits
        # in out_width (see models/sparvar/index_width_utils.py).
        indices = indices[:, :out_width].contiguous()
        pad_n = out_width

    return indices.view(B, H, M, pad_n), counts.view(B, H, M)


def _have_cuda_mask_to_indices():
    """``mask_to_indices`` is a plain CUDA/CUB op (NOT ThunderKittens), so chipmunk
    builds it on every arch incl. A100 -- setup.py compiles the portable indexed_io ops
    and skips the Hopper-only attn/mlp kernels. Detect whether the compiled op is actually
    registered (independent of the attention provider)."""
    if mask_to_indices is None:
        return False
    try:
        import chipmunk  # noqa: F401  -- loads the TORCH_LIBRARY static initializers
    except Exception:
        pass
    try:
        return hasattr(torch.ops.chipmunk, "mask_to_indices")
    except Exception:
        return False


_USE_CUDA_MASK_TO_INDICES = _have_cuda_mask_to_indices()
print(f"[index_map] mask_to_indices backend = "
      f"{'CUDA (chipmunk)' if _USE_CUDA_MASK_TO_INDICES else 'pure-torch fallback'}")


def _mask_to_indices(mask, multiple_of, pad_to_multiple_of, out_width: int = -1):
    """Dispatch mask -> (indices, counts). Prefer the compiled CUDA/CUB op when present
    (int32 warp-scan, O(N), minimal memory -- available on A100 too, regardless of the
    attention provider); otherwise use the pure-torch fallback (chunked cumsum).
    ``out_width > 0`` narrows the output row width (compact allocation; the caller must
    prove every row's padded count fits -- see index_width_utils)."""
    if _USE_CUDA_MASK_TO_INDICES:
        return mask_to_indices(mask, multiple_of, pad_to_multiple_of, out_width)
    return mask_to_indices_torch(mask, multiple_of, pad_to_multiple_of, out_width=out_width)


# ----------------------------
# Index Mapping Visualization
# ----------------------------
@torch.no_grad()
def selection_mask_from_inds(
    inds_K: torch.Tensor,   # [B, H, Gk, Kk]  每个 query-group 的选 K 索引（已经是跨尺度映射后的）
    q_len: int,             # 目标尺度的 Q 序列长度（这一尺度的 token 数）
    kv_len: int,            # 目标尺度可见的 KV 总长度（历史+当前）
    bm: int,                # query-group 大小（64 或 192）
    clamp_kv_len: Optional[int] = None,  # 额外裁剪上界（一般 = kv_len）
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """
    输出：mask，形状 [B, H, q_len, kv_len]，元素∈{0,1}，1 表示该 (query_row, key_col) 被选中。
    实现细节：
      - 行到组的映射：row_groups[r] = clamp(floor(r / bm), 0, Gk-1)
      - 向量化展开：inds_per_row = index_select(inds_K, dim=2, index=row_groups)
      - 构建二值图：mask.scatter_(dim=-1, index=inds_per_row, value=1)
    """
    assert inds_K.ndim == 4, f"{inds_K.shape=}"
    B, H, Gk, Kk = inds_K.shape
    device = device or inds_K.device

    # 逐行对应到它属于哪个 query-group
    row_groups = torch.div(torch.arange(q_len, device=device), bm, rounding_mode='trunc')
    row_groups = torch.clamp(row_groups, max=Gk - 1).to(torch.long)          # [q_len]

    # 取出每一行对应组的 K 索引： [B, H, q_len, Kk_sel]
    inds_per_row = inds_K.index_select(dim=2, index=row_groups)              # [B,H,q_len,Kk]
    if clamp_kv_len is None:
        clamp_kv_len = kv_len
    inds_per_row = inds_per_row.clamp_(0, max(0, clamp_kv_len - 1)).long()

    # 构建二值选择图：一次 scatter 完成
    mask = torch.zeros((B, H, q_len, kv_len), dtype=torch.uint8, device=device)
    mask.scatter_(dim=-1, index=inds_per_row, value=1)
    return mask


@torch.no_grad()
def save_attention_selection_visuals(
    inds_K: torch.Tensor,           # [B, H, Gk, Kk]
    q_len: int,
    kv_len: int,
    bm: int,
    out_dir: str,
    prefix: str = "attn_sel",
    downsample_q: int = 1,          # q 维下采样因子（>=1）
    downsample_k: int = 1,          # k 维下采样因子（>=1）
    sink_len: int = 0,              # 若 >0，在图上画一条竖线标注 sink 的右边界
    clamp_kv_len: Optional[int] = None,
):
    """
    为每个 (batch, head) 保存一张 PNG：
      文件名: f"{prefix}_b{b}_h{h}_Q{q_len}_K{kv_len}.png"
    """
    os.makedirs(out_dir, exist_ok=True)
    B, H = inds_K.shape[:2]

    sel = selection_mask_from_inds(
        inds_K=inds_K, q_len=q_len, kv_len=kv_len, bm=bm, clamp_kv_len=clamp_kv_len
    )  # [B,H,q_len,kv_len], uint8

    # 如需下采样，用最近邻（保持二值视觉语义）
    if downsample_q > 1 or downsample_k > 1:
        # 转为 float 后加 channel 维做 interpolate，再阉回到 uint8 显示
        sel_f = sel.float().unsqueeze(2)  # [B,H,1,q,k]
        new_q = math.ceil(q_len / max(1, downsample_q))
        new_k = math.ceil(kv_len / max(1, downsample_k))
        sel_ds = F.interpolate(sel_f, size=(new_q, new_k), mode='nearest').squeeze(2)
        sel = sel_ds.clamp(0, 1).to(torch.uint8)
        q_vis, k_vis = new_q, new_k
        sink_x = int(round(sink_len / max(1, downsample_k))) if sink_len > 0 else -1
    else:
        q_vis, k_vis = q_len, kv_len
        sink_x = sink_len

    # 逐 (B,H) 存文件
    for b in range(B):
        for h in range(H):
            img = sel[b, h].cpu().numpy()  # [q_vis, k_vis], {0,1}

            fig, ax = plt.subplots(figsize=(max(6, k_vis/600*6), max(4, q_vis/600*4)))
            im = ax.imshow(img, interpolation='nearest', aspect='auto')  # 默认灰度 colormap
            ax.set_title(f"b{b}, h{h}  (Q={q_len}, K={kv_len}, bm={bm})")
            ax.set_xlabel("Keys (columns)")
            ax.set_ylabel("Queries (rows)")

            if sink_len > 0:
                ax.axvline(x=sink_x-0.5, linestyle='--')  # 画出 sink 分界

            fig.tight_layout()
            save_path = os.path.join(out_dir, f"{prefix}_b{b}_h{h}_Q{q_len}_K{kv_len}.png")
            fig.savefig(save_path, dpi=200)
            plt.close(fig)
            print(f'{save_path=}')


# ==============================
# Infinity sparse index mapping
# ==============================
def map_sparse_kv_inds(
    inds_S,      # [B,H,QG,Kkeep] @ scale S
    S, K,
    n_list,      # [n_0,...,n_K]
    cum_S, cum_K,
    hw_list=None,# None => 用 1D比例; 否则用 2D 网格
    sink_first_k: int = 5,    # 按你要求: 一共前5个KV
):
    # A) j, ΔS
    j = torch.bucketize(inds_S, cum_S[1:], right=True)      # each kv index is belong to which scale, e.g. index 0 in scale-0, index 1,2,3,4 in scale-1, ...
    deltaS = inds_S - cum_S[j]      # 每个 kv index 在 scale-S 的各尺度块的偏移量, e.g. index = 7, 在 scale 2 的块内偏移量 = 7 - 5 = 2

    # B) scale-K 的目标块 j'
    jp = (K - (S - j)).clamp_min(0)

    # C) ΔK
    if hw_list is None:
        n   = torch.as_tensor(n_list, device=inds_S.device)
        nj  = n[j]
        njp = n[jp]

        deltaK  = (((deltaS + 0.5) * njp) // nj).clamp_max(njp - 1)
    else:
        h = torch.as_tensor([h for h,_ in hw_list], device=inds_S.device)
        w = torch.as_tensor([w for _,w in hw_list], device=inds_S.device)
        hj,  wj  = h.index_select(0, j),  w.index_select(0, j)
        hjp, wjp = h.index_select(0, jp), w.index_select(0, jp)
        u, v  = deltaS // wj, deltaS % wj
        up    = (u * hjp) // hj
        vp    = (v * wjp) // wj
        deltaK    = (up * wjp + vp).clamp_max(hjp*wjp - 1)

    # D) 还原绝对索引
    inds_K = cum_K[jp] + deltaK

    # E) sink 保护：追加前5个KV索引并截断
    if sink_first_k and sink_first_k > 0:
        # B,H,QG,Kkeep = inds_K.shape
        B,H,QG,Kkeep = inds_S.shape
        sink = torch.arange(min(sink_first_k, int(cum_K[-1].item())), device=inds_K.device, dtype=inds_K.dtype)
        sink = sink.view(1,1,1,-1).expand(B,H,QG,-1)
        inds_K = torch.cat([sink, inds_K], dim=-1)[..., :Kkeep]

    return inds_K.to(torch.int32)  # [B,H,QG,Kkeep]


def _infinity_hw_cum(scale_schedule, upto_scale_ind: int, device: torch.device):
    """Per-scale (h, w) tensors and cumulative token offsets for the Infinity image
    KV layout (Concat of all historical scales). ``scale_schedule`` entries are
    (t, h, w) with t == 1 for image generation."""
    shapes = [tuple(int(x) for x in s) for s in scale_schedule[:upto_scale_ind + 1]]
    if any(s[0] != 1 for s in shapes):
        raise ValueError(
            f"Infinity image index mapping expects (1, h, w) scales, got {shapes}; "
            "use the InfinityStar mapper for video schedules."
        )
    hs = torch.tensor([s[1] for s in shapes], device=device, dtype=torch.long)
    ws = torch.tensor([s[2] for s in shapes], device=device, dtype=torch.long)
    cum = torch.zeros(len(shapes) + 1, device=device, dtype=torch.long)
    cum[1:] = torch.cumsum(hs * ws, dim=0)
    return hs, ws, cum


@torch.no_grad()
def map_infinity_sparse_kv_inds(
    inds_S: torch.Tensor,
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    current_k_len: int,
) -> torch.Tensor:
    """Map CS4A-selected image KV indices from decision scale ``S`` to target scale
    ``k`` by the paper's Decompose--Align--Project mechanism.

    Infinity KV layout: Concat(scale_0, ..., scale_k) — a flattened index j is
      Decompose : source scale l (via cumulative offsets) and offset δ -> (u, v)
      Align     : l' = k - (S - l)          # preserve relative scale distance
      Project   : center-aligned 2D rescale (same formula as the InfinityStar
                  video mapper, with t == 1), then re-linearize j' = C_{l'} + δ'.

    Target scales [0, k-S) receive no mapped index (their aligned source would be
    negative); the caller unions an attention-sink prefix to cover them.
    """
    S, K = int(decision_scale_ind), int(current_scale_ind)
    hs, ws, cum = _infinity_hw_cum(scale_schedule, K, inds_S.device)
    decision_k_len = int(cum[S + 1].item())

    j = inds_S.long().clamp(0, max(0, decision_k_len - 1))
    l = torch.bucketize(j, cum[1:S + 2], right=True)             # source scale of j, in [0, S]
    delta = j - cum[l]
    u = torch.div(delta, ws[l], rounding_mode='trunc')
    v = delta - u * ws[l]

    lp = l + (K - S)                                             # relative scale alignment, in [K-S, K]
    up = torch.div((u * 2 + 1) * hs[lp], 2 * hs[l], rounding_mode='trunc')
    vp = torch.div((v * 2 + 1) * ws[lp], 2 * ws[l], rounding_mode='trunc')
    up = torch.minimum(up, hs[lp] - 1)
    vp = torch.minimum(vp, ws[lp] - 1)

    jp = cum[lp] + up * ws[lp] + vp
    return jp.clamp_(0, max(0, int(current_k_len) - 1)).to(torch.int32)


# Cache of Infinity decision->current coordinate LUTs. Same rationale as the
# InfinityStar ``get_coord_lut``: the map is a pure function of the index VALUE and
# the (S, k, scale_schedule) geometry, so one gather through a [decision_k_len] LUT
# replaces the whole elementwise Decompose-Align-Project chain per call.
_infinity_coord_lut_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_INFINITY_COORD_LUT_CACHE_MAX = 8


@torch.no_grad()
def get_infinity_coord_lut(
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """``[decision_k_len]`` int32 LUT: decision KV index -> current KV index, built
    once per (S, k, schedule) geometry by running the exact
    ``map_infinity_sparse_kv_inds`` arithmetic over ``arange(decision_k_len)``."""
    dk = max(int(decision_k_len), 1)
    key = ('infinity', int(decision_scale_ind), int(current_scale_ind),
           tuple(tuple(int(x) for x in s) for s in scale_schedule[:current_scale_ind + 1]),
           int(current_k_len), dk, str(device))
    cached = _infinity_coord_lut_cache.get(key)
    if cached is not None:
        _infinity_coord_lut_cache.move_to_end(key)
        return cached

    x = torch.arange(dk, device=device, dtype=torch.int32).view(1, 1, 1, -1)
    lut = map_infinity_sparse_kv_inds(
        inds_S=x,
        decision_scale_ind=decision_scale_ind,
        current_scale_ind=current_scale_ind,
        scale_schedule=scale_schedule,
        current_k_len=current_k_len,
    ).view(-1)                                                   # [decision_k_len] int32

    if len(_infinity_coord_lut_cache) >= _INFINITY_COORD_LUT_CACHE_MAX:
        _infinity_coord_lut_cache.popitem(last=False)            # evict oldest
    _infinity_coord_lut_cache[key] = lut
    return lut


@torch.no_grad()
def map_infinity_sparse_kv_inds_lut(
    inds_S: torch.Tensor,
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    current_k_len: int,
    decision_k_len: int,
) -> torch.Tensor:
    """LUT-accelerated drop-in for ``map_infinity_sparse_kv_inds`` (bit-identical)."""
    lut = get_infinity_coord_lut(
        decision_scale_ind=decision_scale_ind,
        current_scale_ind=current_scale_ind,
        scale_schedule=scale_schedule,
        current_k_len=current_k_len,
        decision_k_len=decision_k_len,
        device=inds_S.device,
    )
    idx = inds_S.long().clamp_(0, lut.shape[0] - 1)              # decision indices ∈ [0, decision_k_len)
    return lut[idx]                                              # [B,H,Gk,count] int32, current coords


# Cache of per-scale image local-window masks (geometry only, shared across heads and
# layers). The image analog of the video ``_get_band_mask``: each query group's mask is
# the EXACT union of its member queries' win×win 2D windows inside the current scale's
# KV block (built via one max_pool2d dilation — no per-row loops, no contiguous-range
# approximation).
_image_local_mask_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_IMAGE_LOCAL_MASK_CACHE_MAX = 8


@torch.no_grad()
def _get_image_local_window_mask(current_scale_shape, current_block_start, current_k_len,
                                 bm, q_len, win_size, device) -> torch.Tensor:
    """[G, k_len] bool: per query group, the union of member queries' win×win local
    windows on the (h, w) grid, linearized and shifted by the current scale's KV block
    offset ``current_block_start`` (= C_k). Cached per geometry."""
    h, w = int(current_scale_shape[-2]), int(current_scale_shape[-1])
    win = int(win_size)
    assert win >= 1 and win % 2 == 1, f"image local window must be odd, got {win}"
    G = (int(q_len) + bm - 1) // bm
    k_len = int(current_k_len)
    block_start = int(current_block_start)
    key = (h, w, block_start, k_len, int(bm), int(q_len), win, str(device))
    cached = _image_local_mask_cache.get(key)
    if cached is not None:
        _image_local_mask_cache.move_to_end(key)
        return cached

    mask = torch.zeros(G, k_len, dtype=torch.bool, device=device)
    if h > 0 and w > 0 and G > 0 and int(q_len) > 0:
        rows = torch.arange(int(q_len), device=device)
        groups = torch.div(rows, bm, rounding_mode='trunc').clamp_(max=G - 1)
        occ = torch.zeros(G, h * w, device=device, dtype=torch.float32)
        occ[groups, rows.clamp(max=h * w - 1)] = 1.0             # image: q_len == h*w
        if win > 1:
            occ = F.max_pool2d(occ.view(G, 1, h, w), kernel_size=win, stride=1,
                               padding=win // 2).view(G, h * w)
        local = occ > 0
        end = min(block_start + h * w, k_len)
        if end > block_start:
            mask[:, block_start:end] = local[:, :end - block_start]

    if len(_image_local_mask_cache) >= _IMAGE_LOCAL_MASK_CACHE_MAX:
        _image_local_mask_cache.popitem(last=False)              # evict oldest
    _image_local_mask_cache[key] = mask
    return mask


# Cache of per-(schedule, K) image cross-scale band masks at TOKEN granularity — the
# footprint-path locality prior. Multi-scale generalization of
# ``_get_image_local_window_mask`` (which is the l == K special case): at sparse scale
# K every query keeps, for each historical scale l with win[l] >= 1, the
# win[l] x win[l] window around its center-aligned position on scale l. Same win_size
# semantics as the conference CSLA mask_mod and ``dcsla_image_win_size``. Geometry
# only, shared across layers / heads / images.
_image_band_token_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_IMAGE_BAND_TOKEN_CACHE_MAX = 8


@torch.no_grad()
def _get_image_cross_scale_band_mask_tokens(scale_schedule, current_scale_ind,
                                            current_k_len, bm, q_len, win_size,
                                            device) -> torch.Tensor:
    """[G, k_len] bool: per query group, the union of member queries' cross-scale
    band windows over all scales l <= K. ``win_size`` is indexed by the HISTORICAL
    scale l (win < 1 disables scale l, a short list broadcasts its last entry) and
    the projection is ``(round(q_y / h_K * h_l), round(q_x / w_K * w_l))`` — the
    same fp32 formula as ``dcsla_utils.build_image_cross_scale_band_mask``, at
    token instead of block granularity. Cached per geometry."""
    K = int(current_scale_ind)
    hs = [int(s[-2]) for s in scale_schedule[:K + 1]]
    ws = [int(s[-1]) for s in scale_schedule[:K + 1]]
    wins = [int(win_size[l]) if l < len(win_size) else int(win_size[-1]) for l in range(K + 1)]
    G = (int(q_len) + bm - 1) // bm
    k_len = int(current_k_len)
    key = (tuple(hs), tuple(ws), tuple(wins), int(bm), int(q_len), k_len, str(device))
    cached = _image_band_token_cache.get(key)
    if cached is not None:
        _image_band_token_cache.move_to_end(key)
        return cached

    n_per = [h * w for h, w in zip(hs, ws)]
    assert sum(n_per) == k_len, \
        f'image band tokens: schedule tokens {sum(n_per)} != k_len {k_len} at scale {K}'
    assert hs[K] * ws[K] == int(q_len), \
        f'image band tokens: current scale {hs[K]}x{ws[K]} != q_len {q_len}'

    mask = torch.zeros(G, k_len, dtype=torch.bool, device=device)
    rows = torch.arange(int(q_len), device=device)
    groups = torch.div(rows, bm, rounding_mode='trunc').clamp_(max=G - 1)
    qy = torch.div(rows, ws[K], rounding_mode='trunc').to(torch.float32)
    qx = (rows % ws[K]).to(torch.float32)
    H_q, W_q = float(hs[K]), float(ws[K])
    start = 0
    for l in range(K + 1):
        h_l, w_l, win = hs[l], ws[l], wins[l]
        if win >= 1:
            half = win // 2
            cy = torch.round(qy / H_q * float(h_l)).long()
            cx = torch.round(qx / W_q * float(w_l)).long()
            # +1 padded grid: round() can land on h_l / w_l at a scale boundary; the
            # crop below keeps |c - kv| <= half exact, whereas clamping the center
            # would leak a wrong row/col when half is small.
            occ = torch.zeros(G, (h_l + 1) * (w_l + 1), device=device, dtype=torch.float32)
            occ[groups, cy * (w_l + 1) + cx] = 1.0
            occ = occ.view(G, h_l + 1, w_l + 1)
            if half > 0:
                # max_pool2d dilation == per-center |dy|<=half & |dx|<=half union
                occ = F.max_pool2d(occ.unsqueeze(1), kernel_size=2 * half + 1,
                                   stride=1, padding=half).squeeze(1)
            mask[:, start:start + h_l * w_l] = \
                (occ[:, :h_l, :w_l] > 0).reshape(G, h_l * w_l)
        start += h_l * w_l

    if len(_image_band_token_cache) >= _IMAGE_BAND_TOKEN_CACHE_MAX:
        _image_band_token_cache.popitem(last=False)              # evict oldest
    _image_band_token_cache[key] = mask
    return mask


@torch.no_grad()
def build_image_local_union_indices(
    inds: torch.Tensor,                       # [B, H, G, base_count] mapped top-k centers (current KV coords)
    sink_len: int,                            # first columns force-selected (tokens of the first sink scales)
    current_k_len: int,
    current_scale_shape,                      # (1, h, w) or (h, w)
    current_block_start: int,                 # C_k: current scale's offset on the KV axis
    bm: int,
    q_len: int,
    win_size: int,
    multiple_of: int,
    indices_pad_to: int,
):
    """Image (Infinity) counterpart of ``build_frame_band_union_indices``:

        union_mask   = sink prefix | current-scale local window mask | mapped top-k centers
        inds, counts = mask_to_indices(union_mask)     # variable per-group counts

    Realizes the paper's inds^(k) = A_sink ∪ M_{S->k}(inds^(S)) plus the local prior,
    replacing the old fixed-budget concat (which approximated each group's window union
    by a contiguous KV range and could drop tail top-k indices to fit the budget)."""
    B, H, G, _ = inds.shape
    k_len = int(current_k_len)
    local_mask = _get_image_local_window_mask(
        current_scale_shape, current_block_start, k_len, bm, q_len,
        win_size, inds.device)                                                  # [G, k_len]
    union = local_mask.view(1, 1, G, k_len).expand(B, H, G, k_len).clone()      # [B,H,G,k_len] bool
    sink_len = max(0, min(int(sink_len), k_len))
    if sink_len > 0:
        union[..., :sink_len] = True
    union.scatter_(-1, inds.to(torch.int64).clamp_(0, k_len - 1), True)         # add mapped centers
    out_inds, counts = _mask_to_indices(union, multiple_of, indices_pad_to)
    out_inds = out_inds[:, :, :, :k_len]
    return out_inds, counts


# Cache of Infinity current->decision PARENT LUTs (image footprint mode). Inverse of
# ``map_infinity_sparse_kv_inds``: every current KV index inherits the selected-ness of
# the decision cell whose up-sampling footprint contains it, so a selected decision cell
# expands to its WHOLE footprint block at the target scale (no center-only sampling, no
# budget fill). Same LRU pattern as the video ``get_parent_lut``.
_infinity_parent_lut_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_INFINITY_PARENT_LUT_CACHE_MAX = 8


@torch.no_grad()
def get_infinity_parent_lut(
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """``[current_k_len]`` int64 LUT: current (target-scale k) KV index -> its parent
    decision-scale KV index. Piecewise, mirroring ``map_infinity_sparse_kv_inds``
    INVERTED over the stacked-pyramid image KV layout:

      - parentless prefix ``[0, C_{k-S})`` (target scales shifted past the earliest
        source scale): IDENTITY — reuse the decision indices verbatim, exactly like the
        video parent LUT's ctx-prefix identity. These indices are always < decision_k_len
        (C_{k-S} <= C_{S+1}), and with ``cs4a_sink_scales >= k - S`` the sink union
        force-selects this whole range anyway.
      - every other target block l' >= k-S: inverse Relative Alignment l = l' - (k-S),
        then inverse center-aligned footprint per axis, u = ((2u'+1)*h_l) // (2*h_{l'}).
    """
    S, K = int(decision_scale_ind), int(current_scale_ind)
    ck = max(int(current_k_len), 1)
    dk = max(int(decision_k_len), 1)
    key = ('infinity', S, K,
           tuple(tuple(int(x) for x in s) for s in scale_schedule[:K + 1]),
           ck, dk, str(device))
    cached = _infinity_parent_lut_cache.get(key)
    if cached is not None:
        _infinity_parent_lut_cache.move_to_end(key)
        return cached

    hs, ws, cum = _infinity_hw_cum(scale_schedule, K, device)
    shift = K - S

    c = torch.arange(ck, device=device, dtype=torch.long)
    lp = torch.bucketize(c, cum[1:K + 2], right=True)            # target scale of c, in [0, K]
    delta = c - cum[lp]
    up = torch.div(delta, ws[lp], rounding_mode='trunc')
    vp = delta - up * ws[lp]

    l = (lp - shift).clamp_min(0)                                # inverse relative alignment
    u = torch.div((up * 2 + 1) * hs[l], 2 * hs[lp], rounding_mode='trunc')
    v = torch.div((vp * 2 + 1) * ws[l], 2 * ws[lp], rounding_mode='trunc')
    u = torch.minimum(u, hs[l] - 1)
    v = torch.minimum(v, ws[l] - 1)
    parent = cum[l] + u * ws[l] + v

    # parentless prefix -> identity (decision indices reused verbatim)
    parent = torch.where(lp < shift, c, parent)
    parent = parent.clamp_(0, dk - 1)

    if len(_infinity_parent_lut_cache) >= _INFINITY_PARENT_LUT_CACHE_MAX:
        _infinity_parent_lut_cache.popitem(last=False)           # evict oldest
    _infinity_parent_lut_cache[key] = parent
    return parent


@torch.no_grad()
def build_infinity_footprint_union_indices(
    decision_mask: torch.Tensor,                # [B, H, Gs, decision_k_len] bool decision selection
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    decision_k_len: int,
    current_k_len: int,
    current_q_len: int,
    sink_len: int,                              # tokens of the first cs4a_sink_scales scales, force-unioned
    bm: int,
    multiple_of: int,
    indices_pad_to: int,
    union_local: bool = False,
    current_block_start: int = 0,               # unused (band offsets derive from scale_schedule; kept for API stability)
    win_size=1,                                 # list: per-HISTORICAL-scale band widths (win<1 disables, short list broadcasts last entry); legacy int: self-only window on the current scale. Only read when union_local.
):
    """Footprint expansion for the Infinity image CS4A path (inverse parent-LUT mask
    gather, NO sort) — the image counterpart of ``build_footprint_union_indices``:

        gS_idx       = center-NN q-group remap (Gs -> Gk)
        current_mask = decision_mask[:, :, gS_idx[:, None], parent_lut[None, :]]
        current_mask |= sink prefix   (and optionally the cross-scale band prior)
        inds, counts = mask_to_indices(current_mask)     # variable per-group counts

    Relative alignment is built into the parent LUT, so this path does not consult
    ``cs4a_image_kv_map``. Returns ``(inds [B,H,Gk,k_len], counts [B,H,Gk])``.
    """
    B, H, Gs, _ = decision_mask.shape
    device = decision_mask.device
    k_len = int(current_k_len)
    Gk = (int(current_q_len) + bm - 1) // bm

    # 1) q-group remap Gs -> Gk (bit-identical formula to map_inds_counts_across_scales)
    gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5
    gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)   # [Gk]

    # 2) key-dim footprint map: current key selected iff its parent decision cell was.
    parent_lut = get_infinity_parent_lut(
        decision_scale_ind=decision_scale_ind,
        current_scale_ind=current_scale_ind,
        scale_schedule=scale_schedule,
        current_k_len=k_len,
        decision_k_len=decision_k_len,
        device=device,
    )
    # Fused q-group remap + footprint gather in ONE advanced-indexing kernel (returns a
    # fresh tensor, safe to mutate below).
    current_mask = decision_mask[:, :, gS_idx[:, None], parent_lut[None, :]]             # [B,H,Gk,k_len] bool

    # 3) sink union (image analog of the video prefix identity guarantee)
    sink_len = max(0, min(int(sink_len), k_len))
    if sink_len > 0:
        current_mask[..., :sink_len] = True

    # 4) optional locality-prior union (A/B knob). A win_size LIST follows the
    #    conference win_size / dcsla_image_win_size semantics: at scale K, union the
    #    win[l] x win[l] band on every scale l <= K with win[l] >= 1 (cross-scale +
    #    current-scale). A legacy int keeps the old self-only current-scale window.
    if union_local:
        if isinstance(win_size, (list, tuple)):
            wins = [int(x) for x in win_size]
        else:
            wins = [-1] * int(current_scale_ind) + [int(win_size)]
        eff = [wins[l] if l < len(wins) else wins[-1]
               for l in range(int(current_scale_ind) + 1)] if wins else []
        if any(w >= 1 for w in eff):
            band_mask = _get_image_cross_scale_band_mask_tokens(
                scale_schedule, current_scale_ind, k_len, bm, int(current_q_len),
                wins, device)                                                        # [Gk, k_len]
            # In-place union (same as the video band union): band_mask is LRU-cached
            # and only READ here — never flip the direction, that would poison the cache.
            current_mask |= band_mask.view(1, 1, Gk, k_len)

    # 5) mask -> (inds, variable counts) consumed directly by csp_attn.
    out_inds, counts = _mask_to_indices(current_mask, multiple_of, indices_pad_to)
    out_inds = out_inds[:, :, :, :k_len]
    return out_inds, counts


def _as_hw_shape(scale_schedule, scale_ind: int):
    """(h, w) of a scale entry; accepts (t, h, w) or (h, w) tuples."""
    s = scale_schedule[scale_ind]
    return (int(s[-2]), int(s[-1]))


# ==================================
# InfinityStar sparse index mapping
# ==================================

# ---------------
# 1d / 2d expend
# ---------------
def _make_1d_local_offsets(num_offsets: int, device: torch.device) -> torch.Tensor:
    offsets = [0]
    radius = 1
    while len(offsets) < num_offsets:
        offsets.extend([-radius, radius])
        radius += 1
    return torch.tensor(offsets[:num_offsets], device=device, dtype=torch.int64)


def _make_3d_local_offsets(num_offsets: int, device: torch.device) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    offsets = [(0, 0, 0)]
    radius = 1
    while len(offsets) < num_offsets:
        axis_offsets = [
            (0, 0, -radius), (0, 0, radius),
            (0, -radius, 0), (0, radius, 0),
            (-radius, 0, 0), (radius, 0, 0),
        ]
        offsets.extend(axis_offsets)
        for dt in range(-radius, radius + 1):
            for dh in range(-radius, radius + 1):
                for dw in range(-radius, radius + 1):
                    if (dt, dh, dw) in axis_offsets or (dt, dh, dw) == (0, 0, 0):
                        continue
                    if abs(dt) + abs(dh) + abs(dw) == radius:
                        offsets.append((dt, dh, dw))
        radius += 1

    offsets = offsets[:num_offsets]
    dt = torch.tensor([item[0] for item in offsets], device=device, dtype=torch.int64)
    dh = torch.tensor([item[1] for item in offsets], device=device, dtype=torch.int64)
    dw = torch.tensor([item[2] for item in offsets], device=device, dtype=torch.int64)
    return dt, dh, dw


def _select_unique_by_priority(candidates: torch.Tensor, target_count: int, kv_len: int) -> torch.Tensor:
    """Per query group, pick exactly ``target_count`` UNIQUE keys from ``candidates``
    by priority (earlier position == higher priority); any shortfall (a group with
    fewer than ``target_count`` unique valid candidates) is filled with random
    unselected keys. Counts are uniform (= ``target_count``), as csp_attn requires.

    Batched over heads — loops only over B (not B*H) — so the per-(B,H) Python loop
    that dominated sparse-scale latency is gone.
    """
    B, H, G, M = candidates.shape
    target_count = min(target_count, kv_len, M)
    out = torch.empty(B, H, G, target_count, device=candidates.device, dtype=torch.int32)
    pos = torch.arange(M, device=candidates.device, dtype=torch.int64).view(1, 1, M)
    neg1 = pos.new_full((1, 1, M), -1)
    max_key = kv_len * (M + 1) + M

    for b in range(B):
        chunk = candidates[b]                                          # [H, G, M]
        valid = (chunk >= 0) & (chunk < kv_len)
        sort_key = torch.where(
            valid,
            chunk.to(torch.int64) * (M + 1) + pos,
            chunk.new_full((H, G, M), max_key, dtype=torch.int64),
        )
        sorted_key, order = torch.sort(sort_key, dim=-1)
        sorted_val = torch.div(sorted_key, M + 1, rounding_mode='trunc')
        is_first = sorted_key < max_key
        is_first[..., 1:] &= sorted_val[..., 1:] != sorted_val[..., :-1]

        unique_mask = torch.zeros((H, G, M), device=candidates.device, dtype=torch.bool)
        unique_mask.scatter_(dim=-1, index=order, src=is_first)
        priority = torch.where(unique_mask, M - pos, neg1)
        selected_priority, selected_pos = torch.topk(priority, k=target_count, dim=-1, largest=True)
        selected = torch.gather(chunk, dim=-1, index=selected_pos.to(torch.int64)).to(torch.int32)

        # Shortfall slots (selected_priority < 0) -> random unselected keys.
        rand_fill = torch.randint(0, max(1, kv_len), (H, G, target_count),
                                  device=candidates.device, dtype=torch.int32)
        out[b].copy_(torch.where(selected_priority < 0, rand_fill, selected))

    return out


def expand_mapped_indices_to_budget(
    inds: torch.Tensor,
    target_count: int,
    current_sink_len: int,
    current_k_len: int,
    current_scale_shape: Tuple[int, int, int],
    expand_mode: str,
) -> torch.Tensor:
    if target_count <= 0:
        return inds[..., :0].to(torch.int32)

    B, H, G, base_count = inds.shape
    expand_count = max(1, math.ceil(target_count / max(base_count, 1)))
    sink_mask = inds < current_sink_len
    expand_mode = str(expand_mode).lower()

    if expand_mode == '3d' and current_scale_shape is not None:
        ct, ch, cw = current_scale_shape
        video_len = ct * ch * cw
        offsets_t, offsets_h, offsets_w = _make_3d_local_offsets(expand_count, inds.device)
        video_delta = (inds - current_sink_len).clamp(0, max(0, video_len - 1)).to(torch.int64)
        src_t = torch.div(video_delta, ch * cw, rounding_mode='trunc')
        src_hw = video_delta - src_t * ch * cw
        src_h = torch.div(src_hw, cw, rounding_mode='trunc')
        src_w = src_hw - src_h * cw

        cand_t = (src_t.unsqueeze(-1) + offsets_t.view(1, 1, 1, 1, -1)).clamp(0, ct - 1)
        cand_h = (src_h.unsqueeze(-1) + offsets_h.view(1, 1, 1, 1, -1)).clamp(0, ch - 1)
        cand_w = (src_w.unsqueeze(-1) + offsets_w.view(1, 1, 1, 1, -1)).clamp(0, cw - 1)
        candidates = current_sink_len + (cand_t * ch + cand_h) * cw + cand_w
        candidates = torch.where(sink_mask.unsqueeze(-1), inds.unsqueeze(-1).to(torch.int64), candidates)
    else:
        offsets = _make_1d_local_offsets(expand_count, inds.device)
        candidates = inds.unsqueeze(-1).to(torch.int64) + offsets.view(1, 1, 1, 1, -1)
        video_candidates = candidates.clamp(current_sink_len, max(current_sink_len, current_k_len - 1))
        candidates = torch.where(sink_mask.unsqueeze(-1), inds.unsqueeze(-1).to(torch.int64), video_candidates)

    # Priority is: all mapped topk centers first, then radius-1 expansions, etc.
    candidates = candidates.permute(0, 1, 2, 4, 3).reshape(B, H, G, -1).to(torch.int32)
    return _select_unique_by_priority(candidates, target_count, current_k_len)


# @torch.no_grad()
# def map_inds_counts_across_scales(
#     inds_S: torch.Tensor,               # [B, H, G_S, K_S]      decision scale `S` topk KV indices（已对齐）
#     counts_S: torch.Tensor,             # [B, H, G_S]           decision scale `S` per-q-group count（通常常数&对齐）
#     cur_q_len: int,                     # current target scale `K` Q length (to determine G_K)
#     cur_k_len: int,                     # current target scale `K` KV length
#     bm: int = 192,                      # query-group size {64, 192}
#     indices_pad_to: int = 1,            # {triton: 1, cuda: 4}
#     add_sink: bool = False,             # if add attention sink KV indices
#     sink_len: int = 0,                  # sink [0:sink_len)
#     sink_quota_frac: float = 0.2,       # sink 配额在 indices_count 中占比（仅当 add_sink=True 生效）
#     device: torch.device = None,
# ):
#     """
#     将 (inds_S, counts_S) 从尺度S映射到cur scale，返回:
#       inds_K   [B, H, Gk, Kk]
#       counts_K [B, H, Gk]      （各组均为同一个 indices_count，且是 indices_pad_to 的倍数）

#     设计要点：
#     1) 组映射：Gk 组 -> Gs 组，使用中心点最近邻 (u=(g+0.5)/G * Gs - 0.5) 舍入并 clamp。
#     2) 若 add_sink=True：优先写入长度为 sink_quota 的 [0:sink_len)，剩余配额由模板 inds_S 填充（去除与 sink 的重叠）。
#     3) 模板填充：用掩码 templ>=sink_len，配合 "很大值屏蔽 + topk(取最小)"，一次性选出需要的列；若仍不足则循环补齐。
#     4) 保证所有索引在 [0, K_hist_len)。
#     """
#     assert inds_S.ndim == 4 and counts_S.ndim == 3, f"{inds_S.shape=}, {counts_S.shape=}"
#     B, H, Gs, Ks = inds_S.shape
#     if device is None:
#         device = inds_S.device
    
#     # current target scale `K` query-group number
#     Gk = (cur_q_len + bm - 1) // bm

#     # indices_count（对齐为 indices_pad_to 的倍数），假设各组相同（Chipmunk约定）
#     # 取 (0,0,0) 位置；如工程里确实 per-(B,H) 不同，可改成 per-(B,H) 广播。
#     raw_cnt = int(counts_S[0, 0, 0].item())
#     indices_count = max(indices_pad_to, (raw_cnt + indices_pad_to - 1) // indices_pad_to * indices_pad_to)

#     # query-group mapping: which group in Gk corresponds to each group in Gs (1D index), and then perform index_select on dim=2
#     gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5
#     gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)  # [Gk]
#     templ = inds_S.index_select(dim=2, index=gS_idx)                                    # [B, H, Gk, **Ks**]

#     # init output
#     inds_K = torch.empty(B, H, Gk, indices_count, dtype=torch.int32, device=device)
#     counts_K = torch.full((B, H, Gk), indices_count, dtype=torch.int32, device=device)

#     if not add_sink or sink_len <= 0 or indices_count == 0:
#         # 直接裁切/循环补齐模板到 indices_count
#         if Ks >= indices_count:
#             inds_K.copy_(templ[..., :indices_count])
#         else:
#             rep = math.ceil(indices_count / Ks)
#             inds_K.copy_(templ.repeat_interleave(repeats=rep, dim=-1)[..., :indices_count])
#         # boundary protection
#         inds_K.clamp_(0, max(0, cur_k_len - 1))
#         return inds_K, counts_K

#     # --- 带 sink 的路径 ---
#     # sink 的配额（逐 (B,H,G) 相同），注意上界
#     sink_quota = min(int(round(indices_count * sink_quota_frac)), indices_count, sink_len)
#     templ_quota = indices_count - sink_quota

#     # 1) 写入 sink 前缀：广播写入 [0:sink_quota]
#     if sink_quota > 0:
#         sink = torch.arange(sink_len, device=device, dtype=torch.int32)
#         if sink_len >= sink_quota:
#             sink_block = sink[:sink_quota]  # [sink_quota]
#         else:
#             rep = math.ceil(sink_quota / sink_len)
#             sink_block = sink.repeat(rep)[:sink_quota]
#         inds_K[..., :sink_quota] = sink_block.view(1, 1, 1, -1)

#     if templ_quota == 0:
#         inds_K.clamp_(0, max(0, cur_k_len - 1))
#         return inds_K, counts_K

#     # 2) 从模板中选出不与 sink 冲突的列，数量为 templ_quota：
#     #    向量化：把 < sink_len 的位置屏蔽成“大值”，然后用 topk(取最小)一次性选出 templ_quota 个值。
#     if sink_len > 0:
#         big_val = torch.iinfo(torch.int32).max  # 用 int32 最大值作屏蔽
#         templ_masked = torch.where(templ >= sink_len, templ, templ.new_full(templ.shape, big_val))
#     else:
#         templ_masked = templ

#     Ksel = templ_masked.shape[-1]
#     if templ_quota <= Ksel:
#         # topk 取“最小”的 templ_quota 个值：对负号后取 largest=True
#         neg_vals = -templ_masked.to(torch.int64)  # 防止溢出
#         topk_pos = torch.topk(neg_vals, k=templ_quota, dim=-1, largest=True).indices  # [B,H,Gk,templ_quota]
#         templ_pick = torch.gather(templ_masked, dim=-1, index=topk_pos).to(torch.int32)  # [B,H,Gk,templ_quota]
#     else:
#         # 先取全部合法项，再循环补齐
#         neg_vals = -templ_masked.to(torch.int64)
#         topk_pos = torch.topk(neg_vals, k=Ksel, dim=-1, largest=True).indices
#         base_pick = torch.gather(templ_masked, dim=-1, index=topk_pos).to(torch.int32)   # [B,H,Gk,Ksel]
#         rep = math.ceil(templ_quota / Ksel)
#         templ_pick = base_pick.repeat_interleave(repeats=rep, dim=-1)[..., :templ_quota] # [B,H,Gk,templ_quota]

#     # 3) 写入模板段
#     inds_K[..., sink_quota:sink_quota + templ_quota] = templ_pick

#     # 4) 边界保护
#     inds_K.clamp_(0, max(0, cur_k_len - 1))
#     return inds_K, counts_K


# @torch.no_grad()
# def map_inds_counts_across_scales(
#     inds_S: torch.Tensor,               # [B, H, G_S, K_S]      decision scale `S` topk KV indices
#     counts_S: torch.Tensor,             # [B, H, G_S]           decision scale `S` per-q-group count
#     cur_q_len: int,                     # current target scale `K` Q length (to determine G_K)
#     cur_k_len: int,                     # current target scale `K` KV length
#     bm: int = 192,                      # query-group size {192}
#     indices_pad_to: int = 4,            # {triton: 1, cuda: 4}
#     device: torch.device = None,
# ):
#     """
#     Map `inds_S` and `counts_S` from decision scale `S` to current scale `K`
#     Return:
#         inds_K   [B, H, Gk, Ks]
#         counts_K [B, H, Gk]
#     """
#     assert inds_S.ndim == 4 and counts_S.ndim == 3, f"{inds_S.shape=}, {counts_S.shape=}"
#     B, H, Gs, Ks = inds_S.shape
#     if device is None:
#         device = inds_S.device

#     # current target scale `K` query-group number
#     Gk = (cur_q_len + bm - 1) // bm

#     raw_cnt = int(counts_S[0, 0, 0].item())
#     # raw_cnt = int(counts_S.max().item())
#     indices_count = max(indices_pad_to, (raw_cnt + indices_pad_to - 1) // indices_pad_to * indices_pad_to)
#     # print(f'        [{device=}] {raw_cnt=}, {indices_count=}')

#     # query-group mapping: which group in Gk corresponds to each group in Gs (1D index), and then perform index_select on dim=2
#         # round using center nearest neighbor ($u=(g+0.5)/G \times Gs - 0.5$) and perform clamp operation
#     gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5             # [0.5, 1.5, ..., Gk-0.5]
#     gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)  # [Gk]
#     templ = inds_S.index_select(dim=2, index=gS_idx)                                    # [B, H, Gk, **Ks**]

#     # init output
#     inds_K = torch.empty(B, H, Gk, indices_count, dtype=torch.int32, device=device)
#     counts_K = torch.full((B, H, Gk), indices_count, dtype=torch.int32, device=device)

#     # Complete to the indices_count template
#     inds_K.copy_(templ[..., :indices_count])
#     # boundary protection
#     inds_K.clamp_(0, max(0, cur_k_len - 1))

#     return inds_K, counts_K


@torch.no_grad()
def map_inds_counts_across_scales(
    inds_S: torch.Tensor,               # [B, H, G_S, K_S]      decision scale `S` topk KV indices
    counts_S: torch.Tensor,             # [B, H, G_S]           decision scale `S` per-q-group count
    cur_q_len: int,                     # current target scale `K` Q length (to determine G_K)
    cur_k_len: int,                     # current target scale `K` KV length
    bm: int = 192,                      # query-group size {192}
    indices_pad_to: int = 4,            # {triton: 1, cuda: 4}
    device: torch.device = None,
    qgroup_map_mode: Literal["nearest", "index_linear"] = "nearest",
    qgroup_interp_align: Literal["center", "endpoint"] = "center",
    sink_len: int = 0,
    source_k_len: Optional[int] = None,
    #* [Index Mapping Exps] Optional host-known count avoids a profiling-time GPU synchronization.
    source_indices_count: Optional[int] = None,
):
    """
    Map `inds_S` and `counts_S` from decision scale `S` to current scale `K`
    Return:
        inds_K   [B, H, Gk, Ks]
        counts_K [B, H, Gk]
    """
    assert inds_S.ndim == 4 and counts_S.ndim == 3, f"{inds_S.shape=}, {counts_S.shape=}"
    B, H, Gs, Ks = inds_S.shape
    if device is None:
        device = inds_S.device
    if source_k_len is None:
        source_k_len = cur_k_len
    source_k_len = max(int(source_k_len), 1)
    sink_len = max(0, min(int(sink_len), source_k_len))

    # current target scale `K` query-group number
    Gk = (cur_q_len + bm - 1) // bm

    #* [Index Mapping Exps] Prefer the host-known top-k count over counts_S.item().
    raw_cnt = (
        int(source_indices_count)
        if source_indices_count is not None
        else int(counts_S[0, 0, 0].item())
    )
    # raw_cnt = int(counts_S.max().item())
    indices_count = max(indices_pad_to, (raw_cnt + indices_pad_to - 1) // indices_pad_to * indices_pad_to)
    # print(f'        [{device=}] {raw_cnt=}, {indices_count=}')

    # init output
    inds_K = torch.empty(B, H, Gk, indices_count, dtype=torch.int32, device=device)
    counts_K = torch.full((B, H, Gk), indices_count, dtype=torch.int32, device=device)

    qgroup_map_mode = str(qgroup_map_mode).lower()
    if qgroup_map_mode == "index_linear":
        #* [Index Mapping Exps] Skip min/max GPU synchronizations when the count is host-known.
        if source_indices_count is None:
            count_min = int(counts_S.min().item())
            count_max = int(counts_S.max().item())
        else:
            count_min = count_max = raw_cnt
        if count_min != count_max:
            raise ValueError(
                "index_linear q-group mapping requires uniform per-group counts, "
                f"but got min={count_min}, max={count_max}"
            )
        if indices_count > Ks:
            raise ValueError(
                f"index_linear q-group mapping needs indices_count <= Ks, "
                f"but got {indices_count=} and {Ks=}"
            )

        target_pos = torch.arange(Gk, device=device, dtype=torch.float32)
        source_pos = torch.arange(Gs, device=device, dtype=torch.float32)
        qgroup_interp_align = str(qgroup_interp_align).lower()
        if qgroup_interp_align == "center":
            source_anchors = (source_pos + 0.5) * (Gk / max(Gs, 1)) - 0.5
        elif qgroup_interp_align == "endpoint":
            source_anchors = source_pos * ((Gk - 1) / max(Gs - 1, 1))
        else:
            raise ValueError(f"Unsupported qgroup_interp_align: {qgroup_interp_align}")

        source_anchors = source_anchors.clamp_(0, max(0, Gk - 1))
        g1 = torch.searchsorted(source_anchors, target_pos, right=False).clamp_(0, Gs - 1)
        g0 = (g1 - 1).clamp_(0, Gs - 1)
        anchor0 = source_anchors.index_select(0, g0)
        anchor1 = source_anchors.index_select(0, g1)
        alpha = ((target_pos - anchor0) / (anchor1 - anchor0).clamp_min(1e-6)).clamp_(0.0, 1.0)

        selected = inds_S[..., :indices_count].to(torch.int64).clamp_(0, source_k_len - 1)
        sentinel = source_k_len
        sink_mask = selected < sink_len if sink_len > 0 else torch.zeros_like(selected, dtype=torch.bool)
        video_mask = ~sink_mask

        sink_counts = sink_mask.sum(dim=-1).to(torch.int64)                              # [B,H,Gs]
        video_counts = (indices_count - sink_counts).to(torch.int64)                     # [B,H,Gs]
        invalid = selected.new_full(selected.shape, sentinel)
        sink_sorted = torch.sort(torch.where(sink_mask, selected, invalid), dim=-1).values
        video_sorted = torch.sort(torch.where(video_mask, selected, invalid), dim=-1).values

        alpha_view = alpha.view(1, 1, Gk, 1)

        def _interp_partition(sorted_vals, part_counts, target_counts):
            left_vals = sorted_vals.index_select(dim=2, index=g0)
            right_vals = sorted_vals.index_select(dim=2, index=g1)
            left_counts = part_counts.index_select(dim=2, index=g0).unsqueeze(-1)
            right_counts = part_counts.index_select(dim=2, index=g1).unsqueeze(-1)
            target_counts = target_counts.unsqueeze(-1)

            rank = torch.arange(indices_count, device=device, dtype=torch.int64).view(1, 1, 1, -1)
            safe_target = target_counts.clamp_min(1)
            left_rank = torch.floor((rank.to(torch.float32) + 0.5) * left_counts.to(torch.float32) / safe_target.to(torch.float32)).to(torch.int64)
            right_rank = torch.floor((rank.to(torch.float32) + 0.5) * right_counts.to(torch.float32) / safe_target.to(torch.float32)).to(torch.int64)
            left_rank = torch.minimum(left_rank, (left_counts - 1).clamp_min(0)).clamp_(0, indices_count - 1)
            right_rank = torch.minimum(right_rank, (right_counts - 1).clamp_min(0)).clamp_(0, indices_count - 1)

            left_pick = torch.gather(left_vals, dim=-1, index=left_rank)
            right_pick = torch.gather(right_vals, dim=-1, index=right_rank)
            has_left = left_counts > 0
            has_right = right_counts > 0
            interp = torch.floor(left_pick.to(torch.float32) * (1.0 - alpha_view) + right_pick.to(torch.float32) * alpha_view + 0.5).to(torch.int64)
            picked = torch.where(has_left & has_right, interp, torch.where(has_left, left_pick, right_pick))
            return torch.where(rank < target_counts, picked, picked.new_full(picked.shape, sentinel))

        left_sink_counts = sink_counts.index_select(dim=2, index=g0)
        right_sink_counts = sink_counts.index_select(dim=2, index=g1)
        target_sink_counts = torch.floor(
            left_sink_counts.to(torch.float32) * (1.0 - alpha.view(1, 1, Gk))
            + right_sink_counts.to(torch.float32) * alpha.view(1, 1, Gk)
            + 0.5
        ).to(torch.int64).clamp_(0, indices_count)
        target_video_counts = indices_count - target_sink_counts

        sink_interp = _interp_partition(sink_sorted, sink_counts, target_sink_counts)
        video_interp = _interp_partition(video_sorted, video_counts, target_video_counts)

        out_rank = torch.arange(indices_count, device=device, dtype=torch.int64).view(1, 1, 1, -1)
        target_sink_view = target_sink_counts.unsqueeze(-1)
        video_rank = (out_rank - target_sink_view).clamp_(0, indices_count - 1)
        video_pick = torch.gather(video_interp, dim=-1, index=video_rank)
        inds_K.copy_(torch.where(out_rank < target_sink_view, sink_interp, video_pick).to(torch.int32))
        inds_K.clamp_(0, source_k_len - 1)

        return inds_K, counts_K

    if qgroup_map_mode != "nearest":
        raise ValueError(f"Unsupported qgroup_map_mode: {qgroup_map_mode}")

    # query-group mapping: which group in Gk corresponds to each group in Gs (1D index), and then perform index_select on dim=2
        # round using center nearest neighbor ($u=(g+0.5)/G \times Gs - 0.5$) and perform clamp operation
    gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5             # [0.5, 1.5, ..., Gk-0.5]
    gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)  # [Gk]
    templ = inds_S.index_select(dim=2, index=gS_idx)                                    # [B, H, Gk, **Ks**]

    # Complete to the indices_count template
    inds_K.copy_(templ[..., :indices_count])
    # boundary protection
    inds_K.clamp_(0, max(0, cur_k_len - 1))

    return inds_K, counts_K


def map_infinitystar_sparse_kv_inds(
    inds_S: torch.Tensor,
    decision_sink_len: int,
    current_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    current_scale_shape: Tuple[int, int, int],
    current_k_len: int,
) -> torch.Tensor:
    """Map CS4A-selected video KV indices by relative (t, h, w) position.
    
    decision KV index
    ├─ sink prefix: keep at the current sink prefix
    └─ video token: decision 1D index -> decision (t,h,w)
                    -> current relative (t,h,w)
                    -> current 1D absolute KV index
    """
    dt, dh, dw = decision_scale_shape
    ct, ch, cw = current_scale_shape
    decision_video_len = dt * dh * dw
    current_video_len = ct * ch * cw
    if decision_video_len <= 0 or current_video_len <= 0:
        return inds_S.clamp(0, max(0, current_k_len - 1)).to(torch.int32)

    sink_mask = inds_S < decision_sink_len
    video_delta = (inds_S - decision_sink_len).clamp(0, decision_video_len - 1)

    src_t = torch.div(video_delta, dh * dw, rounding_mode='trunc')
    src_hw = video_delta - src_t * dh * dw
    src_h = torch.div(src_hw, dw, rounding_mode='trunc')
    src_w = src_hw - src_h * dw

    dst_t = torch.div((src_t * 2 + 1) * ct, 2 * dt, rounding_mode='trunc').clamp_max(ct - 1)
    dst_h = torch.div((src_h * 2 + 1) * ch, 2 * dh, rounding_mode='trunc').clamp_max(ch - 1)
    dst_w = torch.div((src_w * 2 + 1) * cw, 2 * dw, rounding_mode='trunc').clamp_max(cw - 1)
    mapped_video = current_sink_len + (dst_t * ch + dst_h) * cw + dst_w

    mapped_sink = inds_S.clamp(0, max(0, current_sink_len - 1))
    inds_K = torch.where(sink_mask, mapped_sink, mapped_video)
    return inds_K.clamp(0, max(0, current_k_len - 1)).to(torch.int32)


# -----------------------
# Spacetime Local Prior
# Neighboring frames
# -----------------------


# Cache of decision->current coordinate look-up tables. ``map_infinitystar_sparse_kv_inds``
# is a pure function of the index VALUE (it never depends on b/h/g), so the whole
# ``[B,H,Gk,count]`` elementwise chain can be replaced by one gather through a
# ``[decision_k_len]`` LUT. The LUT is geometry-only (same across heads/layers/repeats;
# rebuilt only when the prompt's sink_len changes the geometry), so we cache it with a
# small LRU like the band masks.
_coord_lut_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_COORD_LUT_CACHE_MAX = 8


@torch.no_grad()
def get_coord_lut(
    decision_sink_len: int,
    current_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    current_scale_shape: Tuple[int, int, int],
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """``[decision_k_len]`` int32 LUT: decision KV index -> current KV index.

    Built once per geometry by running the exact ``map_infinitystar_sparse_kv_inds``
    arithmetic over ``arange(decision_k_len)`` (so the result is bit-identical to the
    per-tensor path), then cached with a small LRU.
    """
    dk = max(int(decision_k_len), 1)
    key = (int(decision_sink_len), int(current_sink_len),
           tuple(int(x) for x in decision_scale_shape),
           tuple(int(x) for x in current_scale_shape),
           int(current_k_len), dk, str(device))
    cached = _coord_lut_cache.get(key)
    if cached is not None:
        _coord_lut_cache.move_to_end(key)
        return cached

    x = torch.arange(dk, device=device, dtype=torch.int32).view(1, 1, 1, -1)
    lut = map_infinitystar_sparse_kv_inds(
        inds_S=x,
        decision_sink_len=decision_sink_len,
        current_sink_len=current_sink_len,
        decision_scale_shape=decision_scale_shape,
        current_scale_shape=current_scale_shape,
        current_k_len=current_k_len,
    ).view(-1)                                                   # [decision_k_len] int32

    if len(_coord_lut_cache) >= _COORD_LUT_CACHE_MAX:
        _coord_lut_cache.popitem(last=False)                     # evict oldest
    _coord_lut_cache[key] = lut
    return lut


@torch.no_grad()
def map_infinitystar_sparse_kv_inds_lut(
    inds_S: torch.Tensor,
    decision_sink_len: int,
    current_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    current_scale_shape: Tuple[int, int, int],
    current_k_len: int,
    decision_k_len: int,
) -> torch.Tensor:
    """LUT-accelerated drop-in for ``map_infinitystar_sparse_kv_inds`` (bit-identical).

    Instead of ~25 elementwise kernels over the full ``[B,H,Gk,count]`` index tensor every
    call, gather through a cached ``[decision_k_len]`` value-map LUT (built once per
    geometry). ``inds_S`` holds decision KV indices in ``[0, decision_k_len)``.
    """
    lut = get_coord_lut(
        decision_sink_len=decision_sink_len,
        current_sink_len=current_sink_len,
        decision_scale_shape=decision_scale_shape,
        current_scale_shape=current_scale_shape,
        current_k_len=current_k_len,
        decision_k_len=decision_k_len,
        device=inds_S.device,
    )
    idx = inds_S.long().clamp_(0, lut.shape[0] - 1)              # decision indices ∈ [0, decision_k_len)
    return lut[idx]                                              # [B,H,Gk,count] int32, current coords


# Cache of per-scale full-frame band masks (geometry only, shared across heads and
# layers within a generation). Bounded LRU so per-prompt sink_len changes don't grow
# it unboundedly (~28 MB / entry at the last video scale).
_band_mask_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_BAND_MASK_CACHE_MAX = 8


@torch.no_grad()
def _get_band_mask(current_scale_shape, current_sink_len, current_k_len, bm, q_len,
                   neighbor_frames, device) -> torch.Tensor:
    """[G, k_len] bool: each query group's own frame + neighbour frames marked True.
    Boundary-shifted so every group covers exactly ``2*neighbor_frames+1`` full
    CONTIGUOUS frames (frame 0 -> {0,1,2}, last -> {ct-3,ct-2,ct-1}). Cached per scale;
    geometry depends only on (scale_shape, sink_len, k_len, bm, q_len, nf)."""
    ct, ch, cw = (int(x) for x in current_scale_shape)
    frame_size = ch * cw
    G = (int(q_len) + bm - 1) // bm
    k_len = int(current_k_len)
    key = (ct, ch, cw, int(current_sink_len), k_len, int(bm), G, int(neighbor_frames), str(device))
    cached = _band_mask_cache.get(key)
    if cached is not None:
        _band_mask_cache.move_to_end(key)
        return cached

    mask = torch.zeros(G, k_len, dtype=torch.bool, device=device)
    if frame_size > 0 and ct > 0 and G > 0:
        g = torch.arange(G, device=device, dtype=torch.int64)
        row_center = (g * bm + bm // 2).clamp(0, max(0, int(q_len) - 1))
        t_center = torch.div(row_center, frame_size, rounding_mode='trunc').clamp(0, ct - 1)
        W = min(2 * int(neighbor_frames) + 1, ct)
        w_start = (t_center - int(neighbor_frames)).clamp(0, ct - W)            # contiguous window start
        band_start = int(current_sink_len) + w_start * frame_size              # [G]
        cols = (band_start.view(G, 1)
                + torch.arange(W * frame_size, device=device, dtype=torch.int64).view(1, -1)).clamp(0, k_len - 1)
        mask.scatter_(1, cols, True)

    if len(_band_mask_cache) >= _BAND_MASK_CACHE_MAX:
        _band_mask_cache.popitem(last=False)   # evict oldest
    _band_mask_cache[key] = mask
    return mask


@torch.no_grad()
def build_frame_band_union_indices(
    inds: torch.Tensor,                       # [B, H, G, base_count] mapped top-k centers (current KV coords)
    current_sink_len: int,
    current_k_len: int,
    current_scale_shape: Tuple[int, int, int],  # (ct, ch, cw)
    bm: int,
    q_len: int,
    neighbor_frames: int,
    multiple_of: int,
    indices_pad_to: int,
):
    """Spatiotemporal-prior selection for InfinityStar video sparse scales, the
    Chipmunk way (NO sort): UNION a precomputed full-frame band mask with the mapped
    top-k centers, then ``mask_to_indices`` -> (inds, counts).

        union_mask   = band_mask  |  scatter(mapped top-k centers)
        inds, counts = mask_to_indices(union_mask, multiple_of, indices_pad_to)

    csp_attn reads each group's own ``counts[g]`` (variable, rounded up to a multiple
    of ``multiple_of`` by mask_to_indices, which also fills the rounding slack with
    unselected keys), so no uniform-count / priority / sort step is needed. The full
    frame band keeps every group's count large (>= band size), well inside the
    kernel's accurate regime. Returns ``(inds [B,H,G,k_len], counts [B,H,G])``.
    """
    B, H, G, _ = inds.shape
    k_len = int(current_k_len)
    band_mask = _get_band_mask(
        current_scale_shape, current_sink_len, k_len, bm, q_len,
        neighbor_frames, inds.device)                                           # [G, k_len]
    union = band_mask.view(1, 1, G, k_len).expand(B, H, G, k_len).clone()      # [B,H,G,k_len] bool

    # --- v1 ---
    # union.scatter_(-1, inds.to(torch.int64), True)                             # add mapped centers (coord map already clamped to k_len-1)
    # out_inds, counts = mask_to_indices(union, multiple_of, indices_pad_to)
    # out_inds = out_inds[:, :, :, :k_len]

    # --- v2 ---
    # union.scatter_(-1, inds.to(torch.int64).clamp_(0, k_len - 1), True)        # add mapped centers
    # out_inds, counts = mask_to_indices(union, multiple_of, multiple_of)
    # counts = counts.clamp_max((k_len // multiple_of) * multiple_of)
    # out_inds = out_inds[:, :, :, :k_len]

    # --- v3 ---
    union.scatter_(-1, inds.to(torch.int64).clamp_(0, k_len - 1), True)        # add mapped centers
    out_inds, counts = _mask_to_indices(union, multiple_of, indices_pad_to)
    out_inds = out_inds[:, :, :, :k_len]

    return out_inds, counts


# -----------------------
# Footprint expansion
# (inverse parent-LUT mask gather)
# -----------------------

# Cache of current->decision PARENT look-up tables (footprint mode). This is the INVERSE
# of the decision->current center map in ``map_infinitystar_sparse_kv_inds``: each CURRENT
# key inherits the selected-ness of the decision cell whose up-sampling footprint contains
# it, so a selected decision cell expands to its WHOLE current block (not just the center).
# Geometry-only (same across heads/layers/repeats), cached with a small LRU like the coord
# LUT / band masks.
_parent_lut_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_PARENT_LUT_CACHE_MAX = 8


@torch.no_grad()
def get_parent_lut(
    decision_sink_len: int,
    current_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    current_scale_shape: Tuple[int, int, int],
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
) -> torch.Tensor:
    """``[current_k_len]`` int64 LUT: current KV index -> its parent decision KV index.

    Piecewise (mirrors ``map_infinitystar_sparse_kv_inds`` but INVERTED):
      - sink/text prefix ``[0, sink_len)``: identity. The prefix (image-last-scale + text)
        is the same tokens at the same indices across decision/current scales (no resolution
        gap), so a decision-selected sink key is reused VERBATIM, not footprint-expanded.
      - video region: each current ``(t,h,w)`` maps to the decision cell whose footprint
        block contains it, ``dec = floor((cur+0.5) * D / C)`` per axis. Every current token
        sharing a parent inherits that cell's selected-ness -> the full footprint block is
        selected (no center-only sampling, no budget fill).
    """
    ck = max(int(current_k_len), 1)
    key = (int(decision_sink_len), int(current_sink_len),
           tuple(int(x) for x in decision_scale_shape),
           tuple(int(x) for x in current_scale_shape),
           ck, int(decision_k_len), str(device))
    cached = _parent_lut_cache.get(key)
    if cached is not None:
        _parent_lut_cache.move_to_end(key)
        return cached

    dt, dh, dw = (int(x) for x in decision_scale_shape)
    ct, ch, cw = (int(x) for x in current_scale_shape)
    dk = max(int(decision_k_len), 1)

    c = torch.arange(ck, device=device, dtype=torch.int64)
    # sink/text prefix -> identity (clamped into the decision sink range; decision and
    # current sink lengths are equal for InfinityStar, so this is the exact identity).
    parent = c.clamp(0, max(0, int(decision_sink_len) - 1))

    video_len = ct * ch * cw
    if video_len > 0 and dt > 0 and dh > 0 and dw > 0:
        is_video = c >= int(current_sink_len)
        v = (c - int(current_sink_len)).clamp(0, video_len - 1)
        cur_t = torch.div(v, ch * cw, rounding_mode='trunc')
        cur_hw = v - cur_t * ch * cw
        cur_h = torch.div(cur_hw, cw, rounding_mode='trunc')
        cur_w = cur_hw - cur_h * cw
        # inverse footprint: which decision cell's block contains this current token
        dec_t = torch.div((cur_t * 2 + 1) * dt, 2 * ct, rounding_mode='trunc').clamp_max(dt - 1)
        dec_h = torch.div((cur_h * 2 + 1) * dh, 2 * ch, rounding_mode='trunc').clamp_max(dh - 1)
        dec_w = torch.div((cur_w * 2 + 1) * dw, 2 * cw, rounding_mode='trunc').clamp_max(dw - 1)
        parent_video = int(decision_sink_len) + (dec_t * dh + dec_h) * dw + dec_w
        parent = torch.where(is_video, parent_video, parent)

    parent = parent.clamp_(0, dk - 1).to(torch.int64)

    if len(_parent_lut_cache) >= _PARENT_LUT_CACHE_MAX:
        _parent_lut_cache.popitem(last=False)   # evict oldest
    _parent_lut_cache[key] = parent
    return parent


# Kill-switch for the identity bypass in build_footprint_union_indices (A/B lever only;
# the bypass is a mathematical identity, so it has no config key). Read once at import.
_IDENTITY_REUSE_DISABLED = os.environ.get('SPARVAR_IDENTITY_REUSE', '1') == '0'


@torch.no_grad()
def build_footprint_union_indices(
    decision_mask: torch.Tensor,                # [B, H, Gs, decision_k_len] bool (bitunpacked decision selection)
    decision_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    decision_k_len: int,
    current_sink_len: int,
    current_k_len: int,
    current_scale_shape: Tuple[int, int, int],  # (ct, ch, cw)
    current_q_len: int,
    bm: int,
    multiple_of: int,
    indices_pad_to: int,
    neighbor_frames: int = 0,
    use_band: bool = False,
    out_width: int = -1,     # >0: compact mask_to_indices output row width (index_width_utils)
):
    """Footprint expansion for InfinityStar video sparse scales (inverse parent-LUT mask
    gather, NO sort). Carries the decision-scale sparse pattern to the current scale by its
    geometric FOOTPRINT instead of one-to-one centers + budget fill:

        gS_idx       = center-NN q-group remap (Gs -> Gk)              # same as map_inds_counts_across_scales
        current_mask = decision_mask[:, :, gS_idx[:, None], parent_lut[None, :]]
                       # [B,H,Gk,current_k_len]  q-group inherit + footprint-complete, ONE fused gather
        (optional)   current_mask |= full-frame band_mask
        inds, counts = mask_to_indices(current_mask)                   # variable per-group counts

    The sink/text prefix is reused verbatim (parent_lut identity there); only the video
    region is footprint-mapped. No budget, no random fill, no priority sort -- csp_attn
    reads each group's own variable ``counts[g]`` (mask_to_indices rounds up to a multiple
    of ``multiple_of`` and fills the slack with unselected keys). Returns
    ``(inds [B,H,Gk,k_len], counts [B,H,Gk])``.
    """
    B, H, Gs, _ = decision_mask.shape
    device = decision_mask.device
    k_len = int(current_k_len)
    Gk = (int(current_q_len) + bm - 1) // bm

    # 0) IDENTITY BYPASS — decision-scale reuse repeats (decision_repeat_idx:
    #    e.g. 720p scale-27 repeat-1, 480p scale-25 repeat-1/2).
    #    When the current geometry equals the decision geometry, gS_idx and parent_lut
    #    are both arange, so the fused gather below would only COPY decision_mask.
    #    Skip it: alias the mask (no band) or union the band OUT-OF-PLACE onto it.
    #    Bit-exact vs the full pipeline: the same mask VALUES reach the same
    #    deterministic _mask_to_indices kernel. SPARVAR_IDENTITY_REUSE=0 disables.
    identity_geometry = (
        not _IDENTITY_REUSE_DISABLED
        and current_scale_shape is not None and decision_scale_shape is not None
        and tuple(int(x) for x in current_scale_shape) == tuple(int(x) for x in decision_scale_shape)
        and k_len == int(decision_k_len)
        and int(current_sink_len) == int(decision_sink_len)
        and Gk == Gs
    )
    if identity_geometry:
        if use_band:
            band_mask = _get_band_mask(
                current_scale_shape, current_sink_len, k_len, bm, current_q_len,
                neighbor_frames, device)                                                # [Gk, k_len]
            # Out-of-place union on purpose: decision_mask belongs to the caller and
            # the LRU-cached band_mask is read-only — neither may be mutated here.
            current_mask = decision_mask | band_mask.view(1, 1, Gk, k_len)
        else:
            # Pure alias: _mask_to_indices only READS the mask. The CUDA kernel indexes
            # the raw data_ptr, so guard contiguity (bitunpack outputs and leading-dim
            # _match_batch slices are already contiguous; this is a safety net).
            current_mask = decision_mask if decision_mask.is_contiguous() else decision_mask.contiguous()
    else:
        # 1) q-group remap Gs -> Gk (bit-identical formula to map_inds_counts_across_scales):
        #    current group gk inherits decision group gS_idx[gk]'s full selection (sink incl.).
        gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5
        gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)   # [Gk]

        # 2) key-dim footprint map: current key selected iff its parent decision cell was.
        parent_lut = get_parent_lut(
            decision_sink_len=decision_sink_len,
            current_sink_len=current_sink_len,
            decision_scale_shape=decision_scale_shape,
            current_scale_shape=current_scale_shape,
            current_k_len=k_len,
            decision_k_len=decision_k_len,
            device=device,
        )
        # Fused q-group remap + footprint gather in ONE advanced-indexing kernel:
        # out[b,h,g,c] = decision_mask[b,h,gS_idx[g],parent_lut[c]]. Equivalent to
        # index_select(2, gS_idx) -> index_select(3, parent_lut) but never materializes
        # the [B,H,Gk,decision_k_len] intermediate (~0.6 GB at the last 720p scale).
        current_mask = decision_mask[:, :, gS_idx[:, None], parent_lut[None, :]]             # [B,H,Gk,k_len] bool

        # 3) optionally union a precomputed full-frame band (temporal-locality safety net).
        # if use_band and neighbor_frames > 0:
        if use_band:
            band_mask = _get_band_mask(
                current_scale_shape, current_sink_len, k_len, bm, current_q_len,
                neighbor_frames, device)                                                    # [Gk, k_len]
            # In-place union: current_mask is a fresh base tensor (the advanced-indexing
            # gather above always copies), and the LRU-cached band_mask is only READ —
            # never flip this to mutate band_mask, that would poison _band_mask_cache.
            # Saves the second [B,H,Gk,k_len] bool transient at the largest scales.
            current_mask |= band_mask.view(1, 1, Gk, k_len)

    # 4) mask -> (inds, variable counts) consumed directly by csp_attn. With a compact
    #    out_width the [:k_len] slice is a no-op (torch clamps over-long slices).
    out_inds, counts = _mask_to_indices(current_mask, multiple_of, indices_pad_to, out_width)
    out_inds = out_inds[:, :, :, :k_len]
    return out_inds, counts
