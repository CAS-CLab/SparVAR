"""D-CSLA v1: dynamic block-wise sparse attention for InfinityStar video scales
and Infinity image scales (the ``*_image`` builders below swap the video KV
layout for the stacked image pyramid; the selection and forward stages are
shared).

Zero-kernel-change pipeline, an alternative execution path to the token-level
CS4A csp_attn (config-gated via ``attn.dcsla_enabled``):

    colsum at the decision scale (existing Triton kernel; query-group rows set
    by ``dcsla_colsum_group_rows``: 192 legacy / 128 block-aligned / 64 finest)
      -> footprint MASS transfer: block-count matrix C built from the geometric
         parent LUT (``get_parent_lut``) carries the float colsum attention mass
         to each future scale as per-(query-group, kv-block) block mass
      -> per-(head, q-block) coverage-based 128-wide KV-block selection
         (sink blocks force-kept, optional diagonal band)
      -> FlexAttention block-sparse forward (``BlockMask.from_kv_blocks``)

torch >= 2.7 compatibility notes (verified against the v2.7.1 source; the same
API/behavior holds on newer releases, e.g. the 2.11 remote env and 2.12):
  - ``BlockMask.from_kv_blocks`` accepts ``seq_lengths``; with ``mask_mod=None``
    (noop) and exact ``seq_lengths=(q_len, kv_len)`` the compiled kernel handles
    non-multiple-of-128 boundaries itself (IS_DIVISIBLE=False codegen masks the
    KV tail with -inf and store-masks the Q tail).
  - ``kv_indices`` MUST be full-width ``[B, H, U, V]`` with values < V: the
    internal ``_transpose_ordered -> _ordered_to_dense`` scatter-writes at the
    index VALUES into a buffer of width ``kv_indices.shape[-1] + 1``, so
    truncating to a max-count would write out of bounds. Entries beyond
    ``kv_num_blocks`` are ignored by the kernel.
  - Never call the *uncompiled* flex_attention at real sizes: the eager path
    materializes the dense score matrix (72000 x 75643 at 720p scale 29).
"""
import math
import os
from collections import OrderedDict
from functools import partial
from typing import Dict, Optional, Tuple

import torch
from torch import Tensor
from torch.nn import functional as F
from torch.nn.attention.flex_attention import BlockMask, flex_attention

from models.sparvar.dcsla_density_probe_utils import maybe_log_dcsla_density
from models.sparvar.kbench_pattern_dump_utils import maybe_dump_dcsla_pattern
from utils import lprof
from models.sparvar.index_map_utils import get_infinity_parent_lut, get_parent_lut
from models.sparvar.vcsla_utils import build_video_frame_band_mask
from models.sparvar.sparse_index_viz_utils import (dump_dcsla_block_mask_viz,
                                                   dump_dcsla_mass_selection_viz,
                                                   image_kv_scale_boundaries,
                                                   viz_enabled_for)

DCSLA_BLOCK = 128

# FA4 path only: pad KV (and the block-list width) up to this multiple, so the
# per-prompt text-length variation collapses into one shape bucket per scale —
# one CuTe template compile per (scale, 1024-window) per process instead of one
# per prompt (the vbench speed run tripped dynamo's recompile_limit=8 after 4
# prompts and fell back to an eager flex that tried to allocate 134.5 GiB).
# Must be a multiple of DCSLA_BLOCK. Padded blocks are never selected and the
# real ragged boundary is enforced by the runtime kv-len tensor, so it's exact.
_FA4_KV_BUCKET = 1024

# SPARVAR_DCSLA_QBLOCKS=0 skips BlockMask's q-side (transposed) block lists. The
# forward kernel only reads the kv-side lists (q_* lists serve the backward), so
# the output is unchanged; it drops the two dense-scatter + sort transposes per
# layer that dominate the mask-build time at the large scales. Default on (=1)
# keeps the legacy behaviour.
_COMPUTE_Q_BLOCKS = os.environ.get('SPARVAR_DCSLA_QBLOCKS', '1').lower() not in ('0', 'false', 'no', 'off')

# SPARVAR_DCSLA_HOST_OPT=1: cache the geometry-only tensors of the sparse step (the
# spatial/temporal force-include priors, the FA4 pad/partial index permutations, the
# phi remap) across layers and videos instead of rebuilding them per layer, and never
# create small constant tensors from Python lists on the hot path (each such
# torch.tensor(..., device=cuda) is a synchronous pageable H2D copy that stalls the
# CPU behind the GPU and leaves the GPU idle while the next layer is dispatched).
# Values are identical -> bit-exact; default off keeps the legacy behaviour.
_HOST_OPT = os.environ.get('SPARVAR_DCSLA_HOST_OPT', '0').lower() in ('1', 'true', 'yes', 'on')
_geom_cache: Dict[tuple, Tensor] = {}


def _geom(key: tuple, build):
    """Geometry-only tensor cache (device-resident, keyed by ints/strings only)."""
    t = _geom_cache.get(key)
    if t is None:
        t = build()
        if len(_geom_cache) > 256:
            _geom_cache.clear()
        _geom_cache[key] = t
    return t


def _cdiv(a: int, b: int) -> int:
    return (a + b - 1) // b


_warned_once = set()


def dcsla_warn_once(msg: str) -> None:
    if msg not in _warned_once:
        _warned_once.add(msg)
        print(msg)


# -----------------------
# Block-count matrix C (geometry-only, LRU-cached like the parent LUT)
# -----------------------

_block_count_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_BLOCK_COUNT_CACHE_MAX = 4


def _counts_from_parent_lut(
    parent_lut: torch.Tensor,                # [current_k_len] int64, child -> decision index
    current_k_len: int,
    decision_k_len: int,
    V: int,
    block_size: int,
    normalize: bool,
) -> torch.Tensor:
    """``[decision_k_len, V]`` bf16 block-count matrix from a parent LUT.

    Layout-agnostic: the only thing that differs between the video and image KV
    layouts is which LUT builder produced ``parent_lut``.
    """
    device = parent_lut.device
    block_id = torch.arange(int(current_k_len), device=device, dtype=torch.int64) // int(block_size)
    flat = parent_lut * V + block_id
    counts = torch.zeros(int(decision_k_len) * V, device=device, dtype=torch.float32)
    counts.index_add_(0, flat, torch.ones(int(current_k_len), device=device, dtype=torch.float32))
    counts = counts.view(int(decision_k_len), V)
    if normalize:
        counts = counts / counts.sum(dim=-1, keepdim=True).clamp_min(1.0)
    return counts.to(torch.bfloat16)


@torch.no_grad()
def get_block_count_matrix(
    decision_sink_len: int,
    current_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    current_scale_shape: Tuple[int, int, int],
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
    block_size: int = DCSLA_BLOCK,
    normalize: bool = True,
) -> torch.Tensor:
    """``[decision_k_len, V]`` bf16, V = ceil(current_k_len / block_size).

    Raw counts: C[j, v] = #{current KV index j' in kv-block v : parent_lut[j']
    == j}, i.e. how many children of decision token j land in kv-block v.

    normalize=True (default, mass-conserving transfer): each row is divided by
    its footprint multiplicity, so a decision token distributes exactly its own
    colsum mass across the blocks its footprint covers. This is required for a
    FAIR sink-vs-video ranking (dcsla_keep_sink=false): the ctx prefix maps 1:1
    (multiplicity 1) while video parents have {1,2,4} children (t identity x
    ~1.9 on h/w at 720p), so raw counts inflate video blocks by 2-4x relative
    to sink blocks — and add the same rounding artifact within the video
    region. normalize=False keeps the raw count weighting (the original v1
    behavior when sink blocks were unconditionally kept).

    Entries are small (counts <= footprint size; normalized <= 1), bf16 keeps
    the matmul on tensor cores regardless of the process TF32 setting.
    """
    ck = max(int(current_k_len), 1)
    V = _cdiv(ck, int(block_size))
    key = (int(decision_sink_len), int(current_sink_len),
           tuple(int(x) for x in decision_scale_shape),
           tuple(int(x) for x in current_scale_shape),
           ck, int(decision_k_len), int(block_size), bool(normalize), str(device))
    cached = _block_count_cache.get(key)
    if cached is not None:
        # print(f'        use _block_count_cache')      # only first layer need to build cache
        _block_count_cache.move_to_end(key)
        return cached

    parent_lut = get_parent_lut(
        decision_sink_len=decision_sink_len,
        current_sink_len=current_sink_len,
        decision_scale_shape=decision_scale_shape,
        current_scale_shape=current_scale_shape,
        current_k_len=ck,
        decision_k_len=decision_k_len,
        device=device,
    )
    C = _counts_from_parent_lut(parent_lut, ck, int(decision_k_len), V, block_size, normalize)

    if len(_block_count_cache) >= _BLOCK_COUNT_CACHE_MAX:
        _block_count_cache.popitem(last=False)
    _block_count_cache[key] = C
    return C


@torch.no_grad()
def get_image_block_count_matrix(
    decision_scale_ind: int,
    current_scale_ind: int,
    scale_schedule,
    current_k_len: int,
    decision_k_len: int,
    device: torch.device,
    block_size: int = DCSLA_BLOCK,
    normalize: bool = True,
) -> torch.Tensor:
    """Infinity (image) counterpart of ``get_block_count_matrix``.

    Same semantics -- ``C[j, v]`` is how much of decision token ``j``'s mass lands
    in kv-block ``v`` -- over the stacked-pyramid KV layout
    ``Concat(scale_0..scale_k)``, built from the image parent LUT
    (``get_infinity_parent_lut``) instead of the video one. The image LUT already
    encodes the parentless prefix as identity, so no sink bookkeeping is needed
    here.
    """
    ck = max(int(current_k_len), 1)
    V = _cdiv(ck, int(block_size))
    key = ('image', int(decision_scale_ind), int(current_scale_ind),
           tuple(tuple(int(x) for x in s) for s in scale_schedule[:int(current_scale_ind) + 1]),
           ck, int(decision_k_len), int(block_size), bool(normalize), str(device))
    cached = _block_count_cache.get(key)
    if cached is not None:
        _block_count_cache.move_to_end(key)
        return cached

    parent_lut = get_infinity_parent_lut(
        decision_scale_ind=decision_scale_ind,
        current_scale_ind=current_scale_ind,
        scale_schedule=scale_schedule,
        current_k_len=ck,
        decision_k_len=int(decision_k_len),
        device=device,
    )
    C = _counts_from_parent_lut(parent_lut, ck, int(decision_k_len), V, block_size, normalize)

    if len(_block_count_cache) >= _BLOCK_COUNT_CACHE_MAX:
        _block_count_cache.popitem(last=False)
    _block_count_cache[key] = C
    return C


# -----------------------
# Decision-scale mass transfer
# -----------------------

@torch.no_grad()
def compute_decision_block_mass(
    bs: Tensor,                              # [B, H, G_S, decision_k_len] colsum (fp32 triton / bf16 cuda)
    *,
    scale_schedule,
    decision_scale_ind: int,
    decision_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    decision_k_len: int,
    block_size: int = DCSLA_BLOCK,
    normalize: bool = True,
    include_decision_scale: bool = False,
) -> Tuple[Dict[int, Tensor], Dict[int, int]]:
    """At the decision scale, transfer the colsum mass to every later scale.

    For each si > decision_scale_ind: future_k_len = decision_sink_len + T*H*W
    (InfinityStar per-scale KV = [ctx prefix] + [current video scale]; the
    prefix length is runtime-asserted equal across scales in sparse_attn.py),
    and ``mass[si] = bs @ C_si`` -> [B, H, G_S, V_si] bf16, stored per layer.

    include_decision_scale: also emit mass for the decision scale itself (the
    same-scale transfer matrix is an exact token->block membership, so this is
    the plain block-sum of the colsum). Needed when decision_repeat_idx places
    the decision on a non-last repeat: the scale's later repeats replay the
    block-sparse forward on their own geometry.

    Returns (mass_dict, expected_k_len_dict). Non-3D schedule entries are
    skipped (same guard as ``_as_scale_shape``).
    """
    if decision_scale_shape is None or len(tuple(decision_scale_shape)) != 3:
        raise RuntimeError('D-CSLA requires a (T,H,W) decision scale shape; '
                           'pass scale_schedule into SparseDiffAttn.forward.')
    bs_bf16 = bs.to(torch.bfloat16)
    mass_dict: Dict[int, Tensor] = {}
    k_len_dict: Dict[int, int] = {}
    start_si = int(decision_scale_ind) + (0 if include_decision_scale else 1)
    for si in range(start_si, len(scale_schedule)):
        shape = scale_schedule[si]
        if len(shape) != 3:
            continue
        shape = tuple(int(x) for x in shape)
        future_k_len = int(decision_sink_len) + shape[0] * shape[1] * shape[2]
        C = get_block_count_matrix(
            decision_sink_len=decision_sink_len,
            current_sink_len=decision_sink_len,
            decision_scale_shape=decision_scale_shape,
            current_scale_shape=shape,
            current_k_len=future_k_len,
            decision_k_len=int(decision_k_len),
            device=bs.device,
            block_size=block_size,
            normalize=normalize,
        )
        mass_dict[si] = torch.matmul(bs_bf16, C)        # [B, H, G_S, V_si]
        k_len_dict[si] = future_k_len
    return mass_dict, k_len_dict


@torch.no_grad()
def compute_decision_block_mass_image(
    bs: Tensor,                              # [B, H, G_S, decision_k_len] colsum
    *,
    scale_schedule,
    decision_scale_ind: int,
    decision_k_len: int,
    block_size: int = DCSLA_BLOCK,
    normalize: bool = True,
) -> Tuple[Dict[int, Tensor], Dict[int, int]]:
    """Infinity (image) counterpart of ``compute_decision_block_mass``.

    The image KV is the stacked pyramid ``Concat(scale_0..scale_si)``, so a later
    scale's KV length is the cumulative token count through ``si`` rather than the
    video's ``[ctx prefix] + [one scale]``. Everything else is identical:
    ``mass[si] = bs @ C_si`` -> ``[B, H, G_S, V_si]`` bf16.
    """
    bs_bf16 = bs.to(torch.bfloat16)
    mass_dict: Dict[int, Tensor] = {}
    k_len_dict: Dict[int, int] = {}
    cums = [0]
    for shape in scale_schedule:
        shape = tuple(int(x) for x in shape)
        cums.append(cums[-1] + shape[0] * shape[1] * shape[2])      # [0, 1, 5, 21, 57, 121, 265, 521, 921, 1497, 2521, 4121, 6425, 10521]
    for si in range(int(decision_scale_ind) + 1, len(scale_schedule)):
        future_k_len = cums[si + 1]
        C = get_image_block_count_matrix(
            decision_scale_ind=decision_scale_ind,
            current_scale_ind=si,
            scale_schedule=scale_schedule,
            current_k_len=future_k_len,
            decision_k_len=int(decision_k_len),
            device=bs.device,
            block_size=block_size,
            normalize=normalize,
        )
        mass_dict[si] = torch.matmul(bs_bf16, C)        # [B, H, G_S, V_si]
        k_len_dict[si] = future_k_len
    return mass_dict, k_len_dict


# -----------------------
# Cross-scale locality prior (image)
# -----------------------

_image_band_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
# 8: the static CSLA path (sparse_attn_csla) adds up to 3 keys of its own on top
# of D-CSLA's 2-3, and an eviction would rebuild a mask every forward.
_IMAGE_BAND_CACHE_MAX = 8


@torch.no_grad()
def build_image_cross_scale_band_mask(
    scale_schedule,
    current_scale_ind: int,
    q_len: int,
    kv_len: int,
    win_size,
    device: torch.device,
    block_size: int = DCSLA_BLOCK,
    q_chunk: int = 512,
) -> Tensor:
    """``[U, V]`` bool block mask of the CSLA cross-scale locality prior.

    For every historical scale ``l <= current_scale_ind`` a query at ``(q_y, q_x)``
    of the current ``h_k x w_k`` grid keeps the ``win_size[l] x win_size[l]``
    window around its centre-aligned position
    ``(round(q_y / h_k * h_l), round(q_x / w_k * w_l))`` in scale ``l``. That is
    the geometric rule of the conference CSLA mask_mod
    (``basic_sparvar.py: create_cross_scale_block_attn_mask``), reduced to
    ``block_size`` blocks with 'any'; ``win_size[l] < 1`` disables scale ``l``
    (``-1 // 2 == -1``, so ``|d| <= -1`` never holds -- same as the mask_mod).

    This is the image analogue of the video path's intra-frame stripe plus
    temporal shifts: geometry only, so one build per (schedule, scale) is reused
    by every layer and every image.
    """
    K = int(current_scale_ind)
    hs = [int(s[-2]) for s in scale_schedule[:K + 1]]
    ws = [int(s[-1]) for s in scale_schedule[:K + 1]]
    wins = [int(win_size[l]) if l < len(win_size) else int(win_size[-1]) for l in range(K + 1)]
    key = (tuple(hs), tuple(ws), tuple(wins), int(q_len), int(kv_len), int(block_size), str(device))
    cached = _image_band_cache.get(key)
    if cached is not None:
        _image_band_cache.move_to_end(key)
        return cached

    n_per = [h * w for h, w in zip(hs, ws)]
    assert sum(n_per) == int(kv_len), \
        f'image band: schedule tokens {sum(n_per)} != kv_len {kv_len} at scale {K}'
    assert hs[K] * ws[K] == int(q_len), \
        f'image band: current scale {hs[K]}x{ws[K]} != q_len {q_len}'
    U, V = _cdiv(int(q_len), int(block_size)), _cdiv(int(kv_len), int(block_size))

    cum = torch.zeros(K + 2, device=device, dtype=torch.int64)
    cum[1:] = torch.cumsum(torch.tensor(n_per, device=device, dtype=torch.int64), dim=0)
    hs_t = torch.tensor(hs, device=device, dtype=torch.int64)
    ws_t = torch.tensor(ws, device=device, dtype=torch.int64)
    win_t = torch.tensor(wins, device=device, dtype=torch.int64)

    # per-KV-token geometry (cheap, kv_len-sized)
    kv = torch.arange(int(kv_len), device=device, dtype=torch.int64)
    l_of = torch.bucketize(kv, cum[1:K + 2], right=True).clamp_(0, K)   # owning scale
    w_of = ws_t[l_of]
    rel = kv - cum[l_of]
    kv_y = torch.div(rel, w_of, rounding_mode='floor')
    kv_x = rel - kv_y * w_of
    kv_yf = kv_y.to(torch.float32).unsqueeze(0)
    kv_xf = kv_x.to(torch.float32).unsqueeze(0)
    h_of = hs_t[l_of].to(torch.float32).unsqueeze(0)
    w_of_f = w_of.to(torch.float32).unsqueeze(0)
    half = torch.div(win_t[l_of], 2, rounding_mode='floor').to(torch.float32).unsqueeze(0)

    H_q, W_q = float(hs[K]), float(ws[K])
    W_q_i = int(ws[K])
    # chunk the query axis (block-aligned) so the token-level band never
    # materialises the full q_len x kv_len matrix at once
    q_chunk = max(int(q_chunk) // int(block_size), 1) * int(block_size)
    rows = []
    for q0 in range(0, int(q_len), q_chunk):
        q1 = min(q0 + q_chunk, int(q_len))
        qi = torch.arange(q0, q1, device=device, dtype=torch.int64)
        qy_i = torch.div(qi, W_q_i, rounding_mode='floor')
        qx_i = qi - qy_i * W_q_i
        cy = torch.round(qy_i.to(torch.float32).unsqueeze(1) / H_q * h_of)
        cx = torch.round(qx_i.to(torch.float32).unsqueeze(1) / W_q * w_of_f)
        band = ((cy - kv_yf).abs() <= half) & ((cx - kv_xf).abs() <= half)
        n_rows = int(q1 - q0)
        n_pad = _cdiv(n_rows, int(block_size)) * int(block_size)
        buf = torch.zeros(n_pad, V * int(block_size), dtype=torch.bool, device=device)
        buf[:n_rows, :int(kv_len)] = band
        rows.append(buf.view(n_pad // int(block_size), int(block_size), V, int(block_size))
                    .any(dim=3).any(dim=1))                             # [n_blocks, V]
    mask = torch.cat(rows, dim=0)[:U].contiguous()

    if len(_image_band_cache) >= _IMAGE_BAND_CACHE_MAX:
        _image_band_cache.popitem(last=False)
    _image_band_cache[key] = mask
    return mask


# -----------------------
# Per-row block selection
# -----------------------

@torch.no_grad()
def select_kv_blocks(
    mass_rows: Tensor,                       # [B, H, U, V] fp32, already phi-expanded + batch-matched
    *,
    sink_len: int,
    q_len: int,
    kv_len: int,
    block_size: int = DCSLA_BLOCK,
    mode: str = 'coverage',                  # ['coverage', 'topk_ratio']
    tau: float = 0.9,
    topk_ratio: float = 0.2,
    min_blocks: int = 16,
    max_blocks_ratio: float = 0.5,
    local_diag_blocks: int = 1,
    keep_sink: bool = True,
    sink_select_tau: float = 0.0,
    scale_shape=None,                        # (T, H, W) of the CURRENT video scale
    temporal_neighbor_frames: int = 0,       # N: force-keep +/-N frames at the same position
    temporal_diag_blocks: int = 0,           # radius (in blocks) around each shifted center
    diag_offset: Optional[int] = None,       # KV offset the q-block diagonal anchors at; None = sink_len
    extra_block_mask: Optional[Tensor] = None,   # [U, V] bool force-include prior (image cross-scale band | video frame band)
    scale_ind=None,                          # probe labels only
    layer_ind=None,
) -> Tuple[Tensor, Tensor]:
    """Coverage/top-k KV-block selection per (batch, head, q-block) row.

    Coverage semantics (mode='coverage'): tau ALWAYS means "select blocks until
    tau of the row's TOTAL mass is covered". Whatever sink mass the sink policy
    already covers is credited against that target before the video ranking,
    so a sink-dominant row gets a small video budget instead of chasing tau
    over its near-flat video residual (which would blow up to max_blocks).

    Three sink policies (sink = ctx prefix, ceil(sink_len/block_size) blocks;
    the last one may straddle into video tokens -- safe over-inclusion):
      keep_sink=True: all sink blocks force-kept; their full mass is credited.
      keep_sink=False, sink_select_tau>0 (SEPARATE selection): the sink region
        runs its OWN per-row coverage at sink_select_tau over the SINK mass
        (>=1 block/row); the selected sink mass is credited. The sink mass is
        EXACT -- the ctx prefix transfers by identity (multiplicity 1), no
        cross-scale approximation -- so this keeps each head's genuinely
        attended sink blocks without letting block-aggregated video totals
        crowd out spiky sink tokens.
      keep_sink=False, sink_select_tau=0 (joint compete): one ranking over all
        blocks against the total-mass target. Use mass-conserving transfer
        (dcsla_mass_normalize=true) here, otherwise raw count weighting
        under-ranks sink blocks by 2-4x.
    Two force-include locality priors sit on top of the mass selection (cheap
    insurance for rows whose transferred mass is unreliable); both are additive
    and neither replaces the mass ranking:
      local_diag_blocks (SPATIAL, intra-frame): +/-r blocks around each q-block's
        diagonally aligned KV position. Because the layout is t-major and flat,
        a 128-token block spans w*k tokens, so this band is a horizontal STRIPE
        at the same vertical position of the SAME frame (720p s29: +/-1 block =
        384 tokens = 4.8 of 45 rows).
      temporal_neighbor_frames (TEMPORAL, cross-frame): the same spatial position
        in the +/-N neighbouring frames, i.e. the q-block's token span shifted by
        delta*H*W, widened by +/-temporal_diag_blocks. Frame length is generally
        NOT a multiple of block_size (s29: 3600 = 28*128 + 16), so consecutive
        deltas drift in phase; the shifted CENTER is used, which costs at most a
        fraction-of-a-row spatial offset. Shifts that leave the current scale
        (frame 0 looking back, last frame looking forward) are DROPPED, never
        clamped -- clamping would fold them onto block 0 (sink) or V-1 and force
        semantically wrong blocks. Attention within a scale is bidirectional
        (flash_attn_func is called without causal), so +delta is legitimate.
        Requires scale_shape; a no-op for image scales (T <= 1) or N = 0.
      extra_block_mask (cross-scale band | frame band): a precomputed [U, V]
        block prior OR'd in verbatim. The Infinity path passes the CSLA
        cross-scale band (build_image_cross_scale_band_mask); the video path
        passes the V-CSLA full-frame band (build_video_frame_band_mask,
        dcsla_vcsla_frame_band_rho >= 0).

    Returns:
      kv_indices     [B, H, U, V] int32 -- FULL-WIDTH permutation; first
                     kv_num_blocks[b,h,u] entries are the selected block ids in
                     ascending order (mirrors torch's _dense_to_ordered), the
                     tail holds the unselected ids and is ignored downstream.
      kv_num_blocks  [B, H, U] int32
    """
    B, H, U, V = mass_rows.shape
    device = mass_rows.device
    assert V == _cdiv(int(kv_len), int(block_size)), \
        f'mass width {V} != ceil(kv_len={kv_len} / block={block_size})'
    sink_block_end = min(_cdiv(max(int(sink_len), 0), int(block_size)), V)

    mf = mass_rows.float()      # torch.float32
    sink_split = (keep_sink or float(sink_select_tau) > 0) and sink_block_end > 0
    m = mf
    if sink_split:
        m = mf.clone()
        m[..., :sink_block_end] = 0.0

    # Sink handling FIRST: the video coverage target below credits the sink mass
    # already covered, so tau uniformly means "cover tau of the row's TOTAL
    # mass" under all three sink policies. Without the credit, a sink-dominant
    # head leaves a near-flat tiny video residual, covering tau of THAT blows
    # the video budget up to max_blocks, and bf16 ties then select a contiguous
    # column prefix (observed at 480p scale 27 head0: density 0.072 -> 0.506).
    sink_sel = None
    sink_covered = None                                      # [B,H,U] mass already covered
    if sink_split:
        if keep_sink:
            sink_covered = mf[..., :sink_block_end].sum(dim=-1)
        else:
            ms = mf[..., :sink_block_end]
            s_sorted, s_idx = ms.sort(dim=-1, descending=True, stable=True)
            s_csum = s_sorted.cumsum(dim=-1)
            s_raw_total = s_csum[..., -1]
            n_s = (s_csum < float(sink_select_tau) * s_raw_total.unsqueeze(-1).clamp_min(1e-12)).sum(dim=-1) + 1
            # all-zero sink mass rows would otherwise select ALL sink blocks
            n_s = torch.where(s_raw_total > 0, n_s, torch.ones_like(n_s)).clamp_(1, sink_block_end)
            keep_s = torch.arange(sink_block_end, device=device).view(1, 1, 1, -1) < n_s.unsqueeze(-1)
            sink_sel = torch.zeros_like(keep_s)
            sink_sel.scatter_(-1, s_idx, keep_s)
            sink_covered = torch.gather(s_csum, -1, (n_s - 1).unsqueeze(-1)).squeeze(-1)

    sorted_m, sort_idx = m.sort(dim=-1, descending=True, stable=True)
    max_blocks = min(max(int(min_blocks), int(math.ceil(float(max_blocks_ratio) * V))), V)
    if mode == 'coverage':
        csum = sorted_m.cumsum(dim=-1)
        total_all = mf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        target = float(tau) * total_all
        if sink_covered is not None:
            target = (target - sink_covered.unsqueeze(-1)).clamp_min(0.0)
        n = (csum < target).sum(dim=-1) + 1                  # [B,H,U] int64
    elif mode == 'topk_ratio':
        n = torch.full((B, H, U), int(math.ceil(float(topk_ratio) * V)),
                       device=device, dtype=torch.int64)
    else:
        raise ValueError(f'unknown dcsla select mode: {mode}')
    n = n.clamp_(min=int(min_blocks), max=max_blocks)

    keep_sorted = torch.arange(V, device=device).view(1, 1, 1, V) < n.unsqueeze(-1)
    mask = torch.zeros_like(keep_sorted)
    mask.scatter_(-1, sort_idx, keep_sorted)                             # [B,H,U,V] bool

    if sink_block_end > 0:
        if keep_sink:
            mask[..., :sink_block_end] = True
        elif sink_sel is not None:
            mask[..., :sink_block_end] |= sink_sel
    if int(local_diag_blocks) > 0:
        # video: the current scale starts right after the ctx prefix, so the
        # diagonal anchors at sink_len; image: the current scale sits at the END
        # of the stacked pyramid, so the caller passes its cumulative offset.
        diag_base = int(sink_len) if diag_offset is None else int(diag_offset)

        def _build_diag():
            u_ids = torch.arange(U, device=device, dtype=torch.int64)
            center_tok = (u_ids * int(block_size) + int(block_size) // 2).clamp_max(max(int(q_len) - 1, 0))
            c_u = (diag_base + center_tok) // int(block_size)                # [U]
            offs = torch.arange(-int(local_diag_blocks), int(local_diag_blocks) + 1,
                                device=device, dtype=torch.int64)
            diag_cols = (c_u.view(-1, 1) + offs.view(1, -1)).clamp_(0, V - 1)  # [U, 2r+1]
            dm = torch.zeros(U, V, dtype=torch.bool, device=device)
            dm.scatter_(-1, diag_cols, True)
            return dm
        if _HOST_OPT:
            diag_mask = _geom(('diag', U, V, int(block_size), int(q_len), diag_base, int(local_diag_blocks), str(device)),
                              _build_diag)
        else:
            diag_mask = _build_diag()
        mask |= diag_mask.view(1, 1, U, V)

    frame_tokens = 0
    if scale_shape is not None and int(temporal_neighbor_frames) > 0:
        t_s, h_s, w_s = (int(x) for x in scale_shape)
        if t_s > 1 and h_s * w_s > 0 and t_s * h_s * w_s == int(q_len):
            frame_tokens = h_s * w_s
    if frame_tokens > 0:
        N = int(temporal_neighbor_frames)
        r_t = int(temporal_diag_blocks)

        def _build_temporal():
            u_ids = torch.arange(U, device=device, dtype=torch.int64)
            center_tok = (u_ids * int(block_size) + int(block_size) // 2).clamp_max(max(int(q_len) - 1, 0))
            if _HOST_OPT:
                # device-side aranges instead of torch.tensor(list) (a synchronous H2D copy)
                deltas = torch.cat([torch.arange(-N, 0, device=device, dtype=torch.int64),
                                    torch.arange(1, N + 1, device=device, dtype=torch.int64)])
            else:
                deltas = torch.tensor([d for d in range(-N, N + 1) if d != 0],
                                      device=device, dtype=torch.int64)
            shifted = center_tok.view(-1, 1) + deltas.view(1, -1) * frame_tokens   # [U, 2N]
            in_scale = (shifted >= 0) & (shifted < int(q_len))
            c_t = (int(sink_len) + shifted.clamp(0, max(int(q_len) - 1, 0))) // int(block_size)
            offs_t = torch.arange(-r_t, r_t + 1, device=device, dtype=torch.int64)
            cols_t = c_t.unsqueeze(-1) + offs_t.view(1, 1, -1)                     # [U, 2N, 2r_t+1]
            ok = in_scale.unsqueeze(-1) & (cols_t >= 0) & (cols_t < V)
            # Route every dropped column to a throwaway lane V so all scatter writes
            # are True -- masking by writing False would race with a valid True.
            cols_t = torch.where(ok, cols_t, torch.full_like(cols_t, V))
            tm = torch.zeros(U, V + 1, dtype=torch.bool, device=device)
            tm.scatter_(-1, cols_t.reshape(U, -1), True)
            return tm[:, :V]
        if _HOST_OPT:
            temporal_mask = _geom(('temporal', U, V, int(block_size), int(q_len), int(sink_len), frame_tokens, N, r_t, str(device)),
                                  _build_temporal)
        else:
            temporal_mask = _build_temporal()
        if os.environ.get('SPARVAR_DCSLA_DENSITY_LOG'):
            tm = temporal_mask.view(1, 1, U, V)
            fresh = (tm & ~mask).sum().item()
            total = int(tm.sum().item()) * mask.shape[0] * mask.shape[1]
            print(f'[dcsla-temporal] scale={scale_ind} layer={layer_ind} N={N} r_t={r_t} '
                  f'frame_tokens={frame_tokens} cols_per_row={float(tm.sum().item()) / max(U, 1):.2f} '
                  f'fresh_frac={fresh / max(total, 1):.4f}')
        mask |= temporal_mask.view(1, 1, U, V)

    if extra_block_mask is not None:
        # geometry-only force-include prior, precomputed per (schedule, scale)
        mask |= extra_block_mask.view(1, 1, U, V)

    kv_num_blocks = mask.sum(dim=-1, dtype=torch.int32)
    # Stable descending argsort of the 0/1 mask: selected ids first, each part
    # in ascending order -- exactly the layout from_kv_blocks expects.
    kv_indices = mask.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    return kv_indices, kv_num_blocks


def build_dcsla_block_mask(
    kv_indices: Tensor,                      # [B, H, U, V] int32 full-width
    kv_num_blocks: Tensor,                   # [B, H, U] int32
    q_len: int,
    kv_len: int,
    block_size: int = DCSLA_BLOCK,
) -> BlockMask:
    # All selected blocks go through the partial list (no full_kv_*): with
    # mask_mod=None (noop) the boundary masking is done by the compiled kernel
    # via seq_lengths (see module docstring). Triton backend only — the FLASH
    # backend silently computes DENSE on this encoding (see the FA4 builder).
    return BlockMask.from_kv_blocks(
        kv_num_blocks, kv_indices,
        BLOCK_SIZE=(int(block_size), int(block_size)),
        mask_mod=None,
        seq_lengths=(int(q_len), int(kv_len)),
        compute_q_blocks=_COMPUTE_Q_BLOCKS,
    )


# FA4 boundary mask_mod: ONE closure per (device, padded_kv_len) over a 1-D bool
# "valid" tensor of length padded_kv_len — the document-mask pattern the FLASH
# backend officially supports. Shape is bucket-constant (no recompile); CONTENT
# (True for kv_idx < real kv_len) is refreshed at runtime per prompt.
# Rejected alternatives: a captured Python int bakes a per-value dynamo guard
# (recompile per text length); a captured 0-dim CUDA scalar fails in the CuTe
# codegen ("Expected TensorSSA, but got <class 'int'>" — a 0-dim read lowers to
# a constant-int-indexed load, and the CuTe template only accepts TensorSSA
# indices; the 1-D lookup's index is kv_idx, a TensorSSA).
_fa4_boundary: Dict[tuple, dict] = {}


def _get_fa4_boundary_mod(device: torch.device, padded_kv_len: int, kv_len: int):
    key = (str(device), int(padded_kv_len))
    ent = _fa4_boundary.get(key)
    if ent is None:
        valid = torch.zeros(int(padded_kv_len), device=device, dtype=torch.bool)

        def boundary_mask_mod(b, h, q_idx, kv_idx):
            return valid[kv_idx]

        ent = {'valid': valid, 'mod': boundary_mask_mod, 'kv_len': -1}
        _fa4_boundary[key] = ent
    if ent['kv_len'] != int(kv_len):                 # refresh only when the prompt length changes
        ent['valid'].zero_()
        ent['valid'][:int(kv_len)] = True
        ent['kv_len'] = int(kv_len)
    return ent['mod']


def build_dcsla_block_mask_fa4(
    kv_indices: Tensor,                      # [B, H, U, V] int32 full-width (select_kv_blocks layout)
    kv_num_blocks: Tensor,                   # [B, H, U] int32
    q_len: int,
    kv_len: int,
    padded_kv_len: int = None,               # bucket-padded KV length (multiple of block_size, >= kv_len)
    block_size: int = DCSLA_BLOCK,
) -> BlockMask:
    """FA4 (CuTe) backend BlockMask: full/partial split + non-trivial boundary mask_mod.

    torch 2.11's FLASH backend (torch/_inductor/kernel/flex/flex_flash_attention.py)
    derives NEEDS_BLOCK_MASK solely from the mask_mod graph (is_trivial_mask_graph)
    and requires full_kv_* lists — on the partial-only mask_mod=None encoding above
    the CuTe kernel IGNORES the kv lists and computes dense attention. Encoding here:
    selected non-tail blocks -> full_kv_* (no per-element mask eval); the block
    holding the real ragged boundary -> partial list, masked by the 1-D valid-mask
    lookup ``valid[kv_idx]`` (see _get_fa4_boundary_mod; equivalent to
    ``kv_idx < kv_len``), which is also the non-trivial graph that flips
    NEEDS_BLOCK_MASK=True so the lists drive real block skipping. Geometry
    (lists + seq_lengths) is built at ``padded_kv_len``: pad blocks are never
    selected and the valid mask kills positions past the real ``kv_len``, so the
    padded mask is exact. Validated on H800 / torch 2.11 / fa4-b24
    (kernels/bench_dcsla_flex.py + SPARVAR_DCSLA_FA4_CHECK e2e): numerics ==
    triton backend, density-linear, ~1.6x faster at equal density.

    NB: the caller must pad K/V to ``padded_kv_len``; the valid tensor's content
    is refreshed here per real ``kv_len``.
    """
    B, H, U, V = kv_indices.shape
    device = kv_indices.device
    if padded_kv_len is None:
        padded_kv_len = _cdiv(int(kv_len), int(block_size)) * int(block_size)
    assert padded_kv_len % int(block_size) == 0 and padded_kv_len >= int(kv_len), \
        f'padded_kv_len {padded_kv_len} must be a block multiple >= kv_len {kv_len}'
    assert V == _cdiv(int(kv_len), int(block_size)), \
        f'kv_indices width {V} != ceil(kv_len={kv_len} / {block_size})'
    V_pad = padded_kv_len // int(block_size)

    # O(1) full/partial split. select_kv_blocks rows hold the selected ids as an
    # ASCENDING prefix, and the ragged tail block id (V-1) is the maximum, so a
    # selected tail can only sit at the END of the prefix: membership = one
    # gather of the prefix's last entry, and the full list reuses the prefix
    # as-is with the count decremented. The previous sel-scatter + two V_pad-wide
    # argsorts cost ~3ms/call at scale 29 (~22% of the FA4 sparse step, measured
    # via the idxmap timing A/B; today: LPROF=1 LPROF_SCOPES='dcsla.*').
    if V_pad > V:
        if _HOST_OPT:
            pad_ids = _geom(('pad_ids', V, V_pad, str(device)),
                            lambda: torch.arange(V, V_pad, device=device, dtype=torch.int32))
        else:
            pad_ids = torch.arange(V, V_pad, device=device, dtype=torch.int32)
        full_idx = torch.cat(
            [kv_indices, pad_ids.view(1, 1, 1, -1).expand(B, H, U, -1)], dim=-1)
    else:
        full_idx = kv_indices
    if int(kv_len) % int(block_size) != 0:
        last = kv_indices.gather(-1, (kv_num_blocks.long() - 1).clamp_min(0)
                                 .unsqueeze(-1)).squeeze(-1)
        tail_sel = ((last == V - 1) & (kv_num_blocks > 0)).to(torch.int32)   # [B,H,U]
        full_num = kv_num_blocks - tail_sel
        part_num = tail_sel
        # Static full-width permutation with the tail id first; rows with
        # part_num=0 never read it.

        def _build_part_idx():
            # arange-based tail id (device side; torch.tensor([V-1]) would be a sync H2D copy)
            return torch.cat([
                torch.arange(V - 1, V, device=device, dtype=torch.int32),
                torch.arange(V - 1, device=device, dtype=torch.int32),
                torch.arange(V, V_pad, device=device, dtype=torch.int32),
            ]).view(1, 1, 1, -1).expand(B, H, U, -1).contiguous()
        if _HOST_OPT:
            part_idx = _geom(('part_idx', B, H, U, V, V_pad, str(device)), _build_part_idx)
        else:
            part_idx = torch.cat([
                torch.tensor([V - 1], device=device, dtype=torch.int32),
                torch.arange(V - 1, device=device, dtype=torch.int32),
                torch.arange(V, V_pad, device=device, dtype=torch.int32),
            ]).view(1, 1, 1, -1).expand(B, H, U, -1).contiguous()
    else:
        full_num = kv_num_blocks
        part_num = torch.zeros_like(kv_num_blocks)
        part_idx = full_idx                       # ignored: every row has part_num=0

    mod = _get_fa4_boundary_mod(device, int(padded_kv_len), int(kv_len))

    return BlockMask.from_kv_blocks(
        part_num, part_idx,
        full_kv_num_blocks=full_num, full_kv_indices=full_idx,
        BLOCK_SIZE=(int(block_size), int(block_size)),
        mask_mod=mod,
        seq_lengths=(int(q_len), int(padded_kv_len)),
        compute_q_blocks=_COMPUTE_Q_BLOCKS,
    )


# -----------------------
# Compiled flex singleton + sparse-step entry point
# -----------------------

_compiled_flex_cache: Dict[tuple, object] = {}

# Set True after the first FA4 forward failure: every later call falls back to
# the triton backend instead of retrying (and re-raising) per layer.
_fa4_runtime_disabled = False


def fa4_runtime_ok() -> bool:
    """Process-wide FA4 kill-switch state, shared by D-CSLA and static CSLA."""
    return not _fa4_runtime_disabled


def disable_fa4(reason: str) -> None:
    """Permanently fall back to the triton flex backend for this process."""
    global _fa4_runtime_disabled
    _fa4_runtime_disabled = True
    dcsla_warn_once(reason)


def _get_compiled_flex(mode, dynamic: bool):
    """Lazy module-level singletons shared by all layers (never compile at
    import time; per-layer wrappers would each keep their own compile cache)."""
    mode = None if mode in (None, '', 'none', 'default', '~') else str(mode)
    key = (mode, bool(dynamic))
    fn = _compiled_flex_cache.get(key)
    if fn is None:
        kwargs = {'dynamic': bool(dynamic)}
        if mode is not None:
            kwargs['mode'] = mode
        fn = torch.compile(flex_attention, **kwargs)
        _compiled_flex_cache[key] = fn
    return fn


def _get_compiled_flex_flash():
    """FLASH (FA4/CuTe) backend singleton, compiled dynamic=False.

    With the _FA4_KV_BUCKET padding the shapes are constant per scale, so this
    compiles one CuTe template per sparse scale per process (~4.5s each).
    Defense-in-depth on dynamo's recompile limit: past the limit dynamo would
    SILENTLY run eager flex_attention, which materializes the dense score matrix
    (134.5 GiB at 720p scale 28 — the observed vbench-speed OOM). Raise the limit
    far above our bucket count and turn any residual limit hit into an exception,
    which the caller's try/except converts into a clean triton fallback.
    """
    fn = _compiled_flex_cache.get('fa4')
    if fn is None:
        from torch import _dynamo as torch_dynamo  # distinct name: a bare `import torch._dynamo` would shadow `torch` locally
        if hasattr(torch_dynamo.config, 'recompile_limit'):
            torch_dynamo.config.recompile_limit = max(int(torch_dynamo.config.recompile_limit), 64)
        if hasattr(torch_dynamo.config, 'fail_on_recompile_limit_hit'):
            torch_dynamo.config.fail_on_recompile_limit_hit = True
        fn = torch.compile(partial(flex_attention, kernel_options={"BACKEND": "FLASH"}),
                           dynamic=False)
        _compiled_flex_cache['fa4'] = fn
    return fn


def _maybe_check_fa4_vs_triton(o_fa4, q, k, v, scale, kv_indices, kv_num_blocks,
                               *, q_len: int, kv_len: int, block_size: int,
                               scale_ind, layer_ind, attn_config) -> None:
    """SPARVAR_DCSLA_FA4_CHECK=1: recompute the same selection through the triton
    backend and print rel_l2 — the sparse+GQA+FA4 numerics check the bench's small
    MHA shape cannot cover. Diagnostic runs only (doubles the attention cost);
    swallows its own errors so a probe failure never disables the FA4 path."""
    if os.environ.get("SPARVAR_DCSLA_FA4_CHECK", "0") in ("0", "", "false", "False"):
        return
    try:
        bm = build_dcsla_block_mask(kv_indices, kv_num_blocks, q_len, kv_len, block_size)
        flex_tri = _get_compiled_flex(attn_config.get('flex_mode', 'max-autotune-no-cudagraphs'),
                                      bool(attn_config.get('flex_dynamic', True)))
        o_ref = flex_tri(q.to(v.dtype), k.to(v.dtype), v, block_mask=bm, scale=scale, enable_gqa=True)
        num = torch.linalg.norm((o_fa4 - o_ref).float())
        den = torch.linalg.norm(o_ref.float()).clamp_min(1e-12)
        print(f"[dcsla-fa4-check] scale={scale_ind} layer={int(layer_ind):02d} "
              f"rel_l2={(num / den).item():.3e}")
    except Exception as e:  # noqa: BLE001
        print(f"[dcsla-fa4-check] scale={scale_ind} layer={int(layer_ind):02d} "
              f"ERR {str(e).splitlines()[0][:80]}")


@torch.no_grad()
def dcsla_sparse_attention(
    q: Tensor, k: Tensor, v: Tensor, scale,
    mass: Tensor,                            # [B, H, G_S, V] bf16, already batch-matched
    *,
    sink_len: int,
    scale_ind: int,
    layer_ind: int,
    attn_config: dict,
    scale_shape=None,                        # optional (T, H, W): viz dump + temporal locality prior
    image_scale_schedule=None,               # Infinity (image): stacked-pyramid KV geometry; None = video layout
) -> Tensor:
    """Sparse-step forward: phi-expand mass rows -> select blocks -> BlockMask
    -> compiled block-sparse flex_attention. Returns [B, H, L, D] in v.dtype."""
    scale = None if scale is None else float(scale)
    block_size = int(attn_config.get('flex_block_size', DCSLA_BLOCK))
    B, H, q_len, _ = q.shape
    kv_len = int(k.shape[-2])
    G_S = int(mass.shape[2])
    V = int(mass.shape[-1])
    U = _cdiv(int(q_len), block_size)
    assert mass.shape[0] == B and mass.shape[1] == H, \
        f'D-CSLA mass shape {tuple(mass.shape)} does not match q {tuple(q.shape)}'

    # lprof segments: select = phi expand + block selection; build = BlockMask
    # construction; fwd = compiled flex forward (env-gated probes/viz between
    # the marks land in the enclosing segment — they are off by default).
    _t = lprof.timer('dcsla', si=scale_ind, layer=layer_ind)

    # phi: center-NN remap of 128-row target q-blocks onto the decision-scale
    # colsum groups (dcsla_colsum_group_rows wide; G_S is read from the mass
    # shape, same formula as build_footprint_union_indices).
    def _build_phi():
        u_centers = torch.arange(U, device=mass.device, dtype=torch.float32) + 0.5
        return torch.round(u_centers / U * G_S - 0.5).to(torch.int64).clamp_(0, G_S - 1)
    phi = _geom(('phi', U, G_S, str(mass.device)), _build_phi) if _HOST_OPT else _build_phi()
    mass_rows = mass.index_select(2, phi).float()                        # [B,H,U,V]

    # Image (Infinity): the current scale sits at the END of the stacked pyramid,
    # so the diagonal band anchors at its cumulative offset instead of sink_len,
    # and the locality prior is the CSLA cross-scale band (win_size per historical
    # scale) rather than the video's intra-frame stripe + temporal shifts.
    diag_offset = None
    extra_block_mask = None
    if image_scale_schedule is not None:
        n_per = [int(s[-2]) * int(s[-1]) for s in image_scale_schedule[:int(scale_ind) + 1]]
        diag_offset = sum(n_per[:-1])
        win = attn_config.get('dcsla_image_win_size') or None
        if win:
            extra_block_mask = build_image_cross_scale_band_mask(
                image_scale_schedule, int(scale_ind), int(q_len), kv_len, win,
                mass.device, block_size=block_size)
    elif scale_shape is not None:
        # Video (V-CSLA W-A): full-frame band prior, boundary-shifted like the
        # CS4A frame band. Budget-zero recipe -> static V-CSLA; otherwise it is
        # OR'd with the dynamic selection (additive, like local_diag_blocks).
        vcsla_rho = int(attn_config.get('dcsla_vcsla_frame_band_rho', -1))
        t_s, h_s, w_s = (int(x) for x in scale_shape)
        if vcsla_rho >= 0 and t_s * h_s * w_s == int(q_len):
            extra_block_mask = build_video_frame_band_mask(
                scale_shape, q_len=int(q_len), kv_len=kv_len,
                sink_len=int(sink_len), block_size=block_size,
                rho=vcsla_rho, device=mass.device)

    kv_indices, kv_num_blocks = select_kv_blocks(
        mass_rows,
        sink_len=int(sink_len), q_len=int(q_len), kv_len=kv_len, block_size=block_size,
        mode=str(attn_config.get('dcsla_select_mode', 'coverage')),
        tau=float(attn_config.get('dcsla_tau', 0.9)),
        topk_ratio=float(attn_config.get('dcsla_topk_ratio', 0.2)),
        min_blocks=int(attn_config.get('dcsla_min_blocks', 16)),
        max_blocks_ratio=float(attn_config.get('dcsla_max_blocks_ratio', 0.5)),
        local_diag_blocks=int(attn_config.get('dcsla_local_diag_blocks', 1)),
        keep_sink=bool(attn_config.get('dcsla_keep_sink', True)),
        sink_select_tau=float(attn_config.get('dcsla_sink_select_tau', 0.0)),
        scale_shape=scale_shape,
        temporal_neighbor_frames=int(attn_config.get('dcsla_temporal_neighbor_frames', 0)),
        temporal_diag_blocks=int(attn_config.get('dcsla_temporal_diag_blocks', 0)),
        diag_offset=diag_offset, extra_block_mask=extra_block_mask,
        scale_ind=scale_ind, layer_ind=layer_ind,
    )
    _t.mark('select')
    # Env-gated (SPARVAR_KBENCH_DUMP): real block selection for the kernel-speed bench.
    maybe_dump_dcsla_pattern(kv_indices, kv_num_blocks, scale_ind=scale_ind,
                             layer_ind=layer_ind, q=q, k=k, scale=scale,
                             sink_len=int(sink_len), block_size=block_size,
                             is_image=image_scale_schedule is not None)
    # Env-gated real-density probe: the achieved block density under the live
    # coverage/topk policy on REAL colsum mass (unknown from the synthetic bench).
    # One line per (scale, layer); aggregate per scale offline. sink_blocks is the
    # ctx-prefix floor, so density - sink_blocks/V is the video-selected fraction.
    if os.environ.get('SPARVAR_DCSLA_DENSITY_LOG'):
        knb = kv_num_blocks.float()
        sink_blocks = _cdiv(max(int(sink_len), 0), block_size)
        print(f'[dcsla-density] scale={scale_ind} layer={layer_ind} '
              f'density={knb.mean().item() / max(V, 1):.4f} '
              f'max={knb.max().item() / max(V, 1):.4f} '
              f'blocks_mean={knb.mean().item():.1f} V={V} sink_blocks={sink_blocks}')

    # Env-gated (SPARVAR_DCSLA_DENSITY_LOG) real-density probe: where does
    # coverage tau land on real attention mass? Pure logging.
    maybe_log_dcsla_density(kv_num_blocks, V=V, scale_ind=scale_ind,
                            layer_ind=layer_ind, attn_config=attn_config)
    use_fa4 = (str(attn_config.get('flex_backend', 'triton')).lower() == 'fa4'
               and fa4_runtime_ok())
    if use_fa4:
        # Bucket-pad the KV geometry so per-prompt text-length variation maps to
        # one static shape per scale (one CuTe compile, no per-prompt recompiles).
        padded_kv_len = _cdiv(kv_len, _FA4_KV_BUCKET) * _FA4_KV_BUCKET
        block_mask = build_dcsla_block_mask_fa4(kv_indices, kv_num_blocks, int(q_len), kv_len,
                                                padded_kv_len=padded_kv_len, block_size=block_size)
    else:
        block_mask = build_dcsla_block_mask(kv_indices, kv_num_blocks, int(q_len), kv_len, block_size)
    _t.note(backend='fa4' if use_fa4 else 'triton')
    _t.mark('build')

    if viz_enabled_for(scale_ind, layer_ind):
        # Image: per-scale KV token boundaries after the sink (divider lines +
        # x tick labels). Video (image_scale_schedule=None) keeps default ticks.
        kv_bounds = (image_kv_scale_boundaries(image_scale_schedule, kv_len, int(sink_len))
                     if image_scale_schedule is not None else None)
        # Evidence + result: phi-remapped block mass heatmap with the FINAL
        # selection outlined, color-coded by origin (mass prefix / forced /
        # cut-by-cap). SPARVAR_VIZ_MASS_SEL=0 disables just this renderer.
        dump_dcsla_mass_selection_viz(
            mass_rows, kv_indices, kv_num_blocks, scale_ind=scale_ind,
            layer_ind=layer_ind, q_len=int(q_len), kv_len=kv_len,
            block_size=block_size, sink_len=int(sink_len),
            attn_config=attn_config, scale_shape=scale_shape,
            kv_scale_boundaries=kv_bounds, tag="dcsla")
        # Detailed view: full token-coordinate mask + red dashed block grid
        # (SPARVAR_VIZ_BLOCK_GRID=0 disables just this renderer).
        dump_dcsla_block_mask_viz(
            kv_indices, kv_num_blocks, scale_ind=scale_ind, layer_ind=layer_ind,
            q_len=int(q_len), kv_len=kv_len, block_size=block_size,
            sink_len=int(sink_len), tag="dcsla", scale_shape=scale_shape,
            kv_scale_boundaries=kv_bounds)

    if use_fa4:
        try:
            k_f, v_f = k.to(v.dtype), v
            if padded_kv_len != kv_len:      # bucket pad; masked out by the boundary tensor
                k_f = F.pad(k_f, (0, 0, 0, padded_kv_len - kv_len))
                v_f = F.pad(v_f, (0, 0, 0, padded_kv_len - kv_len))
            o = _get_compiled_flex_flash()(q.to(v.dtype), k_f, v_f,
                                           block_mask=block_mask, scale=scale, enable_gqa=True)
            _maybe_check_fa4_vs_triton(o, q, k, v, scale, kv_indices, kv_num_blocks,
                                       q_len=int(q_len), kv_len=kv_len, block_size=block_size,
                                       scale_ind=scale_ind, layer_ind=layer_ind,
                                       attn_config=attn_config)
            _t.stop('fwd')
            return o
        except Exception as e:  # noqa: BLE001
            # One warning, permanent triton fallback: an env/build drift must not
            # kill the whole run 36 layers deep.
            disable_fa4('[D-CSLA] FA4 backend failed, falling back to triton flex: '
                        + str(e).splitlines()[0][:120])
            _t.note(backend='fa4_fallback')  # mask rebuild + triton fwd land in 'fwd'
            block_mask = build_dcsla_block_mask(kv_indices, kv_num_blocks, int(q_len), kv_len, block_size)

    flex = _get_compiled_flex(attn_config.get('flex_mode', 'max-autotune-no-cudagraphs'),
                              bool(attn_config.get('flex_dynamic', True)))
    o = flex(q.to(v.dtype), k.to(v.dtype), v, block_mask=block_mask, scale=scale, enable_gqa=True)
    _t.stop('fwd')
    return o


if __name__ == '__main__':
    pass
