"""V-CSLA: static frame-band block prior for InfinityStar video sparse scales.

Extends the conference static CSLA idea (sink protection + local window) to
video, at D-CSLA block granularity: each query block keeps the KV blocks of its
own frame plus the +/-rho neighbouring FULL frames (W-A scheme of
docs/sparvar/CSLA/vcsla_static_video_csla_design.md).

Boundary handling matches CS4A's ``_get_band_mask``
(models/sparvar/index_map_utils.py): the temporal window is BOUNDARY-SHIFTED so
every query block covers exactly ``min(2*rho+1, T)`` CONTIGUOUS frames
(frame 0 -> {0..2*rho}, last frame -> {T-2*rho-1..T-1}), never truncated at the
clip edges.

The [U, V] mask is consumed through ``select_kv_blocks``'s ``extra_block_mask``
entry (OR'd after the mass selection), so the same builder serves both uses:
  * static V-CSLA: budget-zero recipe (topk_ratio=0 + min_blocks=0 +
    keep_sink=true) -> the mask IS the selection (sink + frame band);
  * locality prior: normal dynamic config -> dynamic mass selection UNION the
    frame band, additive like ``dcsla_local_diag_blocks``.

Geometry only (content/layer/head independent), cached with a small LRU like
the CS4A band-mask / image cross-scale band caches.
"""

from collections import OrderedDict

import torch

_frame_band_cache: "OrderedDict[tuple, torch.Tensor]" = OrderedDict()
_FRAME_BAND_CACHE_MAX = 8


@torch.no_grad()
def build_video_frame_band_mask(
    scale_shape,               # (T, H, W) of the CURRENT video scale
    *,
    q_len: int,                # current-scale token count (== T*H*W)
    kv_len: int,               # ctx prefix + current scale
    sink_len: int,             # ctx prefix length in TOKENS
    block_size: int,
    rho: int,                  # temporal radius in FRAMES (window = min(2*rho+1, T))
    device: torch.device,
) -> torch.Tensor:
    """``[U, V]`` bool: per-query-block frame-band prior (W-A), sink excluded.

    Block quantization uses any-overlap semantics: the band's contiguous token
    range [tok0, tok1) marks every KV block it touches (floor start / ceil
    end), so frame boundaries that straddle a 128-block are safely
    over-included -- the same convention as the sink block rounding.

    Sink blocks are NOT encoded here (the run recipe's ``keep_sink=true`` /
    the sink policy owns them); the mask carries only the frame-band geometry.
    """
    T, Hs, Ws = (int(x) for x in scale_shape)
    hw = Hs * Ws
    q_len, kv_len = int(q_len), int(kv_len)
    block_size, rho = int(block_size), int(rho)
    U = (q_len + block_size - 1) // block_size
    V = (kv_len + block_size - 1) // block_size
    key = (T, Hs, Ws, q_len, kv_len, int(sink_len), block_size, rho, str(device))
    cached = _frame_band_cache.get(key)
    if cached is not None:
        _frame_band_cache.move_to_end(key)
        return cached

    mask = torch.zeros(U, V, dtype=torch.bool, device=device)
    if hw > 0 and T > 0 and U > 0 and rho >= 0:
        u = torch.arange(U, device=device, dtype=torch.int64)
        center = (u * block_size + block_size // 2).clamp_max(max(q_len - 1, 0))
        t_c = torch.div(center, hw, rounding_mode='trunc').clamp_(0, T - 1)
        W_frames = min(2 * rho + 1, T)
        w_start = (t_c - rho).clamp_(0, T - W_frames)              # contiguous window start
        tok0 = int(sink_len) + w_start * hw                        # [U] inclusive token start
        tok1 = int(sink_len) + (w_start + W_frames) * hw           # [U] exclusive token end
        b0 = torch.div(tok0, block_size, rounding_mode='trunc')
        b1 = torch.div(tok1 + block_size - 1, block_size, rounding_mode='trunc').clamp_(max=V)
        band_w = int((b1 - b0).max().item()) if U > 0 else 0
        cols = b0.view(U, 1) + torch.arange(band_w, device=device, dtype=torch.int64).view(1, -1)
        # Route out-of-band columns to a throwaway lane V so every scatter write
        # is True (same pattern as the temporal prior in select_kv_blocks).
        cols = torch.where(cols < b1.view(U, 1), cols, torch.full_like(cols, V))
        buf = torch.zeros(U, V + 1, dtype=torch.bool, device=device)
        buf.scatter_(-1, cols, True)
        mask = buf[:, :V].contiguous()

    if len(_frame_band_cache) >= _FRAME_BAND_CACHE_MAX:
        _frame_band_cache.popitem(last=False)
    _frame_band_cache[key] = mask
    return mask
