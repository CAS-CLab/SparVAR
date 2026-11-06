"""Env-gated real-inference sparse-pattern dump for the kernel-speed benchmark.

Feeds ``kernels/bench_kernel_speed_table.py`` (TPAMI tab:kernel_speed): the
benchmark replays each dumped per-layer pattern against FA2/FA3/sdpa+mask/
csp_attn/FlexAttention at the real last-scale shapes.

Env:
  SPARVAR_KBENCH_DUMP=<dir>     enable; pattern files are written into <dir>
  SPARVAR_KBENCH_SCALES=29      comma list of scale indices to dump (default: all)

One file per (tag, scale, layer); an existing file is never overwritten, so a
repeated scale (multi-repeat / cfg branches) keeps its first pattern only.

CS4A patterns are stored as a bit-packed boolean group mask (uint8, ~k_len/8
bytes per row) instead of the raw [B,H,G,k_len] int32 indices: the video
last-scale indices alone would be ~3.6 GB/layer, the packed mask is ~113 MB.
The benchmark rebuilds the exact csp_attn (inds, counts) layout from it.
"""
import os

import numpy as np
import torch

__all__ = ['maybe_dump_csp_pattern', 'maybe_dump_dcsla_pattern']


def _dump_dir():
    return os.environ.get('SPARVAR_KBENCH_DUMP', '')


def _scale_enabled(scale_ind) -> bool:
    raw = os.environ.get('SPARVAR_KBENCH_SCALES', '').strip()
    if not raw:
        return True
    try:
        wanted = {int(x) for x in raw.split(',') if x.strip()}
    except ValueError:
        return True
    return isinstance(scale_ind, int) and scale_ind in wanted


def _target_path(tag: str, scale_ind, layer_ind) -> str:
    return os.path.join(_dump_dir(), f'{tag}_si{scale_ind}_layer{layer_ind}.pt')


@torch.no_grad()
def maybe_dump_csp_pattern(inds, counts, *, scale_ind, layer_ind, q, k, scale,
                           sink_len, bm, multiple_of, tag):
    """Dump one layer's csp_attn pattern (batch 0) as a bit-packed group mask.

    ``inds``  [B,H,G,W] int32 (first counts[g] entries per row = selected KV ids),
    ``counts`` [B,H,G] int32. Zero-cost unless SPARVAR_KBENCH_DUMP is set.
    """
    if not _dump_dir() or not _scale_enabled(scale_ind):
        return
    path = _target_path(tag, scale_ind, layer_ind)
    if os.path.exists(path):
        return
    os.makedirs(_dump_dir(), exist_ok=True)

    B, H, G, W = (int(x) for x in inds.shape)
    k_len = int(k.shape[-2])
    ar = torch.arange(W, device=inds.device).view(1, W)
    packed = np.empty((H, G, (k_len + 7) // 8), dtype=np.uint8)
    for h in range(H):  # per-head loop bounds the int64/bool temporaries
        valid = ar < counts[0, h].view(G, 1)                       # [G, W]
        mask_h = torch.zeros((G, k_len), dtype=torch.bool, device=inds.device)
        mask_h.scatter_(1, inds[0, h].long().clamp_(0, k_len - 1), valid)
        packed[h] = np.packbits(mask_h.cpu().numpy(), axis=-1)

    torch.save({
        'kind': 'csp',
        'mask_bits': packed,                     # [H, G, ceil(k_len/8)] uint8
        'counts': counts[0].to('cpu', torch.int32).contiguous(),   # [H, G]
        'meta': {
            'tag': tag, 'scale_ind': int(scale_ind), 'layer_ind': int(layer_ind),
            'q_shape': tuple(int(x) for x in q.shape),
            'k_shape': tuple(int(x) for x in k.shape),
            'q_dtype': str(q.dtype), 'v_dtype': str(k.dtype),
            'inds_width': W, 'k_len': k_len, 'q_len': int(q.shape[-2]),
            'sink_len': int(sink_len), 'bm': int(bm),
            'multiple_of': int(multiple_of),
            'scale': None if scale is None else float(scale),
        },
    }, path)
    print(f'[kbench-dump] {os.path.basename(path)}: H={H} G={G} k_len={k_len} '
          f'density={counts[0].to(torch.float64).sum().item() / (H * G * k_len):.4f}')


@torch.no_grad()
def maybe_dump_dcsla_pattern(kv_indices, kv_num_blocks, *, scale_ind, layer_ind,
                             q, k, scale, sink_len, block_size, is_image):
    """Dump one layer's D-CSLA block selection (batch 0): kv lists as produced by
    ``select_kv_blocks`` (ascending prefix layout). Zero-cost unless enabled."""
    if not _dump_dir() or not _scale_enabled(scale_ind):
        return
    tag = 'dcsla_image' if is_image else 'dcsla_video'
    path = _target_path(tag, scale_ind, layer_ind)
    if os.path.exists(path):
        return
    os.makedirs(_dump_dir(), exist_ok=True)

    V = int(kv_indices.shape[-1])
    torch.save({
        'kind': 'dcsla',
        'kv_indices': kv_indices[0].to('cpu', torch.int32).contiguous(),      # [H, U, V]
        'kv_num_blocks': kv_num_blocks[0].to('cpu', torch.int32).contiguous(),  # [H, U]
        'meta': {
            'tag': tag, 'scale_ind': int(scale_ind), 'layer_ind': int(layer_ind),
            'q_shape': tuple(int(x) for x in q.shape),
            'k_shape': tuple(int(x) for x in k.shape),
            'q_dtype': str(q.dtype), 'v_dtype': str(k.dtype),
            'q_len': int(q.shape[-2]), 'k_len': int(k.shape[-2]),
            'sink_len': int(sink_len), 'block_size': int(block_size),
            'scale': None if scale is None else float(scale),
        },
    }, path)
    dens = kv_num_blocks[0].to(torch.float64).mean().item() / max(V, 1)
    print(f'[kbench-dump] {os.path.basename(path)}: U={int(kv_indices.shape[-2])} '
          f'V={V} block_density={dens:.4f}')
