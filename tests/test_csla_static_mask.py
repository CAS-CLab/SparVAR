#!/usr/bin/env python3
"""CPU tests for the modernized static CSLA path (models/sparvar/sparse_attn_csla.py).

Covers:
  1. bit-equivalence of the block-level mask construction against a token-level
     reimplementation of the legacy SparseFlexAttn Stage A+B (mask_mod + sink
     union + block-any reduction), on the small brute-force schedule and on the
     real 13-scale 1M schedule at si=10/11/12,
  2. sink-boundary block rounding (non-multiple boundaries like 121 and 265),
  3. the [U,V] bool -> FlexAttention kv-list layout round trip,
  4. laziness: constructing StaticCSLAAttn on CPU does no device/compile work,
  5. the flex_* config-key hard migration (BASE_CONFIG + both ymls).

    python tests/test_csla_static_mask.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import yaml

from models.sparvar import dcsla_utils as D
from models.sparvar import sparse_attn_csla as C
from kernels.sparse_attn.sparse_attn_config import BASE_CONFIG

DEV = torch.device('cpu')
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

SIDES_SMALL = [1, 2, 4, 6, 8, 12]
SCHED_SMALL = [(s, s) for s in SIDES_SMALL]
SIDES_FULL = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
SCHED_FULL = [(s, s) for s in SIDES_FULL]
WIN_FULL = [-1] * 8 + [1, 1, 3, 5, 7]          # the shipped attn.win_size


def _cum(sched):
    out = [0]
    for h, w in sched:
        out.append(out[-1] + h * w)
    return out


# --------------------------------------------------------------- legacy reference
def _legacy_reference(sched, K, win, sink_scale, block_size):
    """Token-level rebuild of the legacy SparseFlexAttn Stage A+B: the fine
    mask_mod (sink prefix OR centre-aligned per-scale band), materialized at
    [S_q, S_k] and block-reduced with any — i.e. basic_sparvar.py:116-180 as it
    was before the migration."""
    hs = [int(s[-2]) for s in sched[:K + 1]]
    ws = [int(s[-1]) for s in sched[:K + 1]]
    slice_idx = torch.tensor([0] + list(torch.cumsum(
        torch.tensor([h * w for h, w in zip(hs, ws)]), 0)), dtype=torch.long)
    S_k = int(slice_idx[-1])
    H_q, W_q = hs[K], ws[K]
    S_q = H_q * W_q
    sink_boundary = int(slice_idx[sink_scale]) if sink_scale > 0 else 0

    kv = torch.arange(S_k)
    l_of = torch.searchsorted(slice_idx, kv, right=True) - 1
    H_k = torch.tensor(hs)[l_of]
    W_k = torch.tensor(ws)[l_of]
    half = torch.tensor([int(win[l]) for l in range(K + 1)])[l_of] // 2
    rel = kv - slice_idx[l_of]
    kv_y, kv_x = rel // W_k, rel % W_k

    q = torch.arange(S_q)
    q_y, q_x = q // W_q, q % W_q
    cy = torch.round((q_y.float() / H_q).unsqueeze(1) * H_k.float())
    cx = torch.round((q_x.float() / W_q).unsqueeze(1) * W_k.float())
    fine = ((cy - kv_y.float()).abs() <= half) & ((cx - kv_x.float()).abs() <= half)
    fine |= (kv < sink_boundary).unsqueeze(0)

    U = (S_q + block_size - 1) // block_size
    V = (S_k + block_size - 1) // block_size
    buf = torch.zeros(U * block_size, V * block_size, dtype=torch.bool)
    buf[:S_q, :S_k] = fine
    return buf.view(U, block_size, V, block_size).any(3).any(1)


def _build(sched, K, win, sink_scale, block_size):
    D._image_band_cache.clear()
    return C.build_csla_block_mask_bool(sched, K, win, sink_scale, DEV, block_size=block_size)


# ----------------------------------------------------------------------- tests
def test_legacy_equivalence_small():
    win = [-1, -1, 1, 1, 3, 5]
    for K in (3, 4, 5):
        for sink in (0, 2, 3):
            ref = _legacy_reference(SCHED_SMALL, K, win, sink, block_size=8)
            got = _build(SCHED_SMALL, K, win, sink, block_size=8)
            assert torch.equal(got, ref), \
                f'small K={K} sink={sink}: {int((got ^ ref).sum())} blocks differ'
    print('[ok] small-schedule legacy equivalence (K x sink grid)')


def test_legacy_equivalence_full():
    # the real production points: si=10/11/12, block 128, shipped win/sink
    for K in (10, 11, 12):
        ref = _legacy_reference(SCHED_FULL, K, WIN_FULL, 5, block_size=128)
        got = _build(SCHED_FULL, K, WIN_FULL, 5, block_size=128)
        assert torch.equal(got, ref), \
            f'full si={K}: {int((got ^ ref).sum())} blocks differ'
        print(f'[ok] full si={K}: {int(ref.sum())}/{ref.numel()} blocks kept, '
              f'density={ref.float().mean():.3f}')


def test_token_mask_consistency():
    # viz token mask block-anys to EXACTLY the inference block mask
    for K in (10, 11, 12):
        fine = C.build_csla_token_mask_bool(SCHED_FULL, K, WIN_FULL, 5, DEV)
        S_q, S_k = fine.shape
        U, V = (S_q + 127) // 128, (S_k + 127) // 128
        buf = torch.zeros(U * 128, V * 128, dtype=torch.bool)
        buf[:S_q, :S_k] = fine
        blk = buf.view(U, 128, V, 128).any(3).any(1)
        got = _build(SCHED_FULL, K, WIN_FULL, 5, block_size=128)
        assert torch.equal(blk, got), \
            f'si={K}: token-mask block-any differs from the block builder'
    print('[ok] token mask (viz) block-anys to the inference block mask (si=10/11/12)')


def test_sink_rounding():
    cum = _cum(SCHED_FULL)
    # sink=5 -> boundary 121 (< 128): exactly block column 0 fully on
    got5 = _build(SCHED_FULL, 11, WIN_FULL, 5, block_size=128)
    got0 = _build(SCHED_FULL, 11, WIN_FULL, 0, block_size=128)
    assert cum[5] == 121 and got5[:, 0].all()
    assert torch.equal(got5[:, 1:], got0[:, 1:]), 'sink=5 must only change block column 0'
    # sink=6 -> boundary 265 (2 full blocks + a partial third): columns 0-2 on
    got6 = _build(SCHED_FULL, 11, WIN_FULL, 6, block_size=128)
    assert cum[6] == 265 and got6[:, :3].all()
    ref6 = _legacy_reference(SCHED_FULL, 11, WIN_FULL, 6, block_size=128)
    assert torch.equal(got6, ref6), 'sink=6 rounding differs from the token-level reference'
    print('[ok] sink rounding: boundary 121 -> 1 column, 265 -> 3 columns (ceil semantics)')


def test_kv_list_roundtrip():
    torch.manual_seed(0)
    for mask in (torch.rand(9, 23) < 0.3,
                 _build(SCHED_FULL, 12, WIN_FULL, 5, block_size=128)):
        kv_indices, kv_num_blocks = C.mask_uv_to_kv_lists(mask)
        U, V = mask.shape
        assert kv_indices.shape == (1, 1, U, V) and kv_indices.dtype == torch.int32
        assert kv_num_blocks.shape == (1, 1, U) and kv_num_blocks.dtype == torch.int32
        # scatter the prefix back to dense and compare
        back = torch.zeros(U, V, dtype=torch.bool)
        for u in range(U):
            n = int(kv_num_blocks[0, 0, u])
            ids = kv_indices[0, 0, u, :n].long()
            assert torch.equal(ids, ids.sort().values), f'row {u}: prefix not ascending'
            back[u, ids] = True
        assert torch.equal(back, mask), 'kv-list round trip lost blocks'
    print('[ok] kv-list layout round trip (int32, ascending selected prefix)')


def test_lazy_construction():
    cfg = {'attn_sink': 5, 'win_size': WIN_FULL, 'csla_scales': [10, 11, 12],
           'flex_backend': 'fa4', 'flex_block_size': 128,
           'flex_mode': 'max-autotune-no-cudagraphs', 'flex_dynamic': True}
    lst = C.build_csla_spattn_list(SCHED_FULL, cfg, num_scales=len(SCHED_FULL))
    assert [i for i, fn in enumerate(lst) if fn is not None] == [10, 11, 12]
    fn = lst[12]
    assert fn._device is None and fn._bm_tri is None and fn._bm_fa4 is None, \
        'construction must not build any device state'
    assert fn.S_q == 4096 and fn.S_kv == 10521 and fn.sink_boundary == 121
    print('[ok] builder list + lazy construction (no device/compile work at init)')


def test_config_migration():
    old = ('dcsla_backend', 'dcsla_block_size', 'dcsla_flex_mode', 'dcsla_flex_dynamic')
    new = ('flex_backend', 'flex_block_size', 'flex_mode', 'flex_dynamic')
    a = BASE_CONFIG['attn']
    assert all(k in a for k in new) and not any(k in a for k in old)
    assert 'csla_scales' in a
    for yml in ('models/sparvar/sparse-attn-config.yml',
                'models/sparvar/sparse-attn-video-config.yml'):
        with open(os.path.join(REPO, yml), encoding='utf-8') as f:
            attn = yaml.safe_load(f)['attn']
        assert not any(k in attn for k in old), f'{yml} still sets old keys'
    # default backends: image = triton (fa4 has no image-side gain), video = fa4
    with open(os.path.join(REPO, 'models/sparvar/sparse-attn-config.yml'), encoding='utf-8') as f:
        assert yaml.safe_load(f)['attn'].get('flex_backend') == 'triton', \
            'image yml: flex_backend != triton'
    with open(os.path.join(REPO, 'models/sparvar/sparse-attn-video-config.yml'), encoding='utf-8') as f:
        assert yaml.safe_load(f)['attn'].get('flex_backend') == 'fa4', \
            'video yml: flex_backend != fa4'
    print('[ok] flex_* hard migration (BASE_CONFIG + both ymls)')


if __name__ == '__main__':
    test_legacy_equivalence_small()
    test_legacy_equivalence_full()
    test_token_mask_consistency()
    test_sink_rounding()
    test_kv_list_roundtrip()
    test_lazy_construction()
    test_config_migration()
    print('\nstatic CSLA mask tests passed')
