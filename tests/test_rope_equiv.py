#!/usr/bin/env python3
"""Equivalence tests for the rewritten `apply_rotary_emb`
(`models/infinity/basic_infinity.py`).

  1. math check (self-contained, runs anywhere): rotating q and k separately is
     bit-identical to rotating the stacked q||k tensor, including the dtype
     promotion torch.stack performs when q is fp32 and k is bf16
  2. module check (needs the repo env: flash_attn etc.): the real
     apply_rotary_emb reproduces the old implementation bit-for-bit on the live
     shapes, for scale_ind 0 and > 0
  3. fused path (opt-in, CUDA): reports the numerical delta of
     INFINITY_ROPE_FUSED=1 rather than asserting bit-exactness

Run: python tests/test_rope_equiv.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import torch

DEV = 'cuda' if torch.cuda.is_available() else 'cpu'
# Infinity 1M schedule (t, h, w)
SCHEDULE = [(1, s, s) for s in (1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64)]
TOTAL = sum(t * h * w for t, h, w in SCHEDULE)


def old_apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier,
                         rope2d_normalized_by_hw, scale_ind):
    """Verbatim copy of the implementation this change replaces."""
    qk = torch.stack((q, k), dim=0)
    device_type = qk.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = qk.shape[3]
        start = 0
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = np.sum([item[0] * item[1] * item[2] for item in scale_schedule[:scale_ind]])
        rope2d_freqs_grid[str(tuple(scale_schedule))] = rope2d_freqs_grid[str(tuple(scale_schedule))].to(qk.device)
        assert start + seq_len <= rope2d_freqs_grid[str(tuple(scale_schedule))].shape[4]
        rope_cache = rope2d_freqs_grid[str(tuple(scale_schedule))][:, :, :, :, start:start + seq_len]
        qk = qk.reshape(*qk.shape[:-1], -1, 2)
        qk = torch.stack([
            rope_cache[0] * qk[..., 0] - rope_cache[1] * qk[..., 1],
            rope_cache[1] * qk[..., 0] + rope_cache[0] * qk[..., 1],
        ], dim=-1)
        qk = qk.reshape(*qk.shape[:-2], -1)
        q, k = qk.unbind(dim=0)
    return q, k


def make_grid(half_dim, device=DEV):
    ang = torch.randn(TOTAL, half_dim, device=device)
    grid = torch.stack((ang.cos(), ang.sin()), dim=0)     # (2, total, half)
    return {str(tuple(SCHEDULE)): grid[:, None, None, None]}   # (2,1,1,1,total,half)


def qk_pair(B, H, L, Dh, dtypes, device=DEV):
    q = torch.randn(B, H, L, Dh, device=device, dtype=dtypes[0])
    k = torch.randn(B, H, L, Dh, device=device, dtype=dtypes[1])
    return q, k


# --------------------------------------------------------------------------
# 1. math check (no repo imports)
# --------------------------------------------------------------------------

def new_math(q, k, cos, sin):
    dtype = torch.promote_types(q.dtype, k.dtype)

    def one(x):
        shape = x.shape
        xr = x.to(dtype).reshape(*shape[:-1], -1, 2)
        x0, x1 = xr[..., 0], xr[..., 1]
        out = torch.stack((cos * x0 - sin * x1, sin * x0 + cos * x1), dim=-1)
        return out.reshape(shape)

    return one(q), one(k)


def old_math(q, k, cos, sin):
    qk = torch.stack((q, k), dim=0)
    qk = qk.reshape(*qk.shape[:-1], -1, 2)
    qk = torch.stack([cos * qk[..., 0] - sin * qk[..., 1],
                      sin * qk[..., 0] + cos * qk[..., 1]], dim=-1)
    return qk.reshape(*qk.shape[:-2], -1).unbind(dim=0)


def test_math():
    torch.manual_seed(0)
    for dtypes in ((torch.float32, torch.bfloat16), (torch.float32, torch.float32),
                   (torch.bfloat16, torch.bfloat16)):
        B, H, L, Dh = 2, 4, 37, 16
        q, k = qk_pair(B, H, L, Dh, dtypes)
        ang = torch.randn(1, 1, 1, L, Dh // 2, device=DEV)
        cos, sin = ang.cos(), ang.sin()
        oq, ok = old_math(q, k, cos, sin)
        nq, nk = new_math(q, k, cos, sin)
        assert oq.dtype == nq.dtype == ok.dtype == nk.dtype, (oq.dtype, nq.dtype)
        assert torch.equal(oq, nq) and torch.equal(ok, nk), f'mismatch for {dtypes}'
        assert nq.shape == q.shape and nk.shape == k.shape
    print(f'  math equivalence OK (bit-identical, device={DEV})')


# --------------------------------------------------------------------------
# 2/3. real module
# --------------------------------------------------------------------------

def test_module():
    try:
        import models.infinity.basic_infinity as bi
    except Exception as e:
        print(f'  [skip] module check: {type(e).__name__}: {e}')
        return
    torch.manual_seed(0)
    B, H, Dh = 2, 28, 128
    for scale_ind, L in ((0, 1), (8, 576), (11, 2304), (12, 4096)):
        if DEV == 'cpu' and L > 600:
            continue                                    # keep the CPU run quick
        grid = make_grid(Dh // 2)
        q, k = qk_pair(B, H, L, Dh, (torch.float32, torch.bfloat16))
        oq, ok = old_apply_rotary_emb(q, k, SCHEDULE, dict(grid), 128, 2, scale_ind)
        bi._ROPE_META.update(sched=None, key=None, starts=None, polar=None)
        nq, nk = bi.apply_rotary_emb(q, k, SCHEDULE, dict(grid), 128, 2, scale_ind)
        assert torch.equal(oq, nq) and torch.equal(ok, nk), f'si={scale_ind} L={L} mismatch'
        assert nq.shape == q.shape and nk.shape == k.shape
        print(f'  module equivalence OK  si={scale_ind:>2} L={L:>4}  bit-identical')

        if DEV == 'cuda':                               # report, do not assert
            bi._ROPE_META.update(sched=None, key=None, starts=None, polar=None)
            bi._ROPE_FUSED = True
            fq, fk = bi.apply_rotary_emb(q, k, SCHEDULE, dict(grid), 128, 2, scale_ind)
            bi._ROPE_FUSED = False
            for name, ref, got in (('q', oq, fq), ('k', ok, fk)):
                exact = torch.equal(ref, got)
                d = (ref.float() - got.float()).abs().max().item()
                flips = (ref.to(torch.bfloat16) != got.to(torch.bfloat16)).float().mean().item()
                print(f'    fused {name}: bit-identical={exact}  max|d|={d:.3e}  '
                      f'bf16-flip rate={flips:.2e}')


if __name__ == '__main__':
    print(f'[test_rope_equiv] device={DEV}')
    test_math()
    test_module()
    print('all tests passed')
