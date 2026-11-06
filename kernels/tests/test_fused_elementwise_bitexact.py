"""Bitwise check: fused Triton SiLU*up and RoPE vs the eager InfinityStar ops (remote GPU).

  CUDA_VISIBLE_DEVICES=7 python kernels/tests/test_fused_elementwise_bitexact.py
"""
import os
import sys

os.environ['INFINITYSTAR_FUSED_SWIGLU'] = '1'
os.environ['INFINITYSTAR_FUSED_ROPE'] = '1'
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402

from models.infinitystar.fused_elementwise_utils import (fused_rope_inplace, fused_silu_mul,  # noqa: E402
                                                          fused_rope_enabled, fused_swiglu_enabled)
from models.infinitystar.rope import apply_rotary_emb  # noqa: E402


def check_silu_mul():
    assert fused_swiglu_enabled(), 'triton unavailable'
    g = torch.Generator(device='cuda').manual_seed(1)
    n_bad = 0
    cases = []
    for rows, cols in [(1, 16384), (7, 16384), (2 * 144, 16384), (4096, 16384), (72000, 16384), (31800, 16384)]:
        for scale in (0.02, 0.5, 3.0, 40.0):
            for dt in (torch.bfloat16, torch.float32):
                gate = (torch.randn(rows, cols, device='cuda', generator=g) * scale).to(dt)
                up = (torch.randn(rows, cols, device='cuda', generator=g) * scale).to(dt)
                ref = F.silu(gate) * up
                out = fused_silu_mul(gate, up)
                eq = torch.equal(ref, out)
                if not eq:
                    n_bad += 1
                    diff = (ref.float() - out.float()).abs()
                    cases.append((rows, cols, scale, str(dt), int((ref != out).sum().item()), diff.max().item()))
    # special values
    gate = torch.tensor([0.0, -0.0, 1e-3, -1e-3, 80.0, -80.0, 200.0, -200.0, float('inf'), float('-inf')],
                        device='cuda').to(torch.bfloat16).repeat(2048)
    up = torch.linspace(-3, 3, gate.numel(), device='cuda').to(torch.bfloat16)
    ref = F.silu(gate) * up
    out = fused_silu_mul(gate, up)
    eq_special = torch.equal(ref, out) or (torch.isnan(ref).equal(torch.isnan(out)) and torch.equal(ref[~torch.isnan(ref)], out[~torch.isnan(out)]))
    print(f'[silu*mul] mismatching cases: {n_bad} ; special values identical: {eq_special}')
    for c in cases[:10]:
        print('   mismatch rows=%d cols=%d scale=%.2f dtype=%s n_diff=%d maxdiff=%.3e' % c)
    return n_bad == 0 and eq_special


def check_rope():
    assert fused_rope_enabled(), 'triton unavailable'
    g = torch.Generator(device='cuda').manual_seed(2)
    n_bad = 0
    for B, H, Hkv, L in [(2, 32, 8, 1), (2, 32, 8, 144), (2, 32, 8, 3600), (1, 32, 8, 72000), (1, 32, 8, 31800), (2, 32, 8, 20640)]:
        D = 128
        for dt in (torch.bfloat16, torch.float32):
            # eager layout: [B, L, H, D] projections viewed/transposed to [B, H, L, D]
            qb = (torch.randn(B, L, H, D, device='cuda', generator=g) * 2.0).to(dt)
            kb = (torch.randn(B, L, Hkv, D, device='cuda', generator=g) * 2.0).to(dt)
            ang = torch.rand(L, D // 2, device='cuda', generator=g) * 6.2832
            rope = torch.stack([torch.cos(ang), torch.sin(ang)], 0).view(2, 1, 1, 1, L, D // 2).contiguous()
            for tag, (qq, kk, rr) in {'normal': (qb, kb, rope),
                                      'denormal': ((qb.float() * 1e-20).to(dt), (kb.float() * 1e-20).to(dt),
                                                   (rope * torch.tensor([1.0, 1e-20], device='cuda').view(2, 1, 1, 1, 1, 1)).contiguous())}.items():
                q1, k1 = qq.clone().transpose(1, 2), kk.clone().transpose(1, 2)
                q2, k2 = qq.clone().transpose(1, 2), kk.clone().transpose(1, 2)
                rq, rk = apply_rotary_emb(q1, k1, rr)
                fq, fk = fused_rope_inplace(q2, k2, rr)
                eq = torch.equal(rq, fq) and torch.equal(rk, fk)
                if not eq:
                    n_bad += 1
                    dq = (rq.float() - fq.float()).abs().max().item()
                    dk = (rk.float() - fk.float()).abs().max().item()
                    print(f'   mismatch[{tag}] B={B} L={L} dtype={dt}: n_diff_q={(rq != fq).sum().item()} maxdiff q={dq:.3e} k={dk:.3e}')
    print(f'[rope] mismatching cases: {n_bad}')
    return n_bad == 0


def bench():
    import time
    L, H, Hkv, D = 72000, 32, 8, 128
    q = torch.randn(1, L, H, D, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    k = torch.randn(1, L, Hkv, D, device='cuda', dtype=torch.bfloat16).transpose(1, 2)
    ang = torch.rand(L, D // 2, device='cuda') * 6.28
    rope = torch.stack([torch.cos(ang), torch.sin(ang)], 0).view(2, 1, 1, 1, L, D // 2).contiguous()
    for name, fn in (('eager rope', lambda: apply_rotary_emb(q, k, rope)), ('fused rope', lambda: fused_rope_inplace(q, k, rope))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print(f'[bench] {name}: {(time.perf_counter() - t) / 20 * 1e3:.3f} ms  (si29 shapes)')
    gate = torch.randn(L, 16384, device='cuda', dtype=torch.bfloat16)
    up = torch.randn(L, 16384, device='cuda', dtype=torch.bfloat16)
    for name, fn in (('eager silu*mul', lambda: F.silu(gate) * up), ('fused silu*mul', lambda: fused_silu_mul(gate, up))):
        for _ in range(3):
            fn()
        torch.cuda.synchronize(); t = time.perf_counter()
        for _ in range(20):
            fn()
        torch.cuda.synchronize()
        print(f'[bench] {name}: {(time.perf_counter() - t) / 20 * 1e3:.3f} ms  (si29 shapes)')


if __name__ == '__main__':
    ok1 = check_silu_mul()
    ok2 = check_rope()
    bench()
    print('RESULT', 'PASS' if (ok1 and ok2) else 'FAIL', f'silu_mul={ok1} rope={ok2}')
