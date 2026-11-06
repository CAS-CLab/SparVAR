"""Triton dense_colsum_attn correctness checks (independent of the chipmunk CUDA ops).

Verifies both the attention output `o` (vs SDPA) and the normalized column-sum `cs`
(vs colsum of the true softmax, grouped over bm=192 query rows).

Run:  python kernels/tests/test_dense_colsum_attn_triton.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch

from kernels.sparse_attn.triton import dense_attn, dense_colsum_attn

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

BM = 192
ATOL_O, RTOL_O = 2e-2, 2e-2
ATOL_CS, RTOL_CS = 3e-2, 3e-2


def reference_o_and_p(q, k, v, scale):
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale
    P = torch.softmax(logits, dim=-1)            # (B,H,Nq,Nk) fp32
    O = torch.matmul(P.to(v.dtype), v)           # match the kernel's bf16 P@V
    return O, P


def reference_blocksums(P, bm=BM):
    B, H, Nq, Nk = P.shape
    G = (Nq + bm - 1) // bm
    Bs = torch.zeros(B, H, G, Nk, dtype=torch.float32, device=P.device)
    for g in range(G):
        s, e = g * bm, min((g + 1) * bm, Nq)
        Bs[:, :, g, :] = P[:, :, s:e, :].sum(dim=-2)
    return Bs


def check(b, h, nq, nk, d, prev_from_dense=False):
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, nq, d)
    k = torch.randn(b, h, nk, d)
    v = torch.randn(b, h, nk, d)

    prev_lse = None
    if prev_from_dense:
        _, prev_lse = dense_attn(q, k, v, scale=scale)

    o, cs, (M, L) = dense_colsum_attn(q, k, v, prev_lse, scale=scale)

    O_ref, P_ref = reference_o_and_p(q, k, v, scale)
    cs_ref = reference_blocksums(P_ref, BM)

    G = (nq + BM - 1) // BM
    assert o.shape == q.shape
    assert cs.shape == (b, h, G, nk), f"cs shape {cs.shape} != {(b, h, G, nk)}"
    assert M.shape == (b, h, nq, 1) and L.shape == (b, h, nq, 1)

    o_diff = (o - O_ref).abs().max().item()
    cs_diff = (cs - cs_ref).abs().max().item()
    ok_o = torch.allclose(o, O_ref, atol=ATOL_O, rtol=RTOL_O)
    ok_cs = torch.allclose(cs, cs_ref, atol=ATOL_CS, rtol=RTOL_CS)
    print(f"[colsum] B{b} H{h} Nq{nq} Nk{nk} D{d} prev={'dense' if prev_from_dense else 'None'} "
          f"(Nq%192={nq % 192}, Nk%64={nk % 64}) o_diff={o_diff:.4f} ok_o={ok_o} "
          f"cs_diff={cs_diff:.4f} ok_cs={ok_cs}")
    assert ok_o, f"colsum o mismatch ({o_diff})"
    assert ok_cs, f"colsum cs mismatch ({cs_diff})"


def main():
    # self-attention, prev_lse derived internally; Nq spans %192 == 0 and != 0
    for nq in [512, 576, 777, 1152, 1600]:
        check(1, 24, nq, nq, 128, prev_from_dense=False)
    # explicit prev_lse path
    check(1, 24, 768, 768, 128, prev_from_dense=True)
    # cross-attention (Nk % 64 != 0)
    check(2, 3, 777, 1234, 64, prev_from_dense=False)
    check(1, 24, 1600, 4142, 128, prev_from_dense=False)
    print("\nAll dense_colsum_attn Triton checks passed.")


if __name__ == '__main__':
    main()
