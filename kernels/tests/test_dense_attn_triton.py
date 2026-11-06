"""Triton dense_attn correctness checks (independent of the chipmunk CUDA ops).

Run:  python kernels/tests/test_dense_attn_triton.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F

from kernels.sparse_attn.triton import dense_attn

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

ATOL, RTOL = 2e-2, 2e-2


def make_tensor(shape, is_contiguous):
    if is_contiguous:
        return torch.randn(shape)
    # build with a permuted layout so the tensor is non-contiguous
    b, h, n, d = shape
    return torch.randn(n, b, h, d).permute(1, 2, 0, 3)


def check(b, h, nq, nkv, d, is_contiguous=True):
    scale = 1.0 / math.sqrt(d)
    q = make_tensor((b, h, nq, d), is_contiguous)
    k = make_tensor((b, h, nkv, d), is_contiguous)
    v = make_tensor((b, h, nkv, d), is_contiguous)

    o, (M, L) = dense_attn(q, k, v, scale=scale)
    o_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

    assert o.shape == q.shape, f"o shape {o.shape} != {q.shape}"
    assert M.shape == (b, h, nq, 1), f"M shape {M.shape}"
    assert L.shape == (b, h, nq, 1), f"L shape {L.shape}"

    diff = (o - o_ref).abs().max().item()
    ok = torch.allclose(o, o_ref, atol=ATOL, rtol=RTOL)
    print(f"[dense] B{b} H{h} Nq{nq} Nkv{nkv} D{d} contig={is_contiguous} "
          f"(Nkv%64={nkv % 64}, Nq%192={nq % 192}) max_diff={diff:.4f} ok={ok}")
    assert ok, f"dense_attn output mismatch (max_diff={diff})"


def main():
    # self-attention, arbitrary lengths (incl. %64 != 0, %192 != 0), both layouts
    for n in [512, 777, 2048, 2385, 4096, 4592]:
        for contig in [True, False]:
            check(1, 24, n, n, 128, contig)
    # cross-attention (Q != KV)
    check(1, 24, 4096, 10521, 128, True)
    check(2, 24, 1600, 4142, 128, True)
    check(1, 24, 2048, 6425, 128, False)
    # smaller head dim
    check(2, 3, 333, 555, 64, True)
    print("\nAll dense_attn Triton checks passed.")


if __name__ == '__main__':
    main()
