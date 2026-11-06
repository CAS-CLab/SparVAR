"""Triton csp_attn (column-sparse attention) correctness checks.

Two references:
  * full indices (every key selected) must equal dense SDPA;
  * random selected keys must equal a masked-softmax reference.

Run:  python kernels/tests/test_csp_attn_triton.py
"""
import os
import sys
import math

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

import torch
import torch.nn.functional as F

from kernels.sparse_attn.triton import csp_attn

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

BM = 192
ATOL, RTOL = 2e-2, 2e-2


def make_tensor(shape, is_contiguous):
    if is_contiguous:
        return torch.randn(shape)
    b, h, n, d = shape
    return torch.randn(n, b, h, d).permute(1, 2, 0, 3)


def reference_sparse_o(q, k, v, indices, counts, scale, bm=BM):
    B, H, Nq, D = q.shape
    Nk = k.shape[-2]
    G = (Nq + bm - 1) // bm
    logits = torch.matmul(q.float(), k.float().transpose(-2, -1)) * scale  # (B,H,Nq,Nk)
    mask = torch.zeros(B, H, Nq, Nk, dtype=torch.bool, device=q.device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                s, e = g * bm, min((g + 1) * bm, Nq)
                c = int(counts[b, h, g].item())
                sel = indices[b, h, g, :c].long()
                mask[b, h, s:e, sel] = True
    logits = torch.where(mask, logits, torch.full_like(logits, float("-inf")))
    P = torch.softmax(logits, dim=-1)
    return torch.matmul(P.to(v.dtype), v)


def check_full(b, h, n, d, is_contiguous=True):
    """Full indices (every key selected) must equal dense SDPA."""
    scale = 1.0 / math.sqrt(d)
    q = make_tensor((b, h, n, d), is_contiguous)
    k = make_tensor((b, h, n, d), is_contiguous)
    v = make_tensor((b, h, n, d), is_contiguous)
    n_groups = (n + BM - 1) // BM
    indices = torch.arange(n, dtype=torch.int32).repeat((b, h, n_groups, 1)).contiguous()
    counts = torch.full((b, h, n_groups), n, dtype=torch.int32)

    o, (M, L) = csp_attn(q, k, v, indices, counts, 1.0, scale=scale)
    o_ref = F.scaled_dot_product_attention(q, k, v, scale=scale)

    assert o.shape == q.shape
    diff = (o - o_ref).abs().max().item()
    ok = torch.allclose(o, o_ref, atol=ATOL, rtol=RTOL)
    print(f"[csp-full] B{b} H{h} N{n} D{d} contig={is_contiguous} "
          f"(N%64={n % 64}, N%192={n % 192}) max_diff={diff:.4f} ok={ok}")
    assert ok, f"csp full-index mismatch ({diff})"


def check_sparse(b, h, nq, nk, d, num_sel, seed=0):
    """Random selected keys must match the masked-softmax reference."""
    g = torch.Generator(device='cuda').manual_seed(seed)
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, nq, d, generator=g)
    k = torch.randn(b, h, nk, d, generator=g)
    v = torch.randn(b, h, nk, d, generator=g)
    G = (nq + BM - 1) // BM
    indices = torch.empty(b, h, G, num_sel, dtype=torch.int32)
    for bb in range(b):
        for hh in range(h):
            for gg in range(G):
                indices[bb, hh, gg] = torch.randperm(nk, generator=g, device='cuda')[:num_sel].to(torch.int32)
    counts = torch.full((b, h, G), num_sel, dtype=torch.int32)

    o, _ = csp_attn(q, k, v, indices, counts, 1.0, scale=scale)
    o_ref = reference_sparse_o(q, k, v, indices, counts, scale)

    diff = (o - o_ref).abs().max().item()
    ok = torch.allclose(o, o_ref, atol=ATOL, rtol=RTOL)
    print(f"[csp-sparse] B{b} H{h} Nq{nq} Nk{nk} D{d} sel{num_sel} "
          f"(sel%64={num_sel % 64}) max_diff={diff:.4f} ok={ok}")
    assert ok, f"csp sparse mismatch ({diff})"


def check_lean(b, h, nq, nk, d, num_sel, seed=0):
    """lean=True must match lean=False numerically and skip the M/L byproduct.

    Also checks the transpose-friendly [B,N,H,D] output storage: `.transpose(1, 2)`
    of the lean output must be contiguous (so the caller's reshape is a free view).
    """
    g = torch.Generator(device='cuda').manual_seed(seed)
    scale = 1.0 / math.sqrt(d)
    q = torch.randn(b, h, nq, d, generator=g)
    k = torch.randn(b, h, nk, d, generator=g)
    v = torch.randn(b, h, nk, d, generator=g)
    G = (nq + BM - 1) // BM
    indices = torch.empty(b, h, G, num_sel, dtype=torch.int32)
    for bb in range(b):
        for hh in range(h):
            for gg in range(G):
                indices[bb, hh, gg] = torch.randperm(nk, generator=g, device='cuda')[:num_sel].to(torch.int32)
    counts = torch.full((b, h, G), num_sel, dtype=torch.int32)

    o_ref, ml = csp_attn(q, k, v, indices, counts, 1.0, scale=scale, lean=False)
    o_lean, ml_lean = csp_attn(q, k, v, indices, counts, 1.0, scale=scale, lean=True)

    assert ml_lean is None, "lean path must not return the M/L byproduct"
    assert o_lean.shape == o_ref.shape
    assert o_lean.transpose(1, 2).is_contiguous(), \
        "lean output must be [B,N,H,D]-backed so transpose(1,2).reshape is a free view"
    diff = (o_lean.float() - o_ref.float()).abs().max().item()
    ok = torch.equal(o_lean, o_ref)  # same kernel math -> expect bit-identical
    print(f"[csp-lean] B{b} H{h} Nq{nq} Nk{nk} D{d} sel{num_sel} "
          f"max_diff={diff:.6f} bit_identical={ok}")
    assert diff == 0.0, f"lean vs non-lean mismatch ({diff})"


def main():
    # full indices == dense, arbitrary seqlen (incl. %64 != 0), both layouts
    for n in [4480, 4592, 5376]:
        for contig in [True, False]:
            check_full(1, 24, n, 128, contig)
    # lean path: identical numerics + transpose-friendly output + no M/L byproduct
    check_lean(1, 8, 1600, 6400, 128, num_sel=640, seed=1)
    check_lean(2, 4, 768, 4142, 128, num_sel=512, seed=2)
    # random sparse selections; selected-count both a multiple of 64 and not
    check_sparse(1, 8, 1600, 6400, 128, num_sel=640, seed=1)
    check_sparse(2, 4, 768, 4142, 128, num_sel=512, seed=2)
    check_sparse(1, 8, 1024, 4096, 128, num_sel=600, seed=3)  # 600 % 64 != 0
    print("\nAll csp_attn Triton checks passed.")


if __name__ == '__main__':
    main()
