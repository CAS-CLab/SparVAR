"""Verify that the CUDA `torch.ops.chipmunk.csp_attn` (ThunderKittens) honours
PER-GROUP (variable) counts — i.e. each query group attends to its OWN
`counts[b,h,g]` keys, not a single uniform count.

Method: feed deliberately *different* per-group counts (all multiples of 64) and
compare against a per-group reference (masked softmax-SDPA over each group's
selected keys). If the kernel silently assumed a uniform count, the groups whose
count differs from the assumed one would mismatch.

Two checks:
  (A) sanity — select ALL keys (count = kv_len for every group): must equal dense
      SDPA. Confirms the harness + kernel normalisation are correct.
  (B) the real test — strongly varying per-group counts: must equal the per-group
      reference. PASS here means variable counts are supported.

Run on the GPU server (chipmunk extension must be built):
    python tools/test_csp_variable_counts.py
"""

import math
import os.path as osp
import sys

import torch
import torch.nn.functional as F

sys.path.append(osp.dirname(osp.dirname(__file__)))

from kernels.sparse_attn.sparse_attn_config import (GLOBAL_CONFIG,
                                                    get_kernel_config_attn,
                                                    load_from_file)
from kernels.sparse_attn.sparse_attn_ops import csp_attn  # (q,k,v,scale,inds,counts,o,o_scale)


@torch.no_grad()
def reference_sparse_attn(q, k, v, inds, counts, bm, scale):
    """Per query GROUP g (rows [g*bm, (g+1)*bm)), masked softmax-SDPA over that
    group's first counts[g] selected keys (all queries in a group share keys)."""
    B, H, Nq, D = q.shape
    o = torch.zeros_like(q)
    G = inds.shape[2]
    for b in range(B):
        for h in range(H):
            for g in range(G):
                c = int(counts[b, h, g].item())
                q0, q1 = g * bm, min((g + 1) * bm, Nq)
                if q0 >= q1 or c <= 0:
                    continue
                sel = inds[b, h, g, :c].long()
                qg = q[b, h, q0:q1].float()
                attn = (qg @ k[b, h, sel].float().T) * scale     # [qb, c]
                attn = attn.softmax(dim=-1)
                o[b, h, q0:q1] = (attn @ v[b, h, sel].float()).to(o.dtype)
    return o


def _build_inds(B, H, G, Nk, counts, device, sort_ascending=False):
    """First counts[g] entries = unique random keys (optionally sorted asc); rest
    padded with a valid index (not read by the kernel)."""
    inds = torch.zeros(B, H, G, Nk, dtype=torch.int32, device=device)
    for b in range(B):
        for h in range(H):
            for g in range(G):
                c = int(counts[b, h, g].item())
                sel = torch.randperm(Nk, device=device)[:c].to(torch.int32)
                if sort_ascending:
                    sel = sel.sort().values
                inds[b, h, g, :c] = sel
                if c < Nk:
                    inds[b, h, g, c:] = sel[0] if c > 0 else 0
    return inds


def _run(q, k, v, inds, counts, scale, bm, label, tol=0.1):
    o = torch.zeros_like(q).contiguous()
    o = csp_attn(q, k, v, scale, inds.contiguous(), counts.contiguous(), o, 1)
    o_ref = reference_sparse_attn(q, k, v, inds, counts, bm, scale)
    diff = (o.float() - o_ref.float()).abs()
    max_diff, mean_diff = diff.max().item(), diff.mean().item()
    G = inds.shape[2]
    per_g = [round(diff[:, :, g * bm:min((g + 1) * bm, q.shape[2])].max().item(), 3) for g in range(G)]
    ok = max_diff < tol
    print(f"[{label}] max={max_diff:.4f} mean={mean_diff:.4f}  "
          f"counts0={counts[0, 0].tolist()}  per-group-max={per_g}  -> "
          f"{'PASS' if ok else 'FAIL'}")
    return ok


def main():
    assert torch.cuda.is_available(), "CUDA required"
    load_from_file("models/sparvar/sparse-attn-video-config.yml")
    provider = GLOBAL_CONFIG['attn']['provider']
    kc = get_kernel_config_attn()
    bm = int(kc['bm'])
    tile = 64  # kv_tile rows == BLOCK_N == multiple_of; counts must be a multiple of this
    print(f"provider={provider}, bm={bm}, tile(multiple_of)={tile}")

    torch.manual_seed(0)
    device, dtype = "cuda", torch.bfloat16
    B, H, D = 1, 4, 128
    G = 6
    Nq = bm * G
    Nk = 4096
    scale = 1.0 / math.sqrt(D)

    q = torch.randn(B, H, Nq, D, device=device, dtype=dtype)
    k = torch.randn(B, H, Nk, D, device=device, dtype=dtype)
    v = torch.randn(B, H, Nk, D, device=device, dtype=dtype)

    var_counts = torch.zeros(B, H, G, dtype=torch.int32, device=device)
    uni_counts = torch.full((B, H, G), 640, dtype=torch.int32, device=device)
    per_group = [64, 128, 640, 1024, 1600, 2048]
    for g in range(G):
        var_counts[:, :, g] = min(per_group[g % len(per_group)], (Nk // tile) * tile)

    # 4 controls to isolate the cause:
    #   A  all keys selected         -> must equal dense SDPA (sanity)
    #   B  variable counts, RANDOM indices
    #   B1 variable counts, SORTED indices  (isolates index-order sensitivity)
    #   B2 UNIFORM count,  RANDOM indices    (isolates count-variation sensitivity)
    all_inds = torch.arange(Nk, dtype=torch.int32, device=device).view(1, 1, 1, Nk).expand(B, H, G, Nk)
    all_counts = torch.full((B, H, G), Nk, dtype=torch.int32, device=device)
    ok_a = _run(q, k, v, all_inds, all_counts, scale, bm, "A  all-keys==dense ")

    ok_b = _run(q, k, v, _build_inds(B, H, G, Nk, var_counts, device, sort_ascending=False),
                var_counts, scale, bm, "B  var + random   ")
    ok_b1 = _run(q, k, v, _build_inds(B, H, G, Nk, var_counts, device, sort_ascending=True),
                 var_counts, scale, bm, "B1 var + sorted   ")
    ok_b2 = _run(q, k, v, _build_inds(B, H, G, Nk, uni_counts, device, sort_ascending=False),
                 uni_counts, scale, bm, "B2 uniform + rand ")

    #   C  COMPACT-width indices (width < Nk): validates the relaxed csp_attn wrapper
    #      + the kernel's stride(2)-based row addressing with a narrow allocation
    #      (the mask_to_indices out_width path). Counts use the provider's REAL
    #      counts_multiple_of (112 for cuda) so this case is kernel-valid even though
    #      the legacy cases above hardcode tile=64. CUDA provider only: the Triton
    #      wrapper keeps the strict k_len-width contract.
    ok_c = True
    if provider == 'cuda':
        tile_c = int(kc['counts_multiple_of'])
        pad_to = int(kc['indices_pad_to'])
        comp_counts = torch.zeros(B, H, G, dtype=torch.int32, device=device)
        per_group_c = [1, 2, 5, 3, 8, 4]
        for g in range(G):
            comp_counts[:, :, g] = min(per_group_c[g % len(per_group_c)] * tile_c,
                                       (Nk // tile_c) * tile_c)
        W = int(comp_counts.max().item())
        W = (W + pad_to - 1) // pad_to * pad_to
        comp_inds = _build_inds(B, H, G, Nk, comp_counts, device)[:, :, :, :W].contiguous()
        ok_c = _run(q, k, v, comp_inds, comp_counts, scale, bm, f"C  compact W={W} ")
    else:
        print("[C  compact] skipped (triton provider keeps the strict k_len-width contract)")

    print("\n--- diagnosis ---")
    print(f"  A(sanity)={ok_a}  B(var+rand)={ok_b}  B1(var+sorted)={ok_b1}  B2(uni+rand)={ok_b2}  C(compact)={ok_c}")
    if ok_b1 and not ok_b:
        print("  => index ORDER matters (kernel wants sorted indices); variable counts are FINE.")
    elif ok_b2 and not ok_b1:
        print("  => variable COUNTS are the problem (uniform works, variable doesn't).")
    elif ok_b and ok_b1 and ok_b2:
        print("  => everything works; the earlier FAIL was just the tight tolerance.")
    else:
        print("  => inconclusive; look at per-group-max to see which count/group fails.")


if __name__ == "__main__":
    main()
