"""Validate the compact-width (out_width) mask_to_indices path (Phase C).

Sections:
  1. HOST  — the analytic children bound holds against the REAL parent LUT:
             max bincount of get_parent_lut's video region <= prod(_max_axis_children)
             at the real 720p/480p geometries + randomized small shapes (CPU torch).
  2. HOST  — resolve_footprint_out_width returns the expected widths at the 720p
             geometry (27→27: 4816, 27→28/29: 19264) and -1 on every failed
             precondition.
  3. CUDA  — random masks with bounded per-row popcount: compact call (out_width=W)
             vs legacy call must give identical counts and identical per-row
             [0:counts[r]] index prefixes; every padded_total <= W; invalid
             out_width values must raise. Needs the REBUILT chipmunk extension.
  4. TORCH — mask_to_indices_torch out_width parity (same checks, pure torch).

Host/torch sections run anywhere (macOS dev box included); the CUDA section skips
cleanly when the extension is unavailable. Run from the repo root:
    python kernels/tests/test_mask_to_indices_width.py
"""

import os.path as osp
import random
import sys

import torch

sys.path.append(osp.dirname(osp.dirname(osp.dirname(osp.abspath(__file__)))))

from models.sparvar.index_map_utils import get_parent_lut, mask_to_indices_torch
from models.sparvar.index_width_utils import (_max_axis_children,
                                              resolve_footprint_out_width)

# (decision_shape, current_shape, sink_len) — real InfinityStar geometries
REAL_GEOMETRIES = [
    ((20, 24, 43), (20, 24, 43), 3643),   # 720p 27→27 (identity)
    ((20, 24, 43), (20, 30, 53), 3643),   # 720p 27→28
    ((20, 24, 43), (20, 45, 80), 3643),   # 720p 27→29
    ((17, 16, 27), (17, 20, 36), 2048),   # 480p-like
    ((17, 16, 27), (17, 30, 53), 2048),   # 480p-like
]


def check_children_bound_vs_parent_lut():
    """max children per decision cell (true LUT bincount) <= prod(_max_axis_children)."""
    ok = True
    geoms = list(REAL_GEOMETRIES)
    rng = random.Random(0)
    for _ in range(20):
        d = tuple(rng.randint(1, 24) for _ in range(3))
        c = tuple(max(di, rng.randint(1, 48)) for di in d)   # up-scaling (the CS4A case)
        geoms.append((d, c, rng.randint(0, 500)))
    for dec_shape, cur_shape, sink in geoms:
        dk = sink + dec_shape[0] * dec_shape[1] * dec_shape[2]
        ck = sink + cur_shape[0] * cur_shape[1] * cur_shape[2]
        lut = get_parent_lut(
            decision_sink_len=sink, current_sink_len=sink,
            decision_scale_shape=dec_shape, current_scale_shape=cur_shape,
            current_k_len=ck, decision_k_len=dk, device=torch.device('cpu'))
        true_max = int(torch.bincount(lut, minlength=dk).max().item())
        bound = 1
        for d_ax, c_ax in zip(dec_shape, cur_shape):
            bound *= _max_axis_children(d_ax, c_ax)
        good = true_max <= bound
        ok &= good
        print(f"[1 children-bound] dec={dec_shape} cur={cur_shape} sink={sink}: "
              f"true_max={true_max} <= bound={bound} -> {'PASS' if good else 'FAIL'}")
    return ok


def check_resolve_widths():
    ok = True
    base = dict(mode='auto', provider='cuda', decision_indices_count=4816,
                decision_rows_bounded=True, use_band=False, neighbor_frames=1,
                multiple_of=112, indices_pad_to=4)

    cases = [
        # (label, kwargs-overrides, expected)
        ("720p 27→27", dict(decision_scale_shape=(20, 24, 43), current_scale_shape=(20, 24, 43),
                            current_k_len=24283), 4816),
        ("720p 27→28", dict(decision_scale_shape=(20, 24, 43), current_scale_shape=(20, 30, 53),
                            current_k_len=35443), 19264),
        ("720p 27→29", dict(decision_scale_shape=(20, 24, 43), current_scale_shape=(20, 45, 80),
                            current_k_len=75643), 19264),
        ("mode=full", dict(mode='full', decision_scale_shape=(20, 24, 43),
                           current_scale_shape=(20, 45, 80), current_k_len=75643), -1),
        ("provider=triton", dict(provider='triton', decision_scale_shape=(20, 24, 43),
                                 current_scale_shape=(20, 45, 80), current_k_len=75643), -1),
        ("count=None", dict(decision_indices_count=None, decision_scale_shape=(20, 24, 43),
                            current_scale_shape=(20, 45, 80), current_k_len=75643), -1),
        ("rows-unbounded", dict(decision_rows_bounded=False, decision_scale_shape=(20, 24, 43),
                                current_scale_shape=(20, 45, 80), current_k_len=75643), -1),
        ("shape=None", dict(decision_scale_shape=None, current_scale_shape=(20, 45, 80),
                            current_k_len=75643), -1),
        # bound >= k_len (huge S_dec) -> no saving -> -1
        ("no-saving", dict(decision_indices_count=75600, decision_scale_shape=(20, 24, 43),
                           current_scale_shape=(20, 45, 80), current_k_len=75643), -1),
        # band adds (2*1+1)*45*80 = 10800 -> 19264 + 10800 -> round112 = 30128
        ("27→29 +band", dict(use_band=True, decision_scale_shape=(20, 24, 43),
                             current_scale_shape=(20, 45, 80), current_k_len=75643), 30128),
    ]
    for label, over, expect in cases:
        kw = dict(base)
        kw.update(over)
        got = resolve_footprint_out_width(**kw)
        good = (got == expect)
        ok &= good
        print(f"[2 resolve] {label}: got={got} expect={expect} -> {'PASS' if good else 'FAIL'}")
        if got > 0:
            assert got % kw['indices_pad_to'] == 0
    return ok


def _bounded_random_mask(B, H, G, N, max_pop, device, seed=0):
    """[B,H,G,N] bool with per-row popcount in [1, max_pop]; a few rows exactly max_pop."""
    g = torch.Generator(device='cpu').manual_seed(seed)
    mask = torch.zeros(B, H, G, N, dtype=torch.bool)
    for b in range(B):
        for h in range(H):
            for r in range(G):
                pop = max_pop if (r % 3 == 0) else int(torch.randint(1, max_pop + 1, (1,), generator=g))
                sel = torch.randperm(N, generator=g)[:pop]
                mask[b, h, r, sel] = True
    return mask.to(device)


def _compare_compact_vs_full(fn, mask, multiple_of, pad_to, W, label):
    """fn(mask, multiple_of, pad_to[, out_width]) -> (indices, counts). Compare."""
    inds_full, cnt_full = fn(mask, multiple_of, pad_to)
    inds_comp, cnt_comp = fn(mask, multiple_of, pad_to, W)
    ok = True
    if not torch.equal(cnt_full, cnt_comp):
        print(f"[{label}] FAIL: counts differ")
        ok = False
    cmax = int(cnt_full.max().item())
    if cmax > W:
        print(f"[{label}] FAIL: counts.max()={cmax} > W={W} (bound violated)")
        ok = False
    if inds_comp.shape[-1] != W:
        print(f"[{label}] FAIL: compact width {inds_comp.shape[-1]} != {W}")
        ok = False
    # Per-row [0:counts[r]] prefixes must be element-identical (write layout is
    # independent of the output row stride).
    B, H, G = cnt_full.shape
    col = torch.arange(cmax, device=mask.device).view(1, 1, 1, cmax)
    read = col < cnt_full.to(torch.int64).unsqueeze(-1)
    a = inds_full[..., :cmax][read]
    bvals = inds_comp[..., :cmax][read]
    if not torch.equal(a, bvals):
        nbad = int((a != bvals).sum().item())
        print(f"[{label}] FAIL: {nbad} mismatching entries in the counts-bounded read region")
        ok = False
    if a.numel() and (int(a.min().item()) < 0 or int(a.max().item()) >= mask.shape[-1]):
        print(f"[{label}] FAIL: read-region values out of [0, N)")
        ok = False
    print(f"[{label}] counts.max={cmax} W={W} -> {'PASS' if ok else 'FAIL'}")
    return ok


def check_torch_fallback():
    torch.manual_seed(0)
    B, H, G, N = 2, 3, 5, 1123
    multiple_of, pad_to = 112, 4
    max_pop = 300
    W = ((max_pop + multiple_of - 1) // multiple_of) * multiple_of        # 336
    W = ((W + pad_to - 1) // pad_to) * pad_to
    mask = _bounded_random_mask(B, H, G, N, max_pop, torch.device('cpu'))

    def fn(m, mo, pt, ow=-1):
        return mask_to_indices_torch(m, mo, pt, out_width=ow)

    return _compare_compact_vs_full(fn, mask, multiple_of, pad_to, W, "4 torch-fallback")


def check_cuda_op():
    if not torch.cuda.is_available():
        print("[3 cuda] SKIP (no CUDA device)")
        return True
    try:
        import chipmunk  # noqa: F401 -- loads TORCH_LIBRARY initializers
    except Exception:
        pass
    if not hasattr(torch.ops.chipmunk, "mask_to_indices"):
        print("[3 cuda] SKIP (chipmunk op not registered)")
        return True

    ok = True
    torch.manual_seed(0)
    B, H, G, N = 2, 4, 7, 4243                       # deliberately N % 4 != 0
    multiple_of, pad_to = 112, 4
    max_pop = 500
    W = ((max_pop + multiple_of - 1) // multiple_of) * multiple_of        # 560
    W = ((W + pad_to - 1) // pad_to) * pad_to
    mask = _bounded_random_mask(B, H, G, N, max_pop, torch.device('cuda'))

    def fn(m, mo, pt, ow=-1):
        if ow > 0:
            return torch.ops.chipmunk.mask_to_indices(m, mo, pt, ow)
        return torch.ops.chipmunk.mask_to_indices(m, mo, pt)

    ok &= _compare_compact_vs_full(fn, mask, multiple_of, pad_to, W, "3 cuda-op")

    # Invalid out_width values must be rejected by the host TORCH_CHECKs.
    for bad, why in [(W + 1, "not a multiple of pad_to"),
                     (((N + pad_to - 1) // pad_to) * pad_to + pad_to, "exceeds default width")]:
        try:
            torch.ops.chipmunk.mask_to_indices(mask, multiple_of, pad_to, bad)
            print(f"[3 cuda] FAIL: out_width={bad} ({why}) did not raise")
            ok = False
        except RuntimeError:
            print(f"[3 cuda] out_width={bad} ({why}) correctly rejected -> PASS")
    return ok


def main():
    r1 = check_children_bound_vs_parent_lut()
    r2 = check_resolve_widths()
    r4 = check_torch_fallback()
    r3 = check_cuda_op()
    all_ok = r1 and r2 and r3 and r4
    print(f"\n=== test_mask_to_indices_width: {'ALL PASS' if all_ok else 'FAILURES'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
