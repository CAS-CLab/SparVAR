#!/usr/bin/env python3
"""GPU smoke test for the SparseDiffAttn IMAGE (Infinity) path, isolated from the
full model: drives the decision scale + both sparse scales with synthetic q/k/v of
the exact Infinity 1024x1024 KV geometry, for use_o_cache on and off.

If this passes but the full play_sparvar.py run fails, the problem is in the model
side (schedule plumbing / debugger interference), not in the sparse-attn module.

Requires a GPU with the repo env (triton kernels):

    python tools/test_sparse_attn_image_smoke.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG, load_from_file

load_from_file("models/sparvar/sparse-attn-config.yml")
assert GLOBAL_CONFIG['attn']['model_family'] == 'image', \
    f"expected image config, got model_family={GLOBAL_CONFIG['attn']['model_family']}"

from models.sparvar.sparse_attn import SparseDiffAttn
from models.sparvar.sparse_attn_layer_counter import LayerCounter

SIDES = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
SCALE_SCHEDULE = [(1, s, s) for s in SIDES]
N_LIST = [s * s for s in SIDES]
CUM = [0]
for n in N_LIST:
    CUM.append(CUM[-1] + n)

B, H, D = 2, 16, 128
assert torch.cuda.is_available(), "this smoke test needs a GPU"


def run(use_o_cache: bool, expand_mode: str = 'center_union'):
    GLOBAL_CONFIG['attn']['use_o_cache'] = use_o_cache   # read by SparseDiffAttn.__init__
    GLOBAL_CONFIG['attn']['cs4a_image_expand_mode'] = expand_mode
    tag = f'[o_cache={use_o_cache}, expand={expand_mode}]'
    layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True)
    module = SparseDiffAttn(layer_num, layer_counter,
                            decision_scale=GLOBAL_CONFIG['attn']['decision_scale'])
    assert module.model_family == 'image'
    S = module.decision_scale

    torch.manual_seed(0)
    for si in (S, S + 1, S + 2):
        q_len, k_len = N_LIST[si], CUM[si + 1]
        q = torch.randn(B, H, q_len, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, H, k_len, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, H, k_len, D, device='cuda', dtype=torch.bfloat16)
        out = module(q, k, v, scale=1.0, do_full_step=(si == S),
                     scale_ind=si, layer_ind=0, scale_schedule=SCALE_SCHEDULE)
        assert out.shape == q.shape, f'shape mismatch at scale {si}: {tuple(out.shape)}'
        assert out.isfinite().all(), f'non-finite output at scale {si}'
        print(f'{tag} scale {si}: q_len={q_len}, k_len={k_len}, '
              f'out={tuple(out.shape)} finite  OK')

    # drop-uncond batch shrink: sparse scale with a smaller batch than the decision step
    q = torch.randn(1, H, N_LIST[S + 2], D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, H, CUM[S + 3], D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, H, CUM[S + 3], D, device='cuda', dtype=torch.bfloat16)
    out = module(q, k, v, scale=1.0, do_full_step=False,
                 scale_ind=S + 2, layer_ind=0, scale_schedule=SCALE_SCHEDULE)
    assert out.shape == q.shape and out.isfinite().all()
    print(f'{tag} scale {S + 2} with batch shrink 2->1  OK')


def run_dcsla(backend: str = 'fa4', band: bool = False):
    """Same drive, D-CSLA path (block-sparse FlexAttention instead of csp_attn).

    Also reports the achieved block density per scale, which is the number that
    decides whether the image side is worth anything: the decision colsum here is
    synthetic, so treat it as a shape/kernel check, not a density prediction.
    """
    a = GLOBAL_CONFIG['attn']
    a['use_o_cache'] = False                 # required by D-CSLA v1
    a['dcsla_enabled'] = True
    a['flex_backend'] = backend
    a['dcsla_image_win_size'] = a['win_size'] if band else None
    tag = f'[dcsla backend={backend}, band={band}]'
    layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True)
    module = SparseDiffAttn(layer_num, layer_counter, decision_scale=a['decision_scale'])
    S = module.decision_scale

    torch.manual_seed(0)
    for si in (S, S + 1, S + 2):
        q_len, k_len = N_LIST[si], CUM[si + 1]
        q = torch.randn(B, H, q_len, D, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(B, H, k_len, D, device='cuda', dtype=torch.bfloat16)
        v = torch.randn(B, H, k_len, D, device='cuda', dtype=torch.bfloat16)
        out = module(q, k, v, scale=1.0, do_full_step=(si == S),
                     scale_ind=si, layer_ind=0, scale_schedule=SCALE_SCHEDULE)
        assert out.shape == q.shape, f'shape mismatch at scale {si}: {tuple(out.shape)}'
        assert out.isfinite().all(), f'non-finite output at scale {si}'
        extra = ''
        if si > S:
            V = (k_len + int(a['flex_block_size']) - 1) // int(a['flex_block_size'])
            extra = f' V={V}'
        print(f'{tag} scale {si}: q_len={q_len}, k_len={k_len}, out={tuple(out.shape)} finite{extra}  OK')

    # drop-uncond batch shrink (mass is [B,H,G,V] and must _match_batch down)
    q = torch.randn(1, H, N_LIST[S + 2], D, device='cuda', dtype=torch.bfloat16)
    k = torch.randn(1, H, CUM[S + 3], D, device='cuda', dtype=torch.bfloat16)
    v = torch.randn(1, H, CUM[S + 3], D, device='cuda', dtype=torch.bfloat16)
    out = module(q, k, v, scale=1.0, do_full_step=False,
                 scale_ind=S + 2, layer_ind=0, scale_schedule=SCALE_SCHEDULE)
    assert out.shape == q.shape and out.isfinite().all()
    print(f'{tag} scale {S + 2} with batch shrink 2->1  OK')
    a['dcsla_enabled'] = False               # leave the config as we found it


def _rel_l2(a, b):
    num = torch.linalg.norm((a - b).float())
    den = torch.linalg.norm(b.float()).clamp_min(1e-12)
    return (num / den).item()


def run_csla(compare_legacy: bool = True):
    """Static CSLA path: drives the StaticCSLAAttn instances directly, both
    backends, plus a numerical acceptance A/B against the migrated legacy
    SparseFlexAttn (same mask set; encoding/kernel differ, so the bar is
    rel_l2, not bit-exactness)."""
    from models.sparvar.sparse_attn_csla import SparseFlexAttn, build_csla_spattn_list
    a = GLOBAL_CONFIG['attn']
    sched2 = [(s, s) for s in SIDES]
    csla_scales = [int(s) for s in a.get('csla_scales', [10, 11, 12])]

    def make_qkv(si, batch=B):
        torch.manual_seed(1000 + si)
        q_len, k_len = N_LIST[si], CUM[si + 1]
        # mimic the live call site: cos_attn keeps q/k fp32, v is bf16
        q = torch.randn(batch, H, q_len, D, device='cuda', dtype=torch.float32)
        k = torch.randn(batch, H, k_len, D, device='cuda', dtype=torch.float32)
        v = torch.randn(batch, H, k_len, D, device='cuda', dtype=torch.bfloat16)
        return q, k, v

    outs = {}
    for backend in ('triton', 'fa4'):
        cfg = dict(a)
        cfg['flex_backend'] = backend
        lst = build_csla_spattn_list(sched2, cfg, num_scales=len(SIDES))
        for si in csla_scales:
            q, k, v = make_qkv(si)
            out = lst[si](q, k, v, scale=1.0)
            assert out.shape == q.shape, f'[csla {backend}] shape mismatch at scale {si}'
            assert out.isfinite().all(), f'[csla {backend}] non-finite output at scale {si}'
            outs[(backend, si)] = out
            print(f'[csla backend={backend}] scale {si}: out={tuple(out.shape)} finite  OK')
        # drop-uncond batch shrink: the H=1 BlockMask broadcasts over any batch
        q, k, v = make_qkv(csla_scales[-1], batch=1)
        out = lst[csla_scales[-1]](q, k, v, scale=1.0)
        assert out.shape == q.shape and out.isfinite().all()
        print(f'[csla backend={backend}] scale {csla_scales[-1]} with batch 1  OK')

    for si in csla_scales:
        r = _rel_l2(outs[('fa4', si)], outs[('triton', si)])
        assert r < 1e-2, f'[csla] fa4-vs-triton rel_l2={r:.3e} at scale {si}'
        print(f'[csla] scale {si}: fa4 vs triton rel_l2={r:.3e}  OK')

    if compare_legacy:
        for si in csla_scales:
            legacy = SparseFlexAttn(sched2, a['win_size'], si, a['attn_sink'], num_heads=H)
            q, k, v = make_qkv(si)
            ref = legacy(q, k, v, scale=1.0)
            r = _rel_l2(outs[('triton', si)], ref)
            assert r < 1e-3, f'[csla] new-vs-legacy rel_l2={r:.3e} at scale {si}'
            print(f'[csla] scale {si}: modernized(triton) vs legacy rel_l2={r:.3e}  OK')
            del legacy
            torch.cuda.empty_cache()


if __name__ == '__main__':
    dcsla_only = '--dcsla' in sys.argv
    csla_only = '--csla' in sys.argv
    if csla_only:
        run_csla()
    else:
        if not dcsla_only:
            for mode in ('center_union', 'footprint'):
                run(True, mode)
                run(False, mode)
        for backend in ('triton', 'fa4'):
            for band in (False, True):
                run_dcsla(backend, band)
        run_csla()
    print('\nSparseDiffAttn image-path smoke test passed')
