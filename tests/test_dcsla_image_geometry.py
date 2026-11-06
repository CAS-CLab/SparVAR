#!/usr/bin/env python3
"""CPU tests for the D-CSLA image (Infinity) geometry.

Covers the three pieces that differ from the video path:
  1. the block-count matrix built from the image parent LUT (mass conservation),
  2. the cross-scale locality band vs an independent reimplementation of the
     conference CSLA mask_mod (models/sparvar/basic_sparvar.py),
  3. select_kv_blocks staying bit-identical when the two new optional arguments
     are omitted, i.e. the video path is untouched.

    python tests/test_dcsla_image_geometry.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

from models.sparvar import dcsla_utils as D
from models.sparvar.index_map_utils import get_infinity_parent_lut

DEV = torch.device('cpu')

# small schedule for the brute-force comparisons, real 1M schedule for the rest
SIDES_SMALL = [1, 2, 4, 6, 8, 12]
SCHED_SMALL = [(1, s, s) for s in SIDES_SMALL]
SIDES_FULL = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
SCHED_FULL = [(1, s, s) for s in SIDES_FULL]


def _cum(sched):
    out = [0]
    for t, h, w in sched:
        out.append(out[-1] + t * h * w)
    return out


# ---------------------------------------------------------------- block counts
def test_block_count_conserves_mass(S=2, K=5):
    cum = _cum(SCHED_SMALL)
    ck, dk, bs = cum[K + 1], cum[S + 1], 8
    V = (ck + bs - 1) // bs

    raw = D.get_image_block_count_matrix(
        decision_scale_ind=S, current_scale_ind=K, scale_schedule=SCHED_SMALL,
        current_k_len=ck, decision_k_len=dk, device=DEV, block_size=bs, normalize=False)
    assert raw.shape == (dk, V), f'shape {tuple(raw.shape)} != {(dk, V)}'
    # every current token is counted exactly once (index_add over a total function)
    assert abs(raw.float().sum().item() - ck) < 1e-3, \
        f'raw counts sum {raw.float().sum().item()} != current_k_len {ck}'

    # independent recount via bincount on the parent LUT
    lut = get_infinity_parent_lut(decision_scale_ind=S, current_scale_ind=K,
                                  scale_schedule=SCHED_SMALL, current_k_len=ck,
                                  decision_k_len=dk, device=DEV)
    blk = torch.arange(ck) // bs
    ref = torch.bincount(lut * V + blk, minlength=dk * V).view(dk, V).float()
    assert torch.equal(raw.float(), ref), 'block counts differ from the bincount reference'

    D._block_count_cache.clear()
    nrm = D.get_image_block_count_matrix(
        decision_scale_ind=S, current_scale_ind=K, scale_schedule=SCHED_SMALL,
        current_k_len=ck, decision_k_len=dk, device=DEV, block_size=bs, normalize=True)
    row_sum = nrm.float().sum(-1)
    has_child = ref.sum(-1) > 0
    # a decision token spreads exactly its own mass over its footprint blocks
    assert torch.allclose(row_sum[has_child], torch.ones_like(row_sum[has_child]), atol=2e-2), \
        f'normalized rows do not sum to 1: {row_sum[has_child].min()}..{row_sum[has_child].max()}'
    assert row_sum[~has_child].abs().sum().item() == 0.0, 'orphan decision rows carry mass'
    print(f'[ok] block counts: sum={ck}, rows_with_children={int(has_child.sum())}/{dk}')


def test_mass_transfer_kv_lens(S=10):
    cum = _cum(SCHED_FULL)
    B, H, G = 2, 4, 9
    bs_mat = torch.rand(B, H, G, cum[S + 1])
    mass, klens = D.compute_decision_block_mass_image(
        bs_mat, scale_schedule=SCHED_FULL, decision_scale_ind=S,
        decision_k_len=cum[S + 1], block_size=128, normalize=True)
    assert sorted(mass) == [11, 12], f'unexpected target scales {sorted(mass)}'
    for si in mass:
        assert klens[si] == cum[si + 1], f'k_len[{si}]={klens[si]} != {cum[si + 1]}'
        V = (cum[si + 1] + 127) // 128
        assert tuple(mass[si].shape) == (B, H, G, V), \
            f'mass[{si}] shape {tuple(mass[si].shape)} != {(B, H, G, V)}'
        assert mass[si].float().isfinite().all()
    print(f'[ok] mass transfer: k_lens={ {si: klens[si] for si in klens} }')


def test_video_block_count_unchanged():
    """Regression guard for the video path after factoring out the counting step."""
    from models.sparvar.index_map_utils import get_parent_lut
    sink, ds, cs, bs = 100, (2, 8, 8), (2, 16, 16), 128
    dk = sink + ds[0] * ds[1] * ds[2]
    ck = sink + cs[0] * cs[1] * cs[2]
    V = (ck + bs - 1) // bs

    C = D.get_block_count_matrix(
        decision_sink_len=sink, current_sink_len=sink, decision_scale_shape=ds,
        current_scale_shape=cs, current_k_len=ck, decision_k_len=dk,
        device=DEV, block_size=bs, normalize=False)
    lut = get_parent_lut(decision_sink_len=sink, current_sink_len=sink,
                         decision_scale_shape=ds, current_scale_shape=cs,
                         current_k_len=ck, decision_k_len=dk, device=DEV)
    ref = torch.bincount(lut * V + torch.arange(ck) // bs, minlength=dk * V).view(dk, V).float()
    assert torch.equal(C.float(), ref), 'video block-count matrix changed'
    assert abs(C.float().sum().item() - ck) < 1e-3
    print(f'[ok] video block counts unchanged (dk={dk}, ck={ck}, V={V})')


# ------------------------------------------------------------------ band prior
def _band_reference(sched, K, win, block_size):
    """Token-level rebuild of the conference CSLA mask_mod, reduced to blocks."""
    hs = [int(s[-2]) for s in sched[:K + 1]]
    ws = [int(s[-1]) for s in sched[:K + 1]]
    slice_idx = torch.tensor([0] + list(torch.cumsum(
        torch.tensor([h * w for h, w in zip(hs, ws)]), 0)), dtype=torch.long)
    S_k = int(slice_idx[-1])
    H_q, W_q = hs[K], ws[K]
    S_q = H_q * W_q

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

    U = (S_q + block_size - 1) // block_size
    V = (S_k + block_size - 1) // block_size
    buf = torch.zeros(U * block_size, V * block_size, dtype=torch.bool)
    buf[:S_q, :S_k] = fine
    return buf.view(U, block_size, V, block_size).any(3).any(1)


def test_band_matches_conference_mask_mod(K=5, block_size=8):
    win = [-1, -1, 1, 1, 3, 5]
    cum = _cum(SCHED_SMALL)
    q_len, kv_len = SIDES_SMALL[K] ** 2, cum[K + 1]
    ref = _band_reference(SCHED_SMALL, K, win, block_size)

    D._image_band_cache.clear()
    got = D.build_image_cross_scale_band_mask(
        SCHED_SMALL, K, q_len, kv_len, win, DEV, block_size=block_size)
    assert got.shape == ref.shape, f'{tuple(got.shape)} != {tuple(ref.shape)}'
    assert torch.equal(got, ref), \
        f'band mask differs from the mask_mod reference in {int((got ^ ref).sum())} blocks'

    # chunking must not change the result
    D._image_band_cache.clear()
    chunked = D.build_image_cross_scale_band_mask(
        SCHED_SMALL, K, q_len, kv_len, win, DEV, block_size=block_size, q_chunk=block_size)
    assert torch.equal(chunked, ref), 'q_chunk changes the band mask'
    print(f'[ok] band mask: {int(ref.sum())}/{ref.numel()} blocks kept, density={ref.float().mean():.3f}')


def test_band_disabled_and_full_geometry():
    cum = _cum(SCHED_FULL)
    D._image_band_cache.clear()
    off = D.build_image_cross_scale_band_mask(
        SCHED_FULL, 12, SIDES_FULL[12] ** 2, cum[13], [-1] * 13, DEV)
    assert not off.any(), 'win_size=-1 must disable every scale'

    D._image_band_cache.clear()
    win = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 3, 5, 7]
    for K in (11, 12):
        m = D.build_image_cross_scale_band_mask(
            SCHED_FULL, K, SIDES_FULL[K] ** 2, cum[K + 1], win, DEV)
        U, V = (SIDES_FULL[K] ** 2 + 127) // 128, (cum[K + 1] + 127) // 128
        assert m.shape == (U, V), f'scale {K}: {tuple(m.shape)} != {(U, V)}'
        assert m.any(dim=1).all(), f'scale {K}: some q-block has an empty band'
        print(f'[ok] band @si={K}: U={U} V={V} density={m.float().mean():.3f}')


# ------------------------------------------------------- select_kv_blocks A/B
def test_select_backcompat_and_diag_offset():
    torch.manual_seed(0)
    B, H, U, V, blk = 2, 4, 6, 20, 128
    mass = torch.rand(B, H, U, V)
    kw = dict(sink_len=256, q_len=U * blk, kv_len=V * blk, block_size=blk,
              mode='coverage', tau=0.9, min_blocks=2, max_blocks_ratio=0.5,
              local_diag_blocks=1, keep_sink=True)

    base_i, base_n = D.select_kv_blocks(mass, **kw)
    new_i, new_n = D.select_kv_blocks(mass, **kw, diag_offset=None, extra_block_mask=None)
    assert torch.equal(base_i, new_i) and torch.equal(base_n, new_n), \
        'passing the new arguments as None changed the video result'

    # a non-zero diag_offset must move the forced diagonal band
    off_i, off_n = D.select_kv_blocks(mass, **{**kw, 'diag_offset': 12 * blk})
    assert not torch.equal(off_n, base_n) or not torch.equal(off_i, base_i), \
        'diag_offset had no effect'

    extra = torch.zeros(U, V, dtype=torch.bool)
    extra[:, -1] = True
    ex_i, ex_n = D.select_kv_blocks(mass, **kw, extra_block_mask=extra)
    assert (ex_n >= base_n).all(), 'extra_block_mask must only add blocks'
    sel = torch.zeros(B, H, U, V, dtype=torch.bool)
    sel.scatter_(-1, ex_i.long(), torch.arange(V).view(1, 1, 1, V) < ex_n.unsqueeze(-1))
    assert sel[..., -1].all(), 'the forced extra column is not selected everywhere'
    print(f'[ok] select_kv_blocks: back-compatible, diag_offset and extra mask both active')


# --------------------------------------------------- decision -> BlockMask chain
def test_full_chain_to_block_mask(S=10):
    """Everything dcsla_sparse_attention does except the kernel call.

    The forward itself cannot be checked here: with mask_mod=None the block lists
    are only honoured by the COMPILED triton backend (verified on CPU: eager
    flex_attention with this encoding returns the DENSE result), so numerics are
    a GPU-only check.
    """
    cum = _cum(SCHED_FULL)
    blk, B, H, G = 128, 2, 4, 9
    win = [-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 3, 5, 7]
    torch.manual_seed(0)
    bs_mat = torch.rand(B, H, G, cum[S + 1])
    mass, klens = D.compute_decision_block_mass_image(
        bs_mat, scale_schedule=SCHED_FULL, decision_scale_ind=S,
        decision_k_len=cum[S + 1], block_size=blk, normalize=True)

    for si in sorted(mass):
        q_len, kv_len = SIDES_FULL[si] ** 2, klens[si]
        U, V = (q_len + blk - 1) // blk, (kv_len + blk - 1) // blk
        # phi: same centre-NN q-block remap dcsla_sparse_attention applies
        centers = torch.arange(U, dtype=torch.float32) + 0.5
        phi = torch.round(centers / U * G - 0.5).long().clamp_(0, G - 1)
        rows = mass[si].index_select(2, phi).float()
        assert tuple(rows.shape) == (B, H, U, V)

        band = D.build_image_cross_scale_band_mask(
            SCHED_FULL, si, q_len, kv_len, win, DEV, block_size=blk)
        sink_len = cum[min(5, si)]
        kvi, kvn = D.select_kv_blocks(
            rows, sink_len=sink_len, q_len=q_len, kv_len=kv_len, block_size=blk,
            mode='coverage', tau=0.9, min_blocks=4, max_blocks_ratio=0.5,
            local_diag_blocks=0, keep_sink=True,
            diag_offset=cum[si], extra_block_mask=band)

        sel = torch.zeros(B, H, U, V, dtype=torch.bool)
        sel.scatter_(-1, kvi.long(), torch.arange(V).view(1, 1, 1, V) < kvn.unsqueeze(-1))
        assert sel[..., 0].all(), f'si={si}: sink block not force-kept'
        assert (sel | ~band.view(1, 1, U, V)).all(), f'si={si}: band block dropped'
        assert int(kvn.max()) <= V and int(kvn.min()) >= 1

        bm = D.build_dcsla_block_mask(kvi, kvn, q_len, kv_len, blk)
        assert tuple(bm.shape) == (B, H, q_len, kv_len), \
            f'si={si}: BlockMask shape {tuple(bm.shape)}'
        flat = kvn.float().mean().item() / V

        # coverage must react to mass shape: a spiky row needs far fewer blocks
        # than the flat rand() above (which saturates the max_blocks_ratio cap)
        spiky = torch.full_like(rows, 1e-6)
        spiky[..., V // 2] = 1.0
        s_kvi, s_kvn = D.select_kv_blocks(
            spiky, sink_len=sink_len, q_len=q_len, kv_len=kv_len, block_size=blk,
            mode='coverage', tau=0.9, min_blocks=4, max_blocks_ratio=0.5,
            local_diag_blocks=0, keep_sink=True, diag_offset=cum[si])
        spike_d = s_kvn.float().mean().item() / V
        assert spike_d < 0.2, f'si={si}: spiky mass still selects {spike_d:.3f} of V'
        print(f'[ok] chain @si={si}: U={U} V={V} band={band.float().mean():.3f} '
              f'density: uniform-mass {flat:.3f} (cap-bound), spiky-mass {spike_d:.3f}')


if __name__ == '__main__':
    test_block_count_conserves_mass()
    test_mass_transfer_kv_lens()
    test_video_block_count_unchanged()
    test_band_matches_conference_mask_mod()
    test_band_disabled_and_full_geometry()
    test_select_backcompat_and_diag_offset()
    test_full_chain_to_block_mask()
    print('\nall D-CSLA image geometry tests passed')
