#!/usr/bin/env python3
"""Sanity tests for the Infinity (image) CS4A index-mapping utilities in
``models/sparvar/index_map_utils.py``:

  1. LUT gather == elementwise Decompose-Align-Project reference (bit-identical)
  2. structural invariants of the relative mapping (scale shift, value range)
  3. deviation vs the legacy ``map_sparse_kv_inds`` (endpoint- vs center-aligned
     projection: expected <= 1 grid cell per axis, reported not asserted)
  4. image local-window mask == brute-force per-row window union
  5. ``build_image_local_union_indices``: sink prefix / mapped centers / local
     window all present within each group's ``counts[g]``, counts are multiples
     of ``multiple_of``
  8. cross-scale band tokens == brute-force |c - kv| <= half reference (small
     synthetic schedule, exercises the padded-grid boundary case)
  9. band tokens: legacy int equivalence, D-CSLA block-mask cross-check,
     footprint-builder int-vs-list back-compat, scale-coverage columns

Run inside the repo env (imports pull in triton via sparse_attn_ops):

    python tools/test_infinity_index_map.py
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch

import models.sparvar.index_map_utils as imu
from models.sparvar.index_map_utils import (
    _get_image_cross_scale_band_mask_tokens,
    _get_image_local_window_mask,
    build_image_local_union_indices,
    build_infinity_footprint_union_indices,
    get_infinity_parent_lut,
    map_infinity_sparse_kv_inds,
    map_infinity_sparse_kv_inds_lut,
    map_sparse_kv_inds,
)

# Infinity 1024x1024 token schedule
SIDES = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
SCALE_SCHEDULE = [(1, s, s) for s in SIDES]
N_LIST = [s * s for s in SIDES]
CUM = [0]
for n in N_LIST:
    CUM.append(CUM[-1] + n)          # [0, 1, 5, 21, ..., 4121, 6425, 10521]

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# Force the pure-torch mask_to_indices fallback: deterministic within-row layout across
# backends (the CUDA op is contract-identical but thread-strided), and CPU-safe.
imu._USE_CUDA_MASK_TO_INDICES = False
print(f'device = {DEVICE}')


def _rand_decision_inds(B, H, G, count, decision_k_len):
    return torch.randint(0, decision_k_len, (B, H, G, count), device=DEVICE)


def test_lut_matches_reference(S=10):
    torch.manual_seed(0)
    for K in (11, 12):
        current_k_len = CUM[K + 1]
        inds = _rand_decision_inds(2, 4, 12, 832, CUM[S + 1])
        ref = map_infinity_sparse_kv_inds(
            inds, decision_scale_ind=S, current_scale_ind=K,
            scale_schedule=SCALE_SCHEDULE, current_k_len=current_k_len)
        lut = map_infinity_sparse_kv_inds_lut(
            inds.clone(), decision_scale_ind=S, current_scale_ind=K,
            scale_schedule=SCALE_SCHEDULE, current_k_len=current_k_len,
            decision_k_len=CUM[S + 1])
        assert torch.equal(ref, lut), f'LUT != reference at K={K}'
        print(f'[1] LUT == elementwise reference  (S={S}, K={K}, k_len={current_k_len})  OK')


def test_relative_invariants(S=10):
    for K in (11, 12):
        current_k_len = CUM[K + 1]
        j = torch.arange(CUM[S + 1], device=DEVICE).view(1, 1, 1, -1)
        jp = map_infinity_sparse_kv_inds(
            j, decision_scale_ind=S, current_scale_ind=K,
            scale_schedule=SCALE_SCHEDULE, current_k_len=current_k_len).long().view(-1)

        cum_t = torch.tensor(CUM, device=DEVICE)
        l_src = torch.bucketize(j.view(-1), cum_t[1:S + 2], right=True)
        l_dst = torch.bucketize(jp, cum_t[1:K + 2], right=True)
        assert torch.equal(l_dst, l_src + (K - S)), f'scale shift violated at K={K}'
        assert int(jp.min()) >= CUM[K - S] and int(jp.max()) < current_k_len, \
            f'range violated at K={K}: [{int(jp.min())}, {int(jp.max())}]'
        print(f'[2] relative alignment invariants (l\'=l+{K - S}, range [{CUM[K - S]}, {current_k_len}))  OK')


def test_deviation_vs_legacy(S=10):
    """Legacy map_sparse_kv_inds projects with endpoint alignment (u*h'//h); the new
    mapper uses center alignment ((2u+1)h'//(2h)). Same Decompose/Align steps, so the
    per-axis deviation should be at most ~1 grid cell.

    NOTE: the legacy hw_list (2D) branch uses index_select, which only accepts 1-D
    index tensors — so both mappers are fed 1-D input here (the new mapper is
    shape-agnostic either way)."""
    cum_t = torch.tensor(CUM, dtype=torch.long, device=DEVICE)
    hw_list = [(s, s) for s in SIDES]
    for K in (11, 12):
        current_k_len = CUM[K + 1]
        j = torch.arange(CUM[S + 1], device=DEVICE)                  # 1-D
        new = map_infinity_sparse_kv_inds(
            j, decision_scale_ind=S, current_scale_ind=K,
            scale_schedule=SCALE_SCHEDULE, current_k_len=current_k_len).long().view(-1)
        old = map_sparse_kv_inds(
            j.clone(), S=S, K=K, n_list=N_LIST, cum_S=cum_t, cum_K=cum_t,
            hw_list=hw_list, sink_first_k=0).long().view(-1)

        l_dst = torch.bucketize(new, cum_t[1:K + 2], right=True)
        w_dst = torch.tensor(SIDES, device=DEVICE)[l_dst]
        d_new, d_old = new - cum_t[l_dst], old - cum_t[l_dst]
        du = (torch.div(d_new, w_dst, rounding_mode='trunc')
              - torch.div(d_old, w_dst, rounding_mode='trunc')).abs()
        dv = (d_new % w_dst - d_old % w_dst).abs()
        match = float((new == old).float().mean())
        print(f'[3] K={K}: exact-match vs legacy = {match:.3f}, '
              f'max |Δu| = {int(du.max())}, max |Δv| = {int(dv.max())} '
              f'(center- vs endpoint-aligned projection)')


def test_local_window_mask():
    h = w = 8
    bm, win = 16, 3
    q_len, block_start = h * w, 21
    k_len = block_start + h * w
    mask = _get_image_local_window_mask((1, h, w), block_start, k_len, bm, q_len, win, DEVICE)

    G = (q_len + bm - 1) // bm
    ref = torch.zeros(G, k_len, dtype=torch.bool, device=DEVICE)
    r = win // 2
    for row in range(q_len):
        g = min(row // bm, G - 1)
        u, v = row // w, row % w
        for uu in range(max(0, u - r), min(h, u + r + 1)):
            for vv in range(max(0, v - r), min(w, v + r + 1)):
                ref[g, block_start + uu * w + vv] = True
    assert torch.equal(mask, ref), 'local window mask != brute force'
    print(f'[4] image local window mask == brute force (h=w={h}, bm={bm}, win={win})  OK')


def test_union_indices():
    torch.manual_seed(1)
    B, H, bm, win = 2, 3, 16, 3
    h = w = 8
    block_start, sink_len = 21, 5
    q_len, k_len = h * w, block_start + h * w
    multiple_of, indices_pad_to = 8, 1
    G = (q_len + bm - 1) // bm
    centers = torch.randint(0, k_len, (B, H, G, 6), device=DEVICE)

    inds, counts = build_image_local_union_indices(
        centers, sink_len=sink_len, current_k_len=k_len,
        current_scale_shape=(1, h, w), current_block_start=block_start,
        bm=bm, q_len=q_len, win_size=win,
        multiple_of=multiple_of, indices_pad_to=indices_pad_to)

    assert inds.shape[-1] == k_len and counts.shape == (B, H, G)
    assert int(counts.max()) <= k_len and (counts % multiple_of == 0).all()

    local = _get_image_local_window_mask((1, h, w), block_start, k_len, bm, q_len, win, DEVICE)
    for b in range(B):
        for hh in range(H):
            for g in range(G):
                sel = set(inds[b, hh, g, :int(counts[b, hh, g])].tolist())
                expect = set(range(sink_len)) | set(centers[b, hh, g].tolist()) \
                    | set(torch.nonzero(local[g]).view(-1).tolist())
                missing = expect - sel
                assert not missing, f'union missing {sorted(missing)[:8]} at (b={b},h={hh},g={g})'
    print(f'[5] union indices cover sink | local window | mapped centers; '
          f'counts % {multiple_of} == 0  OK')


def test_parent_lut(S=10):
    cum_t = torch.tensor(CUM, device=DEVICE)
    for K in (11, 12):
        ck, dk = CUM[K + 1], CUM[S + 1]
        shift = K - S
        parent = get_infinity_parent_lut(
            decision_scale_ind=S, current_scale_ind=K, scale_schedule=SCALE_SCHEDULE,
            current_k_len=ck, decision_k_len=dk, device=DEVICE)

        # parentless prefix -> identity (decision indices reused verbatim)
        pref = CUM[shift]
        assert torch.equal(parent[:pref], torch.arange(pref, device=DEVICE)), \
            f'prefix identity violated at K={K}'
        # body: parent lives in the aligned source scale l = l' - shift
        c = torch.arange(ck, device=DEVICE)
        l_cur = torch.bucketize(c, cum_t[1:K + 2], right=True)
        l_par = torch.bucketize(parent, cum_t[1:S + 2], right=True)
        body = c >= pref
        assert torch.equal(l_par[body], l_cur[body] - shift), f'parent scale shift violated at K={K}'
        # roundtrip: every decision cell's forward center lies inside its own footprint
        j = torch.arange(dk, device=DEVICE)
        fwd = map_infinity_sparse_kv_inds(
            j, decision_scale_ind=S, current_scale_ind=K,
            scale_schedule=SCALE_SCHEDULE, current_k_len=ck).long()
        assert torch.equal(parent[fwd], j), f'center-footprint roundtrip violated at K={K}'
        print(f'[6] parent LUT (K={K}): prefix identity [0,{pref}) | scale shift +{shift} | '
              f'roundtrip parent[fwd[j]]==j  OK')


def test_footprint_union(S=10, K=12):
    torch.manual_seed(2)
    dk, ck = CUM[S + 1], CUM[K + 1]
    B, H, bm = 1, 2, 192
    q_len = N_LIST[K]
    Gs = (N_LIST[S] + bm - 1) // bm
    Gk = (q_len + bm - 1) // bm
    multiple_of, indices_pad_to = 64, 1
    sink_len = CUM[5]
    decision_mask = torch.rand(B, H, Gs, dk, device=DEVICE) < 0.05

    inds, counts = build_infinity_footprint_union_indices(
        decision_mask=decision_mask, decision_scale_ind=S, current_scale_ind=K,
        scale_schedule=SCALE_SCHEDULE, decision_k_len=dk, current_k_len=ck,
        current_q_len=q_len, sink_len=sink_len, bm=bm,
        multiple_of=multiple_of, indices_pad_to=indices_pad_to)
    assert inds.shape[-1] == ck and counts.shape == (B, H, Gk)

    # expected mask, built independently: q-group inherit -> parent gather -> sink union
    parent = get_infinity_parent_lut(
        decision_scale_ind=S, current_scale_ind=K, scale_schedule=SCALE_SCHEDULE,
        current_k_len=ck, decision_k_len=dk, device=DEVICE)
    gk_centers = torch.arange(Gk, device=DEVICE, dtype=torch.float32) + 0.5
    gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).long().clamp_(0, Gs - 1)
    expected = decision_mask.index_select(2, gS_idx).index_select(3, parent).clone()
    expected[..., :sink_len] = True

    for b in range(B):
        for hh in range(H):
            for g in range(Gk):
                cnt = int(counts[b, hh, g])
                sel = torch.zeros(ck, dtype=torch.bool, device=DEVICE)
                sel[inds[b, hh, g, :cnt].long()] = True
                exp = expected[b, hh, g]
                assert bool(sel[exp].all()), f'footprint union missing keys at (b={b},h={hh},g={g})'
                pop = int(exp.sum())
                assert pop <= cnt <= min(((pop + multiple_of - 1) // multiple_of) * multiple_of, ck)
    print(f'[7] footprint union (K={K}): all expected keys within per-group counts; '
          f'counts rounded to {multiple_of}  OK')


def _band_tokens_reference(schedule, K, bm, wins, device):
    """Independent per-token |c - kv| <= half reference of the cross-scale band."""
    hs = [int(s[-2]) for s in schedule[:K + 1]]
    ws_ = [int(s[-1]) for s in schedule[:K + 1]]
    eff = [wins[l] if l < len(wins) else wins[-1] for l in range(K + 1)]
    q_len = hs[K] * ws_[K]
    k_len = sum(h * w for h, w in zip(hs, ws_))
    G = (q_len + bm - 1) // bm
    ref = torch.zeros(G, k_len, dtype=torch.bool, device=device)
    for row in range(q_len):
        g = min(row // bm, G - 1)
        qy, qx = row // ws_[K], row % ws_[K]
        start = 0
        for l in range(K + 1):
            h_l, w_l, win = hs[l], ws_[l], eff[l]
            if win >= 1:
                half = win // 2
                # same fp32 projection expression as the implementation
                cy = int(torch.round(torch.tensor(qy, dtype=torch.float32) / hs[K] * h_l))
                cx = int(torch.round(torch.tensor(qx, dtype=torch.float32) / ws_[K] * w_l))
                for kv_y in range(max(0, cy - half), min(h_l, cy + half + 1)):
                    for kv_x in range(max(0, cx - half), min(w_l, cx + half + 1)):
                        ref[g, start + kv_y * w_l + kv_x] = True
            start += h_l * w_l
    return ref


def test_band_tokens_reference():
    # small synthetic schedule; side-1/2 scales exercise the padded-grid boundary
    # (round() lands on h_l for late-row queries)
    schedule = [(1, s, s) for s in (1, 2, 4, 6, 8)]
    K, bm = 4, 16
    for wins in ([1, -1, 1, 3, 5], [-1, -1, -1, 3, 3], [3]):        # incl. broadcast
        q_len = schedule[K][1] * schedule[K][2]
        k_len = sum(s[1] * s[2] for s in schedule[:K + 1])
        mask = _get_image_cross_scale_band_mask_tokens(
            schedule, K, k_len, bm, q_len, wins, DEVICE)
        ref = _band_tokens_reference(schedule, K, bm, wins, DEVICE)
        assert torch.equal(mask, ref), f'band tokens != brute force for wins={wins}'
    print('[8] cross-scale band tokens == brute force (incl. boundary + broadcast)  OK')


def test_band_tokens_legacy_and_dcsla(S=10):
    # (a) list [-1]*K + [w] == legacy self-only local window mask
    for K, w in ((11, 7), (12, 9)):
        q_len, k_len, bm = N_LIST[K], CUM[K + 1], 192
        band = _get_image_cross_scale_band_mask_tokens(
            SCALE_SCHEDULE, K, k_len, bm, q_len, [-1] * K + [w], DEVICE)
        legacy = _get_image_local_window_mask(
            (1, SIDES[K], SIDES[K]), CUM[K], k_len, bm, q_len, w, DEVICE)
        assert torch.equal(band, legacy), f'self-only band != legacy local mask at K={K}'

    # (b) 128-blockified band == the D-CSLA block-level band builder (bit-lock the
    # two band geometries against drift)
    from models.sparvar.dcsla_utils import build_image_cross_scale_band_mask
    wins = [-1] * 10 + [3, 5, 7]
    for K in (11, 12):
        q_len, k_len = N_LIST[K], CUM[K + 1]
        tok = _get_image_cross_scale_band_mask_tokens(
            SCALE_SCHEDULE, K, k_len, 128, q_len, wins, DEVICE)
        V = (k_len + 127) // 128
        pad = torch.zeros(tok.shape[0], V * 128 - k_len, dtype=torch.bool, device=DEVICE)
        blk = torch.cat([tok, pad], dim=1).view(-1, V, 128).any(dim=-1)
        ref = build_image_cross_scale_band_mask(SCALE_SCHEDULE, K, q_len, k_len, wins, DEVICE)
        assert torch.equal(blk, ref), f'blockified band != D-CSLA band builder at K={K}'

    # (c) scale coverage columns at K=11: scales 0..9 empty, 10 and 11 populated
    tok = _get_image_cross_scale_band_mask_tokens(
        SCALE_SCHEDULE, 11, CUM[12], 128, N_LIST[11], wins, DEVICE)
    assert not tok[:, :CUM[10]].any(), 'band leaked into disabled scales 0..9'
    assert tok[:, CUM[10]:CUM[11]].any() and tok[:, CUM[11]:CUM[12]].any(), \
        'band missing on scale 10 / 11 segments'
    print('[9] band tokens: legacy equivalence | D-CSLA block cross-check | coverage  OK')


def test_footprint_union_band_compat(S=10, K=12):
    torch.manual_seed(3)
    dk, ck = CUM[S + 1], CUM[K + 1]
    B, H, bm, w = 1, 2, 192, 9
    q_len = N_LIST[K]
    Gs = (N_LIST[S] + bm - 1) // bm
    decision_mask = torch.rand(B, H, Gs, dk, device=DEVICE) < 0.05
    kwargs = dict(
        decision_mask=decision_mask, decision_scale_ind=S, current_scale_ind=K,
        scale_schedule=SCALE_SCHEDULE, decision_k_len=dk, current_k_len=ck,
        current_q_len=q_len, sink_len=CUM[5], bm=bm,
        multiple_of=64, indices_pad_to=1, union_local=True,
        current_block_start=CUM[K])
    inds_int, counts_int = build_infinity_footprint_union_indices(**kwargs, win_size=w)
    inds_lst, counts_lst = build_infinity_footprint_union_indices(
        **kwargs, win_size=[-1] * K + [w])
    assert torch.equal(inds_int, inds_lst) and torch.equal(counts_int, counts_lst), \
        'legacy int win_size != equivalent per-scale list'
    print(f'[10] footprint union: legacy int win == [-1]*K + [win] list (K={K})  OK')


if __name__ == '__main__':
    test_lut_matches_reference()
    test_relative_invariants()
    test_deviation_vs_legacy()
    test_local_window_mask()
    test_union_indices()
    test_parent_lut()
    test_footprint_union()
    test_band_tokens_reference()
    test_band_tokens_legacy_and_dcsla()
    test_footprint_union_band_compat()
    print('\nall Infinity index-map tests passed')
