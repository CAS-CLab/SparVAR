"""CPU tests for the V-CSLA W-A frame-band block prior (models/sparvar/vcsla_utils.py).

Checks, on small synthetic geometries:
  1. bit-exactness against the CS4A reference: block-any reduction of
     ``index_map_utils._get_band_mask`` (the token-level, boundary-shifted band
     the frame-band semantics are defined to match) equals
     ``build_video_frame_band_mask`` when bm == block_size;
  2. boundary shift: frame 0 / last frame cover exactly min(2*rho+1, T)
     contiguous frames, never truncated;
  3. select_kv_blocks integration: budget-zero recipe -> final mask is exactly
     sink + band; dynamic recipe -> final selection is a superset of the band.
"""

import math

import pytest
import torch

from models.sparvar.index_map_utils import _get_band_mask
from models.sparvar.vcsla_utils import build_video_frame_band_mask
from models.sparvar.dcsla_utils import select_kv_blocks


def _selection_mask(kv_indices, kv_num_blocks):
    """[B,H,U,V] bool from the (full-width indices, counts) contract."""
    B, H, U, V = kv_indices.shape
    pos = torch.arange(V).view(1, 1, 1, V)
    valid = pos < kv_num_blocks.to(torch.int64).unsqueeze(-1)
    mask = torch.zeros(B, H, U, V, dtype=torch.bool)
    mask.scatter_(-1, kv_indices.to(torch.int64).clamp(0, V - 1), valid)
    return mask


GEOMETRIES = [
    # (T, H, W, sink_len, block_size, rho)
    (5, 2, 2, 3, 4, 1),      # tiny, frame (4 tokens) == block
    (5, 3, 3, 7, 4, 1),      # frame (9) not a multiple of block (4) -> straddling
    (8, 4, 5, 13, 8, 2),     # wider window, sink straddles a block
    (3, 2, 3, 5, 4, 2),      # window (2*rho+1=5) > T -> clamps to all 3 frames
    (6, 3, 4, 9, 8, 0),      # rho=0: own frame only
]


@pytest.mark.parametrize("T,H,W,sink_len,block_size,rho", GEOMETRIES)
def test_matches_cs4a_band_block_reduction(T, H, W, sink_len, block_size, rho):
    hw = H * W
    q_len = T * hw
    kv_len = sink_len + q_len
    V = math.ceil(kv_len / block_size)

    got = build_video_frame_band_mask(
        (T, H, W), q_len=q_len, kv_len=kv_len, sink_len=sink_len,
        block_size=block_size, rho=rho, device=torch.device("cpu"))

    # CS4A reference: token-level boundary-shifted band, one row per bm-group,
    # reduced to KV blocks with any(). bm == block_size so groups == q-blocks.
    ref_tok = _get_band_mask((T, H, W), sink_len, kv_len, block_size, q_len,
                             rho, torch.device("cpu"))                    # [G, k_len]
    G = ref_tok.shape[0]
    pad = V * block_size - kv_len
    ref_blk = torch.nn.functional.pad(ref_tok, (0, pad)) \
        .view(G, V, block_size).any(dim=-1)                               # [G, V]

    assert got.shape == ref_blk.shape
    assert torch.equal(got, ref_blk)


def test_boundary_shift_full_window():
    T, H, W, sink_len, block_size, rho = 5, 2, 2, 4, 4, 1
    hw = H * W
    q_len = T * hw
    kv_len = sink_len + q_len
    mask = build_video_frame_band_mask(
        (T, H, W), q_len=q_len, kv_len=kv_len, sink_len=sink_len,
        block_size=block_size, rho=rho, device=torch.device("cpu"))

    # hw == block_size == 4 and sink_len == 4: KV block b maps to frame b-1.
    def frames_of(u):
        return {int(b) - 1 for b in torch.nonzero(mask[u]).flatten().tolist()}

    # frame 0 query -> shifted window {0,1,2}; last frame -> {2,3,4}; middle exact.
    assert frames_of(0) == {0, 1, 2}
    assert frames_of(T - 1) == {2, 3, 4}
    assert frames_of(2) == {1, 2, 3}
    # every query block covers exactly 2*rho+1 frames (boundary-shifted, no truncation)
    for u in range(T):
        assert len(frames_of(u)) == 2 * rho + 1


def test_select_kv_blocks_union_and_static():
    torch.manual_seed(0)
    T, H, W, sink_len, block_size, rho = 5, 3, 3, 7, 4, 1
    hw = H * W
    q_len = T * hw
    kv_len = sink_len + q_len
    U = math.ceil(q_len / block_size)
    V = math.ceil(kv_len / block_size)
    sink_block_end = math.ceil(sink_len / block_size)
    band = build_video_frame_band_mask(
        (T, H, W), q_len=q_len, kv_len=kv_len, sink_len=sink_len,
        block_size=block_size, rho=rho, device=torch.device("cpu"))
    mass = torch.rand(1, 2, U, V)

    # Budget-zero recipe: final mask must be exactly sink blocks + frame band.
    inds, counts = select_kv_blocks(
        mass, sink_len=sink_len, q_len=q_len, kv_len=kv_len, block_size=block_size,
        mode='topk_ratio', topk_ratio=0.0, min_blocks=0, max_blocks_ratio=1.0,
        local_diag_blocks=0, keep_sink=True, sink_select_tau=0.0,
        scale_shape=(T, H, W), temporal_neighbor_frames=0, temporal_diag_blocks=0,
        extra_block_mask=band)
    got = _selection_mask(inds, counts)
    expected = band.clone()
    expected[:, :sink_block_end] = True
    assert torch.equal(got, expected.view(1, 1, U, V).expand_as(got))

    # Dynamic recipe: the band is additive -- final selection is a superset.
    inds, counts = select_kv_blocks(
        mass, sink_len=sink_len, q_len=q_len, kv_len=kv_len, block_size=block_size,
        mode='coverage', tau=0.5, min_blocks=1, max_blocks_ratio=1.0,
        local_diag_blocks=1, keep_sink=True, sink_select_tau=0.0,
        scale_shape=(T, H, W), temporal_neighbor_frames=0, temporal_diag_blocks=0,
        extra_block_mask=band)
    with_band = _selection_mask(inds, counts)
    assert bool((with_band | band.view(1, 1, U, V)).eq(with_band).all())

    # And strictly larger than the same dynamic recipe without the band
    # (the band must add blocks the mass ranking did not pick at tau=0.5).
    inds0, counts0 = select_kv_blocks(
        mass, sink_len=sink_len, q_len=q_len, kv_len=kv_len, block_size=block_size,
        mode='coverage', tau=0.5, min_blocks=1, max_blocks_ratio=1.0,
        local_diag_blocks=1, keep_sink=True, sink_select_tau=0.0,
        scale_shape=(T, H, W), temporal_neighbor_frames=0, temporal_diag_blocks=0,
        extra_block_mask=None)
    without_band = _selection_mask(inds0, counts0)
    assert bool((with_band | without_band).eq(with_band).all())
    assert int(with_band.sum()) > int(without_band.sum())
