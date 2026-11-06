"""Host-side analytic width bound for the compact ``mask_to_indices`` output (CS4A
footprint path).

``mask_to_indices`` legacy behavior allocates ``[B, H, G, ceil(k_len/pad_to)*pad_to]``
int32 — 3.63 GB transient at the last 720p scale (B=1, H=32, G=375, k_len=75643). The
CUDA kernel only ever writes ``padded_total = ceil(popcount/multiple_of)*multiple_of``
entries per row, so when every row's ``padded_total`` is provably below an analytic
bound ``W``, the output can be allocated at width ``W`` instead (−2.7 GB at that scale).

The bound (footprint expansion, per ``[b, h, g]`` row):

    popcount(row) <= S_dec * prod_axis(max_children_axis) + band_popcount_bound

where ``S_dec`` is the decision top-k count (each decision-mask row carries at most
``S_dec`` selected keys), every selected decision key contributes at most
``prod_axis(max_children_axis)`` footprint children under the inverse parent LUT
(sink keys map identically and contribute exactly 1 <= that), and the optional
full-frame band adds at most ``(2*neighbor_frames+1) * ch * cw`` keys per row.
``max_children_axis`` is counted EXACTLY per axis (axis dims are tiny), so the bound
needs no conjectured ``ceil(c/d)``-style formula.

Everything here is plain host arithmetic — no torch, no syncs.
"""

from typing import Dict, Optional, Sequence, Tuple

_axis_children_cache: Dict[Tuple[int, int], int] = {}


def _max_axis_children(d: int, c: int) -> int:
    """Exact max preimage size of the inverse-footprint axis map used by
    ``get_parent_lut``: ``dec = floor((2*cur + 1) * d / (2*c))`` for ``cur in [0, c)``
    (then clamped to ``d - 1``).

    Counted by enumeration (``c`` is a t/h/w scale dim, <= ~100), memoized per
    ``(d, c)`` — the result is provably >= every decision cell's children count.
    """
    d = int(d)
    c = int(c)
    hit = _axis_children_cache.get((d, c))
    if hit is not None:
        return hit
    if d <= 0 or c <= 0:
        _axis_children_cache[(d, c)] = 1
        return 1
    counts: Dict[int, int] = {}
    for j in range(c):
        m = ((2 * j + 1) * d) // (2 * c)
        if m > d - 1:
            m = d - 1
        counts[m] = counts.get(m, 0) + 1
    result = max(counts.values())
    _axis_children_cache[(d, c)] = result
    return result


def _round_up(x: int, m: int) -> int:
    return -(-int(x) // int(m)) * int(m)


def resolve_footprint_out_width(
    *,
    mode: str,
    provider: str,
    decision_scale_shape: Optional[Sequence[int]],
    current_scale_shape: Optional[Sequence[int]],
    current_k_len: int,
    decision_indices_count: Optional[int],
    decision_rows_bounded: bool,
    use_band: bool,
    neighbor_frames: int,
    multiple_of: int,
    indices_pad_to: int,
) -> int:
    """Compact ``mask_to_indices`` out_width for a footprint sparse step, or -1.

    Returns -1 (= legacy full-width allocation, exact current behavior) unless ALL
    preconditions hold:
      - ``mode == 'auto'`` (config ``attn.cs4a_compact_index_width``)
      - ``provider == 'cuda'`` (the Triton csp path keeps the strict k_len contract)
      - ``decision_indices_count`` known host-side (S_dec)
      - ``decision_rows_bounded``: the caller certifies each decision-mask row holds
        at most S_dec selected keys (pure top-k; no random fill / singleton unions /
        uninitialized-tail scatters)
      - both scale shapes known

    On success returns ``W`` with the guarantees the CUDA host entry requires:
      ``padded_total(row) <= W`` for every row, ``W % indices_pad_to == 0``, and
      ``W < ceil(k_len/indices_pad_to)*indices_pad_to`` (otherwise -1: no saving).
    """
    if str(mode).lower() != 'auto':
        return -1
    if str(provider).lower() != 'cuda':
        return -1
    if decision_indices_count is None or int(decision_indices_count) <= 0:
        return -1
    if not decision_rows_bounded:
        return -1
    if decision_scale_shape is None or current_scale_shape is None:
        return -1
    if len(decision_scale_shape) != 3 or len(current_scale_shape) != 3:
        return -1
    k_len = int(current_k_len)
    multiple_of = int(multiple_of)
    indices_pad_to = int(indices_pad_to)
    if k_len <= 0 or multiple_of <= 0 or indices_pad_to <= 0:
        return -1

    children = 1
    for d_ax, c_ax in zip(decision_scale_shape, current_scale_shape):
        children *= _max_axis_children(int(d_ax), int(c_ax))

    union_bound = int(decision_indices_count) * children
    if use_band:
        _, ch, cw = (int(x) for x in current_scale_shape)
        union_bound += (2 * max(int(neighbor_frames), 0) + 1) * ch * cw

    # Row-wise: popcount <= min(union_bound, k_len); the kernel pads counts to
    # multiple_of, and the allocation width must additionally satisfy the TMA
    # pad_to contract of the CUDA host entry.
    padded_total_max = _round_up(min(union_bound, k_len), multiple_of)
    w = _round_up(padded_total_max, indices_pad_to)
    pad_n_full = _round_up(k_len, indices_pad_to)
    if w >= pad_n_full:
        return -1        # bound doesn't beat the legacy width -> keep legacy behavior
    return w
