"""GPU-resident, count-aware accumulation of image CS4A identity indices.

The state follows the last sparse scale, independently of the decision O_cache.
Only counts-valid indices are inherited, including keys used for count alignment;
unused/uninitialized row tails never enter the next selection.
"""

from functools import lru_cache

import torch

from models.sparvar.index_map_utils import (
    _get_image_local_window_mask,
    _mask_to_indices,
)


def validate_identity_cumulative_config(config):
    if (config.get('cs4a_image_expand_mode', 'center_union') != 'center_union'
            or config.get('cs4a_image_kv_map', 'relative') != 'identity'
            or config.get('cs4a_qgroup_map_mode', 'nearest') != 'nearest'
            or config.get('dcsla_enabled', False)):
        raise ValueError(
            'cs4a_identity_cumulative requires center_union, identity, '
            'nearest Q-group mapping and dcsla_enabled=false.'
        )


@lru_cache(maxsize=32)
def _nearest_groups(source_groups, target_groups, device):
    # Match map_inds_counts_across_scales, including torch.round ties-to-even.
    centers = torch.arange(target_groups, device=device, dtype=torch.float32) + 0.5
    return torch.round(centers / target_groups * source_groups - 0.5).long().clamp_(
        0, source_groups - 1)


@lru_cache(maxsize=32)
def _column_ranks(width, device):
    return torch.arange(width, device=device).view(1, 1, 1, width)


def _union_count_bound(source_width, sink_len, q_len, shape, bm, win_size, k_len,
                       multiple_of, indices_pad_to):
    """Conservative host-only bound; no counts.max().item() synchronization.

    A contiguous query group touches at most ceil((bm+w-1)/w) image rows.
    Dilation adds at most win-1 rows. Charging their entire width also bounds
    groups wrapping a row boundary, and overcounts overlaps safely.
    """
    h, w = int(shape[-2]), int(shape[-1])
    touched_rows = (min(int(bm), int(q_len)) + w - 2) // w + 1
    local_bound = min(h, touched_rows + int(win_size) - 1) * w
    logical_bound = min(int(k_len), int(source_width) + int(sink_len) + local_bound)
    rounded = (logical_bound + multiple_of - 1) // multiple_of * multiple_of
    return (rounded + indices_pad_to - 1) // indices_pad_to * indices_pad_to


class CumulativeImageIdentityIndices:
    def __init__(self):
        self.clear()

    def clear(self):
        self.indices = None
        self.counts = None
        self.scale_ind = None
        self.kv_len = None

    def reset(self, indices, *, scale_ind, kv_len):
        """Seed with compact, uniform Top-K indices at every new decision step."""
        self.indices = indices
        self.counts = None  # Every column in this compact decision tensor is valid.
        self.scale_ind = int(scale_ind)
        self.kv_len = int(kv_len)

    @torch.no_grad()
    def advance(self, *, scale_ind, batch_size, q_len, kv_len, scale_shape,
                sink_len, bm, win_size, multiple_of, indices_pad_to, provider):
        if self.indices is None:
            raise RuntimeError('Cumulative identity requires a decision step first.')
        if int(scale_ind) <= self.scale_ind or int(kv_len) <= self.kv_len:
            raise ValueError('Cumulative identity requires increasing image scales/KV lengths.')
        if not 0 < batch_size <= self.indices.shape[0]:
            raise ValueError('Cumulative identity permits batch shrink, not batch growth.')

        source = self.indices[:batch_size]
        groups = (int(q_len) + bm - 1) // bm
        row_map = _nearest_groups(source.shape[2], groups, source.device)
        selected = source.index_select(2, row_map)
        local = _get_image_local_window_mask(
            scale_shape, int(kv_len) - int(q_len), kv_len, bm, q_len,
            win_size, source.device)
        union = local.view(1, 1, groups, kv_len).expand(
            batch_size, source.shape[1], groups, kv_len).clone()
        sink_len = max(0, min(int(sink_len), int(kv_len)))
        if sink_len:
            union[..., :sink_len] = True

        if self.counts is None:
            union.scatter_(-1, selected.long(), True)
        else:
            counts = self.counts[:batch_size].index_select(2, row_map)
            valid = _column_ranks(source.shape[-1], source.device) < counts.unsqueeze(-1)
            # Mask BEFORE indexing: unused CUDA output tails may contain arbitrary data.
            safe = torch.where(valid, selected, 0).long()
            # Invalid lanes target column 0 with False. OR reduction prevents them
            # from erasing a real selected key or a local/sink key at that column.
            # CUDA scatter_reduce does not support Bool. Byte views share storage
            # (both dtypes occupy one byte), so this OR needs no dtype-copy kernel.
            union.view(torch.uint8).scatter_reduce_(
                -1, safe, valid.view(torch.uint8), reduce='amax', include_self=True)

        bound = _union_count_bound(
            source.shape[-1], sink_len, q_len, scale_shape, bm, win_size, kv_len,
            multiple_of, indices_pad_to)
        full_width = (int(kv_len) + indices_pad_to - 1) // indices_pad_to * indices_pad_to
        # The CUDA CSP kernel accepts compact, TMA-aligned rows. Keep the existing
        # full-width contract for Triton. No data-dependent allocation or CPU sync.
        out_width = bound if provider == 'cuda' and bound < full_width else -1
        indices, counts = _mask_to_indices(union, multiple_of, indices_pad_to, out_width)
        indices = indices[..., :min(int(kv_len), indices.shape[-1])]
        # Save views/references, not clones; keep only the latest sparse selection.
        self.indices = indices[..., :min(bound, indices.shape[-1])]
        self.counts = counts
        self.scale_ind = int(scale_ind)
        self.kv_len = int(kv_len)
        return indices, counts
