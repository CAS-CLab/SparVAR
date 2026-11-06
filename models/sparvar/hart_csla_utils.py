"""CSLA mask preparation; runtime objects never enter the HART state dict."""

import math

import torch
import torch.nn.functional as F


CSLA_QUERY_SCALES = (11, 12, 13)
CSLA_PRESETS = {
    "hart_legacy": {
        "attn_sink_scales": 5,
        "kernel_schedule": (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 3, 5, 7),
    },
    "sparvar_legacy": {
        "attn_sink_scales": 6,
        "kernel_schedule": (-1, -1, -1, -1, -1, -1, -1, -1, 1, 1, 1, 1, 3, 5),
    },
}


def get_csla_preset(name):
    if name not in CSLA_PRESETS:
        raise ValueError(f"Unknown HART CSLA preset: {name!r}")
    return dict(CSLA_PRESETS[name])


def resolve_csla_geometry(options, patch_nums, depth):
    preset = get_csla_preset(options.get("csla_preset", "hart_legacy"))
    scales = tuple(options.get("csla_query_scales", CSLA_QUERY_SCALES))
    layers = tuple(options.get("csla_layers", ())) or tuple(range(depth))
    sink = options.get("csla_sink", -1)
    if sink == -1:
        sink = preset["attn_sink_scales"]
    windows = tuple(options.get("csla_windows", ())) or preset["kernel_schedule"]
    block_size = options.get("csla_block_size", 128)
    layout = options.get("csla_kv_layout", "native")
    kernel = options.get("csla_kernel", "flex")
    if not scales or len(set(scales)) != len(scales) or any(s < 1 or s >= len(patch_nums) for s in scales):
        raise ValueError("CSLA query scales must be distinct visual scale indices")
    if len(set(layers)) != len(layers) or any(layer < 0 or layer >= depth for layer in layers):
        raise ValueError("CSLA layers must be distinct valid block indices")
    if not 1 <= sink <= min(scales) + 1:
        raise ValueError("CSLA requires at least one sink scale within every active KV sequence")
    if len(windows) != len(patch_nums) or any(w != -1 and (w < 1 or w % 2 == 0) for w in windows):
        raise ValueError("CSLA windows must provide -1 or a positive odd window for each KV scale")
    if block_size not in (64, 128):
        raise ValueError("CSLA block size must be 64 or 128")
    if layout not in ("native", "contiguous"):
        raise ValueError("CSLA KV layout must be native or contiguous")
    if kernel not in ("flex", "prefix_fa3", "uniform_fa3"):
        raise ValueError("CSLA kernel must be flex, prefix_fa3 or uniform_fa3")
    return dict(query_scales=scales, layers=layers, attn_sink_scales=sink,
                kernel_schedule=tuple(windows), block_size=block_size, kv_layout=layout, kernel=kernel)


def build_csla_block_layout(
    patch_nums, context_token, q_scale_idx, attn_sink_scales, kernel_schedule,
    *, block_size=128, device="cpu",
):
    """Expand the legacy sink/local mask to blocks using actual token geometry.

    Windows are indexed by KV scale. Negative windows outside the sink have
    no connections; they do not select dense attention for those KV scales.
    """
    patch_nums = tuple(patch_nums)
    if len(kernel_schedule) != len(patch_nums):
        raise ValueError("CSLA window schedule must match the HART scale schedule")
    if not 0 < q_scale_idx < len(patch_nums):
        raise ValueError("CSLA query scale must be a visual scale")
    if not 0 <= attn_sink_scales <= q_scale_idx + 1:
        raise ValueError("CSLA sink must be within the active KV scales")
    if context_token <= 0 or block_size <= 0 or any(pn <= 0 for pn in patch_nums):
        raise ValueError("CSLA geometry and block size must be positive")
    query_side = patch_nums[q_scale_idx]
    query_length = query_side ** 2
    kv_lengths = (context_token,) + tuple(pn ** 2 for pn in patch_nums[1:q_scale_idx + 1])
    kv_length = sum(kv_lengths)
    mask = torch.zeros((query_length, kv_length), dtype=torch.bool, device=device)
    mask[:, :sum(kv_lengths[:attn_sink_scales])] = True
    query_positions = torch.arange(query_length, device=device)
    query_y = query_positions // query_side
    query_x = query_positions % query_side
    offset = 0
    for scale_idx, scale_length in enumerate(kv_lengths):
        window = kernel_schedule[scale_idx]
        if scale_idx >= attn_sink_scales and scale_idx > 0 and window > 0:
            kv_side = patch_nums[scale_idx]
            kv_positions = torch.arange(scale_length, device=device)
            # Match the original round-based cross-scale coordinate mapping.
            center_y = torch.round((query_y.float() / query_side) * kv_side)
            center_x = torch.round((query_x.float() / query_side) * kv_side)
            local_y = (center_y[:, None] - (kv_positions // kv_side)[None, :]).abs() <= window // 2
            local_x = (center_x[:, None] - (kv_positions % kv_side)[None, :]).abs() <= window // 2
            mask[:, offset:offset + scale_length] = local_y & local_x
        offset += scale_length
    padded = F.pad(mask, (0, -kv_length % block_size, 0, -query_length % block_size))
    return padded.reshape(
        math.ceil(query_length / block_size), block_size,
        math.ceil(kv_length / block_size), block_size,
    ).any(dim=(1, 3))


class SparseFlexAttn:
    """A non-module callable sharing one prepared mask across all HART layers."""

    def __init__(self, flex_attention, block_mask, query_length, kv_length):
        self.flex_attention = flex_attention
        self.block_mask = block_mask
        self.query_length = query_length
        self.kv_length = kv_length

    def __call__(self, q, k, v, scale=None):
        if q.shape[-2] != self.query_length or k.shape[-2] != self.kv_length:
            raise ValueError("HART CSLA requires the complete query and KV scale geometry")
        return self.flex_attention(
            q.to(v.dtype), k.to(v.dtype), v, block_mask=self.block_mask, scale=scale,
        )


def make_csla_mask_mod(layout, block_size=128):
    def mask_mod(batch, head, q_idx, kv_idx):
        row, col = q_idx // block_size, kv_idx // block_size
        valid = (row < layout.shape[0]) & (col < layout.shape[1])
        return valid & layout[row.clamp_max(layout.shape[0] - 1), col.clamp_max(layout.shape[1] - 1)]

    return mask_mod


class PrefixFA3Attention:
    """Execute a query-independent block mask with sliced or gathered dense FA3."""

    def __init__(self, layout, query_length, kv_length, block_size, allow_gaps=False):
        from flash_attn_interface import flash_attn_func

        kept = int(layout[0].sum().item())
        prefix = bool(layout[:, :kept].all()) and not bool(layout[:, kept:].any())
        if kept == 0 or not bool((layout == layout[:1]).all()) or (not allow_gaps and not prefix):
            raise ValueError("prefix_fa3/uniform_fa3 requires the same nonempty KV selection for every query; prefix_fa3 also requires a prefix")
        self.query_length, self.kv_length = query_length, kv_length
        self.prefix_length = min(kept * block_size, kv_length) if prefix else None
        self.indices = None
        if not prefix:
            blocks = layout[0].nonzero().flatten()
            indices = (blocks[:, None] * block_size + torch.arange(block_size, device=layout.device)).flatten()
            self.indices = indices[indices < kv_length]
        self.attention = flash_attn_func

    def __call__(self, q, k, v, scale=None):
        if q.shape[-2] != self.query_length or k.shape[-2] != self.kv_length:
            raise ValueError("HART prefix attention requires complete query/KV geometry")
        if self.indices is None:
            selected_k, selected_v = k[:, :, :self.prefix_length], v[:, :, :self.prefix_length]
        else:
            selected_k, selected_v = k.index_select(2, self.indices), v.index_select(2, self.indices)
        output = self.attention(
            q.to(v.dtype).transpose(1, 2), selected_k.to(v.dtype).transpose(1, 2),
            selected_v.transpose(1, 2), softmax_scale=scale,
        )
        return output.transpose(1, 2)


def build_csla_attention(patch_nums, context_token, num_heads, *, device, preset,
                         query_scales=CSLA_QUERY_SCALES, attn_sink_scales=None,
                         kernel_schedule=None, block_size=128, kernel="flex"):
    # FlexAttention is imported only when the enabled method is prepared.
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    device = torch.device(device)
    if device.type != "cuda":
        raise ValueError("Enabled HART CSLA preparation requires a CUDA device")
    preset_options = get_csla_preset(preset)
    if attn_sink_scales is not None:
        preset_options["attn_sink_scales"] = attn_sink_scales
    if kernel_schedule is not None:
        preset_options["kernel_schedule"] = kernel_schedule
    compiled_attention = torch.compile(
        flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs",
    ) if kernel == "flex" else None
    prepared = {}
    for scale_idx in query_scales:
        layout = build_csla_block_layout(
            patch_nums, context_token, scale_idx, device=device, block_size=block_size, **preset_options,
        )

        query_length = patch_nums[scale_idx] ** 2
        kv_length = context_token + sum(pn ** 2 for pn in patch_nums[1:scale_idx + 1])
        if kernel in ("prefix_fa3", "uniform_fa3"):
            prepared[scale_idx] = PrefixFA3Attention(layout, query_length, kv_length, block_size, allow_gaps=kernel == "uniform_fa3")
            continue
        block_mask = create_block_mask(
            make_csla_mask_mod(layout, block_size), B=1, H=num_heads, Q_LEN=query_length, KV_LEN=kv_length,
            device=device, BLOCK_SIZE=block_size,
        )
        prepared[scale_idx] = SparseFlexAttn(
            compiled_attention, block_mask, query_length, kv_length,
        )
    return prepared


def resolve_hart_cs4a(options, csla_geometry, patch_nums, depth):
    layers = tuple(options.get('cs4a_layers', ()))
    decision = options.get('cs4a_decision_scale', 10)
    ratio = options.get('cs4a_ratio', 0.5)
    use_cache = options.get('cs4a_use_o_cache', True)
    if len(set(layers)) != len(layers) or any(type(i) is not int or i < 0 or i >= depth for i in layers):
        raise ValueError('CS4A layers must be distinct valid block indices')
    if not isinstance(use_cache, bool) or type(ratio) not in (int, float) or not 0 < ratio <= 1:
        raise ValueError('CS4A requires a boolean cache option and ratio in (0,1]')
    if type(decision) is not int or not 1 <= decision < len(patch_nums):
        raise ValueError('CS4A decision must be a valid visual scale')
    if layers and decision >= min(csla_geometry['query_scales']):
        raise ValueError('CS4A decision must precede every sparse query scale')
    if layers:
        if options.get('dense_attn_backend') != 'fa3' or options.get('hart_fa3_start_scale', 0) > decision:
            raise ValueError('CS4A requires FA3 at the decision scale for a matched dense output/LSE')
        if not set(layers).issubset(csla_geometry['layers']):
            raise ValueError('CS4A layers must be a subset of the configured sparse layers')
        if csla_geometry['attn_sink_scales'] > decision + 1:
            raise ValueError('CS4A sink must exist at the decision scale')
    return dict(layers=layers, decision_scale=decision, ratio=ratio, use_o_cache=use_cache,
                query_group_rows=192, mapping='relative_points_with_identity_sink')


def build_hart_cs4a_maps(patch_nums, context_token, query_scales, decision_scale, sink_scales, *, device):
    lengths = [context_token] + [pn * pn for pn in patch_nums[1:]]
    cumulative = [0]
    for length in lengths:
        cumulative.append(cumulative[-1] + length)
    source_len = cumulative[decision_scale + 1]
    sink_len = cumulative[sink_scales]
    source_groups = math.ceil(patch_nums[decision_scale] ** 2 / 192)
    maps = {}
    for target in query_scales:
        # Keep the text/sink prefix literal; align selected visual scales by their distance to the current scale.
        lut = torch.arange(source_len, dtype=torch.int32)
        for source in range(sink_scales, decision_scale + 1):
            destination = source + target - decision_scale
            old_side, new_side = patch_nums[source], patch_nums[destination]
            coordinates = torch.arange(old_side * old_side)
            row, col = coordinates // old_side, coordinates % old_side
            new_row = ((2 * row + 1) * new_side) // (2 * old_side)
            new_col = ((2 * col + 1) * new_side) // (2 * old_side)
            lut[cumulative[source]:cumulative[source + 1]] = cumulative[destination] + new_row * new_side + new_col
        target_groups = math.ceil(patch_nums[target] ** 2 / 192)
        groups = F.interpolate(torch.arange(source_groups).float()[None, None], size=target_groups, mode='nearest')[0, 0].long()
        maps[target] = (lut.to(device), groups.to(device), patch_nums[target])
    return maps, source_len, sink_len


def hart_cs4a_decision(q, k, v, scale, keep, sink_len, use_o_cache):
    from flash_attn_interface import flash_attn_func
    from kernels.sparse_attn.triton.attn_colsum_only import colsum_only_attn
    from kernels.sparse_attn.triton.attn_csp import csp_attn

    output, lse, *_ = flash_attn_func(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2),
                                     softmax_scale=scale, return_attn_probs=True)
    dense = output.transpose(1, 2)
    lse2 = (lse[..., :q.shape[2]].float() * 1.4426950408889634).contiguous()
    # Single-writer 64-row groups, then a deterministic reduction to the paper's 192-row index groups.
    mass = colsum_only_attn(q, k, lse2, scale=scale, group_rows=64)
    groups = math.ceil(q.shape[2] / 192)
    mass = F.pad(mass, (0, 0, 0, groups * 3 - mass.shape[2])).reshape(q.shape[0], q.shape[1], groups, 3, k.shape[2]).sum(3)
    dynamic = mass[..., sink_len:].topk(keep - sink_len, dim=-1).indices + sink_len
    sink = torch.arange(sink_len, device=q.device).expand(*dynamic.shape[:-1], -1)
    indices = torch.cat((sink, dynamic), dim=-1).sort(dim=-1).values
    offset = None
    if use_o_cache:
        counts = torch.full(indices.shape[:-1], keep, dtype=torch.int32, device=q.device)
        old_sparse = csp_attn(q, k, v, indices.int(), counts, scale=scale, lean=True)[0]
        offset = dense - old_sparse
    return dense, (indices, offset)


def hart_cs4a_sparse(q, k, v, scale, state, mapping, keep, source_side):
    from kernels.sparse_attn.triton.attn_csp import csp_attn

    indices, offset = state
    lut, query_groups, target_side = mapping
    mapped = lut[indices.index_select(2, query_groups)].sort(dim=-1).values
    counts = torch.full(mapped.shape[:-1], keep, dtype=torch.int32, device=q.device)
    output = csp_attn(q, k, v, mapped, counts, scale=scale, lean=True)[0]
    if offset is not None:
        batch, heads, _, dim = offset.shape
        spatial = offset.permute(0, 1, 3, 2).reshape(batch, heads * dim, source_side, source_side)
        resized = F.interpolate(spatial, size=(target_side, target_side), mode='bilinear', align_corners=False)
        output = output + resized.reshape(batch, heads, dim, target_side ** 2).permute(0, 1, 3, 2)
    return output
