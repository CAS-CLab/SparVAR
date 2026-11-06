import math
from typing import List, Optional, Tuple, Union

import torch
from torch.nn import functional as F

try: import chipmunk     # chipmunk attn
except ImportError: chipmunk = None
import kernels.sparse_attn.triton as sattn_triton
from .sparse_attn_config import GLOBAL_CONFIG, get_kernel_config_attn
# from flag_attn import flash_attention


def pad_qkvo_tensor(tensor, pad_to):
    n = tensor.shape[-2]
    padded_n = ((n + pad_to - 1) // pad_to) * pad_to
    # IMPORTANT: Do not use torch.empty here, it will cause NaN in the Triton kernel!
    padded_tensor = torch.zeros(tensor.shape[:-2] + (padded_n, tensor.shape[-1]), dtype=tensor.dtype, device=tensor.device)
    padded_tensor[..., :n, :] = tensor
    return padded_tensor


def dense_attn(q, k, v, scale=None):
    """support non-casual cross-attention"""

    if GLOBAL_CONFIG['attn']['provider'] == 'tilelang':
        from kernels.sparse_attn.tilelang import dense_attn as tilelang_dense_attn
        return tilelang_dense_attn(q, k, v, scale=scale)
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        o, lse = sattn_triton.dense_attn(q, k, v, scale=scale)
        assert type(lse) == tuple, "LSE must be a tuple"
        assert lse[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
        assert lse[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    else:
        o, lse = torch.ops.chipmunk.dense_attn(q, k, v, scale=scale)
        assert lse.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "LSE shape mismatch"
    
    assert o.shape == q.shape, "Output shape mismatch"
    
    return o, lse


def _colsum_group_rows() -> int:
    """Colsum query-group rows for the decision scale.

    `dcsla_colsum_group_rows` (multiple of 64; {192, 128, 64}) only takes effect
    when D-CSLA is enabled: the legacy CS4A chain (topk / mask_to_indices /
    csp_attn / singleton masks) assumes the kernel-config bm=192 grouping, so
    with dcsla_enabled=false the override is ignored and 192 is always used.
    """
    attn_cfg = GLOBAL_CONFIG['attn']
    if not attn_cfg.get('dcsla_enabled', False):
        return 192
    return int(attn_cfg.get('dcsla_colsum_group_rows', 192))


# ---------------------------------------------------------------
# Flash-attention-based decision colsum (exact; replaces the double full-attention)
#   legacy triton path: dense_attn (o discarded, lse kept) + fused
#     dense_colsum_attn (recomputes o while accumulating cs) => 2x full attention
#   legacy cuda path: chipmunk dense_attn (o + lse) + chipmunk dense_colsum_attn
#     (recomputes the whole attention incl. P@V, o discarded) => 2x full attention
#   fa2/fa3 path: flash_attn_func returns o AND the per-row logsumexp as a free
#     byproduct; colsum_only_attn then adds a single QK^T-only sweep. The o
#     returned comes from the same kernel family the non-decision dense repeats
#     use, so decision-step numerics match the rest of the pipeline.
#     fa3 = flash_attn_interface (Hopper-only build; ~1.25x over both FA2 and
#     chipmunk TK dense on H800) — enable via attn.colsum_fa3.
# ---------------------------------------------------------------
def _use_fa2_colsum() -> bool:
    # Disable this override to use the provider-native decision path.
    return bool(GLOBAL_CONFIG['attn'].get('colsum_fa2', True))


def _use_fa3_colsum() -> bool:
    # attn.colsum_fa3: true routes the decision o/lse through FA3 regardless of
    # provider (checked FIRST in dense_colsum_attn_s/_q); Hopper-only build.
    return bool(GLOBAL_CONFIG['attn'].get('colsum_fa3', False))


# Resolved on the first FA3 call: name of the flash_attn_func kwarg that makes it
# return (out, softmax_lse) — '' means the build returns the tuple by default.
_FA3_LSE_KWARG = None


def _flash_attn_and_lse2(q, k, v, sm_scale, impl='fa2'):
    """FA2/FA3 dense attention on [B, H, N, D] tensors, GQA-native.

    Returns (o [B, H_q, N, D], lse2 [B, H_q, N] fp32) where lse2 is the kernel's
    natural-log row logsumexp scaled by log2(e) — the exp2-domain combined
    logsumexp expected by colsum_only_attn: exp2(q@k*scale*log2e - lse2)
    equals the softmax probability exactly.
    """
    if impl == 'fa3':
        global _FA3_LSE_KWARG
        import flash_attn_interface as _fai  # lazy: Hopper-only FA3 build
        if _FA3_LSE_KWARG is None:
            # The FA3 interface's return convention drifted across revisions:
            # early builds always return (out, softmax_lse); newer ones return a
            # bare out unless asked via a kwarg whose NAME also drifted
            # (return_attn_probs = FA2-compat spelling, return_softmax_lse =
            # kvcache/vllm spelling). Resolve once per process.
            import inspect
            params = inspect.signature(_fai.flash_attn_func).parameters
            _FA3_LSE_KWARG = next((n for n in ('return_attn_probs', 'return_softmax_lse')
                                   if n in params), '')
        lse_kwargs = {_FA3_LSE_KWARG: True} if _FA3_LSE_KWARG else {}
        ret = _fai.flash_attn_func(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
            softmax_scale=sm_scale, **lse_kwargs,
        )
        assert isinstance(ret, tuple) and len(ret) >= 2, (
            f'FA3 flash_attn_func (build {getattr(_fai, "__version__", "?")}) returned a bare '
            f'tensor; lse kwarg resolved to {_FA3_LSE_KWARG!r} — cannot recover softmax_lse'
        )
    else:
        from flash_attn import flash_attn_func  # lazy: keep this module importable without fa2
        ret = flash_attn_func(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3),
            softmax_scale=sm_scale, return_attn_probs=True,
        )
    o_blhd, lse_e = ret[0], ret[1]
    if lse_e.shape[-1] != q.shape[2]:
        # some FA2 builds round seqlen_q up in the returned lse; drop the padding
        lse_e = lse_e[..., :q.shape[2]]
    assert lse_e.shape == (q.shape[0], q.shape[1], q.shape[2]), \
        f'unexpected {impl} lse shape {tuple(lse_e.shape)} for q {tuple(q.shape)}'
    o = o_blhd.permute(0, 2, 1, 3)
    lse2 = (lse_e.float() * 1.4426950408889634).contiguous()
    return o, lse2


def _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa2'):
    sm_scale = 1 / math.sqrt(q.shape[-1]) if scale is None else scale
    o, lse2 = _flash_attn_and_lse2(q, k, v, sm_scale, impl=impl)
    cs = sattn_triton.colsum_only_attn(q, k, lse2, scale=sm_scale, group_rows=group_rows)
    return o, cs


# legacy names kept for bench_colsum_fa2.py
def _fa2_attn_and_lse2(q, k, v, sm_scale):
    return _flash_attn_and_lse2(q, k, v, sm_scale, impl='fa2')


def _dense_colsum_attn_fa2(q, k, v, scale, group_rows):
    return _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa2')


# --------------------------
# for speedup
# --------------------------
def dense_colsum_attn_s(q, k, v, scale=None):
    """
    Compute variable length attention in ThunderKittens.
    """

    provider = GLOBAL_CONFIG['attn']['provider']
    pad_to = get_kernel_config_attn()['bm']
    group_rows = _colsum_group_rows()

    if _use_fa3_colsum() and q.dtype in (torch.bfloat16, torch.float16):
        # FA3 dense (o + lse byproduct) + QK^T-only Triton colsum. Checked BEFORE
        # provider: with provider=cuda only this decision colsum moves to
        # FA3+Triton — the rest of the CS4A chain (csp_attn / mask_to_indices)
        # stays on the chipmunk kernels.
        # (fp32 inputs fall through: flash-attention kernels are half-precision only)
        o, cs = _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa3')

    elif provider == 'cuda':
        # CUDA implementation
        # --- 2 kernel 2 pass get colsum ---
        assert group_rows == 192, \
            f'CUDA dense_colsum_attn is fixed at 192-row groups (3 consumer warpgroups); got dcsla_colsum_group_rows={group_rows}'
        o, lse = torch.ops.chipmunk.dense_attn(q, k, v, scale)     # lse [bs, heads, q_len, 1]
        _, cs, l = torch.ops.chipmunk.dense_colsum_attn(q, k, v, lse, scale)    # cs: [bs, num_heads, ceil(Nq/bm), Nkv]
        assert l.shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)

    elif _use_fa2_colsum() and q.dtype in (torch.bfloat16, torch.float16):
        # FA2 dense (o + lse byproduct) + QK^T-only Triton colsum.
        # (fp32 inputs fall through to the legacy path: FA2 is half-precision only)
        o, cs = _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa2')

    elif provider == 'tilelang':
        from kernels.sparse_attn.tilelang import dense_colsum_attn as tilelang_dense_colsum_attn
        o, cs = tilelang_dense_colsum_attn(q, k, v, scale=scale, group_rows=group_rows)

    else:
        # Triton implementation (legacy)
        # --- fused kernel; host derives lse via an extra internal dense_attn pass ---
        o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, scale=scale, group_rows=group_rows)

        assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)

    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    # assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, k.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)

    return o, cs


# -----------------------
# for Generation quality
# -----------------------
def dense_colsum_attn_q(q, k, v, scale=None):
    """
    Compute variable length attention in ThunderKittens.
    """

    provider = GLOBAL_CONFIG['attn']['provider']
    pad_to = get_kernel_config_attn()['bm']
    group_rows = _colsum_group_rows()

    if _use_fa3_colsum() and q.dtype in (torch.bfloat16, torch.float16):
        # FA3 dense (o + lse byproduct) + QK^T-only Triton colsum (see the
        # dispatch note in dense_colsum_attn_s; checked BEFORE provider).
        o, cs = _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa3')
    elif provider == 'cuda':
        # CUDA implementation
        # --- 2 kernel 2 pass get colsum ---
        o, _ = torch.ops.chipmunk.dense_attn(q, k, v, scale=scale)
        _, lse = sattn_triton.dense_attn(q, k, v, scale=scale)
        _, cs, _ = sattn_triton.dense_colsum_attn(q, k, v, lse, scale=scale, group_rows=group_rows)     # Using Triton correctly cs
    elif _use_fa2_colsum() and q.dtype in (torch.bfloat16, torch.float16):
        # FA2 dense (o + lse byproduct) + QK^T-only Triton colsum.
        # (fp32 inputs fall through to the legacy path: FA2 is half-precision only)
        o, cs = _dense_colsum_attn_flash(q, k, v, scale, group_rows, impl='fa2')

    elif provider == 'tilelang':
        from kernels.sparse_attn.tilelang import dense_colsum_attn as tilelang_dense_colsum_attn
        o, cs = tilelang_dense_colsum_attn(q, k, v, scale=scale, group_rows=group_rows)

    else:
        # Triton implementation (legacy explicit 2-pass)
        o, lse = sattn_triton.dense_attn(q, k, v, scale=scale)

        assert type(lse) == tuple
        assert lse[0].is_contiguous() and lse[1].is_contiguous(), "LSE must be contiguous"

        o, cs, l = sattn_triton.dense_colsum_attn(q, k, v, lse, scale=scale, group_rows=group_rows)

        assert l[0].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[0].shape, q.shape)
        assert l[1].shape == (q.shape[0], q.shape[1], q.shape[2], 1), "L shape mismatch - l: {}, q: {}".format(l[1].shape, q.shape)

    assert o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)
    # assert cs.shape == (q.shape[0], q.shape[1], (q.shape[-2] + pad_to - 1) // pad_to, k.shape[2]), "CS shape mismatch - cs: {}, q: {}".format(cs.shape, q.shape)

    return o, cs


def csp_attn(q, k, v, scale, indices, indices_counts, o, o_scale):
    if GLOBAL_CONFIG['attn']['provider'] == 'tilelang':
        from kernels.sparse_attn.tilelang import csp_attn as tilelang_csp_attn
        return tilelang_csp_attn(q, k, v, scale, indices, indices_counts, o, o_scale)
    # Ignore the n_groups dimension in Python - the kernel will also double check for us!
    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        # Triton path keeps the strict legacy contract: indices width == k_len.
        assert indices.shape == (q.shape[0], q.shape[1], indices.shape[2], k.shape[-2]), "Indices shape mismatch - indices: {}, q: {}, k: {}".format(indices.shape, q.shape, k.shape)
    else:
        # CUDA kernel reads exactly counts[g] entries per group through
        # indices.stride(2) (never up to k_len), so a COMPACT width <= k_len is legal
        # (mask_to_indices out_width path); stride/TMA legality is TORCH_CHECKed by
        # csp_attn.cu itself. Width must still cover every count (guard-checked under
        # SPARVAR_CSP_GUARD=1).
        assert indices.shape[:2] == (q.shape[0], q.shape[1]) and indices.shape[-1] <= k.shape[-2], "Indices shape mismatch - indices: {}, q: {}, k: {}".format(indices.shape, q.shape, k.shape)
    assert indices_counts.shape == indices.shape[:-1], "Indices counts shape mismatch - indices_counts: {}, indices: {}".format(indices_counts.shape, indices.shape)
    assert o is None or o.shape == q.shape, "Output shape mismatch - o: {}, q: {}".format(o.shape, q.shape)

    if GLOBAL_CONFIG['attn']['provider'] == 'triton':
        pad_to = get_kernel_config_attn()['bm']
        # lean=True: skip the discarded M/L byproduct and return ΔO in transpose-friendly
        # [B,N,H,D] storage. Safe because every SparseDiffAttn consumer finishes with
        # `.transpose(1, 2).reshape(B, L, C)` — layout-agnostic, and free on this layout.
        o_delta, _ = sattn_triton.csp_attn(
            q, k, v,
            indices, indices_counts,
            scale=scale,
            lean=True,
        )
        # o=None means "no O_cache to accumulate into" (o would be all zeros):
        # return ΔO directly instead of zeros-init + two full-size elementwise passes.
        if o is None:
            return o_delta if o_scale == 1 else o_delta * o_scale
        o = o + o_delta * o_scale
    else:
        if o is None:
            # The CUDA kernel accumulates into o in place, so it needs a real tensor.
            o = torch.zeros_like(q)
        torch.ops.chipmunk.csp_attn(q, k, v, o, indices, indices_counts, o_scale, scale)

    return o

# __all__ = ['csp_attn', 'dense_attn', 'dense_colsum_attn']


# ========
# bitpack
# ========
# dynamic=True: `mask` width tracks k_len (varies per video), so dynamic=False
# recompiles every video and the retained compile artifacts pin the activations.
# A single symbolic-shape compilation reused across k_len avoids the per-video churn.
@torch.compile(dynamic=True)
def bitpack(mask: torch.Tensor):
    r"""
    Compresses a boolean tensor into a bit-packed uint8 tensor in parallel on the GPU.
    Each output byte encodes 8 bits (True or False) from the input tensor, in little-endian order.

    Args:
        mask (torch.Tensor): A boolean tensor to compress. Must be on the GPU.

    Returns:
        (torch.Tensor, Tuple[int, ...]):
            A tuple of:
            - A 1-D torch.uint8 tensor of length ceil(numel(mask) / 8)
              storing the packed bits on the GPU.
            - The original shape of the mask tensor (for later unpacking).
    """
    original_shape = mask.shape
    # Flatten the tensor
    flat_mask = mask.flatten()
    n = flat_mask.numel()

    # Number of bits we need to pad so that we can reshape into 8 columns
    pad_size = (-n) % 8  # same as: (8 - (n % 8)) % 8

    # Zero-pad if necessary
    flat_mask = torch.cat([flat_mask, flat_mask.new_zeros(pad_size)])

    # Reshape to [N/8, 8], cast to uint8
    flat_mask = flat_mask.view(-1, 8).to(torch.uint8)

    # For each column j, we multiply by 2^j and sum across columns
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=flat_mask.device)).view(1, -1)
    packed = (flat_mask * shifts).sum(dim=1, dtype=torch.uint8).contiguous()  # [N/8]

    return packed, original_shape


# dynamic=True for the same reason as bitpack: avoid per-video recompilation that
# pins activations in retained compile artifacts.
@torch.compile(dynamic=True)
def bitunpack(packed: torch.Tensor, original_shape: Tuple[int, ...]):
    r"""
    Decompresses a bit-packed tensor (uint8) back to a boolean tensor in parallel on the GPU.

    Args:
        packed (torch.Tensor): A 1-D bit-packed tensor of type torch.uint8 on the GPU.
        original_shape (Tuple[int, ...]): The original shape of the boolean tensor.

    Returns:
        torch.Tensor: A boolean tensor of shape original_shape.
    """
    # Compute total number of bits needed
    total_bits = 1
    for dim in original_shape:
        total_bits *= dim

    # Expand the packed bytes to 8 bits each
    # shifts = [1, 2, 4, 8, 16, 32, 64, 128]
    shifts = (2 ** torch.arange(8, dtype=torch.uint8, device=packed.device)).view(1, -1)
    
    # (packed.unsqueeze(1) >> shift) & 1 gives bits; shape => [N_bytes, 8]
    bits_2d = ((packed.unsqueeze(1) & shifts) > 0).to(torch.bool)

    # Flatten and truncate if there was padding
    bits = bits_2d.view(-1)[:total_bits]

    # Reshape to the original shape
    return bits.view(*original_shape)


# ===========
# indexed_io
# ===========
def copy_indices(
    bm_fc1: torch.Tensor, 
    bm_mid_cache: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor
) -> None:
    torch.ops.chipmunk.copy_indices(bm_fc1, bm_mid_cache, indices, counts)


def topk_indices(
    activations: torch.Tensor, 
    indices_out: torch.Tensor, 
    counts_out: torch.Tensor, 
    sparsity_amount: float, 
    multiple_of: int, 
    rk: float
) -> None:
    torch.ops.chipmunk.topk_indices(activations, indices_out, counts_out, sparsity_amount, multiple_of, rk)


def scatter_add(
    packed: torch.Tensor, 
    unpacked: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    num_sms: int
) -> None:
    torch.ops.chipmunk.csp_scatter_add(packed.unsqueeze(0), unpacked.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), num_sms)


def mask_to_indices(
    mask: torch.Tensor,
    multiple_of: int,
    pad_to_multiple_of: int,
    out_width: int = -1,
) -> List[torch.Tensor]:
    # out_width > 0: caller-proven narrower output row width (compact allocation;
    # bound derived in models/sparvar/index_width_utils.py). Passed positionally only
    # when set so the default path stays compatible with a pre-out_width chipmunk
    # build (the op schema defaults it to -1).
    if out_width is not None and out_width > 0:
        return torch.ops.chipmunk.mask_to_indices(mask, multiple_of, pad_to_multiple_of, out_width)
    return torch.ops.chipmunk.mask_to_indices(mask, multiple_of, pad_to_multiple_of)


# ====
# mlp
# ====
def mm1(
    x: torch.Tensor, 
    fc1w: torch.Tensor, 
    sparse_act_packed: torch.Tensor,
    fc1b: torch.Tensor, 
    sparse_act_T: torch.Tensor,
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    scale_a: torch.Tensor = None,
    scale_b: torch.Tensor = None,
) -> None:
    assert x.dtype == torch.bfloat16
    assert sparse_act_packed.dtype == torch.bfloat16
    assert sparse_act_T.dtype == torch.bfloat16
    
    if fc1w.dtype == torch.float8_e4m3fn:
        # FP8 is not supported in CUDA yet!
        # We only support FP8 in Triton for now.
        # NOTE: needs fp8 tensor cores (sm89+); unusable on A100.
        sattn_triton.csp_mlp_mm1_fp8(x, fc1w.T, fc1b, indices, counts, sparse_act_T, sparse_act_packed, scale_a, scale_b)
    elif fc1w.dtype == torch.bfloat16:
        provider = GLOBAL_CONFIG['mlp']['provider']
        if provider == 'triton':
            sattn_triton.csp_mlp_mm1_bf16(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)
        elif provider == 'cuda':
            torch.ops.chipmunk.csp_mlp_mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts)
        else:
            raise ValueError(f"Unsupported provider: {provider}")
    else:
        raise ValueError(f"Unsupported dtype: {fc1w.dtype}")


def mm2_cuda(
    packed: torch.Tensor,
    unpacked_colmajor: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    sparse_act_packed: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    num_sms_scatter_add: int
) -> None:
    assert sparse_act_packed.dtype == torch.bfloat16
    assert fc2wT.dtype == torch.bfloat16
    assert cached_out.dtype == torch.bfloat16
    assert GLOBAL_CONFIG['mlp']['provider'] == 'cuda'
    # Fused implementation uses CUDAGraphs under the hood to allocate x certain # of SMs to scatter_add cache writeback kernel
    # and then uses the rest of the SMs for the actual matmul.
    torch.ops.chipmunk.csp_mlp_mm2_and_scatter_add(packed.unsqueeze(0), unpacked_colmajor.unsqueeze(0), indices.unsqueeze(0), counts.unsqueeze(0), sparse_act_packed.unsqueeze(0), fc2wT.unsqueeze(0), cached_out.unsqueeze(0), num_sms_scatter_add, chipmunk.triton.csp_mlp_mm2_bf16_function_ptr)


def mm2_triton(
    sparse_act_packed: torch.Tensor,
    _: torch.Tensor,
    indices: torch.Tensor,
    counts: torch.Tensor,
    __: torch.Tensor,
    fc2wT: torch.Tensor,
    cached_out: torch.Tensor,
    ___: int
) -> None:
    assert sparse_act_packed.dtype == torch.bfloat16
    assert fc2wT.dtype == torch.bfloat16
    assert cached_out.dtype == torch.bfloat16
    assert GLOBAL_CONFIG['mlp']['provider'] == 'triton'
    # scatter-add is fused into the csp_mlp_mm1 kernel, so we don't have to take care of it!
    sattn_triton.csp_mlp_mm2_bf16(sparse_act_packed, fc2wT, indices, counts, cached_out)


# REASON FOR DISABLING TORCH COMPILE:
# torch inductor likes to insert a dummy Triton kernel between matmul 1 and 2 that just copies data (`triton_fused_poi_2`) 
# that eats up 68us per invocation :(
# uncompiled is still fast because MLPs are very compute-bound and our custom kernels have everything fused in them!
@torch.compiler.disable
def run_e2e(
    x: torch.Tensor, 
    fc1w: torch.Tensor, 
    fc1b: torch.Tensor,
    fc2w_T: torch.Tensor, 
    indices: torch.Tensor, 
    counts: torch.Tensor, 
    sparse_act_T: torch.Tensor, 
    cached_out: torch.Tensor, 
    num_sms_scatter_add: int,
    mm1_scale_a: torch.Tensor = None,
    mm1_scale_b: torch.Tensor = None,
) -> None:
    # Ensure shapes match
    M, K1 = x.shape
    K2, K1_ = fc1w.shape
    assert K1 == K1_, "K1 must match"
    K2_, N = fc2w_T.shape
    assert K2 == K2_, "K2 must match"

    # Packed intermediate activations matrix
    sparse_act_packed = torch.empty((M, K2), device=x.device, dtype=x.dtype)
    
    mm1(x, fc1w, sparse_act_packed, fc1b, sparse_act_T, indices, counts, mm1_scale_a, mm1_scale_b)
    mm2 = mm2_cuda if GLOBAL_CONFIG['mlp']['provider'] == 'cuda' else mm2_triton
    mm2(sparse_act_packed, sparse_act_T, indices, counts, sparse_act_packed, fc2w_T, cached_out, num_sms_scatter_add)


# if __name__ == "__main__":
#     pass
