"""
Definitions of blocks of VAR transformer model.
"""

import math
import os
import sys
debugger_attached = hasattr(sys, 'gettrace') and sys.gettrace() is not None
from functools import partial
from typing import Optional, Tuple, Union

try: import chipmunk     # chipmunk FA3 attn
except ImportError: chipmunk = None
from kernels.sparse_attn.sparse_attn_ops import dense_attn
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
# Import flash_attn's attention
# from flash_attn_interface import flash_attn_func; print('#'*10 + ' use FA3 ' + '#'*10)
from flash_attn import flash_attn_func  # q, k, or v: BLHc, ret: BLHc
from flash_attn import flash_attn_varlen_kvpacked_func  # qkv: N3Hc, ret: NHc
from timm.layers import DropPath, drop_path
from torch.nn.attention import SDPBackend, sdpa_kernel
# not slow attn, for torch >= 2.0.0 and CUDA backend, it support flash attn 2 for fast inference
# https://docs.pytorch.org/docs/2.6/generated/torch.nn.functional.scaled_dot_product_attention.html#torch.nn.functional.scaled_dot_product_attention
# from torch.nn.functional import scaled_dot_product_attention as slow_attn    # q, k, v: BHLc
# -->
from torch.nn.functional import scaled_dot_product_attention as torch_attn  # q, k, v: BHLc
from torch.utils.checkpoint import checkpoint

from utils import lprof
from utils.attn_backend_utils import dense_self_attn

# SparVAR analysis: env-gated attention pattern statistics (default off; the module
# is imported lazily inside the hook so the baseline import graph stays unchanged).
_PATTERN_STATS_ENABLED = os.environ.get("SPARVAR_PATTERN_STATS", "0").lower() in ("1", "true", "yes", "on")

# Import flash_attn's fused ops
try:
    from flash_attn.ops.fused_dense import fused_mlp_func
    from flash_attn.ops.layer_norm import dropout_add_layer_norm
    from flash_attn.ops.rms_norm import dropout_add_rms_norm
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    flash_fused_op_installed = True
except ImportError:
    dropout_add_layer_norm = dropout_add_rms_norm = fused_mlp_func = None
    flash_fused_op_installed = False
    
    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


def precompute_rope2d_freqs_grid(
        dim,
        dynamic_resolution_h_w,
        rope2d_normalized_by_hw,
        pad_to_multiplier=1,
        max_height=2048 // 16,
        max_width=2048 // 16,
        base=10000.0,
        device=None,
        scaling_factor=1.0
    ):
    # split the dimension into half, one for x and one for y
    half_dim = dim // 2
    inv_freq = 1.0 / (base ** (torch.arange(0, half_dim, 2, dtype=torch.int64).float().to(device) / half_dim)) # namely theta, 1 / (10000^(i/half_dim)), i=0,2,..., half_dim-2
    t_height = torch.arange(max_height, device=device, dtype=torch.int64).type_as(inv_freq)
    t_width = torch.arange(max_width, device=device, dtype=torch.int64).type_as(inv_freq)
    t_height = t_height / scaling_factor
    freqs_height = torch.outer(t_height, inv_freq)  # (max_height, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely y*theta
    t_width = t_width / scaling_factor
    freqs_width = torch.outer(t_width, inv_freq)  # (max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2), namely x*theta
    freqs_grid_map = torch.concat([
        freqs_height[:, None, :].expand(-1, max_width, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
        freqs_width[None, :, :].expand(max_height, -1, -1), # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d) / 2)
    ], dim=-1)  # (max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))
    freqs_grid_map = torch.stack([torch.cos(freqs_grid_map), torch.sin(freqs_grid_map)], dim=0)
    # (2, max_height, max_width, dim / (1 for 1d, 2 for 2d, 3 for 3d))

    rope2d_freqs_grid = {}
    for h_div_w in dynamic_resolution_h_w:
        scale_schedule = dynamic_resolution_h_w[h_div_w]['1M']['scales']
        _, ph, pw = scale_schedule[-1]
        max_edge_length = freqs_grid_map.shape[1]
        if ph >= pw:
            uph, upw = max_edge_length, int(max_edge_length / ph * pw)
        else:
            uph, upw = int(max_edge_length / pw * ph), max_edge_length
        rope_cache_list = []
        for (_, ph, pw) in scale_schedule:
            ph_mul_pw = ph * pw
            if rope2d_normalized_by_hw == 1: # downsample
                rope_cache = F.interpolate(freqs_grid_map[:, :uph, :upw, :].permute([0,3,1,2]), size=(ph, pw), mode='bilinear', align_corners=True)
                rope_cache = rope_cache.permute([0,2,3,1]) # (2, ph, pw, half_head_dim)
            elif rope2d_normalized_by_hw == 2: # star stylee
                _, uph, upw = scale_schedule[-1]
                indices = torch.stack([
                    (torch.arange(ph) * (uph / ph)).reshape(ph, 1).expand(ph, pw),
                    (torch.arange(pw) * (upw / pw)).reshape(1, pw).expand(ph, pw),
                ], dim=-1).round().int() # (ph, pw, 2)
                indices = indices.reshape(-1, 2) # (ph*pw, 2)
                rope_cache = freqs_grid_map[:, indices[:,0], indices[:,1], :] # (2, ph*pw, half_head_dim)
                rope_cache = rope_cache.reshape(2, ph, pw, -1)
            elif rope2d_normalized_by_hw == 0:
                rope_cache = freqs_grid_map[:, :ph, :pw, :] # (2, ph, pw, half_head_dim)
            else:
                raise ValueError(f'Unknown rope2d_normalized_by_hw: {rope2d_normalized_by_hw}')
            rope_cache_list.append(rope_cache.reshape(2, ph_mul_pw, -1))
        cat_rope_cache = torch.cat(rope_cache_list, 1) # (2, seq_len, half_head_dim)
        if cat_rope_cache.shape[1] % pad_to_multiplier:
            pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim)
            # for VSCode debugger debug
            # pad = torch.zeros(2, pad_to_multiplier - cat_rope_cache.shape[1] % pad_to_multiplier, half_dim).to(cat_rope_cache.device)
            cat_rope_cache = torch.cat([cat_rope_cache, pad], dim=1)
        cat_rope_cache = cat_rope_cache[:,None,None,None] # (2, 1, 1, 1, seq_len, half_dim)
        for pn in dynamic_resolution_h_w[h_div_w]:
            scale_schedule = dynamic_resolution_h_w[h_div_w][pn]['scales']
            tmp_scale_schedule = [(1, h, w) for _, h, w in scale_schedule]
            rope2d_freqs_grid[str(tuple(tmp_scale_schedule))] = cat_rope_cache
    
    return rope2d_freqs_grid


# `INFINITY_ROPE_FUSED=1` folds the 4 muls + 2 adds into one complex multiply
# (8 kernels -> 1, ~10x less memory traffic): 3.9x faster on the 1M schedule
# (165.6 -> 42.2 ms/image predicted, kernels/bench_rope.py @H800). It is NOT
# bit-identical to the default path -- nvcc contracts a*b - c*d into an fma, so
# results differ by 1 fp32 ulp (max 4.8e-07), which flips ~2e-5 of the elements
# by one bf16 ulp after the cast that follows. Opt-in until that is A/B'd on
# image quality; the default path stays bit-exact with the original code.
_ROPE_FUSED = os.environ.get('INFINITY_ROPE_FUSED', '0').lower() in ('1', 'true', 'yes', 'on')

# Per-call allocations that are recomputed identically on every forward (bias
# concats, cu_seqlens, AdaLN unbind) are cached instead. Bit-exact by
# construction; INFINITY_MICRO_CACHE=0 falls back to recomputing them, which is
# the A/B used by tools/verify_infinity_exactness.py.
_MICRO_CACHE = os.environ.get('INFINITY_MICRO_CACHE', '1').lower() not in ('0', 'false', 'no', 'off')

# Cross-attention text kv projection is reused while the input text tensor is the
# same object (one image = one text tensor across all scales); same GEMM on the
# same inputs, so the reused result is bit-identical. INFINITY_CA_KVPROJ_CACHE=0 disables.
_CA_KVPROJ_CACHE = os.environ.get('INFINITY_CA_KVPROJ_CACHE', '1').lower() not in ('0', 'false', 'no', 'off')

# scale_schedule is rebuilt once per generation but read by every (layer, scale)
# rope call (520x per image for Infinity-8B @1M), so memoize everything derived
# from it: the freqs-grid dict key (a str() of the whole schedule) and the
# per-scale token offsets (previously an np.sum over a list comprehension).
_ROPE_META = {'sched': None, 'key': None, 'starts': None, 'polar': None}


def _rope_scale_meta(scale_schedule, need_starts):
    if _ROPE_META['sched'] is not scale_schedule:
        _ROPE_META.update(sched=scale_schedule, key=str(tuple(scale_schedule)),
                          starts=None, polar=None)
    if need_starts and _ROPE_META['starts'] is None:
        starts, acc = [0], 0
        for item in scale_schedule:
            acc += item[0] * item[1] * item[2]
            starts.append(acc)
        _ROPE_META['starts'] = starts
    return _ROPE_META['key'], _ROPE_META['starts']


def _rope_polar(grid):
    """cos + i*sin view of the whole freqs grid, built once per generation."""
    polar = _ROPE_META['polar']
    if polar is None or polar.device != grid.device:
        polar = torch.view_as_complex(torch.stack((grid[0], grid[1]), dim=-1).contiguous())
        _ROPE_META['polar'] = polar
    return polar


def _apply_rope_1_fused(x, polar, dtype):
    shape = x.shape
    xc = torch.view_as_complex(x.to(dtype).reshape(*shape[:-1], -1, 2).contiguous())
    return torch.view_as_real(xc * polar).reshape(shape)


def apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, pad_to_multiplier, rope2d_normalized_by_hw, scale_ind):
    device_type = q.device.type
    device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"

    with torch.autocast(device_type=device_type, enabled=False):
        seq_len = q.shape[-2]   # == the old qk.shape[3], for BHLc and BLHc alike
        start = 0
        key, starts = _rope_scale_meta(scale_schedule, need_starts=scale_ind >= 1)
        if scale_ind >= 1:
            assert len(scale_schedule[0]) == 3
            start = starts[scale_ind]
        grid = rope2d_freqs_grid[key]
        if grid.device != q.device:
            grid = grid.to(q.device)
            rope2d_freqs_grid[key] = grid
            _ROPE_META['polar'] = None
        assert start+seq_len <= grid.shape[4]
        if _ROPE_FUSED:
            polar = _rope_polar(grid)[:, :, :, start:start+seq_len]  # (1, 1, 1, seq_len, half_head_dim)
            dtype = torch.promote_types(q.dtype, k.dtype)   # torch.stack((q, k)) promoted both
            return _apply_rope_1_fused(q, polar, dtype), _apply_rope_1_fused(k, polar, dtype)
        rope_cache = grid[:, :, :, :, start:start+seq_len] # rope_cache shape: [2, 1, 1, 1, seq_len, half_head_dim]
        # rotating q and k separately is bit-identical but needs 6 more kernel
        # launches, which costs more than the saved copy in the launch-bound
        # small scales (kernels/bench_rope.py: 1.20x slower at L <= 400), so the
        # stacked form stays the exact path.
        qk = torch.stack((q, k), dim=0)  #(2, batch_size, heads, seq_len, head_dim)
        qk = qk.reshape(*qk.shape[:-1], -1, 2) #(2, batch_size, heads, seq_len, half_head_dim, 2)
        qk = torch.stack([
            rope_cache[0] * qk[...,0] - rope_cache[1] * qk[...,1],
            rope_cache[1] * qk[...,0] + rope_cache[0] * qk[...,1],
        ], dim=-1) # (2, batch_size, heads, seq_len, half_head_dim, 2), here stack + reshape should not be concate
        qk = qk.reshape(*qk.shape[:-2], -1) #(2, batch_size, heads, seq_len, head_dim)
        return qk.unbind(dim=0) # (batch_size, heads, seq_len, head_dim)


class FastRMSNorm(nn.Module):
    def __init__(self, C, eps=1e-6, elementwise_affine=True):
        super().__init__()
        self.C = C
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if self.elementwise_affine:
            self.weight = nn.Parameter(torch.ones(C))
        else:
            self.register_buffer('weight', torch.ones(C))
    
    def forward(self, x):
        src_type = x.dtype
        return rms_norm_impl(x.float(), self.weight, epsilon=self.eps).to(src_type)
    
    def extra_repr(self) -> str:
        return f'C={self.C}, eps={self.eps:g}, elementwise_affine={self.elementwise_affine}'


def get_dropout_layer(p):
    return nn.Dropout(p, inplace=True) if p > 0 else nn.Identity()


class FFN(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = fused_mlp_func if fused_mlp else None
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU(approximate='tanh')
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = get_dropout_layer(drop)
        self.heuristic = -1
    
    def forward(self, x):
        if self.fused_mlp_func is not None:
            return self.drop(self.fused_mlp_func(
                x=x,
                weight1=self.fc1.weight,
                weight2=self.fc2.weight,
                bias1=self.fc1.bias,
                bias2=self.fc2.bias,
                activation='gelu_approx',
                save_pre_act=self.training,
                return_residual=False,
                checkpoint_lvl=0,
                heuristic=self.heuristic,
                process_group=None,
            ))
        else:
            return self.drop(self.fc2( self.act(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class FFNSwiGLU(nn.Module):
    def __init__(self, in_features, hidden_features, out_features=None, drop=0., fused_mlp=False):
        super().__init__()
        self.fused_mlp_func = None
        hidden_features = round(2 * hidden_features / 3 / 256) * 256
        
        out_features = out_features or in_features
        self.fcg = nn.Linear(in_features, hidden_features, bias=False)
        self.fc1 = nn.Linear(in_features, hidden_features, bias=False)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=False)
        self.drop = get_dropout_layer(drop)
    
    def forward(self, x):
        return self.drop(self.fc2( F.silu(self.fcg(x), inplace=True).mul_(self.fc1(x)) ))
    
    def extra_repr(self) -> str:
        return f'fused_mlp={self.fused_mlp_func is not None}'


class SelfAttention(nn.Module):
    def __init__(
        self, embed_dim=768, num_heads=12,
        proj_drop=0., tau=1, cos_attn=False, customized_flash_attn=True, use_flex_attn=False, 
        batch_size=2, pad_to_multiplier=1, rope2d_normalized_by_hw=0,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        :param proj_drop: always 0 for testing
        :param tau: always 1
        :param cos_attn: always True: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        :param customized_flash_attn:
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        self.using_flash = customized_flash_attn
        
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.tau, self.cos_attn = tau, cos_attn
        if self.cos_attn:
            self.scale = 1
            size = (1, 1, self.num_heads, 1) if self.using_flash else (1, self.num_heads, 1, 1)
            # size: 11H1 or 1H11
            self.scale_mul_1H11 = nn.Parameter(torch.full(size=size, fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim) / self.tau
        
        self.mat_qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.q_bias, self.v_bias = nn.Parameter(torch.zeros(embed_dim)), nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)
        
        self.caching = False    # kv caching: only used during inference
        self.cached_k = None    # kv caching: only used during inference
        self.cached_v = None    # kv caching: only used during inference

        self.batch_size = batch_size
        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw

    def kv_caching(self, enable: bool): # kv caching: only used during inference
        self.caching = enable
        self.cached_k = None
        self.cached_v = None

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]],
                attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0,
                layer_idx=0):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        :param (fp32) attn_bias_or_two_vector:
                if not using_flash:
                    a block-wise, lower-triangle matrix, like:
                    [[[[0, -, -, -, -, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, -, -, -, -, -, -, -, -, -],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]]]]
                    where 0 means visible and - means invisible (-inf)
                else:
                    a tuple of two 1-dim int vector (VAR_visible_kvlen, VAR_invisible_qlen)
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be shared
        """
        # x: fp32
        B, L, C = x.shape
        
        # qkv: amp, bf16
        qkv = (F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
               .view(B, L, 3, self.num_heads, self.head_dim))  # BL3Hc
        if self.using_flash: q, k, v = qkv.unbind(dim=2); L_dim = 1           # q or k or v: all are shaped in (B:batch_size, L:seq_len, H:heads, c:head_dim)
        else: q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0); L_dim = 2   # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim)
        
        if self.cos_attn:   # always True
            scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp() # 11H1 (flash), or 1H11 (not flash)
            q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()   # fp32
            k = F.normalize(k, dim=-1, eps=1e-12).contiguous()                  # fp32
            v = v.contiguous()                                                  # bf16
        else:   # be contiguous, to make kernel happy
            q = q.contiguous()      # bf16
            k = k.contiguous()      # bf16
            v = v.contiguous()      # bf16
        if rope2d_freqs_grid is not None:
            q, k = apply_rotary_emb(q, k, scale_schedule, rope2d_freqs_grid, self.pad_to_multiplier, self.rope2d_normalized_by_hw, scale_ind) #, freqs_cis=freqs_cis)

        # drop uncond branch: once input batch shrinks 2B -> B, shrink kv cache to the cond half
        if self.cached_k is not None and 2 * B == self.cached_k.shape[0]:
            self.cached_k = self.cached_k[:B]; self.cached_v = self.cached_v[:B]

        if self.caching:    # kv caching: only used during inference
            if self.cached_k is None: self.cached_k = k; self.cached_v = v
            else: k = self.cached_k = torch.cat((self.cached_k, k), dim=L_dim); v = self.cached_v = torch.cat((self.cached_v, v), dim=L_dim)
            # print(f'[scale-{scale_ind}_layer-{layer_idx}] self-attn caching: q shape: {q.shape}, k shape: {k.shape}, v shape: {v.shape}')

            #* SparVAR analysis hook: attention pattern statistics (no-op unless
            #* SPARVAR_PATTERN_STATS=1). q/k here are post-norm/RoPE/cache-concat.
            if _PATTERN_STATS_ENABLED:
                from models.sparvar.attn_pattern_stats_image_utils import \
                    maybe_record_infinity_attention_stats
                maybe_record_infinity_attention_stats(
                    q, k, scale=self.scale, scale_ind=scale_ind, layer_ind=layer_idx,
                    scale_schedule=scale_schedule, is_bhlc=not self.using_flash)

        if self.using_flash:
            if attn_bias_or_two_vector is not None: # training
                kw = dict(VAR_visible_kvlen=attn_bias_or_two_vector[0], VAR_invisible_qlen=attn_bias_or_two_vector[1])
            else:                                   # inference (autoregressive sampling)
                kw = dict()
            oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, dropout_p=0, softmax_scale=self.scale, **kw).view(B, L, C)
        else:
            # if self.cos_attn: q, k are in fp32; v is in bf16
            # else: q, k, v are in bf16
            if self.use_flex_attn and attn_fn is not None:
                oup = attn_fn(q, k, v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
            else:
                # torch FA2 / chipmunk switch via env LVAR_ATTN_BACKEND; this file's
                # historical default is the chipmunk CUDA kernel
                oup = (dense_self_attn(q, k, v, scale=self.scale, attn_mask=attn_bias_or_two_vector,
                                       default_backend='chipmunk')
                       .transpose(1, 2)
                       .reshape(B, L, C))

                # --- FA3 for H100 ---
                # q, k, v = q.transpose(1,2), k.transpose(1,2), v.transpose(1,2)
                # oup = flash_attn_func(q.to(v.dtype), k.to(v.dtype), v, softmax_scale=self.scale).view(B, L, C)

                # --- Triton attn ---
                # o, _ = dense_attn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale)
                # oup = o.transpose(1, 2).reshape(B, L, C)
            # oup: bf16
        
        return self.proj_drop(self.proj(oup))

    def extra_repr(self) -> str:
        tail = ''
        return f'using_flash={self.using_flash}, tau={self.tau}, cos_attn={self.cos_attn}{tail}'


class SelfAttnBlock(nn.Module):
    def __init__(
        self, embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
    ):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = SelfAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            proj_drop=drop,
            tau=tau,
            cos_attn=cos_attn,
            customized_flash_attn=customized_flash_attn,
            attn_fn = attn_fn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(
            in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        
        self.shared_aln = shared_aln
        if self.shared_aln:
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector):  # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        with torch.amp.autocast('cuda', enabled=False):
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.fused_ada_norm is None:
            x = x + self.drop_path(self.attn( self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1), attn_bias_or_two_vector=attn_bias_or_two_vector ).mul_(gamma1))
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            x = x + self.drop_path(self.attn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1), attn_bias_or_two_vector=attn_bias_or_two_vector).mul_(gamma1))
            x = x + self.drop_path(self.ffn(self.fused_ada_norm(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}'


class CrossAttention(nn.Module):
    def __init__(
        self, for_attn_pool=False, embed_dim=768, kv_dim=4096, num_heads=12,
        proj_drop=0., cos_attn=False,
    ):
        """
        :param for_attn_pool: only used in VAR.text_proj_for_sos
        :param embed_dim: Q's dim
        :param kv_dim: K's and V's dim
        :param num_heads: num heads of multi-head attention
        :param proj_drop: proj drop out
        :param cos_attn: during attention, q and k will be L2-normalized and scaled by a head-wise learnable parameter self.scale_mul_1H11
        """
        cos_attn = False    # todo: never use cos attn in cross attention with T5 kv
        super().__init__()
        self.for_attn_pool = for_attn_pool
        self.embed_dim = embed_dim
        self.kv_dim = kv_dim
        assert embed_dim % num_heads == 0
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads  # =64
        self.cos_attn = cos_attn
        if self.cos_attn:
            self.scale = 1
            self.scale_mul_1H1 = nn.Parameter(torch.full(size=(1, self.num_heads, 1, 1), fill_value=4.0).log(), requires_grad=True)
            self.max_scale_mul = torch.log(torch.tensor(100)).item()
        else:
            self.scale = 1 / math.sqrt(self.head_dim)
        
        if for_attn_pool:
            q = torch.empty(1, self.num_heads, self.head_dim)
            nn.init.trunc_normal_(q, mean=0, std=math.sqrt(1 / embed_dim / 3))
            self.mat_q = nn.Parameter(q)
        else:
            self.mat_q = nn.Linear(embed_dim, embed_dim, bias=True)
        self.mat_kv = nn.Linear(kv_dim, embed_dim*2, bias=False)
        self.v_bias = nn.Parameter(torch.zeros(embed_dim))
        self.register_buffer('zero_k_bias', torch.zeros(embed_dim))
        # both are rebuilt on every forward otherwise (520 allocations + launches
        # per image each); cached values are identical, so results do not change
        self.register_buffer('_kv_bias_cache', None, persistent=False)
        self._cu_seqlens_q_cache = {}       # (B, Lq, device) -> int32 offsets
        # (input kv_compact, projected kv) of the last call: the text kv is the same
        # tensor object for every (layer, scale) of one image, so its projection is
        # computed once per image per layer instead of once per scale
        self._kvproj_cache = None

        self.proj = nn.Linear(embed_dim, embed_dim)
        self.proj_drop = get_dropout_layer(proj_drop)

    def _kv_bias(self):
        stale = self.training or not _MICRO_CACHE
        bias = self._kv_bias_cache
        if (stale or bias is None or bias.dtype != self.v_bias.dtype
                or bias.device != self.v_bias.device):
            bias = torch.cat((self.zero_k_bias, self.v_bias))
            self._kv_bias_cache = None if stale else bias
        return bias

    def _cu_seqlens_q(self, B, Lq, device):
        key = (B, Lq, device)
        out = self._cu_seqlens_q_cache.get(key) if _MICRO_CACHE else None
        if out is None:
            out = torch.arange(0, Lq * (B + 1), Lq, dtype=torch.int32, device=device)
            if _MICRO_CACHE:
                self._cu_seqlens_q_cache[key] = out
        return out
    
    def forward(self, q, ca_kv, scale_ind=None, layer_ind=None):
        """
        :param q: shaped as (batch, seq_len, Q_dim)
        :param ca_kv: contains several vectors, each of which is shaped as (len_i, KV_dim). We have [len_1xKV_dim, len_2xKV_dim, len_3xKV_dim, ...] and lens == [len_1, len_2, len_3, ...]
            - kv_compact: shaped as (sum(lens), KV_dim)
            - cu_seqlens_k: cumulated sum of lens
            - max_seqlen_k: int, max(lens)
        NOTE: seq_len (num of Qs) can reach 10k;  but len_i (num of KVs) must <= 256
        
        :return: shaped as (batch, seq_len, Q_dim)
        """
        kv_compact, cu_seqlens_k, max_seqlen_k = ca_kv
        N = kv_compact.shape[0]
        # lprof ctx: only stamped when the caller knows where it is in the AR loop
        _pctx = {} if scale_ind is None else dict(si=scale_ind, layer=layer_ind)

        with lprof.region('ca.kvproj', **_pctx):
            cached = self._kvproj_cache
            use_cache = _CA_KVPROJ_CACHE and not self.training
            if use_cache and cached is not None and cached[0] is kv_compact:
                kv_compact = cached[1]      # same text tensor as the previous call: reuse its projection
            else:
                kv_in = kv_compact
                kv_compact = F.linear(kv_compact, weight=self.mat_kv.weight, bias=self._kv_bias()).view(N, 2, self.num_heads, self.head_dim) # NC => N2Hc
                self._kvproj_cache = (kv_in, kv_compact) if use_cache else None
        # attn_bias = xformers.ops.fmha.BlockDiagonalMask.from_seqlens

        if not self.for_attn_pool:
            B, Lq = q.shape[:2]
            with lprof.region('ca.qproj', **_pctx):
                q_compact = self.mat_q(q).view(-1, self.num_heads, self.head_dim)
        else:
            B = cu_seqlens_k.shape[0] - 1
            Lq = 1
            q_compact = self.mat_q.repeat(B, 1, 1).to(dtype=kv_compact.dtype)
        
        if self.cos_attn:   # always False
            scale_mul = self.scale_mul_1H1.clamp_max(self.max_scale_mul).exp()
            k, v = kv_compact.unbind(dim=1)
            q_compact = F.normalize(q_compact, dim=-1).mul(scale_mul)
            k = F.normalize(k, dim=-1)
            kv_compact = torch.stack((k, v), dim=1)
        
        q_compact = q_compact.contiguous()
        kv_compact = kv_compact.contiguous()
        
        cu_seqlens_q = self._cu_seqlens_q(B, Lq, q_compact.device)
        with lprof.region('ca.core', k_len=int(max_seqlen_k), **_pctx):
            if q_compact.dtype == torch.float32:    # todo: fp16 or bf16?
                oup = flash_attn_varlen_kvpacked_func(q=q_compact.to(dtype=torch.bfloat16), kv=kv_compact.to(dtype=torch.bfloat16), cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)
                oup = oup.float()
            else:
                oup = flash_attn_varlen_kvpacked_func(q=q_compact, kv=kv_compact, cu_seqlens_q=cu_seqlens_q, cu_seqlens_k=cu_seqlens_k, max_seqlen_q=Lq, max_seqlen_k=max_seqlen_k, dropout_p=0, softmax_scale=self.scale).reshape(B, Lq, -1)

        with lprof.region('ca.proj', **_pctx):
            oup = self.proj_drop(self.proj(oup))
        return oup
    
    def extra_repr(self) -> str:
        return f'Cq={self.embed_dim}, Ckv={self.kv_dim}, cos_attn={self.cos_attn}'


class CrossAttnBlock(nn.Module):
    def __init__(
        self,
        embed_dim, kv_dim, cross_attn_layer_scale, cond_dim, act: bool, shared_aln: bool, norm_layer: partial,
        num_heads, mlp_ratio=4., drop=0., drop_path=0., tau=1, cos_attn=False,
        swiglu=False, customized_flash_attn=False, fused_mlp=False, fused_norm_func=None, checkpointing_sa_only=False,
        use_flex_attn=False, batch_size=2, pad_to_multiplier=1, apply_rope2d=False, rope2d_normalized_by_hw=False,
    ):
        super(CrossAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.drop_path_rate = drop_path
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.sa = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, proj_drop=drop, tau=tau, cos_attn=cos_attn, customized_flash_attn=customized_flash_attn,
            use_flex_attn=use_flex_attn, batch_size=batch_size, pad_to_multiplier=pad_to_multiplier, rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        )
        self.ca = CrossAttention(embed_dim=embed_dim, kv_dim=kv_dim, num_heads=num_heads, proj_drop=drop, cos_attn=cos_attn)
        self.using_swiglu = swiglu
        self.ffn = (FFNSwiGLU if swiglu else FFN)(in_features=embed_dim, hidden_features=round(embed_dim * mlp_ratio / 256) * 256, drop=drop, fused_mlp=fused_mlp)
        
        self.ln_wo_grad = norm_layer(embed_dim, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        self.ca_norm = norm_layer(embed_dim, elementwise_affine=True)
        
        self.shared_aln = shared_aln
        if self.shared_aln: # always True
            self.ada_gss = nn.Parameter(torch.randn(1, 1, 6, embed_dim) / embed_dim**0.5)
        else:
            lin = nn.Linear(cond_dim, 6*embed_dim)
            self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
        
        if cross_attn_layer_scale >= 0:
            self.ca_gamma = nn.Parameter(cross_attn_layer_scale * torch.ones(embed_dim), requires_grad=True)
        else:
            self.ca_gamma = 1
        
        self.checkpointing_sa_only = checkpointing_sa_only
    
    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, scale_schedule=None, rope2d_freqs_grid=None, scale_ind=0, 
                layer_idx=0
    ):    # todo: minGPT and vqgan also uses pre-norm, just like this, while MaskGiT uses post-norm
        with torch.autocast('cuda', enabled=False):     # disable half precision
            if self.shared_aln: # always True;                   (1, 1, 6, C)  + (B, 1, 6, C)
                gamma1, gamma2, scale1, scale2, shift1, shift2 = (self.ada_gss + cond_BD).unbind(2) # 116C + B16C =unbind(2)=> 6 B1C
            else:
                gamma1, gamma2, scale1, scale2, shift1, shift2 = self.ada_lin(cond_BD).view(-1, 1, 6, self.C).unbind(2)
        
        if self.fused_norm_func is None:
            x_sa = self.ln_wo_grad(x.float()).mul(scale1.add(1)).add_(shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind,
                               layer_idx=layer_idx)      #* add scale_ind
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn( self.ln_wo_grad(x.float()).mul(scale2.add(1)).add_(shift2) ).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        else:
            x_sa = self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale1, shift=shift1)
            if self.checkpointing_sa_only and self.training:
                x_sa = checkpoint(self.sa, x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, use_reentrant=False)
            else:
                x_sa = self.sa(x_sa, attn_bias_or_two_vector, attn_fn, scale_schedule, rope2d_freqs_grid, scale_ind=scale_ind, layer_idx=layer_idx)
            x = x + self.drop_path(x_sa.mul_(gamma1))
            x = x + self.ca(self.ca_norm(x), ca_kv).float().mul_(self.ca_gamma)
            x = x + self.drop_path(self.ffn(self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x, scale=scale2, shift=shift2)).mul(gamma2)) # this mul(gamma2) cannot be in-placed cuz we possibly use FusedMLP
        return x
    
    def extra_repr(self) -> str:
        return f'shared_aln={self.shared_aln}, fused_norm={self.fused_norm_func is not None}, ca_gamma={"<learnable>" if isinstance(self.ca_gamma, nn.Parameter) else self.ca_gamma}'


class AdaLNBeforeHead(nn.Module):
    def __init__(self, C, D, act: bool, norm_layer: partial, fused_norm_func=None):   # C: embed_dim, D: cond_dim
        super().__init__()
        self.C, self.D = C, D
        self.ln_wo_grad = norm_layer(C, elementwise_affine=False)
        self.fused_norm_func = fused_norm_func
        self.norm_eps = norm_layer.keywords.get('eps', 1e-6)
        lin = nn.Linear(D, 2*C)
        self.ada_lin = nn.Sequential(nn.SiLU(inplace=False), lin) if act else nn.Sequential(lin)
    
    def forward(self, x_BLC: torch.Tensor, cond_BD: Optional[torch.Tensor]):
        scale, shift = self.ada_lin(cond_BD).view(-1, 1, 2, self.C).unbind(2)
        if self.fused_norm_func is None:
            return self.ln_wo_grad(x_BLC).mul(scale.add(1)).add_(shift)
        else:
            return self.fused_norm_func(C=self.C, eps=self.norm_eps, x=x_BLC, scale=scale, shift=shift)


if __name__ == '__main__':
    embed_dim = 2048
    num_heads = 2048 // 128
    batch_size = 2
    seq_len = 2304
    pad_to_multiplier = 128
    rope2d_normalized_by_hw = 2

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    attn = SelfAttention(
        embed_dim=embed_dim,
        num_heads=num_heads,
        proj_drop=0.0,
        tau=1.0,
        cos_attn=True,
        customized_flash_attn=False,
        use_flex_attn=False,
        batch_size=batch_size,
        pad_to_multiplier=pad_to_multiplier,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw
    ).to(device)

    x = torch.randn(batch_size, seq_len, embed_dim, dtype=torch.float32, device=device)

    from utils.dynamic_resolution import dynamic_resolution_h_w
    scale_schedule = [(1, 1, 1), (1, 2, 2), (1, 4, 4), (1, 6, 6), (1, 8, 8), (1, 12, 12), 
                      (1, 16, 16), (1, 20, 20), (1, 24, 24), (1, 32, 32), (1, 40, 40), 
                      (1, 48, 48), (1, 64, 64)]
    rope2d_freqs_grid = precompute_rope2d_freqs_grid(
        dim=embed_dim // num_heads,
        dynamic_resolution_h_w=dynamic_resolution_h_w,
        pad_to_multiplier=pad_to_multiplier,
        rope2d_normalized_by_hw=rope2d_normalized_by_hw,
        device=x.device
    )

    with torch.no_grad():
        out = attn(
            x,
            attn_bias_or_two_vector=None,
            attn_fn=None,
            scale_schedule=scale_schedule,
            rope2d_freqs_grid=rope2d_freqs_grid,
            scale_ind=0
        )
    print(f"SelfAttention output shape: {out.shape}")
