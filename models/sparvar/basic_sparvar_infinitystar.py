from typing import Optional, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG
from kernels.sparse_attn.sparse_attn_ops import dense_attn
from models.infinitystar.rope import apply_rotary_emb
from models.infinitystar.basic_infinitystar import FastRMSNorm, Qwen3MLP, repeat_kv
from models.infinitystar.fused_elementwise_utils import fused_rope_inplace, fused_silu_mul
from models.sparvar.sparse_attn import SparseDiffAttn
from models.sparvar.sparse_attn_layer_counter import LayerCounter
from utils import lprof
from utils.sequence_parallel import sp_all_to_all, SequenceParallelManager as sp_manager

# Import flash_attn's attention
# from flash_attn_interface import flash_attn_func; print('#'*10 + ' use FA3 ' + '#'*10)
# from flash_attn.cute import flash_attn_func; print('#'*10 + ' use FA4 ' + '#'*10)
from flash_attn import flash_attn_func  # q, k, or v: BLHc, ret: BLHc
try:  # FA3 (flash_attn_interface): Hopper-only build, absent on A100 — keep optional
    from flash_attn_interface import flash_attn_func as flash_attn_func_v3
except ImportError:
    flash_attn_func_v3 = None
# ------ chipmunk FA3 ------
try: import chipmunk     # chipmunk FA3 attn
except ImportError: chipmunk = None

# Import flash_attn's fused ops
try:
    from flash_attn.ops.rms_norm import rms_norm as rms_norm_impl
    from flash_attn.ops.fused_dense import fused_mlp_func

    flash_fused_op_installed = True
except ImportError:
    fused_mlp_func = None
    flash_fused_op_installed = False


    def rms_norm_impl(x, weight, epsilon):
        return (x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True).add_(epsilon))) * weight


#* [Decision-TopK Exps] Pure decision-placement logic, kept separate for CPU-only validation.
def _resolve_cs4a_decision_flags(
    is_decision_scale: bool,
    repeat_idx: int,
    last_repetition_step: bool,
    decision_repeat_idx: Optional[int],
    decision_scale: int,
) -> Tuple[bool, bool]:
    if not is_decision_scale:
        return False, False

    # decision_repeat_idx: null -> decision on the decision scale's last repeat;
    # an integer k -> decision on repeat k (earlier repeats dense, later repeats
    # reuse the decision indices). Out-of-range k raises once the scale ends.
    if decision_repeat_idx is None:
        return bool(last_repetition_step), False

    if last_repetition_step and repeat_idx < decision_repeat_idx:
        raise ValueError(
            f'attn.decision_repeat_idx={decision_repeat_idx} is out of range '
            f'for decision scale {decision_scale}; the last available repeat is {repeat_idx}'
        )
    return repeat_idx == decision_repeat_idx, repeat_idx > decision_repeat_idx


class SelfAttention(nn.Module):
    def __init__(
            self, embed_dim=768, num_heads=12, num_key_value_heads=-1,
            use_flex_attn=False,
            pad_to_multiplier=1, rope2d_normalized_by_hw=0,
            mask_type='var', context_frames=1000000, steps_per_frame=4,
            arch='var',
            qwen_qkvo_bias=False,
    ):
        """
        :param embed_dim: model's width
        :param num_heads: num heads of multi-head attention
        """
        super().__init__()
        assert embed_dim % num_heads == 0
        assert num_key_value_heads == -1 or num_heads % num_key_value_heads == 0

        self.embed_dim = embed_dim
        self.num_heads, self.head_dim = num_heads, embed_dim // num_heads
        self.num_key_value_heads = num_key_value_heads if num_key_value_heads > 0 else num_heads
        self.arch = arch
        if self.arch == 'qwen':
            self.q_proj = nn.Linear(embed_dim, self.num_heads * self.head_dim, bias=qwen_qkvo_bias)
            self.k_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=qwen_qkvo_bias)
            self.v_proj = nn.Linear(embed_dim, self.num_key_value_heads * self.head_dim, bias=qwen_qkvo_bias)
            self.o_proj = nn.Linear(self.num_heads * self.head_dim, embed_dim, bias=qwen_qkvo_bias)
            self.q_norm = FastRMSNorm(self.head_dim)
            self.k_norm = FastRMSNorm(self.head_dim)
            self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        else:
            raise ValueError(f'arch {self.arch} not supported')

        self.caching = False  # kv caching: only used during inference
        self.cached_k = {}  # kv caching: only used during inference
        self.cached_v = {}  # kv caching: only used during inference

        self.use_flex_attn = use_flex_attn
        self.pad_to_multiplier = pad_to_multiplier

        self.rope2d_normalized_by_hw = rope2d_normalized_by_hw
        self.mask_type = mask_type
        self.context_frames = context_frames
        self.steps_per_frame = steps_per_frame
        self.layer_idx = -1
        self.attn_map_recorder = None

        #* ------ CS4A ------
        attn_config = GLOBAL_CONFIG['attn']
        # self.decision_scale = 27
        self.decision_scale = attn_config['decision_scale']
        # Hopper-only: pre-sparse dense scales run FA3 (flash_attn_interface)
        # instead of chipmunk dense_attn (~1.25x on H800); fail fast at model
        # construction if the FA3 build is missing.
        self.dense_attn_fa3 = bool(attn_config.get('dense_attn_fa3', False))
        if self.dense_attn_fa3:
            assert flash_attn_func_v3 is not None, \
                'attn.dense_attn_fa3=true requires the FA3 build (flash_attn_interface)'
        # Decision placement: decision_repeat_idx selects which repeat of the
        # decision scale runs the full colsum decision. null -> the scale's last
        # repeat; an integer k -> repeat k, with its later repeats running
        # csp_attn that reuses the repeat-k indices (validated: a repeat's top-k
        # keeps ~0.995 relative colsum-mass recall on the next repeat).
        configured_repeat_idx = attn_config.get('decision_repeat_idx', None)
        if configured_repeat_idx is None:
            self.decision_repeat_idx = None
        else:
            if isinstance(configured_repeat_idx, bool) or not isinstance(configured_repeat_idx, int):
                raise TypeError(
                    'attn.decision_repeat_idx must be a non-negative integer or null, '
                    f'got {configured_repeat_idx!r}'
                )
            if configured_repeat_idx < 0:
                raise ValueError(
                    'attn.decision_repeat_idx must be non-negative, '
                    f'got {configured_repeat_idx}'
                )
            # D-CSLA supports an explicit decision repeat: the decision scale's later
            # repeats replay the block-sparse forward using the mass entry computed for
            # the decision scale itself (same-scale transfer = exact block membership).
            self.decision_repeat_idx = configured_repeat_idx
        # which repetition step is running; broadcast by the infer loop each (scale, repeat)
        self.repetition_step_idx = 0

        layer_num, layer_counter = LayerCounter.build_for_layer(is_attn_sparse=True)
        self.sparse_attn = SparseDiffAttn(
            layer_num, layer_counter, decision_scale=self.decision_scale,
        )

    def kv_caching(self, enable: bool):  # kv caching: only used during inference
        self.caching = enable
        self.cached_k = {}
        self.cached_v = {}

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, attn_bias_or_two_vector: Union[torch.Tensor, Tuple[torch.IntTensor, torch.IntTensor]],
                attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[], scale_ind=0, context_info=None,
                last_repetition_step=True, ref_text_scale_inds=[]):
        """
        :param (fp32) x: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded (L = raw_seq_len//sp_size)
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
        :return: shaped (B or batch_size, L or seq_length, C or hidden_dim); if seq-parallel is used, the `L` dim would be sharded
        """
        # x: fp32
        B, L, C = x.shape

        if self.arch == 'qwen':  # InfinityStar only support 'qwen' arch
            hidden_states = x
            input_shape = hidden_states.shape[:-1]
            hidden_shape = (*input_shape, -1, self.head_dim)

            with lprof.region('attn.qkv', si=scale_ind, layer=self.layer_idx):
                query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
                key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)  # batch, num_key_value_heads, slen, head_dim
                value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)  # batch, num_key_value_heads, slen, head_dim

            if sp_manager.sp_on():
                # Headnum need to be sharded and L needs to be gathered
                # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
                sdim = 1
                gdim = 2
                L = L * sp_manager.get_sp_size()
                C = C // sp_manager.get_sp_size()
                query_states = sp_all_to_all(query_states, sdim, gdim)
                key_states = sp_all_to_all(key_states, sdim, gdim)
                value_states = sp_all_to_all(value_states, sdim, gdim)

            with lprof.region('attn.rope', si=scale_ind, layer=self.layer_idx):
                # bit-exact fused Triton RoPE when INFINITYSTAR_FUSED_ROPE=1, else the eager apply_rotary_emb
                query_states, key_states = fused_rope_inplace(query_states, key_states, rope2d_freqs_grid)
            with lprof.region('attn.kvcache', si=scale_ind, layer=self.layer_idx):
                if self.caching:  # kv caching: only used during inference
                    if last_repetition_step:
                        self.cached_k[scale_ind] = key_states
                        self.cached_v[scale_ind] = value_states
                    if isinstance(scale_ind, int):
                        ref_scale_inds = context_info[scale_ind]['ref_sids'] + ref_text_scale_inds
                        key_states = torch.cat([self.cached_k[ind] for ind in ref_scale_inds] + [key_states], dim=2)
                        value_states = torch.cat([self.cached_v[ind] for ind in ref_scale_inds] + [value_states], dim=2)

                        ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
                        for si in range(len(context_info)):
                            for ref_si in context_info[si]['ref_sids']:
                                ref_scale_2_last_use_scale[ref_si] = si
                        for ref_si in range(scale_ind):
                            if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si] is not None):
                                tmpk, tmpv = self.cached_k[ref_si], self.cached_v[ref_si]
                                self.cached_k[ref_si], self.cached_v[ref_si] = None, None
                                del tmpk, tmpv

            # [GQA] all attention consumers on this path (chipmunk CUDA / Triton csp &
            # colsum kernels, flash_attn) natively support grouped KV (8 KV heads x
            # 32 Q heads, mapped via hr / H_Q//H_KV inside the kernels), so the 4x
            # repeat_kv materialization is skipped entirely.
            # key_states = repeat_kv(key_states, self.num_key_value_groups)
            # value_states = repeat_kv(value_states, self.num_key_value_groups)
            if self.attn_map_recorder is not None and getattr(self.attn_map_recorder, "verbose", False):
                print(f'[scale-{scale_ind}_layer-{self.layer_idx}] self-attn caching: q shape: {query_states.shape}, k shape: {key_states.shape}, v shape: {value_states.shape}')
            # print(f'[scale-{scale_ind}_layer-{self.layer_idx}] self-attn caching: q shape: {query_states.shape}, k shape: {key_states.shape}, v shape: {value_states.shape}')

            scale = self.head_dim ** -0.5
            if (
                self.attn_map_recorder is not None
                and self.attn_map_recorder.should_capture_attention(
                    layer_idx=self.layer_idx,
                    batch_size=B,
                    num_heads=self.num_heads,
                )
            ):
                if sp_manager.sp_on():
                    raise NotImplementedError("Attention map capture does not support sequence parallel inference.")
                attn_output = self.attn_map_recorder.capture_attention(
                    layer_idx=self.layer_idx,
                    query_states=query_states,
                    key_states=key_states,
                    value_states=value_states,
                    scale=scale,
                ).transpose(1, 2).reshape(B, L, C)
            elif self.use_flex_attn and attn_fn is not None:  # ! inference always False
                attn_output = attn_fn(query_states.to(value_states.dtype), key_states.to(value_states.dtype),
                                      value_states, scale=scale).transpose(1, 2).reshape(B, L, C)
            else:
                # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                # from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
                # attn_output = flash_attn_func(query_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                #                               key_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                #                               value_states.permute([0, 2, 1, 3]).to(torch.bfloat16),
                #                               softmax_scale=scale)
                # attn_output = attn_output.reshape(B, L, C)

                # fa3, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
                # from flash_attn_interface import flash_attn_qkvpacked_func, flash_attn_func
                # attn_output = flash_attn_func(query_states.permute([0,2,1,3]).to(torch.bfloat16), key_states.permute([0,2,1,3]).to(torch.bfloat16), value_states.permute([0,2,1,3]).to(torch.bfloat16), softmax_scale=scale)
                # attn_output = attn_output.reshape(B, L, C)
                # print(f'{value_states.dtype=}')     # torch.bfloat16

                # --- chipmunk ---
                # attn_output, _ = torch.ops.chipmunk.dense_attn(query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale)
                # # print(f'{attn_output.shape=}')
                # attn_output = attn_output.transpose(1, 2).reshape(B, L, C)

                # slow attn
                # attn_output = slow_attn(query=query_states, key=key_states, value=value_states, scale=scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)

                # ------ SparVAR ------
                # Decision placement: decision_repeat_idx picks the decision repeat
                # (null -> last repeat of the decision scale).
                is_decision_scale = isinstance(scale_ind, int) and scale_ind == self.decision_scale
                decision_step, decision_reuse_step = _resolve_cs4a_decision_flags(
                    is_decision_scale=is_decision_scale,
                    repeat_idx=self.repetition_step_idx,
                    last_repetition_step=last_repetition_step,
                    decision_repeat_idx=self.decision_repeat_idx,
                    decision_scale=self.decision_scale,
                )

                if decision_step:
                    # print(f'     [scale-{scale_ind}_layer-{self.layer_idx}] CS4A decision scale')
                    with lprof.region('attn.core.decision', si=scale_ind, layer=self.layer_idx,
                                      repeat=self.repetition_step_idx):
                        attn_output = self.sparse_attn(
                            query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states,
                            scale=scale, do_full_step=True, scale_ind=scale_ind, layer_ind=self.layer_idx,
                            scale_schedule=scale_schedule,
                            # forward repeat metadata for repeat-level profiling
                            repeat_idx=self.repetition_step_idx,
                            last_repetition_step=last_repetition_step
                        ).transpose(1, 2).reshape(B, L, C)

                elif decision_reuse_step or (isinstance(scale_ind, int) and scale_ind > self.decision_scale):
                    # print(f'     [scale-{scale_ind}_layer-{self.layer_idx}] CS4A sparse scale')
                    with lprof.region('attn.core.sparse', si=scale_ind, layer=self.layer_idx,
                                      repeat=self.repetition_step_idx):
                        attn_output = self.sparse_attn(
                            query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states,
                            scale=scale, do_full_step=False, scale_ind=scale_ind, layer_ind=self.layer_idx,
                            scale_schedule=scale_schedule,
                            # forward repeat metadata for repeat-level profiling
                            repeat_idx=self.repetition_step_idx,
                            last_repetition_step=last_repetition_step
                        ).transpose(1, 2).reshape(B, L, C)

                else:
                    with lprof.region('attn.core.dense', si=scale_ind, layer=self.layer_idx,
                                      repeat=self.repetition_step_idx):
                        if self.dense_attn_fa3:
                            # FA3 (bshd layout, GQA-native, returns (out, lse));
                            # out is [B, L, H, D] so no transpose before reshape
                            ret = flash_attn_func_v3(
                                query_states.permute(0, 2, 1, 3).to(value_states.dtype),
                                key_states.permute(0, 2, 1, 3).to(value_states.dtype),
                                value_states.permute(0, 2, 1, 3),
                                softmax_scale=scale)
                            attn_output = (ret[0] if isinstance(ret, tuple) else ret).reshape(B, L, C)
                        elif GLOBAL_CONFIG['attn']['provider'] == 'tilelang':
                            attn_output, _ = dense_attn(
                                query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale)
                            attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
                        else:
                            attn_output, _ = torch.ops.chipmunk.dense_attn(
                                query_states.to(value_states.dtype), key_states.to(value_states.dtype), value_states, scale=scale)
                            attn_output = attn_output.transpose(1, 2).reshape(B, L, C)
                # ----------------------

            if sp_manager.sp_on():
                # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
                sdim = 1
                gdim = 2
                attn_output = sp_all_to_all(attn_output, sdim, gdim)

            attn_output = self.o_proj(attn_output)

            return attn_output

        # qkv: amp, bf16
        qkv = (F.linear(input=x, weight=self.mat_qkv.weight, bias=torch.cat((self.q_bias, self.zero_k_bias, self.v_bias)))
               .view(B, L, 3, self.num_heads, self.head_dim))  # BL3Hc
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(dim=0);
        L_dim = 2  # q or k or v: all are shaped in (B:batch_size, H:heads, L:seq_len, c:head_dim), this way

        scale_mul = self.scale_mul_1H11.clamp_max(self.max_scale_mul).exp()  # 11H1 (flash), or 1H11 (not flash)
        q = F.normalize(q, dim=-1, eps=1e-12).mul(scale_mul).contiguous()  # fp32
        k = F.normalize(k, dim=-1, eps=1e-12).contiguous()  # fp32
        v = v.contiguous()  # bf16

        if sp_manager.sp_on():
            # Headnum need to be sharded and L needs to be gathered
            # [B, H, raw_L/sp, C] --> [B, H/sp, raw_L, C]
            sdim = 1
            gdim = 2

            L = L * sp_manager.get_sp_size()
            C = C // sp_manager.get_sp_size()

            q = sp_all_to_all(q, sdim, gdim)
            k = sp_all_to_all(k, sdim, gdim)
            v = sp_all_to_all(v, sdim, gdim)

        q, k = apply_rotary_emb(q, k, rope2d_freqs_grid)  # , freqs_cis=freqs_cis)
        if self.caching:  # kv caching: only used during inference
            if last_repetition_step:
                self.cached_k.append(k)
                self.cached_v.append(v)
            if scale_ind >= 0:
                ref_scale_inds = context_info[scale_ind]['ref_sids']
                k = torch.cat([self.cached_k[0]] + [self.cached_k[ind + 1] for ind in ref_scale_inds] + [k], dim=L_dim)
                v = torch.cat([self.cached_v[0]] + [self.cached_v[ind + 1] for ind in ref_scale_inds] + [v], dim=L_dim)

            ref_scale_2_last_use_scale = [-1 for _ in range(len(context_info))]
            for si in range(len(context_info)):
                for ref_si in context_info[si]['ref_sids']:
                    ref_scale_2_last_use_scale[ref_si] = si
            for ref_si in range(scale_ind):
                if (ref_scale_2_last_use_scale[ref_si] < scale_ind) and (self.cached_k[ref_si + 1] is not None):
                    tmpk, tmpv = self.cached_k[ref_si + 1], self.cached_v[ref_si + 1]
                    self.cached_k[ref_si + 1], self.cached_v[ref_si + 1] = None, None
                    del tmpk, tmpv

        # if self.cos_attn: q, k are in fp32; v is in bf16
        # else: q, k, v are in bf16
        if self.use_flex_attn and attn_fn is not None:
            oup = attn_fn(q.to(v.dtype), k.to(v.dtype), v, scale=self.scale).transpose(1, 2).reshape(B, L, C)
        else:
            # oup = slow_attn(query=q, key=k, value=v, scale=self.scale, attn_mask=attn_bias_or_two_vector, dropout_p=0).transpose(1, 2).reshape(B, L, C)
            # fa2, flash_attn_func input/output should be (batch_size, seqlen, nheads, headdim)
            # from flash_attn import flash_attn_qkvpacked_func, flash_attn_func
            oup = flash_attn_func(q.permute([0, 2, 1, 3]).to(torch.bfloat16),
                                  k.permute([0, 2, 1, 3]).to(torch.bfloat16),
                                  v.permute([0, 2, 1, 3]).to(torch.bfloat16), softmax_scale=self.scale)
            oup = oup.reshape(B, L, C)
        # oup: bf16

        if sp_manager.sp_on():
            # [B, raw_L, C/sp] --> [B, raw_L/sp, C]
            sdim = 1
            gdim = 2
            oup = sp_all_to_all(oup, sdim, gdim)

        return self.proj_drop(self.proj(oup))


class Qwen3MLPFused(Qwen3MLP):
    """Qwen3MLP with the SiLU*up activation fused into one bit-exact Triton kernel when
    INFINITYSTAR_FUSED_SWIGLU=1 (same parameters / state_dict keys as Qwen3MLP)."""

    def forward(self, x):
        return self.down_proj(fused_silu_mul(self.gate_proj(x), self.up_proj(x)))


class SelfAttnBlock(nn.Module):
    def __init__(
            self,
            embed_dim,
            cond_dim,
            num_heads,
            num_key_value_heads,
            mlp_ratio=4.0,
            use_flex_attn=False,
            pad_to_multiplier=1,
            rope2d_normalized_by_hw=False,
            mask_type="",
            context_frames=-1,
            steps_per_frame=-1,
            arch="var",
            qwen_qkvo_bias=False,
            inject_sync=False,
    ):
        super(SelfAttnBlock, self).__init__()
        self.C, self.D = embed_dim, cond_dim
        self.arch = arch
        self.attn = SelfAttention(
            embed_dim=embed_dim, num_heads=num_heads, num_key_value_heads=num_key_value_heads,
            use_flex_attn=use_flex_attn, pad_to_multiplier=pad_to_multiplier,
            rope2d_normalized_by_hw=rope2d_normalized_by_hw,
            mask_type=mask_type, context_frames=context_frames, steps_per_frame=steps_per_frame, arch=arch,
            qwen_qkvo_bias=qwen_qkvo_bias,
        )
        if self.arch == 'qwen':
            self.mlp = Qwen3MLPFused(hidden_size=embed_dim, intermediate_size=round(embed_dim * mlp_ratio / 256) * 256)
            self.input_layernorm = FastRMSNorm(embed_dim)
            self.post_attention_layernorm = FastRMSNorm(embed_dim)
            self.inject_sync = inject_sync
        else:
            raise ValueError(f'arch {self.arch} not supported')

    # NOTE: attn_bias_or_two_vector is None during inference
    def forward(self, x, cond_BD, ca_kv, attn_bias_or_two_vector, attn_fn=None, rope2d_freqs_grid=[], scale_schedule=[],
                scale_ind=0, context_info=None, last_repetition_step=True, ref_text_scale_inds=[]):
        residual = x
        hidden_states = x
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.attn(hidden_states, attn_bias_or_two_vector, attn_fn, rope2d_freqs_grid, scale_schedule,
                                  scale_ind, context_info, last_repetition_step, ref_text_scale_inds)
        hidden_states = residual + hidden_states
        # Fully Connected
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        return hidden_states


if __name__ == '__main__':
    pass
