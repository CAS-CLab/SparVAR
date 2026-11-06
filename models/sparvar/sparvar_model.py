from pathlib import Path
from contextlib import nullcontext
from functools import partial
import math
import os
import random
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import register_model

from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG, load_from_file
from models.infinity.flex_attn import FlexAttn
from models.infinity.basic_infinity import (
    AdaLNBeforeHead, 
    SelfAttnBlock, 
    flash_attn_func, 
    flash_fused_op_installed, 
    FastRMSNorm, 
    precompute_rope2d_freqs_grid
)
from models.infinity.infinity_model import (
    Infinity,
    MultiInpIdentity,
    MultipleLayers,
    SharedAdaLin,
    TextAttentivePool,
    sample_with_top_k_top_p_also_inplace_modifying_logits_,
    TIMM_KEYS
)
from models.sparvar.basic_sparvar import CrossAttnBlock
from models.sparvar.sparse_attn_csla import build_csla_spattn_list
from models.sparvar.sparvar_cuda_graph_utils import BlockStackGraphRunner, cuda_graph_enabled

import utils.dist as dist
from utils import lprof
from utils.dist import for_visualize
from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates

# level-embedding index tensor cache (bit-identical; see SparVAR_Infinity.add_lvl_embeding); opt-in
_LVL_IDX_CACHE = os.environ.get('SPARVAR_LVL_IDX_CACHE', '0').lower() in ('1', 'true', 'yes', 'on')

try:
    from models.infinity.fused_op import fused_ada_layer_norm, fused_ada_rms_norm
except:
    fused_ada_layer_norm, fused_ada_rms_norm = None, None


@register_model
def sparvar_infinity_2b(
    depth=32, embed_dim=2048, num_heads=2048//128, drop_path_rate=0.1, 
    mlp_ratio=4, block_chunks=8, **kwargs
):
    return SparVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_chunks=block_chunks,
        drop_path_rate=drop_path_rate,

        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


@register_model
def sparvar_infinity_8b(
    depth=40, embed_dim=3584, num_heads=3584//128, drop_path_rate=0.1, 
    mlp_ratio=4, block_chunks=8, **kwargs
):
    return SparVAR_Infinity(
        depth=depth,
        embed_dim=embed_dim,
        num_heads=num_heads,
        mlp_ratio=mlp_ratio,
        block_chunks=block_chunks,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


class SparVAR_Infinity(Infinity):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SPARVAR_CONFIG_FILE lets concurrent ablation jobs use isolated config
        # copies instead of racing on the canonical yml (mirrors the video model).
        load_from_file(os.environ.get("SPARVAR_CONFIG_FILE",
                                      str(Path(__file__).with_name("sparse-attn-config.yml"))))
        attn_config = GLOBAL_CONFIG['attn']

        self.unregistered_blocks = []
        for block_idx in range(self.depth):
            block = (CrossAttnBlock if self.t2i else SelfAttnBlock)(
                embed_dim=self.C,
                kv_dim=self.D,
                cross_attn_layer_scale=self.cross_attn_layer_scale,
                cond_dim=self.D,
                act=True,
                shared_aln=self.shared_aln,
                norm_layer=self.norm_layer,
                num_heads=self.num_heads,
                mlp_ratio=self.mlp_ratio,
                drop=self.drop_rate,
                drop_path=self.dpr[block_idx],
                tau=self.tau,
                cos_attn=self.cos_attn,
                swiglu=self.swiglu,
                customized_flash_attn=self.customized_flash_attn,
                fused_mlp=self.fused_mlp,
                fused_norm_func=self.fused_norm_func,
                checkpointing_sa_only=self.checkpointing == 'self-attn',
                use_flex_attn=self.use_flex_attn,
                batch_size=self.batch_size,
                pad_to_multiplier=self.pad_to_multiplier,
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,
                attn_config=attn_config,
            )
            self.unregistered_blocks.append(block)

        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(
                    self.unregistered_blocks,
                    self.num_blocks_in_a_chunk,
                    i*self.num_blocks_in_a_chunk
                ))

        # print(
        #     f'\n    [skip scales] skip last {skip_last_scales} scales\n',
        #     f'    [drop uncond branch] drop last {drop_uncond_last_scales} scales\n',
        #     end='\n\n', flush=True
        # )
        # self.skip_last_scales = skip_last_scales
        # self.drop_uncond_last_scales = drop_uncond_last_scales
        
        #* ------ CLSA: attn sink + block-wise local sparse attn ------
        assert len(attn_config['win_size']) == 13, f'kernel_schedule mismatch 13 scales'
        self.attn_sink_scale: int = attn_config['attn_sink']

        # Model geometry for the 1M / h_div_w=1.0 pn: the static masks only
        # support this square 13-scale schedule (the forward asserts kv length).
        scale_schedule = [
            (1, 1), (2, 2), (4, 4), (6, 6), (8, 8), (12, 12), (16, 16),
            (20, 20), (24, 24), (32, 32), (40, 40), (48, 48), (64, 64)
        ]
        self.spattn_compile_list = build_csla_spattn_list(
            scale_schedule, attn_config, num_scales=len(scale_schedule))

        # drop_uncond_last_scales='auto': anchor the uncond drop to the CS4A decision scale
        # (decision scale and every later scale run single-branch), mirroring the video model.
        # An int keeps the baseline "last N scales of the full schedule" semantics.
        self._drop_uncond_auto = isinstance(self.drop_uncond_last_scales, str) and \
            self.drop_uncond_last_scales.strip().lower() == 'auto'
        if self._drop_uncond_auto:
            print(f"    [drop uncond branch] mode=auto -> anchored at CS4A decision_scale={attn_config['decision_scale']}")
        else:
            self.drop_uncond_last_scales = int(self.drop_uncond_last_scales)

        # CUDA-graph replay of the dense block stack at the small scales (SPARVAR_CUDA_GRAPH=1)
        self._cuda_graph_runner = BlockStackGraphRunner(self, attn_config) if cuda_graph_enabled() else None
        self._lvl_idx_cache = {}    # (scale_ind, bs, n_tokens, device) -> constant level-index tensor

    def add_lvl_embeding(self, feature, scale_ind, scale_schedule, need_to_pad=0):
        # Same math as Infinity.add_lvl_embeding. The level-index tensor is a constant per
        # (scale, batch, length); building it on the device once removes the per-scale
        # CPU->GPU copy of the base version, which is a host sync on every scale.
        # SPARVAR_LVL_IDX_CACHE=1 enables it; the default keeps the base implementation.
        if not _LVL_IDX_CACHE:
            return super().add_lvl_embeding(feature, scale_ind, scale_schedule, need_to_pad=need_to_pad)
        bs, seq_len, c = feature.shape
        patch_t, patch_h, patch_w = scale_schedule[scale_ind]
        t_mul_h_mul_w = patch_t * patch_h * patch_w
        assert t_mul_h_mul_w + need_to_pad == seq_len
        key = (scale_ind, bs, t_mul_h_mul_w, feature.device)
        idx = self._lvl_idx_cache.get(key)
        if idx is None:
            idx = torch.full((bs, t_mul_h_mul_w), scale_ind, dtype=torch.int, device=feature.device)
            self._lvl_idx_cache[key] = idx
        feature[:, :t_mul_h_mul_w] += self.lvl_embed(idx)
        return feature

    def _resolve_drop_uncond_from(self, scale_schedule):
        """First scale index whose uncond branch is dropped, or None when nothing is dropped."""
        if self._drop_uncond_auto:
            return int(GLOBAL_CONFIG['attn']['decision_scale'])
        if self.drop_uncond_last_scales > 0:
            return len(scale_schedule) - self.drop_uncond_last_scales
        return None

    @torch.no_grad()
    def autoregressive_infer_cfg(
        self,
        vae=None,
        scale_schedule=None,    # [(1, 1, 1), (1, 2, 2), (1, 4, 4), ..., (1, 48, 48), (1, 64, 64)]
        label_B_or_BLT=None,    # tuple(torch.Size([9, 2048]), list[9], tensor[0, 9] shape=[2])
        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
        g_seed=None, cfg_list=[], tau_list=[], cfg_sc=3, top_k=0, top_p=0.0,    # top_k=900, top_p=0.97
        returns_vemb=0, ratio_Bl1=None, gumbel=0, norm_cfg=False,               # returns_vemb=1
        cfg_exp_k: float=0.0, cfg_insertion_layer=[-5],     # cfg_insertion_layer=[0], CFG to Transformer layer: 0 for logits; 1 for probability, 负值表示从末尾开始的第n层
        vae_type=0, softmax_merge_topk=-1, ret_img=False,   # vae_type=32, ret_img=True
        trunk_scale=1000,
        gt_leak=0, gt_ls_Bl=None,
        inference_mode=False,   # True for infra, to start KV-Cache
        save_img_path=None,
        sampling_per_bits=1,
    ):   # returns List[idx_Bl]
        if g_seed is None: rng = None
        else: self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)     # for each scale has cfg and tau
        assert len(tau_list) >= len(scale_schedule)

        # --- for drop uncond branch ---
        drop_uncond_from = self._resolve_drop_uncond_from(scale_schedule)
        if drop_uncond_from is not None:
            with torch.amp.autocast('cuda', enabled=False):
                kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
                # 如果 forward 中训练时可能随机丢弃条件，这里推理时直接使用条件版（可保留丢弃逻辑也可去掉）
                kv_normal = self.text_norm(kv_compact).contiguous()
                sos_normal = self.text_proj_for_sos((kv_normal, cu_seqlens_k, max_seqlen_k)).float().contiguous()
                kv_normal = self.text_proj_for_ca(kv_normal).contiguous()
                cond_BD_normal = self.shared_ada_lin(sos_normal).contiguous()  # cond_BD 依赖 sos_normal
                kv_step12 = kv_normal.clone()
                ca_kv_step12 = (kv_step12, cu_seqlens_k, max_seqlen_k)
                cond_BD_step12 = cond_BD_normal.clone()

        # scale_schedule is used by infinity, vae_scale_schedule is used by vae if there exists a spatial patchify, 
        # we need to convert scale_schedule to vae_scale_schedule by multiply 2 to h and w
        if self.apply_spatial_patchify:     # default: 0
            vae_scale_schedule = [(pt, 2*ph, 2*pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        # ------ text prompt process: for CFG ------
        kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        if any(np.array(cfg_list) != 1):
            bs = 2*B
            if not negative_label_B_or_BLT:
                kv_compact_un = kv_compact.clone()
                total = 0
                for le in lens:
                    kv_compact_un[total:total+le] = (self.cfg_uncond)[:le]
                    total += le
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)      # [2*t_len, text_dim(2048)]
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k[1:]+cu_seqlens_k[-1]), dim=0)      # tensor[0, 9, 18], adjust the cumulative seq_len of 2*bs
            else:
                kv_compact_un, lens_un, cu_seqlens_k_un, max_seqlen_k_un = negative_label_B_or_BLT
                kv_compact = torch.cat((kv_compact, kv_compact_un), dim=0)
                cu_seqlens_k = torch.cat((cu_seqlens_k, cu_seqlens_k_un[1:]+cu_seqlens_k[-1]), dim=0)
                max_seqlen_k = max(max_seqlen_k, max_seqlen_k_un)
        else:
            bs = B
        # ------------------------------------------

        kv_compact = self.text_norm(kv_compact)     # FastRMSNorm
        sos = cond_BD = self.text_proj_for_sos((kv_compact, cu_seqlens_k, max_seqlen_k))    # sos: [2, v_dim(2048)], cond_BD: [2, v_dim]
        kv_compact = self.text_proj_for_ca(kv_compact)                                      # kv_compact: [2*len, v_dim], linear -> tanh -> linear
        ca_kv = kv_compact, cu_seqlens_k, max_seqlen_k
        last_stage = sos.unsqueeze(1).expand(bs, 1, -1) + self.pos_start.expand(bs, 1, -1)      # [bs, 1, v_dim], Transformer first scale input

        with torch.amp.autocast('cuda', enabled=False):
            cond_BD_or_gss = self.shared_ada_lin(cond_BD.float()).float().contiguous()      # shared_ada_lin: silu -> SharedAdaLin; gss: gamma, scale, shift for AdaLN
        accu_BChw, cur_L, ret = None, 0, []     # current length, list of reconstructed images
        idx_Bl_list, idx_Bld_list = [], []      # Bl: 存储每个尺度预测的 token 索引的列表; Bld: 存储每个尺度预测的逐位 token 标签的列表

        if inference_mode:      # start Transformer block's attn KV-Cache
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(True)
            # for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).update_batch(B)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(True)

        # ------ where and how to use CFG ------
        abs_cfg_insertion_layers = []
        add_cfg_on_logits, add_cfg_on_probs = False, False
        leng = len(self.unregistered_blocks)
        for item in cfg_insertion_layer:
            if item == 0: # add cfg on logits
                add_cfg_on_logits = True
            elif item == 1: # add cfg on probs
                add_cfg_on_probs = True # todo in the future, we may want to add cfg on logits and probs
            elif item < 0: # determine to add cfg at item-th layer's output
                assert leng+item > 0, f'cfg_insertion_layer: {item} is not valid since len(unregistered_blocks)={self.num_block_chunks}'
                abs_cfg_insertion_layers.append(leng+item)
            else:
                raise ValueError(f'cfg_insertion_layer: {item} is not valid')
        # ---------------------------------------

        num_stages_minus_1 = len(scale_schedule) - 1
        summed_codes = 0
        # ------ auto-regressive scale ------             si: [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]
        for si, pn in enumerate(scale_schedule):        # si: i-th segment, pn: current scale patch number
            # skip last scales
            if self.skip_last_scales == 2 and (48 == pn[2] or 64 == pn[2]):
                continue
            elif self.skip_last_scales == 1 and 64 == pn[2]:
                continue
            # print(f"\n[debug] --- current scale: {si} ({pn[1]}x{pn[2]}) ---")

            cfg = cfg_list[si]                          # get current scale si's CFG
            if si >= trunk_scale:                       # trunk_scale=1000
                break
            cur_L += np.array(pn).prod()                # current cumulative token length
            # per-scale timer: blocks (transformer stack) / logits (head + sampling)
            # / post (quantizer + interpolation + next-scale embedding)
            _t_scale = lprof.timer('scale', si=si)

            need_to_pad = 0
            attn_fn = None
            if self.use_flex_attn:      # test default = 0
                # need_to_pad = (self.pad_to_multiplier - cur_L % self.pad_to_multiplier) % self.pad_to_multiplier
                # if need_to_pad:
                #     last_stage = F.pad(last_stage, (0, 0, 0, need_to_pad))
                attn_fn = self.attn_fn_compile_dict.get(tuple(scale_schedule[:(si+1)]), None)

            # ------ Transformer blocks ------
            # assert self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].sum() == 0, f'AR with {(self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L] != 0).sum()} / {self.attn_bias_for_masking[:, :, last_L:cur_L, :cur_L].numel()} mask item'
            layer_idx = 0
            cg = self._cuda_graph_runner
            if cg is not None and cg.eligible(si, last_stage, ca_kv, abs_cfg_insertion_layers, drop_uncond_from,
                                              attn_fn, need_to_pad, self.add_lvl_embeding_only_first_block):
                # whole dense block stack of this scale runs as one CUDA graph (sparvar_cuda_graph_utils)
                last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                last_stage = cg.run(si, last_stage, cond_BD_or_gss, ca_kv, scale_schedule,
                                    self.rope2d_freqs_grid, self.spattn_compile_list[si])
                layer_idx = self.depth
            elif cg is not None:
                cg.ensure_eager_ready()
            # the graph path already ran every block: iterate nothing below in that case
            for block_idx, b in enumerate(self.block_chunks if layer_idx == 0 else ()):
                # last_stage shape: [4, 1, 2048], cond_BD_or_gss.shape: [4, 1, 6, 2048], ca_kv[0].shape: [64, 2048], ca_kv[1].shape [5], ca_kv[2]: int
                if self.add_lvl_embeding_only_first_block and block_idx == 0:
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                if not self.add_lvl_embeding_only_first_block: 
                    last_stage = self.add_lvl_embeding(last_stage, si, scale_schedule, need_to_pad=need_to_pad)
                
                # generic form of the former hardcoded (N=1 -> si>=12 / N=2 -> si>=11 / N=3 -> si>=10)
                if layer_idx == 0 and drop_uncond_from is not None and si >= drop_uncond_from:
                    last_stage = last_stage[:B]
                    cond_BD_or_gss = cond_BD_step12 #print(f'{cond_BD_step12.shape=}')
                    ca_kv = ca_kv_step12
                
                for m in b.module:
                    last_stage = m(
                        x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=None, 
                        attn_fn=attn_fn, scale_schedule=scale_schedule, rope2d_freqs_grid=self.rope2d_freqs_grid, 
                        scale_ind=si, layer_ind=layer_idx, 
                        spattn_fn=self.spattn_compile_list[si]
                    )
                    if (cfg != 1) and (layer_idx in abs_cfg_insertion_layers):
                        # print(f'add cfg={cfg} on {layer_idx}-th layer output')
                        last_stage = cfg * last_stage[:B] + (1-cfg) * last_stage[B:]
                        last_stage = torch.cat((last_stage, last_stage), 0)
                    layer_idx += 1
            # --------------------------------
            _t_scale.mark('blocks')

            if (cfg != 1) and add_cfg_on_logits:
                # print(f'add cfg on add_cfg_on_logits')
                if last_stage.shape[0] * 2 == cond_BD.shape[0]:
                    logits_BlV = self.get_logits(last_stage, cond_BD[:B]).mul(1/tau_list[si])
                    logits_BlV = cfg * logits_BlV + (1-cfg) * logits_BlV
                else:
                    logits_BlV = self.get_logits(last_stage, cond_BD).mul(1/tau_list[si])       # get 2*bs(cond and uncond)'s logits
                    logits_BlV = cfg * logits_BlV[:B] + (1-cfg) * logits_BlV[B:]
            else:
                logits_BlV = self.get_logits(last_stage[:B], cond_BD[:B]).mul(1/tau_list[si])
            # ------ sample from logits_BlV, get current scale predicted tokens ------
            if self.use_bit_label:      # use_bit_label = 1
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, 2)
                idx_Bld = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
                idx_Bld = idx_Bld.reshape(tmp_bs, tmp_seq_len, -1)
            else:
                idx_Bl = sample_with_top_k_top_p_also_inplace_modifying_logits_(logits_BlV, rng=rng, top_k=top_k or self.top_k, top_p=top_p or self.top_p, num_samples=1)[:, :, 0]
            _t_scale.mark('logits')
            # ------ process sampled tokens, prepare next AR step's input ------
            if vae_type != 0:           # vae_type = 32
                assert returns_vemb
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[si]
                else:
                    assert pn[0] == 1
                    idx_Bld = idx_Bld.reshape(B, pn[1], pn[2], -1) # shape: [B, h, w, d] or [B, h, w, 4d]
                    if self.apply_spatial_patchify: # unpatchify operation
                        idx_Bld = idx_Bld.permute(0,3,1,2) # [B, 4d, h, w]
                        idx_Bld = torch.nn.functional.pixel_shuffle(idx_Bld, 2) # [B, d, 2h, 2w]
                        idx_Bld = idx_Bld.permute(0,2,3,1) # [B, 2h, 2w, d]
                    idx_Bld = idx_Bld.unsqueeze(1) # [B, 1, h, w, d] or [B, 1, 2h, 2w, d]

                idx_Bld_list.append(idx_Bld)
                codes = vae.quantizer.lfq.indices_to_codes(idx_Bld, label_type='bit_label') # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                if si != num_stages_minus_1:
                    summed_codes += F.interpolate(codes, size=vae_scale_schedule[-1], mode=vae.quantizer.z_interplote_up)
                    last_stage = F.interpolate(summed_codes, size=vae_scale_schedule[si+1], mode=vae.quantizer.z_interplote_up) # [B, d, 1, h, w] or [B, d, 1, 2h, 2w]
                    last_stage = last_stage.squeeze(-3) # [B, d, h, w] or [B, d, 2h, 2w]
                    if self.apply_spatial_patchify: # patchify operation
                        last_stage = torch.nn.functional.pixel_unshuffle(last_stage, 2) # [B, 4d, h, w]
                    last_stage = last_stage.reshape(*last_stage.shape[:2], -1) # [B, d, h*w] or [B, 4d, h*w]
                    last_stage = torch.permute(last_stage, [0,2,1]) # [B, h*w, d] or [B, h*w, 4d]
                else:
                    summed_codes += codes
            else:
                if si < gt_leak:
                    idx_Bl = gt_ls_Bl[si]
                h_BChw = self.quant_only_used_in_inference[0].embedding(idx_Bl).float()   # BlC

                # h_BChw = h_BChw.float().transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1])
                h_BChw = h_BChw.transpose_(1, 2).reshape(B, self.d_vae, scale_schedule[si][0], scale_schedule[si][1], scale_schedule[si][2])
                ret.append(h_BChw if returns_vemb != 0 else idx_Bl)
                idx_Bl_list.append(idx_Bl)
                if si != num_stages_minus_1:
                    accu_BChw, last_stage = self.quant_only_used_in_inference[0].one_step_fuse(si, num_stages_minus_1+1, accu_BChw, h_BChw, scale_schedule)
            # -------------------------------------------------------------------

            if si != num_stages_minus_1:    # not last scale
                last_stage = self.word_embed(self.norm0_ve(last_stage))
                last_stage = last_stage.repeat(bs//B, 1, 1)
            _t_scale.stop('post')

        if inference_mode:  # disable attn's KV-Cache --> free memory
            for b in self.unregistered_blocks: (b.sa if isinstance(b, CrossAttnBlock) else b.attn).kv_caching(False)
        else:
            assert self.num_block_chunks > 1
            for block_chunk_ in self.block_chunks:
                for module in block_chunk_.module.module:
                    (module.sa if isinstance(module, CrossAttnBlock) else module.attn).kv_caching(False)

        if not ret_img:
            return ret, idx_Bl_list, []
        
        # ------ decode latent feature to img ------
        if vae_type != 0:
            img = vae.decode(summed_codes.squeeze(-3))
        else:
            img = vae.viz_from_ms_h_BChw(ret, scale_schedule=scale_schedule, same_shape=True, last_one=True)

        img = (img + 1) / 2
        img = img.permute(0, 2, 3, 1).mul_(255).to(torch.uint8).flip(dims=(3,))
        return ret, idx_Bl_list, img


if __name__ == '__main__':

    import argparse
    import cv2
    import os
    import time
    from torch import autocast
    from tools.run_infinity import load_tokenizer, load_visual_tokenizer, load_transformer, gen_one_img

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    # seed = random.randint(0, 10000)
    seed = 10086
    enable_positive_prompt=0

    # model_type = 'infinity_2b'
    model_type = 'infinity_8b'
    load_weight = True
    latency_profile = True
    prompt = """A group of students in a class"""

    if '2b' in model_type:
        vae_type = 32
        apply_spatial_patchify = 0
        checkpoint_type = 'torch'
        model_path = "pretrained_models/infinity/Infinity/infinity_2b_reg.pth"
        vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d32reg.pth'
        cfg = 3; tau = 0.5
        model_kwargs = dict(
            depth=32, embed_dim=2048, num_heads=2048//128, 
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8) # 2b model
    if '8b' in model_type:
        vae_type=14
        apply_spatial_patchify = 1
        checkpoint_type = 'torch_shard'
        model_path = "pretrained_models/infinity/Infinity/infinity_8b_weights"
        vae_path = 'pretrained_models/infinity/Infinity/infinity_vae_d56_f8_14_patchify.pth'
        cfg = 3; tau = 1.0
        model_kwargs = dict(
            depth=40, embed_dim=3584, num_heads=28, 
            drop_path_rate=0.1, mlp_ratio=4, block_chunks=8)
    text_encoder_ckpt = 'pretrained_models/infinity/flan-t5-xl'
    
    if not load_weight:
        model_path = None
        vae_path = None
        text_encoder_ckpt = None

    args=argparse.Namespace(
        pn='1M',
        model_path=model_path,
        cfg_insertion_layer=0,
        vae_type=vae_type,
        vae_path=vae_path,
        add_lvl_embeding_only_first_block=1,
        use_bit_label=1,
        model_type=model_type,
        rope2d_each_sa_layer=1,
        rope2d_normalized_by_hw=2,
        use_scale_schedule_embedding=0,
        sampling_per_bits=1,
        text_encoder_ckpt=text_encoder_ckpt,
        text_channels=2048,
        apply_spatial_patchify=apply_spatial_patchify,
        h_div_w_template=1.000,
        use_flex_attn=0,
        cache_dir='/dev/shm',
        checkpoint_type=checkpoint_type,
        seed=seed,
        bf16=1,
        save_file='tmp.jpg',
        # 
        enable_model_cache=0,
        # -- exp params ---
        attn_sink_scales=5,
        skip_last_scales=0,
        drop_uncond_last_scales=3)

    if not load_weight:
        # load vae
        vae = load_visual_tokenizer(args)
        # --> do not load transformer weight, for fast debug
        with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True), torch.no_grad():
            infinity_test: SparVAR_Infinity = SparVAR_Infinity(
                vae_local=vae, text_channels=2048, text_maxlen=512,
                shared_aln=True, raw_scale_schedule=None,
                checkpointing='full-block',
                customized_flash_attn=False,    # default: False
                fused_mlp=False,                 # default: False
                fused_norm=True,
                pad_to_multiplier=128,
                use_flex_attn=0,                # default: 0
                add_lvl_embeding_only_first_block=1,
                use_bit_label=1,
                rope2d_each_sa_layer=1,
                rope2d_normalized_by_hw=2,
                pn='1M',
                apply_spatial_patchify=apply_spatial_patchify,
                inference_mode=True,
                train_h_div_w_list=[1.0],
                **model_kwargs,
            ).to(device=device)
            print(f'[you selected Infinity with {model_kwargs=}] model size: {sum(p.numel() for p in infinity_test.parameters())/1e9:.2f}B, bf16={args.bf16}')

            if args.bf16:
                for block in infinity_test.unregistered_blocks:
                    block.bfloat16()

            infinity_test.eval()
            infinity_test.requires_grad_(False)

            infinity_test.cuda()
            torch.cuda.empty_cache()
    else:
        # load text encoder
        text_tokenizer, text_encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        # load vae
        vae = load_visual_tokenizer(args)
        # load infinity
        infinity_test = load_transformer(vae, args)
    
    # *------ t2i forward ------*
    h_div_w = 1/1 # aspect ratio, height:width
    h_div_w_template_ = h_div_w_templates[np.argmin(np.abs(h_div_w_templates-h_div_w))]
    scale_schedule = dynamic_resolution_h_w[h_div_w_template_][args.pn]['scales']
    scale_schedule = [(1, h, w) for (_, h, w) in scale_schedule]

    if not load_weight:
        # dummy_input
        text_cond_tuple = (
            torch.randn(size=[9, 2048], device=device, dtype=torch.float32),
            [9],
            torch.tensor([0, 9], device=device, dtype=torch.int32),
            9
        )

        if not latency_profile:
            _, _, img_list = infinity_test.autoregressive_infer_cfg(
                vae=vae,
                scale_schedule=scale_schedule,
                label_B_or_BLT=text_cond_tuple, g_seed=seed,
                B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                cfg_exp_k=0.0, cfg_insertion_layer=[0],
                vae_type=32, softmax_merge_topk=-1,
                ret_img=True, trunk_scale=1000,
                gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                sampling_per_bits=1,
            )
            img = img_list[0]
            print(f"{img.shape=}")
        
        else:
            # warmup GPU
            warmup_iterations = 10
            print(f"Starting GPU warm-up for {warmup_iterations} iterations...")
            with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                for _ in range(warmup_iterations):
                    _, _, _ = infinity_test.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
                        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                        cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                        returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                        cfg_exp_k=0.0, cfg_insertion_layer=[0],
                        vae_type=32, softmax_merge_topk=-1,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                        sampling_per_bits=1,
                    )
                    torch.cuda.synchronize(device=device)
            print("GPU warm-up finished.")

            # Influence speed test
            num_test_iterations = 100
            timings = []

            print(f"Starting inference speed test for {num_test_iterations} iterations...")
            with autocast("cuda", dtype=torch.bfloat16, enabled=True, cache_enabled=True):
                stt = time.time()
                for i in range(num_test_iterations):
                    start_time = time.perf_counter()    # for accurate timing

                    _, _, img_list = infinity_test.autoregressive_infer_cfg(
                        vae=vae,
                        scale_schedule=scale_schedule,
                        label_B_or_BLT=text_cond_tuple, g_seed=seed,
                        B=1, negative_label_B_or_BLT=None, force_gt_Bhw=None,
                        cfg_sc=3, cfg_list=[3]*len(scale_schedule), tau_list=[0.5]*len(scale_schedule), top_k=900, top_p=0.97,
                        returns_vemb=1, ratio_Bl1=None, gumbel=0, norm_cfg=False,
                        cfg_exp_k=0.0, cfg_insertion_layer=[0],
                        vae_type=32, softmax_merge_topk=-1,
                        ret_img=True, trunk_scale=1000,
                        gt_leak=0, gt_ls_Bl=None, inference_mode=True,
                        sampling_per_bits=1,
                    )
                    torch.cuda.synchronize(device=device)   # *Important*: Ensure that all CUDA operations are completed before recording the time

                    end_time = time.perf_counter()
                    timings.append(end_time - start_time)
                    if (i + 1) % 10 == 0:
                        print(f"Iteration {i+1}/{num_test_iterations} done.")

            print("Inference speed test finished.")
            
            batch_size = 1
            avg_latency = sum(timings) / len(timings)
            std_latency = torch.tensor(timings).std().item()
            throughput = batch_size / avg_latency if avg_latency > 0 else float('inf')

            print(f"\n--- Inference Performance ---")
            print(f"Batch Size: {batch_size}")
            print(f"Average Latency: {avg_latency * 1000:.2f} ms")
            print(f"Latency StdDev: {std_latency * 1000:.2f} ms")
            print(f"Throughput: {throughput:.2f} samples/sec")

            # --- print first and last times to observe warmup effect ---
            # print("\nFirst 5 latencies (ms):")
            # for t in timings[:5]:
            #     print(f"{t*1000:.2f}")

            # print("\nLast 5 latencies (ms):")
            # for t in timings[-5:]:
            #     print(f"{t*1000:.2f}")
    
    else:

        generated_image = gen_one_img(
            infinity_test,
            vae,
            text_tokenizer,
            text_encoder,
            prompt,
            g_seed=seed,
            gt_leak=0,
            gt_ls_Bl=None,
            cfg_list=cfg,
            tau_list=tau,
            scale_schedule=scale_schedule,
            cfg_insertion_layer=[args.cfg_insertion_layer],
            vae_type=args.vae_type,
            sampling_per_bits=args.sampling_per_bits,
            enable_positive_prompt=0,
        )
        args.save_file = f'{model_type}_test.jpg'
        os.makedirs(os.path.dirname(os.path.abspath(args.save_file)), exist_ok=True)
        cv2.imwrite(args.save_file, generated_image.cpu().numpy())
        print(f'Save to {os.path.abspath(args.save_file)}')
