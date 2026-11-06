from pathlib import Path
import json
import os

import tqdm

import numpy as np
from timm.models import register_model
import torch
import torch.nn as nn
import torch.nn.functional as F

from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG, load_from_file
from models.infinitystar.apg import normalized_guidance
from models.infinitystar.infinitystar_model import InfinityStar, MultipleLayers, TIMM_KEYS
from models.schedules.dynamic_resolution import get_first_full_spatial_size_scale_index
from models.sparvar.basic_sparvar_infinitystar import SelfAttnBlock
from models.sparvar.sparvar_video_cuda_graph_utils import (VideoBlockStackGraphRunner,
                                                           video_cuda_graph_enabled)


@register_model
def sparvar_infinitystar(
    depth=36, block_chunks=6, embed_dim=4096, num_heads=4096//128, num_key_value_heads=4096//128//4, 
    mlp_ratio=4, drop_path_rate=0, **kwargs
):
    return SparVAR_InfinityStar(
        arch='qwen',
        depth=depth,
        block_chunks=block_chunks,
        embed_dim=embed_dim,
        num_heads=num_heads,
        num_key_value_heads=num_key_value_heads,
        mlp_ratio=mlp_ratio,
        drop_path_rate=drop_path_rate,
        **{k: v for k, v in kwargs.items() if k not in TIMM_KEYS}
    )


class SparVAR_InfinityStar(InfinityStar):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # SparseDiffAttn reads GLOBAL_CONFIG during construction.
        # SPARVAR_CONFIG_FILE lets concurrent ablation jobs use isolated config
        # copies instead of racing on the canonical yml (ablation scripts honor
        # the same variable as their edit target).
        load_from_file(os.environ.get("SPARVAR_CONFIG_FILE",
                                      str(Path(__file__).with_name("sparse-attn-video-config.yml"))))

        self.unregistered_blocks = []
        for layer_idx in range(self.depth):
            block = SelfAttnBlock(
                embed_dim=self.C,   # 4096
                cond_dim=self.D,    # 4096
                num_heads=self.num_heads,   # 32
                num_key_value_heads=self.num_key_value_heads,   # 8
                mlp_ratio=self.mlp_ratio,   # 8
                use_flex_attn=self.use_flex_attn,   # False
                pad_to_multiplier=self.pad_to_multiplier,   # 128
                rope2d_normalized_by_hw=self.rope2d_normalized_by_hw,   # 2
                mask_type=self.other_args.mask_type,    # 'infinity_elegant_clip20frames_v2'
                context_frames=self.other_args.context_frames,  # 10000
                steps_per_frame=self.other_args.steps_per_frame,    # 3
                arch=self.arch,     # 'qwen'
                qwen_qkvo_bias=self.qwen_qkvo_bias,     # False
                inject_sync=self.other_args.inject_sync,    # 0
            )
            # TODO: ------ for attn map vis ------
            block.attn.layer_idx = layer_idx
            # block.bfloat16()
            self.unregistered_blocks.append(block)

        if self.num_block_chunks == 1:
            self.blocks = nn.ModuleList(self.unregistered_blocks)
            assert self.blocks[0] is self.unregistered_blocks[0]
        else:
            self.block_chunks = nn.ModuleList()
            for i in range(self.num_block_chunks):
                self.block_chunks.append(MultipleLayers(
                    self.unregistered_blocks,
                    self.num_blocks_in_a_chunk,
                    i * self.num_blocks_in_a_chunk,
                ))
            assert self.block_chunks[0].module[0] is self.unregistered_blocks[0]

        print(
            f'\n[SparVAR-InfinityStar]\n'
        )
        # CUDA-graph replay of the small image-pyramid scales (SPARVAR_VIDEO_CUDA_GRAPH=1,
        # default off); built lazily on the first video since it needs the scale schedule
        self._video_graph_runner = None

    def _get_video_graph_runner(self, scale_schedule, context_info, first_full_idx):
        if not video_cuda_graph_enabled():
            return None
        if self._video_graph_runner is None:
            self._video_graph_runner = VideoBlockStackGraphRunner(
                self, GLOBAL_CONFIG['attn'], scale_schedule, context_info, first_full_idx)
            print(f'[sparvar-video-cuda-graph] enabled: scales {self._video_graph_runner.scales}')
        elif self._video_graph_runner.scale_schedule != [tuple(int(v) for v in x) for x in scale_schedule]:
            return None                       # a different schedule: eager for this video
        return self._video_graph_runner

    # CS4A decision placement: tell every attn layer which repetition step is
    # running (used to trigger the repeat-0 full decision on the decision scale).
    def set_repetition_step_idx(self, repeat_idx: int):
        for block in self.unregistered_blocks:
            block.attn.repetition_step_idx = repeat_idx

    def _resolve_drop_uncond_steps(self, visual_repeat_steps):
        """Which (si, repeat_idx) forward steps drop the uncond branch.

        ``drop_uncond_last_scales`` is either a non-negative int (drop uncond on
        the last N visual forward steps) or the literal ``'auto'``. In 'auto'
        mode the drop suffix is anchored to the CS4A decision step so that the
        decision step and every step after it (in forward order) run without the
        uncond branch — matching what one gets by hand-counting N down to the
        decision step. The decision repeat mirrors ``_resolve_cs4a_decision_flags``:
        ``decision_repeat_idx=None`` -> the decision scale's last repeat, an
        integer k -> repeat k.
        """
        dul = self.drop_uncond_last_scales
        if not (isinstance(dul, str) and dul.strip().lower() == "auto"):
            n = int(dul)
            return set(visual_repeat_steps[-n:]) if n > 0 else set()

        attn_config = GLOBAL_CONFIG['attn']
        decision_scale = int(attn_config['decision_scale'])
        decision_repeat_idx = attn_config.get('decision_repeat_idx', None)

        # Repeats of the decision scale that actually run this inference (contiguous 0..max).
        repeats_here = [r for (s, r) in visual_repeat_steps if s == decision_scale]
        if not repeats_here:
            raise ValueError(
                f"drop_uncond_last_scales='auto': decision_scale={decision_scale} has no "
                f"forward step in the current scale schedule (len={len(visual_repeat_steps)})"
            )
        if decision_repeat_idx is None:
            decision_repeat = max(repeats_here)
        else:
            decision_repeat = int(decision_repeat_idx)
            if decision_repeat not in repeats_here:
                raise ValueError(
                    f"drop_uncond_last_scales='auto': decision_repeat_idx={decision_repeat} is out "
                    f"of range for decision_scale={decision_scale} (available repeats: 0..{max(repeats_here)})"
                )

        decision_step = (decision_scale, decision_repeat)
        idx = visual_repeat_steps.index(decision_step)
        drop_uncond_steps = set(visual_repeat_steps[idx:])
        # print(
        #     f"    [InfinityStar Exps] drop_uncond=auto -> anchor at decision step "
        #     f"scale-{decision_scale} repeat-{decision_repeat}; dropping uncond on the last "
        #     f"{len(drop_uncond_steps)} of {len(visual_repeat_steps)} visual steps"
        # )
        return drop_uncond_steps

    @torch.no_grad()
    def ar_infer_infinity_elegant(
            self,
            vae=None,
            scale_schedule=None,
            label_B_or_BLT=None,
            B=1, negative_label_B_or_BLT=None,
            g_seed=None, cfg_list=[], tau_list=[], top_k=0, top_p=0.0,
            trunk_scale=1000,
            gt_leak=0, gt_ls_Bl=None,
            low_vram_mode=False,
            args=None,
            get_visual_rope_embeds=None,
            context_info=None,
            return_summed_code_only=False,
            # TODO: ------ for attn map vis ------
            attn_map_recorder=None,
            cfg_similarity_recorder=None,
            **kwargs,
    ):  # returns List[idx_Bl]
        from models.schedules.infinity_elegant import interpolate

        if g_seed is None:
            rng = None
        else:
            self.rng.manual_seed(g_seed); rng = self.rng
        assert len(cfg_list) >= len(scale_schedule)
        assert len(tau_list) >= len(scale_schedule)
        assert args.use_cfg + args.use_apg == 1  # CFG / APG mutual exclusion
        device = label_B_or_BLT[0].device

        if self.apply_spatial_patchify:
            vae_scale_schedule = [(pt, 2 * ph, 2 * pw) for pt, ph, pw in scale_schedule]
        else:
            vae_scale_schedule = scale_schedule

        # calculate rope cache for this iteration
        self.rope2d_freqs_grid['freqs_text'] = self.rope2d_freqs_grid['freqs_text'].to(device)
        text_maxlen_this_iter = label_B_or_BLT[
            -1]  # self.text_maxlen # kv_compact, lens, cu_seqlens_k, max_seqlen_k = label_B_or_BLT
        prefix_tokens, lens = self.prepare_text_conditions(label_B_or_BLT, cfg_list, B, negative_label_B_or_BLT,
                                                           vae_scale_schedule, text_token_only=False,
                                                           text_maxlen_this_iter=text_maxlen_this_iter)
        bs = prefix_tokens.shape[0]

        ca_kv, cond_BD_or_gss, attn_mask = None, None, None
        ret, idx_Bl_list = [], []  # current length, list of reconstructed images
        for b in self.unregistered_blocks: b.attn.kv_caching(True)

        # TODO: ------ for attn map vis ------
        self.set_attention_map_recorder(attn_map_recorder)

        if attn_map_recorder is not None:
            attn_map_recorder.start_inference(scale_schedule=scale_schedule)
        if cfg_similarity_recorder is not None:
            cfg_similarity_recorder.start_inference(scale_schedule=scale_schedule)

        first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(scale_schedule)
        image_scale_repetition = np.array(json.loads(args.image_scale_repetition))
        video_scale_repetition = np.array(json.loads(args.video_scale_repetition))
        scales_in_one_clip = first_full_spatial_size_scale_index + 1
        assert len(image_scale_repetition) == len(
            video_scale_repetition), f'{len(image_scale_repetition)} != {len(video_scale_repetition)}'
        assert len(
            image_scale_repetition) == scales_in_one_clip, f'{len(image_scale_repetition)} != {scales_in_one_clip}'

        total_steps = image_scale_repetition.sum() + video_scale_repetition.sum() * (
                    len(scale_schedule) // len(video_scale_repetition) - 1) + 1  # +1 is prefix text token forward step
        pbar = tqdm.tqdm(total=total_steps)
        block_chunks = self.block_chunks if self.num_block_chunks > 1 else self.blocks

        #* Count the suffix over actual visual forward steps, e.g. scale_27_repeat1, scale_28_repeat0.
        visual_repeat_steps = []
        for step_si in range(len(scale_schedule)):
            rel_step_si = step_si % scales_in_one_clip
            if step_si < scales_in_one_clip:
                step_repeat_times = image_scale_repetition[rel_step_si]
            else:
                step_repeat_times = video_scale_repetition[rel_step_si]
            step_infer_repeat_times = min(int(step_repeat_times), args.max_repeat_times)
            for step_repeat_idx in range(step_infer_repeat_times):
                visual_repeat_steps.append((step_si, step_repeat_idx))
        # drop_uncond_last_scales: int -> last-N-steps suffix; 'auto' -> anchor the
        # suffix to the CS4A decision step (see _resolve_drop_uncond_steps).
        drop_uncond_steps = self._resolve_drop_uncond_steps(visual_repeat_steps)
        active_branch_repeat = bs // B

        noise_shape = vae_scale_schedule[0]
        if self.other_args.noise_input:
            noise = torch.randn((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype,
                                device=prefix_tokens.device)
        else:
            noise = torch.zeros((1, self.vae_embed_dim, *noise_shape), dtype=prefix_tokens.dtype,
                                device=prefix_tokens.device)  # torch.Size([1, 64, 1, 1, 1])

        summed_codes = [noise[0:1]]
        sos_token = self.embeds_codes2input(noise, active_branch_repeat)  # torch.Size([2, 1, 4096])

        # ------ text tokens forward ------
        rope_cache = self.rope2d_freqs_grid['freqs_text'][
            :, :, :, :, :text_maxlen_this_iter]  # torch.Size([2, 1, 1, 1, text_len, 64])
        last_stage = prefix_tokens  # torch.Size([2, text_len, 4096])
        pbar.update(1)
        if attn_map_recorder is not None:
            attn_map_recorder.set_step(step_key='t0', file_label='scale_t0')
        # get text KV cache
        for block_idx, b in enumerate(block_chunks):
            last_stage = b(
                x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask,
                attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache,
                scale_ind='t0', context_info=context_info, last_repetition_step=True)

        # ---------------------------------------
        # -------- visual tokens forward --------
        # ---------------------------------------
        graph_runner = self._get_video_graph_runner(scale_schedule, context_info, first_full_spatial_size_scale_index)
        if graph_runner is not None:
            graph_runner.begin_video(text_len=int(text_maxlen_this_iter), batch=bs)
        ref_text_scale_inds = ['t0']
        last_stage = sos_token  # torch.Size([2, 1, 4096])
        cum_scales = 0  # real repetition-aware scale
        for si, pn in enumerate(scale_schedule):  # si: i-th segment

            rel_si_in_one_clip = si % scales_in_one_clip
            if si < scales_in_one_clip:  # image
                repeat_times = image_scale_repetition[si % scales_in_one_clip]  # ! repeat time setting
                target_pn = vae_scale_schedule[first_full_spatial_size_scale_index]
            else:
                repeat_times = video_scale_repetition[si % scales_in_one_clip]
                target_pn = vae_scale_schedule[-1]

            cfg = cfg_list[si]
            infer_repeat_times = min(repeat_times, args.max_repeat_times)
            for repeat_idx in range(infer_repeat_times):
                drop_uncond_this_step = (si, repeat_idx) in drop_uncond_steps and active_branch_repeat > 1
                # print(f'{(si, repeat_idx)=}')
                if drop_uncond_this_step:
                    # print(f'    {drop_uncond_this_step=}')
                    last_stage = last_stage[:B]
                    active_branch_repeat = 1
                    self.keep_cond_branch_in_kv_cache(B)
                # print(f'real scale ind is : {cum_scales+repeat_idx}')

                # --- visual RoPE ---
                #       Recalculate visual RoPE for each repetition
                rope_cache = get_visual_rope_embeds(
                    self.rope2d_freqs_grid, scale_schedule, si, cum_scales + repeat_idx,
                    device, args, context_info, first_full_spatial_size_scale_index)
                pbar.update(1)
                # TODO: ------ for attn map vis ------
                if attn_map_recorder is not None:
                    attn_map_recorder.set_step(
                        step_key=f'scale_{si:02d}_repeat{repeat_idx}',
                        file_label=f'scale_{si:02d}_repeat{repeat_idx}',
                    )
                # CS4A: broadcast the current repetition step to all attn layers
                self.set_repetition_step_idx(repeat_idx)

                last_repetition_step = (repeat_idx == (infer_repeat_times - 1))
                graph_out = None
                if graph_runner is not None and graph_runner.eligible(si, last_stage, active_branch_repeat, B):
                    graph_out = graph_runner.run(si, repeat_idx, last_stage, rope_cache, last_repetition_step)
                if graph_out is not None:
                    last_stage = graph_out
                else:
                    for block_idx, b in enumerate(block_chunks):
                        last_stage = b(
                            x=last_stage, cond_BD=cond_BD_or_gss, ca_kv=ca_kv, attn_bias_or_two_vector=attn_mask,
                            attn_fn=None, scale_schedule=scale_schedule, rope2d_freqs_grid=rope_cache,
                            scale_ind=si, context_info=context_info, last_repetition_step=last_repetition_step,
                            ref_text_scale_inds=ref_text_scale_inds
                        )

                if cfg_similarity_recorder is not None and last_repetition_step:
                    cfg_similarity_recorder.capture_last_stage(
                        scale_index=si,
                        repeat_index=repeat_idx,
                        last_stage=last_stage,
                        batch_size=B,
                        is_semantic_scale=rel_si_in_one_clip < args.semantic_scales,
                    )

                logits_BlV = self.get_logits_during_infer(last_stage,
                                                          is_semantic_scale=rel_si_in_one_clip < args.semantic_scales).mul(
                    1 / tau_list[si])
                if cfg != 1:
                    # print(f'add cfg on add_cfg_on_logits')
                    if active_branch_repeat == 1:
                        logits_BlV = logits_BlV[:B]
                    elif args.use_cfg:
                        logits_BlV = cfg * logits_BlV[:B] + (1 - cfg) * logits_BlV[B:]
                    elif args.use_apg:
                        pred_cond = logits_BlV[:B]
                        pred_uncond = logits_BlV[B:]
                        pred_guided = normalized_guidance(pred_cond, pred_uncond, guidance_scale=cfg,
                                                          momentum_buffer=None, eta=0,
                                                          norm_threshold=args.apg_norm_threshold)
                        # pred_guided = cfg * pred_cond + (1-cfg) * pred_uncond
                        logits_BlV = pred_guided
                else:
                    logits_BlV = logits_BlV[:B]

                # -------- bit-wise sample --------
                tmp_bs, tmp_seq_len = logits_BlV.shape[:2]
                logits_BlV = logits_BlV.reshape(tmp_bs, -1, self.num_of_label_value)
                probs_Bld = logits_BlV.softmax(dim=-1)  # [B, thwd or thw4d, 2]
                idx_Bld = torch.multinomial(probs_Bld.view(-1, self.num_of_label_value), num_samples=1,
                                            replacement=True, generator=rng).view(tmp_bs, -1)  # [B, thwd or thw4d]
                probs_Bld = torch.gather(probs_Bld, dim=2, index=idx_Bld.unsqueeze(-1)).squeeze(-1)

                def Bld2Bthwd(item):
                    """Reshape the bit labels back to the spatiotemporal grid of the current scale
                    """
                    item = item.reshape(tmp_bs, tmp_seq_len, -1)  # [B, thw, d or 4d]
                    item = item.reshape(B, pn[0], pn[1], pn[2], -1)  # shape: [B, t, h, w, d] or [B, t, h, w, 4d]
                    if self.apply_spatial_patchify:  # unpatchify operation
                        item = item.permute(0, 1, 4, 2, 3)  # [B, t, 4d, h, w]
                        item = torch.nn.functional.pixel_shuffle(item, 2)  # [B, t, d, 2h, 2w]
                        item = item.permute(0, 1, 3, 4, 2)  # [B, t, 2h, 2w, d]
                    return item

                idx_Bld = Bld2Bthwd(idx_Bld)
                probs_Bld = Bld2Bthwd(probs_Bld)
                # print(f'{si=} {repeat_idx=} idx_Bld.shape={idx_Bld.shape}')

                # for I2V / reference-conditioned inference
                if si < gt_leak:
                    idx_Bld = gt_ls_Bl[cum_scales + repeat_idx]
                # idx_Bld [B, t, h, w, d] or [B, t, 2h, 2w, d]

                # -------- bit-label --> latent code -------
                if self.other_args.use_two_stage_lfq:
                    if pn[1] * pn[2] >= vae.quantizer.detail_scale_min_tokens:
                        is_semantic_scale = False
                        lfq = vae.quantizer.lfq_detail
                    else:
                        is_semantic_scale = True
                        lfq = vae.quantizer.lfq_semantic
                    codes = lfq.indices_to_codes(idx_Bld, 'bit_label')
                    codes = interpolate(codes, size=(self.vae_embed_dim, *target_pn),
                                        mode=vae.quantizer.z_interplote_up, quantizer=vae.quantizer,
                                        is_semantic_scale=is_semantic_scale).contiguous()
                else:
                    codes = vae.quantizer.lfq_detail.indices_to_codes(idx_Bld, 'bit_label')
                    codes = F.interpolate(codes, size=target_pn, mode=vae.quantizer.z_interplote_up)

                # -------- residual accumulation --------
                summed_codes[-1] = F.interpolate(summed_codes[-1], size=target_pn, mode=vae.quantizer.z_interplote_up)
                summed_codes[-1] += codes

                if repeat_idx < repeat_times - 1:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si],
                                               mode=vae.quantizer.z_interplote_down)
                    last_stage = self.embeds_codes2input(last_stage, active_branch_repeat)

            # After a scale is completed, update the real scale counter
            cum_scales += repeat_times

            # -------- prepare next scale input --------
            if si < len(scale_schedule) - 1:
                if scale_schedule[si][-2:] == scale_schedule[-1][-2:]:
                    if self.other_args.noise_input:
                        summed_codes.append(torch.randn((B, summed_codes[-1].shape[1], *vae_scale_schedule[si + 1]),
                                                        device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    else:
                        summed_codes.append(torch.zeros((B, summed_codes[-1].shape[1], *vae_scale_schedule[si + 1]),
                                                        device=summed_codes[-1].device, dtype=summed_codes[-1].dtype))
                    last_stage = summed_codes[-1]
                else:
                    last_stage = F.interpolate(summed_codes[-1], size=vae_scale_schedule[si + 1],
                                               mode=vae.quantizer.z_interplote_down)
                last_stage = self.embeds_codes2input(last_stage, active_branch_repeat)

        summed_codes = torch.cat(summed_codes, dim=-3)
        for b in self.unregistered_blocks: b.attn.kv_caching(False)

        # TODO: ------ for attn map vis ------
        self.set_attention_map_recorder(None)

        if return_summed_code_only:
            return summed_codes
        else:
            if low_vram_mode: vae.to('cuda')
            img = self.summed_codes2images(vae, summed_codes)
            return idx_Bl_list, img
