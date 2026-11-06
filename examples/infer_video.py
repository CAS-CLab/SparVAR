"""Generate a five-second InfinityStar/SparVAR text-to-video sample."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.public_cli_utils import (add_common_arguments, int_or_auto, print_plan,
                                    require_assets, require_cuda, sparse_config, sparse_config_record, write_plan)


def main(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--model', choices=('infinitystar_qwen8b', 'sparvar_infinitystar'), default='sparvar_infinitystar')
    parser.add_argument('--resolution', choices=('480p', '720p'), default='480p')
    parser.add_argument('--attention', choices=('cs4a', 'dcsla'), default='cs4a')
    parser.add_argument('--sparse-config')
    parser.add_argument('--drop-uncond', type=int_or_auto, default=None)
    parser.add_argument('--cfg', type=float, default=34.0)
    parser.add_argument('--tau-image', type=float, default=1.0)
    parser.add_argument('--tau-video', type=float, default=0.4)
    parser.add_argument('--low-vram', action='store_true')
    opts = parser.parse_args(argv)
    is_sparse = opts.model == 'sparvar_infinitystar'
    if opts.drop_uncond == 'auto' and not is_sparse:
        parser.error('auto uncond removal is available only for SparVAR.')
    if not is_sparse and (opts.attention != 'cs4a' or opts.sparse_config):
        parser.error('The dense model does not use a sparse attention configuration.')
    drop = opts.drop_uncond if opts.drop_uncond is not None else ('auto' if is_sparse else 0)
    config = sparse_config(f'configs/sparvar/infinitystar_{opts.resolution}_{opts.attention}.yml', opts.sparse_config) if is_sparse else None
    plan = {**vars(opts), 'sparse_config': config, 'drop_uncond': drop, 'fps': 16, 'frames': 81,
            'guidance': 'APG', 'preset_status': 'migration candidate; independent GPU reproduction pending'}
    plan['sparse_config_record'] = sparse_config_record(config, 'video', opts.attention) if config else None
    if opts.dry_run:
        print_plan(plan)
        return
    require_assets(opts)
    if config:
        os.environ['SPARVAR_CONFIG_FILE'] = config
    torch = require_cuda()
    import numpy as np
    from tools.run_infinity import InferencePipe, gen_one_video, save_video
    from utils.arg_util_video import InferArgs
    from models.schedules.dynamic_resolution import get_dynamic_resolution_meta, get_first_full_spatial_size_scale_index
    args = InferArgs()
    args.model_type = opts.model
    args.model_path = str(Path(opts.model_path).expanduser().resolve())
    args.vae_path = str(Path(opts.vae_path).expanduser().resolve())
    args.text_encoder_ckpt = str(Path(opts.text_encoder).expanduser().resolve())
    args.resolution, args.pn = opts.resolution, '0.40M' if opts.resolution == '480p' else '0.90M'
    args.fps, args.video_frames, args.generation_duration = 16, 81, 5
    args.cfg, args.tau_image, args.tau_video = opts.cfg, opts.tau_image, opts.tau_video
    args.drop_uncond_last_scales, args.enable_rewriter = drop, 0
    args.low_vram_mode, args.baseline_video_path = int(opts.low_vram), ''
    if opts.resolution == '480p':
        args.video_scale_repetition = '[3,3,3,3,3,3,3,3,3,3,3,3,2,1]'
        args.detail_scale_min_tokens, args.semantic_scales = 350, 11
    torch.manual_seed(opts.seed)
    with torch.inference_mode():
        pipe = InferencePipe(args)
        schedules, aspects = get_dynamic_resolution_meta(args.dynamic_scale_schedule, args.video_frames)
        aspect = aspects[np.argmin(np.abs(aspects - 0.571))]
        schedule = schedules[aspect][args.pn]['pt2scale_schedule'][(args.video_frames - 1) // 4 + 1]
        args.first_full_spatial_size_scale_index = get_first_full_spatial_size_scale_index(schedule)
        args.tower_split_index = args.first_full_spatial_size_scale_index + 1
        context = pipe.get_scale_pack_info(schedule, args.first_full_spatial_size_scale_index, args)
        tau = [args.tau_image] * args.tower_split_index + [args.tau_video] * (len(schedule) - args.tower_split_index)
        prompt = f'<<<t=5s>>>{opts.prompt}' if args.append_duration2caption else opts.prompt
        video, _ = gen_one_video(pipe.infinity, pipe.vae, pipe.text_tokenizer, pipe.text_encoder,
                                  prompt, g_seed=opts.seed, cfg_list=opts.cfg, tau_list=tau,
                                  scale_schedule=schedule, cfg_insertion_layer=[0], vae_type=args.vae_type,
                                  sampling_per_bits=1, enable_positive_prompt=0,
                                  low_vram_mode=opts.low_vram, args=args,
                                  get_visual_rope_embeds=pipe.get_visual_rope_embeds, context_info=context)
    output = write_plan(plan, opts.output)
    save_video(video.cpu().numpy(), fps=args.fps, save_filepath=str(output))
    print(f'Saved video: {output}')


if __name__ == '__main__':
    main()
