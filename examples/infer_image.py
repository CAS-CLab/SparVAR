"""Generate an Infinity/SparVAR image using local checkpoints."""
import argparse
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.public_cli_utils import (add_common_arguments, int_or_auto, print_plan,
                                    require_assets, require_cuda, sparse_config, sparse_config_record, write_plan)


def main(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--model', choices=('infinity_2b', 'infinity_8b', 'sparvar_infinity_2b', 'sparvar_infinity_8b'), default='sparvar_infinity_2b')
    parser.add_argument('--cfg', type=float, default=4.0)
    parser.add_argument('--tau', type=float, default=1.0)
    parser.add_argument('--drop-uncond', type=int_or_auto, default=None)
    parser.add_argument('--sparse-config')
    opts = parser.parse_args(argv)
    is_sparse = opts.model.startswith('sparvar_')
    size = '8b' if opts.model.endswith('8b') else '2b'
    if opts.drop_uncond == 'auto' and not is_sparse:
        parser.error('auto uncond removal is available only for SparVAR.')
    drop = opts.drop_uncond if opts.drop_uncond is not None else (3 if is_sparse else 0)
    config = sparse_config(f'configs/sparvar/infinity_{size}.yml', opts.sparse_config) if is_sparse else None
    plan = {**vars(opts), 'sparse_config': config, 'drop_uncond': drop, 'resolution': [1024, 1024],
            'preset_status': 'migration candidate; independent GPU reproduction pending'}
    plan['sparse_config_record'] = sparse_config_record(config, 'image') if config else None
    if opts.dry_run:
        print_plan(plan)
        return
    require_assets(opts)
    if config:
        os.environ['SPARVAR_CONFIG_FILE'] = config
    torch = require_cuda()
    import cv2
    import numpy as np
    from tools.run_infinity import gen_one_img, load_tokenizer, load_transformer, load_visual_tokenizer
    from utils.arg_util import InfinityInferArgs
    from utils.dynamic_resolution import dynamic_resolution_h_w, h_div_w_templates
    args = InfinityInferArgs()
    args.model_type, args.model_path = opts.model, str(Path(opts.model_path).expanduser().resolve())
    args.vae_path = str(Path(opts.vae_path).expanduser().resolve())
    args.text_encoder_ckpt = str(Path(opts.text_encoder).expanduser().resolve())
    args.cfg, args.tau, args.seed, args.drop_uncond_last_scales = opts.cfg, opts.tau, opts.seed, drop
    args.enable_model_cache = 0
    if size == '8b':
        args.vae_type, args.apply_spatial_patchify, args.checkpoint_type = 14, 1, 'torch_shard'
    torch.manual_seed(opts.seed)
    with torch.inference_mode():
        tokenizer, encoder = load_tokenizer(t5_path=args.text_encoder_ckpt)
        vae = load_visual_tokenizer(args)
        model = load_transformer(vae, args)
        aspect = h_div_w_templates[np.argmin(np.abs(h_div_w_templates - 1.0))]
        schedule = [(1, h, w) for _, h, w in dynamic_resolution_h_w[aspect][args.pn]['scales']]
        image = gen_one_img(model, vae, tokenizer, encoder, opts.prompt, g_seed=opts.seed,
                            cfg_list=opts.cfg, tau_list=opts.tau, scale_schedule=schedule,
                            cfg_insertion_layer=[args.cfg_insertion_layer], vae_type=args.vae_type,
                            sampling_per_bits=args.sampling_per_bits, enable_positive_prompt=0)
    output = write_plan(plan, opts.output)
    if not cv2.imwrite(str(output), image.cpu().numpy()):
        raise RuntimeError(f'Could not save image: {output}')
    print(f'Saved image: {output}')


if __name__ == '__main__':
    main()
