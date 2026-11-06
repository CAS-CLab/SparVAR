"""Generate a dense HART or SparVAR HART image."""
import argparse
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from utils.public_cli_utils import (add_common_arguments, load_hart_options, print_plan,
                                    require_assets, require_cuda, write_plan)


def main(argv=None):
    parser = add_common_arguments(argparse.ArgumentParser(description=__doc__))
    parser.add_argument('--model', choices=('hart', 'sparvar_hart'), default='sparvar_hart')
    parser.add_argument('--cfg', type=float, default=4.5)
    parser.add_argument('--options', help='SparVAR HART configure_acceleration JSON options.')
    parser.add_argument('--use-ema', action=argparse.BooleanOptionalAction, default=True)
    opts = parser.parse_args(argv)
    method_options = load_hart_options(opts.options, opts.model)
    plan = {**vars(opts), 'method_options': method_options,
            'preset_status': 'migration candidate; independent GPU reproduction pending'}
    if opts.dry_run:
        print_plan(plan)
        return
    require_assets(opts)
    torch = require_cuda()
    from utils.hart_arg_utils import get_hart_args_class
    from tools.run_hart import gen_one_img_hart, load_hart, load_hart_tokenizer
    from tools.hart_experiment_utils import seed_hart_inference
    args = get_hart_args_class(opts.model)()
    args.model_type, args.model_path = opts.model, str(Path(opts.model_path).expanduser().resolve())
    args.vae_path = str(Path(opts.vae_path).expanduser().resolve())
    args.text_encoder_ckpt = str(Path(opts.text_encoder).expanduser().resolve())
    args.cfg, args.seed, args.use_ema = opts.cfg, opts.seed, opts.use_ema
    output = write_plan(plan, opts.output)
    args.outdir = str(output.parent)
    for key, value in method_options.items():
        if not hasattr(args, key):
            raise ValueError(f'Unsupported HART option: {key}')
        setattr(args, key, value)
    seed_hart_inference(opts.seed)
    with torch.inference_mode(), torch.autocast('cuda', dtype=torch.float16):
        tokenizer, encoder = load_hart_tokenizer(args)
        model, ema_model = load_hart(args)
        image = gen_one_img_hart(model, args.use_ema, ema_model, tokenizer, encoder, opts.prompt,
                                 cfg=opts.cfg, max_token_length=args.max_token_length,
                                 use_llm_system_prompt=args.use_llm_system_prompt,
                                 more_smooth=args.more_smooth, g_seed=opts.seed,
                                 final_stage=args.final_stage, num_maskgit_iters=args.num_maskgit_iters)
    image.save(output)
    print(f'Saved image: {output}')


if __name__ == '__main__':
    main()
