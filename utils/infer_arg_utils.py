"""Helpers to parse inference args for the T2I evaluation scripts.

Model-level args live in Tap classes (InfinityInferArgs / HARTInferArgs in
utils/arg_util.py) and are dispatched on --model_type; benchmark-level args
stay in each eval script's own argparse parser and are parsed from the argv
tokens left over by Tap (known_only=True).

Typical usage in an eval script::

    args = parse_infer_args()
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_samples', type=int, default=4)
    ...
    args = merge_script_args(args, parser)

For video (InfinityStar-family) entry scripts, `parse_video_infer_args` plays
the same dispatch role over the utils/arg_util_video.InferArgs subclasses.
"""

import sys
from utils.arg_util import HARTInferArgs, InfinityInferArgs


def _peek_model_type(argv, default='infinity_2b'):
    """Read --model_type from raw argv without consuming it."""
    for i, tok in enumerate(argv):
        if tok == '--model_type' and i + 1 < len(argv):
            return argv[i + 1]
        if tok.startswith('--model_type='):
            return tok.split('=', 1)[1]
    return default


def parse_infer_args(argv=None):
    """Parse model-level inference args, dispatching the Tap class on --model_type.

    Unrecognized (benchmark-level) tokens are kept in ``args.extra_args`` for
    ``merge_script_args`` to consume.
    """
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    # drop the legacy launcher-injected rank flag (torchrun uses the LOCAL_RANK env var)
    argv = [tok for tok in argv if not tok.startswith(('--local-rank=', '--local_rank='))]

    model_type = _peek_model_type(argv)
    if model_type in ('hart', 'sparvar_hart'):
        from utils.hart_arg_utils import get_hart_args_class
        args_cls = get_hart_args_class(model_type)
    elif model_type in ('infinity_2b', 'infinity_8b',
                        'sparvar_infinity_2b', 'sparvar_infinity_8b'):
        args_cls = InfinityInferArgs
    else:
        raise ValueError(f"Unsupported image release model_type: {model_type!r}")
    return args_cls(explicit_bool=True).parse_args(args=argv, known_only=True)


def parse_hart_infer_args(argv=None, known_only=False):
    """Use the same method dispatch for HART profiling and evaluation."""
    from utils.hart_arg_utils import get_hart_args_class
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    argv = [tok for tok in argv if not tok.startswith(('--local-rank=', '--local_rank='))]
    args_cls = get_hart_args_class(_peek_model_type(argv, default='hart'))
    return args_cls(explicit_bool=True).parse_args(args=argv, known_only=known_only)


VIDEO_INFER_ARGS_REGISTRY = {
    'infinitystar_qwen8b': ('utils.arg_util_video', 'InferArgs'),
    'sparvar_infinitystar': ('utils.arg_util_video', 'InferArgs'),
}


def parse_video_infer_args(argv=None):
    """Parse strict arguments for the two released InfinityStar model types."""
    argv = list(sys.argv[1:]) if argv is None else list(argv)
    argv = [tok for tok in argv if not tok.startswith(('--local-rank=', '--local_rank='))]
    model_type = _peek_model_type(argv, default='infinitystar_qwen8b')
    if model_type not in VIDEO_INFER_ARGS_REGISTRY:
        raise ValueError(f"Unsupported video release model_type: {model_type!r}")
    from utils.arg_util_video import InferArgs
    return InferArgs().parse_args(args=argv)


def merge_script_args(model_args, parser):
    """Parse benchmark-level args from ``model_args.extra_args`` and merge them in.

    Raises if the script parser re-defines a model-level field, which would
    otherwise silently clobber the value Tap already parsed.
    """
    script_args = parser.parse_args(model_args.extra_args)
    for k, v in vars(script_args).items():
        if k in model_args.class_variables:
            raise ValueError(
                f"benchmark arg '--{k}' duplicates a model-level arg of "
                f"{type(model_args).__name__}; define it only in the Tap infer args class"
            )
        setattr(model_args, k, v)
    return model_args
