"""Lightweight release CLI helpers; importing this module never loads a model."""
import argparse
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def int_or_auto(value):
    if str(value).lower() == 'auto':
        return 'auto'
    try:
        result = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError('Use a non-negative integer or auto.') from exc
    if result < 0:
        raise argparse.ArgumentTypeError('The suffix must be non-negative.')
    return result


def add_common_arguments(parser):
    parser.add_argument('--prompt', default='A cat sitting beside a window.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--model-path', help='Local transformer checkpoint file or directory.')
    parser.add_argument('--vae-path', help='Local visual tokenizer checkpoint file or directory.')
    parser.add_argument('--text-encoder', help='Local text encoder/tokenizer directory.')
    parser.add_argument('--output', required=True)
    parser.add_argument('--dry-run', action='store_true', help='Print the resolved plan without importing GPU packages.')
    return parser


def sparse_config(default, override=None):
    value = override or os.environ.get('SPARVAR_CONFIG_FILE')
    path = Path(value).expanduser().resolve() if value else ROOT / default
    if not path.is_file():
        raise FileNotFoundError(f'Sparse configuration does not exist: {path}')
    return str(path)


def load_hart_options(path, model):
    path = Path(path).expanduser().resolve() if path else ROOT / 'configs/sparvar/hart_csla.json'
    data = json.loads(path.read_text())
    if not isinstance(data, dict):
        raise ValueError('HART options must be a JSON object.')
    # Dense references must not inherit the sparse method enable flag.
    from utils.hart_infer_utils import validate_hart_method_options
    if model == 'hart':
        return validate_hart_method_options(model, {'accel_enabled': False})
    return validate_hart_method_options(model, data)


def require_assets(options):
    for field in ('model_path', 'vae_path', 'text_encoder'):
        value = getattr(options, field)
        if not value:
            raise ValueError(f'--{field.replace("_", "-")} is required unless --dry-run is used.')
        if not Path(value).expanduser().exists():
            raise FileNotFoundError(f'Missing {field}: {value}')


def require_cuda():
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError('Generation requires CUDA. Use --dry-run to inspect the plan without a GPU.')
    return torch


def write_plan(plan, output):
    output = Path(output).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(output.suffix + '.run.json').write_text(json.dumps(plan, indent=2) + '\n')
    return output


def print_plan(plan):
    print(json.dumps(plan, indent=2))


def sparse_config_record(path, family, attention=None):
    """Validate the shipped provider boundary and record the input recipe."""
    import hashlib
    import yaml
    payload = Path(path).read_bytes()
    config = yaml.safe_load(payload)
    if not isinstance(config, dict) or not isinstance(config.get('attn'), dict):
        raise ValueError('Sparse configuration must contain an attn mapping.')
    attn = dict(config['attn'])
    if attn.get('model_family') != family:
        raise ValueError(f'Sparse configuration model_family must be {family}.')
    if attn.get('provider') not in ('cuda', 'triton'):
        raise ValueError('This release includes only cuda/triton providers; TileLang is not bundled.')
    if config.get('mlp', {}).get('is_enabled', False):
        raise ValueError('Sparse MLP experiments are not enabled in this release.')
    overrides = {key: value for key, value in os.environ.items()
                 if key.startswith(('SPARVAR_', 'INFINITY_', 'INFINITYSTAR_', 'LPROF'))
                 or key == 'LVAR_ATTN_BACKEND'}
    if 'SPARVAR_DCSLA_ENABLED' in overrides:
        attn['dcsla_enabled'] = overrides['SPARVAR_DCSLA_ENABLED'].lower() in ('1', 'true', 'yes', 'on')
    if attention is not None and bool(attn.get('dcsla_enabled', False)) != (attention == 'dcsla'):
        raise ValueError('--attention disagrees with the sparse config or environment override.')
    if attn.get('dcsla_enabled', False) and attn.get('use_o_cache', False):
        raise ValueError('D-CSLA requires use_o_cache=false.')
    return {'sha256': hashlib.sha256(payload).hexdigest(), 'yaml_attention': config['attn'],
            'environment_overrides': overrides}
