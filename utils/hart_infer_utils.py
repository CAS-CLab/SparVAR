"""Lightweight method selection and runtime options for HART inference."""

from importlib import import_module


HART_PATCH_NUMS = (1, 2, 3, 4, 5, 7, 9, 12, 16, 21, 27, 36, 48, 64)
HART_MODEL_REGISTRY = {
    "hart": ("models.hart.hart_transformer_t2i", "HARTForT2I"),
    "sparvar_hart": ("models.sparvar.sparvar_model_hart", "SparVAR_HART"),
}

HART_METHOD_DEFAULTS = {
    "hart": {},
    "sparvar_hart": {
        "csla_preset": "hart_legacy", "csla_query_scales": [11, 12, 13],
        "csla_layers": [], "csla_sink": -1, "csla_windows": [],
        "csla_block_size": 128, "csla_kv_layout": "native",
        "csla_kernel": "flex",
        "cs4a_layers": [], "cs4a_decision_scale": 10,
        "cs4a_ratio": 0.5, "cs4a_use_o_cache": True,
    },
}


def validate_hart_method_options(model_type, options=None):
    """Resolve defaults without importing models or optional CUDA packages."""
    if model_type not in HART_MODEL_REGISTRY:
        raise ValueError(f"Unknown HART model_type: {model_type!r}")
    defaults = {
        "accel_enabled": model_type != "hart",
        "dense_attn_backend": "flash_attn",
        "use_fused_rope": False,
        "hart_micro_cache": False,
        "hart_rope_cache": False,
        "hart_compile_rope": False,
        "hart_inference_mode": True,
        "hart_triton_rope": False,
        "hart_fa3_bhsd": False,
        "hart_fa3_start_scale": 0,
        "hart_quack_min_tokens": 0,
        "hart_cache_linear_weights": False,
        "hart_ffn_mode": "eager",
        "hart_compile_vae": False,
        **HART_METHOD_DEFAULTS[model_type],
    }
    supplied = dict(options or {})
    unknown = supplied.keys() - defaults.keys()
    if unknown:
        raise ValueError(f"Unsupported {model_type} options: {sorted(unknown)}")
    resolved = {**defaults, **supplied}
    # Do not share mutable preset lists across model instances.
    resolved = {key: list(value) if isinstance(value, (list, tuple)) else value
                for key, value in resolved.items()}
    if not isinstance(resolved["accel_enabled"], bool):
        raise ValueError("accel_enabled must be a boolean")
    if not isinstance(resolved["use_fused_rope"], bool):
        raise ValueError("use_fused_rope must be a boolean")
    if resolved["use_fused_rope"] and model_type not in ("hart", "sparvar_hart"):
        raise ValueError("Fused RoPE currently supports hart and sparvar_hart only")
    for option in ("hart_micro_cache", "hart_rope_cache", "hart_compile_rope", "hart_triton_rope", "hart_cache_linear_weights", "hart_compile_vae"):
        if not isinstance(resolved[option], bool):
            raise ValueError(f"{option} must be a boolean")
        if resolved[option] and model_type not in ("hart", "sparvar_hart"):
            raise ValueError(f"{option} currently supports hart and sparvar_hart only")
    if resolved["hart_rope_cache"] and resolved["use_fused_rope"]:
        raise ValueError("hart_rope_cache currently caches the naive RoPE path only")
    if resolved["hart_compile_rope"] and resolved["use_fused_rope"]:
        raise ValueError("Choose either compiled or original fused RoPE")
    if resolved["hart_triton_rope"] and (resolved["use_fused_rope"] or resolved["hart_compile_rope"]):
        raise ValueError("Triton RoPE cannot be combined with another RoPE implementation")
    if not isinstance(resolved["hart_inference_mode"], bool):
        raise ValueError("hart_inference_mode must be a boolean")
    if not resolved["hart_inference_mode"] and model_type not in ("hart", "sparvar_hart"):
        raise ValueError("The no_grad experiment currently supports hart and sparvar_hart only")
    if model_type == "hart" and resolved["accel_enabled"]:
        raise ValueError("The HART baseline does not implement acceleration")
    if resolved["dense_attn_backend"] not in ("flash_attn", "sdpa", "fa3"):
        raise ValueError("dense_attn_backend must be 'flash_attn', 'sdpa' or 'fa3'")
    if resolved["dense_attn_backend"] == "fa3" and model_type not in ("hart", "sparvar_hart"):
        raise ValueError("FA3 currently supports hart and sparvar_hart only")
    if not isinstance(resolved["hart_fa3_bhsd"], bool):
        raise ValueError("hart_fa3_bhsd must be a boolean")
    if resolved["hart_fa3_bhsd"] and resolved["dense_attn_backend"] != "fa3":
        raise ValueError("hart_fa3_bhsd requires the fa3 backend")
    start_scale = resolved["hart_fa3_start_scale"]
    if not isinstance(start_scale, int) or not 0 <= start_scale <= 13:
        raise ValueError("hart_fa3_start_scale must be an integer from 0 to 13")
    if start_scale and not resolved["hart_fa3_bhsd"]:
        raise ValueError("A mixed SDPA/FA3 schedule requires hart_fa3_bhsd")
    minimum = resolved["hart_quack_min_tokens"]
    if type(minimum) is not int or minimum < 0:
        raise ValueError("hart_quack_min_tokens must be a nonnegative integer")
    if minimum and model_type not in ("hart", "sparvar_hart"):
        raise ValueError("Quack RMSNorm currently supports hart and sparvar_hart only")
    if resolved["hart_ffn_mode"] not in ("eager", "adaptive"):
        raise ValueError("hart_ffn_mode must be eager or adaptive")
    if resolved["hart_ffn_mode"] != "eager" and model_type not in ("hart", "sparvar_hart"):
        raise ValueError("Adaptive FFN currently supports hart and sparvar_hart only")
    return resolved


def get_hart_method_options(args):
    model_type = args.model_type
    defaults = validate_hart_method_options(model_type)
    return validate_hart_method_options(
        model_type, {key: getattr(args, key, value) for key, value in defaults.items()}
    )


def resolve_hart_model_class(model_type):
    if model_type not in HART_MODEL_REGISTRY:
        raise ValueError(f"Unknown HART model_type: {model_type!r}")
    module, name = HART_MODEL_REGISTRY[model_type]
    return getattr(import_module(module), name)


def validate_hart_acceleration_model(model):
    """Reject geometries that have no validated accelerated implementation yet."""
    if tuple(model.patch_nums) != HART_PATCH_NUMS:
        raise ValueError("HART acceleration currently requires the 1024px 14-scale schedule")
    if (model.attn_type != "llama" or not model.disable_aln
            or model.config.shared_aln):
        raise ValueError(
            "HART acceleration requires attn_type='llama', disable_aln=True, shared_aln=False"
        )
