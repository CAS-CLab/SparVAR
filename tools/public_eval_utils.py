"""Validate the model families included in the public SparVAR release."""

INFINITY_IMAGE_MODELS = frozenset({
    "infinity_2b", "infinity_8b", "sparvar_infinity_2b", "sparvar_infinity_8b",
})
HART_MODELS = frozenset({"hart", "sparvar_hart"})
VIDEO_MODELS = frozenset({"infinitystar_qwen8b", "sparvar_infinitystar"})


def validate_public_model_type(model_type, *, kind="image"):
    """Reject unavailable methods before loading weights or initializing CUDA."""
    supported = {
        "image": INFINITY_IMAGE_MODELS | HART_MODELS,
        "infinity_image": INFINITY_IMAGE_MODELS,
        "video": VIDEO_MODELS,
    }
    if kind not in supported:
        raise ValueError(f"Unknown public model family: {kind!r}")
    if model_type not in supported[kind]:
        choices = ", ".join(sorted(supported[kind]))
        raise ValueError(f"Model {model_type!r} is not included in this release; supported {kind} models: {choices}")
    return model_type
