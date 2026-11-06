from importlib import import_module
from timm.models import is_model


EXACT_MODEL_MODULES = {
    "infinity_2b": "models.infinity.infinity_model",
    "infinity_8b": "models.infinity.infinity_model",
    "sparvar_infinity_2b": "models.sparvar.sparvar_model",
    "sparvar_infinity_8b": "models.sparvar.sparvar_model",
    "infinitystar_qwen8b": "models.infinitystar.infinitystar_model",
    "sparvar_infinitystar": "models.sparvar.sparvar_infinitystar_model",
}


def ensure_model_registered(model_type: str) -> None:
    module_name = EXACT_MODEL_MODULES.get(model_type)
    if module_name is None:
        raise ValueError(
            f"Unsupported release model_type: {model_type!r}; "
            f"choose one of {tuple(EXACT_MODEL_MODULES)}"
        )
    if not is_model(model_type):
        import_module(module_name)
    if not is_model(model_type):
        raise RuntimeError(f"Unknown model after registry import: {model_type}")
