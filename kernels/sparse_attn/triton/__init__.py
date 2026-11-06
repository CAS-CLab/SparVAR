from .attn_csp import csp_attn
from .attn_dense import dense_attn
from .attn_dense_colsum import dense_colsum_attn
from .attn_colsum_only import colsum_only_attn

__all__ = [
    'csp_attn', 'dense_attn', 'dense_colsum_attn', 'colsum_only_attn',
    'csp_mlp_mm2_bf16', 'csp_mlp_mm2_bf16_function_ptr', 'csp_mlp_mm1_fp8', 'csp_mlp_mm1_bf16',
]

# Lazy exports for the sparse-MLP kernels: resolving csp_mlp_mm2_bf16_function_ptr
# compiles/launches a CUDA kernel, and the fp8 module needs sm89+. Deferring the
# imports keeps attention-only runs from ever touching the MLP modules.
_LAZY_MLP = {
    'csp_mlp_mm1_bf16': 'mlp_csp_mm1_bf16',
    'csp_mlp_mm1_fp8': 'mlp_csp_mm1_fp8',
    'csp_mlp_mm2_bf16': 'mlp_csp_mm2_bf16',
    'csp_mlp_mm2_bf16_function_ptr': 'mlp_csp_mm2_bf16',
}


def __getattr__(name):
    if name in _LAZY_MLP:
        from importlib import import_module
        return getattr(import_module(f'.{_LAZY_MLP[name]}', __name__), name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
