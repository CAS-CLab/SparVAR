"""Method-specific HART arguments; importing this module does not load models."""

from typing import List

from utils.arg_util import HARTInferArgs
from utils.hart_infer_utils import HART_MODEL_REGISTRY






class SparVARHARTArgs(HARTInferArgs):
    model_type: str = "sparvar_hart"
    accel_enabled: bool = True
    csla_preset: str = "hart_legacy"
    csla_query_scales: List[int] = [11, 12, 13]
    csla_layers: List[int] = []
    csla_sink: int = -1
    csla_windows: List[int] = []
    csla_block_size: int = 128
    csla_kv_layout: str = "native"
    csla_kernel: str = "flex"
    cs4a_layers: List[int] = []
    cs4a_decision_scale: int = 10
    cs4a_ratio: float = 0.5
    cs4a_use_o_cache: bool = True






def get_hart_args_class(model_type):
    classes = {
        "hart": HARTInferArgs,
        "sparvar_hart": SparVARHARTArgs,
    }
    if model_type not in HART_MODEL_REGISTRY:
        raise ValueError(f"Unknown HART model_type: {model_type!r}")
    return classes[model_type]
