"""SparVAR model package.

Keep this package initializer lightweight so importing helper submodules such
as ``models.sparvar.index_map_utils`` does not eagerly import model definitions
(whose transformer blocks need GPU-only packages like flash_attn/triton).
Model loading goes through ``models/registry.py``, which imports
``models.sparvar.sparvar_infinitystar_model`` directly; the lazy ``__getattr__``
below keeps ``from models.sparvar import SparVAR_InfinityStar`` working too.
"""

__all__ = ["SparVAR_InfinityStar"]


def __getattr__(name):
    if name == "SparVAR_InfinityStar":
        from .sparvar_infinitystar_model import SparVAR_InfinityStar
        return SparVAR_InfinityStar
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
