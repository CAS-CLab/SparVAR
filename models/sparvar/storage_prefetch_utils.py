"""Cross-layer prefetch for the per-layer offloaded attention storage.

``MaybeOffloadedTensor``'s PIPELINE_DEPTH=2 slot rotation is designed for a
"load layer i+1 while layer i computes" schedule, but ``SparseDiffAttn``'s
sparse step calls ``load_async() + load_async_wait()`` back-to-back for its
OWN layer. The load stream first waits on all compute enqueued so far and the
compute stream then waits on the load stream, so every layer pays the full
H2D copy as a serial bubble between two compute regions (~0.9 ms/layer for
the ~21 MB packed decision mask).

This module supplies the missing piece:

  * a ``layer_num -> AttnStorage`` registry, filled at ``SparseDiffAttn``
    init;
  * ``prefetch_next_layer_storage(layer_num)``, called by the sparse step
    right AFTER its own ``load_async_wait()`` (so that wait does not cover
    the new copy) and before its heavy compute — the next layer's copy then
    runs on the side load stream, overlapped with this layer's index mapping
    and csp_attn.

Correctness relies on ``load_async()`` being idempotent per ``offload()``:
``gpu_tensor_owners`` (see ``storage/offloaded_tensor.py``) records which
instance's data currently fills the shared GPU slot, so the next layer's own
``load_async()`` finds its prefetched copy and skips the duplicate H2D copy,
while same-parity layers sharing the slot (i and i+2) still reload. Its
``load_async_wait()`` synchronizes on the load stream as before.

There is deliberately NO wrap-around from the last layer to layer 0: the
stored pattern is refreshed by the next decision step, and a cross-video
prefetch would race that offload. The first layer of each sparse scale
therefore pays one un-overlapped load.

Assumes one sparse model per process (re-registration by layer_num simply
overwrites). Everything degrades to a cheap no-op when offloading is
disabled: ``load_async()`` returns before touching the streams.
"""

from typing import Dict

from models.sparvar.storage import AttnStorage

_STORAGE_REGISTRY: Dict[int, AttnStorage] = {}


def register_layer_storage(layer_num: int, storage: AttnStorage) -> None:
    """Called once per attention layer at ``SparseDiffAttn`` init."""
    _STORAGE_REGISTRY[int(layer_num)] = storage


def prefetch_next_layer_storage(layer_num: int) -> None:
    """Kick off the H2D reload of layer ``layer_num + 1``'s stored sparse
    pattern on the side load stream. No-op for the last layer, before anything
    was offloaded, or when offloading is disabled."""
    nxt = _STORAGE_REGISTRY.get(int(layer_num) + 1)
    if nxt is not None:
        nxt.load_async()
