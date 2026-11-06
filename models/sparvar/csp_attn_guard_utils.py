"""Host-side validator for ``csp_attn`` inputs (debug aid).

The cuda/TMA ``csp_attn`` kernel only ``TORCH_CHECK``s shapes / strides / dtypes
(see ``kernels/chipmunk_tk2/csrc/attn/csp_attn.cu``). It does NOT validate the index
VALUES or the per-group counts, so an out-of-range index (kernel gathers K out of
bounds) or an over-large count (kernel reads indices past the row) surfaces as an
*asynchronous* ``CUDA error: an illegal memory access`` deep inside some later op —
with a misleading traceback.

This guard replicates the kernel's host checks AND adds the value/count-range checks
the kernel lacks, raising a clear *synchronous* error at the call site that names the
exact violated invariant and the offending tensor.

Off by default (the ``.max()/.min()/.item()`` checks force GPU syncs and are slow).
Enable with:

    SPARVAR_CSP_GUARD=1
"""

import os

import torch

from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG

_TRUE = {"1", "true", "yes", "on"}
_ENABLED = os.environ.get("SPARVAR_CSP_GUARD", "0").lower() in _TRUE


def csp_guard_enabled() -> bool:
    return _ENABLED


@torch.no_grad()
def check_csp_attn_inputs(q, k, v, inds, counts, o, multiple_of=None, where=""):
    """Raise a clear error if ``(q, k, v, inds, counts, o)`` would violate ``csp_attn``'s
    requirements. No-op unless ``SPARVAR_CSP_GUARD=1``.

    Mirrors the kernel's host ``TORCH_CHECK``s and additionally checks the things the
    kernel does NOT (and which therefore cause a silent illegal memory access):
      * every index value is in ``[0, k_seqlen)``  -> else K is gathered out of bounds
      * every group's count is in ``[0, indices_width]`` -> else indices read past the row

    ``where`` is a short label for the call site (e.g. "reuse", "sparse").
    """
    if not _ENABLED:
        return

    # o=None means "no O_cache accumulate" — csp_attn then returns ΔO shaped like q,
    # so validate against q as the stand-in.
    if o is None:
        o = q

    tag = f"[csp_guard:{where}] " if where else "[csp_guard] "

    def _fail(msg):
        raise RuntimeError(
            tag + msg
            + f"\n    q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)} o={tuple(o.shape)}"
            + f"\n    inds shape={tuple(inds.shape)} stride={tuple(inds.stride())}"
            f" dtype={inds.dtype} contig={inds.is_contiguous()} ptr%16={inds.data_ptr() % 16}"
            + f"\n    counts shape={tuple(counts.shape)} dtype={counts.dtype}"
        )

    # ---- dtype / ndim (csp_attn.cu:351-352) ----
    if inds.dtype != torch.int32:
        _fail(f"indices must be int32, got {inds.dtype}")
    if counts.dtype != torch.int32:
        _fail(f"indices_counts must be int32, got {counts.dtype}")
    if inds.ndim != 4 or counts.ndim != 3:
        _fail(f"indices must be 4D, counts 3D; got inds {inds.ndim}D / counts {counts.ndim}D")
    if q.ndim != 4 or k.ndim != 4 or v.ndim != 4 or o.ndim != 4:
        _fail("q/k/v/o must all be 4D [B, H, S, D]")

    B, Hq, Sq, D = q.shape
    _, Hk, Sk, Dk = k.shape

    # ---- batch / head / seqlen / head_dim consistency (csp_attn.cu:355-385) ----
    if k.shape[0] != B or v.shape[0] != B or o.shape[0] != B \
            or inds.shape[0] != B or counts.shape[0] != B:
        _fail(f"batch (dim0) mismatch: q={B} k={k.shape[0]} v={v.shape[0]} o={o.shape[0]} "
              f"inds={inds.shape[0]} counts={counts.shape[0]}")
    if v.shape[2] != Sk:
        _fail(f"v seqlen {v.shape[2]} != k seqlen {Sk}")
    if o.shape[2] != Sq:
        _fail(f"o seqlen {o.shape[2]} != q seqlen {Sq}")
    if D != 128 or Dk != 128 or v.shape[3] != 128 or o.shape[3] != 128:
        _fail(f"head_dim must be 128; got q={D} k={Dk} v={v.shape[3]} o={o.shape[3]}")
    if o.shape[1] != Hq or inds.shape[1] != Hq or counts.shape[1] != Hq:
        _fail(f"qo_heads mismatch: q={Hq} o={o.shape[1]} inds={inds.shape[1]} counts={counts.shape[1]}")
    if v.shape[1] != Hk:
        _fail(f"kv_heads mismatch: k={Hk} v={v.shape[1]}")
    if Hq < Hk or Hq % Hk != 0:
        _fail(f"qo_heads {Hq} must be a positive multiple of kv_heads {Hk}")
    if inds.shape[2] != counts.shape[2]:
        _fail(f"query-group dim mismatch: inds groups {inds.shape[2]} != counts groups {counts.shape[2]}")
    if inds.shape[3] <= 0:
        _fail(f"indices sparse dim (idx 3) must be > 0, got {inds.shape[3]}")

    # ---- the TMA stride requirement (csp_attn.cu:376) ----
    # cp.async.bulk.tensor needs the per-group stride to be a multiple of 16 bytes (4 int32s).
    if GLOBAL_CONFIG['attn']['provider'] != 'tilelang' and (inds.stride(2) * 4) % 16 != 0:
        _fail(f"indices.stride(2)={inds.stride(2)} -> stride*4 not a multiple of 16 bytes. "
              f"Legal layouts: the mask_to_indices padded (non-contiguous) k_len view — do "
              f"NOT .contiguous()/.clone() the [:k_len] slice when k_len % 4 != 0 — or a "
              f"COMPACT contiguous width that is a multiple of 4 (out_width path).")
    if GLOBAL_CONFIG['attn']['provider'] != 'tilelang' and inds.data_ptr() % 16 != 0:
        _fail(f"indices.data_ptr() not 16-byte aligned (TMA gmem operand): {inds.data_ptr()}")

    # ---- value / count range checks the kernel does NOT do (the silent OOB) ----
    if counts.numel():
        c_min = int(counts.min().item())
        c_max = int(counts.max().item())
        if c_min < 0:
            _fail(f"counts.min()={c_min} < 0")
        if c_max > inds.shape[3]:
            _fail(f"counts.max()={c_max} > indices width {inds.shape[3]} "
                  f"-> kernel reads indices past the row")
        if GLOBAL_CONFIG['attn']['provider'] != 'tilelang' and multiple_of is not None and c_max > 0 and (c_max % multiple_of) != 0:
            _fail(f"counts.max()={c_max} is not a multiple of counts_multiple_of={multiple_of} "
                  f"(kernel iterates counts in blocks)")
    if inds.numel() and counts.numel():
        # csp_attn reads only each group's first counts[g] indices. The [counts[g]:width]
        # slack is intentionally-uninitialized scratch: mask_to_indices allocates the output
        # with torch::empty and writes only [0:padded_total] (== counts[g]) per row, leaving
        # the tail garbage. Validate ONLY that read region, else the harmless slack garbage
        # false-flags min()<0 / max()>=Sk (e.g. on the decision-scale reuse indices).
        width = inds.shape[3]
        col = torch.arange(width, device=inds.device, dtype=torch.int64).view(1, 1, 1, width)
        read_region = col < counts.to(torch.int64).unsqueeze(-1)     # [B,H,G,width] True where csp_attn reads
        read_inds = inds[read_region]
        if read_inds.numel():
            i_min = int(read_inds.min().item())
            i_max = int(read_inds.max().item())
            if i_min < 0:
                _fail(f"inds.min()={i_min} < 0 (within the counts-bounded read region)")
            if i_max >= Sk:
                _fail(f"inds.max()={i_max} >= k seqlen {Sk} -> kernel gathers K/V out of bounds "
                      f"(this is the classic silent illegal-memory-access cause)")
