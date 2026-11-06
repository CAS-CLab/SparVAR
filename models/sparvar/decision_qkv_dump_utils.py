"""Env-gated dump of the REAL decision-step q/k/v for kernel diagnosis.

The CUDA colsum-chain bugs under investigation are data-dependent (fp32 overflow
of the collapsed lse constant only triggers on massive-logit heads), so the
diagnostic in kernels/tests/test_colsum_lse_diagnosis.py needs the actual
decision-scale tensors, not synthetic ones.

Enable by env (all off by default -> zero overhead no-op):
  SPARVAR_DUMP_DECISION_QKV=<dir>     output directory (created if missing)
  SPARVAR_DUMP_DECISION_LAYERS=0,1    comma-separated layer_ind list (default "0,1")
  SPARVAR_DUMP_DECISION_EXIT=1        hard-exit the process once every configured
                                      layer is dumped (skips the expensive
                                      post-decision scales; diagnosis-only runs)

One file per (scale, layer): {dir}/decision_qkv_s{scale}_l{layer}.pt holding
q [B,Hq,Nq,D] / k,v [B,Hkv,Nk,D] (cpu, original dtype), scale, sink_len.
"""

import os

import torch

_DUMP_DIR = os.environ.get("SPARVAR_DUMP_DECISION_QKV", "")
_LAYERS = {
    int(x) for x in os.environ.get("SPARVAR_DUMP_DECISION_LAYERS", "0,1").split(",")
    if x.strip() != ""
}
_EXIT_AFTER = os.environ.get("SPARVAR_DUMP_DECISION_EXIT", "0").lower() in ("1", "true", "yes")
_dumped = set()


@torch.no_grad()
def maybe_dump_decision_qkv(q, k, v, scale, scale_ind, layer_ind):
    """Call at the TOP of the decision (full) step with the raw kernel inputs."""
    if not _DUMP_DIR:
        return
    if layer_ind not in _LAYERS or (scale_ind, layer_ind) in _dumped:
        return
    os.makedirs(_DUMP_DIR, exist_ok=True)
    path = os.path.join(_DUMP_DIR, f"decision_qkv_s{scale_ind}_l{layer_ind}.pt")
    torch.save(
        {
            "q": q.detach().contiguous().cpu(),
            "k": k.detach().contiguous().cpu(),
            "v": v.detach().contiguous().cpu(),
            "scale": None if scale is None else float(scale),
            "scale_ind": int(scale_ind),
            "layer_ind": int(layer_ind),
            "sink_len": max(int(k.shape[-2]) - int(q.shape[-2]), 0),
        },
        path,
    )
    _dumped.add((scale_ind, layer_ind))
    print(f"[decision-dump] saved {path}  q={tuple(q.shape)} k={tuple(k.shape)} scale={scale}")
    if _EXIT_AFTER and {l for (_, l) in _dumped} >= _LAYERS:
        print(f"[decision-dump] all configured layers {sorted(_LAYERS)} dumped -> exiting "
              f"(SPARVAR_DUMP_DECISION_EXIT=1)")
        os._exit(0)
