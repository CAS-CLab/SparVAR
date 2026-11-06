"""Env-gated real-density probe for the D-CSLA sparse path.

``SPARVAR_DCSLA_DENSITY_LOG=1`` prints one line per (scale, layer) call from
``dcsla_sparse_attention``, answering the question every bench speedup currently
hangs on: where does coverage tau actually land on REAL attention mass (the
bench uses synthetic uniform mass, so its "achieved density" is meaningless).

    [dcsla-density] config: {'dcsla_select_mode': 'coverage', 'dcsla_tau': 0.9, ...}
    [dcsla-density] scale=28 layer=03 B=2 V=277 mean=0.212 head_p10=0.148 head_p90=0.286 row_min=0.104 row_max=0.502

``mean`` is the fleet number to read (fraction of KV blocks attended, i.e. the
effective FLOP fraction of the block-sparse forward); head_p10/p90 spread over
per-(batch,head) means shows whether a per-head budget is warranted; row_min/max
are extremes over all (b, h, q-block) rows.

``SPARVAR_DCSLA_DENSITY_CSV=<path>`` (independent of the LOG switch) appends the
RAW per-(batch, head) mean densities as CSV rows ``scale,layer,b,h,density`` —
the input of the per-head budget-allocation figure (a print line can only carry
percentiles). Same cost profile; diagnostic runs only.

Pure logging — never mutates the selection. Costs a few .item() syncs per layer
call; enable for diagnostic runs only.
"""
import os
from typing import Optional

import torch
from torch import Tensor

_ENABLED: Optional[bool] = None
_CSV_PATH: Optional[str] = None
_csv_resolved = False
_csv_header_written = False
_printed_cfg = False

_CFG_KEYS = (
    'dcsla_select_mode', 'dcsla_tau', 'dcsla_topk_ratio', 'dcsla_min_blocks',
    'dcsla_max_blocks_ratio', 'dcsla_local_diag_blocks', 'dcsla_keep_sink',
    'dcsla_sink_select_tau', 'dcsla_mass_normalize', 'dcsla_colsum_group_rows',
)


def _enabled() -> bool:
    global _ENABLED
    if _ENABLED is None:
        _ENABLED = os.environ.get("SPARVAR_DCSLA_DENSITY_LOG", "0") not in ("0", "", "false", "False")
    return _ENABLED


def _csv_path() -> Optional[str]:
    global _CSV_PATH, _csv_resolved
    if not _csv_resolved:
        _csv_resolved = True
        _CSV_PATH = os.environ.get("SPARVAR_DCSLA_DENSITY_CSV") or None
    return _CSV_PATH


@torch.no_grad()
def maybe_log_dcsla_density(
    kv_num_blocks: Tensor,          # [B, H, U] int32 from select_kv_blocks
    *,
    V: int,
    scale_ind,
    layer_ind,
    attn_config: Optional[dict] = None,
) -> None:
    csv_path = _csv_path()
    if not _enabled() and csv_path is None:
        return
    d = kv_num_blocks.float() / float(V)                       # [B, H, U] per-row density
    per_head_bh = d.mean(dim=-1)                               # [B, H] per-(batch, head) mean
    if csv_path is not None:
        global _csv_header_written
        vals = per_head_bh.cpu()
        with open(csv_path, "a", encoding="utf-8") as f:
            if not _csv_header_written:
                _csv_header_written = True
                if f.tell() == 0:
                    f.write("scale,layer,b,h,density\n")
            for b in range(vals.shape[0]):
                for h in range(vals.shape[1]):
                    f.write(f"{scale_ind},{int(layer_ind)},{b},{h},{vals[b, h].item():.6f}\n")
    if not _enabled():
        return
    global _printed_cfg
    if not _printed_cfg and attn_config is not None:
        _printed_cfg = True
        cfg = {k: attn_config.get(k) for k in _CFG_KEYS}
        print(f"[dcsla-density] config: {cfg}")
    per_head = per_head_bh.flatten()                           # [B*H]
    q = per_head.quantile(torch.tensor([0.1, 0.9], device=per_head.device))
    print(f"[dcsla-density] scale={scale_ind} layer={int(layer_ind):02d} B={d.shape[0]} V={V} "
          f"mean={d.mean().item():.3f} head_p10={q[0].item():.3f} head_p90={q[1].item():.3f} "
          f"row_min={d.min().item():.3f} row_max={d.max().item():.3f}")
