"""Env-gated Infinity (image) attention pattern statistics.

Image counterpart of ``attn_pattern_stats_utils.py`` (InfinityStar): by default
processes EVERY query row of the selected scales exactly (streamed in row chunks
of SPARVAR_PATTERN_ROW_CHUNK) over the STACKED-PYRAMID image KV layout; set
SPARVAR_PATTERN_IMG_FULL=0 for the legacy sampled mode (SPARVAR_PATTERN_QBLOCKS /
ROWS_PER_QBLOCK). The sampled mode carries a q-block-alignment bias whenever W_q
divides the 128 q-block size (scale 12 of 1024^2: every block-center row lands on
the x=0 image edge), which is why full-exact is the default. Writes per-layer /
per-head summaries for:

  * sink (first-N-scales), sink1 (scale-0 token), CSLA local-window, scale-band,
    CSLA dead-zone ("blocked": scales with win_size == -1 outside the sink) and
    residual attention mass — all COLUMN-level, kernel-tied region definitions;
  * oracle COLUMN coverage per sampled q-group under fixed budgets (csp_attn picks
    arbitrary columns per q-group, so columns — not 128-blocks — are the achievable
    selection granularity for the image kernel);
  * static CSLA-mask coverage, both fine (column) and 128-block-coarsened (the
    deployed FlexAttention CSLA mask is block-coarsened);
  * transferred CS4A coverage, simulated with the SAME runtime mapping code
    (``get_infinity_parent_lut`` / ``get_infinity_coord_lut`` /
    ``_get_image_local_window_mask``), recorded for BOTH expansion modes:
    ``transfer_*`` = footprint (deployed default), ``transfer_center_*`` =
    center_union.

The decision-scale colsum needed for the transfer simulation is computed inside
this module (exact softmax, bm-grouped) when the decision scale is recorded — no
second hook is required.

Hooked from ``models/infinity/basic_infinity.py`` ``SelfAttention.forward`` right
after the kv-cache concat (dense baseline path). Default behavior is off. Enable
with SPARVAR_PATTERN_STATS=1.
"""

import csv
import functools
import json
import math
import os
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import torch
from torch import Tensor

from models.sparvar.attn_pattern_stats_utils import (
    _TRUE,
    _cdiv,
    _parse_float_list,
    _parse_int_selection,
    _parse_int_set,
    _sample_q_blocks,
    _sample_rows_for_blocks,
    _selected,
    _selected_heads,
    current_pattern_prompt_id,
)
from models.sparvar.index_map_utils import (
    _get_image_local_window_mask,
    get_infinity_coord_lut,
    get_infinity_parent_lut,
)
from utils.misc import time_str


def _parse_int_list(name: str, default: str) -> List[int]:
    raw = os.environ.get(name, default)
    return [int(x) for x in raw.split(",") if x.strip() != ""]


@functools.lru_cache(maxsize=1)
def _img_stats_config() -> dict:
    return {
        "enabled": os.environ.get("SPARVAR_PATTERN_STATS", "0").lower() in _TRUE,
        "out_dir": os.environ.get(
            "SPARVAR_PATTERN_OUT",
            f"work_dir/play_models/SparVAR/Infinity_2b/pattern_stats_{time_str()}",
        ),
        "scales": _parse_int_set("SPARVAR_PATTERN_SCALES", "10,11,12"),
        "layers": _parse_int_selection(os.environ.get("SPARVAR_PATTERN_LAYERS", "all")),
        "heads": _parse_int_selection(os.environ.get("SPARVAR_PATTERN_HEADS", "all")),
        # full-exact (default): every q row, streamed in row chunks; qblocks /
        # rows_per_qblock only apply to the legacy sampled mode (IMG_FULL=0)
        "full_rows": os.environ.get("SPARVAR_PATTERN_IMG_FULL", "1").lower() in _TRUE,
        "row_chunk": max(1, int(os.environ.get("SPARVAR_PATTERN_ROW_CHUNK", "512"))),
        "qblocks": max(1, int(os.environ.get("SPARVAR_PATTERN_QBLOCKS", "16"))),
        "rows_per_qblock": max(1, int(os.environ.get("SPARVAR_PATTERN_ROWS_PER_QBLOCK", "1"))),
        "head_chunk": max(1, int(os.environ.get("SPARVAR_PATTERN_HEAD_CHUNK", "4"))),
        "block_size": max(1, int(os.environ.get("SPARVAR_PATTERN_BLOCK_SIZE", "128"))),
        "budgets": _parse_float_list("SPARVAR_PATTERN_BUDGETS", "0.1,0.2,0.3"),
        # ---- image-specific region geometry ----
        "sink_scales": max(0, int(os.environ.get("SPARVAR_PATTERN_IMG_SINK_SCALES", "5"))),
        "band_s": max(0, int(os.environ.get("SPARVAR_PATTERN_IMG_BAND_S", "1"))),
        # CSLA per-KV-scale window sizes (mask_mod semantics: -1 -> no window)
        "win_size": _parse_int_list(
            "SPARVAR_PATTERN_IMG_WIN_SIZE", "-1,-1,-1,-1,-1,-1,-1,-1,1,1,3,5,7"),
        # ---- CS4A transfer simulation (mirrors the runtime image path) ----
        "decision_scale": int(os.environ.get("SPARVAR_PATTERN_IMG_DECISION_SCALE", "10")),
        "transfer_topk": float(os.environ.get("SPARVAR_PATTERN_IMG_TRANSFER_TOPK", "0.2")),
        "transfer_multiple": max(1, int(os.environ.get("SPARVAR_PATTERN_IMG_TRANSFER_MULTIPLE", "64"))),
        "transfer_bm": max(1, int(os.environ.get("SPARVAR_PATTERN_IMG_TRANSFER_BM", "192"))),
        "transfer_union_local": os.environ.get(
            "SPARVAR_PATTERN_IMG_TRANSFER_UNION_LOCAL", "0").lower() in _TRUE,
        # CS4A per-target-scale local window (center_union always unions it; footprint
        # unions it only when transfer_union_local is on)
        "cs4a_ws": _parse_int_list(
            "SPARVAR_PATTERN_IMG_CS4A_WS", "1,1,1,1,1,1,1,1,1,1,1,7,9"),
        # decision-side top-rho ratios for the token-wise geometric row transfer
        # (analysis-only; independent of the kernel-faithful masks above)
        "transfer_row_ratios": _parse_float_list("SPARVAR_PATTERN_TRANSFER_ROW_RATIOS", "0.2,0.15"),
        "verbose": os.environ.get("SPARVAR_PATTERN_VERBOSE", "1").lower() in _TRUE,
    }


def image_pattern_stats_enabled_for(scale_ind, layer_ind) -> bool:
    cfg = _img_stats_config()
    if not cfg["enabled"]:
        return False
    if not isinstance(scale_ind, int) or not isinstance(layer_ind, int):
        return False
    if cfg["scales"] and int(scale_ind) not in cfg["scales"]:
        return False
    return _selected(cfg["layers"], int(layer_ind))


def _image_hw_cum(scale_schedule, upto_scale_ind: int) -> Tuple[List[Tuple[int, int]], List[int]]:
    """Per-scale (h, w) and cumulative token offsets (python ints) for the image KV."""
    hw = [(int(s[-2]), int(s[-1])) for s in scale_schedule[:upto_scale_ind + 1]]
    cum = [0]
    for h, w in hw:
        cum.append(cum[-1] + h * w)
    return hw, cum


def _win_at(win_list: Sequence[int], scale_ind: int, default: int) -> int:
    if 0 <= scale_ind < len(win_list):
        return int(win_list[scale_ind])
    return int(win_list[-1]) if win_list else int(default)


def _target_indices_count(kv_len: int, top_keys: float, multiple_of: int) -> int:
    indices_count = int(multiple_of * round((float(top_keys) * int(kv_len)) / multiple_of))
    indices_count = max(multiple_of, indices_count)
    return min(indices_count, int(kv_len))


class _InfinityPatternStats:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "stats.jsonl"
        self.csv_path = self.out_dir / "stats_heads.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        # layer -> {"scale_ind", "k_len", "colsum" [Hs, Gs, Kdec] fp32 (selected heads)}
        self._decision: Dict[int, dict] = {}
        self._warned = set()

    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            print(msg)

    # ------------------------------------------------------------------
    # decision-scale colsum (bm-grouped, exact) for the transfer simulation
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _cache_decision_colsum(self, q0: Tensor, k0: Tensor, scale, layer_ind: int, heads: List[int]) -> None:
        cfg = self.cfg
        device = q0.device
        q_len = int(q0.shape[-2])
        kv_len = int(k0.shape[-2])
        bm = int(cfg["transfer_bm"])
        G = _cdiv(q_len, bm)
        colsum = torch.zeros(len(heads), G, kv_len, device=device, dtype=torch.float32)
        key_t = k0.transpose(-2, -1)
        head_chunk = int(cfg["head_chunk"])
        for pos in range(0, len(heads), head_chunk):
            cur = heads[pos:pos + head_chunk]
            ht = torch.tensor(cur, device=device, dtype=torch.int64)
            k_chunk_t = key_t.index_select(1, ht)
            for g in range(G):
                rows = torch.arange(g * bm, min((g + 1) * bm, q_len), device=device, dtype=torch.int64)
                q_chunk = q0.index_select(1, ht).index_select(2, rows)
                logits = torch.matmul(q_chunk, k_chunk_t).float()
                logits.mul_(float(scale))
                colsum[pos:pos + len(cur), g] = torch.softmax(logits, dim=-1).sum(dim=-2).squeeze(0)
        self._decision[int(layer_ind)] = {
            "scale_ind": int(cfg["decision_scale"]),
            "k_len": kv_len,
            "colsum": colsum,
        }
        if cfg["verbose"]:
            print(f"[img-pattern-stats] cached decision colsum: layer={layer_ind} "
                  f"G={G} k_len={kv_len} heads={len(heads)}")

    # ------------------------------------------------------------------
    # CS4A transfer masks — reuses the RUNTIME mapping code (bit-faithful)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _build_transfer_masks(
        self, entry: dict, scale_ind: int, scale_schedule,
        cum: List[int], kv_len: int, q_len: int, device,
    ) -> Dict[str, Tensor]:
        cfg = self.cfg
        S = int(entry["scale_ind"])
        k_dec = int(entry["k_len"])
        colsum = entry["colsum"]                                               # [Hs, Gs, Kdec]
        Hs, Gs, _ = colsum.shape
        bm = int(cfg["transfer_bm"])
        Gk = _cdiv(q_len, bm)

        count = _target_indices_count(k_dec, cfg["transfer_topk"], cfg["transfer_multiple"])
        top_idx = colsum.topk(count, dim=-1).indices                           # [Hs, Gs, count]
        decision_mask = torch.zeros(Hs, Gs, k_dec, dtype=torch.bool, device=device)
        decision_mask.scatter_(-1, top_idx, True)

        # Query Block Homography (same formula as the runtime q-group remap)
        gk_centers = torch.arange(Gk, device=device, dtype=torch.float32) + 0.5
        gS_idx = torch.round(gk_centers / Gk * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)

        sink_len = cum[min(max(int(cfg["sink_scales"]), 0), int(scale_ind))]
        win = _win_at(cfg["cs4a_ws"], int(scale_ind), 1)
        local_mask = _get_image_local_window_mask(
            (scale_schedule[scale_ind][-2], scale_schedule[scale_ind][-1]),
            cum[scale_ind], kv_len, bm, q_len, win, device)                    # [Gk, kv_len]

        # --- footprint (deployed default): inverse parent-LUT mask expansion ---
        parent = get_infinity_parent_lut(
            decision_scale_ind=S, current_scale_ind=int(scale_ind),
            scale_schedule=scale_schedule, current_k_len=kv_len,
            decision_k_len=k_dec, device=device)
        fp = decision_mask[:, gS_idx][:, :, parent]                            # [Hs, Gk, kv_len]
        if sink_len > 0:
            fp[..., :sink_len] = True
        if bool(cfg["transfer_union_local"]):
            fp = fp | local_mask.view(1, Gk, kv_len)

        # --- center_union: coord-LUT centers ∪ local window ∪ sink ---
        coord = get_infinity_coord_lut(
            decision_scale_ind=S, current_scale_ind=int(scale_ind),
            scale_schedule=scale_schedule, current_k_len=kv_len,
            decision_k_len=k_dec, device=device)                               # [Kdec] int32
        grp_idx = top_idx.index_select(1, gS_idx)                              # [Hs, Gk, count]
        mapped = coord[grp_idx.long().clamp_(0, k_dec - 1)].long()             # [Hs, Gk, count]
        ct = local_mask.view(1, Gk, kv_len).expand(Hs, Gk, kv_len).clone()
        ct.scatter_(-1, mapped.clamp_(0, kv_len - 1), True)
        if sink_len > 0:
            ct[..., :sink_len] = True

        return {"footprint": fp, "center": ct}

    # ------------------------------------------------------------------
    # main entry
    # ------------------------------------------------------------------
    @torch.no_grad()
    def record(self, q: Tensor, k: Tensor, *, scale, scale_ind: int, layer_ind: int,
               scale_schedule, is_bhlc: bool) -> None:
        cfg = self.cfg
        if not is_bhlc:
            self._warn_once("[img-pattern-stats] skip: q/k in BLHc (flash) layout; "
                            "run the baseline with the SDPA (non-flash) branch")
            return
        if scale_schedule is None:
            self._warn_once("[img-pattern-stats] skip: scale_schedule is None")
            return
        if q.shape[0] == 0 or q.shape[1] == 0:
            return

        q0 = q[:1]                                                             # cond branch only
        k0 = k[:1]
        device = q0.device
        num_heads = int(q0.shape[1])
        q_len = int(q0.shape[-2])
        kv_len = int(k0.shape[-2])

        hw, cum = _image_hw_cum(scale_schedule, int(scale_ind))
        if cum[-1] != kv_len or hw[-1][0] * hw[-1][1] != q_len:
            self._warn_once(
                f"[img-pattern-stats] skip scale={scale_ind}: geometry mismatch "
                f"(kv_len={kv_len} vs cum={cum[-1]}, q_len={q_len} vs {hw[-1]})")
            return
        H_q, W_q = hw[-1]

        heads = _selected_heads(num_heads, cfg)
        if not heads:
            return

        block_size = int(cfg["block_size"])
        if cfg["full_rows"]:
            # exact mode (default): every q row, streamed in row chunks below.
            # No sampling estimator and no q-block offset alignment bias (with
            # block-aligned offsets, W_q | block_size makes every sampled row
            # share the same x — at scale 12 of 1024^2 all land on x=0).
            q_rows = torch.arange(q_len, device=device, dtype=torch.int64)
            row_to_sample_block = q_rows // block_size
            sampled_q_blocks = torch.arange(_cdiv(q_len, block_size), device=device, dtype=torch.int64)
        else:
            q_block_ids = _sample_q_blocks(q_len, block_size, int(cfg["qblocks"]))
            q_rows, row_to_sample_block, sampled_q_blocks = _sample_rows_for_blocks(
                q_block_ids, q_len, block_size, int(cfg["rows_per_qblock"]), device)
        if q_rows.numel() == 0:
            return
        num_rows = int(q_rows.numel())
        num_sample_blocks = int(sampled_q_blocks.numel())

        # cache the decision colsum for the transfer simulation (once per layer pass)
        if int(scale_ind) == int(cfg["decision_scale"]):
            self._cache_decision_colsum(q0, k0, scale, int(layer_ind), heads)

        # ---------------- column-level region geometry (row-independent) ----------------
        kv = torch.arange(kv_len, device=device, dtype=torch.int64)
        cum_t = torch.tensor(cum, device=device, dtype=torch.int64)
        kv_scale = torch.bucketize(kv, cum_t[1:], right=True)                  # source scale of each column
        hs_t = torch.tensor([h for h, _ in hw], device=device, dtype=torch.int64)
        ws_t = torch.tensor([w for _, w in hw], device=device, dtype=torch.int64)
        H_k, W_k = hs_t[kv_scale], ws_t[kv_scale]
        rel = kv - cum_t[kv_scale]
        kv_y = torch.div(rel, W_k, rounding_mode="trunc")
        kv_x = rel - kv_y * W_k
        win_list = list(cfg["win_size"]) or [-1]
        win_t = torch.tensor(
            [_win_at(win_list, int(j), -1) for j in range(int(scale_ind) + 1)],
            device=device, dtype=torch.int64)
        half_k = torch.div(win_t[kv_scale], 2, rounding_mode="trunc")
        half_k = torch.where(win_t[kv_scale] < 0, torch.full_like(half_k, -1), half_k)

        sink_scales = int(cfg["sink_scales"])
        sink_len = cum[min(sink_scales, int(scale_ind) + 1)]
        sink1_len = cum[min(1, int(scale_ind) + 1)]                            # scale-0 token(s)
        band_start = cum[max(int(scale_ind) - int(cfg["band_s"]), 0)]          # scale-distance <= band_s
        blocked = (kv_scale >= sink_scales) & (half_k < 0)                     # CSLA dead zone
        sink_col = kv < sink_len                                               # [K]

        # ---------------- CS4A transfer masks (built before the row loop) ----------------
        transfer_masks = None
        if int(scale_ind) == int(cfg["decision_scale"]):
            transfer_status = "decision_scale"
        else:
            entry = self._decision.get(int(layer_ind))
            if entry is None:
                transfer_status = "missing_decision"
            else:
                transfer_masks = self._build_transfer_masks(
                    entry, int(scale_ind), scale_schedule, cum, kv_len, q_len, device)
                transfer_status = "ok"
        # token-wise geometric row transfer: per-head top-rho decision tokens
        # (exact column mass = sum of the cached bm-group colsums), expanded to
        # the target KV via the pyramid parent LUT. No query grouping, no forced
        # sink, no kernel alignment — an analysis-only mask.
        row_transfer_masks = None
        row_transfer_density = None
        if transfer_masks is not None:
            entry = self._decision[int(layer_ind)]
            k_dec = int(entry["k_len"])
            col_mass = entry["colsum"].sum(dim=1)                              # [Hs, Kdec]
            parent = get_infinity_parent_lut(
                decision_scale_ind=int(entry["scale_ind"]), current_scale_ind=int(scale_ind),
                scale_schedule=scale_schedule, current_k_len=kv_len,
                decision_k_len=k_dec, device=device)
            row_transfer_masks = {}
            row_transfer_density = {}
            for r in cfg["transfer_row_ratios"]:
                kk = min(max(1, int(math.ceil(float(r) * k_dec))), k_dec)
                sel = torch.zeros(col_mass.shape[0], k_dec, device=device, dtype=torch.bool)
                sel.scatter_(1, col_mass.topk(kk, dim=-1).indices, True)
                key = f"{float(r):.4g}"
                row_transfer_masks[key] = sel.index_select(1, parent)          # [Hs, K]
                row_transfer_density[key] = row_transfer_masks[key].float().mean(dim=-1)

        bm = int(cfg["transfer_bm"])
        row_groups_all = (q_rows // bm).clamp_(0, _cdiv(q_len, bm) - 1)

        # ---------------- streaming pass over the q rows ----------------
        num_sel_heads = len(heads)
        sums = {name: torch.zeros(num_sel_heads, device=device, dtype=torch.float32)
                for name in ("sink", "sink1", "local", "band", "blocked",
                             "static", "static_block", "fp_cov", "fp_den", "ct_cov", "ct_den")}
        group_mass = torch.zeros(num_sel_heads, num_sample_blocks, kv_len, device=device, dtype=torch.float32)
        row_budget_ks = [
            (f"{float(r):.4g}", min(max(1, int(math.ceil(float(r) * kv_len))), kv_len))
            for r in cfg["budgets"]
        ]
        row_k_max = max((kk for _, kk in row_budget_ks), default=0)
        row_oracle_sums = {key: torch.zeros(num_sel_heads, device=device, dtype=torch.float32)
                           for key, _ in row_budget_ks}
        row_transfer_sums = None
        if row_transfer_masks is not None:
            row_transfer_sums = {key: torch.zeros(num_sel_heads, device=device, dtype=torch.float32)
                                 for key in row_transfer_masks}
        local_area_sum = torch.zeros((), device=device, dtype=torch.float32)
        static_density_sum = torch.zeros((), device=device, dtype=torch.float32)
        static_block_density_sum = torch.zeros((), device=device, dtype=torch.float32)

        V = _cdiv(kv_len, block_size)
        kv_block_ids = kv // block_size
        key_t = k0.transpose(-2, -1)
        head_chunk = int(cfg["head_chunk"])
        row_chunk = int(cfg["row_chunk"])
        for row_start in range(0, num_rows, row_chunk):
            rows_c = q_rows[row_start:row_start + row_chunk]
            blocks_c = row_to_sample_block[row_start:row_start + row_chunk]
            groups_c = row_groups_all[row_start:row_start + row_chunk]
            rc = int(rows_c.numel())

            # CSLA local windows (exact mask_mod semantics): per KV scale j, a win//2
            # neighborhood around the rescaled query center round(q/H_q * H_j).
            q_y = torch.div(rows_c, W_q, rounding_mode="trunc").float()
            q_x = (rows_c - torch.div(rows_c, W_q, rounding_mode="trunc") * W_q).float()
            center_y = torch.round(q_y.view(-1, 1) / H_q * H_k.view(1, -1).float())
            center_x = torch.round(q_x.view(-1, 1) / W_q * W_k.view(1, -1).float())
            window = (
                ((center_y - kv_y.view(1, -1).float()).abs() <= half_k.view(1, -1).float())
                & ((center_x - kv_x.view(1, -1).float()).abs() <= half_k.view(1, -1).float())
            )                                                                  # [rc, K]
            static_fine = window | sink_col.view(1, -1)                        # the CSLA mask
            local = window & (~sink_col.view(1, -1))                           # disjoint from sink

            blk_hits = torch.zeros(rc, V, device=device, dtype=torch.float32)
            blk_hits.scatter_add_(1, kv_block_ids.view(1, -1).expand(rc, -1), static_fine.float())
            static_block = blk_hits.gather(1, kv_block_ids.view(1, -1).expand(rc, -1)) > 0

            local_area_sum += local.float().mean(dim=-1).sum()
            static_density_sum += static_fine.float().mean(dim=-1).sum()
            static_block_density_sum += static_block.float().mean(dim=-1).sum()

            for pos in range(0, num_sel_heads, head_chunk):
                cur = heads[pos:pos + head_chunk]
                ht = torch.tensor(cur, device=device, dtype=torch.int64)
                q_chunk = q0.index_select(1, ht).index_select(2, rows_c)       # [1, Hc, rc, c]
                logits = torch.matmul(q_chunk, key_t.index_select(1, ht)).float()
                logits.mul_(float(scale))
                probs = torch.softmax(logits, dim=-1).squeeze(0)               # [Hc, rc, K]
                hsl = slice(pos, pos + len(cur))

                sums["sink"][hsl] += probs[:, :, :sink_len].sum(dim=(-1, -2))
                sums["sink1"][hsl] += probs[:, :, :sink1_len].sum(dim=(-1, -2))
                sums["local"][hsl] += (probs * local.to(probs.dtype).unsqueeze(0)).sum(dim=(-1, -2))
                sums["band"][hsl] += probs[:, :, band_start:].sum(dim=(-1, -2))
                sums["blocked"][hsl] += probs[:, :, blocked].sum(dim=(-1, -2))
                sums["static"][hsl] += (probs * static_fine.to(probs.dtype).unsqueeze(0)).sum(dim=(-1, -2))
                sums["static_block"][hsl] += (probs * static_block.to(probs.dtype).unsqueeze(0)).sum(dim=(-1, -2))
                group_mass[hsl].index_add_(1, blocks_c, probs)

                if transfer_masks is not None:
                    for mask_key, cov_key, den_key in (
                            ("footprint", "fp_cov", "fp_den"), ("center", "ct_cov", "ct_den")):
                        m_c = transfer_masks[mask_key][hsl].index_select(1, groups_c)  # [Hc, rc, K]
                        sums[cov_key][hsl] += (probs * m_c.to(probs.dtype)).sum(dim=(-1, -2))
                        sums[den_key][hsl] += m_c.float().sum(dim=(-1, -2))

                # token-wise per-row entry oracle: top-ceil(rho*K) entries of every row
                if row_k_max > 0:
                    top_vals = probs.topk(row_k_max, dim=-1).values           # sorted descending
                    csum = top_vals.cumsum(dim=-1)
                    for key, kk in row_budget_ks:
                        row_oracle_sums[key][hsl] += csum[..., kk - 1].sum(dim=-1)
                if row_transfer_sums is not None:
                    for key, tm in row_transfer_masks.items():
                        row_transfer_sums[key][hsl] += (
                            probs * tm[hsl].unsqueeze(1).to(probs.dtype)
                        ).sum(dim=(-1, -2))

        # ---------------- reduce ----------------
        denom = float(num_rows)
        sink_mass = sums["sink"] / denom
        sink1_mass = sums["sink1"] / denom
        local_mass = sums["local"] / denom
        band_mass = sums["band"] / denom
        blocked_mass = sums["blocked"] / denom
        other_mass = (1.0 - sink_mass - local_mass).clamp_min(0.0)

        sink_area = float(sink_len) / max(float(kv_len), 1.0)
        local_area = float(local_area_sum) / denom
        band_area = float(kv_len - band_start) / max(float(kv_len), 1.0)
        sink_enrichment = sink_mass / max(sink_area, 1e-12)

        static_cov = sums["static"] / denom
        static_density = float(static_density_sum) / denom
        static_block_cov = sums["static_block"] / denom
        static_block_density = float(static_block_density_sum) / denom

        # ---------------- oracle: column top-k per sampled q-group ----------------
        # Rows of a group share the selection, so the mean-over-rows coverage
        # equals sum_g (rows_g / R) * topk_sum(group mean mass) — no second pass.
        rows_per_block = torch.bincount(row_to_sample_block, minlength=num_sample_blocks).float()
        group_mass = group_mass / rows_per_block.clamp_min(1.0).view(1, -1, 1)
        block_weight = (rows_per_block / denom).view(1, -1)
        oracle: Dict[str, Tensor] = {}
        for ratio in cfg["budgets"]:
            kk = min(max(1, int(math.ceil(float(ratio) * kv_len))), kv_len)
            topk_vals = group_mass.topk(kk, dim=-1).values                     # [Hs, B, kk]
            oracle[f"{float(ratio):.4g}"] = (topk_vals.sum(dim=-1) * block_weight).sum(dim=-1)
        oracle_row = {key: (val / denom) for key, val in row_oracle_sums.items()}
        transfer_row_cov = None
        if row_transfer_sums is not None:
            transfer_row_cov = {key: (val / denom) for key, val in row_transfer_sums.items()}

        # ---------------- CS4A transfer coverage (footprint + center_union) ----------------
        transfer: Dict[str, Optional[Tensor]] = {
            "footprint_cov": None, "footprint_density": None,
            "center_cov": None, "center_density": None,
        }
        if transfer_masks is not None:
            transfer["footprint_cov"] = sums["fp_cov"] / denom
            transfer["footprint_density"] = sums["fp_den"] / (denom * float(kv_len))
            transfer["center_cov"] = sums["ct_cov"] / denom
            transfer["center_density"] = sums["ct_den"] / (denom * float(kv_len))

        # ---------------- write ----------------
        def _opt(x: Optional[Tensor], pos: int):
            return None if x is None else float(x[pos].detach().cpu())

        head_records = []
        for pos, head in enumerate(heads):
            head_records.append({
                "head": int(head),
                "sink_mass": float(sink_mass[pos]),
                "sink1_mass": float(sink1_mass[pos]),
                "local_mass": float(local_mass[pos]),
                "band_mass": float(band_mass[pos]),
                "other_mass": float(other_mass[pos]),
                "blocked_scale_mass": float(blocked_mass[pos]),
                "sink_enrichment": float(sink_enrichment[pos]),
                "static_coverage": float(static_cov[pos]),
                "static_block_coverage": float(static_block_cov[pos]),
                "transfer_coverage": _opt(transfer["footprint_cov"], pos),
                "transfer_density": _opt(transfer["footprint_density"], pos),
                "transfer_center_coverage": _opt(transfer["center_cov"], pos),
                "transfer_center_density": _opt(transfer["center_density"], pos),
                "oracle_coverage": {key: float(val[pos]) for key, val in oracle.items()},
                "oracle_row": {key: float(val[pos]) for key, val in oracle_row.items()},
                "transfer_row": None if transfer_row_cov is None else {
                    key: {
                        "coverage": float(transfer_row_cov[key][pos]),
                        "density": float(row_transfer_density[key][pos]),
                    } for key in transfer_row_cov
                },
            })

        record = {
            "prompt_id": current_pattern_prompt_id(),
            "scale": int(scale_ind),
            "layer": int(layer_ind),
            "q_len": q_len,
            "kv_len": kv_len,
            "sink_len": int(sink_len),
            "sink_scales": sink_scales,
            "band_s": int(cfg["band_s"]),
            "win_size": [int(x) for x in win_t.detach().cpu().tolist()],
            "block_size": block_size,
            "sample_q_blocks": [int(x) for x in sampled_q_blocks.detach().cpu().tolist()],
            "sample_rows": num_rows,
            "full_rows": bool(cfg["full_rows"]),
            "area_fraction": {
                "sink": sink_area,
                "local": local_area,
                "band": band_area,
                "static": static_density,
                "static_block": static_block_density,
            },
            "transfer_status": transfer_status,
            "transfer_row_ratios": [float(r) for r in cfg["transfer_row_ratios"]],
            "transfer_config": {
                "topk": float(cfg["transfer_topk"]),
                "bm": int(cfg["transfer_bm"]),
                "union_local": bool(cfg["transfer_union_local"]),
                "sink_scales": sink_scales,
            },
            "heads": head_records,
            "mean": {
                "sink_mass": float(sink_mass.mean()),
                "local_mass": float(local_mass.mean()),
                "band_mass": float(band_mass.mean()),
                "other_mass": float(other_mass.mean()),
                "blocked_scale_mass": float(blocked_mass.mean()),
                "static_coverage": float(static_cov.mean()),
                "static_block_coverage": float(static_block_cov.mean()),
                "transfer_coverage": None if transfer["footprint_cov"] is None else float(transfer["footprint_cov"].mean()),
                "transfer_center_coverage": None if transfer["center_cov"] is None else float(transfer["center_cov"].mean()),
                "oracle_coverage": {key: float(val.mean()) for key, val in oracle.items()},
                "oracle_row": {key: float(val.mean()) for key, val in oracle_row.items()},
                "transfer_row_coverage": None if transfer_row_cov is None else {
                    key: float(val.mean()) for key, val in transfer_row_cov.items()},
            },
        }
        self._write_record(record, static_density, static_block_density)
        if cfg["verbose"]:
            print(f"[img-pattern-stats] wrote scale={scale_ind} layer={layer_ind} "
                  f"rows={num_rows} heads={len(heads)} transfer={transfer_status}")

    def _write_record(self, record: dict, static_density: float, static_block_density: float) -> None:
        with self.jsonl_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        budget_keys = [f"oracle_{float(r):.4g}" for r in self.cfg["budgets"]]
        row_budget_keys = [f"oracle_row_{float(r):.4g}" for r in self.cfg["budgets"]]
        ratio_strs = [f"{float(r):.4g}" for r in self.cfg["transfer_row_ratios"]]
        row_transfer_keys = []
        for rs in ratio_strs:
            row_transfer_keys += [f"transfer_row_density_{rs}", f"transfer_row_coverage_{rs}"]
        fieldnames = [
            "prompt_id", "scale", "layer", "head", "q_len", "kv_len", "sink_len",
            "sink_mass", "sink1_mass", "local_mass", "band_mass", "other_mass",
            "blocked_scale_mass", "sink_enrichment", "local_area", "band_area",
            "static_density", "static_coverage",
            "static_block_density", "static_block_coverage",
            "transfer_density", "transfer_coverage",
            "transfer_center_density", "transfer_center_coverage",
            "transfer_status",
        ] + budget_keys + row_budget_keys + row_transfer_keys
        rows = []
        for head_rec in record["heads"]:
            row = {
                "prompt_id": record.get("prompt_id", current_pattern_prompt_id()),
                "scale": record["scale"],
                "layer": record["layer"],
                "head": head_rec["head"],
                "q_len": record["q_len"],
                "kv_len": record["kv_len"],
                "sink_len": record["sink_len"],
                "sink_mass": head_rec["sink_mass"],
                "sink1_mass": head_rec["sink1_mass"],
                "local_mass": head_rec["local_mass"],
                "band_mass": head_rec["band_mass"],
                "other_mass": head_rec["other_mass"],
                "blocked_scale_mass": head_rec["blocked_scale_mass"],
                "sink_enrichment": head_rec["sink_enrichment"],
                "local_area": record["area_fraction"]["local"],
                "band_area": record["area_fraction"]["band"],
                "static_density": static_density,
                "static_coverage": head_rec["static_coverage"],
                "static_block_density": static_block_density,
                "static_block_coverage": head_rec["static_block_coverage"],
                "transfer_density": head_rec["transfer_density"],
                "transfer_coverage": head_rec["transfer_coverage"],
                "transfer_center_density": head_rec["transfer_center_density"],
                "transfer_center_coverage": head_rec["transfer_center_coverage"],
                "transfer_status": record["transfer_status"],
            }
            for key, value in head_rec["oracle_coverage"].items():
                row[f"oracle_{key}"] = value
            for key, value in head_rec.get("oracle_row", {}).items():
                row[f"oracle_row_{key}"] = value
            transfer_row = head_rec.get("transfer_row")
            for rs in ratio_strs:
                ent = None if transfer_row is None else transfer_row.get(rs)
                row[f"transfer_row_density_{rs}"] = None if ent is None else ent["density"]
                row[f"transfer_row_coverage_{rs}"] = None if ent is None else ent["coverage"]
            rows.append(row)

        with self.csv_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            if not self._csv_header_written:
                writer.writeheader()
                self._csv_header_written = True
            writer.writerows(rows)


_COLLECTOR: Optional[_InfinityPatternStats] = None


def _get_collector() -> Optional[_InfinityPatternStats]:
    global _COLLECTOR
    cfg = _img_stats_config()
    if not cfg["enabled"]:
        return None
    if _COLLECTOR is None:
        _COLLECTOR = _InfinityPatternStats(cfg)
        if cfg["verbose"]:
            print(f"[img-pattern-stats] enabled, writing to {_COLLECTOR.out_dir}")
    return _COLLECTOR


def maybe_record_infinity_attention_stats(
    q: Tensor,
    k: Tensor,
    *,
    scale,
    scale_ind: int,
    layer_ind: int,
    scale_schedule,
    is_bhlc: bool = True,
) -> None:
    """Hook entry (see basic_infinity.SelfAttention.forward). q/k: post-norm, post-RoPE,
    post-kv-cache-concat; BHLc layout expected (is_bhlc=False -> warn once, skip).

    The decision scale is processed even when it is NOT in SPARVAR_PATTERN_SCALES:
    its colsum is cached (without writing a stats record) so the transfer simulation
    at later scales always has its input."""
    cfg = _img_stats_config()
    if not cfg["enabled"]:
        return
    if not isinstance(scale_ind, int) or not isinstance(layer_ind, int):
        return
    if not _selected(cfg["layers"], int(layer_ind)):
        return
    in_scales = (not cfg["scales"]) or int(scale_ind) in cfg["scales"]
    is_decision = int(scale_ind) == int(cfg["decision_scale"])
    if not (in_scales or is_decision):
        return
    collector = _get_collector()
    if collector is None:
        return
    if not in_scales:
        # decision scale outside the recorded set: cache colsum only
        if not is_bhlc or q.shape[0] == 0 or q.shape[1] == 0:
            return
        heads = _selected_heads(int(q.shape[1]), cfg)
        if heads:
            collector._cache_decision_colsum(q[:1], k[:1], scale, int(layer_ind), heads)
        return
    collector.record(
        q, k, scale=scale, scale_ind=int(scale_ind), layer_ind=int(layer_ind),
        scale_schedule=scale_schedule, is_bhlc=bool(is_bhlc),
    )
