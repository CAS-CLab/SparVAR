"""Env-gated InfinityStar attention pattern statistics.

This module is an analysis-only helper for the SparVAR/InfinityStar path.  It
samples query rows from selected video scales (or, with SPARVAR_PATTERN_FULL=1,
processes EVERY query row exactly — the streaming row-chunk loop keeps memory
flat either way), computes exact attention probabilities over the full KV cache
for those rows, and writes per-layer / per-head summaries for:

  * sink, local-3D, frame-band, and residual attention mass;
  * oracle block coverage under fixed KV-block budgets;
  * static sink+frame-band block coverage;
  * transferred D-CSLA block coverage using the same decision-scale colsum mass
    transfer and block selector as the runtime D-CSLA path.

Default behavior is off. Enable with SPARVAR_PATTERN_STATS=1.
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

from models.sparvar.dcsla_utils import compute_decision_block_mass, select_kv_blocks
from utils.misc import time_str

_TRUE = {"1", "true", "yes", "on"}


def _cdiv(a: int, b: int) -> int:
    return (int(a) + int(b) - 1) // int(b)


def _parse_int_selection(raw: str) -> Optional[set]:
    raw = (raw or "").strip().lower()
    if raw in ("", "all", "*"):
        return None
    return {int(x) for x in raw.split(",") if x.strip() != ""}


def _parse_int_set(name: str, default: str) -> set:
    raw = os.environ.get(name, default)
    return {int(x) for x in raw.split(",") if x.strip() != ""}


def _parse_float_list(name: str, default: str) -> List[float]:
    raw = os.environ.get(name, default)
    return [float(x) for x in raw.split(",") if x.strip() != ""]


def _selected(selection: Optional[set], value: int) -> bool:
    return selection is None or int(value) in selection


# Multi-prompt tagging: runners call set_pattern_prompt_id() between prompts; the
# current id is stamped on every jsonl/CSV record (video AND image collectors —
# attn_pattern_stats_image_utils imports current_pattern_prompt_id from here).
_PROMPT_ID: Optional[str] = None


def set_pattern_prompt_id(prompt_id) -> None:
    global _PROMPT_ID
    _PROMPT_ID = None if prompt_id is None else str(prompt_id)


def current_pattern_prompt_id() -> str:
    if _PROMPT_ID is not None:
        return _PROMPT_ID
    return os.environ.get("SPARVAR_PATTERN_PROMPT_ID", "0")


@functools.lru_cache(maxsize=1)
def _stats_config() -> dict:
    layers_raw = os.environ.get("SPARVAR_PATTERN_LAYERS", "0,17,35")
    heads_raw = os.environ.get("SPARVAR_PATTERN_HEADS", "all")
    return {
        "enabled": os.environ.get("SPARVAR_PATTERN_STATS", "0").lower() in _TRUE,
        "out_dir": os.environ.get(
            "SPARVAR_PATTERN_OUT",
            f"work_dir/play_models/SparVAR/InfinityStar_720p/pattern_stats_{time_str()}",
        ),
        "scales": _parse_int_set("SPARVAR_PATTERN_SCALES", "27,28,29"),
        "layers": _parse_int_selection(layers_raw),
        "heads": _parse_int_selection(heads_raw),
        # full-exact mode: every q row (qblocks / rows_per_qblock ignored); raise
        # row_chunk (e.g. 256) with it to amortize kernel-launch overhead
        "full_rows": os.environ.get("SPARVAR_PATTERN_FULL", "0").lower() in _TRUE,
        "qblocks": max(1, int(os.environ.get("SPARVAR_PATTERN_QBLOCKS", "32"))),
        "rows_per_qblock": max(1, int(os.environ.get("SPARVAR_PATTERN_ROWS_PER_QBLOCK", "2"))),
        "row_chunk": max(1, int(os.environ.get("SPARVAR_PATTERN_ROW_CHUNK", "64"))),
        "head_chunk": max(1, int(os.environ.get("SPARVAR_PATTERN_HEAD_CHUNK", "4"))),
        "block_size": max(1, int(os.environ.get("SPARVAR_PATTERN_BLOCK_SIZE", "128"))),
        "local_t": max(0, int(os.environ.get("SPARVAR_PATTERN_LOCAL_T", "1"))),
        "local_h": max(0, int(os.environ.get("SPARVAR_PATTERN_LOCAL_H", "4"))),
        "local_w": max(0, int(os.environ.get("SPARVAR_PATTERN_LOCAL_W", "4"))),
        "band_t": max(0, int(os.environ.get("SPARVAR_PATTERN_BAND_T", "1"))),
        "budgets": _parse_float_list("SPARVAR_PATTERN_BUDGETS", "0.1,0.2,0.3"),
        "transfer_mode": os.environ.get("SPARVAR_PATTERN_TRANSFER_MODE", "topk_ratio").strip().lower(),
        "transfer_topk_ratio": float(os.environ.get("SPARVAR_PATTERN_TRANSFER_TOPK_RATIO", "-1")),
        "transfer_tau": float(os.environ.get("SPARVAR_PATTERN_TRANSFER_TAU", "0.9")),
        "transfer_min_blocks": int(os.environ.get("SPARVAR_PATTERN_TRANSFER_MIN_BLOCKS", "1")),
        "transfer_max_blocks_ratio": float(os.environ.get("SPARVAR_PATTERN_TRANSFER_MAX_BLOCKS_RATIO", "1.0")),
        "transfer_local_diag_blocks": int(os.environ.get("SPARVAR_PATTERN_TRANSFER_LOCAL_DIAG_BLOCKS", "0")),
        "transfer_keep_sink": os.environ.get("SPARVAR_PATTERN_TRANSFER_KEEP_SINK", "1").lower() in _TRUE,
        "transfer_sink_select_tau": float(os.environ.get("SPARVAR_PATTERN_TRANSFER_SINK_TAU", "0.0")),
        # decision-side top-rho ratios for the token-wise geometric row transfer
        # (analysis-only; independent of the kernel-granular transfer above)
        "transfer_row_ratios": _parse_float_list("SPARVAR_PATTERN_TRANSFER_ROW_RATIOS", "0.2,0.15"),
        "mass_normalize": os.environ.get("SPARVAR_PATTERN_MASS_NORMALIZE", "1").lower() in _TRUE,
        "verbose": os.environ.get("SPARVAR_PATTERN_VERBOSE", "1").lower() in _TRUE,
    }


def pattern_stats_enabled_for(scale_ind, layer_ind) -> bool:
    cfg = _stats_config()
    if not cfg["enabled"]:
        return False
    if not isinstance(scale_ind, int) or not isinstance(layer_ind, int):
        return False
    if cfg["scales"] and int(scale_ind) not in cfg["scales"]:
        return False
    return _selected(cfg["layers"], int(layer_ind))


def pattern_stats_enabled_for_layer(layer_ind) -> bool:
    cfg = _stats_config()
    if not cfg["enabled"]:
        return False
    if not isinstance(layer_ind, int):
        return False
    return _selected(cfg["layers"], int(layer_ind))


def _selected_heads(num_heads: int, cfg: dict) -> List[int]:
    selection = cfg["heads"]
    if selection is None:
        return list(range(int(num_heads)))
    return [h for h in sorted(selection) if 0 <= h < int(num_heads)]


def _sample_q_blocks(q_len: int, block_size: int, target_blocks: int) -> List[int]:
    num_blocks = _cdiv(q_len, block_size)
    if target_blocks >= num_blocks:
        return list(range(num_blocks))
    if target_blocks == 1:
        return [num_blocks // 2]
    # Even coverage over the q-block axis. Use round, then de-duplicate in order.
    raw = torch.linspace(0, num_blocks - 1, steps=target_blocks).round().to(torch.int64).tolist()
    out = []
    seen = set()
    for item in raw:
        item = int(item)
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def _sample_rows_for_blocks(
    q_block_ids: Sequence[int],
    q_len: int,
    block_size: int,
    rows_per_qblock: int,
    device: torch.device,
) -> Tuple[Tensor, Tensor, Tensor]:
    rows: List[int] = []
    row_to_sample_block: List[int] = []
    kept_blocks: List[int] = []
    if rows_per_qblock == 1:
        offsets = [block_size // 2]
    else:
        offsets = torch.linspace(0, block_size - 1, steps=rows_per_qblock).round().to(torch.int64).tolist()
    for block_id in q_block_ids:
        start = int(block_id) * int(block_size)
        if start >= int(q_len):
            continue
        local_rows = []
        for off in offsets:
            row = min(start + int(off), int(q_len) - 1)
            local_rows.append(row)
        local_rows = sorted(set(local_rows))
        if not local_rows:
            continue
        sample_idx = len(kept_blocks)
        kept_blocks.append(int(block_id))
        rows.extend(local_rows)
        row_to_sample_block.extend([sample_idx] * len(local_rows))
    return (
        torch.tensor(rows, device=device, dtype=torch.int64),
        torch.tensor(row_to_sample_block, device=device, dtype=torch.int64),
        torch.tensor(kept_blocks, device=device, dtype=torch.int64),
    )


def _video_coords(indices: Tensor, shape: Tuple[int, int, int]) -> Tuple[Tensor, Tensor, Tensor]:
    t, h, w = (int(x) for x in shape)
    hw = h * w
    ti = torch.div(indices, hw, rounding_mode="trunc")
    rem = indices - ti * hw
    hi = torch.div(rem, w, rounding_mode="trunc")
    wi = rem - hi * w
    return ti, hi, wi


def _row_region_masks(
    q_rows: Tensor,
    *,
    kv_len: int,
    sink_len: int,
    scale_shape: Tuple[int, int, int],
    local_t: int,
    local_h: int,
    local_w: int,
    band_t: int,
) -> Tuple[Tensor, Tensor, Tensor]:
    device = q_rows.device
    t, h, w = (int(x) for x in scale_shape)
    video_len = t * h * w
    kv = torch.arange(int(kv_len), device=device, dtype=torch.int64)
    sink = kv < int(sink_len)
    video = (kv >= int(sink_len)) & (kv < int(sink_len) + video_len)

    kv_video = (kv - int(sink_len)).clamp_(0, max(video_len - 1, 0))
    kt, kh, kw = _video_coords(kv_video, scale_shape)
    qr = q_rows.clamp(0, max(video_len - 1, 0))
    qt, qh, qw = _video_coords(qr, scale_shape)

    local = (
        video.view(1, -1)
        & ((kt.view(1, -1) - qt.view(-1, 1)).abs() <= int(local_t))
        & ((kh.view(1, -1) - qh.view(-1, 1)).abs() <= int(local_h))
        & ((kw.view(1, -1) - qw.view(-1, 1)).abs() <= int(local_w))
    )
    band = (
        video.view(1, -1)
        & ((kt.view(1, -1) - qt.view(-1, 1)).abs() <= int(band_t))
    )
    return sink, local, band


def _static_block_mask(
    q_block_ids: Tensor,
    *,
    kv_len: int,
    sink_len: int,
    q_len: int,
    scale_shape: Tuple[int, int, int],
    block_size: int,
    band_t: int,
) -> Tensor:
    device = q_block_ids.device
    V = _cdiv(kv_len, block_size)
    mask = torch.zeros((int(q_block_ids.numel()), V), device=device, dtype=torch.bool)
    sink_blocks = min(_cdiv(max(int(sink_len), 0), block_size), V)
    if sink_blocks > 0:
        mask[:, :sink_blocks] = True

    center_rows = (q_block_ids * int(block_size) + int(block_size) // 2).clamp_max(max(int(q_len) - 1, 0))
    _, _, band = _row_region_masks(
        center_rows,
        kv_len=kv_len,
        sink_len=sink_len,
        scale_shape=scale_shape,
        local_t=0,
        local_h=0,
        local_w=0,
        band_t=band_t,
    )
    kv_block = torch.arange(int(kv_len), device=device, dtype=torch.int64) // int(block_size)
    for i in range(int(q_block_ids.numel())):
        cols = kv_block[band[i]]
        if cols.numel() > 0:
            mask[i, cols.unique()] = True
    return mask


def _coverage_from_mask(true_block_mass: Tensor, block_mask: Tensor) -> Tuple[Tensor, Tensor]:
    total = true_block_mass.sum(dim=-1).clamp_min(1e-12)
    coverage = (true_block_mass * block_mask.to(true_block_mass.dtype).unsqueeze(0)).sum(dim=-1) / total
    density = block_mask.float().mean(dim=-1).view(1, -1).expand(true_block_mass.shape[0], -1)
    return coverage.mean(dim=-1), density.mean(dim=-1)


def _oracle_coverage(true_block_mass: Tensor, budgets: Sequence[float]) -> Dict[str, Tensor]:
    V = int(true_block_mass.shape[-1])
    total = true_block_mass.sum(dim=-1).clamp_min(1e-12)
    sorted_mass = true_block_mass.sort(dim=-1, descending=True, stable=True).values
    out: Dict[str, Tensor] = {}
    for ratio in budgets:
        k = min(max(1, int(math.ceil(float(ratio) * V))), V)
        cov = sorted_mass[..., :k].sum(dim=-1) / total
        out[f"{float(ratio):.4g}"] = cov.mean(dim=-1)
    return out


def _to_float_list(x: Tensor) -> List[float]:
    return [float(v) for v in x.detach().cpu().tolist()]


def _video_row_parent_map(
    kv_len: int,
    sink_len: int,
    scale_shape: Tuple[int, int, int],
    dec_scale_shape: Tuple[int, int, int],
    dec_sink_len: int,
    device,
) -> Optional[Tensor]:
    """Token-level geometric correspondence target KV -> decision KV.

    Prefix tokens map by identity (the context prefix is shared verbatim across
    scales of a clip); in-scale tokens map by center-aligned floor rescaling of
    their (t, y, x) coordinates. Pure pyramid geometry, no kernel/runtime code.
    Returns None when the geometries are incompatible.
    """
    t_t, h_t, w_t = (int(x) for x in scale_shape)
    t_s, h_s, w_s = (int(x) for x in dec_scale_shape)
    if int(sink_len) != int(dec_sink_len) or t_t != t_s:
        return None
    j = torch.arange(int(kv_len), device=device, dtype=torch.int64)
    p = (j - int(sink_len)).clamp_min(0)
    t = torch.div(p, h_t * w_t, rounding_mode="trunc").clamp_(0, max(t_s - 1, 0))
    rem = p - t * (h_t * w_t)
    y = torch.div(rem, w_t, rounding_mode="trunc")
    x = rem - y * w_t
    ys = ((y.float() + 0.5) * (float(h_s) / float(h_t))).floor().to(torch.int64).clamp_(0, h_s - 1)
    xs = ((x.float() + 0.5) * (float(w_s) / float(w_t))).floor().to(torch.int64).clamp_(0, w_s - 1)
    in_scale = int(dec_sink_len) + t * (h_s * w_s) + ys * w_s + xs
    return torch.where(j < int(sink_len), j, in_scale)


class _InfinityStarPatternStats:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.out_dir = Path(cfg["out_dir"])
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.jsonl_path = self.out_dir / "stats.jsonl"
        self.csv_path = self.out_dir / "stats_heads.csv"
        self._csv_header_written = self.csv_path.exists() and self.csv_path.stat().st_size > 0
        self._decision_mass: Dict[int, dict] = {}
        # layer -> token-level decision cache for the geometric row transfer
        # (built inside record_attention at the decision scale; independent of
        # the runtime colsum hook that feeds _decision_mass)
        self._decision_row_mass: Dict[int, dict] = {}
        self._warned = set()

    def _warn_once(self, msg: str) -> None:
        if msg not in self._warned:
            self._warned.add(msg)
            print(msg)

    @torch.no_grad()
    def record_decision_colsum(
        self,
        bs: Tensor,
        *,
        scale_schedule,
        decision_scale_ind: int,
        layer_ind: int,
        decision_sink_len: int,
        decision_scale_shape: Tuple[int, int, int],
        decision_k_len: int,
    ) -> None:
        if not _selected(self.cfg["layers"], int(layer_ind)):
            return
        if decision_scale_shape is None or scale_schedule is None:
            self._warn_once("[pattern-stats] skip decision transfer cache: missing scale shape/schedule")
            return
        mass_dict, k_len_dict = compute_decision_block_mass(
            bs[:1],
            scale_schedule=scale_schedule,
            decision_scale_ind=int(decision_scale_ind),
            decision_sink_len=int(decision_sink_len),
            decision_scale_shape=decision_scale_shape,
            decision_k_len=int(decision_k_len),
            block_size=int(self.cfg["block_size"]),
            normalize=bool(self.cfg["mass_normalize"]),
        )
        self._decision_mass[int(layer_ind)] = {
            "decision_scale_ind": int(decision_scale_ind),
            "decision_sink_len": int(decision_sink_len),
            "decision_k_len": int(decision_k_len),
            "decision_scale_shape": tuple(int(x) for x in decision_scale_shape),
            "mass": mass_dict,
            "expected_k_len": k_len_dict,
        }
        if self.cfg["verbose"]:
            print(
                f"[pattern-stats] cached transfer mass: scale={decision_scale_ind} "
                f"layer={layer_ind} targets={sorted(mass_dict.keys())}"
            )

    @torch.no_grad()
    def record_attention(
        self,
        q: Tensor,
        k: Tensor,
        *,
        scale,
        scale_ind: int,
        layer_ind: int,
        scale_shape: Tuple[int, int, int],
        sink_len: int,
        do_full_step: bool,
        attn_config: dict,
    ) -> None:
        if scale_shape is None:
            self._warn_once(f"[pattern-stats] skip scale={scale_ind} layer={layer_ind}: missing 3D scale shape")
            return
        if q.shape[0] == 0 or q.shape[1] == 0:
            return

        cfg = self.cfg
        q0 = q[:1]
        k0 = k[:1]
        device = q0.device
        num_heads = int(q0.shape[1])
        q_len = int(q0.shape[-2])
        kv_len = int(k0.shape[-2])
        block_size = int(cfg["block_size"])
        V = _cdiv(kv_len, block_size)
        heads = _selected_heads(num_heads, cfg)
        if not heads:
            return

        if cfg["full_rows"]:
            # exact mode: every q row; the streaming row_chunk loop below keeps
            # memory flat, and all accumulators are block-level (V blocks)
            q_rows = torch.arange(q_len, device=device, dtype=torch.int64)
            row_to_sample_block = q_rows // block_size
            sampled_q_blocks = torch.arange(_cdiv(q_len, block_size), device=device, dtype=torch.int64)
        else:
            q_block_ids = _sample_q_blocks(q_len, block_size, int(cfg["qblocks"]))
            q_rows, row_to_sample_block, sampled_q_blocks = _sample_rows_for_blocks(
                q_block_ids, q_len, block_size, int(cfg["rows_per_qblock"]), device)
        if q_rows.numel() == 0:
            return
        num_sample_blocks = int(sampled_q_blocks.numel())
        num_rows = int(q_rows.numel())

        # ---- token-wise metrics setup (per-row entry oracle + geometric row transfer) ----
        num_sel_heads = len(heads)
        row_budget_ks = [
            (f"{float(r):.4g}", min(max(1, int(math.ceil(float(r) * kv_len))), kv_len))
            for r in cfg["budgets"]
        ]
        row_k_max = max((kk for _, kk in row_budget_ks), default=0)
        row_oracle_sums = {key: torch.zeros(num_sel_heads, device=device, dtype=torch.float32)
                           for key, _ in row_budget_ks}
        decision_scale_ind = int(attn_config.get("decision_scale", -1)) if isinstance(attn_config, dict) else -1
        decision_col_acc = None
        row_transfer_masks = None
        row_transfer_density = None
        if int(scale_ind) == decision_scale_ind:
            decision_col_acc = torch.zeros(num_sel_heads, kv_len, device=device, dtype=torch.float32)
            transfer_row_status = "decision_scale"
        else:
            row_entry = self._decision_row_mass.get(int(layer_ind))
            if row_entry is None:
                transfer_row_status = "missing_decision"
            elif int(row_entry["col_mass"].shape[0]) != num_sel_heads:
                transfer_row_status = "heads_mismatch"
            else:
                parent = _video_row_parent_map(
                    kv_len, int(sink_len), scale_shape,
                    row_entry["scale_shape"], row_entry["sink_len"], device)
                if parent is None:
                    transfer_row_status = "geom_mismatch"
                else:
                    k_dec = int(row_entry["k_len"])
                    row_transfer_masks = {}
                    row_transfer_density = {}
                    for r in cfg["transfer_row_ratios"]:
                        kk = min(max(1, int(math.ceil(float(r) * k_dec))), k_dec)
                        sel = torch.zeros(num_sel_heads, k_dec, device=device, dtype=torch.bool)
                        sel.scatter_(1, row_entry["col_mass"].topk(kk, dim=-1).indices, True)
                        key = f"{float(r):.4g}"
                        row_transfer_masks[key] = sel.index_select(1, parent)  # [Hs, K_target]
                        row_transfer_density[key] = row_transfer_masks[key].float().mean(dim=-1)
                    transfer_row_status = "ok"
        row_transfer_sums = None
        if row_transfer_masks is not None:
            row_transfer_sums = {key: torch.zeros(num_sel_heads, device=device, dtype=torch.float32)
                                 for key in row_transfer_masks}

        true_block_mass = torch.zeros((len(heads), num_sample_blocks, V), device=device, dtype=torch.float32)
        component_sums = {
            "sink": torch.zeros(len(heads), device=device, dtype=torch.float32),
            "local": torch.zeros(len(heads), device=device, dtype=torch.float32),
            "band": torch.zeros(len(heads), device=device, dtype=torch.float32),
        }
        local_area_sum = torch.tensor(0.0, device=device)
        band_area_sum = torch.tensor(0.0, device=device)
        row_count_per_block = torch.zeros(num_sample_blocks, device=device, dtype=torch.float32)
        row_count_per_block.index_add_(0, row_to_sample_block, torch.ones_like(row_to_sample_block, dtype=torch.float32))

        kv_block_ids = torch.arange(kv_len, device=device, dtype=torch.int64) // block_size
        key_t = k0.transpose(-2, -1)

        head_chunk = int(cfg["head_chunk"])
        row_chunk = int(cfg["row_chunk"])
        for row_start in range(0, num_rows, row_chunk):
            row_end = min(row_start + row_chunk, num_rows)
            rows_chunk = q_rows[row_start:row_end]
            row_to_block_chunk = row_to_sample_block[row_start:row_end]
            sink_mask, local_mask, band_mask = _row_region_masks(
                rows_chunk,
                kv_len=kv_len,
                sink_len=int(sink_len),
                scale_shape=scale_shape,
                local_t=int(cfg["local_t"]),
                local_h=int(cfg["local_h"]),
                local_w=int(cfg["local_w"]),
                band_t=int(cfg["band_t"]),
            )
            local_area_sum += local_mask.float().mean(dim=-1).sum()
            band_area_sum += band_mask.float().mean(dim=-1).sum()

            for head_pos in range(0, len(heads), head_chunk):
                cur_heads = heads[head_pos:head_pos + head_chunk]
                head_tensor = torch.tensor(cur_heads, device=device, dtype=torch.int64)
                q_chunk = q0.index_select(1, head_tensor).index_select(2, rows_chunk)
                k_chunk_t = key_t.index_select(1, head_tensor)
                logits = torch.matmul(q_chunk, k_chunk_t).float()
                logits.mul_(float(scale))
                probs = torch.softmax(logits, dim=-1).squeeze(0)  # [Hc, R, K]

                h_slice = slice(head_pos, head_pos + len(cur_heads))
                component_sums["sink"][h_slice] += probs[:, :, sink_mask].sum(dim=(-1, -2))
                component_sums["local"][h_slice] += (
                    probs * local_mask.to(probs.dtype).unsqueeze(0)
                ).sum(dim=(-1, -2))
                component_sums["band"][h_slice] += (
                    probs * band_mask.to(probs.dtype).unsqueeze(0)
                ).sum(dim=(-1, -2))

                # token-wise per-row entry oracle: top-ceil(rho*K) entries of every row
                if row_k_max > 0:
                    top_vals = probs.topk(row_k_max, dim=-1).values           # sorted descending
                    csum = top_vals.cumsum(dim=-1)
                    for key, kk in row_budget_ks:
                        row_oracle_sums[key][h_slice] += csum[..., kk - 1].sum(dim=-1)
                if decision_col_acc is not None:
                    decision_col_acc[h_slice] += probs.sum(dim=1)
                if row_transfer_sums is not None:
                    for key, tm in row_transfer_masks.items():
                        row_transfer_sums[key][h_slice] += (
                            probs * tm[h_slice].unsqueeze(1).to(probs.dtype)
                        ).sum(dim=(-1, -2))

                block_mass = torch.zeros(
                    (len(cur_heads), int(rows_chunk.numel()), V),
                    device=device,
                    dtype=torch.float32,
                )
                scatter_index = kv_block_ids.view(1, 1, kv_len).expand_as(probs)
                block_mass.scatter_add_(-1, scatter_index, probs)
                true_block_mass[h_slice].index_add_(1, row_to_block_chunk, block_mass)

        true_block_mass = true_block_mass / row_count_per_block.clamp_min(1.0).view(1, -1, 1)
        denom_rows = float(num_rows)
        if decision_col_acc is not None:
            self._decision_row_mass[int(layer_ind)] = {
                "scale_ind": int(scale_ind),
                "sink_len": int(sink_len),
                "scale_shape": tuple(int(x) for x in scale_shape),
                "k_len": int(kv_len),
                "col_mass": decision_col_acc,
            }
        oracle_row = {key: (val / denom_rows) for key, val in row_oracle_sums.items()}
        transfer_row_cov = None
        if row_transfer_sums is not None:
            transfer_row_cov = {key: (val / denom_rows) for key, val in row_transfer_sums.items()}
        sink_mass = component_sums["sink"] / denom_rows
        local_mass = component_sums["local"] / denom_rows
        band_mass = component_sums["band"] / denom_rows
        other_mass = (1.0 - sink_mass - local_mass).clamp_min(0.0)
        sink_area = float(sink_len) / max(float(kv_len), 1.0)
        local_area = float(local_area_sum.detach().cpu()) / max(float(num_rows), 1.0)
        band_area = float(band_area_sum.detach().cpu()) / max(float(num_rows), 1.0)
        sink_enrichment = sink_mass / max(sink_area, 1e-12)

        oracle = _oracle_coverage(true_block_mass, cfg["budgets"])
        static_mask = _static_block_mask(
            sampled_q_blocks,
            kv_len=kv_len,
            sink_len=int(sink_len),
            q_len=q_len,
            scale_shape=scale_shape,
            block_size=block_size,
            band_t=int(cfg["band_t"]),
        )
        static_cov, static_density = _coverage_from_mask(true_block_mass, static_mask)
        transfer_cov, transfer_density, transfer_status = self._transfer_coverage(
            true_block_mass=true_block_mass,
            heads=heads,
            sampled_q_blocks=sampled_q_blocks,
            q_len=q_len,
            kv_len=kv_len,
            sink_len=int(sink_len),
            scale_ind=int(scale_ind),
            layer_ind=int(layer_ind),
            attn_config=attn_config,
            block_size=block_size,
        )

        head_records = []
        for pos, head in enumerate(heads):
            rec = {
                "head": int(head),
                "sink_mass": float(sink_mass[pos].detach().cpu()),
                "local_mass": float(local_mass[pos].detach().cpu()),
                "band_mass": float(band_mass[pos].detach().cpu()),
                "other_mass": float(other_mass[pos].detach().cpu()),
                "sink_enrichment": float(sink_enrichment[pos].detach().cpu()),
                "static_density": float(static_density[pos].detach().cpu()),
                "static_coverage": float(static_cov[pos].detach().cpu()),
                "transfer_density": None if transfer_density is None else float(transfer_density[pos].detach().cpu()),
                "transfer_coverage": None if transfer_cov is None else float(transfer_cov[pos].detach().cpu()),
                "oracle_coverage": {key: float(val[pos].detach().cpu()) for key, val in oracle.items()},
                "oracle_row": {key: float(val[pos].detach().cpu()) for key, val in oracle_row.items()},
                "transfer_row": None if transfer_row_cov is None else {
                    key: {
                        "coverage": float(transfer_row_cov[key][pos].detach().cpu()),
                        "density": float(row_transfer_density[key][pos].detach().cpu()),
                    } for key in transfer_row_cov
                },
            }
            head_records.append(rec)

        record = {
            "prompt_id": current_pattern_prompt_id(),
            "scale": int(scale_ind),
            "layer": int(layer_ind),
            "do_full_step": bool(do_full_step),
            "q_len": q_len,
            "kv_len": kv_len,
            "sink_len": int(sink_len),
            "scale_shape": [int(x) for x in scale_shape],
            "block_size": block_size,
            "sample_q_blocks": [int(x) for x in sampled_q_blocks.detach().cpu().tolist()],
            "sample_rows": int(num_rows),
            "full_rows": bool(cfg["full_rows"]),
            "local_radius": {"t": int(cfg["local_t"]), "h": int(cfg["local_h"]), "w": int(cfg["local_w"])},
            "band_t": int(cfg["band_t"]),
            "area_fraction": {
                "sink": sink_area,
                "local": local_area,
                "band": band_area,
            },
            "transfer_status": transfer_status,
            "transfer_row_status": transfer_row_status,
            "transfer_row_ratios": [float(r) for r in cfg["transfer_row_ratios"]],
            "heads": head_records,
            "mean": {
                "sink_mass": float(sink_mass.mean().detach().cpu()),
                "local_mass": float(local_mass.mean().detach().cpu()),
                "band_mass": float(band_mass.mean().detach().cpu()),
                "other_mass": float(other_mass.mean().detach().cpu()),
                "static_coverage": float(static_cov.mean().detach().cpu()),
                "static_density": float(static_density.mean().detach().cpu()),
                "transfer_coverage": None if transfer_cov is None else float(transfer_cov.mean().detach().cpu()),
                "transfer_density": None if transfer_density is None else float(transfer_density.mean().detach().cpu()),
                "oracle_coverage": {key: float(val.mean().detach().cpu()) for key, val in oracle.items()},
                "oracle_row": {key: float(val.mean().detach().cpu()) for key, val in oracle_row.items()},
                "transfer_row_coverage": None if transfer_row_cov is None else {
                    key: float(val.mean().detach().cpu()) for key, val in transfer_row_cov.items()},
            },
        }
        self._write_record(record)
        if cfg["verbose"]:
            print(
                f"[pattern-stats] wrote scale={scale_ind} layer={layer_ind} "
                f"rows={num_rows} heads={len(heads)} transfer={transfer_status}"
            )

    def _transfer_coverage(
        self,
        *,
        true_block_mass: Tensor,
        heads: Sequence[int],
        sampled_q_blocks: Tensor,
        q_len: int,
        kv_len: int,
        sink_len: int,
        scale_ind: int,
        layer_ind: int,
        attn_config: dict,
        block_size: int,
    ) -> Tuple[Optional[Tensor], Optional[Tensor], str]:
        entry = self._decision_mass.get(int(layer_ind))
        if entry is None:
            return None, None, "missing_decision_mass"
        if int(scale_ind) not in entry["mass"]:
            return None, None, "not_a_transfer_target"
        if int(entry["expected_k_len"].get(int(scale_ind), -1)) != int(kv_len):
            return None, None, "kv_len_mismatch"

        mass = entry["mass"][int(scale_ind)]
        V = _cdiv(kv_len, block_size)
        U = _cdiv(q_len, block_size)
        if int(mass.shape[-1]) != V:
            return None, None, "mass_width_mismatch"

        Gs = int(mass.shape[2])
        device = mass.device
        u_centers = torch.arange(U, device=device, dtype=torch.float32) + 0.5
        phi = torch.round(u_centers / U * Gs - 0.5).to(torch.int64).clamp_(0, Gs - 1)
        mass_rows = mass.index_select(2, phi).float()

        select_cfg = dict(attn_config)
        topk_ratio = float(self.cfg["transfer_topk_ratio"])
        if topk_ratio <= 0:
            topk_ratio = float(attn_config.get("top_keys", 0.2))
        select_cfg.update({
            "flex_block_size": int(block_size),
            "dcsla_select_mode": self.cfg["transfer_mode"],
            "dcsla_tau": float(self.cfg["transfer_tau"]),
            "dcsla_topk_ratio": topk_ratio,
            "dcsla_min_blocks": int(self.cfg["transfer_min_blocks"]),
            "dcsla_max_blocks_ratio": float(self.cfg["transfer_max_blocks_ratio"]),
            "dcsla_local_diag_blocks": int(self.cfg["transfer_local_diag_blocks"]),
            "dcsla_keep_sink": bool(self.cfg["transfer_keep_sink"]),
            "dcsla_sink_select_tau": float(self.cfg["transfer_sink_select_tau"]),
        })
        kv_indices, kv_num_blocks = select_kv_blocks(
            mass_rows,
            sink_len=int(sink_len),
            q_len=int(q_len),
            kv_len=int(kv_len),
            block_size=int(block_size),
            mode=str(select_cfg.get("dcsla_select_mode", "topk_ratio")),
            tau=float(select_cfg.get("dcsla_tau", 0.9)),
            topk_ratio=float(select_cfg.get("dcsla_topk_ratio", 0.2)),
            min_blocks=int(select_cfg.get("dcsla_min_blocks", 1)),
            max_blocks_ratio=float(select_cfg.get("dcsla_max_blocks_ratio", 1.0)),
            local_diag_blocks=int(select_cfg.get("dcsla_local_diag_blocks", 0)),
            keep_sink=bool(select_cfg.get("dcsla_keep_sink", True)),
            sink_select_tau=float(select_cfg.get("dcsla_sink_select_tau", 0.0)),
        )

        head_tensor = torch.tensor(list(heads), device=device, dtype=torch.int64)
        ids = kv_indices[:1].index_select(1, head_tensor).index_select(2, sampled_q_blocks)
        counts = kv_num_blocks[:1].index_select(1, head_tensor).index_select(2, sampled_q_blocks)
        ids = ids.squeeze(0)
        counts = counts.squeeze(0)
        valid = torch.arange(V, device=device).view(1, 1, V) < counts.unsqueeze(-1)
        mask = torch.zeros((len(heads), int(sampled_q_blocks.numel()), V), device=device, dtype=torch.bool)
        mask.scatter_(-1, ids.clamp(0, V - 1), valid)
        total = true_block_mass.sum(dim=-1).clamp_min(1e-12)
        coverage = (true_block_mass * mask.to(true_block_mass.dtype)).sum(dim=-1) / total
        density = counts.float() / max(float(V), 1.0)
        return coverage.mean(dim=-1), density.mean(dim=-1), "ok"

    def _write_record(self, record: dict) -> None:
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
            "sink_mass", "local_mass", "band_mass", "other_mass",
            "sink_enrichment", "local_area", "band_area",
            "static_density", "static_coverage",
            "transfer_density", "transfer_coverage", "transfer_status",
            "transfer_row_status",
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
                "local_mass": head_rec["local_mass"],
                "band_mass": head_rec["band_mass"],
                "other_mass": head_rec["other_mass"],
                "sink_enrichment": head_rec["sink_enrichment"],
                "local_area": record["area_fraction"]["local"],
                "band_area": record["area_fraction"]["band"],
                "static_density": head_rec["static_density"],
                "static_coverage": head_rec["static_coverage"],
                "transfer_density": head_rec["transfer_density"],
                "transfer_coverage": head_rec["transfer_coverage"],
                "transfer_status": record["transfer_status"],
                "transfer_row_status": record.get("transfer_row_status"),
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


_COLLECTOR: Optional[_InfinityStarPatternStats] = None


def _get_collector() -> Optional[_InfinityStarPatternStats]:
    global _COLLECTOR
    cfg = _stats_config()
    if not cfg["enabled"]:
        return None
    if _COLLECTOR is None:
        _COLLECTOR = _InfinityStarPatternStats(cfg)
        if cfg["verbose"]:
            print(f"[pattern-stats] enabled, writing to {_COLLECTOR.out_dir}")
    return _COLLECTOR


def maybe_record_infinitystar_decision_colsum(
    bs: Tensor,
    *,
    scale_schedule,
    decision_scale_ind: int,
    layer_ind: int,
    decision_sink_len: int,
    decision_scale_shape: Tuple[int, int, int],
    decision_k_len: int,
) -> None:
    collector = _get_collector()
    if collector is None:
        return
    if not _selected(_stats_config()["layers"], int(layer_ind)):
        return
    collector.record_decision_colsum(
        bs,
        scale_schedule=scale_schedule,
        decision_scale_ind=int(decision_scale_ind),
        layer_ind=int(layer_ind),
        decision_sink_len=int(decision_sink_len),
        decision_scale_shape=decision_scale_shape,
        decision_k_len=int(decision_k_len),
    )


def maybe_record_infinitystar_attention_stats(
    q: Tensor,
    k: Tensor,
    *,
    scale,
    scale_ind: int,
    layer_ind: int,
    scale_shape: Tuple[int, int, int],
    sink_len: int,
    do_full_step: bool,
    attn_config: dict,
) -> None:
    if not pattern_stats_enabled_for(scale_ind, layer_ind):
        return
    collector = _get_collector()
    if collector is None:
        return
    collector.record_attention(
        q,
        k,
        scale=scale,
        scale_ind=int(scale_ind),
        layer_ind=int(layer_ind),
        scale_shape=scale_shape,
        sink_len=int(sink_len),
        do_full_step=bool(do_full_step),
        attn_config=attn_config,
    )
