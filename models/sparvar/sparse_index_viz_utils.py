"""Fast visualization of CS4A sparse KV-index selection.

Renders the *selected* KV indices (decision-scale top-k, and the cross-scale
mapped indices at later sparse scales) as a compact ``[query_group, key_bin]``
density heatmap — one PNG per (scale, layer, head). Unlike the old
``save_attention_selection_visuals`` it never materializes the dense
``[q_len, kv_len]`` mask, so it stays fast and bounded even at 20k-70k sequence
lengths:

  * rows  = query GROUPS (``Gk = ceil(q_len / bm)``), the natural lossless
            resolution of the selection (a whole bm-row group shares one index
            set), so the q axis is only ~108-375 rows — no per-row expansion.
  * cols  = a fixed number of key BINS (default 1024); each cell counts how many
            selected keys of that group fall into that key range.

Only the first ``counts[g]`` entries per group are treated as valid selections
(the rest of ``inds`` is padding garbage), so bin-0 is not polluted.

Enable via environment variables (no pipeline signature changes):

  SPARVAR_VIZ_INDEX=1            master switch (off => zero overhead)
  SPARVAR_VIZ_OUT=<dir>          default work_dir/play_models/sparvar/index_viz
  SPARVAR_VIZ_SCALES=27,28,29    which scale_ind to dump
  SPARVAR_VIZ_LAYERS=0,17,35     which layer_ind to dump (representative layers)
  SPARVAR_VIZ_HEADS=all          'all' or comma list, e.g. 0,1,2
  SPARVAR_VIZ_KBINS=1024         (fast mode) key-axis bins (columns)
  SPARVAR_VIZ_MODE=fast          'fast' (group x binned-key heatmap) or
                                 'full' (complete q_len x kv_len binary mask)
  SPARVAR_VIZ_FULL_QCHUNK=512    (full mode) query-row chunk for bounded GPU memory
  SPARVAR_VIZ_COLOR=gray         (full mode) 'gray' (black/white) or 'viridis'
  SPARVAR_VIZ_FULL_MAXSIDE=4096  1) max long side of the DATA raster in cells: the
                                 mask is max-pooled until long_side/f <= MAXSIDE
                                 (0 => no side cap); 2) also sets the rendered
                                 figure width in px (clamped to [1200, 16384]).
  SPARVAR_VIZ_FULL_MAX_PIXELS=16777216
                                 max total cells (rows x cols) of the DATA raster;
                                 the pool factor grows until under this cap.
                                 Set 0 together with FULL_MAXSIDE=0 for native
                                 1 token = 1 cell resolution (pdf recommended).
  SPARVAR_VIZ_FULL_DPI=200       figure DPI. The mask area is ~MAXSIDE px wide
                                 regardless (fig inches = px / dpi), so in practice
                                 DPI controls how LARGE text/lines appear relative
                                 to the mask (fonts are in points): higher DPI =>
                                 bigger-looking labels, thicker-looking lines.
  SPARVAR_VIZ_FORMAT=png         'png' | 'pdf' | 'both' for the matplotlib renderers
                                 (FULL / BLOCKGRID / TOKENMASK). PDF keeps text,
                                 ticks and all boundary/grid lines as true vectors
                                 and embeds the mask raster UNRESAMPLED at the data
                                 resolution (losslessly compressed), so readers can
                                 zoom to cell/token level. For paper figures:
                                 SPARVAR_VIZ_FORMAT=pdf, and for token-native
                                 detail also FULL_MAXSIDE=0 FULL_MAX_PIXELS=0.
  SPARVAR_VIZ_PNG_COMPRESS_LEVEL=6
                                 (legacy) PNG compression level, 0-9. Unused since the
                                 full-mask renderer moved from PIL to matplotlib.
  SPARVAR_VIZ_BLOCK_GRID=1       (D-CSLA, dump_dcsla_block_mask_viz) render the native
                                 [U,V] block mask as a BLACK/WHITE matplotlib figure
                                 (crisp cells, block-aligned) with a red DASHED per-block
                                 grid, bold sink line, per-scale KV boundary lines with
                                 token-index x ticks (image), cyan dashed frame lines
                                 (video). Line/font styles are keyword args on the
                                 function itself. FULL_MAXSIDE = px width, FULL_DPI = DPI.
  SPARVAR_VIZ_BLOCK_GRID_EVERY=1 red grid every Nth 128-block (1 = every block; 0 = off).
  SPARVAR_VIZ_TOKEN_MASK=1       (CSLA only, dump_csla_token_mask_viz) render the
                                 TOKEN-wise fine mask (pre block-any) once per scale,
                                 notebook-style, with the same grid/sink/scale-boundary
                                 overlays. 0 disables just this renderer.
  SPARVAR_VIZ_MASS_SEL=1         (D-CSLA, dump_dcsla_mass_selection_viz) per sparse
                                 scale: phi-remapped block-mass heatmap + the FINAL
                                 selected blocks as outlines, color-coded by origin
                                 (mass prefix / forced by constraint / cut by cap).
                                 0 disables just this renderer.
  SPARVAR_VIZ_DECISION_COLSUM=1  (D-CSLA, dump_decision_colsum_viz) at the decision
                                 scale: raw token-wise colsum evidence heatmap with
                                 block grid + frame/scale-boundary overlays.
                                 0 disables just this renderer.
  SPARVAR_VIZ_BLOCK_FRAME_LINES=1
                                 (D-CSLA) cyan dotted frame-boundary lines on both
                                 axes (requires scale_shape=(T,H,W) at the call).

Two render modes:
  * fast (default): compact [query_group, key_bin] density heatmap — bounded and
    fast at any sequence length. Good for scanning all heads/layers.
  * full: the exact [q_len, kv_len] binary selection mask (per query ROW). Slow
    and memory-heavy (multi-gigapixel at the last video scale), so restrict it to
    a single layer + a few heads via SPARVAR_VIZ_LAYERS / SPARVAR_VIZ_HEADS.

Only the conditional branch (b=0) is rendered.
"""

import functools
import math
import os

import numpy as np
import torch

from utils.misc import time_str

_TRUE = {"1", "true", "yes", "on"}


@functools.lru_cache(maxsize=1)
def _viz_config():
    def _int_set(name, default):
        raw = os.environ.get(name, default)
        return {int(x) for x in raw.split(",") if x.strip() != ""}

    heads_raw = os.environ.get("SPARVAR_VIZ_HEADS", "all").strip().lower()
    heads = None if heads_raw in ("all", "") else {
        int(x) for x in heads_raw.split(",") if x.strip() != ""}
    mode = os.environ.get("SPARVAR_VIZ_MODE", "fast").strip().lower()
    if mode not in ("fast", "full"):
        mode = "fast"
    return {
        "enabled": os.environ.get("SPARVAR_VIZ_INDEX", "0").lower() in _TRUE,
        "out_dir": os.environ.get("SPARVAR_VIZ_OUT",
                                  f"work_dir/play_models/SparVAR/Infinity_2b/index_viz_{time_str()}"),
        "scales": _int_set("SPARVAR_VIZ_SCALES", "27,28,29"),
        "layers": _int_set("SPARVAR_VIZ_LAYERS", "0,17,35"),
        "heads": heads,
        "kbins": int(os.environ.get("SPARVAR_VIZ_KBINS", "1024")),
        "mode": mode,
        "full_qchunk": int(os.environ.get("SPARVAR_VIZ_FULL_QCHUNK", "512")),
        "color": "viridis" if os.environ.get("SPARVAR_VIZ_COLOR", "gray").strip().lower()
                 == "viridis" else "gray",
        "full_maxside": int(os.environ.get("SPARVAR_VIZ_FULL_MAXSIDE", "4096")),
        "full_max_pixels": int(os.environ.get("SPARVAR_VIZ_FULL_MAX_PIXELS", str(4096 * 4096))),
        "full_dpi": int(os.environ.get("SPARVAR_VIZ_FULL_DPI", "200")),
        "png_compress_level": max(0, min(int(os.environ.get("SPARVAR_VIZ_PNG_COMPRESS_LEVEL", "6")), 9)),
        # 'png' | 'pdf' | 'both' — output format of the matplotlib renderers
        # (FULL / BLOCKGRID / TOKENMASK). PDF keeps lines, text and ticks as
        # true vectors; the mask itself is embedded as a losslessly compressed
        # raster at the DATA resolution (interpolation='none'), so readers can
        # zoom to cell level.
        "format": os.environ.get("SPARVAR_VIZ_FORMAT", "png").strip().lower(),
    }


def _save_fig(fig, base):
    """Save a matplotlib figure as PNG and/or PDF per SPARVAR_VIZ_FORMAT.

    ``base`` is the output path WITHOUT extension. PNG rasterizes at the figure
    DPI; PDF keeps vector overlays and embeds imshow data unresampled."""
    fmt = _viz_config()["format"]
    exts = ("png", "pdf") if fmt == "both" else (fmt if fmt in ("png", "pdf") else "png",)
    for ext in exts:
        path = f"{base}.{ext}"
        fig.savefig(path, bbox_inches='tight')
        print(f"[viz]   saved {path}")


def viz_enabled_for(scale_ind, layer_ind) -> bool:
    """Cheap gate to call before doing any work (no matplotlib import here)."""
    cfg = _viz_config()
    if not cfg["enabled"]:
        return False
    if not isinstance(scale_ind, int) or not isinstance(layer_ind, int):
        return False
    if cfg["scales"] and scale_ind not in cfg["scales"]:
        return False
    if cfg["layers"] and layer_ind not in cfg["layers"]:
        return False
    return True


def image_kv_scale_boundaries(scale_schedule, kv_len, sink_len):
    """Per-scale KV token boundaries of the image stacked-pyramid KV.

    Infinity's KV at a sparse scale is the concat of ALL historical scales, so
    scale i ends at ``sum_{j<=i} h_j*w_j`` tokens. Returns the cumulative
    boundaries strictly between the sink boundary and ``kv_len`` (the sink
    boundary itself is drawn separately as the sink line), e.g. for 8B@1M
    si=12 with sink 5: [265, 521, 921, 1497, 2521, 4121, 6425].
    """
    cum, out = 0, []
    for s in scale_schedule:
        cum += int(s[-2]) * int(s[-1])
        if cum >= int(kv_len):
            break
        if cum > int(sink_len):
            out.append(cum)
    return out


def _apply_boundary_xticks(ax, *, sink_len, kv_scale_boundaries, kv_len,
                           tick_fontsize, tick_rotation, font_family):
    """Replace the default x ticks with the semantic token boundaries
    (0, sink, per-scale boundaries, kv_len) so every divider line is labeled
    with its KV token index. Rotated labels keep the crowded small-scale
    boundaries (e.g. 121/265/521/921 within the first 9% of the axis at si=12)
    readable. Falls back to default ticks when no boundary list is given."""
    import matplotlib.pyplot as plt
    if kv_scale_boundaries:
        ticks = {0, int(kv_len)}
        if 0 < sink_len < kv_len:
            ticks.add(int(sink_len))
        ticks.update(int(b) for b in kv_scale_boundaries if 0 < b < kv_len)
        ticks = sorted(ticks)
        ax.set_xticks(ticks)
        ax.set_xticklabels([str(t) for t in ticks],
                           fontsize=tick_fontsize, rotation=tick_rotation)
    else:
        ax.tick_params(axis='x', labelsize=tick_fontsize)
    ax.tick_params(axis='y', labelsize=tick_fontsize)
    if font_family is not None:
        plt.setp(ax.get_xticklabels() + ax.get_yticklabels(),
                 fontfamily=font_family)


def _choose_full_pool_factor(q_len: int, kv_len: int, max_side: int, max_pixels: int) -> int:
    """Choose a uniform max-pool factor for a macOS-openable full-mask PNG."""
    q_len, kv_len = max(int(q_len), 1), max(int(kv_len), 1)
    f = 1
    if max_side and max_side > 0 and max(q_len, kv_len) > max_side:
        f = max(f, (max(q_len, kv_len) + max_side - 1) // max_side)

    if max_pixels and max_pixels > 0:
        if ((q_len + f - 1) // f) * ((kv_len + f - 1) // f) > max_pixels:
            f = max(f, int(math.ceil(math.sqrt((q_len * kv_len) / max_pixels))))
            while ((q_len + f - 1) // f) * ((kv_len + f - 1) // f) > max_pixels:
                f += 1
    return max(f, 1)


@torch.no_grad()
def _dump_full_selection_masks(inds_b, counts_b, head_iter, *, scale_ind, layer_ind,
                               q_len, kv_len, bm, sink_len, tag, out_dir, qchunk,
                               color="gray", max_side=4096, max_pixels=4096 * 4096,
                               dpi=200,
                               # ---- token-coordinate geometry (block-space callers) ----
                               cell_tokens=1,             # tokens per index cell: 1 for token-space callers, block_size when inds/counts are BLOCK ids (CSLA / D-CSLA)
                               token_q_len=None,          # true q length in TOKENS (crops the padded whole-cell extent); default q_len*cell_tokens
                               token_kv_len=None,         # true kv length in TOKENS; default kv_len*cell_tokens
                               sink_len_tokens=None,      # sink boundary in TOKENS; default sink_len*cell_tokens (exact value preferred: e.g. 121, not 1 block = 128)
                               kv_scale_boundaries=None,  # per-scale KV token boundaries after the sink (image; also used as x ticks); None = no scale lines
                               # ---- line styles (edit defaults here) ----
                               sink_color='magenta',      # sink | rest boundary line color (visible on both the black bg and white selected cells)
                               sink_linestyle='-',        # '-' solid / '--' dashed / (0, (on, off)) custom dash
                               sink_linewidth=2.5,        # sink line width in points (bold = prominent)
                               sink_alpha=1.0,            # sink line opacity (0..1)
                               scale_line_color='orange', # per-scale boundary line color
                               scale_line_style='--',     # per-scale boundary line style
                               scale_line_width=1.2,      # per-scale boundary line width in points
                               scale_line_alpha=0.9,      # per-scale boundary line opacity
                               # ---- text styles ----
                               title_fontsize=24,         # figure title font size
                               axis_label_fontsize=18,    # x/y axis label font size
                               tick_fontsize=13,          # tick label font size
                               tick_rotation=90,          # x tick label rotation in degrees (vertical keeps crowded boundary labels readable)
                               font_family=None):         # font family for title/labels/ticks; None = matplotlib default
    """Render the COMPLETE selection mask (per query ROW), one PNG per head.

    Query row r selects the ``counts[g]`` valid keys of its group ``g = r // bm``.
    To keep files openable, the q_len x kv_len mask is MAX-POOLED down so its long
    side is <= ``max_side`` while preserving the exact aspect ratio (the same pool
    factor on both axes). Max-pool — not nearest/area — because the mask is sparse
    binary: any selected pixel in a pool block survives, so thin diagonal lines are
    kept. Pooling is done incrementally per query-row chunk, so the full-resolution
    mask is never materialized. The pooled canvas is rendered through matplotlib
    (axes in TOKEN coordinates via ``cell_tokens``), with the sink boundary,
    optional per-scale boundary lines and boundary-index x ticks drawn on top.

    color: 'gray' (black bg / white selected) or 'viridis' (purple / yellow).
    """
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    heads = list(head_iter)
    H, Gk, Kk = inds_b.shape
    q_len, kv_len = int(q_len), int(kv_len)
    qchunk = max(1, int(qchunk))
    device = inds_b.device

    # Aspect-preserving pool factor: same f on both axes => exact aspect ratio.
    f = _choose_full_pool_factor(q_len, kv_len, max_side=max_side, max_pixels=max_pixels)
    q_ds = (q_len + f - 1) // f
    k_ds = (kv_len + f - 1) // f
    rows_per_iter = max(1, qchunk // f)          # output rows processed per chunk
    print(f"[viz] FULL mask scale{scale_ind} layer{layer_ind}: {q_len}x{kv_len} "
          f"-> {q_ds}x{k_ds} (max-pool {f}x, {color}) x {len(heads)} head(s)")

    # Token-coordinate geometry: axes are TOKEN indices even when inds/counts are
    # block ids (cell_tokens=block_size), so boundary lines and ticks always land
    # on true token positions.
    cell_tokens = max(1, int(cell_tokens))
    tok_q = int(token_q_len) if token_q_len is not None else q_len * cell_tokens
    tok_kv = int(token_kv_len) if token_kv_len is not None else kv_len * cell_tokens
    sink_tok = int(sink_len_tokens) if sink_len_tokens is not None else int(sink_len) * cell_tokens
    ct = cell_tokens * f                          # tokens per pooled output cell
    cmap_name = 'viridis' if color == 'viridis' else 'gray'
    out_w_px = max(1200, min(int(max_side) if max_side else 4096, 16384))
    fig_width = out_w_px / dpi
    fig_height = max(4.0, fig_width * tok_q / max(tok_kv, 1))
    _ff = {} if font_family is None else {'fontfamily': font_family}

    for h in heads:
        cnt_h = counts_b[h].to(torch.long)
        k_eff = int(cnt_h.max().clamp(min=1).item())
        inds_h = inds_b[h, :, :k_eff].to(torch.long)
        kpos = torch.arange(k_eff, device=device).view(1, -1)
        ds = np.zeros((q_ds, k_ds), dtype=np.uint8)              # downsampled canvas
        for dro in range(0, q_ds, rows_per_iter):
            dr1 = min(dro + rows_per_iter, q_ds)
            n_out = dr1 - dro
            r0, r1 = dro * f, min(dr1 * f, q_len)               # input rows for these out-rows
            cq = r1 - r0
            g = torch.div(torch.arange(r0, r1, device=device), bm,
                          rounding_mode='trunc').clamp(0, Gk - 1)
            sel = inds_h[g].clamp(0, kv_len - 1)
            valid = kpos < cnt_h[g].view(-1, 1)
            row_ids = torch.arange(cq, device=device).view(-1, 1).expand(cq, k_eff)
            mask_chunk = torch.zeros(cq, kv_len, dtype=torch.uint8, device=device)
            mask_chunk[row_ids[valid], sel[valid]] = 1
            # key-axis max-pool (pad to k_ds*f, block-max over f columns)
            if k_ds * f - kv_len > 0:
                mask_chunk = F.pad(mask_chunk, (0, k_ds * f - kv_len))
            mk = mask_chunk.view(cq, k_ds, f).amax(dim=-1)       # [cq, k_ds]
            # row-axis max-pool (pad to n_out*f, block-max over f rows)
            if n_out * f - cq > 0:
                mk = F.pad(mk, (0, 0, 0, n_out * f - cq))
            block = mk.view(n_out, f, k_ds).amax(dim=1)          # [n_out, k_ds]
            ds[dro:dr1] = block.to(torch.uint8).cpu().numpy()
            del mask_chunk, mk, block, sel, valid, g, row_ids
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        # Integer-upscale small canvases (e.g. block-space callers: 32x83 cells)
        # so the raster embedded in a PDF has enough pixels per cell to stay
        # crisp in viewers that force image smoothing. Extent is unchanged
        # (the repeated array spans the same token range).
        ds_show = ds
        rep = max(1, 1024 // max(q_ds, k_ds))
        if rep > 1:
            ds_show = np.repeat(np.repeat(ds, rep, axis=0), rep, axis=1)
        # extent padded to WHOLE pooled cells so each ds cell is exactly ct tokens
        # wide; the xlim/ylim below crop the overhang back to the true token range
        # (same alignment trick as dump_dcsla_block_mask_viz).
        # interpolation='none': vector backends (pdf) embed the DATA array
        # unresampled (zoomable to cell level); Agg (png) treats it as 'nearest'.
        ax.imshow(ds_show, cmap=cmap_name, vmin=0, vmax=1, interpolation='none',
                  aspect='auto', extent=(0, k_ds * ct, q_ds * ct, 0))
        if 0 < sink_tok < tok_kv:                 # sink | rest boundary
            ax.axvline(x=sink_tok, color=sink_color, linestyle=sink_linestyle,
                       linewidth=sink_linewidth, alpha=sink_alpha)
        if kv_scale_boundaries:                   # per-scale KV boundaries after the sink
            xs = [int(x) for x in kv_scale_boundaries if 0 < x < tok_kv]
            if xs:
                ax.vlines(xs, 0, tok_q, color=scale_line_color,
                          linestyle=scale_line_style, linewidth=scale_line_width,
                          alpha=scale_line_alpha)
        ax.set_xlim(0, tok_kv)
        ax.set_ylim(tok_q, 0)
        _apply_boundary_xticks(ax, sink_len=sink_tok,
                               kv_scale_boundaries=kv_scale_boundaries,
                               kv_len=tok_kv, tick_fontsize=tick_fontsize,
                               tick_rotation=tick_rotation, font_family=font_family)
        ax.set_title(f"{tag} FULL mask  scale{scale_ind} layer{layer_ind} head{h}",
                     fontsize=title_fontsize, **_ff)
        ax.set_ylabel(f"Query Token Index (S_q = {tok_q})",
                      fontsize=axis_label_fontsize, **_ff)
        ax.set_xlabel(f"Key/Value Token Index (S_k = {tok_kv})",
                      fontsize=axis_label_fontsize, **_ff)
        _save_fig(fig, os.path.join(
            out_dir, f"{tag}_FULL_s{scale_ind}_l{layer_ind}_h{h}_Q{q_len}_K{kv_len}_ds{f}"))
        plt.close(fig)
        del ds, inds_h


@torch.no_grad()
def dump_selected_index_heatmap(inds, counts, *, scale_ind, layer_ind, q_len,
                                kv_len, bm, sink_len, tag="sel",
                                min_row_px=256,
                                cell_tokens=1,             # (FULL mode) tokens per index cell: block_size when inds are BLOCK ids
                                token_q_len=None,          # (FULL mode) true q length in TOKENS; default q_len*cell_tokens
                                token_kv_len=None,         # (FULL mode) true kv length in TOKENS; default kv_len*cell_tokens
                                sink_len_tokens=None,      # (FULL mode) exact sink boundary in TOKENS (e.g. 121, not 1 block)
                                kv_scale_boundaries=None,  # (FULL mode) per-scale KV token boundaries after the sink (image)
                                full_style=None):          # (FULL mode) dict of style overrides for _dump_full_selection_masks (line/font kwargs; defaults live on that function)
    """Render selected-index density heatmaps for one attention call.

    Args:
        inds:   [B, H, Gk, Kk] int — selected KV indices per query group.
        counts: [B, H, Gk] int    — number of *valid* entries per group.
        scale_ind, layer_ind:       current visual scale / transformer layer.
        q_len, kv_len:              query / KV sequence lengths at this scale.
        bm:                         query-group size (used only for the title).
        sink_len:                   length of the history (image+text) prefix;
                                    a red divider is drawn at this key column.
        tag:                        'decision' or 'mapped' (filename prefix).
    """
    cfg = _viz_config()
    if not cfg["enabled"]:
        return
    from PIL import Image  # local: keep module import cheap when disabled
    try:
        from matplotlib import colormaps
        cmap = colormaps["viridis"]
    except Exception:
        import matplotlib.cm as cm
        cmap = cm.get_cmap("viridis")

    b = 0  # conditional branch only
    inds_b = inds[b]                        # [H, Gk, Kk]
    counts_b = counts[b]                    # [H, Gk]
    H, Gk, Kk = inds_b.shape
    kv_len = int(kv_len)
    kbins = max(1, min(int(cfg["kbins"]), kv_len if kv_len > 0 else int(cfg["kbins"])))

    if cfg["heads"] is None:
        head_iter = range(H)
    else:
        head_iter = sorted(h for h in cfg["heads"] if 0 <= h < H)

    out_dir = os.path.join(cfg["out_dir"], f"scale{scale_ind}", f"layer{layer_ind}")
    os.makedirs(out_dir, exist_ok=True)

    if cfg["mode"] == "full":
        # Complete q_len x kv_len mask (per query row), max-pooled to a bounded size.
        # Meant for one layer + a few heads. Same head selection as fast mode.
        _dump_full_selection_masks(
            inds_b, counts_b, head_iter, scale_ind=scale_ind, layer_ind=layer_ind,
            q_len=int(q_len), kv_len=kv_len, bm=bm, sink_len=int(sink_len),
            tag=tag, out_dir=out_dir, qchunk=cfg["full_qchunk"],
            color=cfg["color"], max_side=cfg["full_maxside"],
            max_pixels=cfg["full_max_pixels"], dpi=cfg["full_dpi"],
            cell_tokens=cell_tokens, token_q_len=token_q_len,
            token_kv_len=token_kv_len, sink_len_tokens=sink_len_tokens,
            kv_scale_boundaries=kv_scale_boundaries, **(full_style or {}))
        return

    sink_col = int((sink_len * kbins) // max(1, kv_len)) if sink_len and sink_len > 0 else -1
    row_rep = max(1, math.ceil(min_row_px / max(1, Gk)))

    # Compute per head so the working set is ~[Gk, Kk] instead of [H, Gk, Kk]
    # (H x less memory): at the last video scale [H, Gk, kv_len] int64 alone is
    # tens of GB and would OOM. Output is identical.
    pos = torch.arange(Kk, device=inds_b.device).view(1, Kk)
    rows = torch.arange(Gk, device=inds_b.device).view(Gk, 1)
    for h in head_iter:
        inds_h = inds_b[h].to(torch.long)                          # [Gk, Kk]
        valid_h = pos < counts_b[h].to(torch.long).unsqueeze(-1)   # [Gk, Kk] bool
        col_h = ((inds_h.clamp(0, max(0, kv_len - 1)) * kbins) // max(1, kv_len)).clamp(0, kbins - 1)
        density = torch.zeros(Gk * kbins, dtype=torch.float32, device=inds_b.device)
        flat_idx = (rows * kbins + col_h)[valid_h]
        density.scatter_add_(0, flat_idx, torch.ones_like(flat_idx, dtype=torch.float32))
        dens_np = density.view(Gk, kbins).cpu().numpy()

        base = os.path.join(
            out_dir, f"{tag}_s{scale_ind}_l{layer_ind}_h{h}_Q{q_len}_K{kv_len}")
        np.save(base + ".npy", dens_np)

        norm = dens_np / (dens_np.max() + 1e-8)
        rgb = (cmap(norm)[..., :3] * 255).astype(np.uint8)    # [Gk, kbins, 3]
        if 0 <= sink_col < kbins:                              # mark sink|video boundary
            rgb[:, sink_col, :] = (255, 0, 0)
        if row_rep > 1:                                        # make short maps readable
            rgb = np.repeat(rgb, row_rep, axis=0)
        Image.fromarray(rgb).save(base + ".png")


@torch.no_grad()
def dump_dcsla_block_mask_viz(kv_indices, kv_num_blocks, *, scale_ind, layer_ind,
                              q_len, kv_len, block_size, sink_len, tag="dcsla",
                              scale_shape=None,
                              kv_scale_boundaries=None,      # per-scale KV token boundaries after the sink (image stacked-pyramid KV; also used as x ticks); None = video / no scale lines
                              # ---- line styles (edit defaults here) ----
                              grid_color='red',              # per-block partition grid color
                              grid_linestyle=(0, (2, 2)),    # dash pattern (offset, (on, off)) in points; NB dash lengths SCALE with linewidth, so thinner lines also mean denser dashes; '-' = solid
                              grid_linewidth=0.3,            # grid line width in points (thin + short dashes keep the block granularity readable when zoomed)
                              grid_alpha=0.75,               # grid line opacity (0..1)
                              sink_color='magenta',          # sink | rest boundary color (visible on both the black bg and white selected cells)
                              sink_linestyle='-',            # sink line style ('-' solid = prominent)
                              sink_linewidth=2.5,            # sink line width in points (bold)
                              sink_alpha=1.0,                # sink line opacity
                              scale_line_color='orange',     # per-scale boundary line color
                              scale_line_style='--',         # per-scale boundary line style
                              scale_line_width=1.2,          # per-scale boundary line width in points
                              scale_line_alpha=0.9,          # per-scale boundary line opacity
                              frame_color='cyan',            # (video) frame boundary line color
                              frame_linestyle=(0, (6, 3)),   # (video) frame boundary dash pattern
                              frame_linewidth=1.8,           # (video) frame boundary line width in points
                              frame_alpha=0.95,              # (video) frame boundary opacity
                              # ---- text styles ----
                              title_fontsize=24,             # figure title font size
                              axis_label_fontsize=18,        # x/y axis label font size
                              tick_fontsize=13,              # tick label font size
                              tick_rotation=90,              # x tick rotation in degrees; vertical keeps the crowded small-scale boundary labels (121/265/521/921) readable
                              font_family=None):             # font family for title/labels/ticks; None = matplotlib default
    """Full token-coordinate visualization of a D-CSLA block selection, one PNG
    per head, in the style of the HART/Infinity CSLA notebook figure
    (models/hart/SparVAR_HART/sparse_attn.ipynb::visualize_mask):

      * the complete [q_len, kv_len] mask (white = attended), max-pooled to the
        FULL_MAXSIDE / FULL_MAX_PIXELS caps — pooling is computed EXACTLY from
        the [U, V] block mask via a 2D integral image (the token mask is
        block-constant), so the dense token mask is never materialized;
      * red dashed lines mark the block_size x block_size block partition;
      * a bold solid vertical line marks the sink | rest boundary;
      * (image) one vertical line per historical scale after the sink
        (``kv_scale_boundaries``), with every boundary's KV token index used
        as a rotated x tick label;
      * optional cyan dotted lines mark video frame boundaries on both axes
        (needs scale_shape=(T, H, W)).

    All line styles (color / width / dash) and text styles (title / axis-label /
    tick font size, tick rotation, font family) are keyword arguments with
    commented defaults in the signature — edit them there.

    Args:
        kv_indices:    [B, H, U, V] int — FULL-WIDTH per-q-block block-id rows
                       (first kv_num_blocks[b,h,u] entries are the selection).
        kv_num_blocks: [B, H, U] int — number of valid entries per row.
        q_len, kv_len: token-level sequence lengths (axes are token indices).
        block_size:    D-CSLA block size (128).
        sink_len:      ctx prefix length in TOKENS.
        scale_shape:   optional (T, H, W) of the current video scale.
    """
    cfg = _viz_config()
    if not cfg["enabled"]:
        return
    if os.environ.get("SPARVAR_VIZ_BLOCK_GRID", "1").lower() not in _TRUE:
        return
    import matplotlib.pyplot as plt  # local: keep module import cheap when disabled

    b = 0  # conditional branch only
    inds_b = kv_indices[b]                     # [H, U, V]
    counts_b = kv_num_blocks[b]                # [H, U]
    H, U, V = inds_b.shape
    q_len, kv_len, block_size = int(q_len), int(kv_len), int(block_size)
    device = inds_b.device
    grid_every = max(0, int(os.environ.get("SPARVAR_VIZ_BLOCK_GRID_EVERY", "1")))
    frame_lines = os.environ.get("SPARVAR_VIZ_BLOCK_FRAME_LINES", "1").lower() in _TRUE

    if cfg["heads"] is None:
        head_iter = list(range(H))
    else:
        head_iter = sorted(h for h in cfg["heads"] if 0 <= h < H)

    out_dir = os.path.join(cfg["out_dir"], f"scale{scale_ind}", f"layer{layer_ind}")
    os.makedirs(out_dir, exist_ok=True)

    # Native block-mask render: imshow the [U, V] block mask directly with
    # interpolation='nearest', so every 128-block is a crisp cell (the old path
    # max-pooled the token mask to a factor that did not divide block_size (128),
    # aliasing the cells). Black/white mask; matplotlib upscales the small [U, V]
    # array to the figure, and extent=token-coords keeps the grid/sink/frame lines
    # aligned. VIZ_COLOR governs only the compact block-space heatmap, not this figure.
    cmap_name = 'gray'                                    # white = selected block, black = unselected
    print(f"[viz] D-CSLA block mask scale{scale_ind} layer{layer_ind}: "
          f"blocks {U}x{V} (native, b/w) x {len(head_iter)} head(s)")

    pos = torch.arange(V, device=device).view(1, V)
    dpi = cfg["full_dpi"]
    # Output width = FULL_MAXSIDE (raise it for higher resolution; capped at 16384 for
    # a sane file). Higher res widens the block spacing so the dashed per-block red grid
    # stays crisp instead of merging. MAXSIDE=0 (native-resolution pooling) falls
    # back to a 4096 px figure so the canvas does not collapse to the 1200 floor.
    out_w_px = max(1200, min(int(cfg["full_maxside"]) if cfg["full_maxside"] else 4096, 16384))
    fig_width = out_w_px / dpi
    fig_height = max(4.0, fig_width * q_len / max(kv_len, 1))

    for h in head_iter:
        valid = pos < counts_b[h].to(torch.long).unsqueeze(-1)            # [U, V]
        blk = torch.zeros(U, V, dtype=torch.bool, device=device)
        blk.scatter_(-1, inds_b[h].to(torch.long).clamp(0, V - 1), valid)  # [U, V]
        density = counts_b[h].float().mean().item() / max(V, 1)

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        # extent padded to WHOLE blocks (V*block_size x U*block_size) so each imshow
        # cell is EXACTLY block_size wide -> the red grid at multiples of block_size
        # lands on cell edges. (With extent=kv_len the V cells were stretched to
        # kv_len/V ~= 128.0085 tokens each and drifted off the 128-spaced grid, so
        # selected cells no longer aligned with the boundary lines.) The xlim/ylim
        # below crop the sub-block overhang back to the true token range.
        # Integer-upscale the tiny [U, V] block array so the raster embedded in
        # a PDF has enough pixels per cell to stay crisp even in viewers that
        # force image smoothing (e.g. macOS Preview). Extent is unchanged.
        blk_np = blk.cpu().numpy()
        rep = max(1, 1024 // max(U, V))
        if rep > 1:
            blk_np = np.repeat(np.repeat(blk_np, rep, axis=0), rep, axis=1)
        # interpolation='none': vector backends (pdf) embed the DATA array
        # unresampled; Agg (png) treats it as 'nearest'.
        ax.imshow(blk_np, cmap=cmap_name, vmin=0, vmax=1,
                  interpolation='none', aspect='auto',
                  extent=(0, V * block_size, U * block_size, 0))

        # Feature 1: block partition grid — red DASHED, token coordinates (aligned to
        # the padded-block cell edges above). SPARVAR_VIZ_BLOCK_GRID_EVERY=N draws every
        # Nth block (1 = every block; 0 = off); raise FULL_MAXSIDE if the dashes merge.
        if grid_every > 0:
            step = block_size * grid_every
            xs = np.arange(step, kv_len, step)
            ys = np.arange(step, q_len, step)
            if len(xs):
                ax.vlines(xs, 0, q_len, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha)
            if len(ys):
                ax.hlines(ys, 0, kv_len, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha)

        # Feature 2: sink | rest boundary (bold solid by default, so the sink
        # region stands out from the per-block grid).
        if 0 < sink_len < kv_len:
            ax.axvline(x=sink_len, color=sink_color, linestyle=sink_linestyle,
                       linewidth=sink_linewidth, alpha=sink_alpha)

        # Feature 3: per-scale KV boundaries after the sink (image stacked-pyramid
        # KV): one vertical line where each historical scale ends.
        if kv_scale_boundaries:
            sxs = np.asarray([int(x) for x in kv_scale_boundaries if 0 < x < kv_len])
            if len(sxs):
                ax.vlines(sxs, 0, q_len, color=scale_line_color,
                          linestyle=scale_line_style, linewidth=scale_line_width,
                          alpha=scale_line_alpha)

        # Feature 4: video frame boundaries on both axes.
        if frame_lines and scale_shape is not None and len(tuple(scale_shape)) == 3:
            T, Hh, Ww = (int(x) for x in scale_shape)
            hw = Hh * Ww
            if T > 1 and hw > 0:
                fx = int(sink_len) + np.arange(1, T) * hw
                fy = np.arange(1, T) * hw
                ax.vlines(fx[fx < kv_len], 0, q_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)
                ax.hlines(fy[fy < q_len], 0, kv_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)

        ax.set_xlim(0, kv_len)
        ax.set_ylim(q_len, 0)
        # Boundary-index x ticks (image): every divider line is labeled with its
        # KV token index; video (no boundary list) keeps the default ticks.
        _apply_boundary_xticks(ax, sink_len=int(sink_len),
                               kv_scale_boundaries=kv_scale_boundaries,
                               kv_len=kv_len, tick_fontsize=tick_fontsize,
                               tick_rotation=tick_rotation, font_family=font_family)
        _ff = {} if font_family is None else {'fontfamily': font_family}
        ax.set_title(f"D-CSLA block mask (block={block_size})  "
                     f"scale{scale_ind} layer{layer_ind} head{h}  "
                     f"density={density:.3f}", fontsize=title_fontsize, **_ff)
        ax.set_ylabel(f"Query Token Index (S_q = {q_len})",
                      fontsize=axis_label_fontsize, **_ff)
        ax.set_xlabel(f"Key/Value Token Index (S_k = {kv_len})",
                      fontsize=axis_label_fontsize, **_ff)

        _save_fig(fig, os.path.join(
            out_dir, f"{tag}_BLOCKGRID_s{scale_ind}_l{layer_ind}_h{h}"
                     f"_Q{q_len}_K{kv_len}_blk{block_size}"))
        plt.close(fig)


@torch.no_grad()
def dump_csla_token_mask_viz(token_mask, *, scale_ind, layer_ind, block_size,
                             sink_len, kv_scale_boundaries=None, tag="csla",
                             # ---- line styles (edit defaults here) ----
                             grid_every=1,                  # draw the block grid every Nth block (1 = every block; 0 = grid off)
                             grid_color='red',              # block partition grid color
                             grid_linestyle=(0, (2, 2)),    # dash pattern (offset, (on, off)) in points; dash lengths SCALE with linewidth; '-' = solid
                             grid_linewidth=0.3,            # grid line width in points
                             grid_alpha=0.75,               # grid line opacity (0..1)
                             sink_color='magenta',          # sink | rest boundary color
                             sink_linestyle='-',            # sink line style ('-' solid = prominent)
                             sink_linewidth=2.5,            # sink line width in points (bold)
                             sink_alpha=1.0,                # sink line opacity
                             scale_line_color='orange',     # per-scale boundary line color
                             scale_line_style='--',         # per-scale boundary line style
                             scale_line_width=1.2,          # per-scale boundary line width in points
                             scale_line_alpha=0.9,          # per-scale boundary line opacity
                             # ---- text styles ----
                             title_fontsize=24,             # figure title font size
                             axis_label_fontsize=18,        # x/y axis label font size
                             tick_fontsize=13,              # tick label font size
                             tick_rotation=90,              # x tick rotation in degrees
                             font_family=None):             # font family for title/labels/ticks; None = matplotlib default
    """TOKEN-wise CSLA fine-mask render (sink prefix OR per-scale band, BEFORE
    the block-any reduction), in the style of the HART/Infinity notebook figure
    (``models/hart/SparVAR_HART/sparse_attn.ipynb::visualize_mask``): one PNG
    per scale (the static mask is layer/head independent), with the block
    partition grid, the bold sink line, per-scale KV boundary lines and
    boundary-index x ticks — same style knobs as the BLOCKGRID render.

    ``SPARVAR_VIZ_TOKEN_MASK=0`` disables just this renderer. The dense mask is
    max-pooled to the FULL_MAXSIDE / FULL_MAX_PIXELS caps (max-pool keeps thin
    selected bands visible).

    Args:
        token_mask: [S_q, S_kv] bool — the dense token-level mask.
        block_size: block partition drawn by the grid (128).
        sink_len:   sink boundary in TOKENS.
        kv_scale_boundaries: per-scale KV token boundaries after the sink.
    """
    cfg = _viz_config()
    if not cfg["enabled"]:
        return
    if os.environ.get("SPARVAR_VIZ_TOKEN_MASK", "1").lower() not in _TRUE:
        return
    import torch.nn.functional as F
    import matplotlib.pyplot as plt

    q_len, kv_len = int(token_mask.shape[0]), int(token_mask.shape[1])
    out_dir = os.path.join(cfg["out_dir"], f"scale{scale_ind}", f"layer{layer_ind}")
    os.makedirs(out_dir, exist_ok=True)

    # Max-pool the dense mask to the openable-size caps (any selected token in a
    # pool cell survives, so the thin win-band diagonals are kept).
    f = _choose_full_pool_factor(q_len, kv_len, max_side=cfg["full_maxside"],
                                 max_pixels=cfg["full_max_pixels"])
    q_ds = (q_len + f - 1) // f
    k_ds = (kv_len + f - 1) // f
    m = token_mask.to(torch.uint8)
    if f > 1:
        m = F.pad(m, (0, k_ds * f - kv_len, 0, q_ds * f - q_len))
        m = m.view(q_ds, f, k_ds, f).amax(dim=(1, 3))
    ds = m.cpu().numpy()
    density = float(token_mask.float().mean().item())
    print(f"[viz] CSLA token mask scale{scale_ind}: {q_len}x{kv_len} "
          f"-> {q_ds}x{k_ds} (max-pool {f}x) density={density:.4f}")

    dpi = cfg["full_dpi"]
    out_w_px = max(1200, min(int(cfg["full_maxside"]) if cfg["full_maxside"] else 4096, 16384))
    fig_width = out_w_px / dpi
    fig_height = max(4.0, fig_width * q_len / max(kv_len, 1))
    _ff = {} if font_family is None else {'fontfamily': font_family}

    fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
    # extent padded to whole pool cells (f tokens each); xlim/ylim crop back to
    # the true token range so all overlays land on exact token positions.
    # interpolation='none': vector backends (pdf) embed the DATA array
    # unresampled (zoom to token level); Agg (png) treats it as 'nearest'.
    ax.imshow(ds, cmap='gray', vmin=0, vmax=1, interpolation='none',
              aspect='auto', extent=(0, k_ds * f, q_ds * f, 0))

    if grid_every and grid_every > 0:                # block partition grid
        step = int(block_size) * int(grid_every)
        xs = np.arange(step, kv_len, step)
        ys = np.arange(step, q_len, step)
        if len(xs):
            ax.vlines(xs, 0, q_len, color=grid_color, linestyle=grid_linestyle,
                      linewidth=grid_linewidth, alpha=grid_alpha)
        if len(ys):
            ax.hlines(ys, 0, kv_len, color=grid_color, linestyle=grid_linestyle,
                      linewidth=grid_linewidth, alpha=grid_alpha)
    if 0 < sink_len < kv_len:                        # sink | rest boundary
        ax.axvline(x=int(sink_len), color=sink_color, linestyle=sink_linestyle,
                   linewidth=sink_linewidth, alpha=sink_alpha)
    if kv_scale_boundaries:                          # per-scale KV boundaries after the sink
        sxs = np.asarray([int(x) for x in kv_scale_boundaries if 0 < x < kv_len])
        if len(sxs):
            ax.vlines(sxs, 0, q_len, color=scale_line_color,
                      linestyle=scale_line_style, linewidth=scale_line_width,
                      alpha=scale_line_alpha)

    ax.set_xlim(0, kv_len)
    ax.set_ylim(q_len, 0)
    _apply_boundary_xticks(ax, sink_len=int(sink_len),
                           kv_scale_boundaries=kv_scale_boundaries,
                           kv_len=kv_len, tick_fontsize=tick_fontsize,
                           tick_rotation=tick_rotation, font_family=font_family)
    ax.set_title(f"CSLA token mask  scale{scale_ind}  density={density:.4f}",
                 fontsize=title_fontsize, **_ff)
    ax.set_ylabel(f"Query Token Index (S_q = {q_len})",
                  fontsize=axis_label_fontsize, **_ff)
    ax.set_xlabel(f"Key/Value Token Index (S_k = {kv_len})",
                  fontsize=axis_label_fontsize, **_ff)

    _save_fig(fig, os.path.join(
        out_dir, f"{tag}_TOKENMASK_s{scale_ind}_Q{q_len}_K{kv_len}_ds{f}"))
    plt.close(fig)


def _dcsla_mass_only_selection(mass_uv, *, sink_block_end, mode, tau, topk_ratio,
                               keep_sink, sink_select_tau):
    """``[U, V]`` float mass -> ``[U, V]`` bool: the UNCONSTRAINED mass-ranking
    selection — mirrors the coverage/topk core of ``select_kv_blocks`` (sink
    zeroing + sink-mass credit included, since those are part of the mass rule)
    but WITHOUT the [min_blocks, max_blocks] clamp, the forced sink fill and
    the locality priors. Used only to color-code the origin of each finally
    selected block in ``dump_dcsla_mass_selection_viz``:

      final & mass_only   -> picked by the mass prefix        (red)
      final & ~mass_only  -> forced by constraint / prior     (orange)
      ~final & mass_only  -> wanted by mass, cut by the cap   (blue)
    """
    U, V = mass_uv.shape
    device = mass_uv.device
    mf = mass_uv.float()
    sink_split = (keep_sink or float(sink_select_tau) > 0) and sink_block_end > 0
    m = mf
    sink_sel = None
    sink_covered = None
    if sink_split:
        m = mf.clone()
        m[:, :sink_block_end] = 0.0
        if keep_sink:
            sink_covered = mf[:, :sink_block_end].sum(dim=-1)
        else:
            ms = mf[:, :sink_block_end]
            s_sorted, s_idx = ms.sort(dim=-1, descending=True, stable=True)
            s_csum = s_sorted.cumsum(dim=-1)
            s_total = s_csum[:, -1]
            n_s = (s_csum < float(sink_select_tau)
                   * s_total.unsqueeze(-1).clamp_min(1e-12)).sum(dim=-1) + 1
            n_s = torch.where(s_total > 0, n_s, torch.ones_like(n_s)).clamp_(1, sink_block_end)
            keep_s = torch.arange(sink_block_end, device=device).view(1, -1) < n_s.unsqueeze(-1)
            sink_sel = torch.zeros_like(keep_s)
            sink_sel.scatter_(-1, s_idx, keep_s)
            sink_covered = torch.gather(s_csum, -1, (n_s - 1).unsqueeze(-1)).squeeze(-1)

    sorted_m, sort_idx = m.sort(dim=-1, descending=True, stable=True)
    if mode == 'coverage':
        csum = sorted_m.cumsum(dim=-1)
        target = float(tau) * mf.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        if sink_covered is not None:
            target = (target - sink_covered.unsqueeze(-1)).clamp_min(0.0)
        n = (csum < target).sum(dim=-1) + 1
    else:  # topk_ratio
        n = torch.full((U,), int(math.ceil(float(topk_ratio) * V)),
                       device=device, dtype=torch.int64)
    n = n.clamp_(max=V)
    keep = torch.arange(V, device=device).view(1, V) < n.unsqueeze(-1)
    sel = torch.zeros(U, V, dtype=torch.bool, device=device)
    sel.scatter_(-1, sort_idx, keep)
    if sink_sel is not None:
        sel[:, :sink_block_end] |= sink_sel
    return sel


def _upscale2d(arr, target=1024):
    """Per-axis integer repeat so a small array embeds with enough pixels per
    cell to stay crisp in smoothing PDF viewers. Extent-based drawing is
    unaffected (the repeated array spans the same coordinate range)."""
    rep_r = max(1, target // max(arr.shape[0], 1))
    rep_c = max(1, target // max(arr.shape[1], 1))
    if rep_r > 1:
        arr = np.repeat(arr, rep_r, axis=0)
    if rep_c > 1:
        arr = np.repeat(arr, rep_c, axis=1)
    return arr


@torch.no_grad()
def dump_dcsla_mass_selection_viz(mass_rows, kv_indices, kv_num_blocks, *,
                                  scale_ind, layer_ind, q_len, kv_len, block_size,
                                  sink_len, attn_config, scale_shape=None,
                                  kv_scale_boundaries=None, tag="dcsla",
                                  # ---- mass background (edit defaults here) ----
                                  mass_clip_pct=99.5,        # percentile clip for the heatmap vmax (block mass is heavy-tailed; sink columns would crush the rest)
                                  mass_log=False,            # True: log1p scale the background instead of clipping
                                  # ---- selection outlines ----
                                  sel_mass_color='red',      # blocks picked by the mass prefix (coverage tau / topk_ratio)
                                  sel_forced_color='orange', # blocks forced by constraints/priors (keep_sink, min_blocks, diag/temporal/band)
                                  sel_cut_color='deepskyblue',  # blocks the raw mass rule wanted but max_blocks_ratio cut (NOT selected)
                                  sel_linewidth=0.7,         # outline width in points
                                  show_legend=True,          # draw the 3-color legend
                                  legend_fontsize=16,        # legend text font size
                                  row_density_strip=False,   # right-hand strip: per-row selected fraction (adaptive density profile)
                                  # ---- overlays (same conventions as the other renders) ----
                                  grid_every=0,              # block grid every Nth block (0 = off: the outlines already delineate cells)
                                  grid_color='red', grid_linestyle=(0, (2, 2)), grid_linewidth=0.3, grid_alpha=0.75,
                                  sink_color='magenta', sink_linestyle='-', sink_linewidth=2.5, sink_alpha=1.0,
                                  scale_line_color='orange', scale_line_style='--', scale_line_width=1.2, scale_line_alpha=0.9,
                                  frame_color='cyan', frame_linestyle=(0, (6, 3)), frame_linewidth=1.8, frame_alpha=0.95,
                                  # ---- text styles ----
                                  title_fontsize=24, axis_label_fontsize=18,
                                  tick_fontsize=13, tick_rotation=90, font_family=None):
    """D-CSLA per-sparse-scale 'evidence + result' figure, one PNG/PDF per head:

      * background: the phi-remapped block mass ([U, V], viridis) — the exact
        input of the block selection at this scale;
      * overlay: the FINAL selected blocks (from the kv-lists fed to the
        kernel) as outlined cells, color-coded by origin via a viz-side replay
        of the unconstrained mass rule (see _dcsla_mass_only_selection);
      * plus the usual sink line, video frame lines / image scale boundaries,
        boundary-index x ticks, optional block grid.

    Replaces the old 'dcsla_blocks' heatmap. SPARVAR_VIZ_MASS_SEL=0 disables.
    """
    cfg = _viz_config()
    if not cfg["enabled"]:
        return
    if os.environ.get("SPARVAR_VIZ_MASS_SEL", "1").lower() not in _TRUE:
        return
    import matplotlib.pyplot as plt
    from matplotlib.collections import PatchCollection
    from matplotlib.patches import Patch, Rectangle

    b = 0  # conditional branch only
    mass_b = mass_rows[b].float()                # [H, U, V]
    inds_b = kv_indices[b]                       # [H, U, V]
    counts_b = kv_num_blocks[b]                  # [H, U]
    H, U, V = mass_b.shape
    q_len, kv_len, block_size = int(q_len), int(kv_len), int(block_size)
    device = mass_b.device
    sink_block_end = min((max(int(sink_len), 0) + block_size - 1) // block_size, V)

    head_iter = (list(range(H)) if cfg["heads"] is None
                 else sorted(h for h in cfg["heads"] if 0 <= h < H))
    out_dir = os.path.join(cfg["out_dir"], f"scale{scale_ind}", f"layer{layer_ind}")
    os.makedirs(out_dir, exist_ok=True)

    mode = str(attn_config.get('dcsla_select_mode', 'coverage'))
    tau = float(attn_config.get('dcsla_tau', 0.9))
    topk_ratio = float(attn_config.get('dcsla_topk_ratio', 0.2))
    keep_sink = bool(attn_config.get('dcsla_keep_sink', True))
    sink_select_tau = float(attn_config.get('dcsla_sink_select_tau', 0.0))
    mode_str = (f"coverage tau={tau:g}" if mode == 'coverage'
                else f"topk_ratio={topk_ratio:g}")
    print(f"[viz] D-CSLA mass+selection scale{scale_ind} layer{layer_ind}: "
          f"blocks {U}x{V} ({mode_str}) x {len(head_iter)} head(s)")

    dpi = cfg["full_dpi"]
    out_w_px = max(1200, min(int(cfg["full_maxside"]) if cfg["full_maxside"] else 4096, 16384))
    fig_width = out_w_px / dpi
    fig_height = max(4.0, fig_width * q_len / max(kv_len, 1))
    _ff = {} if font_family is None else {'fontfamily': font_family}
    pos = torch.arange(V, device=device).view(1, V)

    for h in head_iter:
        # final selection (kernel ground truth) from the kv-lists
        valid = pos < counts_b[h].to(torch.long).unsqueeze(-1)             # [U, V]
        final = torch.zeros(U, V, dtype=torch.bool, device=device)
        final.scatter_(-1, inds_b[h].to(torch.long).clamp(0, V - 1), valid)
        density = counts_b[h].float().mean().item() / max(V, 1)
        # unconstrained mass-rule replay -> origin categories
        mass_only = _dcsla_mass_only_selection(
            mass_b[h], sink_block_end=sink_block_end, mode=mode, tau=tau,
            topk_ratio=topk_ratio, keep_sink=keep_sink,
            sink_select_tau=sink_select_tau)
        cat_mass = (final & mass_only).cpu().numpy()
        cat_forced = (final & ~mass_only).cpu().numpy()
        cat_cut = (~final & mass_only).cpu().numpy()

        # background: clipped / log mass heatmap
        m_np = mass_b[h].cpu().numpy()
        if mass_log:
            m_show, vmax = np.log1p(m_np), None
        else:
            m_show = m_np
            vmax = float(np.percentile(m_np, mass_clip_pct)) if m_np.size else None
            if not vmax or vmax <= 0:
                vmax = None

        if row_density_strip:
            fig, (ax, ax_d) = plt.subplots(
                1, 2, figsize=(fig_width, fig_height), dpi=dpi, sharey=True,
                gridspec_kw={'width_ratios': [24, 2], 'wspace': 0.02})
        else:
            fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
            ax_d = None
        ax.imshow(_upscale2d(m_show), cmap='viridis', vmin=0.0, vmax=vmax,
                  interpolation='none', aspect='auto',
                  extent=(0, V * block_size, U * block_size, 0))

        # selection outlines (token coords; one rect per selected block cell)
        for cat, color in ((cat_mass, sel_mass_color),
                           (cat_forced, sel_forced_color),
                           (cat_cut, sel_cut_color)):
            uu, vv = np.nonzero(cat)
            if len(uu):
                rects = [Rectangle((v * block_size, u * block_size), block_size, block_size)
                         for u, v in zip(uu, vv)]
                ax.add_collection(PatchCollection(
                    rects, facecolor='none', edgecolor=color,
                    linewidth=sel_linewidth))

        if grid_every and grid_every > 0:
            step = block_size * int(grid_every)
            xs = np.arange(step, kv_len, step)
            ys = np.arange(step, q_len, step)
            if len(xs):
                ax.vlines(xs, 0, q_len, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha)
            if len(ys):
                ax.hlines(ys, 0, kv_len, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha)
        if 0 < sink_len < kv_len:
            ax.axvline(x=int(sink_len), color=sink_color, linestyle=sink_linestyle,
                       linewidth=sink_linewidth, alpha=sink_alpha)
        if kv_scale_boundaries:
            sxs = np.asarray([int(x) for x in kv_scale_boundaries if 0 < x < kv_len])
            if len(sxs):
                ax.vlines(sxs, 0, q_len, color=scale_line_color,
                          linestyle=scale_line_style, linewidth=scale_line_width,
                          alpha=scale_line_alpha)
        if scale_shape is not None and len(tuple(scale_shape)) == 3:
            T, Hh, Ww = (int(x) for x in scale_shape)
            hw = Hh * Ww
            if T > 1 and hw > 0:
                fx = int(sink_len) + np.arange(1, T) * hw
                fy = np.arange(1, T) * hw
                ax.vlines(fx[fx < kv_len], 0, q_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)
                ax.hlines(fy[fy < q_len], 0, kv_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)

        ax.set_xlim(0, kv_len)
        ax.set_ylim(q_len, 0)
        _apply_boundary_xticks(ax, sink_len=int(sink_len),
                               kv_scale_boundaries=kv_scale_boundaries,
                               kv_len=kv_len, tick_fontsize=tick_fontsize,
                               tick_rotation=tick_rotation, font_family=font_family)
        if show_legend:
            ax.legend(handles=[
                Patch(facecolor='none', edgecolor=sel_mass_color, label='mass prefix'),
                Patch(facecolor='none', edgecolor=sel_forced_color, label='forced (sink/min/prior)'),
                Patch(facecolor='none', edgecolor=sel_cut_color, label='cut by cap (not selected)')],
                loc='upper right', fontsize=legend_fontsize, framealpha=0.85)
        ax.set_title(f"D-CSLA mass + selection ({mode_str}, block={block_size})  "
                     f"scale{scale_ind} layer{layer_ind} head{h}  "
                     f"density={density:.3f}", fontsize=title_fontsize, **_ff)
        ax.set_ylabel(f"Query Token Index (S_q = {q_len})",
                      fontsize=axis_label_fontsize, **_ff)
        ax.set_xlabel(f"Key/Value Token Index (S_k = {kv_len})",
                      fontsize=axis_label_fontsize, **_ff)

        if ax_d is not None:
            frac = (counts_b[h].float() / max(V, 1)).cpu().numpy()
            yc = (np.arange(U) + 0.5) * block_size
            ax_d.plot(frac, yc, color=sel_mass_color, linewidth=1.0)
            ax_d.set_xlim(0, 1)
            ax_d.set_xlabel('sel/V', fontsize=tick_fontsize, **_ff)
            ax_d.tick_params(axis='x', labelsize=tick_fontsize)
            ax_d.grid(True, alpha=0.3)

        _save_fig(fig, os.path.join(
            out_dir, f"{tag}_MASSSEL_s{scale_ind}_l{layer_ind}_h{h}"
                     f"_Q{q_len}_K{kv_len}_blk{block_size}"))
        plt.close(fig)


@torch.no_grad()
def dump_decision_colsum_viz(bs, *, scale_ind, layer_ind, q_len, k_len, block_size,
                             sink_len, scale_shape=None, kv_scale_boundaries=None,
                             tag="dcsla",
                             # ---- heatmap scaling (edit defaults here) ----
                             colsum_clip_pct=99.5,      # percentile clip for vmax (colsum is heavy-tailed at the sink)
                             colsum_log=False,          # True: log1p scale instead of clipping
                             # ---- overlays ----
                             grid_every=1,              # KV-axis block grid every Nth block (0 = off)
                             grid_color='red', grid_linestyle=(0, (2, 2)), grid_linewidth=0.3, grid_alpha=0.75,
                             sink_color='magenta', sink_linestyle='-', sink_linewidth=2.5, sink_alpha=1.0,
                             scale_line_color='orange', scale_line_style='--', scale_line_width=1.2, scale_line_alpha=0.9,
                             frame_color='cyan', frame_linestyle=(0, (6, 3)), frame_linewidth=1.8, frame_alpha=0.95,
                             # ---- text styles ----
                             title_fontsize=24, axis_label_fontsize=18,
                             tick_fontsize=13, tick_rotation=90, font_family=None):
    """D-CSLA decision-scale raw-evidence figure, one PNG/PDF per head: the
    token-wise colsum ``bs [B, H, G, k_len]`` as a continuous heatmap (rows =
    colsum row-groups stretched to their token span, cols = KV tokens), with
    the KV block grid, sink line, video frame lines / image scale boundaries
    and boundary-index x ticks — shows how token mass clusters inside the
    128-blocks that the selection later ranks. SPARVAR_VIZ_DECISION_COLSUM=0
    disables just this renderer.
    """
    cfg = _viz_config()
    if not cfg["enabled"]:
        return
    if os.environ.get("SPARVAR_VIZ_DECISION_COLSUM", "1").lower() not in _TRUE:
        return
    import matplotlib.pyplot as plt

    b = 0  # conditional branch only
    bs_b = bs[b].float()                          # [H, G, k_len]
    H, G, k_len_bs = bs_b.shape
    q_len, k_len, block_size = int(q_len), int(k_len), int(block_size)
    rows_per_group = max(1, (q_len + G - 1) // G)  # token span of one colsum row-group

    head_iter = (list(range(H)) if cfg["heads"] is None
                 else sorted(h for h in cfg["heads"] if 0 <= h < H))
    out_dir = os.path.join(cfg["out_dir"], f"scale{scale_ind}", f"layer{layer_ind}")
    os.makedirs(out_dir, exist_ok=True)
    print(f"[viz] D-CSLA decision colsum scale{scale_ind} layer{layer_ind}: "
          f"{G}x{k_len_bs} groups x tokens, {len(head_iter)} head(s)")

    dpi = cfg["full_dpi"]
    out_w_px = max(1200, min(int(cfg["full_maxside"]) if cfg["full_maxside"] else 4096, 16384))
    fig_width = out_w_px / dpi
    fig_height = max(4.0, fig_width * q_len / max(k_len, 1))
    _ff = {} if font_family is None else {'fontfamily': font_family}

    for h in head_iter:
        img = bs_b[h].cpu().numpy()               # [G, k_len]
        if colsum_log:
            img_show, vmax = np.log1p(img), None
        else:
            img_show = img
            vmax = float(np.percentile(img, colsum_clip_pct)) if img.size else None
            if not vmax or vmax <= 0:
                vmax = None

        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=dpi)
        # rows stretched to their token span; xlim/ylim crop the group overhang.
        ax.imshow(_upscale2d(img_show), cmap='viridis', vmin=0.0, vmax=vmax,
                  interpolation='none', aspect='auto',
                  extent=(0, k_len_bs, G * rows_per_group, 0))

        if grid_every and grid_every > 0:          # KV-axis block grid
            step = block_size * int(grid_every)
            xs = np.arange(step, k_len, step)
            if len(xs):
                ax.vlines(xs, 0, q_len, color=grid_color, linestyle=grid_linestyle,
                          linewidth=grid_linewidth, alpha=grid_alpha)
        if 0 < sink_len < k_len:
            ax.axvline(x=int(sink_len), color=sink_color, linestyle=sink_linestyle,
                       linewidth=sink_linewidth, alpha=sink_alpha)
        if kv_scale_boundaries:
            sxs = np.asarray([int(x) for x in kv_scale_boundaries if 0 < x < k_len])
            if len(sxs):
                ax.vlines(sxs, 0, q_len, color=scale_line_color,
                          linestyle=scale_line_style, linewidth=scale_line_width,
                          alpha=scale_line_alpha)
        if scale_shape is not None and len(tuple(scale_shape)) == 3:
            T, Hh, Ww = (int(x) for x in scale_shape)
            hw = Hh * Ww
            if T > 1 and hw > 0:
                fx = int(sink_len) + np.arange(1, T) * hw
                fy = np.arange(1, T) * hw
                ax.vlines(fx[fx < k_len], 0, q_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)
                ax.hlines(fy[fy < q_len], 0, k_len, color=frame_color,
                          linestyle=frame_linestyle, linewidth=frame_linewidth,
                          alpha=frame_alpha)

        ax.set_xlim(0, k_len)
        ax.set_ylim(q_len, 0)
        _apply_boundary_xticks(ax, sink_len=int(sink_len),
                               kv_scale_boundaries=kv_scale_boundaries,
                               kv_len=k_len, tick_fontsize=tick_fontsize,
                               tick_rotation=tick_rotation, font_family=font_family)
        ax.set_title(f"D-CSLA decision colsum (group={rows_per_group} rows)  "
                     f"scale{scale_ind} layer{layer_ind} head{h}",
                     fontsize=title_fontsize, **_ff)
        ax.set_ylabel(f"Query Token Index (S_q = {q_len})",
                      fontsize=axis_label_fontsize, **_ff)
        ax.set_xlabel(f"Key/Value Token Index (S_k = {k_len})",
                      fontsize=axis_label_fontsize, **_ff)

        _save_fig(fig, os.path.join(
            out_dir, f"{tag}_COLSUM_s{scale_ind}_l{layer_ind}_h{h}"
                     f"_Q{q_len}_K{k_len}"))
        plt.close(fig)
