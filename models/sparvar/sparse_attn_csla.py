"""Static CSLA (block-wise local sparse attention) for SparVAR(Infinity).

Two implementations live side by side:

- ``SparseFlexAttn``  : the conference-era implementation, migrated verbatim from
  ``basic_sparvar.py``. Kept ONLY as an in-checkout numerical A/B reference and
  rollback backup; the inference path does not use it.
- ``StaticCSLAAttn``  : the modernized path used by ``sparvar_model.py``. Same
  block-mask semantics (band OR sink, block-any reduced; bit-locked by
  ``tests/test_csla_static_mask.py``), but:
    * config-driven via the shared ``flex_*`` keys (``flex_backend`` /
      ``flex_block_size`` / ``flex_mode`` / ``flex_dynamic``) and ``csla_scales``;
    * block-level mask construction through the LRU-cached
      ``dcsla_utils.build_image_cross_scale_band_mask`` (no token-level
      ``S_q x S_kv`` materialization at init);
    * H=1 BlockMask (broadcast over heads) instead of the legacy H=num_heads
      replication;
    * compiled-flex singletons shared with the D-CSLA path (no per-instance
      ``torch.compile``);
    * fa4 (CuTe FLASH) backend support with the shared runtime kill-switch;
    * lazy device-agnostic build on the first forward (CPU-importable, no CUDA
      or compile work at construction time).

Visualization: with ``SPARVAR_VIZ_INDEX=1`` the mask is dumped once per scale
on first build — ``csla_BLOCKGRID`` (block-wise) and ``csla_TOKENMASK``
(token-wise fine mask). The static mask is layer-independent, so it is dumped
as layer 0 — image runs need ``SPARVAR_VIZ_SCALES=10,11,12`` (the default
filter is the video set 27,28,29) and ``SPARVAR_VIZ_LAYERS`` must include 0
(the default does).
"""
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.attention.flex_attention import (
    _DEFAULT_SPARSE_BLOCK_SIZE,
    create_block_mask,
    create_mask,
    flex_attention,
)

from models.sparvar import dcsla_utils as DU
from models.sparvar.dcsla_utils import _cdiv
from models.sparvar.sparse_index_viz_utils import (dump_csla_token_mask_viz,
                                                   dump_dcsla_block_mask_viz,
                                                   image_kv_scale_boundaries,
                                                   viz_enabled_for)


# -----------------------
# Modernized static CSLA
# -----------------------

@torch.no_grad()
def build_csla_block_mask_bool(
    scale_schedule,                    # [(h, w)] or [(t, h, w)]; entries [:q_scale_idx+1] used
    q_scale_idx: int,
    win_size,                          # per-historical-scale band widths (attn.win_size)
    attn_sink_scale: int,              # dense sink = tokens of the first N scales
    device: torch.device,
    block_size: int = DU.DCSLA_BLOCK,
) -> torch.Tensor:
    """``[U, V]`` bool block mask of the full static CSLA pattern (band OR sink).

    Exactly the legacy Stage A+B construction: block-any distributes over OR, so
    band and sink can be reduced independently. The sink prefix ``kv_idx <
    boundary`` block-reduces to the first ``ceil(boundary/block_size)`` columns
    fully on (block j is touched iff ``j*block_size < boundary``).
    """
    n_per = [int(s[-2]) * int(s[-1]) for s in scale_schedule[:q_scale_idx + 1]]
    q_len = n_per[-1]
    kv_len = sum(n_per)
    band = DU.build_image_cross_scale_band_mask(
        scale_schedule, int(q_scale_idx), q_len, kv_len, win_size,
        device, block_size=int(block_size))
    sink_boundary = sum(n_per[:attn_sink_scale]) if attn_sink_scale > 0 else 0
    mask = band.clone()                 # the band is LRU-cached; never mutate it
    if sink_boundary > 0:
        mask[:, :_cdiv(sink_boundary, int(block_size))] = True
    return mask


@torch.no_grad()
def build_csla_token_mask_bool(
    scale_schedule,                    # [(h, w)] or [(t, h, w)]; entries [:q_scale_idx+1] used
    q_scale_idx: int,
    win_size,                          # per-historical-scale band widths (attn.win_size)
    attn_sink_scale: int,              # dense sink = tokens of the first N scales
    device: torch.device,
) -> torch.Tensor:
    """``[S_q, S_kv]`` bool TOKEN-level CSLA fine mask (sink prefix OR
    centre-aligned per-scale band) — the pre-block-any mask of the legacy
    ``gen_fine_grained_mask_mod``, vectorized. Visualization only: the
    inference path uses the block-any reduction of exactly this mask
    (``build_csla_block_mask_bool``). ``win_size`` entries < 1 contribute no
    band (``win // 2 == -1`` -> the |center - kv| test is always false)."""
    K = int(q_scale_idx)
    hs = [int(s[-2]) for s in scale_schedule[:K + 1]]
    ws = [int(s[-1]) for s in scale_schedule[:K + 1]]
    n_per = [h * w for h, w in zip(hs, ws)]
    slice_idx = torch.tensor(
        [0] + list(torch.cumsum(torch.tensor(n_per), 0)), device=device, dtype=torch.long)
    S_kv = int(slice_idx[-1])
    H_q, W_q = hs[K], ws[K]
    S_q = H_q * W_q
    sink_boundary = sum(n_per[:attn_sink_scale]) if attn_sink_scale > 0 else 0

    kv = torch.arange(S_kv, device=device)
    l_of = torch.searchsorted(slice_idx, kv, right=True) - 1   # owning scale of each kv token
    H_k = torch.tensor(hs, device=device)[l_of]
    W_k = torch.tensor(ws, device=device)[l_of]
    half = torch.tensor([int(win_size[l]) for l in range(K + 1)], device=device)[l_of] // 2
    rel = kv - slice_idx[l_of]
    kv_y, kv_x = rel // W_k, rel % W_k

    q = torch.arange(S_q, device=device)
    q_y, q_x = q // W_q, q % W_q
    cy = torch.round((q_y.float() / H_q).unsqueeze(1) * H_k.float())   # [S_q, S_kv] centre rows
    cx = torch.round((q_x.float() / W_q).unsqueeze(1) * W_k.float())
    fine = ((cy - kv_y.float()).abs() <= half) & ((cx - kv_x.float()).abs() <= half)
    fine |= (kv < sink_boundary).unsqueeze(0)
    return fine


def mask_uv_to_kv_lists(mask_uv: torch.Tensor):
    """``[U, V]`` bool -> (``kv_indices [1,1,U,V]`` int32, ``kv_num_blocks [1,1,U]`` int32).

    Same layout contract as ``dcsla_utils.select_kv_blocks``: per row, the
    selected block ids form an ascending prefix of a stable descending argsort
    of the bool mask. H=1 broadcasts over heads in the flex kernel (the static
    mask is head-independent, unlike the legacy H=num_heads replication).
    """
    m = mask_uv.unsqueeze(0).unsqueeze(0)                                  # [1,1,U,V]
    kv_num_blocks = m.sum(dim=-1, dtype=torch.int32)
    kv_indices = m.to(torch.int32).argsort(dim=-1, descending=True, stable=True).to(torch.int32)
    return kv_indices, kv_num_blocks


class StaticCSLAAttn:
    """Static CSLA forward with the ``spattn_fn(q, k, v, scale=...)`` call API.

    Plain object (no parameters); everything device-side is built lazily on the
    first forward and cached, so steady-state forwards do zero mask work — the
    property that makes static CSLA the image-side speed upper bound.
    """

    def __init__(self, scale_schedule, q_scale_idx: int, attn_config: dict):
        n_per = [int(s[-2]) * int(s[-1]) for s in scale_schedule[:q_scale_idx + 1]]
        self.scale_schedule = [tuple(int(x) for x in s) for s in scale_schedule]
        self.q_scale_idx = int(q_scale_idx)
        self.S_q = n_per[-1]
        self.S_kv = sum(n_per)
        self.attn_sink_scale = int(attn_config['attn_sink'])
        self.sink_boundary = sum(n_per[:self.attn_sink_scale]) if self.attn_sink_scale > 0 else 0
        self.win_size = list(attn_config['win_size'])
        self.block_size = int(attn_config.get('flex_block_size', DU.DCSLA_BLOCK))
        self.backend = str(attn_config.get('flex_backend', 'triton')).lower()
        self.flex_mode = attn_config.get('flex_mode', 'max-autotune-no-cudagraphs')
        # self.flex_dynamic = bool(attn_config.get('flex_dynamic', True))
        self.flex_dynamic = False
        # lazy state
        self._device = None
        self._bm_tri = None
        self._bm_fa4 = None
        self._padded_kv_len = None

    def _build_kv_lists(self, device):
        mask = build_csla_block_mask_bool(
            self.scale_schedule, self.q_scale_idx, self.win_size,
            self.attn_sink_scale, device, block_size=self.block_size)
        return mask_uv_to_kv_lists(mask)

    def _maybe_dump_viz(self, kv_indices, kv_num_blocks):
        # Static mask is identical across layers: dump once per scale as layer 0.
        if not viz_enabled_for(self.q_scale_idx, 0):
            return
        # Per-scale KV token boundaries after the sink (image stacked-pyramid KV):
        # drawn as divider lines and used as x tick labels in both renders.
        kv_bounds = image_kv_scale_boundaries(
            self.scale_schedule, self.S_kv, self.sink_boundary)
        dump_dcsla_block_mask_viz(
            kv_indices, kv_num_blocks, scale_ind=self.q_scale_idx, layer_ind=0,
            q_len=self.S_q, kv_len=self.S_kv, block_size=self.block_size,
            sink_len=self.sink_boundary, tag="csla", scale_shape=None,
            kv_scale_boundaries=kv_bounds)
        # Token-wise fine mask (CSLA only): the pre-block-any sink|band mask,
        # notebook-style. Built here on demand (viz is opt-in) and discarded.
        dump_csla_token_mask_viz(
            build_csla_token_mask_bool(self.scale_schedule, self.q_scale_idx,
                                       self.win_size, self.attn_sink_scale,
                                       kv_indices.device),
            scale_ind=self.q_scale_idx, layer_ind=0, block_size=self.block_size,
            sink_len=self.sink_boundary, kv_scale_boundaries=kv_bounds,
            tag="csla")

    def _lazy_build(self, device):
        assert self._device is None or str(self._device) == str(device), \
            f'CSLA scale {self.q_scale_idx}: device changed {self._device} -> {device}'
        kv_indices, kv_num_blocks = self._build_kv_lists(device)
        self._maybe_dump_viz(kv_indices, kv_num_blocks)
        if self.backend == 'fa4' and DU.fa4_runtime_ok():
            # Same bucket rule as the image D-CSLA path, so a mixed bound_layer
            # run shares the (device, padded_kv_len) boundary closure and CuTe
            # shape. Each image bucket maps to exactly ONE kv_len (the KV is
            # text-free and fixed per scale), so the shared closure content
            # never conflicts with the cached BlockMask below.
            self._padded_kv_len = _cdiv(self.S_kv, DU._FA4_KV_BUCKET) * DU._FA4_KV_BUCKET
            self._bm_fa4 = DU.build_dcsla_block_mask_fa4(
                kv_indices, kv_num_blocks, self.S_q, self.S_kv,
                padded_kv_len=self._padded_kv_len, block_size=self.block_size)
        else:
            self._bm_tri = DU.build_dcsla_block_mask(
                kv_indices, kv_num_blocks, self.S_q, self.S_kv, self.block_size)
        self._device = device

    def _build_triton_fallback(self):
        # FA4 died at runtime: rebuild the triton-encoded BlockMask once.
        kv_indices, kv_num_blocks = self._build_kv_lists(self._device)
        self._bm_fa4 = None
        self._bm_tri = DU.build_dcsla_block_mask(
            kv_indices, kv_num_blocks, self.S_q, self.S_kv, self.block_size)

    def __call__(self, q, k, v, scale=None):
        assert int(k.shape[-2]) == self.S_kv, \
            f'CSLA scale {self.q_scale_idx}: kv_len {int(k.shape[-2])} != expected {self.S_kv}'
        scale = None if scale is None else float(scale)
        if self._device is None:
            self._lazy_build(q.device)
        if self._bm_fa4 is not None and DU.fa4_runtime_ok():
            try:
                k_f, v_f = k.to(v.dtype), v
                if self._padded_kv_len != self.S_kv:
                    pad = self._padded_kv_len - self.S_kv
                    k_f = F.pad(k_f, (0, 0, 0, pad))
                    v_f = F.pad(v_f, (0, 0, 0, pad))
                return DU._get_compiled_flex_flash()(
                    q.to(v.dtype), k_f, v_f,
                    block_mask=self._bm_fa4, scale=scale, enable_gqa=True)
            except Exception as e:  # noqa: BLE001
                DU.disable_fa4('[CSLA] FA4 backend failed, falling back to triton flex: '
                               + str(e).splitlines()[0][:120])
        if self._bm_tri is None:
            self._build_triton_fallback()
        flex = DU._get_compiled_flex(self.flex_mode, self.flex_dynamic)
        return flex(q.to(v.dtype), k.to(v.dtype), v,
                    block_mask=self._bm_tri, scale=scale, enable_gqa=True)


def build_csla_spattn_list(scale_schedule, attn_config: dict, num_scales: int):
    """``[None, ..., StaticCSLAAttn, ...]`` indexed by scale, per ``csla_scales``."""
    csla_scales = [int(s) for s in attn_config.get('csla_scales', [10, 11, 12])]
    if not csla_scales:
        return [None] * int(num_scales)
    win_size = attn_config['win_size']
    attn_sink = int(attn_config['attn_sink'])
    assert len(win_size) >= max(csla_scales) + 1, \
        f'win_size has {len(win_size)} entries but csla_scales needs {max(csla_scales) + 1}'
    spattn_list = [None] * int(num_scales)
    for si in csla_scales:
        spattn_list[si] = StaticCSLAAttn(scale_schedule, si, attn_config)
    print(f'[w/ Block-wise Local Sparse Attention] backend={attn_config.get("flex_backend", "triton")}')
    print(f"Attn sink scale: Keep the first {attn_sink} scales KV cache")
    for si in csla_scales:
        print(f'Scale-{si} use sparse-attn, local window size is {win_size[si]}')
    print(f'Local window schedule: {list(win_size)}')
    return spattn_list


# -----------------------
# Legacy conference implementation (A/B reference only; NOT used in inference)
# -----------------------

class SparseFlexAttn(nn.Module):
    def __init__(self, scale_schedule, kernel_schedule, q_scale_idx, attn_sink_scale, block_size=_DEFAULT_SPARSE_BLOCK_SIZE, num_heads: int = 16):
        super().__init__()

        S_q = scale_schedule[q_scale_idx][0] * scale_schedule[q_scale_idx][1]
        S_kv = sum([h * w for h, w in scale_schedule[:q_scale_idx + 1]])
        self.block_size = block_size

        self.flex_attention = torch.compile(flex_attention, dynamic=False, mode="max-autotune-no-cudagraphs")

        self.mask_mod = self.create_cross_scale_block_attn_mask(
            scale_schedule=scale_schedule,
            kernel_schedule=kernel_schedule,
            q_scale_idx=q_scale_idx,
            attn_sink_scale=attn_sink_scale
        )

        self.block_mask = create_block_mask(
            self.mask_mod,
            B=1,
            H=num_heads,
            Q_LEN=S_q,
            KV_LEN=S_kv,
            device='cuda',
            BLOCK_SIZE=block_size
        )

    def create_cross_scale_block_attn_mask(
        self,
        scale_schedule=None,
        kernel_schedule=None,
        q_scale_idx: int = 12,
        attn_sink_scale: int = 5,
        device: str = "cuda",
    ):
        """
        一个统一的稀疏 attention mask mod，用于为Infinity模型的完整注意力矩阵
        （例如 4096x10521）进行局部稀疏化。

        该函数能够根据 Key/Value 的一维索引 kv_idx，自动推断其所属的历史尺度、
        该尺度的网格尺寸，以及在该尺度内的局部坐标，然后应用局部注意力逻辑。
        """

        H_q, W_q = scale_schedule[q_scale_idx]
        S_q = H_q * W_q

        # 计算所有历史 Key/Value 尺度的 token 数量和边界
        relevant_schedule = scale_schedule[:q_scale_idx + 1]
        relevant_kernel_schedule = kernel_schedule[:q_scale_idx + 1]

        kv_tokens_per_scale = [h * w for h, w in relevant_schedule]
        kv_cumulative_tokens = np.cumsum(kv_tokens_per_scale)
        kv_slice_indices = np.insert(kv_cumulative_tokens, 0, 0)
        S_k = kv_cumulative_tokens[-1]

        # Calculate the token boundary for the dense attention sink
        attn_sink_boundary = int(kv_slice_indices[attn_sink_scale]) if attn_sink_scale > 0 else 0

        scale_schedule_tensor = torch.tensor(relevant_schedule, device=device, dtype=torch.int)
        kv_slice_indices_tensor = torch.tensor(kv_slice_indices, device=device, dtype=torch.int)
        kernel_schedule_tensor = torch.tensor(relevant_kernel_schedule, device=device, dtype=torch.int)

        def gen_fine_grained_mask_mod(b, h, q_idx, kv_idx):

            # Condition 1: Is the Key/Value token in the dense "attention sink" region?
            # If so, it's always attended to.
            attn_sink_mask = (kv_idx < attn_sink_boundary)

            # --- The rest of the logic is for the sparse region ---
            # 1. 计算 Query 的 2D 坐标 (这部分不变)
            q_y, q_x = q_idx // W_q, q_idx % W_q

            # 2. 核心逻辑: 确定每个 kv_idx 属于哪个历史尺度
            #    torch.searchsorted 是一个高效的查找操作，它会返回每个 kv_idx
            #    应该插入到 slice_indices 中的哪个位置，从而确定其所属的尺度。
            #    我们减去1来获得正确的0-based尺度索引。
            scale_index_k = torch.searchsorted(kv_slice_indices_tensor, kv_idx, right=True) - 1

            # 3. 根据尺度索引，获取该尺度的网格尺寸 (H_k, W_k)
            #    使用 torch.gather 来避免在 vmap 环境中的索引问题
            H_k_tensor = torch.gather(scale_schedule_tensor[:, 0], 0, scale_index_k)
            W_k_tensor = torch.gather(scale_schedule_tensor[:, 1], 0, scale_index_k)

            # 4. 计算 kv_idx 在其自身尺度内的局部一维索引
            start_indices_k = torch.gather(kv_slice_indices_tensor, 0, scale_index_k)
            kernel_size_k = torch.gather(kernel_schedule_tensor, 0, scale_index_k)
            relative_kv_idx = kv_idx - start_indices_k

            # 5. 计算 Key/Value 的 2D 坐标 Calculate Key/Value's 2D coordinates
            kv_y, kv_x = relative_kv_idx // W_k_tensor, relative_kv_idx % W_k_tensor

            # 6. 坐标重缩放，找到 Query 在 Key 网格中的对应中心点 Rescale coordinates
            center_y = torch.round((q_y.float() / H_q) * H_k_tensor)
            center_x = torch.round((q_x.float() / W_q) * W_k_tensor)

            # 7. 判断 Key/Value token 是否在中心点 KERNEL_SIZE//2 的邻域内
                # Check neighborhood using the per-scale kernel size
            half_kernel = kernel_size_k // 2
            vertical_mask = (center_y - kv_y).abs() <= half_kernel
            horizontal_mask = (center_x - kv_x).abs() <= half_kernel
            is_in_sparse_region = vertical_mask & horizontal_mask

            # return vertical_mask & horizontal_mask
            # A token is attended to if it's in the dense region OR in the local sparse region.
            return attn_sink_mask | is_in_sparse_region

        # print("Step 1: Generating internal fine-grained mask...")
        fine_grained_mask = create_mask(gen_fine_grained_mask_mod, 1, 1, S_q, S_k, device=device).squeeze()

        # --- Step 2: Convert to a Pure Block-Sparse Mask ---
        # print(f"Step 2: Converting to {block_size}x{block_size} block-sparse format...")
        # Pad the fine-grained mask to be divisible by block_size
        pad_q = (self.block_size - S_q % self.block_size) % self.block_size
        pad_k = (self.block_size - S_k % self.block_size) % self.block_size
        padded_mask = F.pad(fine_grained_mask, (0, pad_k, 0, pad_q), "constant", False)

        padded_S_q, padded_S_k = padded_mask.shape
        num_q_blocks = padded_S_q // self.block_size
        num_k_blocks = padded_S_k // self.block_size

        # Reshape, permute, and reduce to get the block-level mask
        # The logic is: if ANY value in a block is True, the whole block becomes True.
        block_level_mask = (
            padded_mask.reshape(num_q_blocks, self.block_size, num_k_blocks, self.block_size)
            .permute(0, 2, 1, 3)
            .any(dim=(-1, -2))
        )

        # --- Step 3: Return a Simple Lookup Function ---
        # print("Step 3: Returning simplified block-lookup mask function.")

        # This closure captures the pre-computed block_level_mask
        def final_block_mask_mod(b, h, q_idx, kv_idx):
            q_block_idx = q_idx // self.block_size
            k_block_idx = kv_idx // self.block_size
            # This is now a simple, fast lookup
            return block_level_mask[q_block_idx, k_block_idx]

        return final_block_mask_mod

    def forward(self, q, k, v, scale=None):
        oup = self.flex_attention(
            q.to(v.dtype), k.to(v.dtype), v,
            block_mask=self.block_mask,
            scale=scale
        )

        return oup
