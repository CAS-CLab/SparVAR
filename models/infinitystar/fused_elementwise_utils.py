"""Bit-exact Triton fusions of the memory-bound elementwise ops in the InfinityStar block.

Both kernels reproduce the eager PyTorch rounding sequence step by step (fp32 opmath,
round-to-nearest-even back to the tensor dtype at every point where eager materializes a
tensor), so their outputs are bit-identical to the eager ops they replace; they only
remove the intermediate global-memory round trips. Verified by
kernels/tests/test_fused_elementwise_bitexact.py (remote GPU).

  * fused_silu_mul(gate, up)  == F.silu(gate) * up            (Qwen3MLP activation)
        eager: silu = bf16(x / (1 + exp(-x))) ; out = bf16(silu * up)
  * fused_rope_inplace(q, k, rope_cache) == models.infinitystar.rope.apply_rotary_emb
        eager (per interleaved pair x0, x1; c = cos, s = sin, fp32):
          tmp1 = x1 * s ; tmp2 = x0 * s                    (fp32 temporaries)
          x0 = dt(dt(x0 * c) - tmp1) ; x1 = dt(dt(x1 * c) + tmp2)    dt = round to x dtype

Enable with INFINITYSTAR_FUSED_SWIGLU=1 / INFINITYSTAR_FUSED_ROPE=1 (default off). Falls
back to the eager ops when Triton is unavailable or the layout is not supported.
"""
import os

import torch
import torch.nn.functional as F

try:
    import triton
    import triton.language as tl
    from triton.language.extra import libdevice
    _HAS_TRITON = True
except Exception:  # noqa: BLE001
    _HAS_TRITON = False

_FUSED_SWIGLU = os.environ.get('INFINITYSTAR_FUSED_SWIGLU', '0').lower() in ('1', 'true', 'yes', 'on')
_FUSED_ROPE = os.environ.get('INFINITYSTAR_FUSED_ROPE', '0').lower() in ('1', 'true', 'yes', 'on')

if _FUSED_SWIGLU or _FUSED_ROPE:
    # announce at import so a run log records which fusions were active
    print(f'[infinitystar-fused-elementwise] enabled: swiglu={_FUSED_SWIGLU and _HAS_TRITON} '
          f'rope={_FUSED_ROPE and _HAS_TRITON} (triton available: {_HAS_TRITON})')


def fused_swiglu_enabled() -> bool:
    return _FUSED_SWIGLU and _HAS_TRITON


def fused_rope_enabled() -> bool:
    return _FUSED_ROPE and _HAS_TRITON


if _HAS_TRITON:

    @triton.jit
    def _silu_mul_kernel(g_ptr, u_ptr, o_ptr, n_elements, BLOCK: tl.constexpr):
        pid = tl.program_id(0)
        offs = pid.to(tl.int64) * BLOCK + tl.arange(0, BLOCK)
        mask = offs < n_elements
        g = tl.load(g_ptr + offs, mask=mask, other=0.0)
        u = tl.load(u_ptr + offs, mask=mask, other=0.0)
        gf = g.to(tl.float32)
        # torch silu: x / (1 + exp(-x)) in fp32 with IEEE division (div.rn), then rounded to
        # the tensor dtype because eager materializes the silu output
        s = libdevice.div_rn(gf, 1.0 + libdevice.exp(-gf))
        s = s.to(g.dtype).to(tl.float32)
        o = (s * u.to(tl.float32)).to(g.dtype)     # plain mul: matches eager incl. denormal flush
        tl.store(o_ptr + offs, o, mask=mask)

    @triton.jit
    def _mul_rn_ftz(a, b):
        return tl.inline_asm_elementwise("mul.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _add_rn_ftz(a, b):
        return tl.inline_asm_elementwise("add.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _sub_rn_ftz(a, b):
        return tl.inline_asm_elementwise("sub.rn.f32 $0, $1, $2;", "=r,r,r", [a, b],
                                         dtype=tl.float32, is_pure=True, pack=1)

    @triton.jit
    def _rope_inplace_kernel(x_ptr, cos_ptr, sin_ptr,
                             stride_b, stride_h, stride_l,
                             H, L, D: tl.constexpr, BLOCK_L: tl.constexpr):
        # one program: (b, h) pair x BLOCK_L rows. The last dim (D, contiguous) holds
        # interleaved (x0, x1) pairs; cos/sin are [L, D/2] fp32 (broadcast over b, h).
        # Products/sums are explicit mul/add/sub.rn.ftz PTX so ptxas cannot contract
        # mul+sub into an FMA (eager PyTorch rounds every op separately, no FTZ).
        pid_bh = tl.program_id(0)
        pid_l = tl.program_id(1)
        b = pid_bh // H
        h = pid_bh % H
        rows = pid_l * BLOCK_L + tl.arange(0, BLOCK_L)
        row_mask = rows < L
        d = tl.arange(0, D)
        base = x_ptr + b.to(tl.int64) * stride_b + h.to(tl.int64) * stride_h
        ptrs = base + rows[:, None].to(tl.int64) * stride_l + d[None, :]
        m2 = row_mask[:, None] & (d[None, :] < D)
        x = tl.load(ptrs, mask=m2, other=0.0)                       # [BLOCK_L, D] coalesced
        x0, x1 = tl.split(tl.reshape(x, (BLOCK_L, D // 2, 2)))       # even / odd lanes
        j = tl.arange(0, D // 2)
        cs_off = rows[:, None].to(tl.int64) * (D // 2) + j[None, :]
        mc = row_mask[:, None] & (j[None, :] < D // 2)
        c = tl.load(cos_ptr + cs_off, mask=mc, other=0.0)
        s = tl.load(sin_ptr + cs_off, mask=mc, other=0.0)
        x0f = x0.to(tl.float32)
        x1f = x1.to(tl.float32)
        tmp1 = _mul_rn_ftz(x1f, s)
        tmp2 = _mul_rn_ftz(x0f, s)
        a0 = _mul_rn_ftz(x0f, c).to(x0.dtype).to(tl.float32)
        a1 = _mul_rn_ftz(x1f, c).to(x1.dtype).to(tl.float32)
        y0 = _sub_rn_ftz(a0, tmp1).to(x0.dtype)
        y1 = _add_rn_ftz(a1, tmp2).to(x1.dtype)
        y = tl.reshape(tl.join(y0, y1), (BLOCK_L, D))
        tl.store(ptrs, y, mask=m2)


def fused_silu_mul(gate: torch.Tensor, up: torch.Tensor) -> torch.Tensor:
    """== F.silu(gate) * up, bit-exact, one memory pass."""
    if not (fused_swiglu_enabled() and gate.is_cuda and gate.dtype == up.dtype
            and gate.dtype in (torch.bfloat16, torch.float16, torch.float32)
            and gate.is_contiguous() and up.is_contiguous() and gate.shape == up.shape):
        return F.silu(gate) * up
    out = torch.empty_like(gate)
    n = gate.numel()
    if n == 0:
        return out
    BLOCK = 2048
    grid = (triton.cdiv(n, BLOCK),)
    _silu_mul_kernel[grid](gate, up, out, n, BLOCK=BLOCK, num_warps=8)
    return out


def _rope_layout_ok(x: torch.Tensor, L: int, HALF: int) -> bool:
    """x [B, H, L, D] with D contiguous (any batch/head/row strides), D = 2*HALF, HALF a power of two."""
    if x.dim() != 4 or x.stride(-1) != 1 or x.shape[-2] != L or x.shape[-1] != 2 * HALF:
        return False
    if HALF > 256 or HALF & (HALF - 1):          # tl.arange needs a power of two
        return False
    return x.dtype in (torch.bfloat16, torch.float16, torch.float32)


def _rope_apply_one(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor):
    """In-place rotary on x [B, H, L, D]; cos/sin: [L, D/2] fp32 contiguous."""
    B, H, L, D = x.shape
    BLOCK_L = 64
    grid = (B * H, triton.cdiv(L, BLOCK_L))
    _rope_inplace_kernel[grid](x, cos, sin, x.stride(0), x.stride(1), x.stride(2), H, L,
                               D=D, BLOCK_L=BLOCK_L, num_warps=4)


def fused_rope_inplace(q: torch.Tensor, k: torch.Tensor, rope_cache: torch.Tensor):
    """== models.infinitystar.rope.apply_rotary_emb(q, k, rope_cache) (in place, returns q, k)."""
    from models.infinitystar.rope import apply_rotary_emb
    if not (fused_rope_enabled() and q.is_cuda):
        return apply_rotary_emb(q, k, rope_cache)
    rc = rope_cache[:, 0]                       # (2, 1, 1, L, D/2): cos = rc[0], sin = rc[1]
    if rc.dim() != 5 or rc.shape[1] != 1 or rc.shape[2] != 1 or rc.dtype != torch.float32:
        return apply_rotary_emb(q, k, rope_cache)
    L, HALF = int(rc.shape[-2]), int(rc.shape[-1])
    if not (_rope_layout_ok(q, L, HALF) and _rope_layout_ok(k, L, HALF)):
        return apply_rotary_emb(q, k, rope_cache)
    cos = rc[0].reshape(L, HALF).contiguous()
    sin = rc[1].reshape(L, HALF).contiguous()
    _rope_apply_one(q, cos, sin)
    _rope_apply_one(k, cos, sin)
    return q, k
