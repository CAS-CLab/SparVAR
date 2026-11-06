import math
import chipmunk
import torch
from einops import rearrange

torch.set_default_device('cuda')
torch.set_default_dtype(torch.bfloat16)

@torch.no_grad()
def naive_dense_colsum_attn(q, k, v, scale=None, q_group_size=192):
    B, H, q_len, D = q.shape
    k_len = k.size(-2)

    scale_factor = 1 / math.sqrt(q.size(-1)) if scale is None else scale

    attn_weight = q @ k.transpose(-2, -1) * scale_factor
    attn_weight = torch.softmax(attn_weight, dim=-1)
    o = attn_weight @ v

    q_groups = (q_len + q_group_size - 1) // q_group_size
    if q_groups * q_group_size != q_len:
        q_pad_len = q_groups * q_group_size - q_len
        attn_pad = F.pad(attn_weight, (0, 0, 0, q_pad_len))
    else:
        attn_pad = attn_weight
    
    attn_g = attn_pad.view(B, H, q_groups, q_group_size, k_len)
    cs = attn_g.sum(dim=3).to(v.dtype)
    
    return o, cs


@torch.no_grad()
def naive_dense_colsum_attn2(
    q: torch.Tensor,                 # [B, qo_heads, q_len, d]
    k: torch.Tensor,                 # [B, kv_heads, kv_len, d]
    v: torch.Tensor,                 # [B, kv_heads, kv_len, d]
    scale: Union[int, float, None] = None,
    q_group_size: int = 192,           # 与 kernel 的 q_g 计算保持一致（FUSE_REDUCE=True）
    out_dtype: Optional[torch.dtype] = None,    # 若为 None，跟随 v.dtype
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    朴素参考实现：
      返回 (o, cs, l_vec)
        o:     [B, qo_heads, q_len, d]，标准密集注意力输出
        cs:    [B, qo_heads, q_groups, kv_len]，按查询组聚合后的列和矩阵
        l_vec: [B, qo_heads, q_len, 1]，每行 softmax 分母的倒数（自然底 e）
    """
    assert q.ndim == k.ndim == v.ndim == 4, "Q/K/V must be [B, H, N, D]"
    B, qo_h, q_len, d = q.shape
    Bk, kv_h, kv_len, dk = k.shape
    Bv, kv_h2, kv_len2, dv = v.shape
    assert B == Bk == Bv, "batch mismatch"
    assert d == dk == dv, "head_dim mismatch"
    assert kv_h == kv_h2, "KV heads mismatch"
    assert q.dtype == k.dtype == v.dtype or True, "dtypes can differ but beware precision"
    assert qo_h % kv_h == 0, "qo_heads must be a multiple of kv_heads"
    hr = qo_h // kv_h

    # 将 K/V 在“组内”复制到 QO heads 维度（广播到 qo_heads）
    # 形状变换: [B, kv_h, kv_len, d] → [B, kv_h, hr, kv_len, d] → [B, qo_h, kv_len, d]
    k_exp = k.unsqueeze(2).expand(B, kv_h, hr, kv_len, d).reshape(B, qo_h, kv_len, d)
    v_exp = v.unsqueeze(2).expand(B, kv_h, hr, kv_len, d).reshape(B, qo_h, kv_len, d)

    # 标准缩放点积注意力（使用 e 底数的稳定 softmax）
    # scores: [B, qo_h, q_len, kv_len]
    sm_scale = 1/math.sqrt(d) if scale is None else scale
    scores = torch.matmul(q.to(torch.float32), k_exp.transpose(-1, -2).to(torch.float32)) / sm_scale

    # log-sum-exp：m, L
    # m = scores.max(dim=-1, keepdim=True).values                          # [B, qo_h, q_len, 1]
    # exps = torch.exp(scores - m)                                         # [B, qo_h, q_len, kv_len]
    # L = exps.sum(dim=-1, keepdim=True)                                   # [B, qo_h, q_len, 1]
    # attn = exps / L                                                      # [B, qo_h, q_len, kv_len]
    attn = torch.softmax(scores, dim=-1)

    # 输出 o
    o = torch.matmul(attn, v_exp.to(torch.float32))                      # [B, qo_h, q_len, d]
    if out_dtype is None:
        out_dtype = v.dtype
    o = o.to(out_dtype)

    # 列和矩阵 cs：按查询组把 attn 沿 q_len 聚合到 q_groups
    q_groups = (q_len + q_group_size - 1) // q_group_size
    if q_groups * q_group_size != q_len:
        # 为了实现方便，可做 0-padding 再重塑；不影响求和（因为 padding 行是 0）
        pad_len = q_groups * q_group_size - q_len
        attn_pad = torch.nn.functional.pad(attn, (0, 0, 0, pad_len))     # pad queries 维
    else:
        attn_pad = attn
    # 现在把 queries 维重塑为 [q_groups, group_size]，再在 group_size 维上求和
    attn_g = attn_pad.view(B, qo_h, q_groups, q_group_size, kv_len)
    cs = attn_g.sum(dim=3)                                               # [B, qo_h, q_groups, kv_len]
    cs = cs.to(out_dtype)  # 与 kernel 保持 bf16/半精输出习惯

    # l_vec：每行 softmax 分母的倒数（自然底 e）
    # Kernel 用的是 base-2 路径，但数值上与 1/sum(exp(scores)) 等价
    # l_vec = (1.0 / (torch.exp(m) * L)).to(torch.float32)                 # [B, qo_h, q_len, 1]
    # return o, cs, l_vec
    return o, cs

for seqlen in range(512, 2048, 64):
    b, h, n, d = 1, 24, seqlen, 128
    n_groups   = (n + 192 - 1) // 192

    q = torch.randn(b, h, n, d)
    k = torch.randn(b, h, n, d)
    v = torch.randn(b, h, n, d)

    indices = torch.arange(n, dtype=torch.int32).repeat((b, h, n_groups, 1)).contiguous()
    counts  = torch.full((b, h, n_groups), n, dtype=torch.int32)

    # ─── Chipmunk kernels ────────────────────────────────────────────────
    o_dense,        lse = torch.ops.chipmunk.dense_attn(q, k, v)
    o_dense_cs, cs, _   = torch.ops.chipmunk.dense_colsum_attn(q, k, v, lse)

    # ─── PyTorch reference output ───────────────────────────────────────
    o_ref = torch.nn.functional.scaled_dot_product_attention(q, k, v)

    # ─── Reference LSE / nv computation ─────────────────────────────────
    softmax_scale = 1.0 / math.sqrt(d)
    logits = (q.float() @ k.float().transpose(-2, -1)) * softmax_scale
    m      = logits.max(dim=-1, keepdim=True).values
    logits -= m
    p      = logits.exp()
    l_ref  = p.sum(dim=-1, keepdim=True)           # softmax denominator
    # cs is the per-192-row-group sum of NORMALIZED softmax probs (the lse2
    # contract makes the kernel exact; the old ref skipped the 1/l factor).
    probs  = p / l_ref
    pad_size = (192 - (p.size(2) % 192)) % 192
    p_padded = torch.nn.functional.pad(probs, (0, 0, 0, pad_size))
    cs_ref = rearrange(p_padded.to(q.dtype), 'b h (m r) n -> b h m r n', r=192).sum(dim=-2).contiguous()

    # --------------------------------------------------------------------

    # diff metrics
    o_max_diff  = (o_dense_cs - o_ref).abs().max()
    o_mean_diff = (o_dense_cs - o_ref).abs().mean()
    cs_mean_diff = (cs - cs_ref).abs().mean()
    cs_max_diff     = (cs  - cs_ref.to(cs.dtype)).abs().max()

    print(
        f"seqlen {seqlen:4d} | "
        f"o_max_diff {o_max_diff:.2e}  o_mean_diff {o_mean_diff:.2e} (o_mean_abs {o_dense_cs.abs().mean():.2e}) | "
        f"cs_max_diff {cs_max_diff:.2e} cs_mean_diff {cs_mean_diff:.2e} (cs_mean_abs {cs.abs().mean():.2e})"
    )
