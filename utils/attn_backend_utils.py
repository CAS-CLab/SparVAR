"""Env-switchable dense self-attention backend shared by the Infinity-family
image models (baseline Infinity / ScaleKV via its patch, SparseVAR, FastVAR,
SkipVAR).

LVAR_ATTN_BACKEND selects the kernel used for the dense (non-flex) BHLc
self-attention path:
  - unset / ''  : keep each call site's historical default (baseline Infinity
                  -> 'chipmunk'; SparseVAR / FastVAR / SkipVAR -> 'torch'),
                  so existing scripts behave exactly as before
  - 'torch'     : torch scaled_dot_product_attention pinned to the FA2 backend
  - 'chipmunk'  : chipmunk TK CUDA dense attention kernel (CUDA-only)

The env value, when set, overrides the default at every call site, so one
variable pins the same kernel stack across baseline and accelerated models
(required for pixel-level metric comparisons; see eval protocol).
"""

import os

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from torch.nn.functional import scaled_dot_product_attention as torch_attn

_VALID_BACKENDS = ('torch', 'chipmunk')
_ENV_BACKEND = os.environ.get('LVAR_ATTN_BACKEND', '').strip().lower()
if _ENV_BACKEND and _ENV_BACKEND not in _VALID_BACKENDS:
    raise ValueError(
        f"LVAR_ATTN_BACKEND={_ENV_BACKEND!r} is invalid; use one of {_VALID_BACKENDS} or unset it"
    )

_printed_backends = set()


def dense_self_attn(q, k, v, scale, attn_mask=None, default_backend='torch'):
    """Dense self-attention on [B, H, L, c] tensors; returns [B, H, L, c].

    Backend = LVAR_ATTN_BACKEND when set, else `default_backend` (each call
    site passes its historical default so unset env keeps legacy behavior).
    """
    backend = _ENV_BACKEND or default_backend
    if backend not in _printed_backends:  # one-time banner per process for eval-log auditing
        _printed_backends.add(backend)
        src = 'env LVAR_ATTN_BACKEND' if _ENV_BACKEND else 'call-site default'
        print(f'[attn-backend] dense self-attn backend = {backend} ({src})')

    if backend == 'chipmunk':
        assert attn_mask is None, 'chipmunk dense_attn does not support attn_mask (training-only path)'
        import chipmunk  # noqa: F401  (registers torch.ops.chipmunk; CUDA-only)
        o, _ = torch.ops.chipmunk.dense_attn(q.to(v.dtype), k.to(v.dtype), v, scale=scale)
        return o
    with sdpa_kernel(SDPBackend.FLASH_ATTENTION):
        return torch_attn(query=q, key=k, value=v, scale=scale, attn_mask=attn_mask, dropout_p=0)
