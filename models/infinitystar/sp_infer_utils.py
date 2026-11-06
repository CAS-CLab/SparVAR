# Copyright (c) 2025 FoundationVision
# SPDX-License-Identifier: MIT

"""
Helpers to run InfinityStar autoregressive inference under Ulysses sequence
parallelism (SP) across multiple GPUs, while keeping Flash-Attention and
producing results numerically identical to single-GPU inference.

Design — "phase-switched SP" (see docs/infinitystar_sp_inference_plan.md):

* Phase 1 (text `t0` + image pyramid, ``si < scales_in_one_clip``): SP is OFF.
  Every rank runs this phase replicated (same weights / inputs / seeds -> bit
  identical), so the KV cache is stored with full heads. These scales are tiny
  and fit easily even when replicated.
* Boundary (first video-tower scale): convert every already-cached KV entry from
  full-head to this rank's head shard, then enable SP.
* Phase 2 (video tower, ``si >= scales_in_one_clip``): SP is ON. The per-scale
  sequence length is large and always even (``pt`` is even), so it shards
  cleanly with no padding. This is where the memory peak lives, and SP halves
  the activation + KV-cache footprint here.

All entry points are no-ops when SP is not initialized (``sp_size <= 1``), so the
single-GPU path is byte-for-byte unchanged.
"""

import os

import torch
import torch.distributed as dist

from utils.sequence_parallel import (
    SequenceParallelManager as sp_manager,
    sp_split_sequence_by_dim,
    sp_gather_sequence_by_dim,
)


def sp_debug_log(tag, x=None):
    """Per-scale diagnostic; enable with env LVAR_SP_DEBUG=1. No-op otherwise."""
    if not os.environ.get('LVAR_SP_DEBUG'):
        return
    rank = dist.get_rank() if dist.is_initialized() else 0
    msg = f'[SP_DEBUG r{rank}] {tag}'
    if x is not None:
        msg += f' shape={tuple(x.shape)} contig={x.is_contiguous()} dev={x.device} dtype={x.dtype}'
    if torch.cuda.is_available():
        msg += (f' | mem={torch.cuda.memory_allocated()/1e9:.1f}G'
                f' max={torch.cuda.max_memory_allocated()/1e9:.1f}G'
                f' resv={torch.cuda.memory_reserved()/1e9:.1f}G sp_on={sp_manager.sp_on()}')
    print(msg, flush=True)


def sp_inference_enabled() -> bool:
    """True iff an SP process group was initialized (launched with sp_size > 1)."""
    return sp_manager.get_sp_group() is not None


def _true_sp_rank():
    """This rank's position within its SP group, valid even while SP is toggled
    inactive. NOTE: sp_manager.get_sp_rank() returns 0 whenever sp_on() is False,
    so it must NOT be used at the boundary (SP not yet enabled) — that would make
    every rank shard to head block 0 and corrupt half the heads."""
    return dist.get_rank() % sp_manager.get_sp_size()


def sp_selftest():
    """Decisive check that the SP collectives actually move data across ranks.

    Run via LVAR_SP_DEBUG=1. Catches the failure mode where all_to_all / gather
    silently return the input or uninitialized memory (custom-op / stream-sync
    issues on bleeding-edge torch), which makes the SP video tower produce
    garbage while the replicated image pyramid still looks fine.
    """
    if not sp_inference_enabled():
        return
    from utils.comm.comm import all_to_all_comm
    grank = dist.get_rank()
    sp_rank = _true_sp_rank()   # NOT get_sp_rank(): SP is inactive here so it would return 0
    size = sp_manager.get_sp_size()
    grp = sp_manager.get_sp_group()
    dev = torch.cuda.current_device()

    # (a) split -> gather round trip must recover a deterministic full tensor.
    L = 4 * size
    full = torch.arange(L, device=dev, dtype=torch.float32).reshape(1, L, 1)
    recov = sp_gather_sequence_by_dim(sp_split_sequence_by_dim(full, 1), 1)
    sg_ok = bool(torch.equal(recov, full))

    # (b) all_to_all (heads<->seq, exactly as attention uses it) must MIX data
    #     across ranks: every rank's id should appear in the output.
    H, Lloc = 2 * size, 3
    x = torch.full((1, H, Lloc, 1), float(sp_rank), device=dev)
    y = all_to_all_comm(x, grp, scatter_dim=1, gather_dim=2)
    uniq = sorted(int(v) for v in torch.unique(y).tolist())
    a2a_ok = (uniq == list(range(size)))

    print(f'[SP_SELFTEST g{grank} sp{sp_rank}/{size}] split_gather_ok={sg_ok} '
          f'a2a_ok={a2a_ok} a2a_out{tuple(y.shape)} a2a_uniq={uniq}', flush=True)


def sp_begin_inference(g_seed):
    """Call once at the very start of AR inference.

    Keeps SP OFF for the replicated image-pyramid phase and synchronizes the
    *global* torch RNG across ranks, so the replicated phase (``torch.randn``
    noise) and the sampling phase stay in lockstep. ``self.rng`` (the multinomial
    generator) is seeded with the same ``g_seed`` by the caller.
    """
    if not sp_inference_enabled():
        return
    assert g_seed is not None, (
        "SP inference requires a fixed g_seed so per-rank RNG (noise + sampling) "
        "stays in lockstep; got g_seed=None."
    )
    sp_manager.set_active(False)
    if os.environ.get('LVAR_SP_DEBUG'):
        sp_selftest()
    torch.manual_seed(g_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(g_seed)


def sp_enter_video_tower(blocks):
    """Call once, right before the first video-tower scale's block loop.

    Head-shards every already-cached (full-head) KV entry so it can be
    concatenated with the head-sharded KV produced inside the SP attention path,
    then enables SP. Idempotent and a no-op when SP is unavailable / already on.
    """
    if not sp_inference_enabled() or sp_manager.sp_on():
        return
    # Use the true sp_rank: SP is still inactive here, so get_sp_rank() would return 0.
    shard_kv_cache_heads_(blocks, _true_sp_rank(), sp_manager.get_sp_size())
    sp_manager.set_active(True)


def shard_kv_cache_heads_(blocks, sp_rank, sp_size):
    """In-place: slice each block's cached_k / cached_v along the head dim.

    Cached entries from the replicated phase have shape ``[B, H_kv, L, D]``. The
    head assignment matches ``_all_to_all_func`` (scatter over the head dim):
    rank ``r`` holds heads ``[r*H_kv/sp : (r+1)*H_kv/sp]``. This keeps historical
    KV consistent with the head-sharded KV produced by the SP attention path, so
    the per-scale ``torch.cat`` over reference scales stays well-formed.
    """
    for b in blocks:
        attn = b.attn
        n_kv = attn.num_key_value_heads // sp_size
        lo, hi = sp_rank * n_kv, (sp_rank + 1) * n_kv
        for cache in (attn.cached_k, attn.cached_v):
            for sid, t in list(cache.items()):
                if t is not None:
                    cache[sid] = t[:, lo:hi].contiguous()


def sp_split_hidden(x):
    """``[B, L, C] -> [B, L/sp, C]`` along the sequence dim; no-op when SP is off.

    ``L`` must be divisible by ``sp_size`` (guaranteed for the video tower, where
    ``pt`` is even). Call right before the per-scale block loop.
    """
    if not sp_manager.sp_on():
        return x
    return sp_split_sequence_by_dim(x, 1)


def sp_gather_hidden(x):
    """``[B, L/sp, C] -> [B, L, C]`` along the sequence dim; no-op when SP is off.

    Call right after the per-scale block loop, before logits / sampling, so the
    full sequence is available identically on every rank.
    """
    if not sp_manager.sp_on():
        return x
    return sp_gather_sequence_by_dim(x, 1)


def sp_broadcast_(tensor):
    """Broadcast ``tensor`` in-place from the SP group's rank-0 to all members.

    Optional safety net (``--sp_bcast_sample``) that removes any residual RNG
    drift on sampled tokens. No-op when SP is unavailable / inactive.
    """
    if not sp_manager.sp_on():
        return tensor
    # global rank of sp_rank==0 within this rank's SP group
    src = dist.get_rank() - _true_sp_rank()
    dist.broadcast(tensor, src=src, group=sp_manager.get_sp_group())
    return tensor
