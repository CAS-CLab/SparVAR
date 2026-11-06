"""CPU recurrence tests; no CUDA, Triton or model weights are required."""

import math

import pytest
import torch

from models.sparvar.image_identity_utils import (
    CumulativeImageIdentityIndices,
    _union_count_bound,
    validate_identity_cumulative_config,
)
from models.sparvar.index_map_utils import mask_to_indices_torch


SIDES = [1, 2, 4, 6, 8, 12, 16, 20, 24, 32, 40, 48, 64]


def _as_mask(indices, counts, kv_len):
    result = torch.zeros(*indices.shape[:3], kv_len, dtype=torch.bool)
    for b in range(indices.shape[0]):
        for h in range(indices.shape[1]):
            for g in range(indices.shape[2]):
                result[b, h, g, indices[b, h, g, :int(counts[b, h, g])].long()] = True
    return result


def _reference(previous, *, side, kv_len, bm, win, sink):
    """Explicit group/member-query window enumeration, independent of the fast path."""
    B, H, Gs, _ = previous.shape
    G = math.ceil(side * side / bm)
    out = torch.zeros(B, H, G, kv_len, dtype=torch.bool)
    for g in range(G):
        src = min(Gs - 1, max(0, round((g + 0.5) / G * Gs - 0.5)))
        out[:, :, g, :previous.shape[-1]] = previous[:, :, src]
        out[:, :, g, :sink] = True
        for q in range(g * bm, min((g + 1) * bm, side * side)):
            y, x = divmod(q, side)
            for yy in range(max(0, y - win // 2), min(side, y + win // 2 + 1)):
                for xx in range(max(0, x - win // 2), min(side, x + win // 2 + 1)):
                    out[:, :, g, kv_len - side * side + yy * side + xx] = True
    return out


@pytest.mark.parametrize('provider', ['triton', 'cuda'])
@pytest.mark.parametrize('sink', [0, 121])
def test_s10_s11_s12_inherits_counts_valid_band_and_alignment(provider, sink):
    bm = 192
    multiple, pad = (112, 4) if provider == 'cuda' else (64, 1)
    K10 = sum(s * s for s in SIDES[:11])
    seed_count = multiple * round(0.2 * K10 / multiple)
    # Different rows/heads/batches, and enough gaps to exercise count padding.
    torch.manual_seed(123)
    seed = torch.stack([torch.randperm(K10)[:seed_count] for _ in range(2 * 2 * 9)])
    seed = seed.reshape(2, 2, 9, seed_count).int()
    state = CumulativeImageIdentityIndices()
    state.reset(seed, scale_ind=10, kv_len=K10)
    previous = _as_mask(seed, torch.full((2, 2, 9), seed_count), K10)

    for si, win, batch in [(11, 5, 2), (12, 7, 1)]:
        side = SIDES[si]
        kv_len = sum(s * s for s in SIDES[:si + 1])
        expected = _reference(previous[:batch], side=side, kv_len=kv_len,
                              bm=bm, win=win, sink=sink)
        expected_indices, expected_counts = mask_to_indices_torch(expected, multiple, pad)
        actual_indices, actual_counts = state.advance(
            scale_ind=si, batch_size=batch, q_len=side * side, kv_len=kv_len,
            scale_shape=(1, side, side), sink_len=sink, bm=bm, win_size=win,
            multiple_of=multiple, indices_pad_to=pad, provider=provider)
        assert torch.equal(actual_counts, expected_counts)
        actual_mask = _as_mask(actual_indices, actual_counts, kv_len)
        assert torch.equal(actual_mask, _as_mask(expected_indices, expected_counts, kv_len))
        assert actual_indices.stride(2) % pad == 0
        if provider == 'cuda':
            assert actual_indices.shape[-1] < kv_len
        previous = actual_mask
        # CUDA packing leaves tails uninitialized. Poison them to detect accidental
        # inheritance, including rows shorter than other heads' selected prefixes.
        valid = torch.arange(state.indices.shape[-1]) < actual_counts.unsqueeze(-1)
        state.indices.masked_fill_(~valid, -123456)

    # Scale-11's band is retained at scale 12; it is not a decision-only replay.
    assert previous[..., K10:K10 + SIDES[11] ** 2].any()


def test_reset_and_scale_order():
    state = CumulativeImageIdentityIndices()
    args = dict(scale_ind=2, batch_size=1, q_len=16, kv_len=21,
                scale_shape=(1, 4, 4), sink_len=0, bm=4, win_size=1,
                multiple_of=1, indices_pad_to=4, provider='triton')
    with pytest.raises(RuntimeError, match='decision'):
        state.advance(**args)
    state.reset(torch.tensor([[[[0]]]], dtype=torch.int32), scale_ind=1, kv_len=5)
    state.advance(**args)
    with pytest.raises(ValueError, match='increasing'):
        state.advance(**args)
    # A re-decision/new image discards all prior state and permits a larger batch.
    seed = torch.full((2, 1, 1, 1), 4, dtype=torch.int32)
    state.reset(seed, scale_ind=1, kv_len=5)
    got, cnt = state.advance(**args)
    mask = _as_mask(got, cnt, 21)
    assert mask[..., 4].all()
    assert not mask[..., :4].any()
    state.clear()
    assert state.indices is None and state.counts is None


def test_compact_bound_covers_rectangular_and_wrapping_groups():
    for h, w in [(3, 5), (8, 7), (48, 48)]:
        for bm in [1, 7, 192]:
            for win in [1, 3, 7]:
                # Independent direct window count for rectangular grids.
                maximum = 0
                for start in range(0, h * w, bm):
                    selected = {4}
                    for q in range(start, min(start + bm, h * w)):
                        y, x = divmod(q, w)
                        selected.update(5 + yy * w + xx
                                        for yy in range(max(0, y-win//2), min(h, y+win//2+1))
                                        for xx in range(max(0, x-win//2), min(w, x+win//2+1)))
                    maximum = max(maximum, len(selected))
                bound = _union_count_bound(1, 0, h*w, (1,h,w), bm, win, h*w+5, 4, 4)
                assert bound >= math.ceil(maximum / 4) * 4


def test_incompatible_mapping_is_rejected():
    cfg = dict(cs4a_image_expand_mode='center_union', cs4a_image_kv_map='identity')
    validate_identity_cumulative_config(cfg)
    for key, value in [('cs4a_image_expand_mode', 'footprint'),
                       ('cs4a_image_kv_map', 'relative'),
                       ('cs4a_qgroup_map_mode', 'index_linear'),
                       ('dcsla_enabled', True)]:
        with pytest.raises(ValueError, match='requires'):
            validate_identity_cumulative_config({**cfg, key: value})


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA/chipmunk integration')
def test_cuda_pack_and_count_aware_scatter(monkeypatch):
    import models.sparvar.index_map_utils as mapping
    from kernels.sparse_attn.sparse_attn_config import GLOBAL_CONFIG
    from kernels.sparse_attn.sparse_attn_ops import csp_attn

    if not mapping._USE_CUDA_MASK_TO_INDICES:
        pytest.skip('Native chipmunk mask_to_indices is not installed')
    monkeypatch.setitem(GLOBAL_CONFIG['attn'], 'provider', 'cuda')
    K10 = sum(s*s for s in SIDES[:11])
    seed = torch.arange(784, dtype=torch.int32).expand(2, 2, 9, 784).clone()
    state = CumulativeImageIdentityIndices()
    state.reset(seed.cuda(), scale_ind=10, kv_len=K10)
    previous = _as_mask(seed, torch.full((2, 2, 9), 784), K10)
    for si, win, batch in [(11, 5, 2), (12, 7, 1)]:
        side = SIDES[si]
        kv_len = sum(s*s for s in SIDES[:si+1])
        expected = _reference(previous[:batch], side=side, kv_len=kv_len,
                              bm=192, win=win, sink=121)
        ref, ref_counts = mask_to_indices_torch(expected, 112, 4)
        got, counts = state.advance(
            scale_ind=si, batch_size=batch, q_len=side*side, kv_len=kv_len,
            scale_shape=(1, side, side), sink_len=121, bm=192, win_size=win,
            multiple_of=112, indices_pad_to=4, provider='cuda')
        assert torch.equal(counts.cpu(), ref_counts)
        previous = _as_mask(got.cpu(), counts.cpu(), kv_len)
        assert torch.equal(previous, _as_mask(ref, ref_counts, kv_len))
        assert got.shape[-1] < kv_len and got.stride(2) % 4 == 0
        # Native CSP must consume compact rows exactly like full-width rows.
        full, full_counts = mapping._mask_to_indices(expected.cuda(), 112, 4)
        full = full[..., :kv_len]
        q = torch.randn(batch, 2, side*side, 128, device='cuda', dtype=torch.bfloat16)
        k = torch.randn(batch, 2, kv_len, 128, device='cuda', dtype=torch.bfloat16)
        v = torch.randn_like(k)
        for with_cache in (False, True):
            cached = torch.randn_like(q) if with_cache else None
            actual = csp_attn(q, k, v, 128**-0.5, got, counts,
                              cached.clone() if with_cache else None, 1)
            reference = csp_attn(q, k, v, 128**-0.5, full, full_counts,
                                 cached.clone() if with_cache else None, 1)
            torch.cuda.synchronize()
            assert actual.isfinite().all()
            assert torch.equal(actual, reference)
        print(f'[identity CUDA] scale={si} batch={batch} width={got.shape[-1]}/{kv_len} '
              'mask=exact compact_CSP=bitwise_equal (cache on/off)')
        valid = torch.arange(state.indices.shape[-1], device='cuda') < counts.unsqueeze(-1)
        state.indices.masked_fill_(~valid, -123456)


@pytest.mark.parametrize('use_o_cache', [False, True])
@pytest.mark.parametrize('enabled', [False, True])
def test_image_dispatch_keeps_decision_storage_separate(use_o_cache, enabled):
    # Execute the real Python dispatch method with CPU kernel/storage doubles.
    # Extracting its AST avoids importing unavailable CUDA/FlashAttention modules.
    import ast
    from contextlib import nullcontext
    from pathlib import Path
    from types import MethodType, SimpleNamespace

    import models.sparvar.index_map_utils as mapping

    path = Path(__file__).resolve().parents[1] / 'models/sparvar/sparse_attn.py'
    tree = ast.parse(path.read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == 'SparseDiffAttn')
    names = {'_as_scale_shape', '_target_indices_count', '_match_batch',
             '_uniform_counts_like', '_pad_uniform_indices'}
    nodes = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    nodes += [n for n in cls.body if isinstance(n, ast.FunctionDef)
              and n.name == '_fast_attention_image']
    config = dict(first_n_dense_layers=0, cs4a_identity_cumulative=enabled, cs4a_image_expand_mode='center_union',
                  cs4a_image_kv_map='identity', cs4a_sink_scales=1,
                  cs4a_ws=[1, 1, 3, 3], top_keys=0.2, provider='triton')

    class Storage:
        def __init__(self):
            self.loads = 0

        def load_async(self):
            self.loads += 1

        def load_async_wait(self):
            pass

        def set_indices(self, x):
            self.indices = x

        def set_counts(self, x):
            self.counts = x

        def set_out_cache(self, x):
            self.out_cache = x

        def get_indices(self):
            return self.indices

        def get_counts(self):
            return self.counts

    packed_calls = []

    def csp(q, k, v, scale, inds, counts, o, sign):
        packed_calls.append((inds.clone(), counts.clone()))
        return torch.zeros_like(q)

    timer = SimpleNamespace(mark=lambda *a: None, stop=lambda *a: None)
    ns = dict(torch=torch, Tensor=torch.Tensor, Tuple=tuple,
              triton=SimpleNamespace(cdiv=lambda a, b: math.ceil(a/b)),
              GLOBAL_CONFIG={'attn': config},
              get_kernel_config_attn=lambda: dict(bm=4, counts_multiple_of=1, indices_pad_to=1),
              validate_identity_cumulative_config=validate_identity_cumulative_config,
              lprof=SimpleNamespace(region=lambda *a, **k: nullcontext(),
                                    timer=lambda *a, **k: timer, lazy=lambda f: f),
              csp_attn=csp, check_csp_attn_inputs=lambda *a, **k: None,
              viz_enabled_for=lambda *a: False, maybe_dump_csp_pattern=lambda *a, **k: None,
              map_inds_counts_across_scales=mapping.map_inds_counts_across_scales,
              build_image_local_union_indices=mapping.build_image_local_union_indices)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(path), 'exec'), ns)
    module = SimpleNamespace(use_o_cache=use_o_cache, storage=Storage(),
                             _image_identity_indices=CumulativeImageIdentityIndices(),
                             _decision_inds=None, _decision_indices_count=None)
    module._fast_attention_image = MethodType(ns['_fast_attention_image'], module)
    cache_loads = []
    module._load_upsampled_o_cache = lambda *a, **k: cache_loads.append(module.storage.out_cache)
    module.dense_colsum_attn = lambda q, k, v, scale: (
        torch.zeros_like(q), torch.arange(k.shape[-2], dtype=torch.float32).expand(
            q.shape[0], 1, math.ceil(q.shape[-2]/4), k.shape[-2]))
    schedule = [(1, s, s) for s in [1, 3, 5, 7]]
    decision_snapshot = None
    previous = None
    for si in [1, 2, 3]:
        q_len = math.prod(schedule[si])
        kv_len = sum(math.prod(s) for s in schedule[:si+1])
        q = torch.zeros(1, 1, q_len, 2)
        kv = torch.zeros(1, 1, kv_len, 2)
        module._fast_attention_image(q, kv, kv, 1.0, 0, si == 1, si, 0, schedule)
        if si == 1:
            seed = module.storage.indices if use_o_cache else module._decision_inds
            counts = module.storage.counts if use_o_cache else torch.full(seed.shape[:3], 2)
            decision_snapshot = seed.clone()
            previous = decision_mask = _as_mask(seed, counts, kv_len)
        else:
            expected = _reference(previous if enabled else decision_mask, side=[1,3,5,7][si],
                                  kv_len=kv_len, bm=4, win=3, sink=1)
            got = _as_mask(*packed_calls[-1], kv_len)
            assert torch.equal(got, expected)
            previous = got
            seed = module.storage.indices if use_o_cache else module._decision_inds
            assert torch.equal(seed, decision_snapshot)
    if use_o_cache:
        assert len(cache_loads) == 2 and cache_loads[0] is cache_loads[1]
        assert module.storage.loads == 2
    elif enabled:
        assert module.storage.loads == 0
