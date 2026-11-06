"""CPU regression tests for CSLA geometry, layout dispatch and preparation."""

import ast
from pathlib import Path
import runpy
from types import SimpleNamespace

import pytest
import torch
import torch.nn.functional as F


ROOT = Path(__file__).resolve().parents[1]
UTILS = runpy.run_path(str(ROOT / "models/sparvar/hart_csla_utils.py"))


def _load_classes(path, namespace):
    tree = ast.parse((ROOT / path).read_text())
    classes = [node for node in tree.body if isinstance(node, ast.ClassDef)]
    exec(compile(ast.Module(body=classes, type_ignores=[]), path, "exec"), namespace)
    return namespace


def _reference_mask(patch_nums, context, query_scale, sink, windows, block_size):
    side = patch_nums[query_scale]
    lengths = [context] + [pn ** 2 for pn in patch_nums[1:query_scale + 1]]
    mask = torch.zeros(side ** 2, sum(lengths), dtype=torch.bool)
    for q in range(side ** 2):
        offset = 0
        for scale, length in enumerate(lengths):
            for k in range(length):
                if scale < sink:
                    allowed = True
                elif scale == 0 or windows[scale] < 0:
                    allowed = False
                else:
                    kv_side = patch_nums[scale]
                    center_y = round((q // side) / side * kv_side)
                    center_x = round((q % side) / side * kv_side)
                    allowed = (abs(center_y - k // kv_side) <= windows[scale] // 2
                               and abs(center_x - k % kv_side) <= windows[scale] // 2)
                mask[q, offset + k] = allowed
            offset += length
    padded = F.pad(mask, (0, -mask.shape[1] % block_size, 0, -mask.shape[0] % block_size))
    return padded.reshape(padded.shape[0] // block_size, block_size,
                          padded.shape[1] // block_size, block_size).any(dim=(1, 3))


@pytest.mark.parametrize("preset", ["hart_legacy", "sparvar_legacy"])
@pytest.mark.parametrize("query_scale", [11, 12, 13])
@pytest.mark.parametrize("context", [3, 7])
def test_mask_matches_legacy_mapping_and_block_expansion(preset, query_scale, context):
    # Small geometry covers the full 14-scale preset, including omitted KV scales.
    patch_nums = (1, 2, 2, 3, 3, 3, 4, 4, 4, 4, 4, 5, 5, 6)
    options = UTILS["get_csla_preset"](preset)
    actual = UTILS["build_csla_block_layout"](
        patch_nums, context, query_scale, block_size=4, **options,
    )
    expected = _reference_mask(patch_nums, context, query_scale,
                               options["attn_sink_scales"], options["kernel_schedule"], 4)
    assert torch.equal(actual, expected)


def test_legacy_presets_and_query_scales_are_distinct():
    assert UTILS["CSLA_QUERY_SCALES"] == (11, 12, 13)
    first = UTILS["get_csla_preset"]("hart_legacy")
    second = UTILS["get_csla_preset"]("sparvar_legacy")
    assert (first["attn_sink_scales"], first["kernel_schedule"][-3:]) == (5, (3, 5, 7))
    assert (second["attn_sink_scales"], second["kernel_schedule"][-3:]) == (6, (1, 3, 5))
    with pytest.raises(ValueError, match="preset"):
        UTILS["get_csla_preset"]("unknown")


class _DenseAttention:
    def _attention(self, q, k, v, **kwargs):
        self.dense_calls += 1
        return "dense"


@pytest.mark.parametrize("using_flash", [False, True])
def test_sparse_hook_preserves_input_layout_and_dispatch(using_flash):
    env = _load_classes("models/sparvar/basic_sparvar_hart.py", {
        "LlamaAttention": _DenseAttention, "AdaLNSelfAttn": object,
    })
    attention = env["SparVARLlamaAttention"]()
    attention.accel_enabled, attention.training, attention.scale = True, False, 0.5
    attention.dense_calls = 0
    calls = []

    def sparse(q, k, v, scale):
        calls.append((q.shape, k.shape, scale))
        return q + 2

    attention.csla_attention = {11: sparse, 12: sparse, 13: sparse}
    q = torch.randn(2, 3, 5, 4)
    k = v = torch.randn(2, 3, 9, 4)
    inputs = [tensor.transpose(1, 2) if using_flash else tensor for tensor in (q, k, v)]
    copies = [tensor.clone() for tensor in inputs]
    kwargs = dict(using_flash=using_flash, attn_bias=None, main_type=q.dtype)
    result = attention._attention(*inputs, si=11, **kwargs)
    torch.testing.assert_close(result, (q + 2).transpose(1, 2).reshape(2, 5, 12))
    assert calls == [(q.shape, k.shape, 0.5)]
    for value, before in zip(inputs, copies):
        assert torch.equal(value, before)
    assert attention._attention(*inputs, si=10, **kwargs) == "dense"
    attention.accel_enabled = False
    assert attention._attention(*inputs, si=11, **kwargs) == "dense"
    assert attention.dense_calls == 2


def test_preparation_is_shared_idempotent_and_disabled_is_lazy():
    class Baseline:
        def prepare_for_inference(self):
            pass

        def configure_acceleration(self, options):
            self.acceleration_config = dict(options)
            self.accel_enabled = options.get("accel_enabled", True)
            return self.acceleration_config

    calls = []

    def prepare(*args, **kwargs):
        calls.append((args, kwargs))
        return {index: object() for index in (11, 12, 13)}

    env = _load_classes("models/sparvar/sparvar_model_hart.py", {
        "HARTForT2I": Baseline, "SparVARAdaLNSelfAttn": object,
        "CSLA_QUERY_SCALES": (11, 12, 13), "build_csla_attention": prepare,
        "get_csla_preset": UTILS["get_csla_preset"],
        "resolve_csla_geometry": UTILS["resolve_csla_geometry"],
        "resolve_hart_cs4a": UTILS["resolve_hart_cs4a"],
        "build_hart_cs4a_maps": UTILS["build_hart_cs4a_maps"],
    })
    model = env["SparVAR_HART"]()
    model.acceleration_config = None
    model.patch_nums, model.context_token, model.num_heads = tuple(range(1, 15)), 9, 2
    model.word_embed = SimpleNamespace(weight=torch.zeros(1))
    model.blocks = [SimpleNamespace(attn=SimpleNamespace()) for _ in range(3)]
    model.prepare_for_inference()
    model.prepare_for_inference()
    assert len(calls) == 1
    assert all(block.attn.csla_attention is model._csla_attention for block in model.blocks)
    model.configure_acceleration({"accel_enabled": False})
    model.prepare_for_inference()
    assert len(calls) == 1
    assert all(block.attn.csla_attention == {} for block in model.blocks)
    prepared = model._csla_attention
    for _ in range(3):
        model.configure_acceleration({"accel_enabled": True})
        model.prepare_for_inference()
        assert model._csla_attention is prepared
        assert all(block.attn.csla_attention is prepared for block in model.blocks)
        model.configure_acceleration({"accel_enabled": False})
    assert len(calls) == 1
    model.context_token += 1
    model.configure_acceleration({"accel_enabled": True})
    model.prepare_for_inference()
    assert len(calls) == 2
    assert model._csla_attention is not prepared
    model.configure_acceleration({"csla_sink": 6})
    model.prepare_for_inference()
    assert len(calls) == 3


def test_sparse_callable_rejects_partial_maskgit_geometry():
    operation = UTILS["SparseFlexAttn"](lambda *a, **k: None, None, 9, 20)
    with pytest.raises(ValueError, match="complete query"):
        operation(torch.zeros(1, 2, 5, 4), torch.zeros(1, 2, 20, 4), torch.zeros(1, 2, 20, 4))


@pytest.mark.parametrize("block_size", [64, 128])
def test_real_flex_mask_handles_partial_blocks_and_batch_broadcast(block_size):
    from torch.nn.attention.flex_attention import create_block_mask, flex_attention

    # Both dimensions cross a 128-token boundary and have a partial final block.
    query_length, kv_length = 289, 303
    layout = UTILS["build_csla_block_layout"]((1, 3, 17), 5, 2, 1, (-1, -1, 1), block_size=block_size)
    assert not layout.all()
    block_mask = create_block_mask(
        UTILS["make_csla_mask_mod"](layout, block_size), B=1, H=2,
        Q_LEN=query_length, KV_LEN=kv_length, device="cpu", BLOCK_SIZE=block_size,
    )
    generator = torch.Generator().manual_seed(3)
    q = torch.randn(2, 2, query_length, 8, generator=generator)
    k = torch.randn(2, 2, kv_length, 8, generator=generator)
    v = torch.randn(2, 2, kv_length, 8, generator=generator)
    # Use eager CPU FlexAttention so this never builds or runs a CUDA kernel.
    actual = flex_attention(q, k, v, block_mask=block_mask, scale=0.25)
    expanded = layout.repeat_interleave(block_size, 0).repeat_interleave(block_size, 1)
    allowed = expanded[:query_length, :kv_length]
    expected = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed, scale=0.25)
    assert actual.shape == q.shape
    torch.testing.assert_close(actual, expected, rtol=2e-5, atol=1e-6)


def test_custom_csla_geometry_validation():
    patches=(1,2,3,4,5,7,9,12,16,21,27,36,48,64)
    resolve=UTILS['resolve_csla_geometry']
    default=resolve({},patches,24)
    assert default['query_scales']==(11,12,13) and default['layers']==tuple(range(24))
    custom=resolve({'csla_query_scales':[12,13],'csla_layers':[2,5],
                    'csla_sink':6,'csla_windows':[-1]*8+[1,1,1,1,3,5],
                    'csla_block_size':64,'csla_kv_layout':'contiguous'},patches,24)
    assert custom['layers']==(2,5) and custom['block_size']==64
    for options in ({'csla_layers':[24]}, {'csla_query_scales':[11,11]},
                    {'csla_sink':0}, {'csla_windows':[3]}, {'csla_block_size':32},
                    {'csla_kv_layout':'typo'}):
        with pytest.raises(ValueError): resolve(options,patches,24)


def test_hart_cs4a_lut_preserves_sink_and_relative_visual_coordinates():
    patches=(1,2,3,4,5,7,9,12,16,21,27,36,48,64)
    maps,source_len,sink_len=UTILS['build_hart_cs4a_maps'](patches,300,(11,12,13),10,5,device='cpu')
    assert source_len==2054 and sink_len==354
    lengths=[300]+[p*p for p in patches[1:]]
    cum=[0]
    for size in lengths:cum.append(cum[-1]+size)
    for target,(lut,groups,side) in maps.items():
        assert torch.equal(lut[:sink_len],torch.arange(sink_len,dtype=torch.int32))
        assert len(lut)==source_len and lut.max()<cum[target+1]
        assert groups.shape==((side*side+191)//192,)
        assert groups.min()>=0 and groups.max()<4
        for source in range(5,11):
            destination=source+target-10
            a,b=patches[source],patches[destination]
            for offset in (0,a*a//2,a*a-1):
                y,x=divmod(offset,a)
                expected=cum[destination]+((2*y+1)*b//(2*a))*b+(2*x+1)*b//(2*a)
                assert lut[cum[source]+offset].item()==expected
        assert len(lut.unique())==source_len


def test_cs4a_configuration_requires_matched_fa3_decision_and_valid_layers():
    patches=(1,2,3,4,5,7,9,12,16,21,27,36,48,64)
    geometry=UTILS['resolve_csla_geometry']({},patches,24)
    resolve=UTILS['resolve_hart_cs4a']
    assert resolve({},geometry,patches,24)['layers']==()
    value=resolve({'cs4a_layers':[0,1],'dense_attn_backend':'fa3'},geometry,patches,24)
    assert value['decision_scale']==10 and value['use_o_cache']
    for options in ({'cs4a_layers':[0]}, {'cs4a_layers':[24]},
                    {'cs4a_layers':[0],'dense_attn_backend':'fa3','hart_fa3_start_scale':11},
                    {'cs4a_ratio':0}):
        with pytest.raises(ValueError):resolve(options,geometry,patches,24)


@pytest.mark.parametrize('using_flash',[False,True])
def test_cs4a_attention_routes_decision_and_requires_request_state(using_flash):
    state=object();calls=[]
    def decision(q,k,v,scale,keep,sink,use_cache):
        calls.append('decision');return q+1,state
    def sparse(q,k,v,scale,cached,mapping,keep,side):
        assert cached is state and mapping=='mapping'
        calls.append('sparse');return q+2
    env=_load_classes('models/sparvar/basic_sparvar_hart.py',{
        'LlamaAttention':_DenseAttention,'AdaLNSelfAttn':object,
        'hart_cs4a_decision':decision,'hart_cs4a_sparse':sparse,
    })
    attn=env['SparVARLlamaAttention']()
    attn.training=False;attn.accel_enabled=True;attn.scale=0.03125;attn.block_idx=0
    attn.cs4a_options={'decision_scale':10,'use_o_cache':True}
    attn.cs4a_maps={11:'mapping'};attn.cs4a_keep=3;attn.cs4a_sink_len=1;attn.cs4a_source_side=27
    attn._hart_request_cache={}
    q=torch.randn(2,3,5,4)
    inputs=[q.transpose(1,2) if using_flash else q]*3
    kwargs=dict(using_flash=using_flash,attn_bias=None,main_type=q.dtype)
    with pytest.raises(RuntimeError,match='missing'):
        attn._attention(*inputs,si=11,**kwargs)
    torch.testing.assert_close(attn._attention(*inputs,si=10,**kwargs),(q+1).transpose(1,2).reshape(2,5,12))
    torch.testing.assert_close(attn._attention(*inputs,si=11,**kwargs),(q+2).transpose(1,2).reshape(2,5,12))
    assert calls==['decision','sparse']
    attn._hart_request_cache.clear()
    with pytest.raises(RuntimeError,match='missing'):
        attn._attention(*inputs,si=11,**kwargs)


def test_prefix_fa3_matches_the_expanded_prefix_mask(monkeypatch):
    import sys
    def fake_fa3(q,k,v,softmax_scale):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale=softmax_scale).transpose(1,2)
    monkeypatch.setitem(sys.modules,'flash_attn_interface',SimpleNamespace(flash_attn_func=fake_fa3))
    layout=torch.tensor([[True,True,False,False]]*3)
    op=UTILS['PrefixFA3Attention'](layout,289,450,128)
    assert op.prefix_length==256
    q=torch.randn(2,2,289,8);k=torch.randn(2,2,450,8);v=torch.randn_like(k)
    mask=torch.arange(450)[None,:]<256
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=0.25)
    torch.testing.assert_close(op(q,k,v,scale=0.25),expected)
    for bad in (torch.tensor([[True,False,True]]),torch.tensor([[True,False],[False,True]]),torch.zeros(1,2,dtype=torch.bool)):
        with pytest.raises(ValueError,match='prefix'):UTILS['PrefixFA3Attention'](bad,1,2,128)


def test_uniform_fa3_gather_matches_disjoint_block_mask(monkeypatch):
    import sys
    def fake_fa3(q,k,v,softmax_scale):
        return F.scaled_dot_product_attention(q.transpose(1,2),k.transpose(1,2),v.transpose(1,2),scale=softmax_scale).transpose(1,2)
    monkeypatch.setitem(sys.modules,'flash_attn_interface',SimpleNamespace(flash_attn_func=fake_fa3))
    layout=torch.tensor([[True,False,True]]*2)
    op=UTILS['PrefixFA3Attention'](layout,129,303,128,allow_gaps=True)
    assert op.prefix_length is None and op.indices.numel()==175
    q=torch.randn(2,2,129,8);k=torch.randn(2,2,303,8);v=torch.randn_like(k)
    mask=layout[0].repeat_interleave(128)[:303][None,:]
    expected=F.scaled_dot_product_attention(q,k,v,attn_mask=mask,scale=0.25)
    torch.testing.assert_close(op(q,k,v,scale=0.25),expected)
