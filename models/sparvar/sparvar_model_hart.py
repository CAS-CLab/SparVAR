"""CSLA acceleration as a checkpoint-compatible HART subclass."""

from models.hart.hart_transformer_t2i import HARTForT2I
from models.sparvar.basic_sparvar_hart import SparVARAdaLNSelfAttn
from models.sparvar.hart_csla_utils import (
    CSLA_QUERY_SCALES,
    build_csla_attention,
    get_csla_preset,
    resolve_csla_geometry,
    resolve_hart_cs4a,
    build_hart_cs4a_maps,
)


class SparVAR_HART(HARTForT2I):
    block_class = SparVARAdaLNSelfAttn
    acceleration_method = "sparvar_hart"
    accel_enabled = True

    def configure_acceleration(self, options=None):
        previous_geometry = (getattr(self, "csla_geometry", None),
                             getattr(self, "cs4a_geometry", None), getattr(self, "csla_preset", None))
        options = dict(options or {})
        preset = options.get("csla_preset", "hart_legacy")
        get_csla_preset(preset)
        resolved = super().configure_acceleration(options)
        self.csla_geometry = resolve_csla_geometry(resolved, self.patch_nums, len(self.blocks))
        if self.csla_geometry["kernel"] in ("prefix_fa3", "uniform_fa3"):
            if resolved.get("dense_attn_backend") != "fa3" or resolved.get("hart_fa3_start_scale", 0) > min(self.csla_geometry["query_scales"]):
                raise ValueError("prefix_fa3 requires FA3 at the corresponding dense reference scales")
        self.cs4a_geometry = resolve_hart_cs4a(resolved, self.csla_geometry, self.patch_nums, len(self.blocks))
        self.csla_preset = preset
        same_geometry = previous_geometry == (self.csla_geometry, self.cs4a_geometry, preset)
        # Reuse prepared masks across enable/disable toggles to avoid recompilation.
        if not same_geometry:
            self.cs4a_runtime_geometry = None
            self._csla_prepared_key = None
            self._csla_attention = {}
        for index, block in enumerate(self.blocks):
            block.attn.csla_attention = self._csla_attention if self.accel_enabled and index in self.csla_geometry["layers"] else {}
            block.attn.csla_kv_layout = self.csla_geometry["kv_layout"]
            block.attn.cs4a_options = self.cs4a_geometry if index in self.cs4a_geometry["layers"] else None
            if not same_geometry:
                block.attn.cs4a_maps = {}
        return resolved

    def prepare_for_inference(self):
        if self.acceleration_config is None:
            self.configure_acceleration({})
        super().prepare_for_inference()
        if not self.accel_enabled:
            return
        device = self.word_embed.weight.device
        key = (tuple(self.patch_nums), self.context_token, self.num_heads,
               str(device), self.csla_preset, tuple(self.csla_geometry.items()), tuple(self.cs4a_geometry.items()))
        if self._csla_prepared_key == key:
            return
        sparse_attention = build_csla_attention(
            self.patch_nums, self.context_token, self.num_heads,
            device=device, preset=self.csla_preset,
            **{key: value for key, value in self.csla_geometry.items() if key not in ("layers", "kv_layout")},
        )
        self._csla_attention = sparse_attention
        for index, block in enumerate(self.blocks):
            block.attn.csla_attention = sparse_attention if index in self.csla_geometry["layers"] else {}
        if self.cs4a_geometry["layers"]:
            maps, source_len, sink_len = build_hart_cs4a_maps(
                self.patch_nums, self.context_token, self.csla_geometry["query_scales"],
                self.cs4a_geometry["decision_scale"], self.csla_geometry["attn_sink_scales"], device=device,
            )
            keep = max(sink_len, int(source_len * self.cs4a_geometry["ratio"]))
            self.cs4a_runtime_geometry = dict(source_kv_length=source_len, sink_length=sink_len,
                                              selected_keys_per_group=keep,
                                              target_keep_ratios={si: keep / (self.context_token + sum(p*p for p in self.patch_nums[1:si+1]))
                                                                  for si in self.csla_geometry["query_scales"]})
            for index in self.cs4a_geometry["layers"]:
                attention = self.blocks[index].attn
                attention.cs4a_maps = maps
                attention.cs4a_sink_len = sink_len
                attention.cs4a_keep = keep
                attention.cs4a_source_side = self.patch_nums[self.cs4a_geometry["decision_scale"]]
        self._csla_prepared_key = key

    @property
    def csla_query_scales(self):
        return self.csla_geometry["query_scales"]
