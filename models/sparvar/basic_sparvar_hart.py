"""HART blocks using CSLA without changing dense attention or cache layouts."""

from models.hart.basic_hart import AdaLNSelfAttn, LlamaAttention
from models.sparvar.hart_csla_utils import hart_cs4a_decision, hart_cs4a_sparse


class SparVARLlamaAttention(LlamaAttention):
    def _attention(self, q, k, v, *, using_flash, attn_bias, main_type, si):
        cs4a = getattr(self, "cs4a_options", None)
        if cs4a and getattr(self, "accel_enabled", True) and not self.training and attn_bias is None:
            is_decision = si == cs4a["decision_scale"]
            if is_decision or si in self.cs4a_maps:
                if using_flash:
                    q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
                q, k = q.to(v.dtype), k.to(v.dtype)
                key = ("cs4a", self.block_idx)
                if is_decision:
                    output, state = hart_cs4a_decision(q, k, v, self.scale, self.cs4a_keep,
                                                       self.cs4a_sink_len, cs4a["use_o_cache"])
                    self._hart_request_cache[key] = state
                else:
                    if key not in self._hart_request_cache:
                        raise RuntimeError("CS4A decision state is missing for this request")
                    output = hart_cs4a_sparse(q, k, v, self.scale, self._hart_request_cache[key],
                                              self.cs4a_maps[si], self.cs4a_keep, self.cs4a_source_side)
                return output.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)
        sparse_attention = getattr(self, "csla_attention", {}).get(si)
        if (
            not getattr(self, "accel_enabled", True)
            or self.training
            or attn_bias is not None
            or sparse_attention is None
        ):
            return super()._attention(
                q, k, v, using_flash=using_flash, attn_bias=attn_bias,
                main_type=main_type, si=si,
            )

        # Keep the parent's cache layout and transpose only the sparse call.
        if using_flash:
            q, k, v = (tensor.transpose(1, 2) for tensor in (q, k, v))
        if getattr(self, "csla_kv_layout", "native") == "contiguous":
            k, v = k.contiguous(), v.contiguous()
        result = sparse_attention(q, k, v, scale=self.scale)
        return result.transpose(1, 2).reshape(q.shape[0], q.shape[2], -1)


class SparVARAdaLNSelfAttn(AdaLNSelfAttn):
    attention_class = SparVARLlamaAttention
