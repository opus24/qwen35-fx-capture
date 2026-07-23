class GraphModule(torch.nn.Module):
    def forward(self, L_position_ids_: "i64[3, 1, 128]", L_self_buffers_inv_freq_: "bf16[32]"):
        l_position_ids_ = L_position_ids_
        l_self_buffers_inv_freq_ = L_self_buffers_inv_freq_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:157 in forward, code: self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        getitem: "bf16[1, 1, 32, 1]" = l_self_buffers_inv_freq_[(None, None, slice(None, None, None), None)];  l_self_buffers_inv_freq_ = None
        float_1: "f32[1, 1, 32, 1]" = getitem.float();  getitem = None
        expand: "f32[3, 1, 32, 1]" = float_1.expand(3, 1, -1, 1);  float_1 = None
        inv_freq_expanded: "f32[3, 1, 32, 1]" = expand.to(device(type='cuda', index=0));  expand = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:159 in forward, code: position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        getitem_1: "i64[3, 1, 1, 128]" = l_position_ids_[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  l_position_ids_ = None
        position_ids_expanded: "f32[3, 1, 1, 128]" = getitem_1.float();  getitem_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:163 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        float_3: "f32[3, 1, 32, 1]" = inv_freq_expanded.float();  inv_freq_expanded = None
        float_4: "f32[3, 1, 1, 128]" = position_ids_expanded.float();  position_ids_expanded = None
        matmul: "f32[3, 1, 32, 128]" = float_3 @ float_4;  float_3 = float_4 = None
        freqs: "f32[3, 1, 128, 32]" = matmul.transpose(2, 3);  matmul = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:181 in apply_interleaved_mrope, code: freqs_t = freqs[0]  # just overwrite the first dimension T
        freqs_t: "f32[1, 128, 32]" = freqs[0]
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:185 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        getitem_3: "f32[1, 128, 11]" = freqs[(1, Ellipsis, slice(1, 33, 3))]
        freqs_t[(Ellipsis, slice(1, 33, 3))] = getitem_3;  setitem = freqs_t;  getitem_3 = setitem = None
        getitem_4: "f32[1, 128, 10]" = freqs[(2, Ellipsis, slice(2, 30, 3))];  freqs = None
        freqs_t[(Ellipsis, slice(2, 30, 3))] = getitem_4;  setitem_1 = freqs_t;  getitem_4 = setitem_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:165 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        emb: "f32[1, 128, 64]" = torch.cat((freqs_t, freqs_t), dim = -1);  freqs_t = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:166 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 128, 64]" = emb.cos()
        cos_1: "f32[1, 128, 64]" = cos * 1.0;  cos = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:167 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 128, 64]" = emb.sin();  emb = None
        sin_1: "f32[1, 128, 64]" = sin * 1.0;  sin = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:169 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        to_1: "bf16[1, 128, 64]" = cos_1.to(dtype = torch.bfloat16);  cos_1 = None
        to_2: "bf16[1, 128, 64]" = sin_1.to(dtype = torch.bfloat16);  sin_1 = None
        return (to_1, to_2)
        