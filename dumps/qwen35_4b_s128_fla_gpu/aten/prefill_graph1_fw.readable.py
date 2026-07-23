class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "i64[3, 1, 128]", arg1_1: "bf16[32]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:157 in forward, code: self.inv_freq[None, None, :, None].float().expand(3, position_ids.shape[1], -1, 1).to(x.device)
        unsqueeze: "bf16[1, 32]" = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
        unsqueeze_1: "bf16[1, 1, 32]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "bf16[1, 1, 32, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        _to_copy: "f32[1, 1, 32, 1]" = torch.ops.aten._to_copy.default(unsqueeze_2, dtype = torch.float32);  unsqueeze_2 = None
        expand: "f32[3, 1, 32, 1]" = torch.ops.aten.expand.default(_to_copy, [3, 1, -1, 1]);  _to_copy = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:159 in forward, code: position_ids_expanded = position_ids[:, :, None, :].float()  # shape (3, bs, 1, positions)
        unsqueeze_3: "i64[3, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
        _to_copy_1: "f32[3, 1, 1, 128]" = torch.ops.aten._to_copy.default(unsqueeze_3, dtype = torch.float32);  unsqueeze_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:163 in forward, code: freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(2, 3)
        expand_1: "f32[3, 1, 32, 1]" = torch.ops.aten.expand.default(expand, [3, 1, 32, 1]);  expand = None
        view: "f32[3, 32, 1]" = torch.ops.aten.view.default(expand_1, [3, 32, 1]);  expand_1 = None
        expand_2: "f32[3, 1, 1, 128]" = torch.ops.aten.expand.default(_to_copy_1, [3, 1, 1, 128]);  _to_copy_1 = None
        view_1: "f32[3, 1, 128]" = torch.ops.aten.view.default(expand_2, [3, 1, 128]);  expand_2 = None
        bmm: "f32[3, 32, 128]" = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
        view_2: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(bmm, [3, 1, 32, 128])
        permute: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:181 in apply_interleaved_mrope, code: freqs_t = freqs[0]  # just overwrite the first dimension T
        select: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute, 0, 0)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:185 in apply_interleaved_mrope, code: freqs_t[..., idx] = freqs[dim, ..., idx]
        select_1: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute, 0, 1);  permute = None
        slice_1: "f32[1, 128, 11]" = torch.ops.aten.slice.Tensor(select_1, 2, 1, 33, 3);  select_1 = None
        slice_2: "f32[1, 128, 11]" = torch.ops.aten.slice.Tensor(select, 2, 1, 33, 3);  select = None
        copy: "f32[1, 128, 11]" = torch.ops.aten.copy.default(slice_2, slice_1);  slice_2 = slice_1 = None
        view_3: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(bmm, [3, 1, 32, 128]);  bmm = None
        permute_1: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
        select_2: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute_1, 0, 0)
        slice_scatter: "f32[1, 128, 32]" = torch.ops.aten.slice_scatter.default(select_2, copy, 2, 1, 33, 3);  select_2 = copy = None
        arange: "i64[3]" = torch.ops.aten.arange.start_step(0, 3, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_4: "i64[3, 1, 1, 1]" = torch.ops.aten.view.default(arange, [-1, 1, 1, 1]);  arange = None
        eq: "b8[3, 1, 1, 1]" = torch.ops.aten.eq.Scalar(view_4, 0);  view_4 = None
        unsqueeze_4: "f32[1, 1, 128, 32]" = torch.ops.aten.unsqueeze.default(slice_scatter, 0);  slice_scatter = None
        expand_3: "f32[3, 1, 128, 32]" = torch.ops.aten.expand.default(unsqueeze_4, [3, 1, 128, 32]);  unsqueeze_4 = None
        where: "f32[3, 1, 128, 32]" = torch.ops.aten.where.self(eq, expand_3, permute_1);  eq = expand_3 = permute_1 = None
        permute_2: "f32[3, 1, 32, 128]" = torch.ops.aten.permute.default(where, [0, 1, 3, 2]);  where = None
        view_5: "f32[3, 32, 128]" = torch.ops.aten.view.default(permute_2, [3, 32, 128]);  permute_2 = None
        view_10: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(view_5, [3, 1, 32, 128])
        permute_7: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_10, [0, 1, 3, 2]);  view_10 = None
        select_7: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute_7, 0, 0);  permute_7 = None
        slice_6: "f32[1, 128, 10]" = torch.ops.aten.slice.Tensor(select_7, 2, 2, 30, 3);  select_7 = None
        view_11: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(view_5, [3, 1, 32, 128])
        permute_8: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_11, [0, 1, 3, 2]);  view_11 = None
        select_8: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute_8, 0, 2);  permute_8 = None
        slice_7: "f32[1, 128, 10]" = torch.ops.aten.slice.Tensor(select_8, 2, 2, 30, 3);  select_8 = None
        copy_1: "f32[1, 128, 10]" = torch.ops.aten.copy.default(slice_6, slice_7);  slice_6 = slice_7 = None
        view_12: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(view_5, [3, 1, 32, 128]);  view_5 = None
        permute_9: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_12, [0, 1, 3, 2]);  view_12 = None
        select_9: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute_9, 0, 0)
        slice_scatter_1: "f32[1, 128, 32]" = torch.ops.aten.slice_scatter.default(select_9, copy_1, 2, 2, 30, 3);  select_9 = copy_1 = None
        arange_1: "i64[3]" = torch.ops.aten.arange.start_step(0, 3, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        view_13: "i64[3, 1, 1, 1]" = torch.ops.aten.view.default(arange_1, [-1, 1, 1, 1]);  arange_1 = None
        eq_1: "b8[3, 1, 1, 1]" = torch.ops.aten.eq.Scalar(view_13, 0);  view_13 = None
        unsqueeze_5: "f32[1, 1, 128, 32]" = torch.ops.aten.unsqueeze.default(slice_scatter_1, 0);  slice_scatter_1 = None
        expand_4: "f32[3, 1, 128, 32]" = torch.ops.aten.expand.default(unsqueeze_5, [3, 1, 128, 32]);  unsqueeze_5 = None
        where_1: "f32[3, 1, 128, 32]" = torch.ops.aten.where.self(eq_1, expand_4, permute_9);  eq_1 = expand_4 = permute_9 = None
        permute_10: "f32[3, 1, 32, 128]" = torch.ops.aten.permute.default(where_1, [0, 1, 3, 2]);  where_1 = None
        view_14: "f32[3, 32, 128]" = torch.ops.aten.view.default(permute_10, [3, 32, 128]);  permute_10 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:181 in apply_interleaved_mrope, code: freqs_t = freqs[0]  # just overwrite the first dimension T
        view_16: "f32[3, 1, 32, 128]" = torch.ops.aten.view.default(view_14, [3, 1, 32, 128]);  view_14 = None
        permute_12: "f32[3, 1, 128, 32]" = torch.ops.aten.permute.default(view_16, [0, 1, 3, 2]);  view_16 = None
        select_11: "f32[1, 128, 32]" = torch.ops.aten.select.int(permute_12, 0, 0);  permute_12 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:165 in forward, code: emb = torch.cat((freqs, freqs), dim=-1)
        cat: "f32[1, 128, 64]" = torch.ops.aten.cat.default([select_11, select_11], -1);  select_11 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:166 in forward, code: cos = emb.cos() * self.attention_scaling
        cos: "f32[1, 128, 64]" = torch.ops.aten.cos.default(cat)
        mul: "f32[1, 128, 64]" = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:167 in forward, code: sin = emb.sin() * self.attention_scaling
        sin: "f32[1, 128, 64]" = torch.ops.aten.sin.default(cat);  cat = None
        mul_1: "f32[1, 128, 64]" = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:169 in forward, code: return cos.to(dtype=x.dtype), sin.to(dtype=x.dtype)
        _to_copy_2: "bf16[1, 128, 64]" = torch.ops.aten._to_copy.default(mul, dtype = torch.bfloat16);  mul = None
        _to_copy_3: "bf16[1, 128, 64]" = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16);  mul_1 = None
        return (_to_copy_2, _to_copy_3)
        