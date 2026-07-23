class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 128, 2560]", arg1_1: "bf16[2560]", arg2_1: "bf16[8192, 2560]", arg3_1: "bf16[256]", arg4_1: "bf16[1024, 2560]", arg5_1: "bf16[256]", arg6_1: "bf16[1024, 2560]", arg7_1: "bf16[1, 128, 64]", arg8_1: "bf16[1, 128, 64]", arg9_1: "bf16[1, 1, 128, 128]", arg10_1: "bf16[2560, 4096]", arg11_1: "bf16[2560]", arg12_1: "bf16[9216, 2560]", arg13_1: "bf16[9216, 2560]", arg14_1: "bf16[2560, 9216]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy: "f32[1, 128, 2560]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_1: "f32[1, 128, 2560]" = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
        mean: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        _to_copy_1: "f32[2560]" = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
        add_1: "f32[2560]" = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
        mul_1: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        _to_copy_2: "bf16[1, 128, 2560]" = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:688 in forward, code: self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        permute: "bf16[2560, 8192]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
        mm: "bf16[128, 8192]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 128, 8192]" = torch.ops.aten.view.default(mm, [1, 128, 8192]);  mm = None
        view_2: "bf16[1, 128, 16, 512]" = torch.ops.aten.view.default(view_1, [1, 128, -1, 512]);  view_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:687 in forward, code: query_states, gate = torch.chunk(
        split_with_sizes = torch.ops.aten.split_with_sizes.default(view_2, [256, 256], -1);  view_2 = None
        getitem: "bf16[1, 128, 16, 256]" = split_with_sizes[0]
        getitem_1: "bf16[1, 128, 16, 256]" = split_with_sizes[1];  split_with_sizes = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:690 in forward, code: gate = gate.reshape(*input_shape, -1)
        clone: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(getitem_1, memory_format = torch.contiguous_format);  getitem_1 = None
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone, [1, 128, 4096]);  clone = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:692 in forward, code: query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        view_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.view.default(getitem, [1, 128, -1, 256]);  getitem = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy_3: "f32[1, 128, 16, 256]" = torch.ops.aten._to_copy.default(view_4, dtype = torch.float32);  view_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_2: "f32[1, 128, 16, 256]" = torch.ops.aten.pow.Tensor_Scalar(_to_copy_3, 2)
        mean_1: "f32[1, 128, 16, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
        add_2: "f32[1, 128, 16, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
        rsqrt_1: "f32[1, 128, 16, 1]" = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
        mul_2: "f32[1, 128, 16, 256]" = torch.ops.aten.mul.Tensor(_to_copy_3, rsqrt_1);  _to_copy_3 = rsqrt_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        _to_copy_4: "f32[256]" = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
        add_3: "f32[256]" = torch.ops.aten.add.Tensor(_to_copy_4, 1.0);  _to_copy_4 = None
        mul_3: "f32[1, 128, 16, 256]" = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        _to_copy_5: "bf16[1, 128, 16, 256]" = torch.ops.aten._to_copy.default(mul_3, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:692 in forward, code: query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        permute_1: "bf16[1, 16, 128, 256]" = torch.ops.aten.permute.default(_to_copy_5, [0, 2, 1, 3]);  _to_copy_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:693 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_2: "bf16[2560, 1024]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_5: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
        mm_1: "bf16[128, 1024]" = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
        view_6: "bf16[1, 128, 1024]" = torch.ops.aten.view.default(mm_1, [1, 128, 1024]);  mm_1 = None
        view_7: "bf16[1, 128, 4, 256]" = torch.ops.aten.view.default(view_6, [1, 128, -1, 256]);  view_6 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy_6: "f32[1, 128, 4, 256]" = torch.ops.aten._to_copy.default(view_7, dtype = torch.float32);  view_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_3: "f32[1, 128, 4, 256]" = torch.ops.aten.pow.Tensor_Scalar(_to_copy_6, 2)
        mean_2: "f32[1, 128, 4, 1]" = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
        add_4: "f32[1, 128, 4, 1]" = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
        rsqrt_2: "f32[1, 128, 4, 1]" = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
        mul_4: "f32[1, 128, 4, 256]" = torch.ops.aten.mul.Tensor(_to_copy_6, rsqrt_2);  _to_copy_6 = rsqrt_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        _to_copy_7: "f32[256]" = torch.ops.aten._to_copy.default(arg5_1, dtype = torch.float32);  arg5_1 = None
        add_5: "f32[256]" = torch.ops.aten.add.Tensor(_to_copy_7, 1.0);  _to_copy_7 = None
        mul_5: "f32[1, 128, 4, 256]" = torch.ops.aten.mul.Tensor(mul_4, add_5);  mul_4 = add_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        _to_copy_8: "bf16[1, 128, 4, 256]" = torch.ops.aten._to_copy.default(mul_5, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:693 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        permute_3: "bf16[1, 4, 128, 256]" = torch.ops.aten.permute.default(_to_copy_8, [0, 2, 1, 3]);  _to_copy_8 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:694 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        permute_4: "bf16[2560, 1024]" = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
        view_8: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_2, [128, 2560]);  _to_copy_2 = None
        mm_2: "bf16[128, 1024]" = torch.ops.aten.mm.default(view_8, permute_4);  view_8 = permute_4 = None
        view_9: "bf16[1, 128, 1024]" = torch.ops.aten.view.default(mm_2, [1, 128, 1024]);  mm_2 = None
        view_10: "bf16[1, 128, 4, 256]" = torch.ops.aten.view.default(view_9, [1, 128, -1, 256]);  view_9 = None
        permute_5: "bf16[1, 4, 128, 256]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:594 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        unsqueeze: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(arg7_1, 1);  arg7_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:595 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        unsqueeze_1: "bf16[1, 1, 128, 64]" = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:599 in apply_rotary_pos_emb, code: q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        slice_1: "bf16[1, 16, 128, 64]" = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64)
        slice_2: "bf16[1, 16, 128, 192]" = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 9223372036854775807);  permute_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:600 in apply_rotary_pos_emb, code: k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        slice_3: "bf16[1, 4, 128, 64]" = torch.ops.aten.slice.Tensor(permute_3, 3, 0, 64)
        slice_4: "bf16[1, 4, 128, 192]" = torch.ops.aten.slice.Tensor(permute_3, 3, 64, 9223372036854775807);  permute_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:603 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_6: "bf16[1, 16, 128, 64]" = torch.ops.aten.mul.Tensor(slice_1, unsqueeze)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:568 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_5: "bf16[1, 16, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 32)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:569 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_6: "bf16[1, 16, 128, 32]" = torch.ops.aten.slice.Tensor(slice_1, 3, 32, 9223372036854775807);  slice_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:570 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[1, 16, 128, 32]" = torch.ops.aten.neg.default(slice_6);  slice_6 = None
        cat: "bf16[1, 16, 128, 64]" = torch.ops.aten.cat.default([neg, slice_5], -1);  neg = slice_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:603 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_7: "bf16[1, 16, 128, 64]" = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
        add_6: "bf16[1, 16, 128, 64]" = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:604 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_8: "bf16[1, 4, 128, 64]" = torch.ops.aten.mul.Tensor(slice_3, unsqueeze);  unsqueeze = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:568 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        slice_7: "bf16[1, 4, 128, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 32)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:569 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        slice_8: "bf16[1, 4, 128, 32]" = torch.ops.aten.slice.Tensor(slice_3, 3, 32, 9223372036854775807);  slice_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:570 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[1, 4, 128, 32]" = torch.ops.aten.neg.default(slice_8);  slice_8 = None
        cat_1: "bf16[1, 4, 128, 64]" = torch.ops.aten.cat.default([neg_1, slice_7], -1);  neg_1 = slice_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:604 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_9: "bf16[1, 4, 128, 64]" = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
        add_7: "bf16[1, 4, 128, 64]" = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:607 in apply_rotary_pos_emb, code: q_embed = torch.cat([q_embed, q_pass], dim=-1)
        cat_2: "bf16[1, 16, 128, 256]" = torch.ops.aten.cat.default([add_6, slice_2], -1);  add_6 = slice_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:608 in apply_rotary_pos_emb, code: k_embed = torch.cat([k_embed, k_pass], dim=-1)
        cat_3: "bf16[1, 4, 128, 256]" = torch.ops.aten.cat.default([add_7, slice_4], -1);  add_7 = slice_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:122 in lazy_initialization, code: self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        _tensor_constant0: "bf16[0]" = self._tensor_constant0
        lift_fresh_copy: "bf16[0]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:123 in lazy_initialization, code: self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        _tensor_constant1: "bf16[0]" = self._tensor_constant1
        lift_fresh_copy_1: "bf16[0]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:143 in update, code: self.keys = torch.cat([self.keys, key_states], dim=-2)
        cat_4: "bf16[1, 4, 128, 256]" = torch.ops.aten.cat.default([lift_fresh_copy, cat_3], -2);  lift_fresh_copy = cat_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        cat_5: "bf16[1, 4, 128, 256]" = torch.ops.aten.cat.default([lift_fresh_copy_1, permute_5], -2);  lift_fresh_copy_1 = permute_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:620 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_2: "bf16[1, 4, 1, 128, 256]" = torch.ops.aten.unsqueeze.default(cat_4, 2)
        expand: "bf16[1, 4, 4, 128, 256]" = torch.ops.aten.expand.default(unsqueeze_2, [1, 4, 4, 128, 256]);  unsqueeze_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:621 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_1: "bf16[1, 4, 4, 128, 256]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_11: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(clone_1, [1, 16, 128, 256]);  clone_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:620 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        unsqueeze_3: "bf16[1, 4, 1, 128, 256]" = torch.ops.aten.unsqueeze.default(cat_5, 2)
        expand_1: "bf16[1, 4, 4, 128, 256]" = torch.ops.aten.expand.default(unsqueeze_3, [1, 4, 4, 128, 256]);  unsqueeze_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:621 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        clone_2: "bf16[1, 4, 4, 128, 256]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_12: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(clone_2, [1, 16, 128, 256]);  clone_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:637 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        permute_6: "bf16[1, 16, 256, 128]" = torch.ops.aten.permute.default(view_11, [0, 1, 3, 2]);  view_11 = None
        expand_2: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(cat_2, [1, 16, 128, 256]);  cat_2 = None
        view_13: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_2, [16, 128, 256]);  expand_2 = None
        expand_3: "bf16[1, 16, 256, 128]" = torch.ops.aten.expand.default(permute_6, [1, 16, 256, 128]);  permute_6 = None
        view_14: "bf16[16, 256, 128]" = torch.ops.aten.view.default(expand_3, [16, 256, 128]);  expand_3 = None
        bmm: "bf16[16, 128, 128]" = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
        view_15: "bf16[1, 16, 128, 128]" = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
        mul_10: "bf16[1, 16, 128, 128]" = torch.ops.aten.mul.Tensor(view_15, 0.0625);  view_15 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:639 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        add_8: "bf16[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(mul_10, arg9_1);  mul_10 = arg9_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:641 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        _to_copy_9: "f32[1, 16, 128, 128]" = torch.ops.aten._to_copy.default(add_8, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  add_8 = None
        _softmax: "f32[1, 16, 128, 128]" = torch.ops.aten._softmax.default(_to_copy_9, -1, False);  _to_copy_9 = None
        _to_copy_10: "bf16[1, 16, 128, 128]" = torch.ops.aten._to_copy.default(_softmax, dtype = torch.bfloat16);  _softmax = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:642 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        clone_3: "bf16[1, 16, 128, 128]" = torch.ops.aten.clone.default(_to_copy_10);  _to_copy_10 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:643 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value_states)
        expand_4: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(clone_3, [1, 16, 128, 128]);  clone_3 = None
        view_16: "bf16[16, 128, 128]" = torch.ops.aten.view.default(expand_4, [16, 128, 128]);  expand_4 = None
        expand_5: "bf16[1, 16, 128, 256]" = torch.ops.aten.expand.default(view_12, [1, 16, 128, 256]);  view_12 = None
        view_17: "bf16[16, 128, 256]" = torch.ops.aten.view.default(expand_5, [16, 128, 256]);  expand_5 = None
        bmm_1: "bf16[16, 128, 256]" = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
        view_18: "bf16[1, 16, 128, 256]" = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 256]);  bmm_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:644 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        permute_7: "bf16[1, 128, 16, 256]" = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
        clone_4: "bf16[1, 128, 16, 256]" = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:717 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        view_19: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(clone_4, [1, 128, -1]);  clone_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:718 in forward, code: attn_output = attn_output * torch.sigmoid(gate)
        sigmoid: "bf16[1, 128, 4096]" = torch.ops.aten.sigmoid.default(view_3);  view_3 = None
        mul_11: "bf16[1, 128, 4096]" = torch.ops.aten.mul.Tensor(view_19, sigmoid);  view_19 = sigmoid = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:720 in forward, code: attn_output = self.o_proj(attn_output)
        permute_8: "bf16[4096, 2560]" = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
        view_20: "bf16[128, 4096]" = torch.ops.aten.view.default(mul_11, [128, 4096]);  mul_11 = None
        mm_3: "bf16[128, 2560]" = torch.ops.aten.mm.default(view_20, permute_8);  view_20 = permute_8 = None
        view_21: "bf16[1, 128, 2560]" = torch.ops.aten.view.default(mm_3, [1, 128, 2560]);  mm_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:805 in forward, code: hidden_states = residual + hidden_states
        add_9: "bf16[1, 128, 2560]" = torch.ops.aten.add.Tensor(arg0_1, view_21);  arg0_1 = view_21 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy_11: "f32[1, 128, 2560]" = torch.ops.aten._to_copy.default(add_9, dtype = torch.float32)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_4: "f32[1, 128, 2560]" = torch.ops.aten.pow.Tensor_Scalar(_to_copy_11, 2)
        mean_3: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
        add_10: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
        mul_12: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(_to_copy_11, rsqrt_3);  _to_copy_11 = rsqrt_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        _to_copy_12: "f32[2560]" = torch.ops.aten._to_copy.default(arg11_1, dtype = torch.float32);  arg11_1 = None
        add_11: "f32[2560]" = torch.ops.aten.add.Tensor(_to_copy_12, 1.0);  _to_copy_12 = None
        mul_13: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul_12, add_11);  mul_12 = add_11 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        _to_copy_13: "bf16[1, 128, 2560]" = torch.ops.aten._to_copy.default(mul_13, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_13 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_9: "bf16[2560, 9216]" = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
        view_22: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_13, [128, 2560])
        mm_4: "bf16[128, 9216]" = torch.ops.aten.mm.default(view_22, permute_9);  view_22 = permute_9 = None
        view_23: "bf16[1, 128, 9216]" = torch.ops.aten.view.default(mm_4, [1, 128, 9216]);  mm_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        _to_copy_14: "f32[1, 128, 9216]" = torch.ops.aten._to_copy.default(view_23, dtype = torch.float32);  view_23 = None
        sigmoid_1: "f32[1, 128, 9216]" = torch.ops.aten.sigmoid.default(_to_copy_14)
        mul_14: "f32[1, 128, 9216]" = torch.ops.aten.mul.Tensor(_to_copy_14, sigmoid_1);  _to_copy_14 = sigmoid_1 = None
        _to_copy_15: "bf16[1, 128, 9216]" = torch.ops.aten._to_copy.default(mul_14, dtype = torch.bfloat16);  mul_14 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_10: "bf16[2560, 9216]" = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
        view_24: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_13, [128, 2560]);  _to_copy_13 = None
        mm_5: "bf16[128, 9216]" = torch.ops.aten.mm.default(view_24, permute_10);  view_24 = permute_10 = None
        view_25: "bf16[1, 128, 9216]" = torch.ops.aten.view.default(mm_5, [1, 128, 9216]);  mm_5 = None
        mul_15: "bf16[1, 128, 9216]" = torch.ops.aten.mul.Tensor(_to_copy_15, view_25);  _to_copy_15 = view_25 = None
        permute_11: "bf16[9216, 2560]" = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
        view_26: "bf16[128, 9216]" = torch.ops.aten.view.default(mul_15, [128, 9216]);  mul_15 = None
        mm_6: "bf16[128, 2560]" = torch.ops.aten.mm.default(view_26, permute_11);  view_26 = permute_11 = None
        view_27: "bf16[1, 128, 2560]" = torch.ops.aten.view.default(mm_6, [1, 128, 2560]);  mm_6 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:811 in forward, code: hidden_states = residual + hidden_states
        add_12: "bf16[1, 128, 2560]" = torch.ops.aten.add.Tensor(add_9, view_27);  add_9 = view_27 = None
        return (add_12, cat_5, cat_4)
        