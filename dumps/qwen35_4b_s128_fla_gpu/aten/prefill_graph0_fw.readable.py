class <lambda>(torch.nn.Module):
    def forward(self):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        arange_2: "i64[128]" = torch.ops.aten.arange.start_step(0, 128, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add: "i64[128]" = torch.ops.aten.add.Tensor(arange_2, 0);  arange_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:510 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        arange_3: "i64[128]" = torch.ops.aten.arange.start_step(0, 128, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        add_1: "i64[128]" = torch.ops.aten.add.Tensor(arange_3, 0);  arange_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:366 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        unsqueeze: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add, 0);  add = None
        unsqueeze_1: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
        unsqueeze_2: "i64[1, 1, 128, 1]" = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:367 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        unsqueeze_3: "i64[1, 128]" = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
        unsqueeze_4: "i64[1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
        unsqueeze_5: "i64[1, 1, 1, 128]" = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:80 in causal_mask_function, code: return kv_idx <= q_idx
        le: "b8[1, 1, 128, 128]" = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:518 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        expand: "b8[1, 1, 128, 128]" = torch.ops.aten.expand.default(le, [1, -1, 128, 128]);  le = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:610 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        _tensor_constant0: "bf16[]" = self._tensor_constant0
        lift_fresh_copy: "bf16[]" = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
        scalar_tensor: "bf16[]" = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
        where: "bf16[1, 1, 128, 128]" = torch.ops.aten.where.self(expand, lift_fresh_copy, scalar_tensor);  expand = lift_fresh_copy = scalar_tensor = None
        return (where,)
        