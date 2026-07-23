class GraphModule(torch.nn.Module):
    def forward(self):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:507 in sdpa_mask, code: batch_arange = torch.arange(batch_size, device=device)
        batch_arange: "i64[1]" = torch.arange(1, device = device(type='cuda', index=0));  batch_arange = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:508 in sdpa_mask, code: head_arange = torch.arange(1, device=device)
        head_arange: "i64[1]" = torch.arange(1, device = device(type='cuda', index=0));  head_arange = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:509 in sdpa_mask, code: q_arange = torch.arange(q_length, device=device) + q_offset
        arange_2: "i64[128]" = torch.arange(128, device = device(type='cuda', index=0))
        q_arange: "i64[128]" = arange_2 + 0;  arange_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:510 in sdpa_mask, code: kv_arange = torch.arange(kv_length, device=device) + kv_offset
        arange_3: "i64[128]" = torch.arange(128, device = device(type='cuda', index=0))
        kv_arange: "i64[128]" = arange_3 + 0;  arange_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:366 in _non_vmap_expansion_sdpa, code: q_indices = q_indices[None, None, :, None]
        q_indices: "i64[1, 1, 128, 1]" = q_arange[(None, None, slice(None, None, None), None)];  q_arange = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:367 in _non_vmap_expansion_sdpa, code: kv_indices = kv_indices[None, None, None, :]
        kv_indices: "i64[1, 1, 1, 128]" = kv_arange[(None, None, None, slice(None, None, None))];  kv_arange = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:80 in causal_mask_function, code: return kv_idx <= q_idx
        attention_mask: "b8[1, 1, 128, 128]" = kv_indices <= q_indices;  kv_indices = q_indices = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:518 in sdpa_mask, code: attention_mask = attention_mask.expand(batch_size, -1, q_length, kv_length)
        attention_mask_1: "b8[1, 1, 128, 128]" = attention_mask.expand(1, -1, 128, 128);  attention_mask = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/masking_utils.py:610 in eager_mask, code: mask = torch.where(mask, torch.tensor(0.0, device=mask.device, dtype=dtype), min_dtype)
        tensor: "bf16[]" = torch.tensor(0.0, device = device(type='cuda', index=0), dtype = torch.bfloat16)
        mask: "bf16[1, 1, 128, 128]" = torch.where(attention_mask_1, tensor, -3.3895313892515355e+38);  attention_mask_1 = tensor = None
        return (mask,)
        