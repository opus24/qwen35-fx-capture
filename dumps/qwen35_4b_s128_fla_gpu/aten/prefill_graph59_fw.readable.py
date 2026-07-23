class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 128, 2560]", arg1_1: "bf16[8192, 2560]", arg2_1: "bf16[4096, 2560]", arg3_1: "bf16[32, 2560]", arg4_1: "bf16[32, 2560]", arg5_1: "bf16[8192, 1, 4]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:464 in forward, code: mixed_qkv = self.in_proj_qkv(hidden_states)
        permute: "bf16[2560, 8192]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        view: "bf16[128, 2560]" = torch.ops.aten.view.default(arg0_1, [128, 2560])
        mm: "bf16[128, 8192]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 128, 8192]" = torch.ops.aten.view.default(mm, [1, 128, 8192]);  mm = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:465 in forward, code: mixed_qkv = mixed_qkv.transpose(1, 2)
        permute_1: "bf16[1, 8192, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:467 in forward, code: z = self.in_proj_z(hidden_states)
        permute_2: "bf16[2560, 4096]" = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
        view_2: "bf16[128, 2560]" = torch.ops.aten.view.default(arg0_1, [128, 2560])
        mm_1: "bf16[128, 4096]" = torch.ops.aten.mm.default(view_2, permute_2);  view_2 = permute_2 = None
        view_3: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(mm_1, [1, 128, 4096]);  mm_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:468 in forward, code: z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        view_4: "bf16[1, 128, 32, 128]" = torch.ops.aten.view.default(view_3, [1, 128, -1, 128]);  view_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:470 in forward, code: b = self.in_proj_b(hidden_states)
        permute_3: "bf16[2560, 32]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view_5: "bf16[128, 2560]" = torch.ops.aten.view.default(arg0_1, [128, 2560])
        mm_2: "bf16[128, 32]" = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
        view_6: "bf16[1, 128, 32]" = torch.ops.aten.view.default(mm_2, [1, 128, 32]);  mm_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:471 in forward, code: a = self.in_proj_a(hidden_states)
        permute_4: "bf16[2560, 32]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_7: "bf16[128, 2560]" = torch.ops.aten.view.default(arg0_1, [128, 2560]);  arg0_1 = None
        mm_3: "bf16[128, 32]" = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
        view_8: "bf16[1, 128, 32]" = torch.ops.aten.view.default(mm_3, [1, 128, 32]);  mm_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/torch/nn/functional.py:5418 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        constant_pad_nd: "bf16[1, 8192, 4]" = torch.ops.aten.constant_pad_nd.default(permute_1, [-124, 0], 0.0)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:954 in lazy_initialization, code: self.conv_states[state_idx] = torch.zeros(
        full: "bf16[1, 8192, 4]" = torch.ops.aten.full.default([1, 8192, 4], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:1004 in update_conv_state, code: self.conv_states[state_idx].copy_(full_conv_states[..., -self.conv_kernel_size[state_idx] :])
        slice_1: "bf16[1, 8192, 4]" = torch.ops.aten.slice.Tensor(constant_pad_nd, 2, -4, 9223372036854775807);  constant_pad_nd = None
        copy: "bf16[1, 8192, 4]" = torch.ops.aten.copy.default(full, slice_1);  full = slice_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:495 in forward, code: weight=self.conv1d.weight.squeeze(1),
        squeeze: "bf16[8192, 4]" = torch.ops.aten.squeeze.dims(arg5_1, [1]);  arg5_1 = None
        return (permute_1, squeeze, view_4, view_6, view_8, copy)
        