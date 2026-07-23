class GraphModule(torch.nn.Module):
    def forward(self, L_hidden_states_: "bf16[1, 128, 2560]", L_self_modules_in_proj_qkv_parameters_weight_: "bf16[8192, 2560]", L_self_modules_in_proj_z_parameters_weight_: "bf16[4096, 2560]", L_self_modules_in_proj_b_parameters_weight_: "bf16[32, 2560]", L_self_modules_in_proj_a_parameters_weight_: "bf16[32, 2560]", L_self_modules_conv1d_parameters_weight_: "bf16[8192, 1, 4]"):
        l_hidden_states_ = L_hidden_states_
        l_self_modules_in_proj_qkv_parameters_weight_ = L_self_modules_in_proj_qkv_parameters_weight_
        l_self_modules_in_proj_z_parameters_weight_ = L_self_modules_in_proj_z_parameters_weight_
        l_self_modules_in_proj_b_parameters_weight_ = L_self_modules_in_proj_b_parameters_weight_
        l_self_modules_in_proj_a_parameters_weight_ = L_self_modules_in_proj_a_parameters_weight_
        l_self_modules_conv1d_parameters_weight_ = L_self_modules_conv1d_parameters_weight_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:464 in forward, code: mixed_qkv = self.in_proj_qkv(hidden_states)
        mixed_qkv: "bf16[1, 128, 8192]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_qkv_parameters_weight_, None);  l_self_modules_in_proj_qkv_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:465 in forward, code: mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv_1: "bf16[1, 8192, 128]" = mixed_qkv.transpose(1, 2);  mixed_qkv = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:467 in forward, code: z = self.in_proj_z(hidden_states)
        z: "bf16[1, 128, 4096]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_z_parameters_weight_, None);  l_self_modules_in_proj_z_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:468 in forward, code: z = z.reshape(batch_size, seq_len, -1, self.head_v_dim)
        z_1: "bf16[1, 128, 32, 128]" = z.reshape(1, 128, -1, 128);  z = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:470 in forward, code: b = self.in_proj_b(hidden_states)
        b: "bf16[1, 128, 32]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_b_parameters_weight_, None);  l_self_modules_in_proj_b_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:471 in forward, code: a = self.in_proj_a(hidden_states)
        a: "bf16[1, 128, 32]" = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_a_parameters_weight_, None);  l_hidden_states_ = l_self_modules_in_proj_a_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/torch/nn/functional.py:5418 in pad, code: return torch._C._nn.pad(input, pad, mode, value)
        new_conv_state: "bf16[1, 8192, 4]" = torch._C._nn.pad(mixed_qkv_1, (-124, 0), 'constant', None)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:954 in lazy_initialization, code: self.conv_states[state_idx] = torch.zeros(
        zeros: "bf16[1, 8192, 4]" = torch.zeros((1, 8192, 4), dtype = torch.bfloat16, device = device(type='cuda', index=0))
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:1004 in update_conv_state, code: self.conv_states[state_idx].copy_(full_conv_states[..., -self.conv_kernel_size[state_idx] :])
        getitem: "bf16[1, 8192, 4]" = new_conv_state[(Ellipsis, slice(-4, None, None))];  new_conv_state = None
        copy_: "bf16[1, 8192, 4]" = zeros.copy_(getitem);  getitem = copy_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:495 in forward, code: weight=self.conv1d.weight.squeeze(1),
        squeeze: "bf16[8192, 4]" = l_self_modules_conv1d_parameters_weight_.squeeze(1);  l_self_modules_conv1d_parameters_weight_ = None
        return (mixed_qkv_1, squeeze, z_1, b, a, zeros)
        