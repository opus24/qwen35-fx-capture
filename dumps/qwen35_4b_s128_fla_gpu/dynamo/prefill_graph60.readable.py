class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_1_: "f32[1, 32, 128, 128]", L_stack0_0_: "bf16[1, 128, 32, 128]", L_z_: "bf16[1, 128, 32, 128]", L_self_modules_norm_parameters_weight_: "bf16[128]", L_self_modules_out_proj_parameters_weight_: "bf16[2560, 4096]"):
        l_stack0_1_ = L_stack0_1_
        l_stack0_0_ = L_stack0_0_
        l_z_ = L_z_
        l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
        l_self_modules_out_proj_parameters_weight_ = L_self_modules_out_proj_parameters_weight_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:966 in lazy_initialization, code: self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
        recurrent_states: "f32[1, 32, 128, 128]" = torch.zeros_like(l_stack0_1_)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:1025 in update_recurrent_state, code: self.recurrent_states[state_idx].copy_(recurrent_states)
        copy_: "f32[1, 32, 128, 128]" = recurrent_states.copy_(l_stack0_1_);  l_stack0_1_ = copy_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:557 in torch_dynamo_resume_in_forward_at_539, code: core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        core_attn_out: "bf16[4096, 128]" = l_stack0_0_.reshape(-1, 128);  l_stack0_0_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:558 in torch_dynamo_resume_in_forward_at_539, code: z = z.reshape(-1, self.head_v_dim)
        z: "bf16[4096, 128]" = l_z_.reshape(-1, 128);  l_z_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/utils/_decorators.py:93 in _contiguous_if_needed, code: return arg.contiguous()
        contiguous: "bf16[4096, 128]" = core_attn_out.contiguous();  core_attn_out = None
        contiguous_1: "bf16[4096, 128]" = z.contiguous();  z = None
        contiguous_2: "bf16[128]" = l_self_modules_norm_parameters_weight_.contiguous();  l_self_modules_norm_parameters_weight_ = None
        
        # No stacktrace found for following nodes
        _cuda_exchange_device = torch._C._cuda_exchangeDevice(0)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:653 in forward, code: x = x.reshape(-1, x.shape[-1])
        x: "bf16[4096, 128]" = contiguous.reshape(-1, 128);  contiguous = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:654 in forward, code: g = g.reshape(-1, g.shape[-1])
        g: "bf16[4096, 128]" = contiguous_1.reshape(-1, 128);  contiguous_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:465 in layer_norm_gated_fwd, code: y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
        y: "bf16[4096, 128]" = torch.empty_like(x, dtype = torch.bfloat16)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:471 in layer_norm_gated_fwd, code: rstd = torch.empty((T,), dtype=torch.float, device=x.device)
        rstd: "f32[4096]" = torch.empty((4096,), dtype = torch.float32, device = device(type='cuda', index=0))
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/triton/runtime/autotuner.py:459 in run, code: return self.fn.run(*args, **kwargs)
        triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 47, grid = [(256, 1, 1), (256, 1, 1), (256, 1, 1), (128, 1, 1), (128, 1, 1), (128, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x': x, 'g': g, 'y': y, 'w': contiguous_2, 'rstd': rstd});  x = g = contiguous_2 = rstd = triton_kernel_wrapper_mutation = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:679 in forward, code: y = y.reshape(x_shape_og)
        y_1: "bf16[4096, 128]" = y.reshape((4096, 128));  y = None
        
        # No stacktrace found for following nodes
        _cuda_maybe_exchange_device = torch._C._cuda_maybeExchangeDevice(_cuda_exchange_device);  _cuda_exchange_device = _cuda_maybe_exchange_device = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:560 in torch_dynamo_resume_in_forward_at_539, code: core_attn_out = core_attn_out.reshape(batch_size, seq_len, -1)
        core_attn_out_1: "bf16[1, 128, 4096]" = y_1.reshape(1, 128, -1);  y_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:562 in torch_dynamo_resume_in_forward_at_539, code: output = self.out_proj(core_attn_out)
        output: "bf16[1, 128, 2560]" = torch._C._nn.linear(core_attn_out_1, l_self_modules_out_proj_parameters_weight_, None);  core_attn_out_1 = l_self_modules_out_proj_parameters_weight_ = None
        return (output, recurrent_states)
        