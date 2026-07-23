class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "f32[1, 32, 128, 128]", arg1_1: "bf16[1, 128, 32, 128]", arg2_1: "bf16[1, 128, 32, 128]", arg3_1: "bf16[128]", arg4_1: "bf16[2560, 4096]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:966 in lazy_initialization, code: self.recurrent_states[state_idx] = torch.zeros_like(recurrent_states)
        full_like: "f32[1, 32, 128, 128]" = torch.ops.aten.full_like.default(arg0_1, 0, pin_memory = False, memory_format = torch.preserve_format)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:1025 in update_recurrent_state, code: self.recurrent_states[state_idx].copy_(recurrent_states)
        copy: "f32[1, 32, 128, 128]" = torch.ops.aten.copy.default(full_like, arg0_1);  full_like = arg0_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:557 in torch_dynamo_resume_in_forward_at_539, code: core_attn_out = core_attn_out.reshape(-1, self.head_v_dim)
        view: "bf16[4096, 128]" = torch.ops.aten.view.default(arg1_1, [-1, 128]);  arg1_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:558 in torch_dynamo_resume_in_forward_at_539, code: z = z.reshape(-1, self.head_v_dim)
        view_1: "bf16[4096, 128]" = torch.ops.aten.view.default(arg2_1, [-1, 128]);  arg2_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:653 in forward, code: x = x.reshape(-1, x.shape[-1])
        view_2: "bf16[4096, 128]" = torch.ops.aten.view.default(view, [-1, 128]);  view = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:654 in forward, code: g = g.reshape(-1, g.shape[-1])
        view_3: "bf16[4096, 128]" = torch.ops.aten.view.default(view_1, [-1, 128]);  view_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:465 in layer_norm_gated_fwd, code: y = torch.empty_like(x, dtype=x.dtype if out_dtype is None else out_dtype)
        empty_permuted: "bf16[4096, 128]" = torch.ops.aten.empty_permuted.default([4096, 128], [0, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/fla/modules/fused_norm_gate.py:471 in layer_norm_gated_fwd, code: rstd = torch.empty((T,), dtype=torch.float, device=x.device)
        empty: "f32[4096]" = torch.ops.aten.empty.memory_format([4096], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/triton/runtime/autotuner.py:459 in run, code: return self.fn.run(*args, **kwargs)
        triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 33, grid = [(256, 1, 1), (256, 1, 1), (256, 1, 1), (128, 1, 1), (128, 1, 1), (128, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x': view_2, 'g': view_3, 'y': empty_permuted, 'w': arg3_1, 'rstd': empty}, tensors_to_clone = ['y', 'rstd']);  view_2 = view_3 = empty_permuted = arg3_1 = empty = None
        getitem: "bf16[4096, 128]" = triton_kernel_wrapper_functional_proxy['y'];  triton_kernel_wrapper_functional_proxy = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:562 in torch_dynamo_resume_in_forward_at_539, code: output = self.out_proj(core_attn_out)
        permute: "bf16[4096, 2560]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_10: "bf16[4096, 128]" = torch.ops.aten.view.default(getitem, [4096, 128]);  getitem = None
        view_11: "bf16[1, 128, 4096]" = torch.ops.aten.view.default(view_10, [1, 128, -1]);  view_10 = None
        view_12: "bf16[128, 4096]" = torch.ops.aten.view.default(view_11, [128, 4096]);  view_11 = None
        mm: "bf16[128, 2560]" = torch.ops.aten.mm.default(view_12, permute);  view_12 = permute = None
        view_13: "bf16[1, 128, 2560]" = torch.ops.aten.view.default(mm, [1, 128, 2560]);  mm = None
        return (view_13, copy)
        