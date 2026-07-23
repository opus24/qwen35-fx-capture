class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 128, 2560]", arg1_1: "bf16[1, 128, 2560]", arg2_1: "bf16[2560]", arg3_1: "bf16[9216, 2560]", arg4_1: "bf16[9216, 2560]", arg5_1: "bf16[2560, 9216]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:805 in torch_dynamo_resume_in_forward_at_788, code: hidden_states = residual + hidden_states
        add: "bf16[1, 128, 2560]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy: "f32[1, 128, 2560]" = torch.ops.aten._to_copy.default(add, dtype = torch.float32)
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_1: "f32[1, 128, 2560]" = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
        mean: "f32[1, 128, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
        add_1: "f32[1, 128, 1]" = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
        rsqrt: "f32[1, 128, 1]" = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
        mul: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        _to_copy_1: "f32[2560]" = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        add_2: "f32[2560]" = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
        mul_1: "f32[1, 128, 2560]" = torch.ops.aten.mul.Tensor(mul, add_2);  mul = add_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        _to_copy_2: "bf16[1, 128, 2560]" = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute: "bf16[2560, 9216]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        view: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
        mm: "bf16[128, 9216]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        view_1: "bf16[1, 128, 9216]" = torch.ops.aten.view.default(mm, [1, 128, 9216]);  mm = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        _to_copy_3: "f32[1, 128, 9216]" = torch.ops.aten._to_copy.default(view_1, dtype = torch.float32);  view_1 = None
        sigmoid: "f32[1, 128, 9216]" = torch.ops.aten.sigmoid.default(_to_copy_3)
        mul_2: "f32[1, 128, 9216]" = torch.ops.aten.mul.Tensor(_to_copy_3, sigmoid);  _to_copy_3 = sigmoid = None
        _to_copy_4: "bf16[1, 128, 9216]" = torch.ops.aten._to_copy.default(mul_2, dtype = torch.bfloat16);  mul_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        permute_1: "bf16[2560, 9216]" = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
        view_2: "bf16[128, 2560]" = torch.ops.aten.view.default(_to_copy_2, [128, 2560]);  _to_copy_2 = None
        mm_1: "bf16[128, 9216]" = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
        view_3: "bf16[1, 128, 9216]" = torch.ops.aten.view.default(mm_1, [1, 128, 9216]);  mm_1 = None
        mul_3: "bf16[1, 128, 9216]" = torch.ops.aten.mul.Tensor(_to_copy_4, view_3);  _to_copy_4 = view_3 = None
        permute_2: "bf16[9216, 2560]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        view_4: "bf16[128, 9216]" = torch.ops.aten.view.default(mul_3, [128, 9216]);  mul_3 = None
        mm_2: "bf16[128, 2560]" = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
        view_5: "bf16[1, 128, 2560]" = torch.ops.aten.view.default(mm_2, [1, 128, 2560]);  mm_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:811 in torch_dynamo_resume_in_forward_at_788, code: hidden_states = residual + hidden_states
        add_3: "bf16[1, 128, 2560]" = torch.ops.aten.add.Tensor(add, view_5);  add = view_5 = None
        return (add_3,)
        