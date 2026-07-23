class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 128, 2560]", arg1_1: "bf16[2560]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        _to_copy: "f32[1, 128, 2560]" = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
        
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
        return (_to_copy_2,)
        