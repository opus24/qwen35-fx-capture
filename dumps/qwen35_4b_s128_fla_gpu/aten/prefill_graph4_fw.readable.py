class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 8192, 128]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/causal_conv1d/cpp_functions.py:103 in causal_conv1d_fwd_function, code: out = torch.empty_like(x)
        empty_permuted: "bf16[1, 8192, 128]" = torch.ops.aten.empty_permuted.default([1, 8192, 128], [0, 2, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        return (empty_permuted,)
        