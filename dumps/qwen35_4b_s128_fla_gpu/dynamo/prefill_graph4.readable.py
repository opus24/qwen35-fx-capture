class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "bf16[1, 8192, 128]"):
        l_x_ = L_x_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/causal_conv1d/cpp_functions.py:103 in causal_conv1d_fwd_function, code: out = torch.empty_like(x)
        out: "bf16[1, 8192, 128]" = torch.empty_like(l_x_);  l_x_ = None
        return (out,)
        