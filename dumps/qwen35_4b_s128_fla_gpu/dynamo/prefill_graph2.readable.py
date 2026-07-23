class GraphModule(torch.nn.Module):
    def forward(self, L_hidden_states_: "bf16[1, 128, 2560]", L_self_modules_input_layernorm_parameters_weight_: "bf16[2560]"):
        l_hidden_states_ = L_hidden_states_
        l_self_modules_input_layernorm_parameters_weight_ = L_self_modules_input_layernorm_parameters_weight_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_1: "f32[1, 128, 2560]" = l_hidden_states_.float()
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_1: "f32[1, 128, 2560]" = float_1.pow(2)
        mean: "f32[1, 128, 1]" = pow_1.mean(-1, keepdim = True);  pow_1 = None
        add: "f32[1, 128, 1]" = mean + 1e-06;  mean = None
        rsqrt: "f32[1, 128, 1]" = torch.rsqrt(add);  add = None
        output: "f32[1, 128, 2560]" = float_1 * rsqrt;  float_1 = rsqrt = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        float_2: "f32[2560]" = l_self_modules_input_layernorm_parameters_weight_.float();  l_self_modules_input_layernorm_parameters_weight_ = None
        add_1: "f32[2560]" = 1.0 + float_2;  float_2 = None
        output_1: "f32[1, 128, 2560]" = output * add_1;  output = add_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        hidden_states: "bf16[1, 128, 2560]" = output_1.type_as(l_hidden_states_);  output_1 = l_hidden_states_ = None
        return (hidden_states,)
        