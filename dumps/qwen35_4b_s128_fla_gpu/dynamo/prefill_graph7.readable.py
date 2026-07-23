class GraphModule(torch.nn.Module):
    def forward(self, L_residual_: "bf16[1, 128, 2560]", L_stack0_: "bf16[1, 128, 2560]", L_self_modules_post_attention_layernorm_parameters_weight_: "bf16[2560]", L_self_modules_mlp_modules_gate_proj_parameters_weight_: "bf16[9216, 2560]", L_self_modules_mlp_modules_up_proj_parameters_weight_: "bf16[9216, 2560]", L_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[2560, 9216]"):
        l_residual_ = L_residual_
        l_stack0_ = L_stack0_
        l_self_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_post_attention_layernorm_parameters_weight_
        l_self_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_mlp_modules_gate_proj_parameters_weight_
        l_self_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_mlp_modules_up_proj_parameters_weight_
        l_self_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_down_proj_parameters_weight_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:805 in torch_dynamo_resume_in_forward_at_788, code: hidden_states = residual + hidden_states
        hidden_states: "bf16[1, 128, 2560]" = l_residual_ + l_stack0_;  l_residual_ = l_stack0_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_1: "f32[1, 128, 2560]" = hidden_states.float()
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_1: "f32[1, 128, 2560]" = float_1.pow(2)
        mean: "f32[1, 128, 1]" = pow_1.mean(-1, keepdim = True);  pow_1 = None
        add_1: "f32[1, 128, 1]" = mean + 1e-06;  mean = None
        rsqrt: "f32[1, 128, 1]" = torch.rsqrt(add_1);  add_1 = None
        output: "f32[1, 128, 2560]" = float_1 * rsqrt;  float_1 = rsqrt = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        float_2: "f32[2560]" = l_self_modules_post_attention_layernorm_parameters_weight_.float();  l_self_modules_post_attention_layernorm_parameters_weight_ = None
        add_2: "f32[2560]" = 1.0 + float_2;  float_2 = None
        output_1: "f32[1, 128, 2560]" = output * add_2;  output = add_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        hidden_states_1: "bf16[1, 128, 2560]" = output_1.type_as(hidden_states);  output_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear: "bf16[1, 128, 9216]" = torch._C._nn.linear(hidden_states_1, l_self_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_mlp_modules_gate_proj_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        silu: "bf16[1, 128, 9216]" = torch.nn.functional.silu(linear);  linear = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_1: "bf16[1, 128, 9216]" = torch._C._nn.linear(hidden_states_1, l_self_modules_mlp_modules_up_proj_parameters_weight_, None);  hidden_states_1 = l_self_modules_mlp_modules_up_proj_parameters_weight_ = None
        mul_2: "bf16[1, 128, 9216]" = silu * linear_1;  silu = linear_1 = None
        down_proj: "bf16[1, 128, 2560]" = torch._C._nn.linear(mul_2, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_2 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:811 in torch_dynamo_resume_in_forward_at_788, code: hidden_states = residual + hidden_states
        hidden_states_2: "bf16[1, 128, 2560]" = hidden_states + down_proj;  hidden_states = down_proj = None
        return (hidden_states_2,)
        