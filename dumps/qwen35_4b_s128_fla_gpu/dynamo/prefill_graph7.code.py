


def forward(self, L_residual_ : torch.Tensor, L_stack0_ : torch.Tensor, L_self_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_residual_ = L_residual_
    l_stack0_ = L_stack0_
    l_self_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_post_attention_layernorm_parameters_weight_
    l_self_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_mlp_modules_gate_proj_parameters_weight_
    l_self_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_mlp_modules_up_proj_parameters_weight_
    l_self_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_down_proj_parameters_weight_
    hidden_states = l_residual_ + l_stack0_;  l_residual_ = l_stack0_ = None
    float_1 = hidden_states.float()
    pow_1 = float_1.pow(2)
    mean = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add_1 = mean + 1e-06;  mean = None
    rsqrt = torch.rsqrt(add_1);  add_1 = None
    output = float_1 * rsqrt;  float_1 = rsqrt = None
    float_2 = l_self_modules_post_attention_layernorm_parameters_weight_.float();  l_self_modules_post_attention_layernorm_parameters_weight_ = None
    add_2 = 1.0 + float_2;  float_2 = None
    output_1 = output * add_2;  output = add_2 = None
    hidden_states_1 = output_1.type_as(hidden_states);  output_1 = None
    linear = torch._C._nn.linear(hidden_states_1, l_self_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear);  linear = None
    linear_1 = torch._C._nn.linear(hidden_states_1, l_self_modules_mlp_modules_up_proj_parameters_weight_, None);  hidden_states_1 = l_self_modules_mlp_modules_up_proj_parameters_weight_ = None
    mul_2 = silu * linear_1;  silu = linear_1 = None
    down_proj = torch._C._nn.linear(mul_2, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_2 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
    hidden_states_2 = hidden_states + down_proj;  hidden_states = down_proj = None
    return (hidden_states_2,)
    