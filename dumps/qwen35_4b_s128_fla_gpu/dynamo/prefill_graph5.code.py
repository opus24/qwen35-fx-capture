


def forward(self, L_stack0_ : torch.Tensor, L_b_ : torch.Tensor, L_self_parameters_A_log_ : torch.nn.parameter.Parameter, L_a_ : torch.Tensor, L_self_parameters_dt_bias_ : torch.nn.parameter.Parameter):
    l_stack0_ = L_stack0_
    l_b_ = L_b_
    l_self_parameters_a_log_ = L_self_parameters_A_log_
    l_a_ = L_a_
    l_self_parameters_dt_bias_ = L_self_parameters_dt_bias_
    mixed_qkv = l_stack0_.transpose(1, 2);  l_stack0_ = None
    split = torch.functional.split(mixed_qkv, [2048, 2048, 4096], dim = -1);  mixed_qkv = None
    query = split[0]
    key = split[1]
    value = split[2];  split = None
    query_1 = query.reshape(1, 128, -1, 128);  query = None
    key_1 = key.reshape(1, 128, -1, 128);  key = None
    value_1 = value.reshape(1, 128, -1, 128);  value = None
    beta = l_b_.sigmoid();  l_b_ = None
    float_1 = l_self_parameters_a_log_.float();  l_self_parameters_a_log_ = None
    exp = float_1.exp();  float_1 = None
    neg = -exp;  exp = None
    float_2 = l_a_.float();  l_a_ = None
    add = float_2 + l_self_parameters_dt_bias_;  float_2 = l_self_parameters_dt_bias_ = None
    softplus = torch._C._nn.softplus(add);  add = None
    g = neg * softplus;  neg = softplus = None
    query_2 = query_1.repeat_interleave(2, dim = 2);  query_1 = None
    key_2 = key_1.repeat_interleave(2, dim = 2);  key_1 = None
    return (query_2, key_2, value_1, g, beta)
    