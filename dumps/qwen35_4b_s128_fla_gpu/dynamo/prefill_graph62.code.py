


def forward(self, L_x_ : torch.Tensor, L_self_parameters_weight_ : torch.nn.parameter.Parameter):
    l_x_ = L_x_
    l_self_parameters_weight_ = L_self_parameters_weight_
    float_1 = l_x_.float()
    pow_1 = float_1.pow(2)
    mean = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = mean + 1e-06;  mean = None
    rsqrt = torch.rsqrt(add);  add = None
    output = float_1 * rsqrt;  float_1 = rsqrt = None
    float_2 = l_self_parameters_weight_.float();  l_self_parameters_weight_ = None
    add_1 = 1.0 + float_2;  float_2 = None
    output_1 = output * add_1;  output = add_1 = None
    type_as = output_1.type_as(l_x_);  output_1 = l_x_ = None
    return (type_as,)
    