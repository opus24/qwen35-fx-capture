


def forward(self, L_hidden_states_ : torch.Tensor, L_self_modules_in_proj_qkv_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_in_proj_z_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_in_proj_b_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_in_proj_a_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_conv1d_parameters_weight_ : torch.nn.parameter.Parameter):
    l_hidden_states_ = L_hidden_states_
    l_self_modules_in_proj_qkv_parameters_weight_ = L_self_modules_in_proj_qkv_parameters_weight_
    l_self_modules_in_proj_z_parameters_weight_ = L_self_modules_in_proj_z_parameters_weight_
    l_self_modules_in_proj_b_parameters_weight_ = L_self_modules_in_proj_b_parameters_weight_
    l_self_modules_in_proj_a_parameters_weight_ = L_self_modules_in_proj_a_parameters_weight_
    l_self_modules_conv1d_parameters_weight_ = L_self_modules_conv1d_parameters_weight_
    mixed_qkv = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_qkv_parameters_weight_, None);  l_self_modules_in_proj_qkv_parameters_weight_ = None
    mixed_qkv_1 = mixed_qkv.transpose(1, 2);  mixed_qkv = None
    z = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_z_parameters_weight_, None);  l_self_modules_in_proj_z_parameters_weight_ = None
    z_1 = z.reshape(1, 128, -1, 128);  z = None
    b = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_b_parameters_weight_, None);  l_self_modules_in_proj_b_parameters_weight_ = None
    a = torch._C._nn.linear(l_hidden_states_, l_self_modules_in_proj_a_parameters_weight_, None);  l_hidden_states_ = l_self_modules_in_proj_a_parameters_weight_ = None
    new_conv_state = torch._C._nn.pad(mixed_qkv_1, (-124, 0), 'constant', None)
    zeros = torch.zeros((1, 8192, 4), dtype = torch.bfloat16, device = device(type='cuda', index=0))
    getitem = new_conv_state[(Ellipsis, slice(-4, None, None))];  new_conv_state = None
    copy_ = zeros.copy_(getitem);  getitem = copy_ = None
    squeeze = l_self_modules_conv1d_parameters_weight_.squeeze(1);  l_self_modules_conv1d_parameters_weight_ = None
    return (mixed_qkv_1, squeeze, z_1, b, a, zeros)
    