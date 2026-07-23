


def forward(self, L_stack0_1_ : torch.Tensor, L_stack0_0_ : torch.Tensor, L_z_ : torch.Tensor, L_self_modules_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_out_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_stack0_1_ = L_stack0_1_
    l_stack0_0_ = L_stack0_0_
    l_z_ = L_z_
    l_self_modules_norm_parameters_weight_ = L_self_modules_norm_parameters_weight_
    l_self_modules_out_proj_parameters_weight_ = L_self_modules_out_proj_parameters_weight_
    recurrent_states = torch.zeros_like(l_stack0_1_)
    copy_ = recurrent_states.copy_(l_stack0_1_);  l_stack0_1_ = copy_ = None
    core_attn_out = l_stack0_0_.reshape(-1, 128);  l_stack0_0_ = None
    z = l_z_.reshape(-1, 128);  l_z_ = None
    contiguous = core_attn_out.contiguous();  core_attn_out = None
    contiguous_1 = z.contiguous();  z = None
    contiguous_2 = l_self_modules_norm_parameters_weight_.contiguous();  l_self_modules_norm_parameters_weight_ = None
    _cuda_exchange_device = torch._C._cuda_exchangeDevice(0)
    x = contiguous.reshape(-1, 128);  contiguous = None
    g = contiguous_1.reshape(-1, 128);  contiguous_1 = None
    y = torch.empty_like(x, dtype = torch.bfloat16)
    rstd = torch.empty((4096,), dtype = torch.float32, device = device(type='cuda', index=0))
    triton_kernel_wrapper_mutation = torch.ops.higher_order.triton_kernel_wrapper_mutation(kernel_idx = 0, constant_args_idx = 27, grid = [(256, 1, 1), (256, 1, 1), (256, 1, 1), (128, 1, 1), (128, 1, 1), (128, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x': x, 'g': g, 'y': y, 'w': contiguous_2, 'rstd': rstd});  x = g = contiguous_2 = rstd = triton_kernel_wrapper_mutation = None
    y_1 = y.reshape((4096, 128));  y = None
    _cuda_maybe_exchange_device = torch._C._cuda_maybeExchangeDevice(_cuda_exchange_device);  _cuda_exchange_device = _cuda_maybe_exchange_device = None
    core_attn_out_1 = y_1.reshape(1, 128, -1);  y_1 = None
    output = torch._C._nn.linear(core_attn_out_1, l_self_modules_out_proj_parameters_weight_, None);  core_attn_out_1 = l_self_modules_out_proj_parameters_weight_ = None
    return (output, recurrent_states)
    