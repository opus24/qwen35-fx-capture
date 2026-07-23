


def forward(self, L_args_0_ : torch.Tensor, L_self_modules_input_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_q_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_q_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_k_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_k_norm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_self_attn_modules_v_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_kwargs_position_embeddings_0_ : torch.Tensor, L_kwargs_position_embeddings_1_ : torch.Tensor, L_kwargs_attention_mask_ : torch.Tensor, L_self_modules_self_attn_modules_o_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_post_attention_layernorm_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_gate_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_up_proj_parameters_weight_ : torch.nn.parameter.Parameter, L_self_modules_mlp_modules_down_proj_parameters_weight_ : torch.nn.parameter.Parameter):
    l_args_0_ = L_args_0_
    l_self_modules_input_layernorm_parameters_weight_ = L_self_modules_input_layernorm_parameters_weight_
    l_self_modules_self_attn_modules_q_proj_parameters_weight_ = L_self_modules_self_attn_modules_q_proj_parameters_weight_
    l_self_modules_self_attn_modules_q_norm_parameters_weight_ = L_self_modules_self_attn_modules_q_norm_parameters_weight_
    l_self_modules_self_attn_modules_k_proj_parameters_weight_ = L_self_modules_self_attn_modules_k_proj_parameters_weight_
    l_self_modules_self_attn_modules_k_norm_parameters_weight_ = L_self_modules_self_attn_modules_k_norm_parameters_weight_
    l_self_modules_self_attn_modules_v_proj_parameters_weight_ = L_self_modules_self_attn_modules_v_proj_parameters_weight_
    l_kwargs_position_embeddings_0_ = L_kwargs_position_embeddings_0_
    l_kwargs_position_embeddings_1_ = L_kwargs_position_embeddings_1_
    l_kwargs_attention_mask_ = L_kwargs_attention_mask_
    l_self_modules_self_attn_modules_o_proj_parameters_weight_ = L_self_modules_self_attn_modules_o_proj_parameters_weight_
    l_self_modules_post_attention_layernorm_parameters_weight_ = L_self_modules_post_attention_layernorm_parameters_weight_
    l_self_modules_mlp_modules_gate_proj_parameters_weight_ = L_self_modules_mlp_modules_gate_proj_parameters_weight_
    l_self_modules_mlp_modules_up_proj_parameters_weight_ = L_self_modules_mlp_modules_up_proj_parameters_weight_
    l_self_modules_mlp_modules_down_proj_parameters_weight_ = L_self_modules_mlp_modules_down_proj_parameters_weight_
    float_1 = l_args_0_.float()
    pow_1 = float_1.pow(2)
    mean = pow_1.mean(-1, keepdim = True);  pow_1 = None
    add = mean + 1e-06;  mean = None
    rsqrt = torch.rsqrt(add);  add = None
    output = float_1 * rsqrt;  float_1 = rsqrt = None
    float_2 = l_self_modules_input_layernorm_parameters_weight_.float();  l_self_modules_input_layernorm_parameters_weight_ = None
    add_1 = 1.0 + float_2;  float_2 = None
    output_1 = output * add_1;  output = add_1 = None
    hidden_states = output_1.type_as(l_args_0_);  output_1 = None
    linear = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_q_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_q_proj_parameters_weight_ = None
    view = linear.view(1, 128, -1, 512);  linear = None
    chunk = torch.chunk(view, 2, dim = -1);  view = None
    query_states = chunk[0]
    gate = chunk[1];  chunk = None
    gate_1 = gate.reshape(1, 128, -1);  gate = None
    view_1 = query_states.view((1, 128, -1, 256));  query_states = None
    float_3 = view_1.float()
    pow_2 = float_3.pow(2)
    mean_1 = pow_2.mean(-1, keepdim = True);  pow_2 = None
    add_2 = mean_1 + 1e-06;  mean_1 = None
    rsqrt_1 = torch.rsqrt(add_2);  add_2 = None
    output_2 = float_3 * rsqrt_1;  float_3 = rsqrt_1 = None
    float_4 = l_self_modules_self_attn_modules_q_norm_parameters_weight_.float();  l_self_modules_self_attn_modules_q_norm_parameters_weight_ = None
    add_3 = 1.0 + float_4;  float_4 = None
    output_3 = output_2 * add_3;  output_2 = add_3 = None
    type_as_1 = output_3.type_as(view_1);  output_3 = view_1 = None
    query_states_1 = type_as_1.transpose(1, 2);  type_as_1 = None
    linear_1 = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_k_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_k_proj_parameters_weight_ = None
    view_2 = linear_1.view((1, 128, -1, 256));  linear_1 = None
    float_5 = view_2.float()
    pow_3 = float_5.pow(2)
    mean_2 = pow_3.mean(-1, keepdim = True);  pow_3 = None
    add_4 = mean_2 + 1e-06;  mean_2 = None
    rsqrt_2 = torch.rsqrt(add_4);  add_4 = None
    output_4 = float_5 * rsqrt_2;  float_5 = rsqrt_2 = None
    float_6 = l_self_modules_self_attn_modules_k_norm_parameters_weight_.float();  l_self_modules_self_attn_modules_k_norm_parameters_weight_ = None
    add_5 = 1.0 + float_6;  float_6 = None
    output_5 = output_4 * add_5;  output_4 = add_5 = None
    type_as_2 = output_5.type_as(view_2);  output_5 = view_2 = None
    key_states = type_as_2.transpose(1, 2);  type_as_2 = None
    linear_2 = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_v_proj_parameters_weight_, None);  hidden_states = l_self_modules_self_attn_modules_v_proj_parameters_weight_ = None
    view_3 = linear_2.view((1, 128, -1, 256));  linear_2 = None
    value_states = view_3.transpose(1, 2);  view_3 = None
    cos = l_kwargs_position_embeddings_0_.unsqueeze(1);  l_kwargs_position_embeddings_0_ = None
    sin = l_kwargs_position_embeddings_1_.unsqueeze(1);  l_kwargs_position_embeddings_1_ = None
    q_rot = query_states_1[(Ellipsis, slice(None, 64, None))]
    q_pass = query_states_1[(Ellipsis, slice(64, None, None))];  query_states_1 = None
    k_rot = key_states[(Ellipsis, slice(None, 64, None))]
    k_pass = key_states[(Ellipsis, slice(64, None, None))];  key_states = None
    mul_6 = q_rot * cos
    x1 = q_rot[(Ellipsis, slice(None, 32, None))]
    x2 = q_rot[(Ellipsis, slice(32, None, None))];  q_rot = None
    neg = -x2;  x2 = None
    cat = torch.cat((neg, x1), dim = -1);  neg = x1 = None
    mul_7 = cat * sin;  cat = None
    q_embed = mul_6 + mul_7;  mul_6 = mul_7 = None
    mul_8 = k_rot * cos;  cos = None
    x1_1 = k_rot[(Ellipsis, slice(None, 32, None))]
    x2_1 = k_rot[(Ellipsis, slice(32, None, None))];  k_rot = None
    neg_1 = -x2_1;  x2_1 = None
    cat_1 = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
    mul_9 = cat_1 * sin;  cat_1 = sin = None
    k_embed = mul_8 + mul_9;  mul_8 = mul_9 = None
    q_embed_1 = torch.cat([q_embed, q_pass], dim = -1);  q_embed = q_pass = None
    k_embed_1 = torch.cat([k_embed, k_pass], dim = -1);  k_embed = k_pass = None
    tensor = torch.tensor([], dtype = torch.bfloat16, device = device(type='cuda', index=0))
    tensor_1 = torch.tensor([], dtype = torch.bfloat16, device = device(type='cuda', index=0))
    keys = torch.cat([tensor, k_embed_1], dim = -2);  tensor = k_embed_1 = None
    values = torch.cat([tensor_1, value_states], dim = -2);  tensor_1 = value_states = None
    getitem_10 = keys[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
    hidden_states_1 = getitem_10.expand(1, 4, 4, 128, 256);  getitem_10 = None
    key_states_1 = hidden_states_1.reshape(1, 16, 128, 256);  hidden_states_1 = None
    getitem_11 = values[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
    hidden_states_2 = getitem_11.expand(1, 4, 4, 128, 256);  getitem_11 = None
    value_states_1 = hidden_states_2.reshape(1, 16, 128, 256);  hidden_states_2 = None
    transpose_3 = key_states_1.transpose(2, 3);  key_states_1 = None
    matmul = torch.matmul(q_embed_1, transpose_3);  q_embed_1 = transpose_3 = None
    attn_weights = matmul * 0.0625;  matmul = None
    attn_weights_1 = attn_weights + l_kwargs_attention_mask_;  attn_weights = l_kwargs_attention_mask_ = None
    softmax = torch.nn.functional.softmax(attn_weights_1, dim = -1, dtype = torch.float32);  attn_weights_1 = None
    attn_weights_2 = softmax.to(torch.bfloat16);  softmax = None
    attn_weights_3 = torch.nn.functional.dropout(attn_weights_2, p = 0.0, training = False);  attn_weights_2 = None
    attn_output = torch.matmul(attn_weights_3, value_states_1);  attn_weights_3 = value_states_1 = None
    transpose_4 = attn_output.transpose(1, 2);  attn_output = None
    attn_output_1 = transpose_4.contiguous();  transpose_4 = None
    reshape_3 = attn_output_1.reshape(1, 128, -1);  attn_output_1 = None
    attn_output_2 = reshape_3.contiguous();  reshape_3 = None
    sigmoid = torch.sigmoid(gate_1);  gate_1 = None
    attn_output_3 = attn_output_2 * sigmoid;  attn_output_2 = sigmoid = None
    attn_output_4 = torch._C._nn.linear(attn_output_3, l_self_modules_self_attn_modules_o_proj_parameters_weight_, None);  attn_output_3 = l_self_modules_self_attn_modules_o_proj_parameters_weight_ = None
    hidden_states_3 = l_args_0_ + attn_output_4;  l_args_0_ = attn_output_4 = None
    float_7 = hidden_states_3.float()
    pow_4 = float_7.pow(2)
    mean_3 = pow_4.mean(-1, keepdim = True);  pow_4 = None
    add_10 = mean_3 + 1e-06;  mean_3 = None
    rsqrt_3 = torch.rsqrt(add_10);  add_10 = None
    output_6 = float_7 * rsqrt_3;  float_7 = rsqrt_3 = None
    float_8 = l_self_modules_post_attention_layernorm_parameters_weight_.float();  l_self_modules_post_attention_layernorm_parameters_weight_ = None
    add_11 = 1.0 + float_8;  float_8 = None
    output_7 = output_6 * add_11;  output_6 = add_11 = None
    hidden_states_4 = output_7.type_as(hidden_states_3);  output_7 = None
    linear_4 = torch._C._nn.linear(hidden_states_4, l_self_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_mlp_modules_gate_proj_parameters_weight_ = None
    silu = torch.nn.functional.silu(linear_4);  linear_4 = None
    linear_5 = torch._C._nn.linear(hidden_states_4, l_self_modules_mlp_modules_up_proj_parameters_weight_, None);  hidden_states_4 = l_self_modules_mlp_modules_up_proj_parameters_weight_ = None
    mul_14 = silu * linear_5;  silu = linear_5 = None
    down_proj = torch._C._nn.linear(mul_14, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_14 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
    hidden_states_5 = hidden_states_3 + down_proj;  hidden_states_3 = down_proj = None
    return (hidden_states_5, values, keys)
    