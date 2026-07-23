class GraphModule(torch.nn.Module):
    def forward(self, L_args_0_: "bf16[1, 128, 2560]", L_self_modules_input_layernorm_parameters_weight_: "bf16[2560]", L_self_modules_self_attn_modules_q_proj_parameters_weight_: "bf16[8192, 2560]", L_self_modules_self_attn_modules_q_norm_parameters_weight_: "bf16[256]", L_self_modules_self_attn_modules_k_proj_parameters_weight_: "bf16[1024, 2560]", L_self_modules_self_attn_modules_k_norm_parameters_weight_: "bf16[256]", L_self_modules_self_attn_modules_v_proj_parameters_weight_: "bf16[1024, 2560]", L_kwargs_position_embeddings_0_: "bf16[1, 128, 64]", L_kwargs_position_embeddings_1_: "bf16[1, 128, 64]", L_kwargs_attention_mask_: "bf16[1, 1, 128, 128]", L_self_modules_self_attn_modules_o_proj_parameters_weight_: "bf16[2560, 4096]", L_self_modules_post_attention_layernorm_parameters_weight_: "bf16[2560]", L_self_modules_mlp_modules_gate_proj_parameters_weight_: "bf16[9216, 2560]", L_self_modules_mlp_modules_up_proj_parameters_weight_: "bf16[9216, 2560]", L_self_modules_mlp_modules_down_proj_parameters_weight_: "bf16[2560, 9216]"):
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
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_1: "f32[1, 128, 2560]" = l_args_0_.float()
        
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
        hidden_states: "bf16[1, 128, 2560]" = output_1.type_as(l_args_0_);  output_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:688 in forward, code: self.q_proj(hidden_states).view(*input_shape, -1, self.head_dim * 2), 2, dim=-1
        linear: "bf16[1, 128, 8192]" = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_q_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_q_proj_parameters_weight_ = None
        view: "bf16[1, 128, 16, 512]" = linear.view(1, 128, -1, 512);  linear = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:687 in forward, code: query_states, gate = torch.chunk(
        chunk = torch.chunk(view, 2, dim = -1);  view = None
        query_states: "bf16[1, 128, 16, 256]" = chunk[0]
        gate: "bf16[1, 128, 16, 256]" = chunk[1];  chunk = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:690 in forward, code: gate = gate.reshape(*input_shape, -1)
        gate_1: "bf16[1, 128, 4096]" = gate.reshape(1, 128, -1);  gate = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:692 in forward, code: query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        view_1: "bf16[1, 128, 16, 256]" = query_states.view((1, 128, -1, 256));  query_states = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_3: "f32[1, 128, 16, 256]" = view_1.float()
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_2: "f32[1, 128, 16, 256]" = float_3.pow(2)
        mean_1: "f32[1, 128, 16, 1]" = pow_2.mean(-1, keepdim = True);  pow_2 = None
        add_2: "f32[1, 128, 16, 1]" = mean_1 + 1e-06;  mean_1 = None
        rsqrt_1: "f32[1, 128, 16, 1]" = torch.rsqrt(add_2);  add_2 = None
        output_2: "f32[1, 128, 16, 256]" = float_3 * rsqrt_1;  float_3 = rsqrt_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        float_4: "f32[256]" = l_self_modules_self_attn_modules_q_norm_parameters_weight_.float();  l_self_modules_self_attn_modules_q_norm_parameters_weight_ = None
        add_3: "f32[256]" = 1.0 + float_4;  float_4 = None
        output_3: "f32[1, 128, 16, 256]" = output_2 * add_3;  output_2 = add_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        type_as_1: "bf16[1, 128, 16, 256]" = output_3.type_as(view_1);  output_3 = view_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:692 in forward, code: query_states = self.q_norm(query_states.view(hidden_shape)).transpose(1, 2)
        query_states_1: "bf16[1, 16, 128, 256]" = type_as_1.transpose(1, 2);  type_as_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:693 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        linear_1: "bf16[1, 128, 1024]" = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_k_proj_parameters_weight_, None);  l_self_modules_self_attn_modules_k_proj_parameters_weight_ = None
        view_2: "bf16[1, 128, 4, 256]" = linear_1.view((1, 128, -1, 256));  linear_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_5: "f32[1, 128, 4, 256]" = view_2.float()
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_3: "f32[1, 128, 4, 256]" = float_5.pow(2)
        mean_2: "f32[1, 128, 4, 1]" = pow_3.mean(-1, keepdim = True);  pow_3 = None
        add_4: "f32[1, 128, 4, 1]" = mean_2 + 1e-06;  mean_2 = None
        rsqrt_2: "f32[1, 128, 4, 1]" = torch.rsqrt(add_4);  add_4 = None
        output_4: "f32[1, 128, 4, 256]" = float_5 * rsqrt_2;  float_5 = rsqrt_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        float_6: "f32[256]" = l_self_modules_self_attn_modules_k_norm_parameters_weight_.float();  l_self_modules_self_attn_modules_k_norm_parameters_weight_ = None
        add_5: "f32[256]" = 1.0 + float_6;  float_6 = None
        output_5: "f32[1, 128, 4, 256]" = output_4 * add_5;  output_4 = add_5 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        type_as_2: "bf16[1, 128, 4, 256]" = output_5.type_as(view_2);  output_5 = view_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:693 in forward, code: key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states: "bf16[1, 4, 128, 256]" = type_as_2.transpose(1, 2);  type_as_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:694 in forward, code: value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)
        linear_2: "bf16[1, 128, 1024]" = torch._C._nn.linear(hidden_states, l_self_modules_self_attn_modules_v_proj_parameters_weight_, None);  hidden_states = l_self_modules_self_attn_modules_v_proj_parameters_weight_ = None
        view_3: "bf16[1, 128, 4, 256]" = linear_2.view((1, 128, -1, 256));  linear_2 = None
        value_states: "bf16[1, 4, 128, 256]" = view_3.transpose(1, 2);  view_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:594 in apply_rotary_pos_emb, code: cos = cos.unsqueeze(unsqueeze_dim)
        cos: "bf16[1, 1, 128, 64]" = l_kwargs_position_embeddings_0_.unsqueeze(1);  l_kwargs_position_embeddings_0_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:595 in apply_rotary_pos_emb, code: sin = sin.unsqueeze(unsqueeze_dim)
        sin: "bf16[1, 1, 128, 64]" = l_kwargs_position_embeddings_1_.unsqueeze(1);  l_kwargs_position_embeddings_1_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:599 in apply_rotary_pos_emb, code: q_rot, q_pass = q[..., :rotary_dim], q[..., rotary_dim:]
        q_rot: "bf16[1, 16, 128, 64]" = query_states_1[(Ellipsis, slice(None, 64, None))]
        q_pass: "bf16[1, 16, 128, 192]" = query_states_1[(Ellipsis, slice(64, None, None))];  query_states_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:600 in apply_rotary_pos_emb, code: k_rot, k_pass = k[..., :rotary_dim], k[..., rotary_dim:]
        k_rot: "bf16[1, 4, 128, 64]" = key_states[(Ellipsis, slice(None, 64, None))]
        k_pass: "bf16[1, 4, 128, 192]" = key_states[(Ellipsis, slice(64, None, None))];  key_states = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:603 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_6: "bf16[1, 16, 128, 64]" = q_rot * cos
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:568 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        x1: "bf16[1, 16, 128, 32]" = q_rot[(Ellipsis, slice(None, 32, None))]
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:569 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        x2: "bf16[1, 16, 128, 32]" = q_rot[(Ellipsis, slice(32, None, None))];  q_rot = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:570 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg: "bf16[1, 16, 128, 32]" = -x2;  x2 = None
        cat: "bf16[1, 16, 128, 64]" = torch.cat((neg, x1), dim = -1);  neg = x1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:603 in apply_rotary_pos_emb, code: q_embed = (q_rot * cos) + (rotate_half(q_rot) * sin)
        mul_7: "bf16[1, 16, 128, 64]" = cat * sin;  cat = None
        q_embed: "bf16[1, 16, 128, 64]" = mul_6 + mul_7;  mul_6 = mul_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:604 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_8: "bf16[1, 4, 128, 64]" = k_rot * cos;  cos = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:568 in rotate_half, code: x1 = x[..., : x.shape[-1] // 2]
        x1_1: "bf16[1, 4, 128, 32]" = k_rot[(Ellipsis, slice(None, 32, None))]
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:569 in rotate_half, code: x2 = x[..., x.shape[-1] // 2 :]
        x2_1: "bf16[1, 4, 128, 32]" = k_rot[(Ellipsis, slice(32, None, None))];  k_rot = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:570 in rotate_half, code: return torch.cat((-x2, x1), dim=-1)
        neg_1: "bf16[1, 4, 128, 32]" = -x2_1;  x2_1 = None
        cat_1: "bf16[1, 4, 128, 64]" = torch.cat((neg_1, x1_1), dim = -1);  neg_1 = x1_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:604 in apply_rotary_pos_emb, code: k_embed = (k_rot * cos) + (rotate_half(k_rot) * sin)
        mul_9: "bf16[1, 4, 128, 64]" = cat_1 * sin;  cat_1 = sin = None
        k_embed: "bf16[1, 4, 128, 64]" = mul_8 + mul_9;  mul_8 = mul_9 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:607 in apply_rotary_pos_emb, code: q_embed = torch.cat([q_embed, q_pass], dim=-1)
        q_embed_1: "bf16[1, 16, 128, 256]" = torch.cat([q_embed, q_pass], dim = -1);  q_embed = q_pass = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:608 in apply_rotary_pos_emb, code: k_embed = torch.cat([k_embed, k_pass], dim=-1)
        k_embed_1: "bf16[1, 4, 128, 256]" = torch.cat([k_embed, k_pass], dim = -1);  k_embed = k_pass = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:122 in lazy_initialization, code: self.keys = torch.tensor([], dtype=self.dtype, device=self.device)
        tensor: "bf16[0]" = torch.tensor([], dtype = torch.bfloat16, device = device(type='cuda', index=0))
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:123 in lazy_initialization, code: self.values = torch.tensor([], dtype=self.dtype, device=self.device)
        tensor_1: "bf16[0]" = torch.tensor([], dtype = torch.bfloat16, device = device(type='cuda', index=0))
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:143 in update, code: self.keys = torch.cat([self.keys, key_states], dim=-2)
        keys: "bf16[1, 4, 128, 256]" = torch.cat([tensor, k_embed_1], dim = -2);  tensor = k_embed_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/cache_utils.py:144 in update, code: self.values = torch.cat([self.values, value_states], dim=-2)
        values: "bf16[1, 4, 128, 256]" = torch.cat([tensor_1, value_states], dim = -2);  tensor_1 = value_states = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:620 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        getitem_10: "bf16[1, 4, 1, 128, 256]" = keys[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
        hidden_states_1: "bf16[1, 4, 4, 128, 256]" = getitem_10.expand(1, 4, 4, 128, 256);  getitem_10 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:621 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        key_states_1: "bf16[1, 16, 128, 256]" = hidden_states_1.reshape(1, 16, 128, 256);  hidden_states_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:620 in repeat_kv, code: hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
        getitem_11: "bf16[1, 4, 1, 128, 256]" = values[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None), slice(None, None, None))]
        hidden_states_2: "bf16[1, 4, 4, 128, 256]" = getitem_11.expand(1, 4, 4, 128, 256);  getitem_11 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:621 in repeat_kv, code: return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)
        value_states_1: "bf16[1, 16, 128, 256]" = hidden_states_2.reshape(1, 16, 128, 256);  hidden_states_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:637 in eager_attention_forward, code: attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
        transpose_3: "bf16[1, 16, 256, 128]" = key_states_1.transpose(2, 3);  key_states_1 = None
        matmul: "bf16[1, 16, 128, 128]" = torch.matmul(q_embed_1, transpose_3);  q_embed_1 = transpose_3 = None
        attn_weights: "bf16[1, 16, 128, 128]" = matmul * 0.0625;  matmul = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:639 in eager_attention_forward, code: attn_weights = attn_weights + attention_mask
        attn_weights_1: "bf16[1, 16, 128, 128]" = attn_weights + l_kwargs_attention_mask_;  attn_weights = l_kwargs_attention_mask_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:641 in eager_attention_forward, code: attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
        softmax: "f32[1, 16, 128, 128]" = torch.nn.functional.softmax(attn_weights_1, dim = -1, dtype = torch.float32);  attn_weights_1 = None
        attn_weights_2: "bf16[1, 16, 128, 128]" = softmax.to(torch.bfloat16);  softmax = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:642 in eager_attention_forward, code: attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
        attn_weights_3: "bf16[1, 16, 128, 128]" = torch.nn.functional.dropout(attn_weights_2, p = 0.0, training = False);  attn_weights_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:643 in eager_attention_forward, code: attn_output = torch.matmul(attn_weights, value_states)
        attn_output: "bf16[1, 16, 128, 256]" = torch.matmul(attn_weights_3, value_states_1);  attn_weights_3 = value_states_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:644 in eager_attention_forward, code: attn_output = attn_output.transpose(1, 2).contiguous()
        transpose_4: "bf16[1, 128, 16, 256]" = attn_output.transpose(1, 2);  attn_output = None
        attn_output_1: "bf16[1, 128, 16, 256]" = transpose_4.contiguous();  transpose_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:717 in forward, code: attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        reshape_3: "bf16[1, 128, 4096]" = attn_output_1.reshape(1, 128, -1);  attn_output_1 = None
        attn_output_2: "bf16[1, 128, 4096]" = reshape_3.contiguous();  reshape_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:718 in forward, code: attn_output = attn_output * torch.sigmoid(gate)
        sigmoid: "bf16[1, 128, 4096]" = torch.sigmoid(gate_1);  gate_1 = None
        attn_output_3: "bf16[1, 128, 4096]" = attn_output_2 * sigmoid;  attn_output_2 = sigmoid = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:720 in forward, code: attn_output = self.o_proj(attn_output)
        attn_output_4: "bf16[1, 128, 2560]" = torch._C._nn.linear(attn_output_3, l_self_modules_self_attn_modules_o_proj_parameters_weight_, None);  attn_output_3 = l_self_modules_self_attn_modules_o_proj_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:805 in forward, code: hidden_states = residual + hidden_states
        hidden_states_3: "bf16[1, 128, 2560]" = l_args_0_ + attn_output_4;  l_args_0_ = attn_output_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:750 in forward, code: output = self._norm(x.float())
        float_7: "f32[1, 128, 2560]" = hidden_states_3.float()
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:747 in _norm, code: return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        pow_4: "f32[1, 128, 2560]" = float_7.pow(2)
        mean_3: "f32[1, 128, 1]" = pow_4.mean(-1, keepdim = True);  pow_4 = None
        add_10: "f32[1, 128, 1]" = mean_3 + 1e-06;  mean_3 = None
        rsqrt_3: "f32[1, 128, 1]" = torch.rsqrt(add_10);  add_10 = None
        output_6: "f32[1, 128, 2560]" = float_7 * rsqrt_3;  float_7 = rsqrt_3 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:753 in forward, code: output = output * (1.0 + self.weight.float())
        float_8: "f32[2560]" = l_self_modules_post_attention_layernorm_parameters_weight_.float();  l_self_modules_post_attention_layernorm_parameters_weight_ = None
        add_11: "f32[2560]" = 1.0 + float_8;  float_8 = None
        output_7: "f32[1, 128, 2560]" = output_6 * add_11;  output_6 = add_11 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:754 in forward, code: return output.type_as(x)
        hidden_states_4: "bf16[1, 128, 2560]" = output_7.type_as(hidden_states_3);  output_7 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_4: "bf16[1, 128, 9216]" = torch._C._nn.linear(hidden_states_4, l_self_modules_mlp_modules_gate_proj_parameters_weight_, None);  l_self_modules_mlp_modules_gate_proj_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/activations.py:103 in forward, code: return nn.functional.silu(input)
        silu: "bf16[1, 128, 9216]" = torch.nn.functional.silu(linear_4);  linear_4 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:736 in forward, code: down_proj = self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))
        linear_5: "bf16[1, 128, 9216]" = torch._C._nn.linear(hidden_states_4, l_self_modules_mlp_modules_up_proj_parameters_weight_, None);  hidden_states_4 = l_self_modules_mlp_modules_up_proj_parameters_weight_ = None
        mul_14: "bf16[1, 128, 9216]" = silu * linear_5;  silu = linear_5 = None
        down_proj: "bf16[1, 128, 2560]" = torch._C._nn.linear(mul_14, l_self_modules_mlp_modules_down_proj_parameters_weight_, None);  mul_14 = l_self_modules_mlp_modules_down_proj_parameters_weight_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:811 in forward, code: hidden_states = residual + hidden_states
        hidden_states_5: "bf16[1, 128, 2560]" = hidden_states_3 + down_proj;  hidden_states_3 = down_proj = None
        return (hidden_states_5, values, keys)
        