class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_: "bf16[1, 8192, 128]", L_b_: "bf16[1, 128, 32]", L_self_parameters_A_log_: "bf16[32]", L_a_: "bf16[1, 128, 32]", L_self_parameters_dt_bias_: "bf16[32]"):
        l_stack0_ = L_stack0_
        l_b_ = L_b_
        l_self_parameters_a_log_ = L_self_parameters_A_log_
        l_a_ = L_a_
        l_self_parameters_dt_bias_ = L_self_parameters_dt_bias_
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:505 in torch_dynamo_resume_in_forward_at_493, code: mixed_qkv = mixed_qkv.transpose(1, 2)
        mixed_qkv: "bf16[1, 128, 8192]" = l_stack0_.transpose(1, 2);  l_stack0_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:506 in torch_dynamo_resume_in_forward_at_493, code: query, key, value = torch.split(
        split = torch.functional.split(mixed_qkv, [2048, 2048, 4096], dim = -1);  mixed_qkv = None
        query: "bf16[1, 128, 2048]" = split[0]
        key: "bf16[1, 128, 2048]" = split[1]
        value: "bf16[1, 128, 4096]" = split[2];  split = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:516 in torch_dynamo_resume_in_forward_at_493, code: query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        query_1: "bf16[1, 128, 16, 128]" = query.reshape(1, 128, -1, 128);  query = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:517 in torch_dynamo_resume_in_forward_at_493, code: key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        key_1: "bf16[1, 128, 16, 128]" = key.reshape(1, 128, -1, 128);  key = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:518 in torch_dynamo_resume_in_forward_at_493, code: value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        value_1: "bf16[1, 128, 32, 128]" = value.reshape(1, 128, -1, 128);  value = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:520 in torch_dynamo_resume_in_forward_at_493, code: beta = b.sigmoid()
        beta: "bf16[1, 128, 32]" = l_b_.sigmoid();  l_b_ = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:522 in torch_dynamo_resume_in_forward_at_493, code: g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        float_1: "f32[32]" = l_self_parameters_a_log_.float();  l_self_parameters_a_log_ = None
        exp: "f32[32]" = float_1.exp();  float_1 = None
        neg: "f32[32]" = -exp;  exp = None
        float_2: "f32[1, 128, 32]" = l_a_.float();  l_a_ = None
        add: "f32[1, 128, 32]" = float_2 + l_self_parameters_dt_bias_;  float_2 = l_self_parameters_dt_bias_ = None
        softplus: "f32[1, 128, 32]" = torch._C._nn.softplus(add);  add = None
        g: "f32[1, 128, 32]" = neg * softplus;  neg = softplus = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:524 in torch_dynamo_resume_in_forward_at_493, code: query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        query_2: "bf16[1, 128, 32, 128]" = query_1.repeat_interleave(2, dim = 2);  query_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:525 in torch_dynamo_resume_in_forward_at_493, code: key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        key_2: "bf16[1, 128, 32, 128]" = key_1.repeat_interleave(2, dim = 2);  key_1 = None
        return (query_2, key_2, value_1, g, beta)
        