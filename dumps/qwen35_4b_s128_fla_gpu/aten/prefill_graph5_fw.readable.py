class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[1, 8192, 128]", arg1_1: "bf16[1, 128, 32]", arg2_1: "bf16[32]", arg3_1: "bf16[1, 128, 32]", arg4_1: "bf16[32]"):
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:505 in torch_dynamo_resume_in_forward_at_493, code: mixed_qkv = mixed_qkv.transpose(1, 2)
        permute: "bf16[1, 128, 8192]" = torch.ops.aten.permute.default(arg0_1, [0, 2, 1]);  arg0_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:506 in torch_dynamo_resume_in_forward_at_493, code: query, key, value = torch.split(
        split_with_sizes = torch.ops.aten.split_with_sizes.default(permute, [2048, 2048, 4096], -1);  permute = None
        getitem: "bf16[1, 128, 2048]" = split_with_sizes[0]
        getitem_1: "bf16[1, 128, 2048]" = split_with_sizes[1]
        getitem_2: "bf16[1, 128, 4096]" = split_with_sizes[2];  split_with_sizes = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:516 in torch_dynamo_resume_in_forward_at_493, code: query = query.reshape(batch_size, seq_len, -1, self.head_k_dim)
        view: "bf16[1, 128, 16, 128]" = torch.ops.aten.view.default(getitem, [1, 128, 16, 128]);  getitem = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:517 in torch_dynamo_resume_in_forward_at_493, code: key = key.reshape(batch_size, seq_len, -1, self.head_k_dim)
        view_1: "bf16[1, 128, 16, 128]" = torch.ops.aten.view.default(getitem_1, [1, 128, 16, 128]);  getitem_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:518 in torch_dynamo_resume_in_forward_at_493, code: value = value.reshape(batch_size, seq_len, -1, self.head_v_dim)
        view_2: "bf16[1, 128, 32, 128]" = torch.ops.aten.view.default(getitem_2, [1, 128, 32, 128]);  getitem_2 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:520 in torch_dynamo_resume_in_forward_at_493, code: beta = b.sigmoid()
        sigmoid: "bf16[1, 128, 32]" = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:522 in torch_dynamo_resume_in_forward_at_493, code: g = -self.A_log.float().exp() * F.softplus(a.float() + self.dt_bias)
        _to_copy: "f32[32]" = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
        exp: "f32[32]" = torch.ops.aten.exp.default(_to_copy);  _to_copy = None
        neg: "f32[32]" = torch.ops.aten.neg.default(exp);  exp = None
        _to_copy_1: "f32[1, 128, 32]" = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
        add: "f32[1, 128, 32]" = torch.ops.aten.add.Tensor(_to_copy_1, arg4_1);  _to_copy_1 = arg4_1 = None
        exp_1: "f32[1, 128, 32]" = torch.ops.aten.exp.default(add)
        log1p: "f32[1, 128, 32]" = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
        gt: "b8[1, 128, 32]" = torch.ops.aten.gt.Scalar(add, 20)
        where: "f32[1, 128, 32]" = torch.ops.aten.where.self(gt, add, log1p);  gt = add = log1p = None
        mul: "f32[1, 128, 32]" = torch.ops.aten.mul.Tensor(neg, where);  neg = where = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:524 in torch_dynamo_resume_in_forward_at_493, code: query = query.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        unsqueeze: "bf16[1, 128, 16, 1, 128]" = torch.ops.aten.unsqueeze.default(view, 3);  view = None
        expand: "bf16[1, 128, 16, 2, 128]" = torch.ops.aten.expand.default(unsqueeze, [1, 128, 16, 2, 128]);  unsqueeze = None
        clone: "bf16[1, 128, 16, 2, 128]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_3: "bf16[1, 128, 32, 128]" = torch.ops.aten.view.default(clone, [1, 128, 32, 128]);  clone = None
        
        # File: /root/real_root/qwen35_fx_capture/.venv-fla/lib/python3.12/site-packages/transformers/models/qwen3_5/modeling_qwen3_5.py:525 in torch_dynamo_resume_in_forward_at_493, code: key = key.repeat_interleave(self.num_v_heads // self.num_k_heads, dim=2)
        unsqueeze_1: "bf16[1, 128, 16, 1, 128]" = torch.ops.aten.unsqueeze.default(view_1, 3);  view_1 = None
        expand_1: "bf16[1, 128, 16, 2, 128]" = torch.ops.aten.expand.default(unsqueeze_1, [1, 128, 16, 2, 128]);  unsqueeze_1 = None
        clone_1: "bf16[1, 128, 16, 2, 128]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_4: "bf16[1, 128, 32, 128]" = torch.ops.aten.view.default(clone_1, [1, 128, 32, 128]);  clone_1 = None
        return (view_3, view_4, view_2, mul, sigmoid)
        