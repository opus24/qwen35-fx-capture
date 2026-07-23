


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    add = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
    _to_copy = torch.ops.aten._to_copy.default(add, dtype = torch.float32)
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add_1 = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add_1);  add_1 = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
    add_2 = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, add_2);  mul = add_2 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_1 = None
    permute = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    view = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
    mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
    view_1 = torch.ops.aten.view.default(mm, [1, 128, 9216]);  mm = None
    _to_copy_3 = torch.ops.aten._to_copy.default(view_1, dtype = torch.float32);  view_1 = None
    sigmoid = torch.ops.aten.sigmoid.default(_to_copy_3)
    mul_2 = torch.ops.aten.mul.Tensor(_to_copy_3, sigmoid);  _to_copy_3 = sigmoid = None
    _to_copy_4 = torch.ops.aten._to_copy.default(mul_2, dtype = torch.bfloat16);  mul_2 = None
    permute_1 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    view_2 = torch.ops.aten.view.default(_to_copy_2, [128, 2560]);  _to_copy_2 = None
    mm_1 = torch.ops.aten.mm.default(view_2, permute_1);  view_2 = permute_1 = None
    view_3 = torch.ops.aten.view.default(mm_1, [1, 128, 9216]);  mm_1 = None
    mul_3 = torch.ops.aten.mul.Tensor(_to_copy_4, view_3);  _to_copy_4 = view_3 = None
    permute_2 = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
    view_4 = torch.ops.aten.view.default(mul_3, [128, 9216]);  mul_3 = None
    mm_2 = torch.ops.aten.mm.default(view_4, permute_2);  view_4 = permute_2 = None
    view_5 = torch.ops.aten.view.default(mm_2, [1, 128, 2560]);  mm_2 = None
    add_3 = torch.ops.aten.add.Tensor(add, view_5);  add = view_5 = None
    return (add_3,)
    