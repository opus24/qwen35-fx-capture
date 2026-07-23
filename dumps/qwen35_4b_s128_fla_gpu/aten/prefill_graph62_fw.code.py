


def forward(self, arg0_1, arg1_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32);  arg0_1 = None
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
    add_1 = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_1 = None
    return (_to_copy_2,)
    