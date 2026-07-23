


def forward(self):
    arange_2 = torch.ops.aten.arange.start_step(0, 128, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    add = torch.ops.aten.add.Tensor(arange_2, 0);  arange_2 = None
    arange_3 = torch.ops.aten.arange.start_step(0, 128, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    add_1 = torch.ops.aten.add.Tensor(arange_3, 0);  arange_3 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(add, 0);  add = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(add_1, 0);  add_1 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(unsqueeze_3, 1);  unsqueeze_3 = None
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(unsqueeze_4, 2);  unsqueeze_4 = None
    le = torch.ops.aten.le.Tensor(unsqueeze_5, unsqueeze_2);  unsqueeze_5 = unsqueeze_2 = None
    expand = torch.ops.aten.expand.default(le, [1, -1, 128, 128]);  le = None
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    scalar_tensor = torch.ops.aten.scalar_tensor.default(-3.3895313892515355e+38, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0))
    where = torch.ops.aten.where.self(expand, lift_fresh_copy, scalar_tensor);  expand = lift_fresh_copy = scalar_tensor = None
    return (where,)
    