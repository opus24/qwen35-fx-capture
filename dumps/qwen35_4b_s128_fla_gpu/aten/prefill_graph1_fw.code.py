


def forward(self, arg0_1, arg1_1):
    unsqueeze = torch.ops.aten.unsqueeze.default(arg1_1, 0);  arg1_1 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(unsqueeze, 1);  unsqueeze = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(unsqueeze_1, 3);  unsqueeze_1 = None
    _to_copy = torch.ops.aten._to_copy.default(unsqueeze_2, dtype = torch.float32);  unsqueeze_2 = None
    expand = torch.ops.aten.expand.default(_to_copy, [3, 1, -1, 1]);  _to_copy = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(arg0_1, 2);  arg0_1 = None
    _to_copy_1 = torch.ops.aten._to_copy.default(unsqueeze_3, dtype = torch.float32);  unsqueeze_3 = None
    expand_1 = torch.ops.aten.expand.default(expand, [3, 1, 32, 1]);  expand = None
    view = torch.ops.aten.view.default(expand_1, [3, 32, 1]);  expand_1 = None
    expand_2 = torch.ops.aten.expand.default(_to_copy_1, [3, 1, 1, 128]);  _to_copy_1 = None
    view_1 = torch.ops.aten.view.default(expand_2, [3, 1, 128]);  expand_2 = None
    bmm = torch.ops.aten.bmm.default(view, view_1);  view = view_1 = None
    view_2 = torch.ops.aten.view.default(bmm, [3, 1, 32, 128])
    permute = torch.ops.aten.permute.default(view_2, [0, 1, 3, 2]);  view_2 = None
    select = torch.ops.aten.select.int(permute, 0, 0)
    select_1 = torch.ops.aten.select.int(permute, 0, 1);  permute = None
    slice_1 = torch.ops.aten.slice.Tensor(select_1, 2, 1, 33, 3);  select_1 = None
    slice_2 = torch.ops.aten.slice.Tensor(select, 2, 1, 33, 3);  select = None
    copy = torch.ops.aten.copy.default(slice_2, slice_1);  slice_2 = slice_1 = None
    view_3 = torch.ops.aten.view.default(bmm, [3, 1, 32, 128]);  bmm = None
    permute_1 = torch.ops.aten.permute.default(view_3, [0, 1, 3, 2]);  view_3 = None
    select_2 = torch.ops.aten.select.int(permute_1, 0, 0)
    slice_scatter = torch.ops.aten.slice_scatter.default(select_2, copy, 2, 1, 33, 3);  select_2 = copy = None
    arange = torch.ops.aten.arange.start_step(0, 3, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    view_4 = torch.ops.aten.view.default(arange, [-1, 1, 1, 1]);  arange = None
    eq = torch.ops.aten.eq.Scalar(view_4, 0);  view_4 = None
    unsqueeze_4 = torch.ops.aten.unsqueeze.default(slice_scatter, 0);  slice_scatter = None
    expand_3 = torch.ops.aten.expand.default(unsqueeze_4, [3, 1, 128, 32]);  unsqueeze_4 = None
    where = torch.ops.aten.where.self(eq, expand_3, permute_1);  eq = expand_3 = permute_1 = None
    permute_2 = torch.ops.aten.permute.default(where, [0, 1, 3, 2]);  where = None
    view_5 = torch.ops.aten.view.default(permute_2, [3, 32, 128]);  permute_2 = None
    view_10 = torch.ops.aten.view.default(view_5, [3, 1, 32, 128])
    permute_7 = torch.ops.aten.permute.default(view_10, [0, 1, 3, 2]);  view_10 = None
    select_7 = torch.ops.aten.select.int(permute_7, 0, 0);  permute_7 = None
    slice_6 = torch.ops.aten.slice.Tensor(select_7, 2, 2, 30, 3);  select_7 = None
    view_11 = torch.ops.aten.view.default(view_5, [3, 1, 32, 128])
    permute_8 = torch.ops.aten.permute.default(view_11, [0, 1, 3, 2]);  view_11 = None
    select_8 = torch.ops.aten.select.int(permute_8, 0, 2);  permute_8 = None
    slice_7 = torch.ops.aten.slice.Tensor(select_8, 2, 2, 30, 3);  select_8 = None
    copy_1 = torch.ops.aten.copy.default(slice_6, slice_7);  slice_6 = slice_7 = None
    view_12 = torch.ops.aten.view.default(view_5, [3, 1, 32, 128]);  view_5 = None
    permute_9 = torch.ops.aten.permute.default(view_12, [0, 1, 3, 2]);  view_12 = None
    select_9 = torch.ops.aten.select.int(permute_9, 0, 0)
    slice_scatter_1 = torch.ops.aten.slice_scatter.default(select_9, copy_1, 2, 2, 30, 3);  select_9 = copy_1 = None
    arange_1 = torch.ops.aten.arange.start_step(0, 3, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    view_13 = torch.ops.aten.view.default(arange_1, [-1, 1, 1, 1]);  arange_1 = None
    eq_1 = torch.ops.aten.eq.Scalar(view_13, 0);  view_13 = None
    unsqueeze_5 = torch.ops.aten.unsqueeze.default(slice_scatter_1, 0);  slice_scatter_1 = None
    expand_4 = torch.ops.aten.expand.default(unsqueeze_5, [3, 1, 128, 32]);  unsqueeze_5 = None
    where_1 = torch.ops.aten.where.self(eq_1, expand_4, permute_9);  eq_1 = expand_4 = permute_9 = None
    permute_10 = torch.ops.aten.permute.default(where_1, [0, 1, 3, 2]);  where_1 = None
    view_14 = torch.ops.aten.view.default(permute_10, [3, 32, 128]);  permute_10 = None
    view_16 = torch.ops.aten.view.default(view_14, [3, 1, 32, 128]);  view_14 = None
    permute_12 = torch.ops.aten.permute.default(view_16, [0, 1, 3, 2]);  view_16 = None
    select_11 = torch.ops.aten.select.int(permute_12, 0, 0);  permute_12 = None
    cat = torch.ops.aten.cat.default([select_11, select_11], -1);  select_11 = None
    cos = torch.ops.aten.cos.default(cat)
    mul = torch.ops.aten.mul.Tensor(cos, 1.0);  cos = None
    sin = torch.ops.aten.sin.default(cat);  cat = None
    mul_1 = torch.ops.aten.mul.Tensor(sin, 1.0);  sin = None
    _to_copy_2 = torch.ops.aten._to_copy.default(mul, dtype = torch.bfloat16);  mul = None
    _to_copy_3 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16);  mul_1 = None
    return (_to_copy_2, _to_copy_3)
    