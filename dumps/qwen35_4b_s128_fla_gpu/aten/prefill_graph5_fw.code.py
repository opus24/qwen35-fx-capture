


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    permute = torch.ops.aten.permute.default(arg0_1, [0, 2, 1]);  arg0_1 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(permute, [2048, 2048, 4096], -1);  permute = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1]
    getitem_2 = split_with_sizes[2];  split_with_sizes = None
    view = torch.ops.aten.view.default(getitem, [1, 128, 16, 128]);  getitem = None
    view_1 = torch.ops.aten.view.default(getitem_1, [1, 128, 16, 128]);  getitem_1 = None
    view_2 = torch.ops.aten.view.default(getitem_2, [1, 128, 32, 128]);  getitem_2 = None
    sigmoid = torch.ops.aten.sigmoid.default(arg1_1);  arg1_1 = None
    _to_copy = torch.ops.aten._to_copy.default(arg2_1, dtype = torch.float32);  arg2_1 = None
    exp = torch.ops.aten.exp.default(_to_copy);  _to_copy = None
    neg = torch.ops.aten.neg.default(exp);  exp = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
    add = torch.ops.aten.add.Tensor(_to_copy_1, arg4_1);  _to_copy_1 = arg4_1 = None
    exp_1 = torch.ops.aten.exp.default(add)
    log1p = torch.ops.aten.log1p.default(exp_1);  exp_1 = None
    gt = torch.ops.aten.gt.Scalar(add, 20)
    where = torch.ops.aten.where.self(gt, add, log1p);  gt = add = log1p = None
    mul = torch.ops.aten.mul.Tensor(neg, where);  neg = where = None
    unsqueeze = torch.ops.aten.unsqueeze.default(view, 3);  view = None
    expand = torch.ops.aten.expand.default(unsqueeze, [1, 128, 16, 2, 128]);  unsqueeze = None
    clone = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_3 = torch.ops.aten.view.default(clone, [1, 128, 32, 128]);  clone = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(view_1, 3);  view_1 = None
    expand_1 = torch.ops.aten.expand.default(unsqueeze_1, [1, 128, 16, 2, 128]);  unsqueeze_1 = None
    clone_1 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_4 = torch.ops.aten.view.default(clone_1, [1, 128, 32, 128]);  clone_1 = None
    return (view_3, view_4, view_2, mul, sigmoid)
    