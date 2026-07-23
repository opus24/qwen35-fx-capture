


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1):
    permute = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
    view = torch.ops.aten.view.default(arg0_1, [128, 2560])
    mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
    view_1 = torch.ops.aten.view.default(mm, [1, 128, 8192]);  mm = None
    permute_1 = torch.ops.aten.permute.default(view_1, [0, 2, 1]);  view_1 = None
    permute_2 = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    view_2 = torch.ops.aten.view.default(arg0_1, [128, 2560])
    mm_1 = torch.ops.aten.mm.default(view_2, permute_2);  view_2 = permute_2 = None
    view_3 = torch.ops.aten.view.default(mm_1, [1, 128, 4096]);  mm_1 = None
    view_4 = torch.ops.aten.view.default(view_3, [1, 128, -1, 128]);  view_3 = None
    permute_3 = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
    view_5 = torch.ops.aten.view.default(arg0_1, [128, 2560])
    mm_2 = torch.ops.aten.mm.default(view_5, permute_3);  view_5 = permute_3 = None
    view_6 = torch.ops.aten.view.default(mm_2, [1, 128, 32]);  mm_2 = None
    permute_4 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    view_7 = torch.ops.aten.view.default(arg0_1, [128, 2560]);  arg0_1 = None
    mm_3 = torch.ops.aten.mm.default(view_7, permute_4);  view_7 = permute_4 = None
    view_8 = torch.ops.aten.view.default(mm_3, [1, 128, 32]);  mm_3 = None
    constant_pad_nd = torch.ops.aten.constant_pad_nd.default(permute_1, [-124, 0], 0.0)
    full = torch.ops.aten.full.default([1, 8192, 4], 0, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    slice_1 = torch.ops.aten.slice.Tensor(constant_pad_nd, 2, -4, 9223372036854775807);  constant_pad_nd = None
    copy = torch.ops.aten.copy.default(full, slice_1);  full = slice_1 = None
    squeeze = torch.ops.aten.squeeze.dims(arg5_1, [1]);  arg5_1 = None
    return (permute_1, squeeze, view_4, view_6, view_8, copy)
    