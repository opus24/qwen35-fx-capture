


def forward(self, arg0_1, arg1_1):
    slice_1 = torch.ops.aten.slice.Tensor(arg0_1, 1, -1, 9223372036854775807);  arg0_1 = None
    permute = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
    view = torch.ops.aten.view.default(slice_1, [1, 2560]);  slice_1 = None
    mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
    view_1 = torch.ops.aten.view.default(mm, [1, 1, 248320]);  mm = None
    return (view_1,)
    