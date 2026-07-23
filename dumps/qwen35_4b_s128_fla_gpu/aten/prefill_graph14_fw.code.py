


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1):
    full_like = torch.ops.aten.full_like.default(arg0_1, 0, pin_memory = False, memory_format = torch.preserve_format)
    copy = torch.ops.aten.copy.default(full_like, arg0_1);  full_like = arg0_1 = None
    view = torch.ops.aten.view.default(arg1_1, [-1, 128]);  arg1_1 = None
    view_1 = torch.ops.aten.view.default(arg2_1, [-1, 128]);  arg2_1 = None
    view_2 = torch.ops.aten.view.default(view, [-1, 128]);  view = None
    view_3 = torch.ops.aten.view.default(view_1, [-1, 128]);  view_1 = None
    empty_permuted = torch.ops.aten.empty_permuted.default([4096, 128], [0, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    empty = torch.ops.aten.empty.memory_format([4096], dtype = torch.float32, device = device(type='cuda', index=0), pin_memory = False)
    triton_kernel_wrapper_functional_proxy = torch.ops.higher_order.triton_kernel_wrapper_functional(kernel_idx = 0, constant_args_idx = 27, grid = [(256, 1, 1), (256, 1, 1), (256, 1, 1), (128, 1, 1), (128, 1, 1), (128, 1, 1), (64, 1, 1), (64, 1, 1), (64, 1, 1)], tma_descriptor_metadata = {}, kwargs = {'x': view_2, 'g': view_3, 'y': empty_permuted, 'w': arg3_1, 'rstd': empty}, tensors_to_clone = ['y', 'rstd']);  view_2 = view_3 = empty_permuted = arg3_1 = empty = None
    getitem = triton_kernel_wrapper_functional_proxy['y'];  triton_kernel_wrapper_functional_proxy = None
    permute = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    view_10 = torch.ops.aten.view.default(getitem, [4096, 128]);  getitem = None
    view_11 = torch.ops.aten.view.default(view_10, [1, 128, -1]);  view_10 = None
    view_12 = torch.ops.aten.view.default(view_11, [128, 4096]);  view_11 = None
    mm = torch.ops.aten.mm.default(view_12, permute);  view_12 = permute = None
    view_13 = torch.ops.aten.view.default(mm, [1, 128, 2560]);  mm = None
    return (view_13, copy)
    