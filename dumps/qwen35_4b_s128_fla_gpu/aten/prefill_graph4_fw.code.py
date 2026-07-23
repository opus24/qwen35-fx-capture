


def forward(self, arg0_1):
    empty_permuted = torch.ops.aten.empty_permuted.default([1, 8192, 128], [0, 2, 1], dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
    return (empty_permuted,)
    