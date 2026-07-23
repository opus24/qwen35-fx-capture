


def forward(self):
    batch_arange = torch.arange(1, device = device(type='cuda', index=0));  batch_arange = None
    head_arange = torch.arange(1, device = device(type='cuda', index=0));  head_arange = None
    arange_2 = torch.arange(128, device = device(type='cuda', index=0))
    q_arange = arange_2 + 0;  arange_2 = None
    arange_3 = torch.arange(128, device = device(type='cuda', index=0))
    kv_arange = arange_3 + 0;  arange_3 = None
    q_indices = q_arange[(None, None, slice(None, None, None), None)];  q_arange = None
    kv_indices = kv_arange[(None, None, None, slice(None, None, None))];  kv_arange = None
    attention_mask = kv_indices <= q_indices;  kv_indices = q_indices = None
    attention_mask_1 = attention_mask.expand(1, -1, 128, 128);  attention_mask = None
    tensor = torch.tensor(0.0, device = device(type='cuda', index=0), dtype = torch.bfloat16)
    mask = torch.where(attention_mask_1, tensor, -3.3895313892515355e+38);  attention_mask_1 = tensor = None
    return (mask,)
    