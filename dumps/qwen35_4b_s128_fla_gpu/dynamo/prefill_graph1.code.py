


def forward(self, L_position_ids_ : torch.Tensor, L_self_buffers_inv_freq_ : torch.Tensor):
    l_position_ids_ = L_position_ids_
    l_self_buffers_inv_freq_ = L_self_buffers_inv_freq_
    getitem = l_self_buffers_inv_freq_[(None, None, slice(None, None, None), None)];  l_self_buffers_inv_freq_ = None
    float_1 = getitem.float();  getitem = None
    expand = float_1.expand(3, 1, -1, 1);  float_1 = None
    inv_freq_expanded = expand.to(device(type='cuda', index=0));  expand = None
    getitem_1 = l_position_ids_[(slice(None, None, None), slice(None, None, None), None, slice(None, None, None))];  l_position_ids_ = None
    position_ids_expanded = getitem_1.float();  getitem_1 = None
    float_3 = inv_freq_expanded.float();  inv_freq_expanded = None
    float_4 = position_ids_expanded.float();  position_ids_expanded = None
    matmul = float_3 @ float_4;  float_3 = float_4 = None
    freqs = matmul.transpose(2, 3);  matmul = None
    freqs_t = freqs[0]
    getitem_3 = freqs[(1, Ellipsis, slice(1, 33, 3))]
    freqs_t[(Ellipsis, slice(1, 33, 3))] = getitem_3;  setitem = freqs_t;  getitem_3 = setitem = None
    getitem_4 = freqs[(2, Ellipsis, slice(2, 30, 3))];  freqs = None
    freqs_t[(Ellipsis, slice(2, 30, 3))] = getitem_4;  setitem_1 = freqs_t;  getitem_4 = setitem_1 = None
    emb = torch.cat((freqs_t, freqs_t), dim = -1);  freqs_t = None
    cos = emb.cos()
    cos_1 = cos * 1.0;  cos = None
    sin = emb.sin();  emb = None
    sin_1 = sin * 1.0;  sin = None
    to_1 = cos_1.to(dtype = torch.bfloat16);  cos_1 = None
    to_2 = sin_1.to(dtype = torch.bfloat16);  sin_1 = None
    return (to_1, to_2)
    