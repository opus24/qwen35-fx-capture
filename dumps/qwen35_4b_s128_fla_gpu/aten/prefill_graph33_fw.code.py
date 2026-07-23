


def forward(self, arg0_1, arg1_1, arg2_1, arg3_1, arg4_1, arg5_1, arg6_1, arg7_1, arg8_1, arg9_1, arg10_1, arg11_1, arg12_1, arg13_1, arg14_1):
    _to_copy = torch.ops.aten._to_copy.default(arg0_1, dtype = torch.float32)
    pow_1 = torch.ops.aten.pow.Tensor_Scalar(_to_copy, 2)
    mean = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None
    add = torch.ops.aten.add.Tensor(mean, 1e-06);  mean = None
    rsqrt = torch.ops.aten.rsqrt.default(add);  add = None
    mul = torch.ops.aten.mul.Tensor(_to_copy, rsqrt);  _to_copy = rsqrt = None
    _to_copy_1 = torch.ops.aten._to_copy.default(arg1_1, dtype = torch.float32);  arg1_1 = None
    add_1 = torch.ops.aten.add.Tensor(_to_copy_1, 1.0);  _to_copy_1 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, add_1);  mul = add_1 = None
    _to_copy_2 = torch.ops.aten._to_copy.default(mul_1, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_1 = None
    permute = torch.ops.aten.permute.default(arg2_1, [1, 0]);  arg2_1 = None
    view = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
    mm = torch.ops.aten.mm.default(view, permute);  view = permute = None
    view_1 = torch.ops.aten.view.default(mm, [1, 128, 8192]);  mm = None
    view_2 = torch.ops.aten.view.default(view_1, [1, 128, -1, 512]);  view_1 = None
    split_with_sizes = torch.ops.aten.split_with_sizes.default(view_2, [256, 256], -1);  view_2 = None
    getitem = split_with_sizes[0]
    getitem_1 = split_with_sizes[1];  split_with_sizes = None
    clone = torch.ops.aten.clone.default(getitem_1, memory_format = torch.contiguous_format);  getitem_1 = None
    view_3 = torch.ops.aten.view.default(clone, [1, 128, 4096]);  clone = None
    view_4 = torch.ops.aten.view.default(getitem, [1, 128, -1, 256]);  getitem = None
    _to_copy_3 = torch.ops.aten._to_copy.default(view_4, dtype = torch.float32);  view_4 = None
    pow_2 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_3, 2)
    mean_1 = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None
    add_2 = torch.ops.aten.add.Tensor(mean_1, 1e-06);  mean_1 = None
    rsqrt_1 = torch.ops.aten.rsqrt.default(add_2);  add_2 = None
    mul_2 = torch.ops.aten.mul.Tensor(_to_copy_3, rsqrt_1);  _to_copy_3 = rsqrt_1 = None
    _to_copy_4 = torch.ops.aten._to_copy.default(arg3_1, dtype = torch.float32);  arg3_1 = None
    add_3 = torch.ops.aten.add.Tensor(_to_copy_4, 1.0);  _to_copy_4 = None
    mul_3 = torch.ops.aten.mul.Tensor(mul_2, add_3);  mul_2 = add_3 = None
    _to_copy_5 = torch.ops.aten._to_copy.default(mul_3, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_3 = None
    permute_1 = torch.ops.aten.permute.default(_to_copy_5, [0, 2, 1, 3]);  _to_copy_5 = None
    permute_2 = torch.ops.aten.permute.default(arg4_1, [1, 0]);  arg4_1 = None
    view_5 = torch.ops.aten.view.default(_to_copy_2, [128, 2560])
    mm_1 = torch.ops.aten.mm.default(view_5, permute_2);  view_5 = permute_2 = None
    view_6 = torch.ops.aten.view.default(mm_1, [1, 128, 1024]);  mm_1 = None
    view_7 = torch.ops.aten.view.default(view_6, [1, 128, -1, 256]);  view_6 = None
    _to_copy_6 = torch.ops.aten._to_copy.default(view_7, dtype = torch.float32);  view_7 = None
    pow_3 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_6, 2)
    mean_2 = torch.ops.aten.mean.dim(pow_3, [-1], True);  pow_3 = None
    add_4 = torch.ops.aten.add.Tensor(mean_2, 1e-06);  mean_2 = None
    rsqrt_2 = torch.ops.aten.rsqrt.default(add_4);  add_4 = None
    mul_4 = torch.ops.aten.mul.Tensor(_to_copy_6, rsqrt_2);  _to_copy_6 = rsqrt_2 = None
    _to_copy_7 = torch.ops.aten._to_copy.default(arg5_1, dtype = torch.float32);  arg5_1 = None
    add_5 = torch.ops.aten.add.Tensor(_to_copy_7, 1.0);  _to_copy_7 = None
    mul_5 = torch.ops.aten.mul.Tensor(mul_4, add_5);  mul_4 = add_5 = None
    _to_copy_8 = torch.ops.aten._to_copy.default(mul_5, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_5 = None
    permute_3 = torch.ops.aten.permute.default(_to_copy_8, [0, 2, 1, 3]);  _to_copy_8 = None
    permute_4 = torch.ops.aten.permute.default(arg6_1, [1, 0]);  arg6_1 = None
    view_8 = torch.ops.aten.view.default(_to_copy_2, [128, 2560]);  _to_copy_2 = None
    mm_2 = torch.ops.aten.mm.default(view_8, permute_4);  view_8 = permute_4 = None
    view_9 = torch.ops.aten.view.default(mm_2, [1, 128, 1024]);  mm_2 = None
    view_10 = torch.ops.aten.view.default(view_9, [1, 128, -1, 256]);  view_9 = None
    permute_5 = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
    unsqueeze = torch.ops.aten.unsqueeze.default(arg7_1, 1);  arg7_1 = None
    unsqueeze_1 = torch.ops.aten.unsqueeze.default(arg8_1, 1);  arg8_1 = None
    slice_1 = torch.ops.aten.slice.Tensor(permute_1, 3, 0, 64)
    slice_2 = torch.ops.aten.slice.Tensor(permute_1, 3, 64, 9223372036854775807);  permute_1 = None
    slice_3 = torch.ops.aten.slice.Tensor(permute_3, 3, 0, 64)
    slice_4 = torch.ops.aten.slice.Tensor(permute_3, 3, 64, 9223372036854775807);  permute_3 = None
    mul_6 = torch.ops.aten.mul.Tensor(slice_1, unsqueeze)
    slice_5 = torch.ops.aten.slice.Tensor(slice_1, 3, 0, 32)
    slice_6 = torch.ops.aten.slice.Tensor(slice_1, 3, 32, 9223372036854775807);  slice_1 = None
    neg = torch.ops.aten.neg.default(slice_6);  slice_6 = None
    cat = torch.ops.aten.cat.default([neg, slice_5], -1);  neg = slice_5 = None
    mul_7 = torch.ops.aten.mul.Tensor(cat, unsqueeze_1);  cat = None
    add_6 = torch.ops.aten.add.Tensor(mul_6, mul_7);  mul_6 = mul_7 = None
    mul_8 = torch.ops.aten.mul.Tensor(slice_3, unsqueeze);  unsqueeze = None
    slice_7 = torch.ops.aten.slice.Tensor(slice_3, 3, 0, 32)
    slice_8 = torch.ops.aten.slice.Tensor(slice_3, 3, 32, 9223372036854775807);  slice_3 = None
    neg_1 = torch.ops.aten.neg.default(slice_8);  slice_8 = None
    cat_1 = torch.ops.aten.cat.default([neg_1, slice_7], -1);  neg_1 = slice_7 = None
    mul_9 = torch.ops.aten.mul.Tensor(cat_1, unsqueeze_1);  cat_1 = unsqueeze_1 = None
    add_7 = torch.ops.aten.add.Tensor(mul_8, mul_9);  mul_8 = mul_9 = None
    cat_2 = torch.ops.aten.cat.default([add_6, slice_2], -1);  add_6 = slice_2 = None
    cat_3 = torch.ops.aten.cat.default([add_7, slice_4], -1);  add_7 = slice_4 = None
    _tensor_constant0 = self._tensor_constant0
    lift_fresh_copy = torch.ops.aten.lift_fresh_copy.default(_tensor_constant0);  _tensor_constant0 = None
    _tensor_constant1 = self._tensor_constant1
    lift_fresh_copy_1 = torch.ops.aten.lift_fresh_copy.default(_tensor_constant1);  _tensor_constant1 = None
    cat_4 = torch.ops.aten.cat.default([lift_fresh_copy, cat_3], -2);  lift_fresh_copy = cat_3 = None
    cat_5 = torch.ops.aten.cat.default([lift_fresh_copy_1, permute_5], -2);  lift_fresh_copy_1 = permute_5 = None
    unsqueeze_2 = torch.ops.aten.unsqueeze.default(cat_4, 2)
    expand = torch.ops.aten.expand.default(unsqueeze_2, [1, 4, 4, 128, 256]);  unsqueeze_2 = None
    clone_1 = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
    view_11 = torch.ops.aten.view.default(clone_1, [1, 16, 128, 256]);  clone_1 = None
    unsqueeze_3 = torch.ops.aten.unsqueeze.default(cat_5, 2)
    expand_1 = torch.ops.aten.expand.default(unsqueeze_3, [1, 4, 4, 128, 256]);  unsqueeze_3 = None
    clone_2 = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
    view_12 = torch.ops.aten.view.default(clone_2, [1, 16, 128, 256]);  clone_2 = None
    permute_6 = torch.ops.aten.permute.default(view_11, [0, 1, 3, 2]);  view_11 = None
    expand_2 = torch.ops.aten.expand.default(cat_2, [1, 16, 128, 256]);  cat_2 = None
    view_13 = torch.ops.aten.view.default(expand_2, [16, 128, 256]);  expand_2 = None
    expand_3 = torch.ops.aten.expand.default(permute_6, [1, 16, 256, 128]);  permute_6 = None
    view_14 = torch.ops.aten.view.default(expand_3, [16, 256, 128]);  expand_3 = None
    bmm = torch.ops.aten.bmm.default(view_13, view_14);  view_13 = view_14 = None
    view_15 = torch.ops.aten.view.default(bmm, [1, 16, 128, 128]);  bmm = None
    mul_10 = torch.ops.aten.mul.Tensor(view_15, 0.0625);  view_15 = None
    add_8 = torch.ops.aten.add.Tensor(mul_10, arg9_1);  mul_10 = arg9_1 = None
    _to_copy_9 = torch.ops.aten._to_copy.default(add_8, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0));  add_8 = None
    _softmax = torch.ops.aten._softmax.default(_to_copy_9, -1, False);  _to_copy_9 = None
    _to_copy_10 = torch.ops.aten._to_copy.default(_softmax, dtype = torch.bfloat16);  _softmax = None
    clone_3 = torch.ops.aten.clone.default(_to_copy_10);  _to_copy_10 = None
    expand_4 = torch.ops.aten.expand.default(clone_3, [1, 16, 128, 128]);  clone_3 = None
    view_16 = torch.ops.aten.view.default(expand_4, [16, 128, 128]);  expand_4 = None
    expand_5 = torch.ops.aten.expand.default(view_12, [1, 16, 128, 256]);  view_12 = None
    view_17 = torch.ops.aten.view.default(expand_5, [16, 128, 256]);  expand_5 = None
    bmm_1 = torch.ops.aten.bmm.default(view_16, view_17);  view_16 = view_17 = None
    view_18 = torch.ops.aten.view.default(bmm_1, [1, 16, 128, 256]);  bmm_1 = None
    permute_7 = torch.ops.aten.permute.default(view_18, [0, 2, 1, 3]);  view_18 = None
    clone_4 = torch.ops.aten.clone.default(permute_7, memory_format = torch.contiguous_format);  permute_7 = None
    view_19 = torch.ops.aten.view.default(clone_4, [1, 128, -1]);  clone_4 = None
    sigmoid = torch.ops.aten.sigmoid.default(view_3);  view_3 = None
    mul_11 = torch.ops.aten.mul.Tensor(view_19, sigmoid);  view_19 = sigmoid = None
    permute_8 = torch.ops.aten.permute.default(arg10_1, [1, 0]);  arg10_1 = None
    view_20 = torch.ops.aten.view.default(mul_11, [128, 4096]);  mul_11 = None
    mm_3 = torch.ops.aten.mm.default(view_20, permute_8);  view_20 = permute_8 = None
    view_21 = torch.ops.aten.view.default(mm_3, [1, 128, 2560]);  mm_3 = None
    add_9 = torch.ops.aten.add.Tensor(arg0_1, view_21);  arg0_1 = view_21 = None
    _to_copy_11 = torch.ops.aten._to_copy.default(add_9, dtype = torch.float32)
    pow_4 = torch.ops.aten.pow.Tensor_Scalar(_to_copy_11, 2)
    mean_3 = torch.ops.aten.mean.dim(pow_4, [-1], True);  pow_4 = None
    add_10 = torch.ops.aten.add.Tensor(mean_3, 1e-06);  mean_3 = None
    rsqrt_3 = torch.ops.aten.rsqrt.default(add_10);  add_10 = None
    mul_12 = torch.ops.aten.mul.Tensor(_to_copy_11, rsqrt_3);  _to_copy_11 = rsqrt_3 = None
    _to_copy_12 = torch.ops.aten._to_copy.default(arg11_1, dtype = torch.float32);  arg11_1 = None
    add_11 = torch.ops.aten.add.Tensor(_to_copy_12, 1.0);  _to_copy_12 = None
    mul_13 = torch.ops.aten.mul.Tensor(mul_12, add_11);  mul_12 = add_11 = None
    _to_copy_13 = torch.ops.aten._to_copy.default(mul_13, dtype = torch.bfloat16, layout = torch.strided, device = device(type='cuda', index=0));  mul_13 = None
    permute_9 = torch.ops.aten.permute.default(arg12_1, [1, 0]);  arg12_1 = None
    view_22 = torch.ops.aten.view.default(_to_copy_13, [128, 2560])
    mm_4 = torch.ops.aten.mm.default(view_22, permute_9);  view_22 = permute_9 = None
    view_23 = torch.ops.aten.view.default(mm_4, [1, 128, 9216]);  mm_4 = None
    _to_copy_14 = torch.ops.aten._to_copy.default(view_23, dtype = torch.float32);  view_23 = None
    sigmoid_1 = torch.ops.aten.sigmoid.default(_to_copy_14)
    mul_14 = torch.ops.aten.mul.Tensor(_to_copy_14, sigmoid_1);  _to_copy_14 = sigmoid_1 = None
    _to_copy_15 = torch.ops.aten._to_copy.default(mul_14, dtype = torch.bfloat16);  mul_14 = None
    permute_10 = torch.ops.aten.permute.default(arg13_1, [1, 0]);  arg13_1 = None
    view_24 = torch.ops.aten.view.default(_to_copy_13, [128, 2560]);  _to_copy_13 = None
    mm_5 = torch.ops.aten.mm.default(view_24, permute_10);  view_24 = permute_10 = None
    view_25 = torch.ops.aten.view.default(mm_5, [1, 128, 9216]);  mm_5 = None
    mul_15 = torch.ops.aten.mul.Tensor(_to_copy_15, view_25);  _to_copy_15 = view_25 = None
    permute_11 = torch.ops.aten.permute.default(arg14_1, [1, 0]);  arg14_1 = None
    view_26 = torch.ops.aten.view.default(mul_15, [128, 9216]);  mul_15 = None
    mm_6 = torch.ops.aten.mm.default(view_26, permute_11);  view_26 = permute_11 = None
    view_27 = torch.ops.aten.view.default(mm_6, [1, 128, 2560]);  mm_6 = None
    add_12 = torch.ops.aten.add.Tensor(add_9, view_27);  add_9 = view_27 = None
    return (add_12, cat_5, cat_4)
    