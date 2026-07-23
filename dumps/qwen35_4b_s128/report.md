# FX capture report — Qwen/Qwen3.5-4B

weights=random dtype=bfloat16 attn=eager layers=32 backend=dump decomp=core dynamic=false

## prefill (batch=1 seq_len=128 past=0)

| level | graph | nodes | placeholders | call_function |
| --- | --- | --- | --- | --- |
| dynamo | prefill_graph0 | 19897 | 429 | 11089 |
| aten | prefill_graph0_fw | 42023 | 429 | 41576 |

### top ops — prefill_graph0_fw

| op | count |
| --- | --- |
| torch._ops.aten.slice.Tensor | 6166 |
| torch._ops.aten.select.int | 5119 |
| torch._ops.aten.unsqueeze.default | 3693 |
| torch._ops.aten.view.default | 3478 |
| torch._ops.aten.clone.default | 3256 |
| torch._ops.aten.mul.Tensor | 2340 |
| torch._ops.aten.expand.default | 2287 |
| torch._ops.aten.add.Tensor | 1980 |
| torch._ops.aten.arange.start_step | 1756 |
| torch._ops.aten.where.self | 1659 |
| torch._ops.aten.copy.default | 1610 |
| torch._ops.aten.eq.Scalar | 1562 |
| torch._ops.aten.sum.dim_IntList | 1560 |
| torch._ops.aten.slice_scatter.default | 1514 |
| torch._ops.aten._to_copy.default | 663 |
| torch._ops.aten.permute.default | 609 |
| torch._ops.aten.bmm.default | 329 |
| torch._ops.aten.mm.default | 249 |
| torch._ops.aten.exp.default | 240 |
| torch._ops.aten.sub.Tensor | 192 |
| torch._ops.aten.rsqrt.default | 153 |
| torch._ops.aten.constant_pad_nd.default | 144 |
| torch._ops.aten.sigmoid.default | 112 |
| torch._ops.aten.pow.Tensor_Scalar | 105 |
| torch._ops.aten.mean.dim | 105 |

## decode (batch=1 seq_len=1 past=128)

| level | graph | nodes | placeholders | call_function |
| --- | --- | --- | --- | --- |
| dynamo | decode_graph0 | 4561 | 493 | 2097 |
| aten | decode_graph0_fw | 5783 | 493 | 5288 |

### top ops — decode_graph0_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 862 |
| torch._ops.aten.mul.Tensor | 660 |
| torch._ops.aten._to_copy.default | 639 |
| torch._ops.aten.permute.default | 489 |
| torch._ops.aten.add.Tensor | 372 |
| torch._ops.aten.unsqueeze.default | 309 |
| torch._ops.aten.mm.default | 249 |
| torch._ops.aten.rsqrt.default | 153 |
| torch._ops.aten.select.int | 151 |
| torch._ops.aten.expand.default | 127 |
| torch._ops.aten.slice.Tensor | 118 |
| torch._ops.aten.sigmoid.default | 112 |
| torch._ops.aten.pow.Tensor_Scalar | 105 |
| torch._ops.aten.mean.dim | 105 |
| torch._ops.aten.sum.dim_IntList | 96 |
| _operator.getitem | 88 |
| torch._ops.aten.clone.default | 80 |
| torch._ops.aten.copy.default | 74 |
| torch._ops.aten.cat.default | 73 |
| torch._ops.aten.exp.default | 72 |
| torch._ops.aten.where.self | 51 |
| torch._ops.aten.neg.default | 40 |
| torch._ops.aten.split_with_sizes.default | 32 |
| torch._ops.aten.arange.start_step | 28 |
| torch._ops.aten.eq.Scalar | 26 |
