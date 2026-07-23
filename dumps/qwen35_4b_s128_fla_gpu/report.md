# FX capture report — Qwen/Qwen3.5-4B (fla kernels, cuda)

weights=random dtype=bfloat16 attn=eager layers=32 decomp=core dynamic=false

## linear-attention kernel path

| | |
| --- | --- |
| requested | `fla` |
| torch_version | `2.10.0+cu128` |
| cuda_available | `True` |
| gpu | `NVIDIA A40` |
| transformers_version | `5.14.1` |
| fla_available | `True` |
| causal_conv1d_available | `True` |
| fla_version | `0.5.1` |
| delta_rule_chunk | `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` |
| delta_rule_recurrent | `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule` |
| conv1d_fn | `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` |
| conv1d_update | `causal_conv1d.causal_conv1d_interface.causal_conv1d_update` |
| gated_rmsnorm | `fla.modules.fused_norm_gate.FusedRMSNormGated` |
| fla_delta_rule_active | `True` |

## eager latency (CUDA events)

| step | median ms | mean ms | min ms | notes |
| --- | --- | --- | --- | --- |
| prefill (b1 s128) | 98.658 | 98.591 | 97.323 | 1297.4 tok/s |
| decode ×32 (past=128) | 1378.887 | 1380.087 | 1369.948 | **43.09 ms/token** |

peak memory: 8.594 GB

prefill logits: shape=[1, 1, 248320] mean=-0.000777 std=1.010787 absmax=5.0625
params: numel=4205751296 sum_finite=4043.9586 nonfinite=5

## prefill (batch=1 seq_len=128 past=0)

dynamo: graph_count=64 graph_break_count=63

| level | graph | nodes | placeholders | call_function |
| --- | --- | --- | --- | --- |
| dynamo | prefill_graph0 | 13 | 0 | 11 |
| aten | prefill_graph0_fw | 17 | 0 | 15 |
| dynamo | prefill_graph1 | 25 | 2 | 11 |
| aten | prefill_graph1_fw | 65 | 2 | 62 |
| dynamo | prefill_graph2 | 13 | 2 | 5 |
| aten | prefill_graph2_fw | 13 | 2 | 10 |
| dynamo | prefill_graph3 | 18 | 6 | 7 |
| aten | prefill_graph3_fw | 30 | 6 | 23 |
| dynamo | prefill_graph4 | 3 | 1 | 1 |
| aten | prefill_graph4_fw | 3 | 1 | 1 |
| dynamo | prefill_graph5 | 24 | 5 | 8 |
| aten | prefill_graph5_fw | 33 | 5 | 27 |
| dynamo | prefill_graph6 | 23 | 5 | 7 |
| aten | prefill_graph6_fw | 22 | 5 | 16 |
| dynamo | prefill_graph7 | 24 | 6 | 12 |
| aten | prefill_graph7_fw | 36 | 6 | 29 |
| dynamo | prefill_graph8 | 18 | 6 | 7 |
| aten | prefill_graph8_fw | 30 | 6 | 23 |
| dynamo | prefill_graph9 | 23 | 5 | 7 |
| aten | prefill_graph9_fw | 22 | 5 | 16 |
| dynamo | prefill_graph10 | 18 | 6 | 7 |
| aten | prefill_graph10_fw | 30 | 6 | 23 |
| dynamo | prefill_graph11 | 23 | 5 | 7 |
| aten | prefill_graph11_fw | 22 | 5 | 16 |
| dynamo | prefill_graph12 | 124 | 15 | 68 |
| aten | prefill_graph12_fw | 163 | 15 | 145 |
| dynamo | prefill_graph13 | 18 | 6 | 7 |
| aten | prefill_graph13_fw | 30 | 6 | 23 |
| dynamo | prefill_graph14 | 23 | 5 | 7 |
| aten | prefill_graph14_fw | 22 | 5 | 16 |
| dynamo | prefill_graph15 | 18 | 6 | 7 |
| aten | prefill_graph15_fw | 30 | 6 | 23 |
| dynamo | prefill_graph16 | 23 | 5 | 7 |
| aten | prefill_graph16_fw | 22 | 5 | 16 |
| dynamo | prefill_graph17 | 18 | 6 | 7 |
| aten | prefill_graph17_fw | 30 | 6 | 23 |
| dynamo | prefill_graph18 | 23 | 5 | 7 |
| aten | prefill_graph18_fw | 22 | 5 | 16 |
| dynamo | prefill_graph19 | 124 | 15 | 68 |
| aten | prefill_graph19_fw | 163 | 15 | 145 |
| dynamo | prefill_graph20 | 18 | 6 | 7 |
| aten | prefill_graph20_fw | 30 | 6 | 23 |
| dynamo | prefill_graph21 | 23 | 5 | 7 |
| aten | prefill_graph21_fw | 22 | 5 | 16 |
| dynamo | prefill_graph22 | 18 | 6 | 7 |
| aten | prefill_graph22_fw | 30 | 6 | 23 |
| dynamo | prefill_graph23 | 23 | 5 | 7 |
| aten | prefill_graph23_fw | 22 | 5 | 16 |
| dynamo | prefill_graph24 | 18 | 6 | 7 |
| aten | prefill_graph24_fw | 30 | 6 | 23 |
| dynamo | prefill_graph25 | 23 | 5 | 7 |
| aten | prefill_graph25_fw | 22 | 5 | 16 |
| dynamo | prefill_graph26 | 124 | 15 | 68 |
| aten | prefill_graph26_fw | 163 | 15 | 145 |
| dynamo | prefill_graph27 | 18 | 6 | 7 |
| aten | prefill_graph27_fw | 30 | 6 | 23 |
| dynamo | prefill_graph28 | 23 | 5 | 7 |
| aten | prefill_graph28_fw | 22 | 5 | 16 |
| dynamo | prefill_graph29 | 18 | 6 | 7 |
| aten | prefill_graph29_fw | 30 | 6 | 23 |
| dynamo | prefill_graph30 | 23 | 5 | 7 |
| aten | prefill_graph30_fw | 22 | 5 | 16 |
| dynamo | prefill_graph31 | 18 | 6 | 7 |
| aten | prefill_graph31_fw | 30 | 6 | 23 |
| dynamo | prefill_graph32 | 23 | 5 | 7 |
| aten | prefill_graph32_fw | 22 | 5 | 16 |
| dynamo | prefill_graph33 | 124 | 15 | 68 |
| aten | prefill_graph33_fw | 163 | 15 | 145 |
| dynamo | prefill_graph34 | 18 | 6 | 7 |
| aten | prefill_graph34_fw | 30 | 6 | 23 |
| dynamo | prefill_graph35 | 23 | 5 | 7 |
| aten | prefill_graph35_fw | 22 | 5 | 16 |
| dynamo | prefill_graph36 | 18 | 6 | 7 |
| aten | prefill_graph36_fw | 30 | 6 | 23 |
| dynamo | prefill_graph37 | 23 | 5 | 7 |
| aten | prefill_graph37_fw | 22 | 5 | 16 |
| dynamo | prefill_graph38 | 18 | 6 | 7 |
| aten | prefill_graph38_fw | 30 | 6 | 23 |
| dynamo | prefill_graph39 | 23 | 5 | 7 |
| aten | prefill_graph39_fw | 22 | 5 | 16 |
| dynamo | prefill_graph40 | 124 | 15 | 68 |
| aten | prefill_graph40_fw | 163 | 15 | 145 |
| dynamo | prefill_graph41 | 18 | 6 | 7 |
| aten | prefill_graph41_fw | 30 | 6 | 23 |
| dynamo | prefill_graph42 | 23 | 5 | 7 |
| aten | prefill_graph42_fw | 22 | 5 | 16 |
| dynamo | prefill_graph43 | 18 | 6 | 7 |
| aten | prefill_graph43_fw | 30 | 6 | 23 |
| dynamo | prefill_graph44 | 23 | 5 | 7 |
| aten | prefill_graph44_fw | 22 | 5 | 16 |
| dynamo | prefill_graph45 | 18 | 6 | 7 |
| aten | prefill_graph45_fw | 30 | 6 | 23 |
| dynamo | prefill_graph46 | 23 | 5 | 7 |
| aten | prefill_graph46_fw | 22 | 5 | 16 |
| dynamo | prefill_graph47 | 124 | 15 | 68 |
| aten | prefill_graph47_fw | 163 | 15 | 145 |
| dynamo | prefill_graph48 | 18 | 6 | 7 |
| aten | prefill_graph48_fw | 30 | 6 | 23 |
| dynamo | prefill_graph49 | 23 | 5 | 7 |
| aten | prefill_graph49_fw | 22 | 5 | 16 |
| dynamo | prefill_graph50 | 18 | 6 | 7 |
| aten | prefill_graph50_fw | 30 | 6 | 23 |
| dynamo | prefill_graph51 | 23 | 5 | 7 |
| aten | prefill_graph51_fw | 22 | 5 | 16 |
| dynamo | prefill_graph52 | 18 | 6 | 7 |
| aten | prefill_graph52_fw | 30 | 6 | 23 |
| dynamo | prefill_graph53 | 23 | 5 | 7 |
| aten | prefill_graph53_fw | 22 | 5 | 16 |
| dynamo | prefill_graph54 | 124 | 15 | 68 |
| aten | prefill_graph54_fw | 163 | 15 | 145 |
| dynamo | prefill_graph55 | 18 | 6 | 7 |
| aten | prefill_graph55_fw | 30 | 6 | 23 |
| dynamo | prefill_graph56 | 23 | 5 | 7 |
| aten | prefill_graph56_fw | 22 | 5 | 16 |
| dynamo | prefill_graph57 | 18 | 6 | 7 |
| aten | prefill_graph57_fw | 30 | 6 | 23 |
| dynamo | prefill_graph58 | 23 | 5 | 7 |
| aten | prefill_graph58_fw | 22 | 5 | 16 |
| dynamo | prefill_graph59 | 18 | 6 | 7 |
| aten | prefill_graph59_fw | 30 | 6 | 23 |
| dynamo | prefill_graph60 | 23 | 5 | 7 |
| aten | prefill_graph60_fw | 22 | 5 | 16 |
| dynamo | prefill_graph61 | 124 | 15 | 68 |
| aten | prefill_graph61_fw | 163 | 15 | 145 |
| dynamo | prefill_graph62 | 13 | 2 | 5 |
| aten | prefill_graph62_fw | 13 | 2 | 10 |
| dynamo | prefill_graph63 | 5 | 2 | 2 |
| aten | prefill_graph63_fw | 8 | 2 | 5 |

### top ops — prefill_graph0_fw

| op | count |
| --- | --- |
| torch._ops.aten.unsqueeze.default | 6 |
| torch._ops.aten.arange.start_step | 2 |
| torch._ops.aten.add.Tensor | 2 |
| torch._ops.aten.le.Tensor | 1 |
| torch._ops.aten.expand.default | 1 |
| torch._ops.aten.lift_fresh_copy.default | 1 |
| torch._ops.aten.scalar_tensor.default | 1 |
| torch._ops.aten.where.self | 1 |

### top ops — prefill_graph1_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 12 |
| torch._ops.aten.permute.default | 8 |
| torch._ops.aten.select.int | 7 |
| torch._ops.aten.unsqueeze.default | 6 |
| torch._ops.aten.expand.default | 5 |
| torch._ops.aten._to_copy.default | 4 |
| torch._ops.aten.slice.Tensor | 4 |
| torch._ops.aten.copy.default | 2 |
| torch._ops.aten.slice_scatter.default | 2 |
| torch._ops.aten.arange.start_step | 2 |
| torch._ops.aten.eq.Scalar | 2 |
| torch._ops.aten.where.self | 2 |
| torch._ops.aten.mul.Tensor | 2 |
| torch._ops.aten.bmm.default | 1 |
| torch._ops.aten.cat.default | 1 |
| torch._ops.aten.cos.default | 1 |
| torch._ops.aten.sin.default | 1 |

### top ops — prefill_graph2_fw

| op | count |
| --- | --- |
| torch._ops.aten._to_copy.default | 3 |
| torch._ops.aten.add.Tensor | 2 |
| torch._ops.aten.mul.Tensor | 2 |
| torch._ops.aten.pow.Tensor_Scalar | 1 |
| torch._ops.aten.mean.dim | 1 |
| torch._ops.aten.rsqrt.default | 1 |

### top ops — prefill_graph3_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph4_fw

| op | count |
| --- | --- |
| torch._ops.aten.empty_permuted.default | 1 |

### top ops — prefill_graph5_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 5 |
| _operator.getitem | 3 |
| torch._ops.aten._to_copy.default | 2 |
| torch._ops.aten.exp.default | 2 |
| torch._ops.aten.unsqueeze.default | 2 |
| torch._ops.aten.expand.default | 2 |
| torch._ops.aten.clone.default | 2 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten.sigmoid.default | 1 |
| torch._ops.aten.neg.default | 1 |
| torch._ops.aten.add.Tensor | 1 |
| torch._ops.aten.log1p.default | 1 |
| torch._ops.aten.gt.Scalar | 1 |
| torch._ops.aten.where.self | 1 |
| torch._ops.aten.mul.Tensor | 1 |

### top ops — prefill_graph6_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph7_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 6 |
| torch._ops.aten._to_copy.default | 5 |
| torch._ops.aten.add.Tensor | 4 |
| torch._ops.aten.mul.Tensor | 4 |
| torch._ops.aten.permute.default | 3 |
| torch._ops.aten.mm.default | 3 |
| torch._ops.aten.pow.Tensor_Scalar | 1 |
| torch._ops.aten.mean.dim | 1 |
| torch._ops.aten.rsqrt.default | 1 |
| torch._ops.aten.sigmoid.default | 1 |

### top ops — prefill_graph8_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph9_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph10_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph11_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph12_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph13_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph14_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph15_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph16_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph17_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph18_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph19_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph20_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph21_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph22_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph23_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph24_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph25_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph26_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph27_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph28_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph29_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph30_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph31_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph32_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph33_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph34_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph35_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph36_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph37_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph38_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph39_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph40_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph41_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph42_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph43_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph44_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph45_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph46_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph47_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph48_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph49_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph50_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph51_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph52_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph53_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph54_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph55_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph56_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph57_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph58_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph59_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 9 |
| torch._ops.aten.permute.default | 5 |
| torch._ops.aten.mm.default | 4 |
| torch._ops.aten.constant_pad_nd.default | 1 |
| torch._ops.aten.full.default | 1 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.squeeze.dims | 1 |

### top ops — prefill_graph60_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 8 |
| torch._ops.aten.full_like.default | 1 |
| torch._ops.aten.copy.default | 1 |
| torch._ops.aten.empty_permuted.default | 1 |
| torch._ops.aten.empty.memory_format | 1 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 1 |
| _operator.getitem | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

### top ops — prefill_graph61_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 28 |
| torch._ops.aten._to_copy.default | 16 |
| torch._ops.aten.mul.Tensor | 16 |
| torch._ops.aten.add.Tensor | 13 |
| torch._ops.aten.permute.default | 12 |
| torch._ops.aten.slice.Tensor | 8 |
| torch._ops.aten.mm.default | 7 |
| torch._ops.aten.cat.default | 6 |
| torch._ops.aten.expand.default | 6 |
| torch._ops.aten.clone.default | 5 |
| torch._ops.aten.pow.Tensor_Scalar | 4 |
| torch._ops.aten.mean.dim | 4 |
| torch._ops.aten.rsqrt.default | 4 |
| torch._ops.aten.unsqueeze.default | 4 |
| _operator.getitem | 2 |
| torch._ops.aten.neg.default | 2 |
| torch._ops.aten.lift_fresh_copy.default | 2 |
| torch._ops.aten.bmm.default | 2 |
| torch._ops.aten.sigmoid.default | 2 |
| torch._ops.aten.split_with_sizes.default | 1 |
| torch._ops.aten._softmax.default | 1 |

### top ops — prefill_graph62_fw

| op | count |
| --- | --- |
| torch._ops.aten._to_copy.default | 3 |
| torch._ops.aten.add.Tensor | 2 |
| torch._ops.aten.mul.Tensor | 2 |
| torch._ops.aten.pow.Tensor_Scalar | 1 |
| torch._ops.aten.mean.dim | 1 |
| torch._ops.aten.rsqrt.default | 1 |

### top ops — prefill_graph63_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 2 |
| torch._ops.aten.slice.Tensor | 1 |
| torch._ops.aten.permute.default | 1 |
| torch._ops.aten.mm.default | 1 |

## decode (batch=1 seq_len=1 past=128)

dynamo: graph_count=1 graph_break_count=0

| level | graph | nodes | placeholders | call_function |
| --- | --- | --- | --- | --- |
| dynamo | decode_graph0 | 3337 | 493 | 1545 |
| aten | decode_graph0_fw | 4439 | 493 | 3944 |

### top ops — decode_graph0_fw

| op | count |
| --- | --- |
| torch._ops.aten.view.default | 910 |
| torch._ops.aten.permute.default | 393 |
| torch._ops.aten._to_copy.default | 375 |
| torch._ops.aten.mul.Tensor | 300 |
| torch._ops.aten.add.Tensor | 276 |
| torch._ops.aten.mm.default | 249 |
| _operator.getitem | 208 |
| torch._ops.aten.expand.default | 103 |
| torch._ops.aten.unsqueeze.default | 93 |
| torch._ops.aten.pow.Tensor_Scalar | 81 |
| torch._ops.aten.mean.dim | 81 |
| torch._ops.aten.rsqrt.default | 81 |
| torch._ops.aten.split_with_sizes.default | 80 |
| torch._ops.aten.clone.default | 80 |
| torch._ops.aten.empty_permuted.default | 72 |
| torch._ops.aten.slice.Tensor | 70 |
| torch._ops.aten.sigmoid.default | 64 |
| torch._ops.aten.cat.default | 49 |
| torch._ops.aten.exp.default | 48 |
| torch._ops.aten.empty.memory_format | 48 |
| torch.ops.higher_order.triton_kernel_wrapper_functional | 48 |
| torch._ops.aten.neg.default | 40 |
| torch._ops.aten.where.self | 27 |
| torch._ops.aten.copy.default | 26 |
| torch._ops.aten.squeeze.dims | 24 |
