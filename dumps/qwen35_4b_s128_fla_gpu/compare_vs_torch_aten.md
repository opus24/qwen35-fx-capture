# kernel-path diff — qwen35_4b_s128_fla_gpu vs qwen35_4b_s128_torch_gpu

## bound kernels

|  | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu |
| --- | --- | --- |
| delta_rule_chunk | `fla.ops.gated_delta_rule.chunk.chunk_gated_delta_rule` | `transformers.models.qwen3_5.modeling_qwen3_5.torch_chunk_gated_delta_rule` |
| delta_rule_recurrent | `fla.ops.gated_delta_rule.fused_recurrent.fused_recurrent_gated_delta_rule` | `transformers.models.qwen3_5.modeling_qwen3_5.torch_recurrent_gated_delta_rule` |
| conv1d_fn | `causal_conv1d.causal_conv1d_interface.causal_conv1d_fn` | `none` |
| conv1d_update | `causal_conv1d.causal_conv1d_interface.causal_conv1d_update` | `transformers.models.qwen3_5.modeling_qwen3_5.torch_causal_conv1d_update` |
| gated_rmsnorm | `fla.modules.fused_norm_gate.FusedRMSNormGated` | `transformers.models.qwen3_5.modeling_qwen3_5.Qwen3_5RMSNormGated` |
| fla_delta_rule_active | `True` | `False` |

## eager latency (CUDA events, median)

|  | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu | qwen35_4b_s128_torch_gpu / qwen35_4b_s128_fla_gpu |
| --- | --- | --- | --- |
| prefill (b1 s128) | 98.658 ms | 195.367 ms | 1.98× |
| decode (past=128) | 43.09 ms/token | 46.798 ms/token | 1.09× |
| peak memory | 8.594 GB | 8.611 GB |  |

weights identical: **True** (`{'numel': 4205751296, 'sum_finite': 4043.9586, 'nonfinite': 5}` vs `{'numel': 4205751296, 'sum_finite': 4043.9586, 'nonfinite': 5}`)
prefill logits — qwen35_4b_s128_fla_gpu: mean=-0.000777 std=1.010787 absmax=5.0625 | qwen35_4b_s128_torch_gpu: mean=-0.000897 std=1.010811 absmax=5.0625

## prefill — aten graphs

|  | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu |
| --- | --- | --- |
| dynamo graph count | 64 | 1 |
| graph breaks | 63 | 0 |
| aten graphs dumped | 64 | 1 |
| total nodes | 2740 | 42023 |
| largest single graph | 163 | 42023 |
| call_function nodes | 2255 | 41576 |

### top ops — prefill / aten

| op | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu |
| --- | --- | --- |
| slice.Tensor | 93 | 6166 |
| select.int | 7 | 5119 |
| view.default | 657 | 3478 |
| unsqueeze.default | 46 | 3693 |
| clone.default | 42 | 3256 |
| mul.Tensor | 139 | 2340 |
| expand.default | 56 | 2287 |
| add.Tensor | 115 | 1980 |
| arange.start_step | 4 | 1756 |
| where.self | 4 | 1659 |
| copy.default | 50 | 1610 |
| eq.Scalar | 2 | 1562 |
| sum.dim_IntList | 0 | 1560 |
| slice_scatter.default | 2 | 1514 |
| permute.default | 253 | 609 |
| _to_copy.default | 145 | 663 |
| mm.default | 180 | 249 |
| bmm.default | 17 | 329 |
| exp.default | 2 | 240 |
| sub.Tensor | 0 | 192 |
| rsqrt.default | 35 | 153 |
| constant_pad_nd.default | 24 | 144 |
| mean.dim | 35 | 105 |
| pow.Tensor_Scalar | 35 | 105 |
| _operator.getitem | 43 | 88 |

## decode — aten graphs

|  | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu |
| --- | --- | --- |
| dynamo graph count | 1 | 1 |
| graph breaks | 0 | 0 |
| aten graphs dumped | 1 | 1 |
| total nodes | 4439 | 5783 |
| largest single graph | 4439 | 5783 |
| call_function nodes | 3944 | 5288 |

### top ops — decode / aten

| op | qwen35_4b_s128_fla_gpu | qwen35_4b_s128_torch_gpu |
| --- | --- | --- |
| view.default | 910 | 862 |
| _to_copy.default | 375 | 639 |
| mul.Tensor | 300 | 660 |
| permute.default | 393 | 489 |
| add.Tensor | 276 | 372 |
| mm.default | 249 | 249 |
| unsqueeze.default | 93 | 309 |
| _operator.getitem | 208 | 88 |
| rsqrt.default | 81 | 153 |
| expand.default | 103 | 127 |
| slice.Tensor | 70 | 118 |
| mean.dim | 81 | 105 |
| pow.Tensor_Scalar | 81 | 105 |
| sigmoid.default | 64 | 112 |
| clone.default | 80 | 80 |
| select.int | 7 | 151 |
| cat.default | 49 | 73 |
| exp.default | 48 | 72 |
| split_with_sizes.default | 80 | 32 |
| copy.default | 26 | 74 |
| sum.dim_IntList | 0 | 96 |
| neg.default | 40 | 40 |
| where.self | 27 | 51 |
| empty_permuted.default | 72 | 0 |
| empty.memory_format | 48 | 0 |
