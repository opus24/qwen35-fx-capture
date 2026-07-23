# prefill vs decode — `qwen35_4b_s128` (aten level)

## 1. 그래프 크기

|  | prefill | decode | 차이 |
| --- | --- | --- | --- |
| 노드 수 | 42023 | 5783 | 7.3x |
| call_function | 41576 | 5288 | 7.9x |
| placeholder(입력) | 429 | 493 | 64 |
| output(출력) | 65 | 65 | 0 |

## 2. 입력 (placeholder)

### decode에만 있는 입력 = 캐시 상태

| shape | 개수 |
| --- | --- |
| `['1', '8192', '4']` | 24 |
| `['1', '32', '128', '128']` | 24 |
| `['1', '4', '128', '256']` | 16 |
| `['1', '1']` | 2 |

### prefill에만 있거나 개수가 다른 입력

| shape | prefill | decode |
| --- | --- | --- |
| `['1', '128']` | 2 | 0 |

## 3. 출력 (logits + 캐시 write-back)

### prefill — 총 65개

| 생성 op | shape | dtype | 개수 |
| --- | --- | --- | --- |
| `copy.default` | `['1', '8192', '4']` | torch.bfloat16 | 24 |
| `copy.default` | `['1', '32', '128', '128']` | torch.float32 | 24 |
| `cat.default` | `['1', '4', '128', '256']` | torch.bfloat16 | 16 |
| `view.default` | `['1', '1', '248320']` | torch.bfloat16 | 1 |

### decode — 총 65개

| 생성 op | shape | dtype | 개수 |
| --- | --- | --- | --- |
| `copy.default` | `['1', '8192', '4']` | torch.bfloat16 | 24 |
| `copy.default` | `['1', '32', '128', '128']` | torch.float32 | 24 |
| `cat.default` | `['1', '4', '129', '256']` | torch.bfloat16 | 16 |
| `view.default` | `['1', '1', '248320']` | torch.bfloat16 | 1 |


## 4. op 히스토그램

공통 op 39종 / prefill 전용 8종 / decode 전용 1종

### 4-1. 개수까지 동일한 op (= 레이어 구조 그 자체)

| op | 개수 |
| --- | --- |
| `mm.default` | 249 |
| `rsqrt.default` | 153 |
| `sigmoid.default` | 112 |
| `mean.dim` | 105 |
| `pow.Tensor_Scalar` | 105 |
| `_operator.getitem` | 88 |
| `split_with_sizes.default` | 32 |
| `convolution.default` | 24 |
| `gt.Scalar` | 24 |
| `log1p.default` | 24 |
| `_softmax.default` | 8 |
| `cos.default` | 1 |
| `embedding.default` | 1 |
| `le.Tensor` | 1 |
| `sin.default` | 1 |

### 4-2. 개수가 다른 op (= 토큰 축 / 시퀀스 처리 차이)

| op | prefill | decode | 비율 |
| --- | --- | --- | --- |
| `slice.Tensor` | 6166 | 118 | 52.3x |
| `select.int` | 5119 | 151 | 33.9x |
| `unsqueeze.default` | 3693 | 309 | 12.0x |
| `view.default` | 3478 | 862 | 4.0x |
| `clone.default` | 3256 | 80 | 40.7x |
| `mul.Tensor` | 2340 | 660 | 3.5x |
| `expand.default` | 2287 | 127 | 18.0x |
| `add.Tensor` | 1980 | 372 | 5.3x |
| `arange.start_step` | 1756 | 28 | 62.7x |
| `where.self` | 1659 | 51 | 32.5x |
| `copy.default` | 1610 | 74 | 21.8x |
| `eq.Scalar` | 1562 | 26 | 60.1x |
| `sum.dim_IntList` | 1560 | 96 | 16.2x |
| `slice_scatter.default` | 1514 | 2 | 757.0x |
| `_to_copy.default` | 663 | 639 | 1.0x |
| `permute.default` | 609 | 489 | 1.2x |
| `bmm.default` | 329 | 17 | 19.4x |
| `exp.default` | 240 | 72 | 3.3x |
| `sub.Tensor` | 192 | 24 | 8.0x |
| `scalar_tensor.default` | 73 | 1 | 73.0x |

### 4-3. 한쪽에만 있는 op

| op | prefill | decode |
| --- | --- | --- |
| `constant_pad_nd.default` | 144 | - |
| `le.Scalar` | 48 | - |
| `full_like.default` | 48 | - |
| `ge.Scalar` | 24 | - |
| `logical_and.default` | 24 | - |
| `cumsum.default` | 24 | - |
| `eq.Tensor` | 24 | - |
| `alias.default` | 24 | - |
| `squeeze.dims` | - | 24 |

## 5. 노드가 어느 소스에서 나왔나 (readable 덤프의 `# File:` 주석 기준)

### prefill — 귀속된 노드 41562개

| source | 노드 | 비중 |
| --- | --- | --- |
| `modeling_qwen3_5.py:torch_chunk_gated_delta_rule` | 36840 | 88.6% |
| `modeling_qwen3_5.py:forward` | 2830 | 6.8% |
| `modeling_qwen3_5.py:l2norm` | 576 | 1.4% |
| `modeling_qwen3_5.py:_norm` | 405 | 1.0% |
| `modeling_qwen3_5.py:eager_attention_forward` | 168 | 0.4% |
| `functional.py:pad` | 144 | 0.3% |
| `activations.py:forward` | 128 | 0.3% |
| `modeling_qwen3_5.py:apply_rotary_pos_emb` | 112 | 0.3% |

### decode — 귀속된 노드 5258개

| source | 노드 | 비중 |
| --- | --- | --- |
| `modeling_qwen3_5.py:forward` | 2710 | 51.5% |
| `modeling_qwen3_5.py:torch_recurrent_gated_delta_rule` | 816 | 15.5% |
| `modeling_qwen3_5.py:l2norm` | 456 | 8.7% |
| `modeling_qwen3_5.py:_norm` | 405 | 7.7% |
| `modeling_qwen3_5.py:torch_causal_conv1d_update` | 240 | 4.6% |
| `modeling_qwen3_5.py:eager_attention_forward` | 160 | 3.0% |
| `activations.py:forward` | 128 | 2.4% |
| `modeling_qwen3_5.py:apply_rotary_pos_emb` | 112 | 2.1% |

