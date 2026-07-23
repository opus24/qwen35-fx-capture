# prefill vs decode — `qwen35_4b_s128` (dynamo level)

## 1. 그래프 크기

|  | prefill | decode | 차이 |
| --- | --- | --- | --- |
| 노드 수 | 19897 | 4561 | 4.4x |
| call_function | 11089 | 2097 | 5.3x |
| placeholder(입력) | 429 | 493 | 64 |
| output(출력) | 65 | 17 | -48 |

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
| `torch.zeros` | `['1', '8192', '4']` | torch.bfloat16 | 24 |
| `torch.zeros_like` | `['1', '32', '128', '128']` | torch.float32 | 24 |
| `torch.cat` | `['1', '4', '128', '256']` | torch.bfloat16 | 16 |
| `torch._C._nn.linear` | `['1', '1', '248320']` | torch.bfloat16 | 1 |

### decode — 총 17개

| 생성 op | shape | dtype | 개수 |
| --- | --- | --- | --- |
| `torch.cat` | `['1', '4', '129', '256']` | torch.bfloat16 | 16 |
| `torch._C._nn.linear` | `['1', '1', '248320']` | torch.bfloat16 | 1 |


## 4. op 히스토그램

공통 op 44종 / prefill 전용 9종 / decode 전용 1종

### 4-1. 개수까지 동일한 op (= 레이어 구조 그 자체)

| op | 개수 |
| --- | --- |
| `torch._C._nn.linear` | 249 |
| `contiguous` | 160 |
| `torch.rsqrt` | 153 |
| `mean` | 105 |
| `pow` | 105 |
| `type_as` | 81 |
| `torch.nn.functional.silu` | 80 |
| `copy_` | 48 |
| `repeat_interleave` | 48 |
| `view` | 32 |
| `sigmoid` | 24 |
| `torch._C._nn.softplus` | 24 |
| `torch.conv1d` | 24 |
| `torch.functional.split` | 24 |
| `expand` | 19 |
| `torch.matmul` | 16 |
| `torch.chunk` | 8 |
| `torch.nn.functional.dropout` | 8 |
| `torch.nn.functional.softmax` | 8 |
| `torch.sigmoid` | 8 |

### 4-2. 개수가 다른 op (= 토큰 축 / 시퀀스 처리 차이)

| op | prefill | decode | 비율 |
| --- | --- | --- | --- |
| `_operator.getitem` | 3754 | 346 | 10.8x |
| `_operator.mul` | 2260 | 580 | 3.9x |
| `_operator.add` | 1980 | 372 | 5.3x |
| `unsqueeze` | 1648 | 208 | 7.9x |
| `_operator.setitem` | 1562 | 26 | 60.1x |
| `sum` | 1560 | 96 | 16.2x |
| `reshape` | 368 | 200 | 1.8x |
| `transpose` | 353 | 233 | 1.5x |
| `_operator.matmul` | 313 | 1 | 313.0x |
| `to` | 251 | 323 | 0.8x |
| `float` | 238 | 214 | 1.1x |
| `exp` | 216 | 48 | 4.5x |
| `_operator.sub` | 120 | 24 | 5.0x |
| `_operator.neg` | 64 | 40 | 1.6x |
| `torch.cat` | 49 | 73 | 0.7x |
| `torch.zeros` | 48 | 24 | 2.0x |
| `torch.tensor` | 17 | 1 | 17.0x |

### 4-3. 한쪽에만 있는 op

| op | prefill | decode |
| --- | --- | --- |
| `clone` | 3024 | - |
| `torch._C._nn.pad` | 144 | - |
| `torch.ones` | 48 | - |
| `torch.triu` | 48 | - |
| `tril` | 48 | - |
| `torch.zeros_like` | 48 | - |
| `cumsum` | 24 | - |
| `masked_fill` | 24 | - |
| `torch.eye` | 24 | - |
| `squeeze` | - | 24 |

## 5. 노드가 어느 소스에서 나왔나 (readable 덤프의 `# File:` 주석 기준)

### prefill — 귀속된 노드 17873개

| source | 노드 | 비중 |
| --- | --- | --- |
| `modeling_qwen3_5.py:torch_chunk_gated_delta_rule` | 14448 | 80.8% |
| `modeling_qwen3_5.py:forward` | 1770 | 9.9% |
| `modeling_qwen3_5.py:l2norm` | 600 | 3.4% |
| `modeling_qwen3_5.py:_norm` | 405 | 2.3% |
| `functional.py:pad` | 144 | 0.8% |
| `modeling_qwen3_5.py:apply_rotary_pos_emb` | 112 | 0.6% |
| `modeling_qwen3_5.py:eager_attention_forward` | 80 | 0.4% |
| `cache_utils.py:lazy_initialization` | 64 | 0.4% |

### decode — 귀속된 노드 4009개

| source | 노드 | 비중 |
| --- | --- | --- |
| `modeling_qwen3_5.py:forward` | 1722 | 43.0% |
| `modeling_qwen3_5.py:torch_recurrent_gated_delta_rule` | 672 | 16.8% |
| `modeling_qwen3_5.py:l2norm` | 600 | 15.0% |
| `modeling_qwen3_5.py:_norm` | 405 | 10.1% |
| `modeling_qwen3_5.py:torch_causal_conv1d_update` | 216 | 5.4% |
| `modeling_qwen3_5.py:apply_rotary_pos_emb` | 112 | 2.8% |
| `modeling_qwen3_5.py:eager_attention_forward` | 80 | 2.0% |
| `modeling_qwen3_5.py:rotate_half` | 64 | 1.6% |

