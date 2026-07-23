# prefill vs decode FX 그래프 분석

대상: 이 저장소에 커밋된 [`dumps/qwen35_4b_s128/`](../dumps/qwen35_4b_s128) — `Qwen/Qwen3.5-4B`,
32 레이어, batch 1, prefill 128토큰 / decode 1토큰(past 128), bf16, **ATen 레벨**(AOTAutograd 분해 후) 기준.
테스트 조건 전체는 [README](../README.md), 도구 설계는 [`design.md`](design.md).

기계 생성 원본: [`compare_aten.md`](../dumps/qwen35_4b_s128/compare_aten.md)
(재생성: `python compare_graphs.py dumps/qwen35_4b_s128 --level aten`).

---

## 0. 한 줄 요약

> **가중치를 태우는 부분(`aten.mm` 249개)은 완전히 동일하고, 토큰 축을 다루는 부분만 다르다.**
> prefill은 시퀀스 128을 청크로 병렬 처리(`chunk_gated_delta_rule`)하느라 노드가 7.3배 크고,
> decode는 캐시 상태 64개를 입력으로 더 받아 1토큰 재귀(`recurrent_gated_delta_rule`)를 돈다.

## 1. 크기

| | prefill | decode | 차이 |
|---|---|---|---|
| 노드 수 | 42,023 | 5,783 | 7.3× |
| call_function | 41,576 | 5,288 | 7.9× |
| placeholder(입력) | 429 | 493 | **+64** |
| output(출력) | 65 | 65 | 0 |

---

## 2. 공통점

### (a) 출력 구조가 동일하다 — 둘 다 정확히 65개

| 출력 | 개수 | prefill shape | decode shape |
|---|---|---|---|
| logits | 1 | `[1, 1, 248320]` | `[1, 1, 248320]` |
| KV 캐시 (full_attention 8층 × K,V) | 16 | `[1, 4, 128, 256]` | `[1, 4, **129**, 256]` |
| conv state (linear 24층) | 24 | `[1, 8192, 4]` | `[1, 8192, 4]` |
| recurrent state (linear 24층) | 24 | `[1, 32, 128, 128]` fp32 | `[1, 32, 128, 128]` fp32 |

캐시 갱신은 HF 쪽에서 in-place지만, AOT functionalize를 거쳐 **그래프 출력**(`copy` / `cat`)으로 나온다.
KV만 128 → 129로 자라고 conv/recurrent state는 **크기가 고정**이다 — linear attention의 성질이 그래프에 그대로 보인다.

### (b) 가중치 GEMM이 개수·N·K까지 완전히 동일 — `aten.mm` 249개

M(토큰 수)만 128 vs 1이고 **N·K와 개수는 같다** (`M×K @ K×N` 표기):

| K×N | 개수 | 정체 (모듈 정의로 확인) |
|---|---|---|
| `2560×9216` | 64 | MLP `gate_proj` + `up_proj` (32층 × 2) |
| `9216×2560` | 32 | MLP `down_proj` (32층) |
| `2560×8192` | 32 | linear-attn `in_proj_qkv` 24 (= key_dim×2 + value_dim = 2048×2+4096) + full-attn `q_proj` 8 (**gated라 16헤드×256×2 = 8192**) |
| `4096×2560` | 32 | linear-attn `out_proj` 24 + full-attn `o_proj` 8 |
| `2560×4096` | 24 | linear-attn `in_proj_z` (24층) |
| `2560×32` | 48 | linear-attn `in_proj_b` / `in_proj_a` (24층 × 2, out = num_v_heads = 32) |
| `2560×1024` | 16 | full-attn `k_proj` / `v_proj` (8층 × 2, KV 4헤드 × 256) |
| `2560×248320` | 1 | `lm_head` — **둘 다 M=1** (마지막 토큰만 태우므로) |

합계 249. → 커널 관점에서 **"prefill = 같은 GEMM의 M만 큰 것"**으로 읽으면 된다.
짚어둘 포인트: full-attention의 `q_proj`가 **gated**라 출력이 `heads×head_dim×2`(8192)이고, 그래서
linear-attn의 `in_proj_qkv`와 같은 `2560×8192` 버킷에 합쳐진다 (24 + 8 = 32).

### (c) 레이어 구조를 드러내는 op은 개수가 같다 — 숫자가 그대로 레이어 수를 증명한다

| op | 개수 | 분해 (`readable.py`의 소스 주석으로 확인) |
|---|---|---|
| `_softmax` | 8 | full_attention 8층 |
| `convolution` | 24 | linear_attention 24층 |
| `mean.dim` | 105 | RMSNorm 적용 횟수 = 81 (`_norm`: 32층×2 + 최종 1 + q/k_norm 8×2) + 24 (`RMSNormGated`) |
| `rsqrt` | 153 | 105 (위 RMSNorm) + 48 (`l2norm`: 24층 × q,k) |
| `sigmoid` | 112 | 32 MLP SiLU + 24 conv SiLU + 24 `beta=b.sigmoid()` + 24 gated-norm SiLU + **8 attention 출력 게이팅** |
| `split_with_sizes` | 32 | linear-attn QKV 분할 24 + full-attn `query_states, gate = torch.chunk(...)` 8 |
| `embedding` | 1 | `embed_tokens` |

core-ATen 분해에서 **SiLU는 `mul(x, sigmoid(x))`로 풀리므로 `silu` op이 따로 안 보이고 `sigmoid`에 합산**된다.

decode의 conv 24개는 prefill의 `F.conv1d`(시퀀스 전체)가 아니라 `torch_causal_conv1d_update` 안의
4탭 conv다 — **op 이름과 개수는 같지만 의미가 다르다**는 게 함정.

---

## 3. 차이점

### (a) decode에만 있는 입력 64개 = 캐시가 그래프 입력으로 들어온다

| shape | 개수 | 정체 |
|---|---|---|
| `[1, 8192, 4]` | 24 | linear 24층의 conv state |
| `[1, 32, 128, 128]` | 24 | linear 24층의 recurrent state (fp32) |
| `[1, 4, 128, 256]` | 16 | full 8층의 K, V 캐시 |

(그 외 `[1,1]` 2개 = `input_ids` / `position_ids`, prefill에선 `[1,128]` 2개.)
prefill은 캐시가 비어 있어 입력이 아니라 **출력으로만** 나온다 — 이게 429 vs 493의 전부다.

### (b) 서로 다른 커널을 탄다 (같은 모듈, 다른 그래프)

| | prefill | decode |
|---|---|---|
| conv | `F.conv1d` (causal, 시퀀스 전체) | `torch_causal_conv1d_update` (state 4탭 + 1토큰) |
| delta rule | `torch_chunk_gated_delta_rule` | `torch_recurrent_gated_delta_rule` |
| `aten.bmm` | **329** | **17** |

bmm 내역: prefill은 청크 delta rule이 312개(예: `32×64×128 @ 32×128×128` 96개, chunk 64 단위)를,
full attention이 16개(`16×128×256 @ 16×256×128`)를, rope가 1개를 만든다. decode는 attention 16개
(`16×1×256 @ 16×256×129`)와 rope 1개뿐이다.

### (c) 노드가 어디서 나왔나 — prefill 그래프의 88.6%가 청크 delta rule

| prefill | 노드 | 비중 |
|---|---|---|
| `torch_chunk_gated_delta_rule` | 36,840 | **88.6%** |
| `forward` (레이어 본체) | 2,830 | 6.8% |
| `l2norm` | 576 | 1.4% |
| `_norm` (RMSNorm) | 405 | 1.0% |
| `eager_attention_forward` | 168 | 0.4% |

| decode | 노드 | 비중 |
|---|---|---|
| `forward` (레이어 본체) | 2,710 | 51.5% |
| `torch_recurrent_gated_delta_rule` | 816 | 15.5% |
| `l2norm` | 456 | 8.7% |
| `_norm` (RMSNorm) | 405 | 7.7% |
| `torch_causal_conv1d_update` | 240 | 4.6% |

**이게 prefill이 7.3배 큰 이유의 전부다.** `fla`(flash-linear-attention) 라이브러리가 없어 순수 torch
참조 구현으로 폴백되는데, 그 구현 안에 **파이썬 for 루프**가 있어 그래프에 통째로 언롤된다
(`modeling_qwen3_5.py:293`, `chunk_size=64`):

```python
for i in range(1, chunk_size):            # 63회 — 청크 내 forward substitution
    row = attn[..., i, :i].clone()
    sub = attn[..., :i, :i].clone()
    attn[..., i, :i] = row + (row.unsqueeze(-1) * sub).sum(-2)
```

Dynamo는 루프를 트레이스하면서 **63번을 전부 펼친다.** 반복마다 clone 2 + slice/select 여러 개 +
mul/sum + 슬라이스 대입(→ `slice_scatter`)이 생기니 layer당 약 1,535 노드(36,840 ÷ 24층)가 나온다.
`slice` 6,166 / `select` 5,119 / `clone` 3,256 / `sum.dim_IntList` 1,560이 폭증한 것도 전부 이 루프다.

**모델의 본질적 연산량이 아니라 참조 구현의 언롤**이라는 점이 중요하다. 모든 연산이 ATen으로 드러나
분석에는 오히려 유리하지만, "prefill이 원래 이렇게 무겁다"로 읽히면 오해다.

### (d) 한쪽에만 있는 op

| op | prefill | decode | 이유 |
|---|---|---|---|
| `constant_pad_nd` | 144 | - | `F.pad` — 청크 경계 패딩 + prefill 경로에서만 도는 conv state 패딩 |
| `cumsum` | 24 | - | 청크 내 gate 누적합 |
| `full_like`, `le.Scalar` | 48, 48 | - | 청크 내 causal mask 생성 |
| `eq.Tensor`, `ge.Scalar`, `logical_and`, `alias` | 24씩 | - | 위와 동일 계열 |
| `squeeze.dims` | - | 24 | 1토큰 축 제거 (recurrent 경로) |

즉 **prefill 전용 op = 전부 "청크 처리" 부산물**, decode 전용 op = "토큰 축이 1이라 생기는 squeeze" 하나뿐.

---

## 4. 이 결과를 볼 때 같이 알아야 할 것

- **CPU 캡처**다. 그래프 메타의 `device='cpu'`만 타깃과 다르고 구조는 동일하다.
- **랜덤 가중치**(기본). 값 검증이 필요하면 `--weights real`.
- **batch 1, static shape** 캡처다. 다른 seq_len은 `--seq-len`으로 다시 뽑아야 한다
  (`--dynamic true`로 심볼릭 캡처도 되지만 커널 작업엔 static이 낫다).
- **`fla` / `causal-conv1d`를 설치하면 prefill 그래프가 완전히 달라진다** (custom op 하나로 뭉치거나
  graph break가 생길 수 있음). 이 덤프는 "설치 안 된 상태 = 전부 ATen으로 보이는" 버전이다.
- vision 타워는 캡처 대상이 아니다 (텍스트 경로만).
