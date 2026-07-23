# 도구 설계 — 무엇을 어떻게 캡처하나

`capture_qwen35_fx.py` 한 파일이 캡처 전부를 담당한다. 결과 해석은 [`analysis.md`](analysis.md).

---

## 0. 한 줄 요약

`torch.compile(module, backend=<덤프 백엔드>)`로 **Dynamo FX 그래프**를 받고, 그 안에서
`aot_autograd(fw_compiler=..., decompositions=core_aten_decompositions())`로 **ATen 레벨 그래프**까지
받아 readable python / node table / JSON으로 떨군다. prefill과 decode는 **서로 다른 그래프**로 따로 캡처한다.

---

## 1. 왜 prefill / decode를 따로 캡처하나

Qwen3.5는 **하이브리드**다. `layer_types`가 `[linear_attention ×3, full_attention] × 8` 패턴으로 32 레이어:

| | |
|---|---|
| hidden_size / intermediate_size | 2560 / 9216 |
| full_attention 8층 | heads 16, KV heads 4, head_dim 256, mrope(section `[11,11,10]`, `rope_theta 1e7`, partial_rotary 0.25) |
| linear_attention 24층 | `Qwen3_5GatedDeltaNet` — key head 16×128, value head 32×128, conv kernel 4 |
| vocab | 248320 (`tie_word_embeddings=True`) |

`Qwen3_5GatedDeltaNet.forward`는 파이썬 레벨에서 갈라진다
(`transformers/models/qwen3_5/modeling_qwen3_5.py:473, :527`):

```python
if use_precomputed_states and seq_len == 1:
    mixed_qkv = self.causal_conv1d_update(...)                  # decode: conv state in-place 갱신
    core_attn_out, ... = self.recurrent_gated_delta_rule(...)   # decode: step 재귀
else:
    mixed_qkv = F.silu(self.conv1d(mixed_qkv)[...])             # prefill: 진짜 causal conv
    core_attn_out, ... = self.chunk_gated_delta_rule(...)       # prefill: 청크 병렬
```

즉 **같은 모듈이 두 개의 다른 그래프로 트레이스된다.** Dynamo는 `seq_len == 1`을 가드로 특수화하므로,
prefill 그래프에는 `chunk_gated_delta_rule`이, decode 그래프에는 `recurrent_gated_delta_rule`이 들어간다.
그래서 스크립트는 두 스텝을 **별도의 `torch._dynamo.reset()` + 별도 캡처**로 돌린다.

decode 캡처는 캐시가 채워져 있어야 의미가 있으므로, **eager로 prefill을 먼저 한 번 돌려 `DynamicCache`를
채운 뒤** 1토큰을 컴파일해 통과시킨다.

---

## 2. 무엇을 컴파일하나

`TextDecoderForCapture` 래퍼 하나만 컴파일한다 (vision 타워 제외, 텍스트 경로만):

```
input_ids, position_ids, past_key_values
   → Qwen3_5TextModel (embed → 32 hybrid layers → RMSNorm)
   → lm_head(hidden[:, -1:, :])        # 마지막 위치만 → logits [B, 1, 248320]
```

`lm_head`를 마지막 토큰에만 태우는 건 실제 추론과 같고, prefill에서 `[1, 128, 248320]`(≈63M 원소)
logits가 그래프에 생기는 걸 막는다. `--no-lm-head`로 디코더만 캡처할 수도 있다.

**가중치는 기본 랜덤**(`--weights random`): `config.json`만 받아 `from_config`로 초기화한다. 그래프
구조·shape·dtype은 실제 체크포인트와 동일하고 모델 다운로드가 없다. 실제 값이 필요하면 `--weights real`.

---

## 3. 코드 읽는 순서 (= 실행 순서)

### (1) `main()` — 왜 `import torch`가 함수 안에 있는가
`TORCH_LOGS` / `TORCH_LOGS_OUT` / `TORCH_COMPILE_DEBUG`는 **torch import 시점에 읽힌다.** 그래서 env를
먼저 세팅하고 그 다음에 import한다. 이 순서가 뒤집히면 로그 덤프가 조용히 안 나온다.

### (2) `build_module()` — 컴파일 대상 조립 (2장)

### (3) `make_inputs()` — prefill / decode 입력
- prefill: `input_ids [1,128]`, `position_ids = arange(128)`, 빈 `DynamicCache`
- decode: eager prefill로 캐시를 채운 뒤 `input_ids [1,1]`, `position_ids = [[128]]`

### (4) `make_capture_backend()` — 핵심, seam 2개
```python
def backend(gm, example_inputs):          # ← seam 1: Dynamo가 넘겨주는 FX 그래프
    save_graph_module(gm, ..., "dynamo/")
    def fw_compiler(aot_gm, aot_inputs):  # ← seam 2: AOTAutograd가 ATen 분해 후 넘겨주는 그래프
        save_graph_module(aot_gm, ..., "aten/")
        return make_boxed_func(aot_gm.forward)
    return aot_autograd(fw_compiler=fw_compiler,
                        decompositions=core_aten_decompositions())(gm, example_inputs)
```
**seam 2가 커스텀 컴파일러 백엔드(Inductor 대체 백엔드)가 실제로 붙는 자리**다. 이 스크립트는 거기에
"파일로 저장"만 하는 백엔드를 끼운 것이라, `aten/` 덤프는 그런 백엔드가 받게 될 그래프와 같은 레벨이다.

### (5) `save_graph_module()` / `write_report()` — 산출물 (README 참고)

---

## 4. 덤프 방식 3가지

### (a) 커스텀 백엔드 — 기본, `--backend dump`
위 seam 2개에서 그래프를 파일로 저장. 그래프마다 6개 파일 + 스텝별 `report.md`.

### (b) `TORCH_LOGS` — `--torch-logs`
`TORCH_LOGS=graph_code,aot_graphs,graph_breaks,recompiles` + `TORCH_LOGS_OUT=<out>/torch_logs.txt`를
torch import 전에 설정한다. 파일 하나에 Dynamo 그래프 코드와 AOT 그래프가 순서대로 쌓인다.

### (c) 인덕터 트레이스 — `--backend inductor --inductor-trace`
`TORCH_COMPILE_DEBUG=1` + `torch._inductor.config.trace.enabled`. 진짜 인덕터로 컴파일하면서
`<out>/torchinductor/model__0_inference_0.0/` 아래에 남긴다:

```
fx_graph_readable.py     fx_graph_transformed.py     fx_graph_runnable.py
ir_pre_fusion.txt        ir_post_fusion.txt          output_code.py
```

인덕터의 fusion 전/후 IR과 최종 C++ 커널까지 보고 싶을 때만 쓴다 (`setuptools` 필요).

추가로 `--explain`은 `torch._dynamo.explain()` 결과를 `<step>.explain.txt`로 남긴다.

---

## 5. 자주 쓰는 옵션

- `--seq-len N` prefill 길이, `--decode-past N` decode 시점의 캐시 길이 (기본 `--seq-len`)
- `--attn-impl eager|sdpa` — `eager`(기본)는 softmax/matmul이 그래프에 그대로 보이고, `sdpa`는 fused op 하나
- `--aten-decomp core|inductor|none` — AOTAutograd에 넘길 분해 테이블
- `--dynamic false|true|auto` — 기본 `false` (shape 고정; 커널 작업엔 이게 맞다)
- `--layers N`, `--vocab-size V` — 스모크용 축소
- `--fullgraph` — graph break가 있으면 실패시킴
- `--weights real` — 실제 체크포인트 로드
- `--to-folder` — `gm.to_folder()`로 재로드 가능한 모듈까지 (가중치를 쓰므로 큼)

> `--layers N`으로 줄일 때 스크립트가 **full_attention을 최소 1개 유지**한다. 전부 linear만 남으면
> `create_causal_mask → get_seq_length()`가 "Cache seem to only contain LinearAttention layers"로 터진다.
> 그래서 스모크 최소 단위가 `--layers 4` (= `[linear, linear, linear, full]`)다.
