# qwen35-fx-capture

`Qwen/Qwen3.5-4B`의 **prefill / decode** 스텝을 `torch.compile`로 캡처해 FX 그래프를 덤프하고,
두 그래프를 비교 분석한 **결과까지 함께 담은** 저장소.

Qwen3.5는 하이브리드 모델이라 `Qwen3_5GatedDeltaNet`이 `seq_len == 1`에서 파이썬 레벨로 갈린다.
그래서 prefill과 decode는 **서로 다른 그래프**로 캡처된다 — 이 저장소의 출발점.

---

## 결과 바로 보기

| 보고 싶은 것 | 파일 |
|---|---|
| **prefill vs decode 비교 리포트** (ATen) | [`dumps/qwen35_4b_s128/compare_aten.md`](dumps/qwen35_4b_s128/compare_aten.md) |
| 같은 비교 (Dynamo 레벨) | [`dumps/qwen35_4b_s128/compare_dynamo.md`](dumps/qwen35_4b_s128/compare_dynamo.md) |
| 그래프 크기 + op 히스토그램 | [`dumps/qwen35_4b_s128/report.md`](dumps/qwen35_4b_s128/report.md) |
| **decode FX 그래프** (5,783 노드 — 통독 가능) | [`aten/decode_graph0_fw.readable.py`](dumps/qwen35_4b_s128/aten/decode_graph0_fw.readable.py) |
| **prefill FX 그래프** (42,023 노드 — grep 권장) | [`aten/prefill_graph0_fw.readable.py`](dumps/qwen35_4b_s128/aten/prefill_graph0_fw.readable.py) |
| 두 그래프 차이 상세 분석 | [`docs/analysis.md`](docs/analysis.md) |
| 도구 설계 + 덤프 방식 3가지 | [`docs/design.md`](docs/design.md) |

한 줄 결론: **가중치 GEMM(`aten.mm` 249개)은 prefill/decode가 완전히 동일하고 M(토큰 수)만 128 vs 1이다.
차이는 전부 토큰 축 처리에서 나온다.** 자세한 건 [`docs/analysis.md`](docs/analysis.md).

---

## 테스트 조건

이 저장소에 커밋된 `dumps/qwen35_4b_s128/`는 아래 조건으로 실제 캡처한 결과다.

| 항목 | 값 |
|---|---|
| 모델 | `Qwen/Qwen3.5-4B` — `text_config`만 사용 (vision 타워 제외) |
| 레이어 수 | **32** = linear_attention 24 + full_attention 8 (`[L,L,L,F] × 8`) |
| 파라미터 | 4.21B / hidden 2560 / FFN 9216 / vocab 248320 |
| 가중치 | 랜덤 초기화 (`config.json` 기반). 그래프 구조·shape·dtype은 실제 체크포인트와 동일 |
| dtype | bfloat16 (delta-rule 내부 일부는 fp32로 승격) |
| batch | 1 |
| **prefill 입력 토큰** | **128** → 출력 1 토큰분 logits `[1, 1, 248320]` |
| **decode 입력 토큰** | **1** (캐시 past=128) → 출력 1 토큰분 logits `[1, 1, 248320]` |
| attention 구현 | `eager` (sdpa면 attention이 fused op 하나로 뭉쳐 안 보인다) |
| shape | static (`dynamic=False`) |
| ATen 분해 | `core_aten_decompositions()` (1013 룰) |
| graph break | **0** — `--fullgraph` 통과, `torch._dynamo.explain()` = `Graph Count: 1, Break Count: 0` |
| 실행 환경 | CPU / torch 2.10.0 / transformers 5.14.1 / python 3.10 |
| linear-attn 커널 | **torch 참조 구현** (`torch_chunk_gated_delta_rule` / `torch_recurrent_gated_delta_rule`) — FLA fast path **미적용**. prefill 그래프 크기에 결정적이다 ([아래](#fla-fast-path-버전으로-뽑으려면), [`docs/analysis.md`](docs/analysis.md)) |
| 소요 시간 | 모델 빌드 267s / prefill 캡처 203s / decode 캡처 39s (48코어 CPU, 부하에 따라 변동) |

---

## 사용법

```bash
pip install "torch>=2.10" "transformers>=5.14" accelerate tabulate

# 캡처 (위 테스트 조건 그대로 재현)
python capture_qwen35_fx.py --seq-len 128 --mode both --out dumps/qwen35_4b_s128

# prefill vs decode 비교 리포트 생성
python compare_graphs.py dumps/qwen35_4b_s128 --level aten --out dumps/qwen35_4b_s128/compare_aten.md
```

빠른 스모크는 `--layers 4 --vocab-size 4096 --seq-len 32` (수십 초).
그 외 옵션은 `--help` 또는 [`docs/design.md`](docs/design.md).

### FLA fast path 버전으로 뽑으려면

이 저장소의 덤프는 **FLA(flash-linear-attention) fast path가 꺼진** 상태다. Qwen3.5는 커널을
import 시점에 `fla_fn or torch_fn`으로 고르는데(`modeling_qwen3_5.py:421-424`), transformers가
FLA 사용 여부를 **`is_torch_cuda_available()`으로 게이팅**하기 때문에 **CUDA GPU가 있어야만** 켜진다:

```python
# transformers/utils/import_utils.py
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(fla_version) >= version.parse("0.2.2")
```

GPU 머신에서 아래처럼 하면 FLA 버전 덤프가 나온다:

```bash
pip install flash-linear-attention        # delta rule 커널 (필수)
pip install causal-conv1d                 # conv 커널 (선택, CUDA 빌드 필요)

python capture_qwen35_fx.py --seq-len 128 --mode both --require-fla \
    --out dumps/qwen35_4b_s128_fla
```

`--require-fla`는 fast path가 실제로 활성화되지 않았으면 **덤프를 쓰지 않고 즉시 중단**한다
(torch 폴백 결과가 FLA 이름으로 잘못 저장되는 걸 막는다). 실제로 어떤 커널이 잡혔는지는
매 실행 시 `kernel path:` 줄로 출력되고 `report.md` / `report.json`에도 기록된다.

`causal-conv1d` 없이 FLA만 설치해도 **delta rule 커널만 교체**된다 (conv는 torch 경로 유지) —
심볼별 `or` 폴백이라 부분 활성화가 가능하다. prefill 그래프의 88.6%가 delta rule이므로
FLA만으로도 그래프는 크게 달라진다.

> 환경 메모: 이 저장소를 만든 CPU 전용 머신(python 3.10 / torch 2.10 / triton 3.6.0)에서는
> `flash-linear-attention==0.5.1`이 **import 단계에서 실패**했다
> (`fla/ops/simple_gla/parallel.py` → `triton/runtime/jit.py` `AttributeError: 'NoneType' object has no attribute 'start'`).
> fla는 python 3.11 이상을 권장한다. GPU 머신에서 FLA 버전을 뽑을 때 python/triton/fla 조합을
> 먼저 맞춰야 할 수 있다.

---

## 출력물

```
dumps/<name>/
├── dynamo/<step>_graph0.*      # Dynamo가 만든 FX 그래프
├── aten/<step>_graph0_fw.*     # AOTAutograd가 core-ATen으로 분해한 그래프
├── report.md / report.json     # 그래프 크기 + op 히스토그램
└── compare_aten.md             # compare_graphs.py 결과
```

그래프 하나당 `readable.py`(shape/dtype + 원본 소스 라인 주석), `code.py`, `nodes.json`(후처리용),
`inputs.json`, `summary.json`, `tabular.txt`가 나온다.
**`tabular.txt`는 `readable.py`와 중복이고 커서(57MB) 저장소에서만 제외했다** — 실행하면 생성된다.

`<step>_graph0`의 `0`은 그 스텝에서 백엔드가 호출된 순번이다. **`graph0` 하나만 있다 = graph break 없이
한 그래프로 잡혔다**는 뜻. `_fw`는 AOTAutograd forward 그래프(추론이라 backward 없음).

---

## 주의

- **transformers 5.x 필요.** `qwen3_5` 모델 타입은 4.57.x에 없다.
- 기본은 랜덤 가중치라 모델 다운로드가 없다. 실제 체크포인트가 필요하면 `--weights real`.
- CPU에서 캡처했다. 그래프 메타의 `device='cpu'`만 다르고 구조는 동일.
