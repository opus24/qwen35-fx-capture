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
| **FLA fast path를 켠 GPU 캡처** (그래프가 완전히 달라진다) | [`docs/fla_gpu.md`](docs/fla_gpu.md) |
| FLA vs torch 커널 비교 리포트 | [`dumps/qwen35_4b_s128_fla_gpu/compare_vs_torch_aten.md`](dumps/qwen35_4b_s128_fla_gpu/compare_vs_torch_aten.md) |
| **FLA decode 그래프** (break 0, Triton 커널 48노드 포함) | [`aten/decode_graph0_fw.readable.py`](dumps/qwen35_4b_s128_fla_gpu/aten/decode_graph0_fw.readable.py) |

한 줄 결론: **가중치 GEMM(`aten.mm` 249개)은 prefill/decode가 완전히 동일하고 M(토큰 수)만 128 vs 1이다.
차이는 전부 토큰 축 처리에서 나온다.** 자세한 건 [`docs/analysis.md`](docs/analysis.md).

FLA를 켜면(GPU 전용) 얘기가 달라진다: **prefill은 graph break 63개로 쪼개지고 delta rule이 그래프
밖으로 나가지만, decode는 break 0으로 Triton 커널까지 그래프 안에 잡힌다.** 갈림길은
`chunk_gated_delta_rule`에만 붙은 `@torch.compiler.disable` 하나다 — [`docs/fla_gpu.md`](docs/fla_gpu.md).

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
| linear-attn 커널 | **torch 참조 구현** — FLA fast path **미적용**. prefill 그래프 크기에 결정적이다 ([`docs/analysis.md`](docs/analysis.md), [`docs/fla_gpu.md`](docs/fla_gpu.md)) |
| 소요 시간 | 모델 빌드 249s / prefill 캡처 148s / decode 캡처 32s |

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

### FLA fast path 버전 (GPU 전용)

FLA는 transformers가 `is_torch_cuda_available()`로 게이팅하므로 **CUDA GPU에서만** 켜지고,
켜지면 그래프가 완전히 달라진다. 전용 스크립트는 `capture_qwen35_fx_fla_gpu.py`다
(`capture_qwen35_fx.py`는 건드리지 않고 직렬화 부분만 import해서 쓴다):

```bash
./.venv-fla/bin/python capture_qwen35_fx_fla_gpu.py --kernels fla --require-fla \
    --seq-len 128 --mode both --out dumps/qwen35_4b_s128_fla_gpu

# 같은 GPU·같은 가중치의 torch 참조 커널 베이스라인
./.venv-fla/bin/python capture_qwen35_fx_fla_gpu.py --kernels torch \
    --seq-len 128 --mode both --out dumps/qwen35_4b_s128_torch_gpu

./.venv-fla/bin/python compare_kernel_paths.py \
    dumps/qwen35_4b_s128_fla_gpu dumps/qwen35_4b_s128_torch_gpu --level aten
```

**python 3.11+ 필요** (3.10에서는 `import fla`가 깨진다) — 이유·환경 구성·결과는 전부
[`docs/fla_gpu.md`](docs/fla_gpu.md).

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
