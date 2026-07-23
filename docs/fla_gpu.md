# FLA fast path를 켜고 GPU에서 캡처하면 그래프가 어떻게 되나

`capture_qwen35_fx.py`(CPU / torch 참조 커널)로 뽑은 [`dumps/qwen35_4b_s128/`](../dumps/qwen35_4b_s128/)는
**FLA(flash-linear-attention) fast path가 꺼진** 버전이다. 이 문서는 실제 GPU에서 fast path를 켜고
`capture_qwen35_fx_fla_gpu.py`로 다시 뽑은 결과다.

> 한 줄 결론: **FLA를 켜면 prefill 그래프는 42,023노드 한 덩어리에서 ~1.7천 노드 31조각으로 바뀐다.**
> 노드 수가 24배 줄어드는 게 아니라, **delta rule 본체가 그래프 밖으로 빠지면서 그래프가 산산조각 난다.**
> decode는 반대로 **break 0으로 온전히 잡히고, FLA Triton 커널이 그래프 안에 노드로 들어온다.**

---

## 0. 왜 CPU에서는 이걸 볼 수 없나

Qwen3.5의 `Qwen3_5GatedDeltaNet`은 커널을 **import 시점에 심볼별로** 고른다
(`modeling_qwen3_5.py:421-424`):

```python
self.causal_conv1d_fn           = causal_conv1d_fn                                    # 없으면 None
self.causal_conv1d_update       = causal_conv1d_update       or torch_causal_conv1d_update
self.chunk_gated_delta_rule     = chunk_gated_delta_rule     or torch_chunk_gated_delta_rule
self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule
```

그리고 `chunk_gated_delta_rule` 쪽 import가 `is_flash_linear_attention_available()`로 게이팅되는데,
이게 **`is_torch_cuda_available()`을 포함**한다:

```python
# transformers/utils/import_utils.py
def is_flash_linear_attention_available():
    is_available, fla_version = _is_package_available("fla", return_version=True)
    return is_torch_cuda_available() and is_available and version.parse(fla_version) >= version.parse("0.2.2")
```

즉 **CUDA GPU가 없으면 fla가 설치돼 있어도 fast path는 절대 안 켜진다.** CPU 박스에서 이 그래프를
뽑는 건 불가능하고, 그래서 이 스크립트가 따로 있다.

덜 알려진 부수 효과: **gated RMSNorm도 같이 바뀐다.** fast path가 켜지면 `self.norm`이
`Qwen3_5RMSNormGated`(torch) 대신 `fla.modules.fused_norm_gate.FusedRMSNormGated`(Triton)가 된다.
delta rule만 바뀌는 게 아니다.

---

## 1. 환경 — 왜 별도 venv(`.venv-fla`)가 필요했나

기존 `.venv`(python 3.10)에서는 **`import fla` 자체가 실패**한다:

```
File "triton/runtime/jit.py", line 469, in __init__
    src = src[re.search(r"^def\s+\w+\s*\(", src, re.MULTILINE).start():]
AttributeError: 'NoneType' object has no attribute 'start'
```

원인은 triton이 아니라 **CPython 3.10의 `inspect` 버그**다. `inspect.getsource()`는 데코레이터부터
소스를 떠오는데, 3.10의 `BlockFinder`는 **데코레이터 인자 안에 `lambda`가 있으면 거기서 블록이 끝났다고
판단**한다. fla의 커널은 전부 이런 모양이라:

```python
@triton.heuristics({'NV': lambda args: triton.cdiv(args['V'], args['BV']), ...})   # ← lambda
@triton.jit
def parallel_simple_gla_fwd_kernel(...):
```

`getsource()`가 **데코레이터 6줄만** 돌려주고 `def` 줄이 없어서, triton의 `re.search(r"^def...")`가
`None`을 반환한다. python 3.11에서 고쳐진 문제라 **fla는 3.11+가 사실상 필수**다.

> **주의: 이 때문에 기존 `.venv`는 GPU 머신에서 아예 못 쓴다.** CPU 박스에서는
> `is_flash_linear_attention_available()`이 False였으니 fla를 import할 일이 없었는데, GPU가 생기는
> 순간 True가 되면서 `modeling_qwen3_5`가 `from fla.modules import ...`를 타고 위 에러로 죽는다.
> 증상은 엉뚱하게 `ModuleNotFoundError: Could not import module 'Qwen3_5ForCausalLM'`로 나온다
> (transformers의 lazy import가 원인을 삼킨다).

### 패키징 함정 — `fla` != `flash-linear-attention`

`pip install flash-linear-attention`만으로는 **`fla.ops`가 안 들어온다.** 이 휠에는
`fla/layers`, `fla/models`밖에 없고, 정작 커널이 있는 `fla.ops` / `fla.modules`는 **별도 배포판
`fla-core`**에 있다 (`flash-linear-attention`의 의존성). `--no-deps`로 깔면 `import fla`는 성공하는데
`fla.ops`가 없는 상태가 되고, `fla/__init__.py`가 `_import_optional_public_module()`로 실패를
**조용히 삼켜서** 한참 헤맬 수 있다.

```bash
uv venv --python 3.12 .venv-fla
uv pip install --python .venv-fla/bin/python --index-url https://download.pytorch.org/whl/cu128 torch==2.10.0
uv pip install --python .venv-fla/bin/python transformers==5.14.1 accelerate einops
uv pip install --python .venv-fla/bin/python flash-linear-attention   # fla-core를 같이 끌고 온다
uv pip install --python .venv-fla/bin/python --no-build-isolation causal-conv1d   # 선택, nvcc 필요
```

---

## 2. 재현

```bash
# FLA fast path
./.venv-fla/bin/python capture_qwen35_fx_fla_gpu.py --kernels fla --require-fla \
    --seq-len 128 --mode both --out dumps/qwen35_4b_s128_fla_gpu

# 같은 GPU / 같은 가중치의 torch 참조 커널 베이스라인
./.venv-fla/bin/python capture_qwen35_fx_fla_gpu.py --kernels torch \
    --seq-len 128 --mode both --out dumps/qwen35_4b_s128_torch_gpu

# 둘 비교
./.venv-fla/bin/python compare_kernel_paths.py \
    dumps/qwen35_4b_s128_fla_gpu dumps/qwen35_4b_s128_torch_gpu --level aten
```

`--require-fla`는 fast path가 실제로 안 붙었으면 **덤프를 쓰지 않고 즉시 중단**한다 (torch 폴백
결과가 FLA 이름으로 저장되는 사고 방지). `--kernels torch`는 `modeling_qwen3_5` import **전에**
`is_flash_linear_attention_available`을 False로 패치해서 참조 구현을 강제한다 — CPU 덤프는 기기가
달라서 레이턴시 비교의 베이스라인이 될 수 없기 때문에 필요하다.

측정 조건: NVIDIA A40 46GB / torch 2.10.0+cu128 / python 3.12.11 / bf16 / batch 1 / 랜덤 가중치 /
`attn_impl=eager` / static shape. 레이턴시는 CUDA event median (warmup 5, iters 20, decode 32스텝).
두 런의 가중치가 같다는 건 `param_checksum`으로 확인한다 (아래 표).

---

## 3. 결과 — 레이턴시

| | FLA fast path | torch 참조 커널 | torch / FLA |
|---|---|---|---|
| prefill (b1 s128) | **98.7 ms** | 195.4 ms | **1.98×** |
| decode (past=128) | **43.09 ms/token** | 46.80 ms/token | **1.09×** |
| peak memory | 8.594 GB | 8.611 GB | |
| 캡처(컴파일+실행) 시간 | prefill 23.0s / decode 20.4s | prefill 81.1s / decode 16.3s | |

두 런의 가중치는 동일하다 — `numel=4,205,751,296 sum_finite=4043.9586 nonfinite=5` (양쪽 일치).
prefill logits도 사실상 같다: FLA `mean=-0.000777 std=1.010787 absmax=5.0625` vs
torch `mean=-0.000897 std=1.010811 absmax=5.0625`.

**FLA 이득은 거의 전부 prefill에서 나온다** (2배). decode는 9%뿐인데, 단일 토큰 decode는
커널 자체보다 **커널 런치 바운드**라서 그렇다 — 실제로 CPU가 다른 작업으로 바쁠 때 decode 수치가
50%까지 부풀었다. 위 숫자는 유휴 머신 기준이다.

---

## 4. 결과 — prefill 그래프 (핵심)

| | FLA fast path | torch 참조 커널 |
|---|---|---|
| dynamo graph count | **64** | **1** |
| graph break | **63** | **0** |
| ATen 노드 합계 | 2,740 | 42,023 |
| 가장 큰 단일 그래프 | 163 | 42,023 |
| `aten.mm` | 180 | 249 |
| `aten.bmm` | 17 | 329 |
| `aten.convolution` | 0 | 24 |

**노드가 15배 줄어든 게 아니라, 그래프가 64조각으로 부서진 것이다.** delta rule 본체(torch 경로에서
prefill 그래프의 88.6%를 차지하던 언롤)가 **그래프 밖으로** 나가면서 ATen에서 사라졌다.

### 왜 끊기나 — 원인 두 가지

1. **`chunk_gated_delta_rule`이 `@torch.compiler.disable`이다.**
   `fla/ops/gated_delta_rule/chunk.py:379`에 직접 붙어 있다 (`_torchdynamo_disable == True`).
   Dynamo가 아예 진입을 포기하므로 무조건 break다.
2. **causal-conv1d 커스텀 op이 non-contiguous `out=`으로 호출된다.**
   `Attempted to call op with non-contiguous 'out=' tensor` — `DaoAILab::_causal_conv1d_fwd_cpp`.
   즉 **causal-conv1d를 설치하면 break가 오히려 늘어난다** (안 깔면 `F.silu(self.conv1d(...))`로
   평범한 `aten.convolution`이 되어 그래프 안에 남는다).

### 64개 그래프의 정체 — 10종류뿐이다

break가 생기면 Dynamo는 상위 프레임에 인라인하지 못하고 **모듈 프레임을 따로 컴파일**하며,
같은 code object + 같은 guard면 **컴파일 캐시를 재사용**한다. 그래서 64개는 순차적인 64조각이 아니라
10종류의 반복이다:

| 개수 | ATen 노드 | `mm` | 정체 |
|---|---|---|---|
| ×24 | 30 | 4 | `Qwen3_5GatedDeltaNet` 앞부분 — `in_proj_{qkv,z,b,a}` |
| ×24 | 22 | 1 | delta rule break 이후 꼬리 — `out_proj` |
| ×8 | 163 | 7 | `full_attention` 디코더 레이어 (break 없어 통째로 인라인) |
| ×1 | 36 | 3 | MLP — **한 번만 컴파일되고 24개 linear 레이어가 캐시 재사용** |
| ×1 | 8 | 1 | `lm_head` |
| 나머지 | 3~65 | 0 | embedding / mask / 최종 norm 조각 |

`aten.mm` 180 = 24×4 + 24×1 + 8×7 + 3 + 1. torch 경로의 249와 다른 건 **레이어가 빠져서가 아니라
MLP가 캐시 재사용돼 한 번만 덤프되기 때문**이다 (32개 레이어 전부 커버돼 있다).

### 함정: `recompile_limit` 기본값 8은 여기서 너무 작다

break 하나하나가 `Qwen3_5GatedDeltaNet.forward`를 다른 guard로 재진입시켜서, 기본값 8이면
**8개 레이어쯤에서 Dynamo가 포기하고 나머지를 eager로 돌린다** — 그러면 덤프에 그 레이어들이
아예 안 들어오는데 에러도 안 난다. 실제로 기본값에서는 `31 graphs / 30 breaks`, `mm=93`만 나왔다.
`--recompile-limit`(기본 256)로 올리면 `64 graphs / 63 breaks`, `mm=180`으로 전 레이어가 잡힌다.

---

## 5. 결과 — decode 그래프 (prefill과 정반대)

| | FLA fast path | torch 참조 커널 |
|---|---|---|
| dynamo graph count | 1 | 1 |
| graph break | **0** | 0 |
| ATen 노드 | 4,439 | 5,783 |
| `higher_order.triton_kernel_wrapper_functional` | **48** | 0 |
| `higher_order.auto_functionalized_v2` | 24 | 0 |
| `aten.convolution` | 0 | 24 |

**decode는 FLA를 켜도 break 없이 한 그래프로 잡힌다.** 이유는 단순하다 —
`fused_recurrent_gated_delta_rule`에는 `@torch.compiler.disable`이 **없다**
(`_torchdynamo_disable`가 None). 그래서 Dynamo가 Triton 커널을 그래프 안으로 끌고 들어와
`triton_kernel_wrapper_functional` 노드 **48개**(= linear 레이어 24개 × 커널 2개)로 남긴다.
`causal_conv1d_update`는 커스텀 op이라 `auto_functionalized_v2` 24개로 들어온다.

즉 **같은 FLA인데 prefill은 그래프 밖으로 나가고, decode는 그래프 안에 들어온다.** 갈림길은
데코레이터 하나다.

---

## 6. ATen 백엔드(legato) 입장에서의 함의

- **torch 참조 커널 경로**: prefill/decode 모두 순수 ATen 한 그래프. 지금 `dumps/qwen35_4b_s128/`이
  그거고, 백엔드가 받기에 가장 다루기 쉽다 (대신 prefill 42,023노드).
- **FLA 경로 prefill**: 하나의 ATen 그래프로는 **받을 수 없다**. 63개 break로 쪼개지고 delta rule
  본체는 그래프에 아예 안 보인다. 백엔드가 이 모델의 prefill을 통으로 컴파일하려면
  **FLA를 끄거나**, `chunk_gated_delta_rule`을 custom op으로 등록해 `@torch.compiler.disable`을
  걷어내야 한다.
- **FLA 경로 decode**: 그래프는 온전하지만 안에 Triton HOP 노드가 48개 박혀 있다. 백엔드는 이걸
  자기 커널로 **매핑하거나 거절**해야 한다 — ATen만 안다고 가정하면 안 된다.

---

## 7. 주의

- **랜덤 가중치 부작용**: `A_log = log(uniform(0, 16))`이라 uniform이 정확히 0을 뽑으면 `-inf`가 된다.
  이 런에서는 4.2B 파라미터 중 5개가 non-finite였다. 그래프 구조에는 영향이 없고 logits도 유한하게
  나오지만(`std=1.01`), 그래서 가중치 지문은 단순 합이 아니라 **유한값 합 + non-finite 개수**로 받는다.
  실제 체크포인트가 필요하면 `--weights real`.
- `dumps/qwen35_4b_s128_torch_gpu/`의 그래프 파일은 **CPU 덤프와 노드 단위로 완전히 동일**해서
  (prefill 19,897/42,023 · decode 4,561/5,783) 저장소에서는 제외했다. 남긴 건 `report.md`(레이턴시)다.
- `causal-conv1d`는 FLA가 아니라 별도 프로젝트(Dao-AILab)이고 nvcc 빌드가 필요하다. 없어도
  delta rule은 FLA로 바뀐다 — 심볼별 `or` 폴백이라 부분 활성화가 가능하다.
