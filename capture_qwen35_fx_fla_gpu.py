#!/usr/bin/env python3
"""Capture Qwen3.5 prefill/decode FX graphs **on a CUDA GPU**, with the FLA fast path.

Companion to `capture_qwen35_fx.py` (CPU, torch reference kernels). Nothing here
overwrites that script — the graph-serialization plumbing is imported from it and
only the GPU/FLA-specific parts live in this file.

Why a separate script
---------------------
Qwen3.5's `Qwen3_5GatedDeltaNet` binds its linear-attention kernels **once, at
import time** (`modeling_qwen3_5.py:421-424`)::

    self.causal_conv1d_fn        = causal_conv1d_fn                                  # None if not installed
    self.causal_conv1d_update    = causal_conv1d_update    or torch_causal_conv1d_update
    self.chunk_gated_delta_rule  = chunk_gated_delta_rule  or torch_chunk_gated_delta_rule
    self.recurrent_gated_delta_rule = fused_recurrent_gated_delta_rule or torch_recurrent_gated_delta_rule

and the `fla` branch is gated on `is_flash_linear_attention_available()`, which is
`is_torch_cuda_available() and fla>=0.2.2`. So the FLA graph simply cannot be
captured on a CPU box — it needs this script, on a GPU.

`--kernels {fla,torch}` picks which side to capture. `torch` forces the reference
path by patching `is_flash_linear_attention_available` to False *before*
`modeling_qwen3_5` is imported, which gives a same-GPU, same-weights baseline to
diff the FLA graph against (the CPU dump is not a fair baseline — different device,
different machine).

Each run writes, next to the usual `dynamo/` + `aten/` graph dumps:
  * `kernel_path` in report.json/report.md — the *actually bound* implementations,
    so a dump can never be silently mislabeled;
  * `bench` — eager prefill/decode latency measured with CUDA events;
  * `logits.pt` + a param checksum, so the two runs can be compared numerically.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import capture_qwen35_fx as base

DEFAULT_MODEL = base.DEFAULT_MODEL


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture Qwen3.5 prefill/decode FX graphs on GPU with the FLA fast path",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    p.add_argument("--out", default=None, help="output dir (default: ./dumps/<slug>_s<S>_<kernels>_gpu)")
    p.add_argument(
        "--kernels",
        default="fla",
        choices=["fla", "torch"],
        help="'fla' = flash-linear-attention fast path; 'torch' = reference kernels on the same GPU "
        "(forced by disabling is_flash_linear_attention_available before the model module is imported)",
    )
    p.add_argument("--device", default="cuda", help="device to build and run on")
    p.add_argument("--mode", default="both", choices=["prefill", "decode", "both"])

    # shapes
    p.add_argument("--batch", type=int, default=1)
    p.add_argument("--seq-len", type=int, default=128, help="prefill sequence length")
    p.add_argument(
        "--decode-past", type=int, default=None, help="cache length before decode (default: --seq-len)"
    )

    # model construction (mirrors capture_qwen35_fx.py so base.build_module works)
    p.add_argument("--weights", default="random", choices=["random", "real"])
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument("--attn-impl", default="eager", choices=["eager", "sdpa", "flash_attention_2"])
    p.add_argument("--layers", type=int, default=None, help="override num_hidden_layers (smoke tests)")
    p.add_argument("--vocab-size", type=int, default=None, help="override vocab_size (smoke tests)")
    p.add_argument("--no-lm-head", action="store_true")

    # capture knobs
    p.add_argument("--aten-decomp", default="core", choices=["core", "inductor", "none"])
    p.add_argument("--dynamic", default="false", choices=["false", "true", "auto"])
    p.add_argument(
        "--fullgraph", action="store_true", help="fail on graph break (FLA triton kernels may break)"
    )
    p.add_argument(
        "--recompile-limit",
        type=int,
        default=256,
        help="torch._dynamo.config.recompile_limit. The default of 8 is far too low here: every "
        "FLA graph break re-enters Qwen3_5GatedDeltaNet.forward with different guards, so Dynamo "
        "gives up after 8 layers and runs the rest eagerly — silently dropping them from the dump",
    )
    p.add_argument("--to-folder", action="store_true")
    p.add_argument("--no-capture", action="store_true", help="benchmark only, skip torch.compile capture")

    # measurement
    p.add_argument("--no-bench", action="store_true", help="skip the eager latency benchmark")
    p.add_argument("--warmup", type=int, default=3, help="benchmark warmup iterations")
    p.add_argument("--iters", type=int, default=10, help="benchmark timed iterations")
    p.add_argument(
        "--decode-steps", type=int, default=16, help="single-token steps to average per decode timing"
    )

    p.add_argument(
        "--require-fla",
        action="store_true",
        help="abort unless the FLA fast path is actually bound; prevents writing a torch-fallback "
        "dump under an FLA name (no-op with --kernels torch)",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# which kernels are really bound
# --------------------------------------------------------------------------- #
def force_torch_kernels() -> None:
    """Make transformers bind the torch reference kernels even on a CUDA box.

    Must run *before* `modeling_qwen3_5` is first imported: that module does
    `from ...utils.import_utils import is_flash_linear_attention_available` and then
    calls it at module scope, so patching the attribute here is what it will see.
    """
    from transformers.utils import import_utils

    import_utils.is_flash_linear_attention_available = lambda *a, **k: False
    import_utils.is_causal_conv1d_available = lambda *a, **k: False


def _impl_name(fn) -> str:
    """'module.qualname' of whatever callable the model actually bound."""
    if fn is None:
        return "none"
    inner = getattr(fn, "__wrapped__", fn)
    return f"{getattr(inner, '__module__', '?')}.{getattr(inner, '__qualname__', repr(inner))}"


def describe_kernel_path(module, args) -> dict:
    """Which linear-attention kernels this model will actually run and trace.

    Qwen3.5 picks per-symbol at import time (`fla_fn or torch_fn`), so the fast path
    can be *partially* active: fla delta rule + torch conv when causal-conv1d is
    missing. Recorded so a dump can never be silently mislabeled.
    """
    import torch

    status: dict = {
        "requested": args.kernels,
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
    try:
        import transformers

        status["transformers_version"] = transformers.__version__
        from transformers.utils.import_utils import (
            is_causal_conv1d_available,
            is_flash_linear_attention_available,
        )

        status["fla_available"] = bool(is_flash_linear_attention_available())
        status["causal_conv1d_available"] = bool(is_causal_conv1d_available())
    except Exception as exc:  # older transformers / non-qwen3.5 model
        status["availability_error"] = str(exc)[:200]
    try:
        import fla

        status["fla_version"] = getattr(fla, "__version__", "?")
    except Exception as exc:
        status["fla_version"] = f"import failed: {type(exc).__name__}: {str(exc)[:120]}"

    layer = next((m for m in module.modules() if hasattr(m, "chunk_gated_delta_rule")), None)
    if layer is not None:
        status["delta_rule_chunk"] = _impl_name(layer.chunk_gated_delta_rule)
        status["delta_rule_recurrent"] = _impl_name(layer.recurrent_gated_delta_rule)
        status["conv1d_fn"] = _impl_name(layer.causal_conv1d_fn)
        status["conv1d_update"] = _impl_name(layer.causal_conv1d_update)
        # FusedRMSNormGated also comes from fla, so the gated norm swaps with the fast path
        status["gated_rmsnorm"] = f"{type(layer.norm).__module__}.{type(layer.norm).__name__}"
        status["fla_delta_rule_active"] = status["delta_rule_chunk"].startswith("fla")
    return status


# --------------------------------------------------------------------------- #
# inputs on device
# --------------------------------------------------------------------------- #
def make_inputs(args, text_config, step: str, device):
    """(input_ids, position_ids, cache, info) for a prefill or a decode step, on `device`."""
    import torch

    b, s = args.batch, args.seq_len
    past = args.decode_past if args.decode_past is not None else s
    vocab = text_config.vocab_size
    cache = base.make_cache(text_config, b)

    def ids(n):
        return torch.randint(0, vocab, (b, n), dtype=torch.long, device=device)

    def pos(n, start=0):
        return (
            torch.arange(start, start + n, dtype=torch.long, device=device)
            .unsqueeze(0)
            .expand(b, -1)
            .contiguous()
        )

    if step == "prefill":
        return ids(s), pos(s), cache, {"batch": b, "seq_len": s, "past_len": 0}

    # decode: warm the cache with an eager prefill first, then step one token
    return (
        ids(1),
        torch.full((b, 1), past, dtype=torch.long, device=device),
        cache,
        {"batch": b, "seq_len": 1, "past_len": past, "warmup": (ids(past), pos(past))},
    )


# --------------------------------------------------------------------------- #
# measurement
# --------------------------------------------------------------------------- #
def param_fingerprint(module) -> dict:
    """Weight fingerprint the fla and torch runs must agree on before their logits mean anything.

    A plain sum is useless here: Qwen3.5 builds `A_log = log(uniform(0, 16))`, so a random-weight
    model can legitimately carry -inf entries and every checksum collapses to -inf. Sum the finite
    values and count the rest instead. Reduces in fp64 without materializing an fp64 copy.
    """
    import torch

    total, nonfinite, numel = 0.0, 0, 0
    for p in module.parameters():
        d = p.detach()
        finite = torch.isfinite(d)
        nonfinite += int((~finite).sum().item())
        total += float(
            torch.where(finite, d, torch.zeros((), dtype=d.dtype, device=d.device))
            .sum(dtype=torch.float64)
            .item()
        )
        numel += d.numel()
    return {"numel": numel, "sum_finite": round(total, 4), "nonfinite": nonfinite}


def _time_cuda(fn, warmup: int, iters: int, setup=None) -> dict:
    """Median/mean wall time of `fn` in ms, measured with CUDA events.

    `setup` runs before each iteration and is excluded from the timing — decode needs
    a freshly warmed cache per iteration, and that prefill must not land in the number.
    """
    import torch

    for _ in range(warmup):
        if setup is not None:
            setup()
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(iters):
        if setup is not None:
            setup()
            torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))

    times.sort()
    return {
        "iters": iters,
        "median_ms": round(times[len(times) // 2], 3),
        "mean_ms": round(sum(times) / len(times), 3),
        "min_ms": round(times[0], 3),
        "max_ms": round(times[-1], 3),
    }


def benchmark(module, args, text_config, device) -> tuple[dict, "object"]:
    """Eager prefill / decode latency + the logits needed to diff kernels numerically."""
    import torch

    out: dict = {}
    b, s = args.batch, args.seq_len
    past = args.decode_past if args.decode_past is not None else s
    vocab = text_config.vocab_size

    # fixed inputs so the fla and torch runs are numerically comparable
    torch.manual_seed(1234)
    prefill_ids = torch.randint(0, vocab, (b, s), dtype=torch.long, device=device)
    prefill_pos = torch.arange(s, dtype=torch.long, device=device).unsqueeze(0).expand(b, -1).contiguous()
    step_ids = torch.randint(0, vocab, (b, 1), dtype=torch.long, device=device)

    torch.cuda.reset_peak_memory_stats()
    with torch.no_grad():
        print("  benchmarking prefill...", flush=True)
        logits_holder = {}

        def run_prefill():
            cache = base.make_cache(text_config, b)
            logits_holder["out"] = module(prefill_ids, prefill_pos, cache)

        out["prefill"] = {"batch": b, "seq_len": s, **_time_cuda(run_prefill, args.warmup, args.iters)}
        out["prefill"]["tokens_per_s"] = round(b * s / (out["prefill"]["median_ms"] / 1000), 1)

        # logits of the fixed prefill: lets the two kernel paths be compared numerically
        logits = logits_holder["out"].float().cpu()
        out["prefill_logits"] = {
            "shape": list(logits.shape),
            "mean": round(logits.mean().item(), 6),
            "std": round(logits.std().item(), 6),
            "absmax": round(logits.abs().max().item(), 6),
        }

        print("  benchmarking decode...", flush=True)
        # each iteration gets a freshly warmed cache (setup, untimed), then `--decode-steps`
        # single-token steps are timed — so past_len grows the same way every iteration
        cache_holder = {}

        def setup_decode():
            cache_holder["cache"] = base.make_cache(text_config, b)
            module(prefill_ids, prefill_pos, cache_holder["cache"])

        def run_decode():
            cache = cache_holder["cache"]
            for i in range(args.decode_steps):
                pos_i = torch.full((b, 1), past + i, dtype=torch.long, device=device)
                module(step_ids, pos_i, cache)

        stats = _time_cuda(run_decode, max(1, args.warmup - 1), max(3, args.iters // 2), setup=setup_decode)
        per_step = {k: round(v / args.decode_steps, 3) for k, v in stats.items() if k.endswith("_ms")}
        out["decode"] = {
            "batch": b,
            "past_len": past,
            "steps_per_iter": args.decode_steps,
            "prefill_excluded_from_timing": True,
            **stats,
            "per_step": per_step,
        }

    out["peak_memory_gb"] = round(torch.cuda.max_memory_allocated() / 1e9, 3)
    return out, logits


# --------------------------------------------------------------------------- #
# capture
# --------------------------------------------------------------------------- #
def capture_step(module, args, text_config, step: str, out_root: Path, device) -> dict:
    import torch

    print(f"\n=== capturing {step} ({args.kernels} kernels) ===", flush=True)
    input_ids, position_ids, cache, info = make_inputs(args, text_config, step, device)
    warm = info.pop("warmup", None)

    summaries: list[dict] = []
    dynamic = {"false": False, "true": True, "auto": None}[args.dynamic]
    record: dict = {"step": step, **info}

    with torch.no_grad():
        if warm is not None:
            print(f"  eager warmup prefill {tuple(warm[0].shape)} -> filling cache", flush=True)
            module(warm[0], warm[1], cache)
            torch.cuda.synchronize()

        # graph-break accounting: the FLA path calls triton kernels, which dynamo may
        # not be able to trace through — the break count *is* part of the answer here
        try:
            torch._dynamo.reset()
            explanation = torch._dynamo.explain(module)(input_ids, position_ids, cache)
            (out_root / f"{step}.explain.txt").write_text(str(explanation))
            record["graph_count"] = explanation.graph_count
            record["graph_break_count"] = explanation.graph_break_count
            print(f"  graphs={explanation.graph_count} breaks={explanation.graph_break_count}", flush=True)
        except Exception as exc:
            record["explain_error"] = f"{type(exc).__name__}: {str(exc)[:400]}"
            print(f"  explain failed: {record['explain_error']}", flush=True)

        # explain() consumed cache state; rebuild it
        input_ids, position_ids, cache, info2 = make_inputs(args, text_config, step, device)
        warm2 = info2.pop("warmup", None)
        if warm2 is not None:
            module(warm2[0], warm2[1], cache)
            torch.cuda.synchronize()

        torch._dynamo.reset()
        backend = base.make_capture_backend(out_root, step, args, summaries)
        compiled = torch.compile(module, backend=backend, dynamic=dynamic, fullgraph=args.fullgraph)

        t0 = time.time()
        try:
            out = compiled(input_ids, position_ids, cache)
            torch.cuda.synchronize()
            record["output_shape"] = list(out.shape)
        except Exception as exc:
            record["capture_error"] = f"{type(exc).__name__}: {str(exc)[:600]}"
            print(f"  compile FAILED: {record['capture_error']}", flush=True)
        elapsed = time.time() - t0

    record["compile_and_run_seconds"] = round(elapsed, 2)
    record["graphs"] = summaries
    print(f"  compiled+ran in {elapsed:.1f}s, captured {len(summaries)} graph(s) -> {out_root}", flush=True)
    return record


# --------------------------------------------------------------------------- #
# report
# --------------------------------------------------------------------------- #
def write_report(
    out_root: Path,
    args,
    results: list[dict],
    text_config,
    kernel_path: dict,
    bench: dict | None,
    param_checksum: dict | None,
) -> None:
    report = {
        "model": args.model,
        "kernels": args.kernels,
        "device": args.device,
        "kernel_path": kernel_path,
        "weights": args.weights,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "num_hidden_layers": text_config.num_hidden_layers,
        "layer_types": list(getattr(text_config, "layer_types", []) or []),
        "vocab_size": text_config.vocab_size,
        "aten_decomp": args.aten_decomp,
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
        "recompile_limit": args.recompile_limit,
        "param_checksum": param_checksum,
        "bench": bench,
        "steps": results,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))

    lines = [f"# FX capture report — {args.model} ({args.kernels} kernels, {args.device})", ""]
    lines.append(
        f"weights={args.weights} dtype={args.dtype} attn={args.attn_impl} "
        f"layers={text_config.num_hidden_layers} decomp={args.aten_decomp} dynamic={args.dynamic}"
    )
    lines += ["", "## linear-attention kernel path", "", "| | |", "| --- | --- |"]
    lines += [f"| {k} | `{v}` |" for k, v in kernel_path.items()]

    if bench:
        lines += [
            "",
            "## eager latency (CUDA events)",
            "",
            "| step | median ms | mean ms | min ms | notes |",
            "| --- | --- | --- | --- | --- |",
        ]
        pf = bench["prefill"]
        lines.append(
            f"| prefill (b{pf['batch']} s{pf['seq_len']}) | {pf['median_ms']} | {pf['mean_ms']} | "
            f"{pf['min_ms']} | {pf['tokens_per_s']} tok/s |"
        )
        dc = bench["decode"]
        lines.append(
            f"| decode ×{dc['steps_per_iter']} (past={dc['past_len']}) | {dc['median_ms']} | {dc['mean_ms']} | "
            f"{dc['min_ms']} | **{dc['per_step']['median_ms']} ms/token** |"
        )
        lines += ["", f"peak memory: {bench['peak_memory_gb']} GB", ""]
        pl = bench["prefill_logits"]
        lines.append(
            f"prefill logits: shape={pl['shape']} mean={pl['mean']} std={pl['std']} absmax={pl['absmax']}"
        )
        if param_checksum is not None:
            lines.append(
                f"params: numel={param_checksum['numel']} sum_finite={param_checksum['sum_finite']} "
                f"nonfinite={param_checksum['nonfinite']}"
            )

    for res in results:
        lines += [
            "",
            f"## {res['step']} (batch={res['batch']} seq_len={res['seq_len']} past={res['past_len']})",
            "",
        ]
        if "graph_count" in res:
            lines.append(
                f"dynamo: graph_count={res['graph_count']} graph_break_count={res['graph_break_count']}"
            )
        if "capture_error" in res:
            lines.append(f"\n**capture error:** `{res['capture_error']}`")
        if res["graphs"]:
            lines += [
                "",
                "| level | graph | nodes | placeholders | call_function |",
                "| --- | --- | --- | --- | --- |",
            ]
            for g in res["graphs"]:
                lines.append(
                    f"| {g['level']} | {g['name']} | {g['num_nodes']} | "
                    f"{g['num_placeholders']} | {g['num_call_function']} |"
                )
            for g in res["graphs"]:
                if g["level"] != "aten":
                    continue
                top = list(g["op_counts"].items())[:25]
                lines += ["", f"### top ops — {g['name']}", "", "| op | count |", "| --- | --- |"]
                lines += [f"| {op} | {n} |" for op, n in top]
    (out_root / "report.md").write_text("\n".join(lines) + "\n")


# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    slug = args.model.rstrip("/").split("/")[-1].replace(".", "_")
    out_root = Path(
        args.out or Path(__file__).parent / "dumps" / f"{slug}_s{args.seq_len}_{args.kernels}_gpu"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit(
            "--device cuda but torch.cuda.is_available() is False. The FLA fast path is gated on "
            "is_torch_cuda_available(), so this script needs a real GPU; use capture_qwen35_fx.py on CPU."
        )

    # must happen before transformers imports modeling_qwen3_5 (kernels bind at import time)
    if args.kernels == "torch":
        force_torch_kernels()
        print("forcing torch reference kernels (fla/causal-conv1d availability patched to False)", flush=True)

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    torch._dynamo.config.recompile_limit = args.recompile_limit
    device = torch.device(args.device)

    print(f"torch {torch.__version__} | device {device} | out: {out_root}", flush=True)
    t0 = time.time()
    with device:  # build the 4B params straight on the GPU instead of CPU-then-copy
        module, text_config = base.build_module(args)
    module = module.to(device)
    n_params = sum(p.numel() for p in module.parameters())
    print(
        f"module built in {time.time() - t0:.1f}s | {n_params / 1e9:.2f}B params | "
        f"{text_config.num_hidden_layers} layers | vocab {text_config.vocab_size}",
        flush=True,
    )

    kernel_path = describe_kernel_path(module, args)
    print(
        "kernel path: delta_rule={} | conv={} | norm={} | cuda={} fla={} causal_conv1d={}".format(
            kernel_path.get("delta_rule_chunk", "?"),
            kernel_path.get("conv1d_fn", "?"),
            kernel_path.get("gated_rmsnorm", "?"),
            kernel_path.get("cuda_available"),
            kernel_path.get("fla_available"),
            kernel_path.get("causal_conv1d_available"),
        ),
        flush=True,
    )
    if args.require_fla and args.kernels == "fla" and not kernel_path.get("fla_delta_rule_active"):
        raise SystemExit(
            "--require-fla: the flash-linear-attention fast path is NOT active "
            f"(delta rule = {kernel_path.get('delta_rule_chunk')}). transformers gates it on "
            "is_torch_cuda_available() and fla>=0.2.2. Aborting instead of writing a torch-fallback "
            "dump under an FLA name."
        )

    # fingerprint lets the fla and torch runs prove they hold identical weights before
    # their logits are compared
    param_checksum = param_fingerprint(module)
    print(f"params: {param_checksum}", flush=True)

    bench = None
    if not args.no_bench:
        print("\n=== benchmark ===", flush=True)
        bench, logits = benchmark(module, args, text_config, device)
        torch.save(logits, out_root / "prefill_logits.pt")
        print(
            f"  prefill {bench['prefill']['median_ms']} ms | decode {bench['decode']['per_step']['median_ms']} ms/step",
            flush=True,
        )

    results = []
    if not args.no_capture:
        for step in ["prefill", "decode"] if args.mode == "both" else [args.mode]:
            torch._dynamo.reset()
            results.append(capture_step(module, args, text_config, step, out_root, device))

    write_report(out_root, args, results, text_config, kernel_path, bench, param_checksum)
    print(f"\nreport: {out_root / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
