#!/usr/bin/env python3
"""Capture prefill / decode FX graphs of Qwen/Qwen3.5-4B with torch.compile.

Three dump mechanisms, all driven from this one script:

1. custom backend  (--backend dump, default)
     torch.compile(module, backend=<capture backend>) hands us the Dynamo FX graph,
     and aot_autograd(fw_compiler=...) hands us the ATen-level forward graph
     (the same seam a custom compiler backend would plug into). Both are written
     to disk as readable python + node tables + JSON.
2. torch logs      (--torch-logs)
     sets TORCH_LOGS / TORCH_LOGS_OUT before torch is imported, i.e. the
     `TORCH_LOGS="graph_code,aot_graphs" TORCH_LOGS_OUT=file` env knob.
3. inductor trace  (--backend inductor --inductor-trace)
     sets TORCH_COMPILE_DEBUG=1 + torch._inductor.config.trace, which dumps
     fx_graph_readable.py / fx_graph_transformed.py / output_code.py under
     <out>/torch_compile_debug/.

Prefill and decode are captured as *separate* graphs: Qwen3.5 is a hybrid model
whose GatedDeltaNet layers branch in python on `use_precomputed_states and
seq_len == 1`, so the single-token decode step traces to a different graph
(recurrent_gated_delta_rule + causal_conv1d_update) than prefill
(chunk_gated_delta_rule + conv1d).
"""

from __future__ import annotations

import argparse
import contextlib
import io
import json
import os
import time
from pathlib import Path

DEFAULT_MODEL = "Qwen/Qwen3.5-4B"


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Capture prefill/decode FX graphs of Qwen3.5 via torch.compile",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", default=DEFAULT_MODEL, help="HF model id or local path")
    p.add_argument(
        "--out",
        default=None,
        help="output dir (default: ./dumps/<model-slug>_<mode>_b<B>s<S>)",
    )
    p.add_argument(
        "--mode",
        default="both",
        choices=["prefill", "decode", "both"],
        help="which step(s) to capture",
    )

    # shapes
    p.add_argument("--batch", type=int, default=1, help="batch size")
    p.add_argument("--seq-len", type=int, default=128, help="prefill sequence length")
    p.add_argument(
        "--decode-past",
        type=int,
        default=None,
        help="cache length before the decode step (default: --seq-len)",
    )

    # model construction
    p.add_argument(
        "--weights",
        default="random",
        choices=["random", "real"],
        help="'random' builds from config.json only (no 8GB download); "
        "'real' downloads/loads the checkpoint (graph shape is identical)",
    )
    p.add_argument("--dtype", default="bfloat16", choices=["bfloat16", "float16", "float32"])
    p.add_argument(
        "--attn-impl",
        default="eager",
        choices=["eager", "sdpa", "flash_attention_2"],
        help="eager keeps softmax/matmul visible in the ATen graph; sdpa keeps one fused op",
    )
    p.add_argument(
        "--layers",
        type=int,
        default=None,
        help="override num_hidden_layers (smoke tests; keeps the layer_types prefix)",
    )
    p.add_argument(
        "--vocab-size",
        type=int,
        default=None,
        help="override vocab_size (smoke tests; shrinks embedding/lm_head)",
    )
    p.add_argument(
        "--no-lm-head",
        action="store_true",
        help="capture the text decoder only (no lm_head matmul)",
    )

    # capture knobs
    p.add_argument(
        "--backend",
        default="dump",
        choices=["dump", "inductor", "eager"],
        help="'dump' = custom capture backend; 'inductor' = real inductor (use with --inductor-trace)",
    )
    p.add_argument(
        "--aten-decomp",
        default="core",
        choices=["core", "inductor", "none"],
        help="decomposition table handed to AOTAutograd for the ATen-level graph",
    )
    p.add_argument(
        "--dynamic",
        default="false",
        choices=["false", "true", "auto"],
        help="torch.compile dynamic= (false pins static shapes, recommended for kernel work)",
    )
    p.add_argument("--fullgraph", action="store_true", help="fail on graph break instead of splitting")
    p.add_argument(
        "--explain",
        action="store_true",
        help="also run torch._dynamo.explain() and write a graph-break report",
    )
    p.add_argument(
        "--to-folder",
        action="store_true",
        help="also emit gm.to_folder() re-loadable modules (large: writes weights)",
    )

    # env-level dumps (must be set before `import torch`)
    p.add_argument(
        "--torch-logs",
        nargs="?",
        const="graph_code,aot_graphs,graph_breaks,recompiles",
        default=None,
        help="set TORCH_LOGS=<value> and tee it to <out>/torch_logs.txt",
    )
    p.add_argument(
        "--inductor-trace",
        action="store_true",
        help="set TORCH_COMPILE_DEBUG=1 -> <out>/torch_compile_debug/ (needs --backend inductor)",
    )
    return p.parse_args(argv)


# --------------------------------------------------------------------------- #
# FX graph serialization
# --------------------------------------------------------------------------- #
def _tensor_meta(val) -> dict | None:
    shape = getattr(val, "shape", None)
    if shape is None:
        return None
    return {
        "shape": [str(d) for d in shape],
        "dtype": str(getattr(val, "dtype", "")),
        "device": str(getattr(val, "device", "")),
    }


def _node_meta(node) -> dict | None:
    for key in ("val", "example_value", "tensor_meta"):
        if key in node.meta:
            val = node.meta[key]
            if isinstance(val, (tuple, list)):
                metas = [_tensor_meta(v) for v in val]
                return {"tuple": [m for m in metas if m is not None]} if any(metas) else None
            meta = _tensor_meta(val)
            if meta is not None:
                return meta
    return None


def _target_str(node) -> str:
    target = node.target
    if node.op in ("placeholder", "output"):
        return node.op
    if hasattr(target, "__module__") and hasattr(target, "__name__"):
        return f"{target.__module__}.{target.__name__}"
    return str(target)


def save_graph_module(gm, example_inputs, out_dir: Path, name: str, *, to_folder: bool = False) -> dict:
    """Write every useful view of an FX GraphModule; return a summary dict."""
    out_dir.mkdir(parents=True, exist_ok=True)

    with contextlib.suppress(Exception):
        (out_dir / f"{name}.readable.py").write_text(gm.print_readable(print_output=False))
    (out_dir / f"{name}.code.py").write_text(gm.code)

    buf = io.StringIO()
    try:  # print_tabular needs `tabulate`; plain graph print is the fallback
        with contextlib.redirect_stdout(buf):
            gm.graph.print_tabular()
        table = buf.getvalue()
    except Exception:
        table = str(gm.graph)
    (out_dir / f"{name}.tabular.txt").write_text(table)

    nodes = []
    op_counts: dict[str, int] = {}
    for node in gm.graph.nodes:
        target = _target_str(node)
        nodes.append(
            {
                "name": node.name,
                "op": node.op,
                "target": target,
                "args": [str(a) for a in node.args],
                "kwargs": {k: str(v) for k, v in node.kwargs.items()},
                "meta": _node_meta(node),
            }
        )
        if node.op in ("call_function", "call_method", "call_module"):
            op_counts[target] = op_counts.get(target, 0) + 1
    (out_dir / f"{name}.nodes.json").write_text(json.dumps(nodes, indent=2))

    inputs = []
    for i, inp in enumerate(example_inputs or []):
        meta = _tensor_meta(inp)
        inputs.append({"index": i, **(meta or {"repr": str(inp)[:120]})})
    (out_dir / f"{name}.inputs.json").write_text(json.dumps(inputs, indent=2))

    if to_folder:
        with contextlib.suppress(Exception):
            gm.to_folder(out_dir / f"{name}.module", name.replace(".", "_"))

    summary = {
        "name": name,
        "dir": str(out_dir),
        "num_nodes": len(nodes),
        "num_placeholders": sum(1 for n in nodes if n["op"] == "placeholder"),
        "num_call_function": sum(1 for n in nodes if n["op"] == "call_function"),
        "op_counts": dict(sorted(op_counts.items(), key=lambda kv: -kv[1])),
    }
    (out_dir / f"{name}.summary.json").write_text(json.dumps(summary, indent=2))
    return summary


def make_capture_backend(out_root: Path, tag: str, args, summaries: list[dict]):
    """torch.compile backend: dump the Dynamo graph, then the AOT/ATen graph."""
    import torch
    from torch._dynamo.backends.common import aot_autograd
    from torch._functorch.aot_autograd import make_boxed_func

    if args.aten_decomp == "core":
        from torch._decomp import core_aten_decompositions

        decompositions = core_aten_decompositions()
    elif args.aten_decomp == "inductor":
        from torch._inductor.decomposition import select_decomp_table

        decompositions = select_decomp_table()
    else:
        decompositions = None

    counter = {"n": 0}

    def backend(gm: "torch.fx.GraphModule", example_inputs):
        idx = counter["n"]
        counter["n"] += 1
        gname = f"{tag}_graph{idx}"
        summaries.append(
            {
                "level": "dynamo",
                **save_graph_module(gm, example_inputs, out_root / "dynamo", gname, to_folder=args.to_folder),
            }
        )

        def fw_compiler(aot_gm, aot_inputs):
            summaries.append(
                {
                    "level": "aten",
                    **save_graph_module(aot_gm, aot_inputs, out_root / "aten", f"{gname}_fw"),
                }
            )
            return make_boxed_func(aot_gm.forward)

        return aot_autograd(fw_compiler=fw_compiler, decompositions=decompositions)(gm, example_inputs)

    return backend


# --------------------------------------------------------------------------- #
# model construction
# --------------------------------------------------------------------------- #
def build_module(args):
    """Return (module_to_compile, text_config). module.forward(input_ids, position_ids, cache)."""
    import torch
    from torch import nn
    from transformers import AutoConfig

    dtype = getattr(torch, args.dtype)
    config = AutoConfig.from_pretrained(args.model)
    text_config = getattr(config, "text_config", config)

    if args.layers is not None:
        text_config.num_hidden_layers = args.layers
        types = list(getattr(text_config, "layer_types", None) or [])
        if types:
            truncated = types[: args.layers]
            # the cache/mask code needs at least one full_attention layer to answer
            # get_seq_length(), so keep one even when the prefix is all linear_attention
            if "full_attention" in types and "full_attention" not in truncated:
                truncated[-1] = "full_attention"
            text_config.layer_types = truncated
    if args.vocab_size is not None:
        text_config.vocab_size = args.vocab_size
        config.vocab_size = args.vocab_size
    text_config._attn_implementation = args.attn_impl
    config._attn_implementation = args.attn_impl

    if args.weights == "real":
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_pretrained(
            args.model, dtype=dtype, attn_implementation=args.attn_impl
        )
    else:
        from transformers import AutoModelForCausalLM

        model = AutoModelForCausalLM.from_config(config, attn_implementation=args.attn_impl)
        model = model.to(dtype)

    model.eval()

    inner = getattr(model, "model", model)
    text_model = getattr(inner, "language_model", inner)
    lm_head = None if args.no_lm_head else getattr(model, "lm_head", None)

    class TextDecoderForCapture(nn.Module):
        """The text-only prefill/decode step: embed -> 32 hybrid layers -> norm -> lm_head."""

        def __init__(self):
            super().__init__()
            self.text_model = text_model
            self.lm_head = lm_head

        def forward(self, input_ids, position_ids, past_key_values):
            out = self.text_model(
                input_ids=input_ids,
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=True,
            )
            hidden = out.last_hidden_state
            if self.lm_head is not None:
                # only the last position is needed to sample the next token
                hidden = self.lm_head(hidden[:, -1:, :])
            return hidden

    return TextDecoderForCapture().eval(), text_config


def make_cache(text_config, batch: int):
    from transformers import DynamicCache

    return DynamicCache(config=text_config)


def make_inputs(args, text_config, step: str):
    """(input_ids, position_ids, cache) for a prefill or a decode step."""
    import torch

    b, s = args.batch, args.seq_len
    past = args.decode_past if args.decode_past is not None else s
    vocab = text_config.vocab_size
    cache = make_cache(text_config, b)

    if step == "prefill":
        input_ids = torch.randint(0, vocab, (b, s), dtype=torch.long)
        position_ids = torch.arange(s, dtype=torch.long).unsqueeze(0).expand(b, -1).contiguous()
        return input_ids, position_ids, cache, {"batch": b, "seq_len": s, "past_len": 0}

    # decode: warm the cache with an eager prefill first, then step one token
    warm_ids = torch.randint(0, vocab, (b, past), dtype=torch.long)
    warm_pos = torch.arange(past, dtype=torch.long).unsqueeze(0).expand(b, -1).contiguous()
    input_ids = torch.randint(0, vocab, (b, 1), dtype=torch.long)
    position_ids = torch.full((b, 1), past, dtype=torch.long)
    return input_ids, position_ids, cache, {
        "batch": b,
        "seq_len": 1,
        "past_len": past,
        "warmup": (warm_ids, warm_pos),
    }


# --------------------------------------------------------------------------- #
# capture driver
# --------------------------------------------------------------------------- #
def capture_step(module, args, text_config, step: str, out_root: Path) -> dict:
    import torch

    print(f"\n=== capturing {step} ===", flush=True)
    input_ids, position_ids, cache, info = make_inputs(args, text_config, step)
    warmup = info.pop("warmup", None)

    summaries: list[dict] = []
    dynamic = {"false": False, "true": True, "auto": None}[args.dynamic]

    with torch.no_grad():
        if warmup is not None:
            print(f"  eager warmup prefill: {tuple(warmup[0].shape)} -> filling cache", flush=True)
            t0 = time.time()
            module(warmup[0], warmup[1], cache)
            print(f"  warmup done in {time.time() - t0:.1f}s", flush=True)

        if args.explain:
            explanation = torch._dynamo.explain(module)(input_ids, position_ids, cache)
            (out_root / f"{step}.explain.txt").write_text(str(explanation))
            print(f"  graphs={explanation.graph_count} breaks={explanation.graph_break_count}", flush=True)
            torch._dynamo.reset()
            # explain() consumed the cache state; rebuild it
            input_ids, position_ids, cache, info2 = make_inputs(args, text_config, step)
            warm = info2.pop("warmup", None)
            if warm is not None:
                module(warm[0], warm[1], cache)

        if args.backend == "dump":
            backend = make_capture_backend(out_root, step, args, summaries)
        else:
            backend = args.backend

        torch._dynamo.reset()
        compiled = torch.compile(module, backend=backend, dynamic=dynamic, fullgraph=args.fullgraph)

        t0 = time.time()
        out = compiled(input_ids, position_ids, cache)
        elapsed = time.time() - t0

    print(f"  compiled+ran in {elapsed:.1f}s, output {tuple(out.shape)} {out.dtype}", flush=True)
    print(f"  captured {len(summaries)} graph(s) -> {out_root}", flush=True)

    return {
        "step": step,
        **info,
        "output_shape": list(out.shape),
        "compile_and_run_seconds": round(elapsed, 2),
        "graphs": summaries,
    }


def write_report(out_root: Path, args, results: list[dict], text_config) -> None:
    report = {
        "model": args.model,
        "weights": args.weights,
        "dtype": args.dtype,
        "attn_impl": args.attn_impl,
        "num_hidden_layers": text_config.num_hidden_layers,
        "layer_types": list(getattr(text_config, "layer_types", []) or []),
        "vocab_size": text_config.vocab_size,
        "backend": args.backend,
        "aten_decomp": args.aten_decomp,
        "dynamic": args.dynamic,
        "fullgraph": args.fullgraph,
        "steps": results,
    }
    (out_root / "report.json").write_text(json.dumps(report, indent=2))

    lines = [f"# FX capture report — {args.model}", ""]
    lines.append(
        f"weights={args.weights} dtype={args.dtype} attn={args.attn_impl} "
        f"layers={text_config.num_hidden_layers} backend={args.backend} "
        f"decomp={args.aten_decomp} dynamic={args.dynamic}"
    )
    for res in results:
        lines += [
            "",
            f"## {res['step']} (batch={res['batch']} seq_len={res['seq_len']} past={res['past_len']})",
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


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    slug = args.model.rstrip("/").split("/")[-1].replace(".", "_")
    out_root = Path(
        args.out or Path(__file__).parent / "dumps" / f"{slug}_{args.mode}_b{args.batch}s{args.seq_len}"
    ).resolve()
    out_root.mkdir(parents=True, exist_ok=True)

    # env knobs must be set before torch is imported
    if args.torch_logs:
        os.environ["TORCH_LOGS"] = args.torch_logs
        log_path = out_root / "torch_logs.txt"
        log_path.write_text("")  # TORCH_LOGS_OUT appends; start each run clean
        os.environ["TORCH_LOGS_OUT"] = str(log_path)
    if args.inductor_trace:
        os.environ["TORCH_COMPILE_DEBUG"] = "1"
        os.environ["TORCH_COMPILE_DEBUG_DIR"] = str(out_root)
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    import torch

    torch.manual_seed(0)
    torch.set_grad_enabled(False)
    if args.inductor_trace:
        import torch._inductor.config as inductor_config

        inductor_config.trace.enabled = True
        inductor_config.trace.debug_dir = str(out_root)

    print(f"torch {torch.__version__} | out: {out_root}", flush=True)
    t0 = time.time()
    module, text_config = build_module(args)
    n_params = sum(p.numel() for p in module.parameters())
    print(
        f"module built in {time.time() - t0:.1f}s | {n_params / 1e9:.2f}B params | "
        f"{text_config.num_hidden_layers} layers | vocab {text_config.vocab_size}",
        flush=True,
    )

    steps = ["prefill", "decode"] if args.mode == "both" else [args.mode]
    results = []
    for step in steps:
        torch._dynamo.reset()
        results.append(capture_step(module, args, text_config, step, out_root))

    write_report(out_root, args, results, text_config)
    print(f"\nreport: {out_root / 'report.md'}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
