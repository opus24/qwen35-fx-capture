#!/usr/bin/env python3
"""Diff two dumps that differ only in which linear-attention kernels were bound.

    python compare_kernel_paths.py dumps/qwen35_4b_s128_fla_gpu dumps/qwen35_4b_s128_torch_gpu

`compare_graphs.py` answers "prefill vs decode inside one dump" and only ever looks at
`*_graph0`. That is not enough here: the FLA prefill shatters into ~31 graphs, so the
interesting numbers (node totals, op histograms) only exist *summed over all graphs*.

Reads the `report.json` + `<step>_graph*.summary.json` written by
capture_qwen35_fx_fla_gpu.py and prints a markdown comparison of kernel path, graph
count / graph breaks, aggregate node counts, eager latency and the op histogram.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

ATEN = "torch._ops.aten."


def short(op: str) -> str:
    return op.replace(ATEN, "").replace("torch._ops.", "")


def load_dump(path: Path, level: str) -> dict:
    report = json.loads((path / "report.json").read_text())
    steps: dict[str, dict] = {}
    for res in report.get("steps", []):
        graphs = [g for g in res.get("graphs", []) if g["level"] == level]
        ops: Counter = Counter()
        for g in graphs:
            ops.update(g.get("op_counts", {}))
        steps[res["step"]] = {
            "graphs": len(graphs),
            "graph_count": res.get("graph_count"),
            "graph_break_count": res.get("graph_break_count"),
            "nodes": sum(g["num_nodes"] for g in graphs),
            "call_function": sum(g["num_call_function"] for g in graphs),
            "largest": max((g["num_nodes"] for g in graphs), default=0),
            "ops": ops,
        }
    return {"name": path.name, "report": report, "steps": steps}


def table(rows: list[tuple], header: tuple) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def kernel_path_section(a: dict, b: dict) -> list[str]:
    keys = [
        "delta_rule_chunk",
        "delta_rule_recurrent",
        "conv1d_fn",
        "conv1d_update",
        "gated_rmsnorm",
        "fla_delta_rule_active",
    ]
    ka, kb = a["report"].get("kernel_path", {}), b["report"].get("kernel_path", {})
    rows = [(k, f"`{ka.get(k, '?')}`", f"`{kb.get(k, '?')}`") for k in keys]
    return ["## bound kernels", ""] + table(rows, ("", a["name"], b["name"]))


def latency_section(a: dict, b: dict) -> list[str]:
    ba, bb = a["report"].get("bench"), b["report"].get("bench")
    if not (ba and bb):
        return []
    rows = []
    pa, pb = ba["prefill"], bb["prefill"]
    rows.append(
        (
            f"prefill (b{pa['batch']} s{pa['seq_len']})",
            f"{pa['median_ms']} ms",
            f"{pb['median_ms']} ms",
            f"{pb['median_ms'] / pa['median_ms']:.2f}×",
        )
    )
    da, db = ba["decode"], bb["decode"]
    rows.append(
        (
            f"decode (past={da['past_len']})",
            f"{da['per_step']['median_ms']} ms/token",
            f"{db['per_step']['median_ms']} ms/token",
            f"{db['per_step']['median_ms'] / da['per_step']['median_ms']:.2f}×",
        )
    )
    rows.append(("peak memory", f"{ba['peak_memory_gb']} GB", f"{bb['peak_memory_gb']} GB", ""))
    lines = ["## eager latency (CUDA events, median)", ""]
    lines += table(rows, ("", a["name"], b["name"], f"{b['name']} / {a['name']}"))

    # the two runs are only comparable if they hold the same weights
    fa, fb = a["report"].get("param_checksum"), b["report"].get("param_checksum")
    if isinstance(fa, dict) and isinstance(fb, dict):
        same = fa == fb
        lines += ["", f"weights identical: **{same}** (`{fa}` vs `{fb}`)"]
        la = ba.get("prefill_logits")
        lb = bb.get("prefill_logits")
        if same and la and lb:
            lines.append(
                f"prefill logits — {a['name']}: mean={la['mean']} std={la['std']} absmax={la['absmax']} | "
                f"{b['name']}: mean={lb['mean']} std={lb['std']} absmax={lb['absmax']}"
            )
    return lines


def graph_section(a: dict, b: dict, level: str, top: int) -> list[str]:
    lines: list[str] = []
    for step in ("prefill", "decode"):
        sa, sb = a["steps"].get(step), b["steps"].get(step)
        if not (sa and sb):
            continue
        lines += ["", f"## {step} — {level} graphs", ""]
        rows = [
            ("dynamo graph count", sa["graph_count"], sb["graph_count"]),
            ("graph breaks", sa["graph_break_count"], sb["graph_break_count"]),
            (f"{level} graphs dumped", sa["graphs"], sb["graphs"]),
            ("total nodes", sa["nodes"], sb["nodes"]),
            ("largest single graph", sa["largest"], sb["largest"]),
            ("call_function nodes", sa["call_function"], sb["call_function"]),
        ]
        lines += table(rows, ("", a["name"], b["name"]))

        ops = sorted(
            set(sa["ops"]) | set(sb["ops"]),
            key=lambda op: -(sa["ops"].get(op, 0) + sb["ops"].get(op, 0)),
        )[:top]
        lines += ["", f"### top ops — {step} / {level}", ""]
        lines += table(
            [(short(op), sa["ops"].get(op, 0), sb["ops"].get(op, 0)) for op in ops],
            ("op", a["name"], b["name"]),
        )
    return lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_a", type=Path, help="first dump dir (conventionally the FLA one)")
    ap.add_argument("dump_b", type=Path, help="second dump dir (conventionally the torch baseline)")
    ap.add_argument("--level", default="aten", choices=["aten", "dynamo"])
    ap.add_argument("--top", type=int, default=25, help="rows in the op histogram")
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = ap.parse_args()

    a = load_dump(args.dump_a, args.level)
    b = load_dump(args.dump_b, args.level)

    lines = [f"# kernel-path diff — {a['name']} vs {b['name']}", ""]
    lines += kernel_path_section(a, b)
    lines += [""] + latency_section(a, b)
    lines += graph_section(a, b, args.level, args.top)

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(text)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
