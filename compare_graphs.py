#!/usr/bin/env python3
"""Diff the captured prefill vs decode FX graphs of a dump directory.

    .venv/bin/python compare_graphs.py dumps/qwen35_4b_s128 --level aten

Reads the *.summary.json / *.nodes.json / *.inputs.json written by
capture_qwen35_fx.py and prints a markdown comparison: graph size, input
(placeholder) classification, output (cache write-back) classification and the
op histogram diff.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from pathlib import Path

ATEN = "torch._ops.aten."


def short(op: str) -> str:
    return op.replace(ATEN, "").replace("torch._ops.", "")


def load_step(level_dir: Path, step: str) -> dict | None:
    hits = sorted(level_dir.glob(f"{step}_graph*.summary.json"))
    if not hits:
        return None
    base = str(hits[0])[: -len(".summary.json")]
    data = {"name": Path(base).name}
    for key, suffix in (("summary", ".summary.json"), ("nodes", ".nodes.json"), ("inputs", ".inputs.json")):
        path = Path(base + suffix)
        data[key] = json.loads(path.read_text()) if path.exists() else None
    return data


def shape_key(entry: dict) -> tuple:
    return tuple(entry.get("shape") or ())


def outputs_of(nodes: list[dict]) -> list[dict]:
    """Resolve the output node's args back to the producing nodes, in order."""
    by_name = {n["name"]: n for n in nodes}
    out_nodes = [n for n in nodes if n["op"] == "output"]
    if not out_nodes:
        return []
    names = re.findall(r"[A-Za-z_][A-Za-z_0-9]*", str(out_nodes[0]["args"]))
    seen, result = set(), []
    for name in names:
        if name in by_name and name not in seen:
            seen.add(name)
            node = by_name[name]
            result.append(
                {
                    "name": name,
                    "target": short(node["target"]),
                    "shape": tuple((node["meta"] or {}).get("shape") or ()),
                    "dtype": (node["meta"] or {}).get("dtype", ""),
                }
            )
    return result


def attribute_nodes(readable_path: Path) -> Counter:
    """Count graph nodes per originating source function, using print_readable's
    `# File: <path>:<line> in <fn>, code: ...` annotations."""
    counts: Counter = Counter()
    if not readable_path.exists():
        return counts
    current = None
    for line in readable_path.read_text().splitlines():
        stripped = line.strip()
        match = re.match(r"# File: (.+?):\d+ in ([A-Za-z_0-9]+),", stripped)
        if match:
            path, fn = match.groups()
            current = f"{path.split('/')[-1]}:{fn}"
            continue
        if not stripped or stripped.startswith("#"):
            continue
        if re.match(r'[a-z_0-9]+\s*:\s*"', stripped) or stripped.startswith("return"):
            counts[current or "(unattributed)"] += 1
    return counts


def table(rows: list[tuple], header: tuple) -> list[str]:
    out = ["| " + " | ".join(header) + " |", "| " + " | ".join("---" for _ in header) + " |"]
    out += ["| " + " | ".join(str(c) for c in row) + " |" for row in rows]
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("dump_dir", type=Path, help="output dir produced by capture_qwen35_fx.py")
    ap.add_argument("--level", default="aten", choices=["aten", "dynamo"])
    ap.add_argument("--top", type=int, default=20, help="rows in the op histogram")
    ap.add_argument("--out", type=Path, default=None, help="also write the markdown here")
    args = ap.parse_args()

    level_dir = args.dump_dir / args.level
    if not level_dir.is_dir():
        raise SystemExit(f"no such level dir: {level_dir}")

    pre, dec = load_step(level_dir, "prefill"), load_step(level_dir, "decode")
    if pre is None or dec is None:
        raise SystemExit(f"need both prefill and decode graphs in {level_dir}")

    lines = [f"# prefill vs decode — `{args.dump_dir.name}` ({args.level} level)", ""]

    # ---- size ------------------------------------------------------------- #
    outs = {k: outputs_of(v["nodes"] or []) for k, v in (("prefill", pre), ("decode", dec))}
    lines += ["## 1. 그래프 크기", ""]
    lines += table(
        [
            (
                "노드 수",
                pre["summary"]["num_nodes"],
                dec["summary"]["num_nodes"],
                f"{pre['summary']['num_nodes'] / max(dec['summary']['num_nodes'], 1):.1f}x",
            ),
            (
                "call_function",
                pre["summary"]["num_call_function"],
                dec["summary"]["num_call_function"],
                f"{pre['summary']['num_call_function'] / max(dec['summary']['num_call_function'], 1):.1f}x",
            ),
            (
                "placeholder(입력)",
                pre["summary"]["num_placeholders"],
                dec["summary"]["num_placeholders"],
                dec["summary"]["num_placeholders"] - pre["summary"]["num_placeholders"],
            ),
            ("output(출력)", len(outs["prefill"]), len(outs["decode"]), len(outs["decode"]) - len(outs["prefill"])),
        ],
        ("", "prefill", "decode", "차이"),
    )

    # ---- inputs ----------------------------------------------------------- #
    pre_in = Counter(shape_key(e) for e in (pre["inputs"] or []))
    dec_in = Counter(shape_key(e) for e in (dec["inputs"] or []))
    only_dec = {k: v for k, v in dec_in.items() if k not in pre_in}
    only_pre = {k: v for k, v in pre_in.items() if k not in dec_in}
    changed = {k: (pre_in[k], dec_in[k]) for k in pre_in if k in dec_in and pre_in[k] != dec_in[k]}

    lines += ["", "## 2. 입력 (placeholder)", "", "### decode에만 있는 입력 = 캐시 상태", ""]
    lines += table(
        [(f"`{list(k)}`", v) for k, v in sorted(only_dec.items(), key=lambda kv: -kv[1])],
        ("shape", "개수"),
    )
    if only_pre or changed:
        lines += ["", "### prefill에만 있거나 개수가 다른 입력", ""]
        lines += table(
            [(f"`{list(k)}`", v, 0) for k, v in only_pre.items()]
            + [(f"`{list(k)}`", a, b) for k, (a, b) in changed.items()],
            ("shape", "prefill", "decode"),
        )

    # ---- outputs ---------------------------------------------------------- #
    lines += ["", "## 3. 출력 (logits + 캐시 write-back)", ""]
    for step in ("prefill", "decode"):
        grouped = Counter((o["target"], o["shape"], o["dtype"]) for o in outs[step])
        lines += [f"### {step} — 총 {len(outs[step])}개", ""]
        lines += table(
            [(f"`{t}`", f"`{list(s)}`", d, n) for (t, s, d), n in sorted(grouped.items(), key=lambda kv: -kv[1])],
            ("생성 op", "shape", "dtype", "개수"),
        )
        lines += [""]

    # ---- op histogram ----------------------------------------------------- #
    pre_ops = Counter({short(k): v for k, v in pre["summary"]["op_counts"].items()})
    dec_ops = Counter({short(k): v for k, v in dec["summary"]["op_counts"].items()})
    same = sorted(k for k in pre_ops if k in dec_ops and pre_ops[k] == dec_ops[k])
    diff = sorted((k for k in pre_ops if k in dec_ops and pre_ops[k] != dec_ops[k]), key=lambda k: -pre_ops[k])
    p_only = sorted((k for k in pre_ops if k not in dec_ops), key=lambda k: -pre_ops[k])
    d_only = sorted((k for k in dec_ops if k not in pre_ops), key=lambda k: -dec_ops[k])

    lines += ["", "## 4. op 히스토그램", ""]
    lines += [f"공통 op {len(set(pre_ops) & set(dec_ops))}종 / prefill 전용 {len(p_only)}종 / decode 전용 {len(d_only)}종", ""]
    lines += ["### 4-1. 개수까지 동일한 op (= 레이어 구조 그 자체)", ""]
    lines += table([(f"`{k}`", pre_ops[k]) for k in sorted(same, key=lambda k: -pre_ops[k])[: args.top]], ("op", "개수"))
    lines += ["", "### 4-2. 개수가 다른 op (= 토큰 축 / 시퀀스 처리 차이)", ""]
    lines += table(
        [(f"`{k}`", pre_ops[k], dec_ops[k], f"{pre_ops[k] / max(dec_ops[k], 1):.1f}x") for k in diff[: args.top]],
        ("op", "prefill", "decode", "비율"),
    )
    lines += ["", "### 4-3. 한쪽에만 있는 op", ""]
    lines += table(
        [(f"`{k}`", pre_ops[k], "-") for k in p_only[: args.top]]
        + [(f"`{k}`", "-", dec_ops[k]) for k in d_only[: args.top]],
        ("op", "prefill", "decode"),
    )

    # ---- source attribution ----------------------------------------------- #
    attrib = {step: attribute_nodes(level_dir / f"{data['name']}.readable.py") for step, data in (("prefill", pre), ("decode", dec))}
    if any(attrib.values()):
        lines += ["", "## 5. 노드가 어느 소스에서 나왔나 (readable 덤프의 `# File:` 주석 기준)", ""]
        for step in ("prefill", "decode"):
            counts = attrib[step]
            if not counts:
                continue
            total = sum(counts.values())
            lines += [f"### {step} — 귀속된 노드 {total}개", ""]
            lines += table(
                [(f"`{src}`", n, f"{100 * n / total:.1f}%") for src, n in counts.most_common(8)],
                ("source", "노드", "비중"),
            )
            lines += [""]

    text = "\n".join(lines) + "\n"
    print(text)
    if args.out:
        args.out.write_text(text)
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
