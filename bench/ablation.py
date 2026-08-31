"""Attribution: how much does each optimization actually contribute?

Runs the official benchmark repeatedly with individual optimizations disabled
through the TJ_* environment knobs, so the report can claim a number per
technique rather than one aggregate speedup.

Two views are produced:

  ladder      cumulative -- each row adds one layer of optimization
  leave-one   full stack minus a single component, showing what it was worth

    python bench/ablation.py --cases 1,6,13
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.run_all import RESULTS_DIR, run_case  # noqa: E402
from bench.shapes import by_index  # noqa: E402

LADDER = [
    ("0 baseline (control)",
     {"TJ_DISABLE": "1"}),
    ("1 packed QKV + SDPA + fused residual",
     {"TJ_NO_TRITON": "1", "TJ_NO_GRAPH": "1", "TJ_ATTN": "sdpa"}),
    ("2 + Triton norm/FFN kernels",
     {"TJ_NO_GRAPH": "1", "TJ_ATTN": "sdpa"}),
    ("3 + Triton causal flash attention",
     {"TJ_NO_GRAPH": "1"}),
    ("4 + CUDA Graph capture (full stack)",
     {}),
]

LEAVE_ONE = [
    ("full stack", {}),
    ("without CUDA Graphs", {"TJ_NO_GRAPH": "1"}),
    ("without Triton flash attn", {"TJ_ATTN": "sdpa"}),
    ("without fused FFN", {"TJ_FFN": "torch"}),
    ("without fused add+LayerNorm", {"TJ_NORM": "torch"}),
    ("without fp16 compute", {"TJ_COMPUTE_DTYPE": "float32"}),
]


def sweep(name, variants, cases, python, quick, dtype, seed) -> List[Dict]:
    rows = []
    for label, env in variants:
        env = dict(env)
        # The tuned plans.json would override the knobs being ablated.
        env["TJ_IGNORE_PLANS"] = "1"
        for case in cases:
            record = run_case(case, python, quick, dtype, seed, env, [])
            record["variant"] = label
            record["group"] = name
            rows.append(record)
            speed = record.get("speedup")
            print(f"  {label:<40} case {case.idx:2d}  "
                  f"{record['accuracy']:<5} "
                  + (f"{record.get('optimized_median_ms', float('nan')):8.3f} ms  "
                     f"{speed:6.2f}x" if isinstance(speed, float) else "  n/a"),
                  flush=True)
    return rows


def table(rows: List[Dict], cases) -> str:
    variants = []
    for row in rows:
        if row["variant"] not in variants:
            variants.append(row["variant"])
    header = "| variant | " + " | ".join(f"case {c.idx}" for c in cases) + " |"
    sep = "|---|" + "---:|" * len(cases)
    lines = [header, sep]
    for variant in variants:
        cells = []
        for case in cases:
            match = [
                r for r in rows
                if r["variant"] == variant and r["case"] == case.idx
            ]
            if match and isinstance(match[0].get("speedup"), float):
                cells.append(f"{match[0]['speedup']:.2f}x")
            else:
                cells.append("n/a")
        lines.append(f"| {variant} | " + " | ".join(cells) + " |")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="1,6,13")
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--quick", action="store_true", default=True)
    parser.add_argument("--full", dest="quick", action="store_false")
    args = parser.parse_args()

    cases = [by_index(int(c)) for c in args.cases.split(",")]
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    print("=== cumulative ladder ===")
    ladder_rows = sweep("ladder", LADDER, cases, args.python,
                        args.quick, args.dtype, args.seed)
    print("\n=== leave-one-out ===")
    leave_rows = sweep("leave_one", LEAVE_ONE, cases, args.python,
                       args.quick, args.dtype, args.seed)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    text = (
        "### Cumulative optimization ladder (speedup vs the reference)\n\n"
        + table(ladder_rows, cases)
        + "\n\n### Leave-one-out (full stack minus one component)\n\n"
        + table(leave_rows, cases)
        + "\n"
    )
    (RESULTS_DIR / f"ablation-{stamp}.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / "latest-ablation.md").write_text(text, encoding="utf-8")
    (RESULTS_DIR / f"ablation-{stamp}.json").write_text(
        json.dumps(ladder_rows + leave_rows, indent=2)
    )
    print("\n" + text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
