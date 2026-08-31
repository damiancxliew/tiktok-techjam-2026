"""Run the official benchmark across the whole test matrix and tabulate.

Each case runs in its own subprocess so that a CUDA OOM or a crash in one
shape cannot poison the rest of the sweep, and so every case gets a clean
allocator and a cold cuBLAS workspace.

    python bench/run_all.py                 # full sweep, official iteration counts
    python bench/run_all.py --quick         # fast smoke sweep
    python bench/run_all.py --cases 2,6,13  # subset
    python bench/run_all.py --tag ablation-no-graph --env TJ_NO_GRAPH=1
"""

from __future__ import annotations

import argparse
import json
import os
import re
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.shapes import ALL_CASES, Case, by_index  # noqa: E402

RESULTS_DIR = ROOT / "report" / "results"

RE_SUMMARY = re.compile(
    r"summary: (PASS|FAIL) \| max_abs=([\d.eE+-]+) \| max_rel=([\d.eE+-]+) \| "
    r"failed=(\d+)/(\d+)"
)
RE_BASELINE = re.compile(r"baseline\s*: median=([\d.]+) ms.*?min=([\d.]+) ms")
RE_OPTIMIZED = re.compile(r"optimized: median=([\d.]+) ms.*?min=([\d.]+) ms")
RE_SPEEDUP = re.compile(r"speedup\s*: ([\d.]+)x")


def run_case(
    case: Case,
    python: str,
    quick: bool,
    dtype: str,
    seed: int,
    extra_env: Dict[str, str],
    extra_args: List[str],
) -> Dict[str, object]:
    cmd = [python, str(ROOT / "torch_transformer_benchmark.py")]
    cmd += case.cli()
    cmd += ["--seed", str(seed), "--dtype", dtype]
    if quick:
        cmd += ["--accuracy-trials", "2", "--warmup", "5",
                "--repeats", "20", "--benchmark-rounds", "1"]
    cmd += extra_args

    env = dict(os.environ)
    env.update(extra_env)
    env.setdefault("PYTHONIOENCODING", "utf-8")

    started = datetime.now()
    proc = subprocess.run(
        cmd, capture_output=True, text=True, env=env, cwd=str(ROOT)
    )
    out = proc.stdout + proc.stderr

    record: Dict[str, object] = {
        "case": case.idx,
        "name": case.name,
        "note": case.note,
        "batch": case.batch,
        "seq": case.seq,
        "d_model": case.d_model,
        "heads": case.heads,
        "ffn": case.ffn,
        "layers": case.layers,
        "tokens": case.tokens,
        "dtype": dtype,
        "seed": seed,
        "returncode": proc.returncode,
        "wall_seconds": (datetime.now() - started).total_seconds(),
        "cmd": " ".join(cmd[1:]),
    }

    summary = RE_SUMMARY.search(out)
    if summary:
        record["accuracy"] = summary.group(1)
        record["max_abs"] = float(summary.group(2))
        record["max_rel"] = float(summary.group(3))
        record["failed_elements"] = int(summary.group(4))
        record["total_elements"] = int(summary.group(5))
    else:
        record["accuracy"] = "ERROR"

    for key, pattern in (("baseline", RE_BASELINE), ("optimized", RE_OPTIMIZED)):
        match = pattern.search(out)
        if match:
            record[f"{key}_median_ms"] = float(match.group(1))
            record[f"{key}_min_ms"] = float(match.group(2))

    speedup = RE_SPEEDUP.search(out)
    if speedup:
        record["speedup"] = float(speedup.group(1))

    if record["accuracy"] == "ERROR" or proc.returncode != 0:
        tail = [ln for ln in out.strip().splitlines() if ln.strip()][-6:]
        record["error_tail"] = "\n".join(tail)
        if "out of memory" in out.lower():
            record["accuracy"] = "OOM"

    return record


def format_table(records: List[Dict[str, object]]) -> str:
    header = (
        "| # | shape (B,S,d,H,F,L) | tokens | accuracy | max_abs | "
        "baseline ms | ours ms | speedup |"
    )
    sep = "|---|---|---:|---|---:|---:|---:|---:|"
    lines = [header, sep]
    for r in records:
        shape = (f"{r['batch']},{r['seq']},{r['d_model']},"
                 f"{r['heads']},{r['ffn']},{r['layers']}")
        base = r.get("baseline_median_ms")
        ours = r.get("optimized_median_ms")
        speed = r.get("speedup")
        lines.append(
            f"| {r['case']} | {shape} | {r['tokens']:,} | {r['accuracy']} | "
            f"{r.get('max_abs', float('nan')):.2e} | "
            f"{base if base is None else f'{base:.3f}'} | "
            f"{ours if ours is None else f'{ours:.3f}'} | "
            f"{speed if speed is None else f'{speed:.2f}x'} |"
        )
    finished = [r for r in records if isinstance(r.get("speedup"), float)]
    if finished:
        speedups = [float(r["speedup"]) for r in finished]
        geomean = float(
            __import__("math").exp(
                sum(__import__("math").log(s) for s in speedups) / len(speedups)
            )
        )
        lines.append("")
        lines.append(
            f"**{len(finished)}/{len(records)} cases benchmarked - "
            f"geometric-mean speedup {geomean:.2f}x**"
        )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--quick", action="store_true")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--cases", default="", help="comma-separated case ids")
    parser.add_argument("--tag", default="main")
    parser.add_argument("--env", action="append", default=[],
                        help="KEY=VALUE passed to each subprocess")
    parser.add_argument("--extra", nargs=argparse.REMAINDER, default=[],
                        help="extra args forwarded to the benchmark")
    args = parser.parse_args()

    cases = ALL_CASES
    if args.cases:
        cases = [by_index(int(c)) for c in args.cases.split(",")]

    extra_env = {}
    for item in args.env:
        key, _, value = item.partition("=")
        extra_env[key] = value

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    records: List[Dict[str, object]] = []
    for case in cases:
        print(f"--- case {case.idx:2d}  B={case.batch} S={case.seq} "
              f"d={case.d_model} H={case.heads} F={case.ffn}  ({case.note})",
              flush=True)
        record = run_case(
            case, args.python, args.quick, args.dtype, args.seed,
            extra_env, args.extra,
        )
        records.append(record)
        status = record["accuracy"]
        speed = record.get("speedup")
        print(f"    {status}"
              + (f"  speedup={speed:.2f}x" if isinstance(speed, float) else "")
              + f"  ({record['wall_seconds']:.1f}s)", flush=True)
        if "error_tail" in record:
            print("    " + str(record["error_tail"]).replace("\n", "\n    "),
                  flush=True)

    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    payload = {
        "tag": args.tag,
        "timestamp": stamp,
        "dtype": args.dtype,
        "seed": args.seed,
        "quick": args.quick,
        "env": extra_env,
        "records": records,
    }
    json_path = RESULTS_DIR / f"{args.tag}-{stamp}.json"
    json_path.write_text(json.dumps(payload, indent=2))
    table = format_table(records)
    (RESULTS_DIR / f"{args.tag}-{stamp}.md").write_text(table, encoding="utf-8")
    (RESULTS_DIR / f"latest-{args.tag}.md").write_text(table, encoding="utf-8")

    print()
    print(table)
    print(f"\nwrote {json_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
