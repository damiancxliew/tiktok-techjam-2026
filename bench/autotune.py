"""Accuracy-gated autotuner.

The usual autotuner asks "which configuration is fastest". That is the wrong
question here, because the fastest configuration is not always one that stays
inside the competition's error budget -- and the budget is what makes the
result count at all.

So this searches for **the fastest plan that still passes a deliberately
tightened version of the official tolerance**. The official rule is

    abs_err <= 2e-3  OR  rel_err <= 2e-2

A plan is admitted only if, across several input draws including seeds it was
not tuned on, no element uses more than `--margin` (default 0.85) of its own
error budget, where an element's budget is

    budget = max(atol, rtol * |reference|)

This *budget usage* is the right quantity, and getting there took two tries.
Gating on absolute error alone says case 7 uses 96% of its allowance and looks
one unlucky seed from failing -- but the elements carrying that error have
|reference| near 1, so their real budget is the relative one, 10x larger, and
they are nowhere near the edge. Measuring usage per element instead of against
a single scalar keeps the headroom guarantee without paying 2.4x in speed to
escape a threshold that was never really binding.

Search is coordinate descent from the heuristic plan: change one knob, keep it
if it is both admissible and faster. That is ~12 evaluations per shape instead
of the 64 a full cross product would need.

    python bench/autotune.py                 # all cases, writes tjkernels/plans.json
    python bench/autotune.py --cases 7,8     # subset
    python bench/autotune.py --dry-run       # do not write plans.json
"""

from __future__ import annotations

import argparse
import gc
import json
import sys
import time
from dataclasses import replace
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch_transformer_benchmark as bench  # noqa: E402
from tjkernels.plans import Plan, default_plan, save_plans, set_override, shape_key  # noqa: E402
from bench.shapes import ALL_CASES, Case, by_index  # noqa: E402

ATOL = 2e-3
RTOL = 2e-2
HELD_OUT_SEEDS = [777, 4242]


def budget_usage(ref: torch.Tensor, got: torch.Tensor) -> float:
    """Worst fraction of its own error budget any element consumes.

    Computed with in-place ops: at case 6 these tensors are 655 MB each.
    """
    err = (got.detach().float() - ref.detach().float()).abs_()
    budget = ref.detach().float().abs().mul_(RTOL).clamp_min_(ATOL)
    return float(err.div_(budget).max().item())


def make_config(case: Case) -> "bench.TransformerConfig":
    return bench.TransformerConfig(
        batch_size=case.batch, seq_len=case.seq, d_model=case.d_model,
        num_heads=case.heads, ffn_dim=case.ffn, num_layers=case.layers,
        causal=case.causal,
    )


def candidate_knobs(case: Case, base: Plan) -> List[Tuple[str, object]]:
    """One-knob mutations to try, cheapest-to-evaluate first."""
    other_compute = (
        torch.float16 if base.compute_dtype == torch.float32 else torch.float32
    )
    knobs: List[Tuple[str, object]] = [
        ("cuda_graph", not base.cuda_graph),
        ("attn", "sdpa" if base.attn == "triton" else "triton"),
        ("norm", "torch" if base.norm == "triton" else "triton"),
        ("residual_dtype", torch.float16),
        ("compute_dtype", other_compute),
    ]
    from tjkernels.kernels import ffn_supports

    if ffn_supports(case.d_model, case.ffn, base.compute_dtype.itemsize):
        knobs.insert(2, ("ffn", "torch" if base.ffn == "triton" else "triton"))
    return knobs


def evaluate(
    plan: Plan,
    baseline: torch.nn.Module,
    config: "bench.TransformerConfig",
    case: Case,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    margin: float,
    timing_iters: int,
) -> Dict[str, object]:
    """Accuracy (tightened tolerance) then latency, for one plan."""
    set_override(plan)
    optimized = bench.UserOptimizedTransformer(config)
    bench.copy_model_weights(baseline, optimized, strict=True)
    optimized = optimized.to(device=device, dtype=dtype).eval()

    worst_abs = 0.0
    worst_usage = 0.0
    failed = 0
    total = 0
    # Seeds 1..trials are the ones the official harness uses at --seed 1; the
    # rest are held out, so a plan cannot be tuned into passing by luck.
    seeds = list(range(1, trials + 1)) + HELD_OUT_SEEDS
    try:
        with torch.inference_mode():
            for seed in seeds:
                x, mask = bench.generate_random_case(
                    config, device, dtype, seed=seed,
                    padding_ratio=0.0, input_scale=1.0,
                )
                ref = baseline(x, mask)
                got = optimized(x, mask)
                result = bench.compare_outputs(ref, got, rtol=RTOL, atol=ATOL)
                worst_abs = max(worst_abs, result.max_abs_error)
                failed += result.failed_elements
                total += result.total_elements
                worst_usage = max(worst_usage, budget_usage(ref, got))
                del ref, got, x, mask

            # Latency on a fixed input, matching the official harness.
            x, mask = bench.generate_random_case(
                config, device, dtype, seed=100001,
                padding_ratio=0.0, input_scale=1.0,
            )
            for _ in range(5):
                optimized(x, mask)
            torch.cuda.synchronize()
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(timing_iters):
                optimized(x, mask)
            end.record()
            torch.cuda.synchronize()
            latency = start.elapsed_time(end) / timing_iters
    except Exception as exc:  # OOM, smem limits, unsupported tiles
        failure = {"plan": plan, "error": str(exc)[:120], "admissible": False}
    else:
        failure = None
    finally:
        set_override(None)
        del optimized
        gc.collect()
        torch.cuda.empty_cache()

    if failure is not None:
        return failure

    return {
        "plan": plan,
        "latency_ms": latency,
        "max_abs": worst_abs,
        "failed": failed,
        "total": total,
        "usage": worst_usage,
        "admissible": failed == 0 and worst_usage <= margin,
    }


def tune_case(
    case: Case, device: torch.device, dtype: torch.dtype,
    trials: int, margin: float, timing_iters: int, verbose: bool,
) -> Optional[Dict[str, object]]:
    config = make_config(case)
    torch.manual_seed(1)
    baseline = bench.BaselineTransformer(config).to(device=device, dtype=dtype).eval()

    base_plan = replace(
        default_plan(
            case.batch, case.seq, case.d_model, case.heads, case.ffn,
            case.layers, case.causal, dtype, device,
        ),
        label="autotuned",
    )

    results: List[Dict[str, object]] = []
    current = evaluate(
        base_plan, baseline, config, case, device, dtype,
        trials, margin, timing_iters,
    )
    results.append(current)
    if verbose:
        _print_result("heuristic", current)

    if not current.get("admissible"):
        # The heuristic itself is out of budget: escalate precision first.
        for fallback in (
            replace(base_plan, compute_dtype=torch.float32, label="autotuned-fp32"),
        ):
            probe = evaluate(
                fallback, baseline, config, case, device, dtype,
                trials, margin, timing_iters,
            )
            results.append(probe)
            if verbose:
                _print_result("escalate-fp32", probe)
            if probe.get("admissible"):
                current = probe
                break

    best = current
    for knob, value in candidate_knobs(case, best["plan"]):
        trial_plan = replace(best["plan"], **{knob: value})
        probe = evaluate(
            trial_plan, baseline, config, case, device, dtype,
            trials, margin, timing_iters,
        )
        results.append(probe)
        if verbose:
            _print_result(f"{knob}={_short(value)}", probe)
        if (
            probe.get("admissible")
            and best.get("admissible")
            and float(probe["latency_ms"]) < float(best["latency_ms"]) * 0.98
        ):
            best = probe
        elif probe.get("admissible") and not best.get("admissible"):
            best = probe

    del baseline
    gc.collect()
    torch.cuda.empty_cache()

    if not best.get("admissible"):
        return None
    best["case"] = case.idx
    best["explored"] = len(results)
    return best


def _short(value) -> str:
    return str(value).replace("torch.", "")


def _print_result(label: str, result: Dict[str, object]) -> None:
    if "error" in result:
        print(f"    {label:<24} ERROR {result['error']}")
        return
    flag = "ok " if result["admissible"] else "REJ"
    print(f"    {label:<24} {flag} {result['latency_ms']:8.3f} ms   "
          f"max_abs={result['max_abs']:.2e}  budget={result['usage']:.0%}  "
          f"failed={result['failed']}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cases", default="")
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--trials", type=int, default=4)
    parser.add_argument("--margin", type=float, default=0.85,
                        help="max fraction of its own budget any element may use")
    parser.add_argument("--timing-iters", type=int, default=30)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()

    cases = ALL_CASES
    if args.cases:
        cases = [by_index(int(c)) for c in args.cases.split(",")]

    device = torch.device("cuda")
    dtype = bench.resolve_dtype(args.dtype)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    plans: Dict[str, Plan] = {}
    summary = []
    started = time.time()

    for case in cases:
        print(f"case {case.idx:2d}  B={case.batch} S={case.seq} d={case.d_model} "
              f"H={case.heads} F={case.ffn}  ({case.note})", flush=True)
        best = tune_case(
            case, device, dtype, args.trials, args.margin,
            args.timing_iters, not args.quiet,
        )
        if best is None:
            print("    no admissible plan found -- leaving heuristic in place\n")
            continue
        key = shape_key(
            case.batch, case.seq, case.d_model, case.heads, case.ffn,
            case.layers, case.causal, dtype,
        )
        plans[key] = best["plan"]
        summary.append({
            "case": case.idx,
            "key": key,
            "latency_ms": best["latency_ms"],
            "max_abs": best["max_abs"],
            "usage": best["usage"],
            "explored": best["explored"],
            "plan": best["plan"].to_json(),
        })
        print(f"    -> {best['latency_ms']:.3f} ms  "
              f"max_abs={best['max_abs']:.2e}  {best['plan'].to_json()}\n", flush=True)

    if not args.dry_run and plans:
        save_plans(plans)
        print(f"wrote {ROOT / 'tjkernels' / 'plans.json'} ({len(plans)} shapes)")

    out = ROOT / "report" / "results" / "autotune-summary.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({
        "margin": args.margin,
        "trials": args.trials,
        "atol": ATOL,
        "rtol": RTOL,
        "seconds": time.time() - started,
        "cases": summary,
    }, indent=2))
    print(f"tuning took {time.time() - started:.0f}s")
    return 0


if __name__ == "__main__":
    sys.exit(main())
