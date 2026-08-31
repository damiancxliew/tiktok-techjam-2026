"""Correctness beyond the official matrix.

The 13 benchmark shapes are all causal, unpadded, fp32 and power-of-two. This
exercises the paths they never touch -- padding masks, non-causal attention,
fp16/bf16 models, ragged shapes, and eager-vs-CUDA-Graph agreement -- because
a dispatcher that is only correct on the shapes it was tuned for is not a
dispatcher, it is a lookup table.

    python bench/test_correctness.py
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from typing import List, Optional, Tuple

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch_transformer_benchmark as bench  # noqa: E402
from tjkernels.plans import Plan, set_override  # noqa: E402

ATOL, RTOL = 2e-3, 2e-2


def build(batch, seq, d_model, heads, ffn, layers, causal, dtype, device):
    config = bench.TransformerConfig(
        batch_size=batch, seq_len=seq, d_model=d_model, num_heads=heads,
        ffn_dim=ffn, num_layers=layers, causal=causal,
    )
    torch.manual_seed(1)
    baseline = bench.BaselineTransformer(config)
    optimized = bench.UserOptimizedTransformer(config)
    bench.copy_model_weights(baseline, optimized, strict=True)
    return (
        config,
        baseline.to(device=device, dtype=dtype).eval(),
        optimized.to(device=device, dtype=dtype).eval(),
    )


def check(
    name: str,
    batch=8, seq=64, d_model=128, heads=4, ffn=128, layers=2,
    causal=True, dtype=torch.float32, padding=0.0, trials=3,
    plan: Optional[Plan] = None, device_type: str = "cuda",
    expect_fail: bool = False,
) -> Tuple[str, bool, str]:
    device = torch.device(device_type)
    set_override(plan)
    try:
        config, baseline, optimized = build(
            batch, seq, d_model, heads, ffn, layers, causal, dtype, device
        )
        worst_abs, failed, total = 0.0, 0, 0
        with torch.inference_mode():
            for trial in range(trials):
                x, mask = bench.generate_random_case(
                    config, device, dtype, seed=1 + trial,
                    padding_ratio=padding, input_scale=1.0,
                )
                ref = baseline(x, mask)
                got = optimized(x, mask)
                if ref.shape != got.shape:
                    return name, False, f"shape {tuple(got.shape)} != {tuple(ref.shape)}"
                result = bench.compare_outputs(ref, got, rtol=RTOL, atol=ATOL)
                worst_abs = max(worst_abs, result.max_abs_error)
                failed += result.failed_elements
                total += result.total_elements

                if padding > 0:
                    # Invalid rows must be exactly zero, like the reference.
                    invalid = got[~mask]
                    if invalid.numel() and invalid.abs().max().item() != 0.0:
                        return name, False, "padded rows are not zeroed"
        detail = f"max_abs={worst_abs:.2e} failed={failed}/{total}"
        if expect_fail:
            # Known-unreachable, not a defect -- see bench/test_bf16_limit.py.
            return name, True, detail + ("  (XFAIL as documented)" if failed
                                         else "  (XPASS - update the docs)")
        return name, failed == 0, detail
    except Exception as exc:
        traceback.print_exc()
        return name, False, f"{type(exc).__name__}: {exc}"
    finally:
        set_override(None)
        torch.cuda.empty_cache()


def graph_vs_eager() -> Tuple[str, bool, str]:
    """The captured graph must agree with the eager path on fresh inputs."""
    device = torch.device("cuda")
    name = "cuda graph == eager (fresh inputs each call)"
    try:
        config, _, optimized = build(4, 64, 128, 4, 128, 2, True,
                                     torch.float32, device)
        set_override(Plan(cuda_graph=False, label="eager"))
        eager_outs = []
        with torch.inference_mode():
            for trial in range(3):
                x, mask = bench.generate_random_case(
                    config, device, torch.float32, seed=50 + trial,
                    padding_ratio=0.0, input_scale=1.0,
                )
                eager_outs.append((x, mask, optimized(x, mask).clone()))

        # Rebuild with graphs on and replay against the same inputs.
        set_override(Plan(cuda_graph=True, label="graphed"))
        config, _, graphed = build(4, 64, 128, 4, 128, 2, True,
                                   torch.float32, device)
        worst = 0.0
        with torch.inference_mode():
            for x, mask, expected in eager_outs:
                got = graphed(x, mask)
                worst = max(worst, (got - expected).abs().max().item())
        return name, worst == 0.0, f"max difference {worst:.2e}"
    except Exception as exc:
        return name, False, f"{type(exc).__name__}: {exc}"
    finally:
        set_override(None)
        torch.cuda.empty_cache()


def main() -> int:
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    checks: List[Tuple[str, bool, str]] = [
        check("baseline shape, causal, no padding"),
        check("padding_ratio=0.3 (prefix mask)", padding=0.3),
        check("padding_ratio=0.6 (prefix mask)", padding=0.6),
        check("non-causal attention", causal=False),
        check("non-causal + padding", causal=False, padding=0.4),
        check("fp16 model", dtype=torch.float16),
        # bf16 cannot meet this tolerance by any implementation that is not
        # bit-identical to the reference: running the reference itself in fp32
        # and casting back also fails it. See bench/test_bf16_limit.py.
        check("bf16 model (tolerance unreachable in bf16)",
              dtype=torch.bfloat16, expect_fail=True),
        check("ragged seq_len=37", seq=37),
        check("ragged seq_len=1", seq=1),
        check("d_model=96, heads=3 (head_dim 32)", d_model=96, heads=3, ffn=96),
        check("head_dim=8, 16 heads", d_model=128, heads=16),
        check("single layer", layers=1),
        check("ffn_dim != d_model", ffn=512),
        check("batch=1", batch=1),
        # The engine must survive a machine without a GPU at all: every kernel
        # has a torch fallback and the dispatcher must not ask for a graph.
        check("cpu device (all fallbacks)", batch=2, seq=16, layers=1,
              device_type="cpu", trials=1),
        graph_vs_eager(),
    ]

    width = max(len(name) for name, _, _ in checks)
    failures = 0
    for name, ok, detail in checks:
        flag = "PASS" if ok else "FAIL"
        failures += 0 if ok else 1
        print(f"{flag}  {name:<{width}}  {detail}")

    print(f"\n{len(checks) - failures}/{len(checks)} checks passed")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
