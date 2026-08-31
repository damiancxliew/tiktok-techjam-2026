"""Per-kernel attribution for one case, for both baseline and optimized.

    python bench/profile_case.py --case 7
    python bench/profile_case.py --case 11 --top 12

CUDA Graph replay hides the internal kernels from the profiler, so this always
runs with graphs disabled -- it answers "which kernel is expensive", not
"what is the end-to-end latency".
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ["TJ_NO_GRAPH"] = "1"

import torch  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

import torch_transformer_benchmark as bench  # noqa: E402
from bench.shapes import by_index  # noqa: E402


def build(case, dtype, device):
    config = bench.TransformerConfig(
        batch_size=case.batch, seq_len=case.seq, d_model=case.d_model,
        num_heads=case.heads, ffn_dim=case.ffn, num_layers=case.layers,
        causal=case.causal,
    )
    torch.manual_seed(1)
    baseline = bench.BaselineTransformer(config)
    optimized = bench.UserOptimizedTransformer(config)
    bench.copy_model_weights(baseline, optimized, strict=True)
    baseline = baseline.to(device=device, dtype=dtype).eval()
    optimized = optimized.to(device=device, dtype=dtype).eval()
    x, mask = bench.generate_random_case(
        config, device, dtype, seed=1, padding_ratio=0.0, input_scale=1.0
    )
    return baseline, optimized, x, mask


def timed(model, x, mask, iters=30):
    with torch.inference_mode():
        for _ in range(10):
            model(x, mask)
        torch.cuda.synchronize()
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)
        start.record()
        for _ in range(iters):
            model(x, mask)
        end.record()
        torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def kernel_table(model, x, mask, top):
    with torch.inference_mode():
        for _ in range(10):
            model(x, mask)
        torch.cuda.synchronize()
        with profile(activities=[ProfilerActivity.CUDA], record_shapes=False) as prof:
            for _ in range(20):
                model(x, mask)
            torch.cuda.synchronize()
    events = [e for e in prof.key_averages() if e.device_time_total > 0]
    events.sort(key=lambda e: e.device_time_total, reverse=True)
    total = sum(e.device_time_total for e in events)
    rows = []
    for e in events[:top]:
        rows.append(
            f"  {e.device_time_total / 20:9.1f} us  {100 * e.device_time_total / total:5.1f}%"
            f"  x{e.count // 20:<4d} {e.key[:64]}"
        )
    return total / 20, rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--case", type=int, required=True)
    parser.add_argument("--dtype", default="float32")
    parser.add_argument("--top", type=int, default=10)
    parser.add_argument("--skip-baseline", action="store_true")
    args = parser.parse_args()

    case = by_index(args.case)
    device = torch.device("cuda")
    dtype = bench.resolve_dtype(args.dtype)
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    baseline, optimized, x, mask = build(case, dtype, device)
    print(f"case {case.idx}: B={case.batch} S={case.seq} d={case.d_model} "
          f"H={case.heads} F={case.ffn} L={case.layers} ({case.note})")
    print(f"plan: {optimized._tj_engine.plan if hasattr(optimized, '_tj_engine') else 'not built'}")

    if not args.skip_baseline:
        print(f"\nbaseline  end-to-end: {timed(baseline, x, mask):8.3f} ms")
        total, rows = kernel_table(baseline, x, mask, args.top)
        print(f"  gpu kernel total: {total:.1f} us")
        print("\n".join(rows))

    print(f"\noptimized end-to-end: {timed(optimized, x, mask):8.3f} ms  (graphs off)")
    total, rows = kernel_table(optimized, x, mask, args.top)
    print(f"  gpu kernel total: {total:.1f} us")
    print("\n".join(rows))
    print(f"\nplan: {optimized._tj_engine.plan}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
