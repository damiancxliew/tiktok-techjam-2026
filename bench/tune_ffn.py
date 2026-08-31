"""Kernel-level tile search for the fused FFN.

The plan autotuner picks *which* kernel to use; this picks the tile shape
inside one. Sweeps BLOCK_M / num_warps / num_stages against the real
(tokens, d_model, ffn_dim) combinations from the test matrix and reports the
winner, which is what the table in tjkernels/kernels/ffn.py is derived from.

    python bench/tune_ffn.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import List, Tuple

import torch
import triton

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.shapes import ALL_CASES  # noqa: E402
from tjkernels.kernels.ffn import _ffn_fwd, _precision  # noqa: E402

BLOCK_MS = [16, 32, 64, 128]
WARPS = [2, 4, 8]
STAGES = [1, 2, 3]


def run(h, w1, b1, w2, b2, block_m, block_d, block_f, warps, stages):
    tokens, d_model = h.shape
    ffn_dim = w1.shape[1]
    out = torch.empty_like(h)
    _ffn_fwd[(triton.cdiv(tokens, block_m),)](
        h, w1, b1, w2, b2, out,
        tokens, d_model, ffn_dim,
        h.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_F=block_f,
        PREC=_precision(h.dtype),
        num_warps=warps, num_stages=stages,
    )
    return out


def bench(fn, iters=50):
    for _ in range(10):
        fn()
    torch.cuda.synchronize()
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    start.record()
    for _ in range(iters):
        fn()
    end.record()
    torch.cuda.synchronize()
    return start.elapsed_time(end) / iters


def main() -> int:
    torch.manual_seed(1)
    combos: List[Tuple[int, int, int]] = []
    for case in ALL_CASES:
        if case.d_model > 256 or case.ffn > 256:
            continue  # falls back to cuBLAS
        combo = (min(case.tokens, 262144), case.d_model, case.ffn)
        if combo not in combos:
            combos.append(combo)

    print(f"{'tokens,d,ffn':>20} | {'best config':>28} | {'ms':>8} | vs torch")
    print("-" * 78)

    for tokens, d_model, ffn_dim in combos:
        dtype = torch.float16
        h = torch.randn(tokens, d_model, device="cuda", dtype=dtype)
        w1 = torch.randn(d_model, ffn_dim, device="cuda", dtype=dtype) * 0.05
        b1 = torch.randn(ffn_dim, device="cuda", dtype=dtype) * 0.05
        w2 = torch.randn(ffn_dim, d_model, device="cuda", dtype=dtype) * 0.05
        b2 = torch.randn(d_model, device="cuda", dtype=dtype) * 0.05

        block_d = max(16, triton.next_power_of_2(d_model))
        block_f = max(16, triton.next_power_of_2(ffn_dim))

        reference = torch.nn.functional.gelu(
            torch.addmm(b1, h, w1), approximate="none"
        )
        reference = torch.addmm(b2, reference, w2)
        torch_ms = bench(lambda: torch.addmm(
            b2,
            torch.nn.functional.gelu(torch.addmm(b1, h, w1), approximate="none"),
            w2,
        ))

        best = None
        for block_m in BLOCK_MS:
            if block_m > max(16, triton.next_power_of_2(tokens)):
                continue
            for warps in WARPS:
                for stages in STAGES:
                    try:
                        got = run(h, w1, b1, w2, b2, block_m, block_d,
                                  block_f, warps, stages)
                        err = (got.float() - reference.float()).abs().max().item()
                        if err > 5e-2:
                            continue
                        ms = bench(lambda: run(h, w1, b1, w2, b2, block_m,
                                               block_d, block_f, warps, stages))
                    except Exception:
                        continue
                    if best is None or ms < best[0]:
                        best = (ms, block_m, warps, stages)

        if best is None:
            print(f"{str((tokens, d_model, ffn_dim)):>20} | {'no viable config':>28} |")
            continue
        ms, block_m, warps, stages = best
        label = f"BLOCK_M={block_m} warps={warps} stages={stages}"
        print(f"{str((tokens, d_model, ffn_dim)):>20} | {label:>28} | "
              f"{ms:8.4f} | {torch_ms / ms:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
