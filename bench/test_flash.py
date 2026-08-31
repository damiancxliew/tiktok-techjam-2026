"""Correctness + speed of the Triton flash kernel vs PyTorch SDPA.

Checks against an fp64 reference (so both implementations are measured against
the true value, not against each other) on every attention shape in the matrix.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.shapes import ALL_CASES  # noqa: E402
from tjkernels.kernels.attention import flash_attention  # noqa: E402


def bench(fn, iters=30):
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


def reference(qkv, heads, causal=True):
    batch, seq, _, _, hd = qkv.shape
    q = qkv[:, :, 0].transpose(1, 2).double()
    k = qkv[:, :, 1].transpose(1, 2).double()
    v = qkv[:, :, 2].transpose(1, 2).double()
    scores = (q @ k.transpose(-2, -1)) * (hd ** -0.5)
    if causal:
        scores = scores.masked_fill(
            torch.ones(seq, seq, device=qkv.device, dtype=torch.bool).triu(1),
            float("-inf"),
        )
    ctx = torch.softmax(scores, dim=-1) @ v
    return ctx.transpose(1, 2).reshape(batch * seq, heads * hd)


def main() -> int:
    torch.manual_seed(1)
    print(f"{'B,H,S,hd':>22} | {'triton ms':>9} | {'sdpa ms':>9} | {'gain':>6} | "
          f"{'triton err':>10} | {'sdpa err':>9}")
    print("-" * 82)

    seen = set()
    gains = []
    for case in ALL_CASES:
        hd = case.d_model // case.heads
        key = (case.batch, case.heads, case.seq, hd)
        if key in seen:
            continue
        seen.add(key)
        # Case 6 at full batch just measures allocator pressure; scale it down.
        batch = min(case.batch, 512)

        qkv = torch.randn(
            batch, case.seq, 3, case.heads, hd, device="cuda", dtype=torch.float16
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        def sdpa():
            ctx = F.scaled_dot_product_attention(q, k, v, is_causal=True)
            return ctx.transpose(1, 2).reshape(batch * case.seq, case.d_model)

        got = flash_attention(qkv, case.heads, causal=True)
        ref = reference(qkv, case.heads)
        sdpa_out = sdpa()

        err_triton = (got.double() - ref).abs().max().item()
        err_sdpa = (sdpa_out.double() - ref).abs().max().item()

        t_triton = bench(lambda: flash_attention(qkv, case.heads, causal=True))
        t_sdpa = bench(sdpa)
        gains.append(t_sdpa / t_triton)
        print(f"{str((batch,) + key[1:]):>22} | {t_triton:9.4f} | {t_sdpa:9.4f} | "
              f"{t_sdpa / t_triton:5.2f}x | {err_triton:10.2e} | {err_sdpa:9.2e}")

    print(f"\nmean gain over SDPA: {sum(gains) / len(gains):.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
