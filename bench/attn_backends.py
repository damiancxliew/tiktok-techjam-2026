"""Which SDPA backend is fastest for each official attention shape?

PyTorch picks a backend by its own heuristics; this measures all of them on
the exact [B, H, S, head_dim] tensors the engine produces (including the
strided packed-QKV views, which is what actually gets fed to SDPA).
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from bench.shapes import ALL_CASES  # noqa: E402

BACKENDS = {
    "flash": SDPBackend.FLASH_ATTENTION,
    "mem_eff": SDPBackend.EFFICIENT_ATTENTION,
    "math": SDPBackend.MATH,
}


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


def main() -> int:
    torch.manual_seed(1)
    seen = set()
    print(f"{'shape (B,H,S,hd)':>26} | " + " | ".join(f"{k:>9}" for k in BACKENDS)
          + " |   default | best")
    print("-" * 92)

    for case in ALL_CASES:
        head_dim = case.d_model // case.heads
        key = (case.batch, case.heads, case.seq, head_dim)
        if key in seen:
            continue
        seen.add(key)

        # Reproduce the exact layout the engine feeds to SDPA: one packed
        # [T, 3d] buffer, viewed as three strided [B, H, S, hd] tensors.
        qkv = torch.randn(
            case.batch, case.seq, 3, case.heads, head_dim,
            device="cuda", dtype=torch.float16,
        )
        q = qkv[:, :, 0].transpose(1, 2)
        k = qkv[:, :, 1].transpose(1, 2)
        v = qkv[:, :, 2].transpose(1, 2)

        row = {}
        for name, backend in BACKENDS.items():
            try:
                with sdpa_kernel(backend):
                    row[name] = bench(
                        lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
                    )
            except Exception:
                row[name] = float("nan")
        default = bench(
            lambda: F.scaled_dot_product_attention(q, k, v, is_causal=True)
        )
        best = min((v_, k_) for k_, v_ in row.items() if v_ == v_)[1]
        cells = " | ".join(f"{row[k_]:9.4f}" for k_ in BACKENDS)
        print(f"{str(key):>26} | {cells} | {default:9.4f} | {best}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
