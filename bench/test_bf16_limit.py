"""Is the accuracy rule achievable at all in bfloat16?

Our optimized path fails the tolerance for bf16 models. Before calling that a
bug in our kernels, this asks whether *any* non-bit-identical implementation
could pass: it compares the reference against itself, run in a way that is
mathematically equivalent but rounds differently.

Two controls:

  fp32-then-cast   the same reference model in fp32, output cast to bf16 --
                   i.e. a strictly *more accurate* implementation
  reordered        the same reference in bf16, with the batch split in two and
                   concatenated back -- identical math, different kernel tiling

If those controls also fail, the tolerance is not achievable in bf16 by any
implementation that is not bit-identical to the reference, and the honest
conclusion is that bf16 is outside the usable range of the accuracy rule --
not that our kernels are wrong.

    python bench/test_bf16_limit.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch_transformer_benchmark as bench  # noqa: E402

ATOL, RTOL = 2e-3, 2e-2


def report(label: str, ref: torch.Tensor, got: torch.Tensor) -> None:
    result = bench.compare_outputs(ref, got, rtol=RTOL, atol=ATOL)
    status = "PASS" if result.passed else "FAIL"
    print(f"  {status}  {label:<44} max_abs={result.max_abs_error:.2e}  "
          f"failed={result.failed_elements}/{result.total_elements}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-dim", type=int, default=128)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--causal", action="store_true", default=True)
    parser.add_argument("--no-causal", dest="causal", action="store_false")
    args = parser.parse_args()

    device = torch.device("cuda")
    torch.set_float32_matmul_precision("high")
    torch.backends.cuda.matmul.allow_tf32 = True

    config = bench.TransformerConfig(
        batch_size=args.batch_size, seq_len=args.seq_len, d_model=args.d_model,
        num_heads=args.heads, ffn_dim=args.ffn_dim, num_layers=args.layers,
        causal=args.causal,
    )
    print(f"shape: {config}")

    for dtype, name in ((torch.bfloat16, "bfloat16"), (torch.float16, "float16")):
        torch.manual_seed(1)
        model = bench.BaselineTransformer(config)
        model_lp = model.to(device=device, dtype=dtype).eval()
        model_fp32 = bench.BaselineTransformer(config)
        model_fp32.load_state_dict(
            {k: v.float() for k, v in model.state_dict().items()}
        )
        model_fp32 = model_fp32.to(device=device, dtype=torch.float32).eval()

        x, mask = bench.generate_random_case(
            config, device, dtype, seed=1, padding_ratio=0.0, input_scale=1.0
        )

        print(f"\n{name}: reference compared against equivalent references")
        with torch.inference_mode():
            reference = model_lp(x, mask)

            # Control 1: a strictly more accurate implementation.
            more_accurate = model_fp32(x.float(), mask).to(dtype)
            report("same model in fp32, output cast back", reference, more_accurate)

            # Control 2: identical math, different kernel tiling.
            half = config.batch_size // 2
            split = torch.cat(
                [model_lp(x[:half], mask[:half]), model_lp(x[half:], mask[half:])],
                dim=0,
            )
            report("same model, batch split and concatenated", reference, split)

    print("\nIf the controls fail, the tolerance is unreachable in that dtype for")
    print("any implementation that is not bit-identical to the reference.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
