#!/usr/bin/env python3
"""Stress-test tjkernels (normal/tiny/large/outlier, 40 trials) under both
TF32 settings, using our stress-pattern generator against their model classes."""
import copy
import sys
from dataclasses import dataclass, field
from typing import List, Tuple

sys.path.insert(0, "/local1/stsj/misc/tiktok-techjam-2026")

import torch
import torch_transformer_benchmark as tj


# Inlined (not imported, to avoid a transformer_ablation_benchmark.py name
# collision between the two repos) copy of our stress-pattern generator.
def generate_random_case(config, device, dtype, seed, padding_ratio, input_scale, pattern="normal"):
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)
    x = torch.randn(config.batch_size, config.seq_len, config.d_model,
                     generator=generator, device=device, dtype=dtype)
    if pattern == "normal":
        x = x * input_scale
    elif pattern == "tiny":
        x = x * (0.01 * input_scale)
    elif pattern == "large":
        x = x * (3.0 * input_scale)
    elif pattern == "outlier":
        outlier_values = torch.randn(x.shape, generator=generator, device=device, dtype=dtype) * 10.0
        outlier_mask = torch.rand(x.shape, generator=generator, device=device) < 0.001
        x = (x + outlier_values * outlier_mask.to(dtype=dtype)) * input_scale
    else:
        raise ValueError(f"unknown pattern: {pattern}")
    valid_token_mask = torch.ones(config.batch_size, config.seq_len, device=device, dtype=torch.bool)
    return x, valid_token_mask


@dataclass
class AccuracySummary:
    passed: bool = True
    total_elements: int = 0
    failed_elements: int = 0
    max_abs_error: float = 0.0
    max_relative_error: float = 0.0


def update_accuracy_summary(summary, reference, candidate, rtol, atol):
    ref = reference.detach().float()
    opt = candidate.detach().float()
    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()
    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)
    failed = int((~passed_mask).sum().item())
    summary.passed = summary.passed and (failed == 0)
    summary.total_elements += reference.numel()
    summary.failed_elements += failed
    summary.max_abs_error = max(summary.max_abs_error, float(abs_error.max().item()))
    denom = ref.abs().clamp_min(1e-12)
    summary.max_relative_error = max(summary.max_relative_error, float((abs_error / denom).max().item()))

device = torch.device("cuda")

configs = {
    1: tj.TransformerConfig(64, 128, 128, 4, 128, 4, True),
    6: tj.TransformerConfig(10000, 128, 128, 4, 128, 4, True),
    8: tj.TransformerConfig(64, 128, 1024, 4, 1024, 4, True),
    13: tj.TransformerConfig(64, 1024, 128, 4, 128, 4, True),
}

for allow_tf32 in (True, False):
    torch.backends.cuda.matmul.allow_tf32 = allow_tf32
    torch.backends.cudnn.allow_tf32 = allow_tf32
    print(f"\n{'='*70}\nallow_tf32={allow_tf32}\n{'='*70}")
    for shape_id, config in configs.items():
        baseline = tj.BaselineTransformer(config).to(device=device, dtype=torch.float32).eval()
        optimized = tj.UserOptimizedTransformer(config).to(device=device, dtype=torch.float32).eval()
        tj.copy_model_weights(baseline, optimized, strict=True)

        summary = AccuracySummary()
        patterns = ("normal", "tiny", "large", "outlier")
        with torch.inference_mode():
            for p_idx, pattern in enumerate(patterns):
                for trial in range(10):
                    seed = 1234 + p_idx * 100_000 + trial
                    x, mask = generate_random_case(
                        config=config, device=device, dtype=torch.float32,
                        seed=seed, padding_ratio=0.0, input_scale=1.0, pattern=pattern,
                    )
                    ref = baseline(x, mask)
                    out = optimized(x, mask)
                    update_accuracy_summary(summary, ref, out, rtol=0.02, atol=0.002)
        status = "PASS" if summary.passed else "FAIL"
        print(f"shape {shape_id:>2} [{status}]  max_abs={summary.max_abs_error:.6g}  "
              f"max_rel={summary.max_relative_error:.6g}  failed={summary.failed_elements}/{summary.total_elements}")
