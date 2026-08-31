# Technical Report — TechJam 2026 Problem 3

**GPU kernel implementation for a Transformer layer**

Generated 2026-08-31 19:12 from
`main-20260831-184942.json`. Every number below is read from measurement JSON, not
transcribed.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 3060 (sm_86, 12 GB, 28 SMs) |
| Driver | 591.86 |
| CPU | 12th Gen Intel(R) Core(TM) i7-12700F |
| OS | Windows 11 (build 22631) |
| Python | 3.11.16 |
| PyTorch | 2.13.0+cu126 (CUDA 12.6) |
| Triton | 3.8.0 |

Run configuration: `--seed 1`,
`--dtype float32`, TF32 enabled (harness default),
accuracy rule `abs_err <= 2e-3 OR rel_err <= 2e-2` per element.

## 2. Results

| # | B | S | d_model | H | ffn | L | accuracy | max_abs | reference (ms) | ours (ms) | speedup |
|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 128 | 4 | PASS | 1.30e-03 | 6.006 | 0.881 | **6.82x** |
| 2 | 1 | 128 | 128 | 4 | 128 | 4 | PASS | 1.14e-03 | 2.508 | 0.087 | **28.82x** |
| 3 | 4 | 128 | 128 | 4 | 128 | 4 | PASS | 9.83e-04 | 2.468 | 0.132 | **18.69x** |
| 4 | 16 | 128 | 128 | 4 | 128 | 4 | PASS | 1.14e-03 | 2.245 | 0.263 | **8.53x** |
| 5 | 128 | 128 | 128 | 4 | 128 | 4 | PASS | 1.43e-03 | 12.103 | 1.576 | **7.68x** |
| 6 | 10000 | 128 | 128 | 4 | 128 | 4 | PASS | 1.67e-03 | 962.897 | 123.562 | **7.79x** |
| 7 | 64 | 128 | 32 | 4 | 32 | 4 | PASS | 2.33e-03 | 4.524 | 0.279 | **16.24x** |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 4 | PASS | 1.56e-03 | 53.136 | 21.850 | **2.43x** |
| 9 | 64 | 128 | 128 | 1 | 128 | 4 | PASS | 1.31e-03 | 3.434 | 0.920 | **3.73x** |
| 10 | 64 | 128 | 128 | 2 | 128 | 4 | PASS | 1.27e-03 | 4.591 | 0.891 | **5.15x** |
| 11 | 64 | 128 | 128 | 16 | 128 | 4 | PASS | 1.22e-03 | 14.797 | 1.046 | **14.14x** |
| 12 | 64 | 32 | 128 | 4 | 128 | 4 | PASS | 1.15e-03 | 2.215 | 0.257 | **8.62x** |
| 13 | 64 | 1024 | 128 | 4 | 128 | 4 | PASS | 1.30e-03 | 210.621 | 8.907 | **23.65x** |

**13/13 shapes pass** the accuracy rule with zero failing
elements. Geometric-mean speedup across the 13 benchmarked shapes:
**9.34x** (best 28.82x, worst 2.43x).

### 2.1 Held-out seed

The plans in `plans.json` were tuned at seed 1, so they were also validated at a seed no tuning ever saw. At `--seed 99`, **13/13 shapes pass** with a geometric-mean speedup of 9.27x -- the plans generalize rather than fitting the seed they were tuned on.

## 3. What was optimized, and why

### 3.1 The reference is a TF32 reference

The harness enables TF32 by default. TF32 and fp16 have the same 10-bit
mantissa, so fp16 tensor-core compute with fp32 accumulation is not
systematically worse than what we are being compared against — while running
about 2x faster on Ampere consumer parts. Measured directly: with our fp32
plan the deviation from the reference is 1.01e-3, *identical* to running the
whole model in eager fp32, which means that residual error is the reference's
own TF32 rounding rather than ours.

### 3.2 PyTorch on Windows ships without FlashAttention

`bench/attn_backends.py` measures every SDPA backend on every attention shape
in the matrix. `FLASH_ATTENTION` is unavailable for all of them on the official
Windows wheels, leaving the cutlass memory-efficient kernel at 6-11 TFLOP/s
against ~51 TFLOP/s of fp16 tensor-core peak. Our Triton kernel
(`tjkernels/kernels/attention.py`) is a FlashAttention-2 style online-softmax
implementation that additionally reads Q/K/V in place from the packed QKV
buffer and writes `[B*S, d]` directly, removing the layout copies on both
sides. `bench/test_flash.py` verifies it against an fp64 reference and measures
1.5-2.2x over SDPA on the large shapes at identical accuracy.

### 3.3 Causal attention makes the padding mask a no-op

The harness generates prefix masks. Under causal masking an invalid key
`j >= len` is never visible to a valid query `i < len`, so key masking cannot
change any valid output, and invalid rows cannot contaminate valid ones. We
therefore never build an `attn_mask` (which would disable the fast attention
path) and zero the invalid rows once at the end instead of after every block.
Mask classification costs one sync per *distinct mask tensor*, memoized on
tensor identity, so the timing loop never syncs.

### 3.4 Memory traffic, not FLOPs, is the limit at these shapes

Profiling (`bench/profile_case.py`) showed the fused norms sitting at ~94% of
achievable bandwidth while the GEMMs ran well below tensor-core peak — these
shapes are bandwidth-bound. Two structural changes followed: the FFN
intermediate never reaches global memory, and residual additions are deferred
into the next LayerNorm so a block does two residual passes instead of four.

### 3.5 Case 8 is close to its precision-bound ceiling

Case 8 (d_model = ffn = 1024) shows the smallest gain, and that is the honest
answer rather than a gap to close. It is dominated by three large GEMMs per
layer, which the reference already runs on tensor cores in TF32. Our advantage
there is the fp16-vs-TF32 rate difference on Ampere consumer silicon, which is
2x, plus what we save by not materializing the attention scores. A speedup of
about 2.4x is therefore roughly what the hardware allows for this shape without
dropping below the error budget. The fused FFN also cannot help here: at
1024x1024 the two weight matrices do not fit in 96 KB of shared memory, so the
dispatcher correctly routes it to cuBLAS.

### 3.6 Small shapes are launch-bound

A 4-layer forward is ~120 kernel launches, and Windows WDDM charges several
microseconds each. Whole-forward CUDA Graph capture collapses that to one
submission; correctness for arbitrary inputs is preserved by copying into the
captured static buffer and cloning the output back out.

## 4. Per-shape plans chosen by the accuracy-gated autotuner

The autotuner admits a plan only if no element uses more than
85% of its own error budget
(`max(atol, rtol * |reference|)`), measured over
6 input draws including two
held-out seeds, then picks the fastest admissible plan by coordinate descent.

Two of its decisions are worth calling out. Case 7 (d_model 32) carries the
largest absolute error in the matrix, 2.3e-3, which is *above* atol -- yet it
is admitted, because the elements carrying that error have |reference| near 1
and are using only 70% of their relative budget. Conversely an fp16 residual
stream was rejected on every shape: it looks 10% faster and lands at 124% of
budget with real failing elements.

| # | compute | residual | attention | FFN | norm | CUDA graph | max_abs | plans explored |
|---:|:--|:--|:--|:--|:--|:--|---:|---:|
| 1 | float16 | float32 | triton | triton | triton | yes | 1.33e-03 | 7 |
| 2 | float16 | float32 | triton | torch | triton | yes | 1.14e-03 | 7 |
| 3 | float16 | float32 | triton | triton | triton | yes | 1.23e-03 | 7 |
| 4 | float16 | float32 | triton | triton | triton | yes | 1.33e-03 | 7 |
| 5 | float16 | float32 | triton | triton | triton | no | 1.33e-03 | 7 |
| 6 | float16 | float32 | triton | triton | triton | no | 1.67e-03 | 7 |
| 7 | float16 | float32 | triton | torch | triton | yes | 2.33e-03 | 7 |
| 8 | float16 | float32 | triton | torch | triton | yes | 1.56e-03 | 6 |
| 9 | float16 | float32 | triton | triton | triton | yes | 1.31e-03 | 7 |
| 10 | float16 | float32 | triton | triton | triton | yes | 1.27e-03 | 7 |
| 11 | float16 | float32 | triton | triton | triton | yes | 1.43e-03 | 7 |
| 12 | float16 | float32 | triton | triton | triton | yes | 1.33e-03 | 7 |
| 13 | float16 | float32 | triton | triton | triton | no | 1.52e-03 | 7 |

## 5. Optimization attribution

### Cumulative optimization ladder (speedup vs the reference)

| variant | case 2 | case 1 | case 6 | case 13 |
|---|---:|---:|---:|---:|
| 0 baseline (control) | 1.09x | 1.01x | 0.99x | 1.00x |
| 1 packed QKV + SDPA + fused residual | 1.90x | 3.43x | 3.77x | 12.36x |
| 2 + Triton norm/FFN kernels | 2.48x | 6.48x | 7.23x | 18.51x |
| 3 + Triton causal flash attention | 2.83x | 7.04x | 7.76x | 23.79x |
| 4 + CUDA Graph capture (full stack) | 17.96x | 6.83x | 7.76x | 22.75x |

### Leave-one-out (full stack minus one component)

| variant | case 2 | case 1 | case 6 | case 13 |
|---|---:|---:|---:|---:|
| full stack | 21.63x | 6.76x | 7.78x | 22.77x |
| without CUDA Graphs | 3.20x | 7.02x | 7.76x | 23.70x |
| without Triton flash attn | 21.91x | 6.24x | 7.18x | 17.93x |
| without fused FFN | 28.55x | 6.36x | 7.17x | 21.84x |
| without fused add+LayerNorm | 17.45x | 3.77x | 4.03x | 14.50x |
| without fp16 compute | 14.04x | 3.13x | 3.23x | 6.35x |


## 6. Where the accuracy rule itself runs out

The tolerance is achievable in fp32 and fp16, and *not* achievable in bfloat16
by any implementation that is not bit-identical to the reference.
`bench/test_bf16_limit.py` shows this without involving our kernels at all: it
runs the reference model in fp32 and casts the output back to bf16 -- a
strictly more accurate implementation of the same network -- and compares that
against the bf16 reference. It fails, with 1432 of 65536 elements outside the
budget and a max absolute error of 3.1e-2, because bf16 quantizes the output of
the final LayerNorm in steps of about 0.004 while the rule allows 0.002 near
zero. The same control passes comfortably in fp16 (3.9e-3, zero failures).

We therefore support fp32 (the harness default, and what the test matrix uses)
and fp16, and document bf16 as outside the usable range of the rule rather than
claiming a pass we cannot honestly get. `bench/test_correctness.py` records it
as an expected failure with that reasoning attached.

## 7. Case 14

Excluded, with arithmetic rather than excuses: the harness allocates
32 x 100000 x 1024 x 4 B = 13.1 GB for the input before any model runs, and the
reference materializes a 32 x 16 x 100000^2 = 5.1e12-element score tensor. It
needs >=40 GB of VRAM and a memory-safe replacement for the reference; neither
exists on a 12 GB RTX 3060. No number is claimed for it.

## 8. AI-assisted workflow

This solution was built in a single overnight session with Claude Code driving
the profile-hypothesize-measure loop: read the harness to find what the
tolerance actually permits, profile to find the dominant kernel, propose a
kernel variant, measure it, keep it only if it was both faster and inside the
error budget. Two findings came directly out of that loop and neither was
guessed up front: that the Windows PyTorch build has no flash backend at all,
and that Triton's `tl.dot` silently uses TF32 for fp32 inputs, which made the
"safe" fp32 fallback less accurate than fp16 until `input_precision="ieee"`
was requested explicitly.
