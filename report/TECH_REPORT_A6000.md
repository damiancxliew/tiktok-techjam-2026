# Technical Report — TechJam 2026 Problem 3 (RTX A6000 validation)

**GPU kernel implementation for a Transformer layer — same code, a second GPU**

Generated 2026-08-31 19:40 from
`a6000_comparison-20260831-182245.json`. Every number below is read from measurement JSON, not
transcribed, via `bench/make_report.py` (this session fixed two small bugs in that
script exposed by a multi-GPU Linux host it wasn't originally run on: a
malformed multi-line driver-version field, and a CPU name that fell back to
the bare "x86_64" architecture string instead of reading `/proc/cpuinfo`).

**About this report.** This is a second, independent validation of the exact
same `tjkernels/` implementation and `plans.json` used in the original
`report/TECH_REPORT.md` (RTX 3060, Windows) — no code or plan changes, run on
different hardware to check whether the design holds up off its original
target platform. §2 gives the results; §2.2 compares the two GPUs directly;
§9 adds an adversarial stress-test (not part of the original report, or of
the official grading script) applied to this same implementation. The prose
in §3 and §4 describing *why* each optimization was chosen is carried over
from the original report where it explains a design decision (e.g. §3.2's
"PyTorch on Windows ships without FlashAttention" describes the platform that
motivated writing a custom attention kernel in the first place) — it is
historical reasoning, not a claim about the platform this specific run used.

## 1. Environment

| | |
|---|---|
| GPU | NVIDIA RTX A6000 (sm_86, 47 GB, 84 SMs) |
| Driver | 595.80 |
| CPU | AMD EPYC 7763 64-Core Processor |
| OS | Linux 5.14.0-611.55.1.el9_7.0.3.x86_64 |
| Python | 3.10.20 |
| PyTorch | 2.5.1+cu121 (CUDA 12.1) |
| Triton | 3.1.0 |

Run configuration: `--seed 1`,
`--dtype float32`, TF32 enabled (harness default),
accuracy rule `abs_err <= 2e-3 OR rel_err <= 2e-2` per element.

## 2. Results

| # | B | S | d_model | H | ffn | L | accuracy | max_abs | reference (ms) | ours (ms) | speedup |
|---:|---:|---:|---:|---:|---:|---:|:--|---:|---:|---:|---:|
| 1 | 64 | 128 | 128 | 4 | 128 | 4 | PASS | 1.42e-03 | 2.728 | 0.386 | **7.07x** |
| 2 | 1 | 128 | 128 | 4 | 128 | 4 | PASS | 1.10e-03 | 2.711 | 0.104 | **25.95x** |
| 3 | 4 | 128 | 128 | 4 | 128 | 4 | PASS | 1.02e-03 | 2.814 | 0.110 | **25.69x** |
| 4 | 16 | 128 | 128 | 4 | 128 | 4 | PASS | 1.38e-03 | 2.722 | 0.160 | **17.04x** |
| 5 | 128 | 128 | 128 | 4 | 128 | 4 | PASS | 1.42e-03 | 5.575 | 1.071 | **5.21x** |
| 6 | 10000 | 128 | 128 | 4 | 128 | 4 | PASS | 1.82e-03 | 402.554 | 50.896 | **7.91x** |
| 7 | 64 | 128 | 32 | 4 | 32 | 4 | PASS | 1.87e-03 | 2.736 | 0.131 | **20.88x** |
| 8 | 64 | 128 | 1024 | 4 | 1024 | 4 | PASS | 1.50e-03 | 16.288 | 6.667 | **2.44x** |
| 9 | 64 | 128 | 128 | 1 | 128 | 4 | PASS | 1.42e-03 | 2.495 | 0.405 | **6.17x** |
| 10 | 64 | 128 | 128 | 2 | 128 | 4 | PASS | 1.24e-03 | 2.833 | 0.390 | **7.26x** |
| 11 | 64 | 128 | 128 | 16 | 128 | 4 | PASS | 1.30e-03 | 6.874 | 0.474 | **14.50x** |
| 12 | 64 | 32 | 128 | 4 | 128 | 4 | PASS | 1.38e-03 | 2.736 | 0.153 | **17.93x** |
| 13 | 64 | 1024 | 128 | 4 | 128 | 4 | PASS | 1.33e-03 | 98.464 | 3.468 | **28.39x** |

**13/13 shapes pass** the accuracy rule with zero failing
elements. Geometric-mean speedup across the 13 benchmarked shapes:
**11.41x** (best 28.39x, worst 2.44x).

### 2.1 Held-out seed

The plans in `plans.json` were tuned at seed 1, so they were also validated at a seed no tuning ever saw. At `--seed 99`, **13/13 shapes pass** with a geometric-mean speedup of 11.57x -- the plans generalize rather than fitting the seed they were tuned on.

### 2.2 Cross-GPU comparison: RTX 3060 (original) vs. RTX A6000 (this report)

Same `tjkernels/plans.json`, same seed (1), no re-tuning for the new GPU. Both are Ampere, sm_86 -- the Triton kernels needed no porting.

| # | 3060 ref (ms) | 3060 ours (ms) | 3060 speedup | A6000 ref (ms) | A6000 ours (ms) | A6000 speedup | A6000 ours / 3060 ours |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | 6.006 | 0.881 | 6.82x | 2.728 | 0.386 | 7.07x | 2.28x |
| 2 | 2.508 | 0.087 | 28.82x | 2.711 | 0.104 | 25.95x | 0.83x |
| 3 | 2.468 | 0.132 | 18.69x | 2.814 | 0.110 | 25.69x | 1.21x |
| 4 | 2.245 | 0.263 | 8.53x | 2.722 | 0.160 | 17.04x | 1.65x |
| 5 | 12.103 | 1.576 | 7.68x | 5.575 | 1.071 | 5.21x | 1.47x |
| 6 | 962.897 | 123.562 | 7.79x | 402.554 | 50.896 | 7.91x | 2.43x |
| 7 | 4.524 | 0.279 | 16.24x | 2.736 | 0.131 | 20.88x | 2.12x |
| 8 | 53.136 | 21.850 | 2.43x | 16.288 | 6.667 | 2.44x | 3.28x |
| 9 | 3.434 | 0.920 | 3.73x | 2.495 | 0.405 | 6.17x | 2.27x |
| 10 | 4.591 | 0.891 | 5.15x | 2.833 | 0.390 | 7.26x | 2.28x |
| 11 | 14.797 | 1.046 | 14.14x | 6.874 | 0.474 | 14.50x | 2.21x |
| 12 | 2.215 | 0.257 | 8.62x | 2.736 | 0.153 | 17.93x | 1.68x |
| 13 | 210.621 | 8.907 | 23.65x | 98.464 | 3.468 | 28.39x | 2.57x |

The relative speedup (last three columns of §2) is similar on both cards -- 9.34x geomean on the 3060, 11.41x on the A6000 -- because the same design decisions (fp16 compute, fused kernels, mask elimination, graph capture) address the same bottlenecks regardless of which Ampere part runs them. The absolute latency gap is not uniform, though: the A6000's larger SM count and memory bandwidth show up as a 1.2-3.3x wall-clock advantage on every compute- or bandwidth-bound shape, but **shape 2 (batch=1) actually runs faster in absolute terms on the 3060** (0.087ms vs. 0.104ms). At batch=1 the workload is pure launch/dispatch overhead, not compute -- there's nothing for the A6000's extra SMs to parallelize across a single sequence, and its overhead characteristics for a near-empty kernel graph aren't automatically better than the smaller card's. The gap here (17 microseconds) is small enough that we would not treat it as conclusive without repeated-run variance data we did not collect for either GPU -- but it is at minimum a caution against assuming a bigger GPU is strictly faster on every shape without measuring, since every other shape in the matrix does show the A6000 ahead.

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

| variant | case 1 | case 6 | case 13 |
|---|---:|---:|---:|
| 0 baseline (control) | 0.99x | 1.00x | 1.00x |
| 1 packed QKV + SDPA + fused residual | 2.57x | 3.84x | 14.78x |
| 2 + Triton norm/FFN kernels | 2.35x | 6.76x | 23.23x |
| 3 + Triton causal flash attention | 2.59x | 7.91x | 29.34x |
| 4 + CUDA Graph capture (full stack) | 7.23x | 7.91x | 27.46x |

### Leave-one-out (full stack minus one component)

| variant | case 1 | case 6 | case 13 |
|---|---:|---:|---:|
| full stack | 7.19x | 7.91x | 27.10x |
| without CUDA Graphs | 2.58x | 7.91x | 28.75x |
| without Triton flash attn | 6.32x | 6.76x | 22.30x |
| without fused FFN | 6.66x | 7.04x | 24.93x |
| without fused add+LayerNorm | 4.10x | 4.45x | 17.05x |
| without fp16 compute | 3.45x | 3.55x | 7.83x |


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

## 7. Beyond the harness's own accuracy check: an adversarial stress test

Neither the original report nor the official grading script's own `run_accuracy_tests` exercises anything beyond `normal`-pattern Gaussian inputs. As part of a broader comparison exercise (see `techjam-transformer-gpu-kernel` — a separate, independent submission for this same challenge, benchmarked against this one on this same A6000), we additionally stress-tested *this* implementation with `tiny` (0.01x), `large` (3x), and `outlier` (0.1% of elements receive an injected `N(0,10)` spike) input distributions, 10 trials each, 40 trials total per shape, under both TF32 settings (`stress_test_theirs.py`, not part of this repo's own test suite):

| Shape | TF32-on (this repo's design target) | TF32-off |
|---|---|---|
| 1 | FAIL (271/41,943,040) | FAIL (7,056/41,943,040) |
| 6 | FAIL (37,288/6.55B) | FAIL (1,013,455/6.55B) |
| 8 | FAIL (9,577/335.5M) | FAIL (52,062/335.5M) |
| 13 | FAIL (644/335.5M) | FAIL (9,821/335.5M) |

**This design does not survive that harsher test, under either TF32 setting** -- forcing TF32 off makes it measurably worse (1,013,455 failed elements at shape 6, vs. 37,288 with TF32 on as designed), confirming §3.1's reasoning is genuinely load-bearing: this implementation's correctness depends on TF32 staying enabled, and even then only holds for the specific input distribution the official script actually tests. This is not a defect unique to this implementation -- the independent submission we compared against shows the same failure pattern under the same test, at every precision level we tried, including full fp32 -- but it means "13/13 PASS" in §2 should be read precisely as "passes the literal, as-provided grading script," not as a general robustness claim. We think this is worth stating plainly rather than leaving implicit, since it is the one place either submission's confidence could be overstated.

## 8. Case 14

Excluded, with arithmetic rather than excuses: the harness allocates
32 x 100000 x 1024 x 4 B = 13.1 GB for the input before any model runs, and the
reference materializes a 32 x 16 x 100000^2 = 5.1e12-element score tensor. It
needs >=40 GB of VRAM and a memory-safe replacement for the reference; neither
exists on a 12 GB RTX 3060. No number is claimed for it.

**On the A6000 (48 GB), the first constraint is no longer binding** -- ~19.6 GB
for `x`/reference/candidate held simultaneously fits comfortably -- but the
second is unchanged: the reference's `32 x 16 x 100000^2` score tensor is
~5.1e12 elements regardless of which GPU tries to hold it, i.e. several TB even
in fp16, so *no* GPU runs the unmodified reference at this shape. We did not
attempt case 14 with this repo's kernels (that would require a memory-safe
reference and a chunked/streamed variant of `tjkernels/kernels/attention.py`,
neither of which exist yet per §"Limitations" in the main README); a
memory-safe cross-machine attention implementation with a different design
(batch-microbatched against an SDPA-based reference rather than the exact
manual-attention one) validated this shape on the same A6000 in the separate
comparison submission referenced in §7.

## 9. AI-assisted workflow

This solution was built in a single overnight session with Claude Code driving
the profile-hypothesize-measure loop: read the harness to find what the
tolerance actually permits, profile to find the dominant kernel, propose a
kernel variant, measure it, keep it only if it was both faster and inside the
error budget. Two findings came directly out of that loop and neither was
guessed up front: that the Windows PyTorch build has no flash backend at all,
and that Triton's `tl.dot` silently uses TF32 for fp32 inputs, which made the
"safe" fp32 fallback less accurate than fp16 until `input_precision="ieee"`
was requested explicitly.
