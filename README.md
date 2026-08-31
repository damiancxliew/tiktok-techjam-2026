# tjkernels — a shape-dispatched fused Transformer layer for the RTX 3060

TechJam 2026, Problem 3: *Implement a GPU Kernel for a Transformer Layer*.

This repository replaces the `UserOptimizedTransformer` hook in the official
`torch_transformer_benchmark.py` with a set of hand-written Triton kernels and
a per-shape dispatcher. Everything else in the official script — the CLI, the
accuracy rule, the timing loop — is untouched, and an unmodified copy is kept
at `report/torch_transformer_benchmark_original.py` for diffing.

**Results on an RTX 3060 (12 GB, sm_86): all 13 runnable shapes pass with zero
failing elements, geometric-mean speedup 9.34x** (best 28.8x, worst 2.4x).
Full table and methodology in [report/TECH_REPORT.md](report/TECH_REPORT.md);
raw measurement JSON in [report/results/](report/results/).

## The short version

Four observations shaped the design:

1. **The reference is not fp32.** The harness enables TF32 by default
   (`--allow-tf32`, `matmul_precision=high`), so the numbers we are compared
   against already carry ~1e-3 of relative error from a 10-bit mantissa. fp16
   has the *same* 10-bit mantissa, so fp16 tensor-core compute with fp32
   accumulation does not systematically lose ground — and it runs at roughly
   2× TF32 on Ampere consumer silicon.

2. **PyTorch on Windows has no FlashAttention.** `SDPBackend.FLASH_ATTENTION`
   raises for every shape in the test matrix on the official Windows wheels
   (measured in `bench/attn_backends.py`), leaving the cutlass memory-efficient
   kernel at 6–11 TFLOP/s against ~51 TFLOP/s of fp16 peak. So we wrote the
   missing kernel.

3. **Causal attention makes the padding mask a no-op.** The harness builds
   *prefix* masks (`positions < length`). Under causal masking, an invalid key
   `j ≥ len` can never be attended by a valid query `i < len`, so key masking
   provably cannot change any valid output — and invalid rows can never
   contaminate valid ones. Instead of masking after every block, we zero the
   invalid rows once at the very end, and we never pay for an `attn_mask`
   (which would disable the fast attention path anyway).

4. **Small shapes are launch-bound, not compute-bound.** A 4-layer forward is
   ~120 kernel launches; on Windows WDDM that is milliseconds of pure
   submission overhead. Case 2 (batch 1) spends almost all of its baseline time
   there. Whole-forward CUDA Graph capture collapses it to one submission.

## What is actually implemented

| Component | File | What it does |
|---|---|---|
| Causal flash attention | `tjkernels/kernels/attention.py` | FlashAttention-2 style online softmax. Reads Q/K/V in place from the packed `[B,S,3,H,hd]` buffer and writes `[B*S,d]` directly, so neither side needs a layout copy. Skips above-diagonal blocks rather than masking them. |
| Fused residual add + LayerNorm | `tjkernels/kernels/layernorm.py` | One pass computes `res += delta` and its normalization, emitting the compute dtype directly. Row-tiled so each thread gets 16 elements instead of 1. |
| Single-kernel FFN | `tjkernels/kernels/ffn.py` | GEMM → bias → exact-erf GELU → GEMM → bias in one kernel, with the `[T, ffn]` intermediate never reaching global memory. |
| Packed QKV | `tjkernels/engine.py` | One `[d, 3d]` GEMM instead of three, pre-transposed at load time. |
| Deferred residuals | `tjkernels/engine.py` | The FFN output is not added immediately but handed to the *next* block's LayerNorm, so a block costs two residual passes instead of four. |
| CUDA Graph capture | `tjkernels/graphs.py` | Whole-forward capture keyed by shape/dtype/mask-mode, with copy-in and clone-out so any input stays correct. |
| Shape dispatch | `tjkernels/plans.py` | Maps each shape to a tuned `Plan`; `plans.json` is written by the autotuner. |

### The autotuner is accuracy-gated

`bench/autotune.py` does not search for the fastest plan. It searches for **the
fastest plan that leaves every element comfortably inside its error budget**,
verified on seeds the plan was never tuned on. Each element's budget is

```
budget = max(atol, rtol * |reference|)
```

and a plan is admitted only if no element uses more than 85% of its own budget.
That per-element formulation matters. Gating on absolute error alone reported
that case 7 was using 96% of its allowance and looked one unlucky seed from
failing — but the elements carrying that error have `|reference| ≈ 1`, so their
real budget is the relative one, 10× larger, and they were never near the edge.
The first version of this gate "fixed" that phantom risk by escalating the
shape to fp32 and paying 2.4× in latency for nothing. Measuring the right
quantity gave the speed back and kept the guarantee.

The tuner also found a real bug: Triton's `tl.dot` defaults to TF32 for fp32
inputs, whose 10-bit mantissa is *coarser* than the fp16 path's 11 bits — so
the "safe" fp32 escalation was measuring **worse** than the fast path until the
kernels were made to request `input_precision="ieee"`. An autotuner that
assumed higher precision is more accurate would never have caught it.

## Repository layout

```
torch_transformer_benchmark.py      official harness; only the UserOptimizedTransformer
                                    hook is modified (see bench/patch_benchmark.py)
tjkernels/
  engine.py                         weight packing, the fused forward, mask handling
  plans.py                          shape -> plan dispatch + env overrides
  plans.json                        autotuned plans for this GPU
  graphs.py                         whole-forward CUDA Graph capture
  kernels/
    attention.py                    Triton causal flash attention
    layernorm.py                    Triton fused residual-add + LayerNorm
    ffn.py                          Triton single-kernel FFN
    __init__.py                     dispatch to kernels or PyTorch fallbacks
bench/
  run_all.py                        sweep the official matrix -> results JSON + table
  autotune.py                       accuracy-gated plan search -> plans.json
  ablation.py                       per-optimization attribution
  test_correctness.py               paths the official matrix never exercises
  test_flash.py                     flash kernel vs SDPA vs an fp64 reference
  attn_backends.py                  which SDPA backends exist on this platform
  tune_ffn.py                       FFN tile-shape search
  profile_case.py                   per-kernel attribution for one shape
  make_report.py                    renders report/TECH_REPORT.md from measurement JSON
  patch_benchmark.py                wires tjkernels into the official script
report/
  TECH_REPORT.md                    generated, never hand-transcribed
  DEVPOST.md, DEMO_SCRIPT.md        submission material
  results/                          raw measurement JSON and tables
  torch_transformer_benchmark_original.py    pristine copy for diffing
```

## Setup

Requires an NVIDIA GPU (developed on sm_86), Python 3.11, and a recent Triton.

```bash
conda create -n tj26 -y python=3.11
conda activate tj26
pip install torch --index-url https://download.pytorch.org/whl/cu126
pip install triton-windows numpy    # on Linux: pip install triton
python bench/patch_benchmark.py     # wires tjkernels into the official script
```

`bench/patch_benchmark.py` is idempotent and only rewrites the
`UserOptimizedTransformer.forward` body plus one import.

## Reproducing the results

Everything below uses `--seed 1`, as specified.

Single shape, exactly as a judge would run it:

```bash
python torch_transformer_benchmark.py --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128 --causal --seed 1
```

The whole 13-shape matrix:

```bash
python bench/run_all.py --tag main
```

Re-run the accuracy-gated tuning (overwrites `tjkernels/plans.json`):

```bash
python bench/autotune.py --trials 4 --margin 0.7
```

Attribution of each optimization:

```bash
python bench/ablation.py --cases 1,6,13
```

Supporting measurements:

```bash
python bench/attn_backends.py
python bench/test_flash.py
python bench/profile_case.py --case 1
```

## Case 14 is excluded, and here is the arithmetic

The matrix's case 14 (batch 32, seq 100000, d_model 1024, 16 heads) does not
run on a 12 GB card, and not because of our implementation:

- The harness allocates the input before any model runs:
  32 × 100000 × 1024 × 4 B = **13.1 GB** in fp32. In fp16 it is 6.55 GB, and
  the harness holds `x`, `reference` and `candidate` at once — ~19.6 GB.
- The *reference* implementation materializes a score tensor of
  32 × 16 × 100000² ≈ **5.1 × 10¹² elements**. No GPU runs that, so there is
  nothing to be compared against.
- Even a perfect implementation is ~1.3 × 10¹⁵ FLOPs per forward: about 50 s
  per call on this card, ~9 s on an A100.

Running it would need ≥40 GB of VRAM and a memory-safe replacement for the
reference. The other 13 shapes are reported in full.

## Limitations and what we would do next

- **Case 14 is untested**, per the above. The engine's attention kernel is
  sequence-blocked and would stream it correctly given the memory, but we have
  not run it, and we do not claim a number for it.
- **The FFN kernel only fuses while `d_model` and `ffn_dim` are ≤ 256.**
  Case 8 (1024/1024) falls back to cuBLAS + a separate GELU, which is why it
  shows the smallest gain in the table. A split-K tiled version with the
  weights staged through shared memory is the obvious next step.
- **The whole-network megakernel was not built.** Every sequence is
  independent through all layers, so for the `S≤128, d≤128` family the entire
  4-layer model could run as a single kernel launch with one CTA per sequence
  and weights resident in L2 (~780 KB). We ran out of night, not ideas.
- **`plans.json` is tuned for this GPU.** On different hardware the dispatcher
  falls back to heuristics, which are reasonable but not tuned; re-running
  `bench/autotune.py` is the intended remedy.
- **Non-causal shapes take a slower path.** Every official case is causal, so
  the flash kernel assumes it; non-causal input falls back to SDPA. The kernel
  supports it, but it is not the tuned path.
- **bfloat16 models cannot meet this tolerance** -- and neither can anything
  else. `bench/test_bf16_limit.py` shows that running the *reference itself* in
  fp32 and casting the output back to bf16, a strictly more accurate
  implementation, also fails the rule (1432/65536 elements, max abs 3.1e-2),
  because bf16 quantizes the final LayerNorm output in steps of ~0.004 against
  a 0.002 absolute budget. fp32 and fp16 are supported; bf16 is recorded as an
  expected failure with that reasoning rather than papered over.
