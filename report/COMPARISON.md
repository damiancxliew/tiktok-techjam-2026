# Reading the ablation results, and how they compare to ours

All runs below: shape `batch=8, seq=128, d_model=512, heads=8, ffn=2048,
layers=6`, `--seed 1`, RTX 3060.

## 1. The two scripts answer different questions

They look similar and print similar tables, but they are not measuring the same
thing, and the numbers are not interchangeable.

| | `transformer_ablation_benchmark.py` (friend's) | `torch_transformer_benchmark.py` (official) |
|---|---|---|
| Question | "Of these three specific PyTorch tweaks, how much is each one worth?" | "How much faster is the submission than the reference?" |
| Compares | eight variants against **variant A**, which it builds itself | our implementation against the **official reference** |
| Result | an attribution table | the number the competition scores |

The good news: **the two baselines are the same model.** Variant A measured
7.38 ms and the official reference measured 7.35 ms on the identical shape, a
0.4% difference. So the two scripts' numbers can be placed side by side.

The three things the ablation toggles are:

- **packed QKV** — do the Q, K and V projections as one matrix multiply instead
  of three.
- **SDPA** — call PyTorch's built-in attention instead of writing out
  `softmax(QK^T)V` by hand.
- **mask skip** — notice when the padding mask marks everything valid, and skip
  the masking work entirely.

Those are all *rearrangements of PyTorch calls*. None of them writes a new GPU
kernel. That is the ceiling being measured, and it is why the total comes out
around 1.3x rather than several times.

## 2. How to read the ablation table

```
Var  Name                      Accuracy    Median ms   Speedup  Latency Δ    Token/s
A    baseline                  REF            7.3815    1.000x     +0.00%   138725
H    full                      PASS           5.7764    1.278x    -21.75%   177273
```

- **Accuracy** — `REF` is the yardstick; `PASS` means that variant's output
  matched it within the competition tolerance; `FAIL` means it did not, and the
  speed number next to a FAIL should not be claimed.
- **Median ms** — typical time for one forward pass. Lower is better.
- **Speedup** — variant A's time divided by this variant's. 1.278x means "28%
  faster than A".
- **Token/s** — the same information as throughput. It moves with speedup.

The eight variants are every on/off combination of the three tweaks, which is
what lets the script separate their individual contributions:

```
Packed QKV  : matched geometric-mean effect = 1.069x
SDPA        : matched geometric-mean effect = 1.093x
Mask skip   : matched geometric-mean effect = 1.094x
```

"Matched" means each factor is measured four times, each time changing only
that one thing (A->B, C->E, D->F, G->H) and averaging. That is a genuinely
careful way to do it — it is the right method, applied to a limited set of
optimizations.

## 3. Head to head, same shape, same seed

### fp32, non-causal (their first command)

| | median ms | speedup | accuracy |
|---|---:|---:|:--|
| reference (variant A / official baseline) | 7.38 | 1.00x | — |
| their best variant, H (all three tweaks) | 5.78 | **1.28x** | PASS |
| our kernels | 2.63 | **2.80x** | PASS |

### fp32, causal — closest to the competition settings

| | median ms | speedup | accuracy |
|---|---:|---:|:--|
| reference | 7.73 | 1.00x | — |
| our kernels | 2.59 | **2.98x** | PASS |

### fp16, causal (their second command)

| | median ms | speedup | accuracy |
|---|---:|---:|:--|
| reference (variant A) | 4.44 | 1.00x | — |
| their best variant, H | 2.76 | 1.61x | **FAIL** (92 elements) |
| our kernels | 2.58 | 1.72x | **FAIL** (108 elements) |

## 4. About that fp16 FAIL — it is not a bug in either implementation

In fp16 the ablation reports that every SDPA variant (C, E, G, H) fails the
accuracy check, and our kernels fail it too. That looks alarming. It is not a
defect in anyone's code: **at this shape, in fp16, the tolerance cannot be met
by any implementation that is not bit-for-bit identical to the reference.**

`bench/test_bf16_limit.py` demonstrates this without involving custom kernels
at all. It takes the reference model and compares it against itself, computed
two mathematically identical ways:

```
float16: reference compared against equivalent references
  FAIL  same model in fp32, output cast back      max_abs=1.17e-02  failed=5/524288
  FAIL  same model, batch split and concatenated  max_abs=7.81e-03  failed=3/524288
```

The second line is the clincher. Split the batch into two halves, run the
*same* model on each, glue the results back together — no maths changed at
all, only the size of the chunks the GPU worked on — and the result no longer
matches itself closely enough to pass. At 6 layers and d_model 512, fp16
rounding accumulates until a handful of output values that should be near zero
are dominated by noise, and the rule's 0.002 absolute allowance is tighter than
that noise.

Practical consequences:

- **fp16 FAILs at this shape tell you nothing about implementation quality.**
  Both implementations are inside the noise floor of the reference itself.
- **This is not a problem for the actual competition.** The official test
  matrix runs fp32 (the harness default), where everything passes: our 13
  shapes all pass with zero failing elements, and the same control passes
  comfortably in fp32.
- If your friend wants a meaningful fp16 comparison, the fair move is to drop
  `--dtype float16` and compare in fp32, or to report fp16 timings explicitly
  labelled as "accuracy not achievable at this configuration".

## 5. One caveat about the shape

`batch=8, seq=128, d_model=512, heads=8, ffn=2048, layers=6` is the ablation
script's built-in default. It is **not** one of the 14 shapes in the official
appendix — those use d_model of 32, 128 or 1024, ffn equal to d_model, and 4
layers (2 for shape 14). So these runs are useful for understanding the
optimizations, but they do not produce a number that can be quoted as a
competition result.

Our headline numbers come from the official matrix and are in
[TECH_REPORT.md](TECH_REPORT.md): 13 of 14 shapes (shape 14 does not fit on a
12 GB GPU), all passing, geometric-mean **9.34x**.

## 6. Why the gap between 1.28x and 2.80x

The ablation's three tweaks are the first rung of the same ladder we climbed.
Our own attribution table (`report/results/latest-ablation.md`) starts there and
keeps going:

| step | case 1 speedup |
|---|---:|
| packed QKV + SDPA + deferred residuals | 3.43x |
| + Triton LayerNorm and FFN kernels | 6.48x |
| + Triton causal flash attention | 7.04x |
| + CUDA Graph capture | 6.83x (and 17.96x on batch 1) |

The friend's script measures rung one and stops, because measuring rungs two
through four requires writing the kernels. Both numbers are correct; they are
answers to different questions.
