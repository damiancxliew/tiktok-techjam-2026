# Demo video script (~4 minutes)

Backend track, so this is a walkthrough of measurements rather than a UI. Run
everything live in one terminal; the numbers are the demo.

---

### 0:00 — The problem in one screen (20s)

> "Problem 3 gives you a Transformer layer, a fixed accuracy budget, and 14
> input shapes. The optimized implementation has to be numerically
> indistinguishable from the reference and as fast as possible on my own GPU —
> an RTX 3060."

Show `torch_transformer_benchmark.py`, scroll to `UserOptimizedTransformer`.

> "This is the only hook we're allowed to change. Everything else — the
> accuracy rule, the timing loop — stays exactly as shipped."

---

### 0:20 — Run it once, live (40s)

```bash
python torch_transformer_benchmark.py --batch-size 64 --d-model 128 --heads 4 --seq-len 128 --layers 4 --ffn-dim 128 --causal --seed 1
```

Let it finish on camera. Point at the two lines that matter: `summary: PASS`
and `speedup: N.NNx`.

> "Same weights, same inputs, zero failing elements out of 65 thousand."

---

### 1:00 — What we actually built (60s)

Show the table in `README.md`, then open the three kernels briefly:

- `tjkernels/kernels/attention.py` — "PyTorch's Windows build ships **without**
  FlashAttention. We measured every backend" — show `bench/attn_backends.py`
  output with the `nan` column — "so we wrote the missing kernel. It also reads
  Q, K and V in place out of the packed QKV buffer and writes the exact layout
  the output projection wants, so the two layout copies disappear too."
- `tjkernels/kernels/layernorm.py` — "residual add and LayerNorm in one pass,
  and the FFN output isn't added immediately, it's handed to the next block's
  norm. Two residual passes per block instead of four."
- `tjkernels/kernels/ffn.py` — "the whole feed-forward network in one kernel;
  the `[tokens, ffn]` intermediate never reaches global memory."

---

### 2:00 — The part we're actually proud of (60s)

Open `bench/autotune.py`.

> "A normal autotuner asks which config is fastest. That's the wrong question
> when there's an error budget — the fastest config isn't always inside it. So
> this one only admits a plan if it passes at **70% of the official
> tolerance**, across several input draws. The margin is what keeps it passing
> on a seed it was never tuned on."

Show the tuner log for case 7, where fp16 gets rejected and the shape is
escalated to true fp32.

> "And it caught a real bug: Triton's `tl.dot` quietly uses TF32 for fp32
> inputs — a *coarser* mantissa than the fp16 path — so our 'safe' fallback was
> measuring worse than the fast path until we asked for IEEE math explicitly."

---

### 3:00 — All 13 shapes (45s)

Show `report/results/latest-main.md` — the full table — and the ablation table.

> "Thirteen of the fourteen shapes, all passing, geometric mean N.NNx. Case 14
> we excluded, and we show the arithmetic instead of hiding it: its input
> tensor alone is 13.1 gigabytes on a 12-gigabyte card, and the reference
> implementation would need a five-trillion-element score matrix. It isn't a
> limitation of our kernel — nothing runs it here."

---

### 3:45 — Close (15s)

> "Every number in the report is generated from measurement JSON, not typed in.
> `bench/make_report.py` rebuilds it. Everything reproduces with `--seed 1`."

---

## Recording checklist

- [ ] Close background GPU apps (Chrome, Steam) — they perturb the timings
- [ ] Terminal font large enough to read at 1080p
- [ ] `conda activate tj26` before recording
- [ ] Pre-warm the Triton cache with one run so compile time doesn't stall the take
- [ ] Upload to YouTube as **public**, link it in the Devpost description
