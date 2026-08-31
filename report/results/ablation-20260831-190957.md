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
