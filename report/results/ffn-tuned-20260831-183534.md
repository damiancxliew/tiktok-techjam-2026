| # | shape (B,S,d,H,F,L) | tokens | accuracy | max_abs | baseline ms | ours ms | speedup |
|---|---|---:|---|---:|---:|---:|---:|
| 1 | 64,128,128,4,128,4 | 8,192 | PASS | 1.30e-03 | 6.059 | 0.876 | 6.91x |
| 6 | 10000,128,128,4,128,4 | 1,280,000 | PASS | 1.64e-03 | 956.059 | 122.090 | 7.83x |
| 7 | 64,128,32,4,32,4 | 8,192 | PASS | 9.90e-04 | 4.591 | 0.707 | 6.49x |

**3/3 cases benchmarked - geometric-mean speedup 7.06x**