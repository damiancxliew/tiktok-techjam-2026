"""Single-kernel feed-forward network.

For the shapes in this task the FFN is `[T, d] -> [T, ffn] -> [T, d]` with
d and ffn both small (32, 128, and 1024 in the official matrix). When both
fit in one tile, the whole FFN -- GEMM, bias, exact-erf GELU, second GEMM,
bias -- runs inside a single kernel, and the [T, ffn] intermediate never
touches global memory.

That intermediate is the point. Eager PyTorch writes it out and reads it back:
for case 6 (T = 1.28M, ffn = 128) that is ~330 MB of write plus ~330 MB of
read per layer in fp16, on a card with 360 GB/s. This kernel deletes it.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl


@triton.jit
def _ffn_fwd(
    H, W1, B1, W2, B2, OUT,
    T, D: tl.constexpr, Fdim: tl.constexpr,
    stride_h, stride_out,
    BLOCK_M: tl.constexpr, BLOCK_D: tl.constexpr, BLOCK_F: tl.constexpr,
    PREC: tl.constexpr,
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_d = tl.arange(0, BLOCK_D)
    offs_f = tl.arange(0, BLOCK_F)

    mask_m = offs_m < T
    mask_d = offs_d < D
    mask_f = offs_f < Fdim

    # ---- first GEMM: [BLOCK_M, D] x [D, F] -------------------------------
    h = tl.load(
        H + offs_m[:, None] * stride_h + offs_d[None, :],
        mask=mask_m[:, None] & mask_d[None, :],
        other=0.0,
    )
    w1 = tl.load(
        W1 + offs_d[:, None] * Fdim + offs_f[None, :],
        mask=mask_d[:, None] & mask_f[None, :],
        other=0.0,
    )
    acc = tl.dot(h, w1, out_dtype=tl.float32, input_precision=PREC)
    acc += tl.load(B1 + offs_f, mask=mask_f, other=0.0).to(tl.float32)[None, :]

    # ---- exact GELU (erf form, matching approximate="none") --------------
    hidden = acc * 0.5 * (1.0 + tl.erf(acc * 0.7071067811865476))

    # ---- second GEMM: [BLOCK_M, F] x [F, D] ------------------------------
    w2 = tl.load(
        W2 + offs_f[:, None] * D + offs_d[None, :],
        mask=mask_f[:, None] & mask_d[None, :],
        other=0.0,
    )
    out = tl.dot(hidden.to(w2.dtype), w2, out_dtype=tl.float32,
                 input_precision=PREC)
    out += tl.load(B2 + offs_d, mask=mask_d, other=0.0).to(tl.float32)[None, :]

    tl.store(
        OUT + offs_m[:, None] * stride_out + offs_d[None, :],
        out.to(OUT.dtype.element_ty),
        mask=mask_m[:, None] & mask_d[None, :],
    )


# Tile shapes measured by bench/tune_ffn.py over the token counts in the test
# matrix. The [BLOCK_M, BLOCK_F] fp32 accumulator dominates register pressure,
# so the tile shrinks as ffn_dim grows. Note that below ~4k tokens every config
# measures the same, because the kernel is under the launch-overhead floor
# there -- these numbers are chosen from the large-token rows, which are the
# only ones that discriminate.
def _config(tokens: int, d_model: int, ffn_dim: int):
    block_d = max(16, triton.next_power_of_2(d_model))
    block_f = max(16, triton.next_power_of_2(ffn_dim))
    area = block_f * block_d
    if area <= 4096:            # d, ffn <= 64 (case 7)
        block_m, num_warps, num_stages = 32, 8, 3
    elif area <= 32768:         # the 128 x 128 family (cases 1-6, 9-13)
        block_m, num_warps, num_stages = 64, 4, 2
    else:
        block_m, num_warps, num_stages = 16, 8, 2
    if tokens < block_m:
        block_m = max(16, triton.next_power_of_2(tokens))
    return block_d, block_f, block_m, num_warps, num_stages


def _precision(dtype) -> str:
    """fp32 inputs default to TF32 in tl.dot, whose 10-bit mantissa is actually
    coarser than the fp16 path -- so the fp32 plan, which exists precisely to
    buy accuracy, asks for true IEEE math."""
    return "ieee" if dtype == torch.float32 else "tf32"


SMEM_BUDGET = 96 * 1024


def supports(d_model: int, ffn_dim: int, itemsize: int = 2) -> bool:
    """Tile-resident FFN only pays off while both weight matrices fit in one
    tile -- and only works at all while they fit in shared memory, which
    depends on the dtype (fp32 needs twice the room of fp16)."""
    if d_model > 256 or ffn_dim > 256:
        return False
    weights = 2 * d_model * ffn_dim * itemsize
    return weights <= SMEM_BUDGET


def fused_ffn(h, w1, b1, w2, b2):
    tokens, d_model = h.shape
    ffn_dim = w1.shape[1]
    block_d, block_f, block_m, num_warps, num_stages = _config(
        tokens, d_model, ffn_dim
    )
    out = torch.empty_like(h)
    _ffn_fwd[(triton.cdiv(tokens, block_m),)](
        h, w1, b1, w2, b2, out,
        tokens, d_model, ffn_dim,
        h.stride(0), out.stride(0),
        BLOCK_M=block_m, BLOCK_D=block_d, BLOCK_F=block_f,
        PREC=_precision(h.dtype),
        num_warps=num_warps, num_stages=num_stages,
    )
    return out
