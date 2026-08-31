"""Fused residual-add + LayerNorm.

The eager sequence for one block is

    res = res + attn_out                     # kernel 1
    h   = layer_norm(res).to(fp16)           # kernels 2 and 3

which touches the residual tensor five times. This kernel does it in one pass:
read `res` and `delta`, write the updated residual and its normalization.

Combined with the deferred-delta restructuring in engine.py -- where the FFN
output is not added immediately but handed to the *next* block's norm1 -- a
transformer block ends up with exactly two residual passes instead of four.

Rows are processed in tiles rather than one-row-per-CTA: at d_model = 128 a
one-row CTA gives every thread a single 4-byte element and reaches only about
half of peak bandwidth. A [BLOCK_M, BLOCK_N] tile gives each thread 16
elements and lets the compiler emit vector loads.
"""

from __future__ import annotations

from typing import Optional

import torch
import triton
import triton.language as tl


@triton.jit
def _add_layernorm_tiled(
    RES, DELTA, RES_OUT, OUT, W, B,
    M, N,
    stride_res, stride_delta, stride_res_out, stride_out,
    eps,
    HAS_DELTA: tl.constexpr,
    WRITE_RES: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    pid = tl.program_id(0)
    rows = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    row_mask = rows < M
    col_mask = cols < N
    mask = row_mask[:, None] & col_mask[None, :]

    x = tl.load(
        RES + rows[:, None] * stride_res + cols[None, :], mask=mask, other=0.0
    ).to(tl.float32)
    if HAS_DELTA:
        x += tl.load(
            DELTA + rows[:, None] * stride_delta + cols[None, :],
            mask=mask, other=0.0,
        ).to(tl.float32)
        if WRITE_RES:
            tl.store(
                RES_OUT + rows[:, None] * stride_res_out + cols[None, :],
                x.to(RES_OUT.dtype.element_ty),
                mask=mask,
            )

    mean = tl.sum(x, axis=1) / N
    centered = tl.where(mask, x - mean[:, None], 0.0)
    var = tl.sum(centered * centered, axis=1) / N
    rstd = 1.0 / tl.sqrt(var + eps)

    w = tl.load(W + cols, mask=col_mask, other=0.0).to(tl.float32)
    b = tl.load(B + cols, mask=col_mask, other=0.0).to(tl.float32)
    y = centered * rstd[:, None] * w[None, :] + b[None, :]
    tl.store(
        OUT + rows[:, None] * stride_out + cols[None, :],
        y.to(OUT.dtype.element_ty),
        mask=mask,
    )


def _config(n: int):
    """Aim for ~2048 elements per CTA so each of 128 threads gets 16 of them."""
    block_n = triton.next_power_of_2(n)
    block_m = max(1, min(16, 2048 // block_n))
    num_warps = 4 if block_n <= 1024 else 8
    return block_m, block_n, num_warps


def _run(residual, delta, weight, bias, eps, out_dtype, write_residual):
    rows, n = residual.shape
    out_dtype = out_dtype or residual.dtype
    out = torch.empty((rows, n), dtype=out_dtype, device=residual.device)
    res_out = (
        torch.empty_like(residual)
        if (delta is not None and write_residual)
        else residual
    )
    block_m, block_n, num_warps = _config(n)
    _add_layernorm_tiled[(triton.cdiv(rows, block_m),)](
        residual,
        delta if delta is not None else residual,
        res_out, out, weight, bias,
        rows, n,
        residual.stride(0),
        (delta if delta is not None else residual).stride(0),
        res_out.stride(0), out.stride(0),
        eps,
        HAS_DELTA=delta is not None,
        WRITE_RES=write_residual,
        BLOCK_M=block_m, BLOCK_N=block_n,
        num_warps=num_warps,
    )
    return res_out, out


def layernorm(x, weight, bias, eps, out_dtype=None):
    """Plain LayerNorm over the last dim, emitting `out_dtype` directly."""
    return _run(x, None, weight, bias, eps, out_dtype, False)[1]


def fused_add_layernorm(
    residual, delta: Optional[torch.Tensor], weight, bias, eps,
    out_dtype=None, write_residual: bool = True,
):
    """Return (residual + delta, layernorm(residual + delta)).

    With `write_residual=False` the updated residual is not materialized --
    used for the final norm, where nothing downstream reads it again.
    """
    return _run(residual, delta, weight, bias, eps, out_dtype, write_residual)
