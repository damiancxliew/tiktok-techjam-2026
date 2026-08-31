"""FlashAttention-2 style causal attention, in Triton.

Why write this at all, when PyTorch has `scaled_dot_product_attention`?

Because the official PyTorch Windows wheels are built *without* the
FlashAttention backend. On this machine `SDPBackend.FLASH_ATTENTION` raises for
every shape in the test matrix (see bench/attn_backends.py), leaving the
cutlass memory-efficient kernel, which measures 6-11 TFLOP/s on these shapes
against ~51 TFLOP/s of fp16 tensor-core peak.

This kernel additionally reads Q, K and V straight out of the packed
[B, S, 3, H, hd] QKV buffer and writes its result as a contiguous [B*S, d]
matrix -- exactly the layout the output projection wants. That removes the two
layout-shuffling copies the SDPA path needs on either side of it.

Online softmax keeps a running max and normalizer per query row, so the
[S, S] score matrix is never materialized; causal blocks past the diagonal are
skipped entirely rather than computed and masked.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Optional, Tuple

import torch
import triton
import triton.language as tl
from triton.runtime.errors import OutOfResources

LOG2E = 1.4426950408889634


@triton.jit
def _flash_fwd(
    QKV, OUT,
    stride_qkv_b, stride_qkv_s, stride_qkv_c, stride_qkv_h,
    stride_out_t, stride_out_h,
    seq, scale,
    HEADS: tl.constexpr,
    HD: tl.constexpr,
    BLOCK_HD: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    CAUSAL: tl.constexpr,
    PREC: tl.constexpr,
):
    start_m = tl.program_id(0)
    bh = tl.program_id(1)
    batch = bh // HEADS
    head = bh % HEADS

    offs_m = start_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    offs_d = tl.arange(0, BLOCK_HD)
    d_mask = offs_d < HD

    base = QKV + batch * stride_qkv_b + head * stride_qkv_h
    q_ptrs = base + offs_m[:, None] * stride_qkv_s + offs_d[None, :]
    k_base = base + stride_qkv_c
    v_base = base + 2 * stride_qkv_c

    q = tl.load(
        q_ptrs,
        mask=(offs_m[:, None] < seq) & d_mask[None, :],
        other=0.0,
    )

    acc = tl.zeros((BLOCK_M, BLOCK_HD), dtype=tl.float32)
    m_i = tl.full((BLOCK_M,), float("-inf"), dtype=tl.float32)
    l_i = tl.zeros((BLOCK_M,), dtype=tl.float32)

    # Causal: query block `start_m` only ever sees keys up to its last row.
    hi = tl.minimum((start_m + 1) * BLOCK_M, seq) if CAUSAL else seq

    for start_n in range(0, hi, BLOCK_N):
        cols = start_n + offs_n
        col_ok = cols < seq
        k = tl.load(
            k_base + cols[:, None] * stride_qkv_s + offs_d[None, :],
            mask=col_ok[:, None] & d_mask[None, :],
            other=0.0,
        )
        # `scale` already carries the log2(e) factor so the softmax can
        # use the faster exp2 instead of exp.
        qk = tl.dot(q, tl.trans(k), out_dtype=tl.float32,
                    input_precision=PREC) * scale

        if CAUSAL:
            qk = tl.where(offs_m[:, None] >= cols[None, :], qk, float("-inf"))
        else:
            qk = tl.where(col_ok[None, :], qk, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(qk, axis=1))
        # A fully-masked row (impossible here, but cheap to guard) would give
        # -inf - -inf; clamping keeps the exponentials finite.
        m_safe = tl.where(m_new == float("-inf"), 0.0, m_new)
        alpha = tl.exp2(m_i - m_safe)
        p = tl.exp2(qk - m_safe[:, None])

        l_i = l_i * alpha + tl.sum(p, axis=1)
        acc = acc * alpha[:, None]

        v = tl.load(
            v_base + cols[:, None] * stride_qkv_s + offs_d[None, :],
            mask=col_ok[:, None] & d_mask[None, :],
            other=0.0,
        )
        acc += tl.dot(p.to(v.dtype), v, out_dtype=tl.float32,
                      input_precision=PREC)
        m_i = m_new

    acc = acc / l_i[:, None]

    out_ptrs = (
        OUT
        + (batch * seq + offs_m)[:, None] * stride_out_t
        + head * stride_out_h
        + offs_d[None, :]
    )
    tl.store(
        out_ptrs,
        acc.to(OUT.dtype.element_ty),
        mask=(offs_m[:, None] < seq) & d_mask[None, :],
    )


# Preference order for (BLOCK_M, BLOCK_N, num_warps, num_stages). The launcher
# walks this list and takes the first entry that fits in shared memory, then
# falls further down it if the hardware still refuses -- head_dim 256 in fp32
# needs 139 KB at the top entry against a 101 KB limit on sm_86.
_CONFIGS = [
    (128, 64, 8, 3),
    (64, 64, 4, 3),
    (64, 32, 4, 2),
    (32, 32, 4, 2),
    (16, 16, 4, 1),
]

SMEM_BUDGET = 96 * 1024


def _preferred(seq: int, block_hd: int):
    """Measured best-first choice before the generic fallbacks.

    Block shape matters more than it looks: at seq 128 a 128-row query block
    means one CTA per (sequence, head), which both starves the GPU of
    parallelism and throws away causal skipping, since a single block spans the
    whole triangle. Smaller query blocks win there.
    """
    if seq <= 64:
        edge = max(16, triton.next_power_of_2(seq))
        return (edge, edge, 4, 2)
    if block_hd >= 128:
        return (64, 32, 4, 2)
    if seq <= 256:
        return (64, 64, 4, 3)
    return (128, 64, 8, 3)


@lru_cache(maxsize=64)
def _configs(seq: int, head_dim: int, itemsize: int) -> Tuple:
    """Viable launch configs, best-first. Cached: this runs on every call,
    and at these shapes a few microseconds of Python is visible."""
    block_hd = max(16, triton.next_power_of_2(head_dim))
    seq_cap = max(16, triton.next_power_of_2(seq))
    out = []
    seen = set()
    ordered = [_preferred(seq, block_hd)] + _CONFIGS
    for block_m, block_n, num_warps, num_stages in ordered:
        block_m = min(block_m, seq_cap)
        block_n = min(block_n, seq_cap)
        # q tile plus the pipelined k/v tiles.
        smem = (block_m * block_hd + num_stages * 2 * block_n * block_hd) * itemsize
        key = (block_m, block_n, num_warps, num_stages)
        if smem > SMEM_BUDGET or key in seen:
            continue
        seen.add(key)
        out.append((block_hd, block_m, block_n, num_warps, num_stages))
    if not out:
        out.append((block_hd, 16, 16, 4, 1))
    return tuple(out)


def supports(head_dim: int, causal: bool) -> bool:
    return head_dim <= 256 and causal


def flash_attention(qkv: torch.Tensor, heads: int, causal: bool = True):
    """qkv: [B, S, 3, H, hd] -> out: [B*S, H*hd] contiguous."""
    batch, seq, three, n_heads, head_dim = qkv.shape
    assert three == 3 and n_heads == heads
    d_model = heads * head_dim
    out = torch.empty((batch * seq, d_model), dtype=qkv.dtype, device=qkv.device)

    configs = _configs(seq, head_dim, qkv.element_size())
    last_error: Optional[BaseException] = None

    for block_hd, block_m, block_n, num_warps, num_stages in configs:
        grid = (triton.cdiv(seq, block_m), batch * heads)
        try:
            _flash_fwd[grid](
                qkv, out,
                qkv.stride(0), qkv.stride(1), qkv.stride(2), qkv.stride(3),
                out.stride(0), head_dim,
                seq, (head_dim ** -0.5) * LOG2E,
                HEADS=heads,
                HD=head_dim,
                BLOCK_HD=block_hd,
                BLOCK_M=block_m,
                BLOCK_N=block_n,
                CAUSAL=causal,
                PREC="ieee" if qkv.dtype == torch.float32 else "tf32",
                num_warps=num_warps,
                num_stages=num_stages,
            )
            return out
        except OutOfResources as exc:
            # The smem estimate is approximate; believe the compiler over it.
            last_error = exc
            continue

    raise RuntimeError(
        f"no viable flash config for seq={seq} head_dim={head_dim} "
        f"dtype={qkv.dtype}: {last_error}"
    )
