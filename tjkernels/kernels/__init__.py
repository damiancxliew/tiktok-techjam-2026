"""Triton kernels, with transparent PyTorch fallbacks.

Everything here is optional: if Triton is unavailable, or a shape falls outside
a kernel's supported range, the engine keeps running on eager PyTorch and stays
correct. That is deliberate -- the submission must not be one bad import away
from failing on a judge's machine. `force_torch` exposes the same fallbacks to
bench/ablation.py so the contribution of each kernel can be measured.
"""

from __future__ import annotations

from typing import Optional, Tuple

import torch
import torch.nn.functional as F

try:
    from .layernorm import fused_add_layernorm as _triton_add_ln
    from .ffn import fused_ffn as _triton_ffn
    from .ffn import supports as ffn_supports
    from .attention import flash_attention
    from .attention import supports as flash_supports

    HAVE_TRITON = True
    HAVE_FLASH = True
except Exception as _exc:  # pragma: no cover - exercised only without triton
    HAVE_TRITON = False
    HAVE_FLASH = False
    _IMPORT_ERROR = _exc

    def flash_supports(head_dim: int, causal: bool) -> bool:
        return False

    def flash_attention(*args, **kwargs):
        raise RuntimeError('triton unavailable')

    def ffn_supports(d_model: int, ffn_dim: int, itemsize: int = 2) -> bool:
        return False


def _torch_add_layernorm(residual, delta, weight, bias, eps, out_dtype):
    if delta is not None:
        residual = residual + delta
    out = F.layer_norm(residual.float(), (residual.shape[-1],), weight, bias, eps)
    return residual, (out if out_dtype is None else out.to(out_dtype))


def fused_add_layernorm(
    residual: torch.Tensor,
    delta: Optional[torch.Tensor],
    weight: torch.Tensor,
    bias: torch.Tensor,
    eps: float,
    out_dtype: Optional[torch.dtype] = None,
    write_residual: bool = True,
    force_torch: bool = False,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """residual += delta (optional); return (residual, layernorm(residual))."""
    usable = (
        HAVE_TRITON
        and not force_torch
        and residual.is_cuda
        and residual.shape[-1] <= 8192
    )
    if usable:
        return _triton_add_ln(
            residual, delta, weight, bias, eps,
            out_dtype=out_dtype, write_residual=write_residual,
        )
    return _torch_add_layernorm(residual, delta, weight, bias, eps, out_dtype)


def layernorm(x, weight, bias, eps, out_dtype=None, force_torch: bool = False):
    return fused_add_layernorm(
        x, None, weight, bias, eps,
        out_dtype=out_dtype, write_residual=False, force_torch=force_torch,
    )[1]


def fused_ffn(h, w1, b1, w2, b2, force_torch: bool = False):
    if (
        HAVE_TRITON
        and not force_torch
        and h.is_cuda
        and ffn_supports(h.shape[-1], w1.shape[-1], h.element_size())
    ):
        return _triton_ffn(h, w1, b1, w2, b2)
    hidden = F.gelu(torch.addmm(b1, h, w1), approximate="none")
    return torch.addmm(b2, hidden, w2)


__all__ = [
    "HAVE_TRITON",
    "HAVE_FLASH",
    "flash_attention",
    "flash_supports",
    "layernorm",
    "fused_add_layernorm",
    "fused_ffn",
    "ffn_supports",
]
