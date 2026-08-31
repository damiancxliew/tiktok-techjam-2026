"""Shape -> execution-plan dispatch.

A Plan is a small bundle of knobs describing *how* to execute one transformer
layer for one input shape.  Defaults come from the heuristics in
`default_plan()`; `plans.json` (written by bench/autotune.py) overrides them
per shape with the fastest configuration that still passed the accuracy gate.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, Optional

import torch

_PLAN_FILE = Path(__file__).with_name("plans.json")

_DTYPES = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}
_DTYPE_NAMES = {v: k for k, v in _DTYPES.items()}


@dataclass(frozen=True)
class Plan:
    """Execution knobs for a single (shape, dtype) combination."""

    # Precision policy.  compute_dtype drives every GEMM / attention op;
    # residual_dtype is the precision the residual stream is accumulated in.
    compute_dtype: torch.dtype = torch.float16
    residual_dtype: torch.dtype = torch.float32

    # Kernel selection.
    attn: str = "triton"        # triton | sdpa | math
    ffn: str = "torch"          # torch | triton
    norm: str = "torch"         # torch | triton
    packed_qkv: bool = True

    # Whole-forward CUDA Graph capture (kills per-launch overhead on WDDM).
    cuda_graph: bool = True

    label: str = "default"

    def to_json(self) -> Dict[str, object]:
        return {
            "compute_dtype": _DTYPE_NAMES[self.compute_dtype],
            "residual_dtype": _DTYPE_NAMES[self.residual_dtype],
            "attn": self.attn,
            "ffn": self.ffn,
            "norm": self.norm,
            "packed_qkv": self.packed_qkv,
            "cuda_graph": self.cuda_graph,
            "label": self.label,
        }

    @staticmethod
    def from_json(blob: Dict[str, object]) -> "Plan":
        blob = dict(blob)
        blob["compute_dtype"] = _DTYPES[str(blob.get("compute_dtype", "float16"))]
        blob["residual_dtype"] = _DTYPES[str(blob.get("residual_dtype", "float32"))]
        fields = Plan.__dataclass_fields__.keys()
        return Plan(**{k: v for k, v in blob.items() if k in fields})


def shape_key(
    batch: int, seq: int, d_model: int, heads: int, ffn_dim: int,
    layers: int, causal: bool, dtype: torch.dtype,
) -> str:
    return (
        f"b{batch}_s{seq}_d{d_model}_h{heads}_f{ffn_dim}_l{layers}"
        f"_{'causal' if causal else 'full'}_{_DTYPE_NAMES[dtype]}"
    )


def default_plan(
    batch: int, seq: int, d_model: int, heads: int, ffn_dim: int,
    layers: int, causal: bool, dtype: torch.dtype, device: torch.device,
) -> Plan:
    """Heuristic plan used before (or without) autotuning."""
    if device.type != "cuda":
        return Plan(
            compute_dtype=dtype, residual_dtype=torch.float32,
            attn="sdpa", ffn="torch", norm="torch",
            cuda_graph=False, label="cpu",
        )

    major, _ = torch.cuda.get_device_capability(device)
    tokens = batch * seq

    # fp16 tensor cores run 2x TF32 on Ampere consumer parts, and the reference
    # itself is TF32 (10-bit mantissa) -- same mantissa width as fp16, so this
    # does not systematically lose ground against the comparison target.
    compute_dtype = dtype
    if dtype == torch.float32 and major >= 7:
        compute_dtype = torch.float16

    # An fp32 residual stream costs bandwidth but buys accuracy everywhere it
    # was measured, including for fp16 models (matching the reference's fp16
    # residual rounding instead measured *worse*, 4.9e-3 vs 3.9e-3).
    residual_dtype = torch.float32

    # A whole-forward graph costs one copy-in + one copy-out of the activation
    # tensor.  That is free for small shapes and real money for huge ones.
    x_bytes = tokens * d_model * dtype.itemsize
    cuda_graph = x_bytes <= 64 * 1024 * 1024

    # Our Triton kernel handles the causal case; anything else falls back to
    # SDPA.  The explicit-score "math" branch loses to both by ~3x (it pays for
    # a materialized [B,H,S,S] score tensor, a separate fp32 softmax and a
    # masked_fill) and survives only as an ablation control.
    attn = "triton" if causal else "sdpa"

    # The tile-resident FFN needs both weight matrices in shared memory, which
    # depends on the compute dtype -- fp32 needs twice the room of fp16.
    from .kernels import ffn_supports

    ffn = "triton" if ffn_supports(d_model, ffn_dim, compute_dtype.itemsize) else "torch"
    norm = "triton"

    return Plan(
        compute_dtype=compute_dtype,
        residual_dtype=residual_dtype,
        attn=attn,
        ffn=ffn,
        norm=norm,
        packed_qkv=True,
        cuda_graph=cuda_graph,
        label="heuristic",
    )


_cache: Optional[Dict[str, Plan]] = None
_override: Optional[Plan] = None


def set_override(plan: Optional[Plan]) -> None:
    """Force a specific plan regardless of shape. Used by bench/autotune.py."""
    global _override
    _override = plan


def _load_tuned() -> Dict[str, Plan]:
    global _cache
    if _cache is None:
        _cache = {}
        if _PLAN_FILE.exists() and os.environ.get("TJ_IGNORE_PLANS") != "1":
            try:
                raw = json.loads(_PLAN_FILE.read_text())
                _cache = {k: Plan.from_json(v) for k, v in raw.items()}
            except Exception as exc:  # pragma: no cover - never break a run
                print(f"[tjkernels] ignoring unreadable plans.json: {exc}")
    return _cache


def select_plan(
    batch: int, seq: int, d_model: int, heads: int, ffn_dim: int,
    layers: int, causal: bool, dtype: torch.dtype, device: torch.device,
) -> Plan:
    if _override is not None:
        return _override

    key = shape_key(batch, seq, d_model, heads, ffn_dim, layers, causal, dtype)
    tuned = _load_tuned().get(key)
    if tuned is not None:
        return tuned

    plan = default_plan(batch, seq, d_model, heads, ffn_dim, layers,
                        causal, dtype, device)

    # Environment overrides make the ablation study in bench/ a one-liner.
    if os.environ.get("TJ_NO_GRAPH") == "1":
        plan = replace(plan, cuda_graph=False)
    if os.environ.get("TJ_NO_TRITON") == "1":
        plan = replace(plan, ffn="torch", norm="torch")
    forced = os.environ.get("TJ_COMPUTE_DTYPE")
    if forced in _DTYPES:
        plan = replace(plan, compute_dtype=_DTYPES[forced])
    forced = os.environ.get("TJ_RESIDUAL_DTYPE")
    if forced in _DTYPES:
        plan = replace(plan, residual_dtype=_DTYPES[forced])
    for var, field in (("TJ_ATTN", "attn"), ("TJ_FFN", "ffn"), ("TJ_NORM", "norm")):
        value = os.environ.get(var)
        if value:
            plan = replace(plan, **{field: value})
    return plan


def save_plans(plans: Dict[str, Plan]) -> None:
    _PLAN_FILE.write_text(
        json.dumps({k: v.to_json() for k, v in plans.items()}, indent=2)
    )
    global _cache
    _cache = None
