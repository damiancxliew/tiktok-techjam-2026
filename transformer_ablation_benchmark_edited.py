#!/usr/bin/env python3
"""
Transformer ablation benchmark.

Automatically evaluates a 2^3 ablation over three optimizations:

A  baseline                 separate QKV + manual attention + normal masking
B  packed_qkv              packed QKV   + manual attention + normal masking
C  sdpa                    separate QKV + SDPA             + normal masking
D  mask_skip               separate QKV + manual attention + skip no-op all-valid masks
E  packed_qkv+sdpa         packed QKV   + SDPA             + normal masking
F  packed_qkv+mask_skip    packed QKV   + manual attention + skip no-op all-valid masks
G  sdpa+mask_skip          separate QKV + SDPA             + skip no-op all-valid masks
H  full                    packed QKV   + SDPA             + skip no-op all-valid masks

Every candidate is compared against the exact baseline output using:

    abs(candidate - reference) <= atol
    OR
    abs(candidate - reference) <= rtol * abs(reference)

The benchmark uses the same fixed input for every variant and rotates measurement
order across rounds to reduce thermal / clock-order bias.

Competition-oriented additions:
- causal attention is ON by default (use --no-causal for non-causal experiments)
- CUDA SDPA backend selection: auto / flash / efficient / cudnn / math
- official 14-shape sweep via --competition-shapes
- an explicit-memory guard prevents accidental multi-terabyte manual-attention runs
- --quick provides a low-cost development pass

Examples
--------
Mac / Apple Silicon:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --batch-size 8 --seq-len 128 --d-model 512 \
  --heads 8 --ffn-dim 2048 --layers 6 \
  --accuracy-trials 5 --warmup 20 --repeats 100 --benchmark-rounds 3

Only A/B/C/H:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --variants A,B,C,H

CUDA, force FlashAttention SDPA where supported:

python transformer_ablation_benchmark.py \
  --device cuda --dtype bfloat16 \
  --sdpa-backend flash --quick

Official competition shapes 1-13 (shape 14 is guarded by default):

python transformer_ablation_benchmark.py \
  --device cuda --dtype bfloat16 \
  --competition-shapes --shape-ids 1-13 \
  --variants A,C,H --csv competition_ablation.csv

Save CSV:

python transformer_ablation_benchmark.py \
  --device mps --dtype float32 \
  --csv ablation_results.csv
"""

from __future__ import annotations

import argparse
import copy
import csv
import math
import statistics
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


# =============================================================================
# Configuration
# =============================================================================

@dataclass(frozen=True)
class TransformerConfig:
    batch_size: int
    seq_len: int
    d_model: int
    num_heads: int
    ffn_dim: int
    num_layers: int
    causal: bool

    def validate(self) -> None:
        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if self.seq_len <= 0:
            raise ValueError("seq_len must be positive")
        if self.d_model <= 0:
            raise ValueError("d_model must be positive")
        if self.num_heads <= 0:
            raise ValueError("num_heads must be positive")
        if self.d_model % self.num_heads != 0:
            raise ValueError(
                f"d_model ({self.d_model}) must be divisible by "
                f"num_heads ({self.num_heads})"
            )
        if self.ffn_dim <= 0:
            raise ValueError("ffn_dim must be positive")
        if self.num_layers <= 0:
            raise ValueError("num_layers must be positive")


# Official challenge shapes are encoded as:
# (batch_size, d_model, num_heads, seq_len, num_layers, causal, ffn_dim)
COMPETITION_SHAPES: Dict[int, Tuple[int, int, int, int, int, bool, int]] = {
    1: (64, 128, 4, 128, 4, True, 128),
    2: (1, 128, 4, 128, 4, True, 128),
    3: (4, 128, 4, 128, 4, True, 128),
    4: (16, 128, 4, 128, 4, True, 128),
    5: (128, 128, 4, 128, 4, True, 128),
    6: (10000, 128, 4, 128, 4, True, 128),
    7: (64, 32, 4, 128, 4, True, 32),
    8: (64, 1024, 4, 128, 4, True, 1024),
    9: (64, 128, 1, 128, 4, True, 128),
    10: (64, 128, 2, 128, 4, True, 128),
    11: (64, 128, 16, 128, 4, True, 128),
    12: (64, 128, 4, 32, 4, True, 128),
    13: (64, 128, 4, 1024, 4, True, 1024),
    14: (32, 1024, 16, 100000, 2, True, 1024),
}


def competition_config(shape_id: int) -> TransformerConfig:
    try:
        batch, d_model, heads, seq_len, layers, causal, ffn_dim = (
            COMPETITION_SHAPES[shape_id]
        )
    except KeyError as exc:
        raise ValueError(f"unknown competition shape id: {shape_id}") from exc

    return TransformerConfig(
        batch_size=batch,
        seq_len=seq_len,
        d_model=d_model,
        num_heads=heads,
        ffn_dim=ffn_dim,
        num_layers=layers,
        causal=causal,
    )


@dataclass(frozen=True)
class VariantSpec:
    key: str
    name: str
    packed_qkv: bool
    sdpa: bool
    skip_all_valid_mask: bool


VARIANT_SPECS: Dict[str, VariantSpec] = {
    "A": VariantSpec(
        key="A",
        name="baseline",
        packed_qkv=False,
        sdpa=False,
        skip_all_valid_mask=False,
    ),
    "B": VariantSpec(
        key="B",
        name="packed_qkv",
        packed_qkv=True,
        sdpa=False,
        skip_all_valid_mask=False,
    ),
    "C": VariantSpec(
        key="C",
        name="sdpa",
        packed_qkv=False,
        sdpa=True,
        skip_all_valid_mask=False,
    ),
    "D": VariantSpec(
        key="D",
        name="mask_skip",
        packed_qkv=False,
        sdpa=False,
        skip_all_valid_mask=True,
    ),
    "E": VariantSpec(
        key="E",
        name="packed_qkv+sdpa",
        packed_qkv=True,
        sdpa=True,
        skip_all_valid_mask=False,
    ),
    "F": VariantSpec(
        key="F",
        name="packed_qkv+mask_skip",
        packed_qkv=True,
        sdpa=False,
        skip_all_valid_mask=True,
    ),
    "G": VariantSpec(
        key="G",
        name="sdpa+mask_skip",
        packed_qkv=False,
        sdpa=True,
        skip_all_valid_mask=True,
    ),
    "H": VariantSpec(
        key="H",
        name="full",
        packed_qkv=True,
        sdpa=True,
        skip_all_valid_mask=True,
    ),
}


def sdpa_backend_context(backend: str, device: torch.device):
    """Return a context manager selecting a CUDA SDPA backend when requested."""
    if backend == "auto" or device.type != "cuda":
        return nullcontext()

    # PyTorch 2.5+ API. Keep a compatibility fallback for older builds.
    try:
        from torch.nn.attention import SDPBackend, sdpa_kernel

        mapping = {
            "flash": SDPBackend.FLASH_ATTENTION,
            "efficient": SDPBackend.EFFICIENT_ATTENTION,
            "math": SDPBackend.MATH,
        }
        if hasattr(SDPBackend, "CUDNN_ATTENTION"):
            mapping["cudnn"] = SDPBackend.CUDNN_ATTENTION

        if backend not in mapping:
            if backend == "cudnn":
                raise RuntimeError(
                    "this PyTorch build does not expose the cuDNN SDPA backend"
                )
            raise ValueError(f"unknown SDPA backend: {backend}")

        return sdpa_kernel(mapping[backend])
    except ImportError:
        if backend == "cudnn":
            raise RuntimeError(
                "cuDNN SDPA backend selection requires a newer PyTorch build"
            )

        flags = {
            "flash": dict(
                enable_flash=True,
                enable_math=False,
                enable_mem_efficient=False,
            ),
            "efficient": dict(
                enable_flash=False,
                enable_math=False,
                enable_mem_efficient=True,
            ),
            "math": dict(
                enable_flash=False,
                enable_math=True,
                enable_mem_efficient=False,
            ),
        }
        return torch.backends.cuda.sdp_kernel(**flags[backend])


def estimate_manual_attention_working_set_gib(
    config: TransformerConfig,
    dtype: torch.dtype,
) -> float:
    """Estimate the dominant explicit-attention score + fp32-softmax buffers."""
    score_elements = (
        config.batch_size
        * config.num_heads
        * config.seq_len
        * config.seq_len
    )
    score_element_size = torch.empty((), dtype=dtype).element_size()

    # The manual path materializes scores in input dtype and softmax in fp32.
    # This deliberately ignores smaller Q/K/V/context buffers, so it is a lower
    # bound rather than a full allocator model.
    bytes_estimate = score_elements * (score_element_size + 4)
    return bytes_estimate / (1024**3)


def parse_shape_ids(raw: str) -> List[int]:
    """Parse forms such as '1-13', '1,2,7-9', or 'all'."""
    raw = raw.strip().lower()
    if raw == "all":
        return list(COMPETITION_SHAPES)

    result: List[int] = []
    for piece in raw.split(","):
        piece = piece.strip()
        if not piece:
            continue
        if "-" in piece:
            start_s, end_s = piece.split("-", 1)
            start, end = int(start_s), int(end_s)
            if start > end:
                raise ValueError(f"invalid shape range: {piece}")
            result.extend(range(start, end + 1))
        else:
            result.append(int(piece))

    deduped: List[int] = []
    for shape_id in result:
        if shape_id not in COMPETITION_SHAPES:
            raise ValueError(
                f"shape id {shape_id} is not in 1-{max(COMPETITION_SHAPES)}"
            )
        if shape_id not in deduped:
            deduped.append(shape_id)

    if not deduped:
        raise ValueError("shape id selection is empty")
    return deduped


# =============================================================================
# Exact baseline
# =============================================================================

class BaselineSelfAttention(nn.Module):
    """Explicit multi-head self-attention implemented with native PyTorch ops."""

    def __init__(self, d_model: int, num_heads: int) -> None:
        super().__init__()
        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(d_model, d_model, bias=True)
        self.k_proj = nn.Linear(d_model, d_model, bias=True)
        self.v_proj = nn.Linear(d_model, d_model, bias=True)
        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _split_heads(self, x: torch.Tensor) -> torch.Tensor:
        batch, seq_len, _ = x.shape
        return (
            x.view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
            .contiguous()
        )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q = self._split_heads(self.q_proj(x))
        k = self._split_heads(self.k_proj(x))
        v = self._split_heads(self.v_proj(x))

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len),
                device=x.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=x.dtype)
        context = torch.matmul(probs, v)

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )

        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)

        return output


class BaselineTransformerBlock(nn.Module):
    def __init__(self, d_model: int, num_heads: int, ffn_dim: int) -> None:
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attention = BaselineSelfAttention(d_model, num_heads)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(self.norm1(x), valid_token_mask, causal)
        x = x + self.ffn_out(
            F.gelu(
                self.ffn_in(self.norm2(x)),
                approximate="none",
            )
        )

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


class BaselineTransformer(nn.Module):
    def __init__(self, config: TransformerConfig) -> None:
        super().__init__()
        self.config = config
        self.layers = nn.ModuleList(
            [
                BaselineTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                )
                for _ in range(config.num_layers)
            ]
        )
        self.final_norm = nn.LayerNorm(config.d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            x = layer(x, valid_token_mask, self.config.causal)

        x = self.final_norm(x)

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


# =============================================================================
# Generic ablation implementation
# =============================================================================

class AblationSelfAttention(nn.Module):
    """
    Attention implementation controlled by a VariantSpec.

    packed_qkv=False:
        use three independent q_proj / k_proj / v_proj layers.

    packed_qkv=True:
        use one qkv_proj layer.

    sdpa=False:
        reproduce the explicit baseline attention path.

    sdpa=True:
        use torch.nn.functional.scaled_dot_product_attention.
    """

    def __init__(
        self,
        d_model: int,
        num_heads: int,
        *,
        packed_qkv: bool,
        sdpa: bool,
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        if d_model % num_heads != 0:
            raise ValueError("d_model must be divisible by num_heads")

        self.d_model = d_model
        self.num_heads = num_heads
        self.head_dim = d_model // num_heads
        self.scale = self.head_dim**-0.5
        self.packed_qkv = packed_qkv
        self.use_sdpa = sdpa
        self.sdpa_backend = sdpa_backend

        if packed_qkv:
            self.qkv_proj = nn.Linear(d_model, 3 * d_model, bias=True)
        else:
            self.q_proj = nn.Linear(d_model, d_model, bias=True)
            self.k_proj = nn.Linear(d_model, d_model, bias=True)
            self.v_proj = nn.Linear(d_model, d_model, bias=True)

        self.out_proj = nn.Linear(d_model, d_model, bias=True)

    def _project_qkv(
        self,
        x: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch, seq_len, _ = x.shape

        if self.packed_qkv:
            qkv = self.qkv_proj(x).view(
                batch,
                seq_len,
                3,
                self.num_heads,
                self.head_dim,
            )
            q, k, v = qkv.unbind(dim=2)

            q = q.transpose(1, 2)
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)

            # Manual attention is deliberately kept close to the baseline's
            # layout behavior for a cleaner packed-QKV ablation.
            if not self.use_sdpa:
                q = q.contiguous()
                k = k.contiguous()
                v = v.contiguous()

            return q, k, v

        q = (
            self.q_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        k = (
            self.k_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )
        v = (
            self.v_proj(x)
            .view(batch, seq_len, self.num_heads, self.head_dim)
            .transpose(1, 2)
        )

        # The exact baseline materializes contiguous head layouts.
        if not self.use_sdpa:
            q = q.contiguous()
            k = k.contiguous()
            v = v.contiguous()

        return q, k, v

    def _manual_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        seq_len = q.shape[-2]

        scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale

        if causal:
            causal_mask = torch.ones(
                (seq_len, seq_len),
                device=q.device,
                dtype=torch.bool,
            ).triu(diagonal=1)
            scores = scores.masked_fill(causal_mask, float("-inf"))

        if valid_token_mask is not None:
            invalid_keys = ~valid_token_mask[:, None, None, :]
            scores = scores.masked_fill(invalid_keys, float("-inf"))

        probs = torch.softmax(scores.float(), dim=-1).to(dtype=q.dtype)
        return torch.matmul(probs, v)

    def _sdpa_attention(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        seq_len = q.shape[-2]

        attn_mask: Optional[torch.Tensor] = None
        use_is_causal = causal

        if valid_token_mask is not None:
            key_mask = valid_token_mask[:, None, None, :]

            if causal:
                # Boolean SDPA masks use True = allowed.
                causal_allowed = torch.ones(
                    (seq_len, seq_len),
                    device=q.device,
                    dtype=torch.bool,
                ).tril()

                attn_mask = key_mask & causal_allowed[None, None, :, :]
                use_is_causal = False
            else:
                attn_mask = key_mask

        with sdpa_backend_context(self.sdpa_backend, q.device):
            return F.scaled_dot_product_attention(
                q,
                k,
                v,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=use_is_causal,
            )

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
        causal: bool = False,
    ) -> torch.Tensor:
        batch, seq_len, _ = x.shape

        q, k, v = self._project_qkv(x)

        if self.use_sdpa:
            context = self._sdpa_attention(
                q,
                k,
                v,
                valid_token_mask,
                causal,
            )
        else:
            context = self._manual_attention(
                q,
                k,
                v,
                valid_token_mask,
                causal,
            )

        context = (
            context.transpose(1, 2)
            .contiguous()
            .view(batch, seq_len, self.d_model)
        )

        output = self.out_proj(context)

        if valid_token_mask is not None:
            output = output.masked_fill(~valid_token_mask[..., None], 0)

        return output


class AblationTransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        ffn_dim: int,
        *,
        packed_qkv: bool,
        sdpa: bool,
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        self.norm1 = nn.LayerNorm(d_model)
        self.attention = AblationSelfAttention(
            d_model,
            num_heads,
            packed_qkv=packed_qkv,
            sdpa=sdpa,
            sdpa_backend=sdpa_backend,
        )
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn_in = nn.Linear(d_model, ffn_dim)
        self.ffn_out = nn.Linear(ffn_dim, d_model)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor],
        causal: bool,
    ) -> torch.Tensor:
        x = x + self.attention(
            self.norm1(x),
            valid_token_mask,
            causal,
        )

        x = x + self.ffn_out(
            F.gelu(
                self.ffn_in(self.norm2(x)),
                approximate="none",
            )
        )

        if valid_token_mask is not None:
            x = x.masked_fill(~valid_token_mask[..., None], 0)

        return x


class AblationTransformer(nn.Module):
    def __init__(
        self,
        config: TransformerConfig,
        spec: VariantSpec,
        sdpa_backend: str = "auto",
    ) -> None:
        super().__init__()

        if spec.key == "A":
            raise ValueError("Variant A uses BaselineTransformer directly")

        self.config = config
        self.spec = spec

        self.layers = nn.ModuleList(
            [
                AblationTransformerBlock(
                    config.d_model,
                    config.num_heads,
                    config.ffn_dim,
                    packed_qkv=spec.packed_qkv,
                    sdpa=spec.sdpa,
                    sdpa_backend=sdpa_backend,
                )
                for _ in range(config.num_layers)
            ]
        )

        self.final_norm = nn.LayerNorm(config.d_model)
        self._assume_all_valid_mask = False

    def set_assume_all_valid_mask(self, enabled: bool) -> None:
        self._assume_all_valid_mask = bool(enabled)

    def forward(
        self,
        x: torch.Tensor,
        valid_token_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        effective_mask = valid_token_mask

        if self.spec.skip_all_valid_mask and self._assume_all_valid_mask:
            effective_mask = None

        for layer in self.layers:
            x = layer(x, effective_mask, self.config.causal)

        x = self.final_norm(x)

        if effective_mask is not None:
            x = x.masked_fill(~effective_mask[..., None], 0)

        return x


# =============================================================================
# Weight copy
# =============================================================================

def copy_baseline_weights(
    baseline: BaselineTransformer,
    candidate: AblationTransformer,
) -> None:
    if len(baseline.layers) != len(candidate.layers):
        raise ValueError("baseline/candidate layer count mismatch")

    with torch.no_grad():
        for src_layer, dst_layer in zip(
            baseline.layers,
            candidate.layers,
        ):
            dst_layer.norm1.load_state_dict(
                copy.deepcopy(src_layer.norm1.state_dict())
            )
            dst_layer.norm2.load_state_dict(
                copy.deepcopy(src_layer.norm2.state_dict())
            )
            dst_layer.ffn_in.load_state_dict(
                copy.deepcopy(src_layer.ffn_in.state_dict())
            )
            dst_layer.ffn_out.load_state_dict(
                copy.deepcopy(src_layer.ffn_out.state_dict())
            )
            dst_layer.attention.out_proj.load_state_dict(
                copy.deepcopy(src_layer.attention.out_proj.state_dict())
            )

            if candidate.spec.packed_qkv:
                dst_layer.attention.qkv_proj.weight.copy_(
                    torch.cat(
                        (
                            src_layer.attention.q_proj.weight,
                            src_layer.attention.k_proj.weight,
                            src_layer.attention.v_proj.weight,
                        ),
                        dim=0,
                    )
                )
                dst_layer.attention.qkv_proj.bias.copy_(
                    torch.cat(
                        (
                            src_layer.attention.q_proj.bias,
                            src_layer.attention.k_proj.bias,
                            src_layer.attention.v_proj.bias,
                        ),
                        dim=0,
                    )
                )
            else:
                dst_layer.attention.q_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.q_proj.state_dict())
                )
                dst_layer.attention.k_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.k_proj.state_dict())
                )
                dst_layer.attention.v_proj.load_state_dict(
                    copy.deepcopy(src_layer.attention.v_proj.state_dict())
                )

        candidate.final_norm.load_state_dict(
            copy.deepcopy(baseline.final_norm.state_dict())
        )


# =============================================================================
# Data and correctness
# =============================================================================

def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if (
            hasattr(torch.backends, "mps")
            and torch.backends.mps.is_available()
        ):
            return torch.device("mps")
        return torch.device("cpu")

    device = torch.device(device_arg)

    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is False"
        )

    if device.type == "mps":
        if (
            not hasattr(torch.backends, "mps")
            or not torch.backends.mps.is_available()
        ):
            raise RuntimeError(
                "MPS was requested, but torch.backends.mps.is_available() is False"
            )

    return device


def resolve_dtype(dtype_name: str) -> torch.dtype:
    return {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }[dtype_name]


def generate_random_case(
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator(device=device)
    generator.manual_seed(seed)

    x = torch.randn(
        config.batch_size,
        config.seq_len,
        config.d_model,
        generator=generator,
        device=device,
        dtype=dtype,
    )
    x = x * input_scale

    if padding_ratio <= 0:
        valid_token_mask = torch.ones(
            config.batch_size,
            config.seq_len,
            device=device,
            dtype=torch.bool,
        )
        return x, valid_token_mask

    min_valid = max(
        1,
        int(round(config.seq_len * (1.0 - padding_ratio))),
    )

    lengths = torch.randint(
        low=min_valid,
        high=config.seq_len + 1,
        size=(config.batch_size,),
        generator=generator,
        device=device,
    )

    positions = torch.arange(
        config.seq_len,
        device=device,
    )[None, :]

    valid_token_mask = positions < lengths[:, None]
    x = x.masked_fill(~valid_token_mask[..., None], 0)

    return x, valid_token_mask


@dataclass
class AccuracySummary:
    passed: bool = True
    total_elements: int = 0
    failed_elements: int = 0
    max_abs_error: float = 0.0
    max_relative_error: float = 0.0
    mean_abs_error_sum: float = 0.0
    trials: int = 0

    @property
    def mean_abs_error(self) -> float:
        if self.trials == 0:
            return 0.0
        return self.mean_abs_error_sum / self.trials


def update_accuracy_summary(
    summary: AccuracySummary,
    reference: torch.Tensor,
    candidate: torch.Tensor,
    *,
    rtol: float,
    atol: float,
) -> None:
    if reference.shape != candidate.shape:
        raise AssertionError(
            f"shape mismatch: reference={tuple(reference.shape)} "
            f"candidate={tuple(candidate.shape)}"
        )

    ref = reference.detach().float()
    opt = candidate.detach().float()

    finite_mask = torch.isfinite(ref) & torch.isfinite(opt)
    abs_error = (opt - ref).abs()

    abs_ok = abs_error <= atol
    rel_ok = abs_error <= rtol * ref.abs()
    passed_mask = finite_mask & (abs_ok | rel_ok)

    failed = int((~passed_mask).sum().item())

    denominator = ref.abs().clamp_min(1e-12)
    relative_error = abs_error / denominator

    summary.passed = summary.passed and (failed == 0)
    summary.total_elements += reference.numel()
    summary.failed_elements += failed
    summary.max_abs_error = max(
        summary.max_abs_error,
        float(abs_error.max().item()),
    )
    summary.max_relative_error = max(
        summary.max_relative_error,
        float(relative_error.max().item()),
    )
    summary.mean_abs_error_sum += float(abs_error.mean().item())
    summary.trials += 1


def run_accuracy_suite(
    models: Dict[str, nn.Module],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    trials: int,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    rtol: float,
    atol: float,
) -> Dict[str, AccuracySummary]:
    print("\n=== Accuracy ablation ===")
    print(
        f"criterion: abs_error <= {atol:g} OR "
        f"relative_error <= {rtol:.2%}"
    )

    summaries: Dict[str, AccuracySummary] = {
        key: AccuracySummary()
        for key in variant_keys
        if key != "A"
    }

    if not summaries:
        return {}

    with torch.inference_mode():
        for trial in range(trials):
            x, valid_mask = generate_random_case(
                config=config,
                device=device,
                dtype=dtype,
                seed=seed + trial,
                padding_ratio=padding_ratio,
                input_scale=input_scale,
            )

            reference = models["A"](x, valid_mask)

            for key in variant_keys:
                if key == "A":
                    continue

                candidate = models[key](x, valid_mask)

                update_accuracy_summary(
                    summaries[key],
                    reference,
                    candidate,
                    rtol=rtol,
                    atol=atol,
                )

    print(
        f"{'Var':<4} {'Name':<25} {'Status':<7} "
        f"{'Max abs':>12} {'Max rel':>12} {'Failed':>14}"
    )
    print("-" * 84)

    for key in variant_keys:
        if key == "A":
            print(
                f"{'A':<4} {'baseline':<25} {'REF':<7} "
                f"{0.0:>12.6g} {0.0:>12.6g} {'0':>14}"
            )
            continue

        s = summaries[key]
        status = "PASS" if s.passed else "FAIL"

        print(
            f"{key:<4} {VARIANT_SPECS[key].name:<25} {status:<7} "
            f"{s.max_abs_error:>12.6g} "
            f"{s.max_relative_error:>12.6g} "
            f"{s.failed_elements:>14d}"
        )

    return summaries


# =============================================================================
# Timing
# =============================================================================

@dataclass
class TimingResult:
    samples_ms: List[float]

    @property
    def mean_ms(self) -> float:
        return statistics.fmean(self.samples_ms)

    @property
    def median_ms(self) -> float:
        return statistics.median(self.samples_ms)

    @property
    def p90_ms(self) -> float:
        return percentile(self.samples_ms, 0.90)

    @property
    def min_ms(self) -> float:
        return min(self.samples_ms)


def percentile(values: List[float], q: float) -> float:
    if not values:
        raise ValueError("values must not be empty")

    ordered = sorted(values)

    if len(ordered) == 1:
        return ordered[0]

    position = (len(ordered) - 1) * q
    lower = math.floor(position)
    upper = math.ceil(position)

    if lower == upper:
        return ordered[lower]

    weight = position - lower

    return (
        ordered[lower] * (1.0 - weight)
        + ordered[upper] * weight
    )


def synchronize_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def warmup_model(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> None:
    with torch.inference_mode():
        for _ in range(iterations):
            model(x, valid_mask)

    synchronize_device(device)


def benchmark_once(
    model: nn.Module,
    x: torch.Tensor,
    valid_mask: torch.Tensor,
    iterations: int,
    device: torch.device,
) -> List[float]:
    samples_ms: List[float] = []

    with torch.inference_mode():
        if device.type == "cuda":
            starts = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]
            ends = [
                torch.cuda.Event(enable_timing=True)
                for _ in range(iterations)
            ]

            torch.cuda.synchronize(device)

            for index in range(iterations):
                starts[index].record()
                model(x, valid_mask)
                ends[index].record()

            torch.cuda.synchronize(device)

            samples_ms.extend(
                start.elapsed_time(end)
                for start, end in zip(starts, ends)
            )

        elif device.type == "mps":
            # MPS is asynchronous, so synchronize around every measured call.
            for _ in range(iterations):
                torch.mps.synchronize()
                start = time.perf_counter_ns()
                model(x, valid_mask)
                torch.mps.synchronize()
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

        else:
            for _ in range(iterations):
                start = time.perf_counter_ns()
                model(x, valid_mask)
                end = time.perf_counter_ns()
                samples_ms.append((end - start) / 1e6)

    return samples_ms


def rotated_order(
    keys: Sequence[str],
    round_index: int,
) -> List[str]:
    """
    Rotate and reverse measurement order to reduce systematic order bias.

    Round 0: A B C D ...
    Round 1: ... D C B A
    Round 2: C D ... A B
    """
    keys = list(keys)

    if not keys:
        return []

    shift = (round_index // 2) % len(keys)
    order = keys[shift:] + keys[:shift]

    if round_index % 2 == 1:
        order.reverse()

    return order


def benchmark_variants(
    models: Dict[str, nn.Module],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
    padding_ratio: float,
    input_scale: float,
    warmup: int,
    repeats: int,
    rounds: int,
) -> Dict[str, TimingResult]:
    print("\n=== Performance ablation ===")
    print("all variants use the same fixed benchmark input")

    if device.type == "cuda":
        print("CUDA: torch.cuda.Event timing")
    elif device.type == "mps":
        print("MPS: synchronized wall-clock timing")
    else:
        print("CPU: wall-clock timing")

    x, valid_mask = generate_random_case(
        config=config,
        device=device,
        dtype=dtype,
        seed=seed + 100000,
        padding_ratio=padding_ratio,
        input_scale=input_scale,
    )

    print("warming up:", ", ".join(variant_keys))

    for key in variant_keys:
        warmup_model(
            models[key],
            x,
            valid_mask,
            warmup,
            device,
        )

    samples: Dict[str, List[float]] = {
        key: []
        for key in variant_keys
    }

    for round_index in range(rounds):
        order = rotated_order(variant_keys, round_index)
        print(
            f"round {round_index + 1}/{rounds}: "
            + " -> ".join(order)
        )

        for key in order:
            samples[key].extend(
                benchmark_once(
                    models[key],
                    x,
                    valid_mask,
                    repeats,
                    device,
                )
            )

    return {
        key: TimingResult(samples[key])
        for key in variant_keys
    }


# =============================================================================
# Reporting
# =============================================================================

def print_variant_definition_table(
    variant_keys: Sequence[str],
    mask_skip_effective: bool,
) -> None:
    print("\n=== Variants ===")
    print(
        f"{'Var':<4} {'Name':<25} "
        f"{'Packed QKV':<12} {'SDPA':<8} {'Mask skip':<12}"
    )
    print("-" * 70)

    for key in variant_keys:
        spec = VARIANT_SPECS[key]

        if key == "A":
            mask_text = "no"
        elif spec.skip_all_valid_mask:
            mask_text = "yes" if mask_skip_effective else "inactive"
        else:
            mask_text = "no"

        print(
            f"{key:<4} "
            f"{spec.name:<25} "
            f"{('yes' if spec.packed_qkv else 'no'):<12} "
            f"{('yes' if spec.sdpa else 'no'):<8} "
            f"{mask_text:<12}"
        )


def print_timing_table(
    timings: Dict[str, TimingResult],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    accuracy: Dict[str, AccuracySummary],
) -> None:
    baseline_ms = timings["A"].median_ms
    tokens_per_call = config.batch_size * config.seq_len

    print("\n=== Ablation summary ===")
    print(
        f"{'Var':<4} {'Name':<25} {'Accuracy':<9} "
        f"{'Median ms':>11} {'Speedup':>9} {'Latency Δ':>10} "
        f"{'Token/s':>13}"
    )
    print("-" * 92)

    for key in variant_keys:
        result = timings[key]
        speedup = baseline_ms / result.median_ms
        latency_delta = (
            (result.median_ms / baseline_ms) - 1.0
        ) * 100.0
        token_s = (
            tokens_per_call * 1000.0 / result.median_ms
        )

        if key == "A":
            accuracy_text = "REF"
        else:
            accuracy_text = (
                "PASS"
                if accuracy[key].passed
                else "FAIL"
            )

        print(
            f"{key:<4} "
            f"{VARIANT_SPECS[key].name:<25} "
            f"{accuracy_text:<9} "
            f"{result.median_ms:>11.4f} "
            f"{speedup:>8.3f}x "
            f"{latency_delta:>+9.2f}% "
            f"{token_s:>13.2f}"
        )

    print("\nLatency distribution:")
    print(
        f"{'Var':<4} {'Mean ms':>11} {'P90 ms':>11} {'Min ms':>11}"
    )
    print("-" * 44)

    for key in variant_keys:
        result = timings[key]
        print(
            f"{key:<4} "
            f"{result.mean_ms:>11.4f} "
            f"{result.p90_ms:>11.4f} "
            f"{result.min_ms:>11.4f}"
        )


def pairwise_speedup(
    timings: Dict[str, TimingResult],
    before: str,
    after: str,
) -> Optional[float]:
    if before not in timings or after not in timings:
        return None

    return (
        timings[before].median_ms
        / timings[after].median_ms
    )


def geometric_mean(values: Iterable[float]) -> float:
    vals = list(values)

    if not vals:
        raise ValueError("cannot compute geometric mean of an empty set")

    return math.exp(
        statistics.fmean(math.log(v) for v in vals)
    )


def print_effect_analysis(
    timings: Dict[str, TimingResult],
) -> None:
    """
    Report matched comparisons in the full 2^3 design.

    For each factor, the factorial estimate is the geometric mean of four
    matched speedup ratios with that factor toggled while the other two
    factors are held fixed.
    """

    print("\n=== Optimization effect analysis ===")

    comparisons = {
        "Packed QKV": [
            ("A", "B"),
            ("C", "E"),
            ("D", "F"),
            ("G", "H"),
        ],
        "SDPA": [
            ("A", "C"),
            ("B", "E"),
            ("D", "G"),
            ("F", "H"),
        ],
        "Mask skip": [
            ("A", "D"),
            ("B", "F"),
            ("C", "G"),
            ("E", "H"),
        ],
    }

    any_effect = False

    for factor, pairs in comparisons.items():
        ratios: List[float] = []
        labels: List[str] = []

        for before, after in pairs:
            ratio = pairwise_speedup(
                timings,
                before,
                after,
            )

            if ratio is not None:
                ratios.append(ratio)
                labels.append(
                    f"{before}->{after} {ratio:.3f}x"
                )

        if not ratios:
            continue

        any_effect = True
        average_effect = geometric_mean(ratios)

        print(
            f"{factor:<12}: "
            f"matched geometric-mean effect = {average_effect:.3f}x"
        )
        print(" " * 14 + " | ".join(labels))

    if not any_effect:
        print(
            "Need more matched variants to estimate factor effects."
        )

    if "A" in timings and "H" in timings:
        full_speedup = (
            timings["A"].median_ms
            / timings["H"].median_ms
        )
        print(
            f"\nFull A->H speedup: {full_speedup:.3f}x"
        )

    print(
        "\nNote: factor effects need not multiply exactly because "
        "optimizations interact."
    )


def save_csv_results(
    path: str,
    timings: Dict[str, TimingResult],
    variant_keys: Sequence[str],
    config: TransformerConfig,
    accuracy: Dict[str, AccuracySummary],
    device: torch.device,
    dtype: torch.dtype,
    padding_ratio: float,
    sdpa_backend: str,
    shape_id: Optional[int] = None,
    append: bool = False,
) -> None:
    baseline_ms = timings["A"].median_ms
    tokens_per_call = config.batch_size * config.seq_len

    output_path = Path(path)
    output_path.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    should_write_header = (not append) or (not output_path.exists())
    mode = "a" if append else "w"

    with output_path.open(
        mode,
        newline="",
        encoding="utf-8",
    ) as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "shape_id",
                "variant",
                "name",
                "packed_qkv",
                "sdpa",
                "sdpa_backend",
                "mask_skip",
                "accuracy",
                "failed_elements",
                "max_abs_error",
                "max_relative_error",
                "median_ms",
                "mean_ms",
                "p90_ms",
                "min_ms",
                "speedup_vs_A",
                "latency_delta_percent",
                "tokens_per_second",
                "device",
                "dtype",
                "batch_size",
                "seq_len",
                "d_model",
                "heads",
                "ffn_dim",
                "layers",
                "causal",
                "padding_ratio",
            ],
        )
        if should_write_header:
            writer.writeheader()

        for key in variant_keys:
            spec = VARIANT_SPECS[key]
            result = timings[key]

            if key == "A":
                accuracy_text = "REF"
                failed = 0
                max_abs = 0.0
                max_rel = 0.0
            else:
                s = accuracy[key]
                accuracy_text = (
                    "PASS"
                    if s.passed
                    else "FAIL"
                )
                failed = s.failed_elements
                max_abs = s.max_abs_error
                max_rel = s.max_relative_error

            writer.writerow(
                {
                    "shape_id": shape_id if shape_id is not None else "",
                    "variant": key,
                    "name": spec.name,
                    "packed_qkv": spec.packed_qkv,
                    "sdpa": spec.sdpa,
                    "sdpa_backend": sdpa_backend if spec.sdpa else "",
                    "mask_skip": spec.skip_all_valid_mask,
                    "accuracy": accuracy_text,
                    "failed_elements": failed,
                    "max_abs_error": max_abs,
                    "max_relative_error": max_rel,
                    "median_ms": result.median_ms,
                    "mean_ms": result.mean_ms,
                    "p90_ms": result.p90_ms,
                    "min_ms": result.min_ms,
                    "speedup_vs_A": (
                        baseline_ms / result.median_ms
                    ),
                    "latency_delta_percent": (
                        (result.median_ms / baseline_ms) - 1.0
                    ) * 100.0,
                    "tokens_per_second": (
                        tokens_per_call
                        * 1000.0
                        / result.median_ms
                    ),
                    "device": str(device),
                    "dtype": str(dtype),
                    "batch_size": config.batch_size,
                    "seq_len": config.seq_len,
                    "d_model": config.d_model,
                    "heads": config.num_heads,
                    "ffn_dim": config.ffn_dim,
                    "layers": config.num_layers,
                    "causal": config.causal,
                    "padding_ratio": padding_ratio,
                }
            )

    print(f"\nCSV written to: {output_path}")


# =============================================================================
# Compilation
# =============================================================================

def maybe_compile(
    model: nn.Module,
    enabled: bool,
    mode: str,
) -> nn.Module:
    if not enabled:
        return model

    if not hasattr(torch, "compile"):
        raise RuntimeError(
            "this PyTorch build does not provide torch.compile"
        )

    return torch.compile(
        model,
        mode=mode,
    )


# =============================================================================
# CLI
# =============================================================================

def parse_variant_keys(raw: str) -> List[str]:
    raw = raw.strip().upper()

    if raw == "ALL":
        return list(VARIANT_SPECS.keys())

    keys = [
        item.strip()
        for item in raw.split(",")
        if item.strip()
    ]

    if "A" not in keys:
        # Baseline is always required for reference correctness and speedup.
        keys.insert(0, "A")

    unknown = [
        key
        for key in keys
        if key not in VARIANT_SPECS
    ]

    if unknown:
        raise ValueError(
            "unknown variants: "
            + ", ".join(unknown)
            + ". Valid variants are A,B,C,D,E,F,G,H"
        )

    # Stable deduplication.
    deduped: List[str] = []

    for key in keys:
        if key not in deduped:
            deduped.append(key)

    return deduped


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Automatic Transformer optimization ablation benchmark"
        )
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--seq-len",
        type=int,
        default=128,
    )
    parser.add_argument(
        "--d-model",
        type=int,
        default=512,
    )
    parser.add_argument(
        "--heads",
        type=int,
        default=8,
    )
    parser.add_argument(
        "--ffn-dim",
        type=int,
        default=2048,
    )
    parser.add_argument(
        "--layers",
        type=int,
        default=6,
    )
    parser.add_argument(
        "--causal",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Use causal attention. Default: enabled for competition parity.",
    )

    parser.add_argument(
        "--device",
        default="auto",
        help="auto, cpu, mps, cuda, cuda:0, ...",
    )
    parser.add_argument(
        "--dtype",
        choices=(
            "float32",
            "float16",
            "bfloat16",
        ),
        default="float32",
    )

    parser.add_argument(
        "--padding-ratio",
        type=float,
        default=0.0,
    )
    parser.add_argument(
        "--input-scale",
        type=float,
        default=1.0,
    )

    parser.add_argument(
        "--accuracy-trials",
        type=int,
        default=5,
    )
    parser.add_argument(
        "--rtol",
        type=float,
        default=0.02,
    )
    parser.add_argument(
        "--atol",
        type=float,
        default=0.002,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=1234,
    )

    parser.add_argument(
        "--warmup",
        type=int,
        default=20,
    )
    parser.add_argument(
        "--repeats",
        type=int,
        default=100,
    )
    parser.add_argument(
        "--benchmark-rounds",
        type=int,
        default=3,
    )

    parser.add_argument(
        "--variants",
        default="all",
        help=(
            "Comma-separated variants, e.g. A,B,C,H. "
            "Default: all"
        ),
    )

    parser.add_argument(
        "--benchmark-on-failure",
        action="store_true",
        help=(
            "Benchmark even if one or more candidate variants "
            "fail correctness."
        ),
    )

    parser.add_argument(
        "--compile",
        action="store_true",
        help=(
            "Compile every selected model, including baseline A, "
            "before benchmarking. Disabled by default for clean "
            "operator ablations."
        ),
    )
    parser.add_argument(
        "--compile-mode",
        choices=(
            "default",
            "reduce-overhead",
            "max-autotune",
        ),
        default="default",
    )

    parser.add_argument(
        "--matmul-precision",
        choices=(
            "highest",
            "high",
            "medium",
        ),
        default="high",
    )

    parser.add_argument(
        "--allow-tf32",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Enable/disable TF32 on CUDA.",
    )

    parser.add_argument(
        "--sdpa-backend",
        choices=("auto", "flash", "efficient", "cudnn", "math"),
        default="auto",
        help=(
            "Backend used by SDPA variants on CUDA. 'auto' lets PyTorch "
            "choose. Backend forcing is CUDA-only."
        ),
    )

    parser.add_argument(
        "--competition-shapes",
        action="store_true",
        help="Sweep official challenge shapes instead of the single CLI shape.",
    )
    parser.add_argument(
        "--shape-ids",
        default="1-13",
        help=(
            "Competition shape IDs, e.g. '1-13', '1,2,7-9', or 'all'. "
            "Default excludes shape 14 because explicit attention is enormous."
        ),
    )
    parser.add_argument(
        "--max-manual-attention-gib",
        type=float,
        default=12.0,
        help=(
            "Safety guard for estimated score+softmax buffers in explicit "
            "attention. Set <=0 to disable the guard."
        ),
    )
    parser.add_argument(
        "--force-manual-attention",
        action="store_true",
        help="Override the explicit-attention memory guard (OOM risk).",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help=(
            "Fast development pass: accuracy=1, warmup=2, repeats=10, rounds=1."
        ),
    )

    parser.add_argument(
        "--csv",
        default=None,
        help=(
            "Optional CSV output path, e.g. ablation_results.csv"
        ),
    )

    return parser.parse_args()


def validate_args(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
) -> None:
    if not 0.0 <= args.padding_ratio < 1.0:
        raise ValueError(
            "padding_ratio must be in [0, 1)"
        )

    if args.input_scale <= 0:
        raise ValueError(
            "input_scale must be positive"
        )

    if args.accuracy_trials <= 0:
        raise ValueError(
            "accuracy_trials must be positive"
        )

    if args.rtol < 0 or args.atol < 0:
        raise ValueError(
            "rtol and atol must be non-negative"
        )

    if args.warmup < 0:
        raise ValueError(
            "warmup must be non-negative"
        )

    if (
        args.repeats <= 0
        or args.benchmark_rounds <= 0
    ):
        raise ValueError(
            "repeats and benchmark_rounds must be positive"
        )

    if args.sdpa_backend != "auto" and device.type != "cuda":
        raise ValueError(
            "forcing --sdpa-backend requires a CUDA device; use 'auto' on CPU/MPS"
        )

    if args.max_manual_attention_gib < 0:
        raise ValueError("max_manual_attention_gib must be >= 0")

    if (
        device.type == "cpu"
        and dtype == torch.float16
    ):
        print(
            "[warning] float16 CPU kernels may be unsupported or slow"
        )


# =============================================================================
# Main
# =============================================================================

@dataclass
class ShapeRunResult:
    shape_id: Optional[int]
    config: TransformerConfig
    timings: Dict[str, TimingResult]
    accuracy: Dict[str, AccuracySummary]


def run_one_configuration(
    args: argparse.Namespace,
    device: torch.device,
    dtype: torch.dtype,
    variant_keys: Sequence[str],
    config: TransformerConfig,
    *,
    shape_id: Optional[int] = None,
    csv_append: bool = False,
) -> Optional[ShapeRunResult]:
    config.validate()

    estimated_gib = estimate_manual_attention_working_set_gib(config, dtype)
    label = f"competition shape {shape_id}" if shape_id is not None else "configuration"

    print("\n" + "=" * 96)
    print(f"=== {label} ===")
    print(config)
    print(
        f"estimated explicit-attention score+softmax working set "
        f"(lower bound)={estimated_gib:.2f} GiB"
    )

    guard_enabled = args.max_manual_attention_gib > 0
    unsafe_manual = (
        guard_enabled
        and estimated_gib > args.max_manual_attention_gib
        and not args.force_manual_attention
    )
    if unsafe_manual:
        print(
            f"[SKIP] explicit baseline estimate exceeds "
            f"--max-manual-attention-gib={args.max_manual_attention_gib:g}."
        )
        print(
            "       The A-H harness requires explicit variant A for correctness, "
            "so this shape is not safe here. Use the official reference harness "
            "or raise the guard only if you know the target can accommodate it."
        )
        return None

    baseline = BaselineTransformer(config)
    models: Dict[str, nn.Module] = {"A": baseline}

    for key in variant_keys:
        if key == "A":
            continue

        candidate = AblationTransformer(
            config,
            VARIANT_SPECS[key],
            sdpa_backend=args.sdpa_backend,
        )
        copy_baseline_weights(baseline, candidate)
        candidate.set_assume_all_valid_mask(args.padding_ratio == 0.0)
        models[key] = candidate

    for key in variant_keys:
        models[key] = (
            models[key]
            .to(device=device, dtype=dtype)
            .eval()
        )

    if args.compile:
        print(
            f"[compile] compiling all selected variants "
            f"with mode={args.compile_mode}"
        )
        for key in variant_keys:
            models[key] = maybe_compile(
                models[key],
                enabled=True,
                mode=args.compile_mode,
            )

    print("=== Configuration ===")
    print(
        f"device={device}, dtype={dtype}, torch={torch.__version__}, "
        f"sdpa_backend={args.sdpa_backend}"
    )
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(device)}")

    print(
        f"padding_ratio={args.padding_ratio:g}, "
        f"mask-skip effective={args.padding_ratio == 0.0}"
    )
    print(f"selected variants={','.join(variant_keys)}")

    print_variant_definition_table(
        variant_keys,
        mask_skip_effective=(args.padding_ratio == 0.0),
    )

    accuracy = run_accuracy_suite(
        models=models,
        variant_keys=variant_keys,
        config=config,
        device=device,
        dtype=dtype,
        trials=args.accuracy_trials,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        rtol=args.rtol,
        atol=args.atol,
    )

    failed_variants = [
        key
        for key, summary in accuracy.items()
        if not summary.passed
    ]
    if failed_variants:
        print("\nCorrectness failures: " + ", ".join(failed_variants))
        if not args.benchmark_on_failure:
            print(
                "Performance benchmark skipped for this configuration. "
                "Use --benchmark-on-failure to time failures anyway."
            )
            return ShapeRunResult(shape_id, config, {}, accuracy)

    timings = benchmark_variants(
        models=models,
        variant_keys=variant_keys,
        config=config,
        device=device,
        dtype=dtype,
        seed=args.seed,
        padding_ratio=args.padding_ratio,
        input_scale=args.input_scale,
        warmup=args.warmup,
        repeats=args.repeats,
        rounds=args.benchmark_rounds,
    )

    print_timing_table(
        timings=timings,
        variant_keys=variant_keys,
        config=config,
        accuracy=accuracy,
    )
    print_effect_analysis(timings)

    if args.csv:
        save_csv_results(
            path=args.csv,
            timings=timings,
            variant_keys=variant_keys,
            config=config,
            accuracy=accuracy,
            device=device,
            dtype=dtype,
            padding_ratio=args.padding_ratio,
            sdpa_backend=args.sdpa_backend,
            shape_id=shape_id,
            append=csv_append,
        )

    # Release large per-shape allocations before the next sweep item.
    del models
    del baseline
    if device.type == "cuda":
        torch.cuda.empty_cache()

    return ShapeRunResult(shape_id, config, timings, accuracy)


def print_competition_sweep_summary(
    results: Sequence[Optional[ShapeRunResult]],
    variant_keys: Sequence[str],
) -> None:
    print("\n=== Competition sweep summary ===")
    print(
        f"{'Shape':>5} {'B':>7} {'S':>8} {'D':>6} {'H':>4} "
        f"{'Best':>6} {'Best speedup':>13} {'H speedup':>11}"
    )
    print("-" * 76)

    for result in results:
        if result is None:
            continue
        cfg = result.config
        if not result.timings:
            print(
                f"{str(result.shape_id):>5} {cfg.batch_size:>7} {cfg.seq_len:>8} "
                f"{cfg.d_model:>6} {cfg.num_heads:>4} {'FAIL':>6} {'-':>13} {'-':>11}"
            )
            continue

        baseline_ms = result.timings["A"].median_ms
        valid_keys = [
            key for key in variant_keys
            if key in result.timings
            and (key == "A" or result.accuracy.get(key, AccuracySummary()).passed)
        ]
        best_key = min(valid_keys, key=lambda key: result.timings[key].median_ms)
        best_speedup = baseline_ms / result.timings[best_key].median_ms

        h_text = "-"
        if "H" in result.timings and result.accuracy.get("H", AccuracySummary()).passed:
            h_text = f"{baseline_ms / result.timings['H'].median_ms:.3f}x"

        print(
            f"{str(result.shape_id):>5} {cfg.batch_size:>7} {cfg.seq_len:>8} "
            f"{cfg.d_model:>6} {cfg.num_heads:>4} {best_key:>6} "
            f"{best_speedup:>12.3f}x {h_text:>11}"
        )


def main() -> int:
    args = parse_args()

    if args.quick:
        args.accuracy_trials = 1
        args.warmup = 2
        args.repeats = 10
        args.benchmark_rounds = 1
        print(
            "[quick] accuracy_trials=1, warmup=2, repeats=10, benchmark_rounds=1"
        )

    device = resolve_device(args.device)
    dtype = resolve_dtype(args.dtype)
    variant_keys = parse_variant_keys(args.variants)
    validate_args(args, device, dtype)

    torch.manual_seed(args.seed)
    torch.set_float32_matmul_precision(args.matmul_precision)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
        torch.backends.cuda.matmul.allow_tf32 = args.allow_tf32
        torch.backends.cudnn.allow_tf32 = args.allow_tf32

    if args.competition_shapes:
        shape_ids = parse_shape_ids(args.shape_ids)
        print(
            "=== Official competition sweep ===\n"
            f"shape_ids={shape_ids}\n"
            "Note: shape 14 is intentionally excluded by the default --shape-ids 1-13."
        )

        results: List[Optional[ShapeRunResult]] = []
        wrote_csv = False
        for shape_id in shape_ids:
            config = competition_config(shape_id)
            result = run_one_configuration(
                args,
                device,
                dtype,
                variant_keys,
                config,
                shape_id=shape_id,
                csv_append=wrote_csv,
            )
            results.append(result)
            if args.csv and result is not None and result.timings:
                wrote_csv = True

        print_competition_sweep_summary(results, variant_keys)
        failures = [
            result
            for result in results
            if result is not None
            and result.accuracy
            and any(not s.passed for s in result.accuracy.values())
        ]
        return 2 if failures and not args.benchmark_on_failure else 0

    config = TransformerConfig(
        batch_size=args.batch_size,
        seq_len=args.seq_len,
        d_model=args.d_model,
        num_heads=args.heads,
        ffn_dim=args.ffn_dim,
        num_layers=args.layers,
        causal=args.causal,
    )

    result = run_one_configuration(
        args,
        device,
        dtype,
        variant_keys,
        config,
    )
    if result is None:
        return 3
    if result.accuracy and any(not s.passed for s in result.accuracy.values()):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
