"""The official TechJam test matrix (problem statement 3.7, Appendix).

Case 14 (batch 32, seq 100000, d_model 1024) is defined here for completeness
but excluded from the default sweep: its activation tensor alone is
32 x 100000 x 1024 x 4 B = 13.1 GB in fp32, which the harness allocates before
any model runs, and the reference implementation materializes a score tensor of
32 x 16 x 100000^2 = 5.1e12 elements. Neither fits on a 12 GB RTX 3060 -- see
report/TECH_REPORT.md for the full arithmetic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List


@dataclass(frozen=True)
class Case:
    idx: int
    batch: int
    d_model: int
    heads: int
    seq: int
    layers: int
    causal: bool
    ffn: int
    note: str = ""

    @property
    def name(self) -> str:
        return f"case{self.idx:02d}"

    @property
    def tokens(self) -> int:
        return self.batch * self.seq

    def cli(self) -> List[str]:
        args = [
            "--batch-size", str(self.batch),
            "--d-model", str(self.d_model),
            "--heads", str(self.heads),
            "--seq-len", str(self.seq),
            "--layers", str(self.layers),
            "--ffn-dim", str(self.ffn),
        ]
        if self.causal:
            args.append("--causal")
        return args


ALL_CASES: List[Case] = [
    Case(1,     64,  128,  4,    128, 4, True,  128, "reference shape"),
    Case(2,      1,  128,  4,    128, 4, True,  128, "min batch - launch bound"),
    Case(3,      4,  128,  4,    128, 4, True,  128, "small batch"),
    Case(4,     16,  128,  4,    128, 4, True,  128, "small batch"),
    Case(5,    128,  128,  4,    128, 4, True,  128, "large batch"),
    Case(6,  10000,  128,  4,    128, 4, True,  128, "huge batch - bandwidth bound"),
    Case(7,     64,   32,  4,    128, 4, True,   32, "tiny dim"),
    Case(8,     64, 1024,  4,    128, 4, True, 1024, "large dim - GEMM bound"),
    Case(9,     64,  128,  1,    128, 4, True,  128, "single head"),
    Case(10,    64,  128,  2,    128, 4, True,  128, "two heads"),
    Case(11,    64,  128, 16,    128, 4, True,  128, "16 heads - head_dim 8"),
    Case(12,    64,  128,  4,     32, 4, True,  128, "short sequence"),
    Case(13,    64,  128,  4,   1024, 4, True,  128, "long sequence - attention bound"),
]

OOM_CASES: List[Case] = [
    Case(14,    32, 1024, 16, 100000, 2, True, 1024, "excluded: needs >24 GB VRAM"),
]


def by_index(idx: int) -> Case:
    for case in ALL_CASES + OOM_CASES:
        if case.idx == idx:
            return case
    raise KeyError(f"no case {idx}")
