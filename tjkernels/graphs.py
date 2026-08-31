"""Whole-forward CUDA Graph capture.

The benchmark calls the model many times with a fixed shape, which is exactly
the situation CUDA Graphs exist for: the entire 4-layer forward collapses to a
single work submission instead of ~120 individual kernel launches.  On Windows
(WDDM) each launch costs several microseconds, so for the small shapes this is
the dominant cost in the baseline.

Correctness is preserved for *any* input, not just the captured one: every call
copies the live input into the captured static buffer, replays, and clones the
static output back out.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

_WARNED = False


def _warn_once(msg: str) -> None:
    global _WARNED
    if not _WARNED:
        _WARNED = True
        print(f"[tjkernels] {msg}")


class CapturedGraph:
    """One captured forward, bound to one input shape/dtype/mask-mode."""

    def __init__(
        self,
        fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
        warmup: int = 3,
    ) -> None:
        self.static_x = x.clone()
        self.static_mask = None if mask is None else mask.clone()

        # Warm up on a side stream: allocates cuBLAS/Triton workspaces and runs
        # any Triton autotuning *before* capture, where syncs are illegal.
        side = torch.cuda.Stream()
        side.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(side):
            for _ in range(warmup):
                fn(self.static_x, self.static_mask)
        torch.cuda.current_stream().wait_stream(side)
        torch.cuda.synchronize()

        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.static_out = fn(self.static_x, self.static_mask)

    def run(
        self, x: torch.Tensor, mask: Optional[torch.Tensor]
    ) -> torch.Tensor:
        self.static_x.copy_(x)
        if self.static_mask is not None and mask is not None:
            self.static_mask.copy_(mask)
        self.graph.replay()
        return self.static_out.clone()


class GraphCache:
    """Keyed by everything that changes the captured work."""

    def __init__(self) -> None:
        self._graphs: Dict[Tuple, Optional[CapturedGraph]] = {}

    def get(
        self,
        key: Tuple,
        fn: Callable[[torch.Tensor, Optional[torch.Tensor]], torch.Tensor],
        x: torch.Tensor,
        mask: Optional[torch.Tensor],
    ) -> Optional[CapturedGraph]:
        if key in self._graphs:
            return self._graphs[key]
        try:
            graph = CapturedGraph(fn, x, mask)
        except Exception as exc:  # capture is an optimization, never a hard dep
            _warn_once(f"CUDA Graph capture failed ({exc}); running eager.")
            graph = None
        self._graphs[key] = graph
        return graph

    def clear(self) -> None:
        self._graphs.clear()
