"""
tjkernels - shape-dispatched, fused GPU kernels for the TechJam 2026
Transformer-layer optimization task.

Public entry point:

    import tjkernels
    class UserOptimizedTransformer(BaselineTransformer):
        def forward(self, x, valid_token_mask=None):
            return tjkernels.forward(self, x, valid_token_mask)

The engine reads parameters directly off the baseline module, so the
benchmark's strict `load_state_dict` weight copy keeps working unchanged.
"""

from .engine import forward, get_engine, reset_engines
from .plans import Plan, select_plan

__all__ = ["forward", "get_engine", "reset_engines", "Plan", "select_plan"]
