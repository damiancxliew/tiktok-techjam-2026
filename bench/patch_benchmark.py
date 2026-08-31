"""Rewrite the `UserOptimizedTransformer` hook in the official benchmark.

The official script is left untouched everywhere else -- same CLI, same
accuracy rules, same timing loop -- so results stay directly comparable with
an unmodified copy (kept at report/torch_transformer_benchmark_original.py).

Usage:  python bench/patch_benchmark.py [--check]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TARGET = ROOT / "torch_transformer_benchmark.py"

IMPORT_ANCHOR = "import torch\nimport torch.nn as nn\nimport torch.nn.functional as F\n"

IMPORT_BLOCK = """import torch
import torch.nn as nn
import torch.nn.functional as F

# --- TechJam 2026 optimized kernels -----------------------------------------
sys.path.insert(0, str(Path(__file__).resolve().parent))
import tjkernels
# ----------------------------------------------------------------------------
"""

BODY_START = "        # ====================== your codes here ======================"
BODY_END = "        # ============================================================"

NEW_BODY = '''        # ====================== your codes here ======================
        # Dispatches on (batch, seq, d_model, heads, ffn, layers, causal, dtype)
        # to a tuned plan: packed-QKV projection, causal flash attention,
        # fused add+LayerNorm, single-kernel FFN, fp16 tensor-core compute with
        # an fp32 residual stream, and whole-forward CUDA Graph replay.
        # See tjkernels/plans.py for the dispatch table.
        return tjkernels.forward(self, x, valid_token_mask)
        # ============================================================'''


def patch(text: str) -> str:
    if "import tjkernels" not in text:
        if IMPORT_ANCHOR not in text:
            raise SystemExit("could not find the torch import block to anchor to")
        text = text.replace(IMPORT_ANCHOR, IMPORT_BLOCK, 1)
        text = text.replace(
            "import argparse\nimport copy\n",
            "import argparse\nimport copy\nimport sys\nfrom pathlib import Path\n",
            1,
        )

    start = text.index(BODY_START)
    end = text.index(BODY_END, start) + len(BODY_END)
    return text[:start] + NEW_BODY + text[end:]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()

    text = TARGET.read_text(encoding="utf-8")
    patched = patch(text)
    if args.check:
        print("already patched" if text == patched else "needs patching")
        return 0 if text == patched else 1
    TARGET.write_text(patched, encoding="utf-8")
    print(f"patched {TARGET}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
