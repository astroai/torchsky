"""Optional Torchsky native accelerator loading."""

from __future__ import annotations

import os

if os.environ.get("TORCHSKY_DISABLE_CPP", "0") == "1":
    cpp = None
else:
    # The extension itself is torch-free; torch is imported first because
    # loading it registers the torch.ops.torchsky operators.
    import torch  # noqa: F401

    try:
        from . import _cpp as cpp
    except ImportError:  # pragma: no cover - source tree without a native build
        cpp = None
