"""Optional Torchsky native accelerator loading."""

from __future__ import annotations

import os

if os.environ.get("TORCHSKY_DISABLE_CPP", "0") == "1":
    cpp = None
else:
    # Preload libtorch before the extension: PyPI torch wheels rely on the
    # already-loaded libtorch symbols rather than an RPATH for the extension.
    import torch  # noqa: F401

    try:
        from . import _cpp as cpp
    except ImportError:  # pragma: no cover - source tree without a native build
        cpp = None
