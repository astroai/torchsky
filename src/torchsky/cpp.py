"""Merged torchsky.cpp — re-exports torchsky._cpp (native) + torchfits.cpp (FITS).

This shim lets all existing ``import torchsky.cpp as cpp`` and
``from torchsky.cpp import ...`` calls work unchanged after FITS I/O moved
to ``torchfits.cpp``.
"""

import os

import torch  # noqa: F401 - preload libtorch: PyPI torch libs are not on the loader path

# ── FITS I/O module (import first so torchsky._cpp can overwrite colliding
#    HEALPix / WCS names with its own implementations) ────────────────────
from torchfits.cpp import *  # noqa: F403

# ── native (non-FITS) module: HEALPix, WCS, geometry, hardware ──────────
if os.environ.get("TORCHSKY_DISABLE_CPP", "0") != "1":
    from torchsky._cpp import *  # noqa: F403
