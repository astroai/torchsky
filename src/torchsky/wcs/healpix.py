"""Compatibility shim for HEALPix primitives now owned by ``torchsky.sphere``."""

from __future__ import annotations

from ..sphere.healpix import *  # noqa: F403
from ..sphere.healpix import __all__ as __all__
