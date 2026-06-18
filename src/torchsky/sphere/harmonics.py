"""Spherical harmonic primitives for HEALPix maps.

This module is the preferred public name for harmonic transforms. The older
``torchsky.sphere.spectral`` module remains available for compatibility.
"""

from .spectral import *  # noqa: F403
from .spectral import __all__ as __all__
