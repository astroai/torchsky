"""Sky geometry facade. Kernels live in ``torchsky.wcs``.

Projects import ``torchsky.coords`` for catalog frames, observed place, and
propagation. FITS WCS stays ``torchsky.wcs``. This is not a SkyCoord graph
and not ``torchsky.transforms`` (image ops).
"""

from ..wcs.catalog_state import CatalogState
from ..wcs.coords import transform
from ..wcs.frames import (
    convert_celestial,
    propagate_proper_motion,
    propagate_space_motion,
)
from ..wcs.observed import EarthOrientation, Site, observed_altaz

__all__ = [
    "CatalogState",
    "EarthOrientation",
    "Site",
    "convert_celestial",
    "observed_altaz",
    "propagate_proper_motion",
    "propagate_space_motion",
    "transform",
]
