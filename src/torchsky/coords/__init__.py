"""Sky geometry facade. Kernels live in ``torchsky.wcs``.

Projects import ``torchsky.coords`` for catalog frames, observed place, and
propagation. FITS WCS stays ``torchsky.wcs``. This is not a SkyCoord graph
and not ``torchsky.transforms`` (image ops).
"""

from ..wcs.catalog_state import CatalogState
from ..wcs.coords import transform
from ..wcs.frames import (
    convert_celestial,
    heliocentric_rv_to_lsr,
    icrs_to_ecliptic_of_date,
    icrs_to_galactocentric,
    propagate_proper_motion,
    propagate_space_motion,
)
from ..wcs.iau2006 import (
    apply_dcr,
    apply_ephemeris_offset,
    apply_refraction,
    apply_stellar_aberration,
    dcr_offset,
)
from ..wcs.observed import EarthOrientation, Site, observed_altaz

__all__ = [
    "CatalogState",
    "EarthOrientation",
    "Site",
    "apply_dcr",
    "apply_ephemeris_offset",
    "apply_refraction",
    "apply_stellar_aberration",
    "convert_celestial",
    "dcr_offset",
    "heliocentric_rv_to_lsr",
    "icrs_to_ecliptic_of_date",
    "icrs_to_galactocentric",
    "observed_altaz",
    "propagate_proper_motion",
    "propagate_space_motion",
    "transform",
]
