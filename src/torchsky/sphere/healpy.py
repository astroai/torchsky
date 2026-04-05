"""Drop-in *name* for the healpy-style HEALPix API implemented in torch.

Use::

    from torchsky.sphere import healpy as hp

as the torch-native analogue of ``import healpy as hp``. This is the HEALPix slice
of torchsky’s wider compatibility story (Astropy, fitsio, spherical_geometry, …);
see ``docs/compatibility.md``. All symbols come from :mod:`torchsky.sphere.compat`.
"""

from __future__ import annotations

from .compat import *  # noqa: F403

from .compat import __all__ as __all__
