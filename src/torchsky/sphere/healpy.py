"""Drop-in *name* for the healpy-style API implemented in torch.

Use::

    from torchsky.sphere import healpy as hp

as the torch-native analogue of ``import healpy as hp``. All callables and
constants live in :mod:`torchsky.sphere.compat`; this module re-exports them
without adding behavior.
"""

from __future__ import annotations

from .compat import *  # noqa: F403

from .compat import __all__ as __all__
