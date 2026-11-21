"""Deprecated module; use :mod:`torchsky.map.operators.lensing`.

Removed in torchsky 0.7.
"""

from ..._deprecation import deprecated_module_getattr

_NEW = "torchsky.map.operators.lensing"

__getattr__ = deprecated_module_getattr(
    __name__,
    {
        "ConvergenceMagnification": (_NEW, "ConvergenceMagnification"),
        "MagnificationResidual": (_NEW, "LearnedConvergenceCorrection"),
        "magnification_bias": (_NEW, "magnification_bias"),
        "magnify_flux": (_NEW, "magnify_flux"),
        "scale_source_sizes": (_NEW, "scale_source_sizes"),
    },
)
