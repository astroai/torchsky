"""Deprecated module; use :mod:`torchsky.map.field.source_quadrature_read`.

Removed in torchsky 0.7.
"""

from ..._deprecation import deprecated_module_getattr

_NEW = "torchsky.map.field.source_quadrature_read"

__getattr__ = deprecated_module_getattr(
    __name__,
    {
        "NativeIntegratedReadLikelihood": (_NEW, "SourceQuadratureReadLikelihood"),
        "bind_native_integrated_read_likelihood": (
            _NEW,
            "bind_source_quadrature_read_likelihood",
        ),
    },
)
