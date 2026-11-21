"""Deprecated module; use :mod:`torchsky.map.field.source_quadrature`.

Removed in torchsky 0.7.
"""

from ..._deprecation import deprecated_module_getattr

_NEW = "torchsky.map.field.source_quadrature"

__getattr__ = deprecated_module_getattr(
    __name__,
    {
        "NativeFieldResponse": (_NEW, "SourceQuadratureResponse"),
        "bind_native_field_likelihood": (_NEW, "bind_source_quadrature_likelihood"),
        "render_native_field_observation": (
            _NEW,
            "render_source_quadrature_observation",
        ),
    },
)
