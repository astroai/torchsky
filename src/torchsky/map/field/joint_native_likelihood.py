"""Deprecated module; use :mod:`torchsky.map.field.shared_nuisance_likelihood`.

Removed in torchsky 0.7.
"""

from ..._deprecation import deprecated_module_getattr

__getattr__ = deprecated_module_getattr(
    __name__,
    {
        "bind_joint_native_likelihood": (
            "torchsky.map.field.shared_nuisance_likelihood",
            "bind_shared_nuisance_likelihood",
        ),
    },
)
