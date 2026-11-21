"""Shared scalar helpers for WCS projection rotation code."""

from __future__ import annotations

from typing import Sequence, Union

import torch
from torch import Tensor


_HAS_TORCH_SINCOS = hasattr(torch, "sincos")


def _sincos(x: Tensor) -> tuple[Tensor, Tensor]:
    if _HAS_TORCH_SINCOS:
        return torch.sincos(x)  # type: ignore[attr-defined]
    return torch.sin(x), torch.cos(x)


def _wrap_lon360(angle: Tensor) -> Tensor:
    """Wrap longitudes to [0, 360)."""
    return torch.remainder(angle, 360.0)


def _wrap_lon180(angle: Tensor) -> Tensor:
    """Wrap longitudes to [-180, 180)."""
    return torch.remainder(angle + 180.0, 360.0) - 180.0


def _traced_on_mps(x: Tensor) -> bool:
    return torch.compiler.is_compiling() and x.device.type == "mps"


def _hypot(a: Tensor, b: Tensor) -> Tensor:
    """``torch.hypot``; ``sqrt(a*a + b*b)`` when compiling for MPS.

    Inductor's Metal code generator has no ``hypot`` (PyTorch 2.14), so a
    traced MPS graph would fail to compile. WCS operands are angles and
    plane coordinates far from float32 overflow, so the two agree to an ulp.
    """
    if _traced_on_mps(a):
        return torch.sqrt(a * a + b * b)
    return torch.hypot(a, b)


def _copysign(a: Tensor, b: Tensor) -> Tensor:
    """``torch.copysign``; a ``where`` on ``b < 0`` when compiling for MPS
    (no Metal codegen for ``copysign``; differs only for ``b = -0.0``)."""
    if _traced_on_mps(a):
        return torch.where(b < 0.0, -a.abs(), a.abs())
    return torch.copysign(a, b)


PVParams = Union[Tensor, Sequence[float], None]


def _pv_value(params: PVParams, idx: int, default: float) -> float:
    """PV parameter ``idx`` as a Python float, or ``default`` when absent.

    :class:`~torchsky.wcs.WCS` passes its PV vectors as tuples of floats so
    projection constants are host values: no device sync per call and no
    ``.item()`` graph break under ``torch.compile``. Tensors are accepted for
    direct callers of the projection functions.
    """
    if params is None:
        return default
    if isinstance(params, Tensor):
        if idx >= int(params.numel()):
            return default
        return float(params[idx].item())
    if idx >= len(params):
        return default
    return float(params[idx])


# WCSLIB ``prjbchk`` tolerance (degrees) for native coordinates from ``*x2s``.
_NATIVE_BOUNDS_TOL = 1.0e-13


def _native_bounds(phi: Tensor, theta: Tensor) -> tuple[Tensor, Tensor]:
    """WCSLIB ``prjbchk``: reject native ``|phi| > 180`` or ``|theta| > 90``.

    Values within :data:`_NATIVE_BOUNDS_TOL` of a bound are clamped onto it,
    values beyond it become NaN (both coordinates), exactly as WCSLIB does for
    every plane-to-sphere projection with strict bounds checking (the Astropy
    default). Without it, plane points outside a projection's domain (e.g.
    ``|y| > 90`` in CAR) are rotated onto the sphere as if they were valid.
    """
    tol = _NATIVE_BOUNDS_TOL
    invalid = (phi.abs() > 180.0 + tol) | (theta.abs() > 90.0 + tol)
    phi = torch.clamp(phi, -180.0, 180.0)
    theta = torch.clamp(theta, -90.0, 90.0)
    nan = torch.full_like(phi, float("nan"))
    return torch.where(invalid, nan, phi), torch.where(invalid, nan, theta)
