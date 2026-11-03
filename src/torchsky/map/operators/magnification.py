"""Differentiable gravitational-lensing (magnification) operators.

Native-frame forward models need the lensing Jacobian in one place. This
module provides the magnification (kappa) sector as plain ``torch.nn``
components with no torchsky-internal coupling:

- ``ConvergenceMagnification``: maps a convergence map to the magnification
  map ``mu`` under an explicit law (``linear``: ``mu = 1 + 2 kappa``, or
  ``exponential``: ``mu = exp(2 kappa)``), optionally with a bounded learned
  residual whose gate starts at zero (identity at initialisation) and is
  L2-regularised toward zero by the caller;
- ``magnify_flux``: unresolved-source flux magnification ``F -> mu F``;
- ``scale_source_sizes``: resolved-source size dilation ``sigma -> sqrt(mu) sigma``
  (surface brightness conserved);
- ``magnification_bias``: number-density modulation ``n -> mu^(alpha - 1) n``
  with the cumulative count slope ``alpha = 2.5 dlog10 N/dm`` supplied by the
  caller's selection model.

Conventions: ``kappa`` is dimensionless, positive for overdense lines of
sight; ``mu`` is clamped to ``[1/max_magnification, max_magnification]`` when a
bound is given. The shear (distortion) sector is deliberately not implemented
here; it is gated by the representation shape-retention probe.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from ...device import _as_tensor

__all__ = [
    "ConvergenceMagnification",
    "MagnificationResidual",
    "magnification_bias",
    "magnify_flux",
    "scale_source_sizes",
]

_MAGNIFICATION_MODES = ("linear", "exponential")


def _as_kappa_map(kappa: Tensor) -> tuple[Tensor, torch.Size]:
    if not torch.is_tensor(kappa):
        raise TypeError("kappa must be a torch.Tensor")
    if not kappa.is_floating_point():
        raise TypeError("kappa must be a floating-point tensor")
    original = kappa.shape
    if kappa.ndim == 1:
        kappa = kappa.view(-1, 1, 1, 1)
    elif kappa.ndim == 2:
        kappa = kappa.unsqueeze(0).unsqueeze(0)
    elif kappa.ndim == 3:
        kappa = kappa.unsqueeze(1)
    if kappa.ndim != 4 or kappa.shape[1] != 1:
        raise ValueError(
            "kappa must have shape [N], [H,W], [B,H,W] or [B,1,H,W], got "
            f"{tuple(original)}"
        )
    if not bool(torch.isfinite(kappa).all()):
        raise ValueError("kappa must be finite")
    return kappa, original


def _restore_shape(value: Tensor, original: torch.Size) -> Tensor:
    return value.view(original)


def _magnification_from_kappa(
    kappa: Tensor,
    *,
    mode: str,
    max_magnification: float | None,
) -> Tensor:
    if mode not in _MAGNIFICATION_MODES:
        raise ValueError(f"mode must be one of {_MAGNIFICATION_MODES}, got {mode!r}")
    mu = 1.0 + 2.0 * kappa if mode == "linear" else torch.exp(2.0 * kappa)
    if max_magnification is not None:
        if not math.isfinite(max_magnification) or max_magnification <= 1.0:
            raise ValueError("max_magnification must be finite and greater than 1")
        mu = torch.clamp(mu, min=1.0 / max_magnification, max=max_magnification)
    return mu


def magnify_flux(
    flux: Tensor,
    kappa: Tensor,
    *,
    mode: str = "linear",
    max_magnification: float | None = None,
) -> Tensor:
    """Unresolved-source flux magnification ``F -> mu F``."""
    kappa_map, original = _as_kappa_map(kappa)
    mu = _magnification_from_kappa(
        kappa_map, mode=mode, max_magnification=max_magnification
    )
    return flux * _restore_shape(mu, original)


def scale_source_sizes(
    sigma: Tensor,
    kappa: Tensor,
    *,
    mode: str = "linear",
    max_magnification: float | None = None,
) -> Tensor:
    """Resolved-source size dilation ``sigma -> sqrt(mu) sigma`` at fixed surface brightness."""
    kappa_map, original = _as_kappa_map(kappa)
    mu = _magnification_from_kappa(
        kappa_map, mode=mode, max_magnification=max_magnification
    )
    return sigma * torch.sqrt(_restore_shape(mu, original))


def magnification_bias(
    density: Tensor,
    kappa: Tensor,
    count_slope: Tensor | float,
    *,
    mode: str = "linear",
    max_magnification: float | None = None,
) -> Tensor:
    """Magnification-bias number-density modulation ``n -> mu^(alpha - 1) n``."""
    kappa_map, original = _as_kappa_map(kappa)
    mu = _magnification_from_kappa(
        kappa_map, mode=mode, max_magnification=max_magnification
    )
    mu = _restore_shape(mu, original)
    exponent = _as_tensor(count_slope, dtype=mu.dtype, device=mu.device) - 1.0
    return density * mu.pow(exponent)


class MagnificationResidual(nn.Module):
    """Small bounded residual added to kappa, gated at zero at initialisation.

    The residual is a two-layer convolutional correction on the convergence
    map. The output gate starts at zero so the operator is exactly the
    physics law at initialisation; the caller regularises the gate toward zero
    (``gate_penalty``) so the residual only opens when the data demand it.
    """

    def __init__(self, *, channels: int = 8, kernel_size: int = 3) -> None:
        super().__init__()
        if channels < 1:
            raise ValueError("channels must be positive")
        if kernel_size < 1 or kernel_size % 2 == 0:
            raise ValueError("kernel_size must be a positive odd integer")
        padding = kernel_size // 2
        self.conv = nn.Sequential(
            nn.Conv2d(1, channels, kernel_size, padding=padding),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size, padding=padding),
        )
        self.gate = nn.Parameter(torch.zeros(1))
        self.max_abs_delta_kappa = 0.5

    def forward(self, kappa: Tensor) -> Tensor:
        correction = torch.tanh(self.conv(kappa)) * self.max_abs_delta_kappa
        return kappa + self.gate * correction

    def gate_penalty(self) -> Tensor:
        return torch.square(self.gate).sum()


class ConvergenceMagnification(nn.Module):
    """Convergence-to-magnification operator with an optional learned residual.

    ``forward`` returns ``mu`` with the same leading shape as ``kappa``.
    """

    def __init__(
        self,
        *,
        mode: str = "linear",
        max_magnification: float | None = 100.0,
        residual: bool = False,
        residual_channels: int = 8,
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if mode not in _MAGNIFICATION_MODES:
            raise ValueError(
                f"mode must be one of {_MAGNIFICATION_MODES}, got {mode!r}"
            )
        if max_magnification is not None and (
            not math.isfinite(max_magnification) or max_magnification <= 1.0
        ):
            raise ValueError("max_magnification must be finite and greater than 1")
        self.mode = mode
        self.max_magnification = max_magnification
        self.residual = (
            MagnificationResidual(
                channels=residual_channels, kernel_size=residual_kernel_size
            )
            if residual
            else None
        )

    def effective_kappa(self, kappa: Tensor) -> Tensor:
        kappa_map, _ = _as_kappa_map(kappa)
        if self.residual is not None:
            kappa_map = self.residual(kappa_map)
        return kappa_map

    def forward(self, kappa: Tensor) -> Tensor:
        kappa_map, original = _as_kappa_map(kappa)
        effective = self.effective_kappa(kappa_map)
        mu = _magnification_from_kappa(
            effective, mode=self.mode, max_magnification=self.max_magnification
        )
        return _restore_shape(mu, original)

    def regularization(self) -> Tensor:
        if self.residual is None:
            return torch.zeros((), dtype=torch.get_default_dtype())
        return self.residual.gate_penalty()
