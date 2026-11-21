"""Differentiable weak-lensing magnification operators (experimental, opt-in).

The magnification (convergence) sector as plain ``torch.nn`` components with no
torchsky-internal coupling. Import it explicitly from
``torchsky.map.operators.lensing``; it is not re-exported by
``torchsky.map.operators``.

- ``ConvergenceMagnification`` maps a convergence map ``kappa`` to the
  magnification map ``mu`` under an explicit law (``linear``:
  ``mu = 1 + 2 kappa``; ``exponential``: ``mu = exp(2 kappa)``), optionally
  through a ``LearnedConvergenceCorrection`` that is the identity at
  initialisation;
- ``magnify_flux``: unresolved-source flux magnification ``F -> mu F``;
- ``scale_source_sizes``: resolved-source size dilation
  ``sigma -> sqrt(mu) sigma`` at fixed surface brightness;
- ``magnification_bias``: number-density modulation ``n -> mu^(alpha - 1) n``
  with the cumulative count slope ``alpha = 2.5 dlog10 N/dm`` supplied by the
  caller's selection model.

Conventions: ``kappa`` is dimensionless and positive for overdense lines of
sight. Every entry point takes ``max_magnification`` with the same default,
``None``: the law is applied exactly and the linear law fails closed where it is
undefined (``kappa <= -1/2``, i.e. ``mu <= 0``). A finite
``max_magnification > 1`` instead clamps ``mu`` to
``[1/max_magnification, max_magnification]``. The shear (distortion) sector is
not implemented.
"""

from __future__ import annotations

import math

import torch
from torch import Tensor, nn
from ...device import _as_tensor

__all__ = [
    "ConvergenceMagnification",
    "LearnedConvergenceCorrection",
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


def _validate_bound(max_magnification: float | None) -> None:
    if max_magnification is not None and (
        not math.isfinite(max_magnification) or max_magnification <= 1.0
    ):
        raise ValueError("max_magnification must be finite and greater than 1")


def _magnification_from_kappa(
    kappa: Tensor,
    *,
    mode: str,
    max_magnification: float | None,
) -> Tensor:
    if mode not in _MAGNIFICATION_MODES:
        raise ValueError(f"mode must be one of {_MAGNIFICATION_MODES}, got {mode!r}")
    _validate_bound(max_magnification)
    mu = 1.0 + 2.0 * kappa if mode == "linear" else torch.exp(2.0 * kappa)
    if max_magnification is None:
        # The linear weak-lensing law is undefined for kappa <= -1/2; without a
        # bound it would give negative flux and NaN sizes or densities.
        if mode == "linear" and bool((mu <= 0.0).any()):
            raise ValueError(
                "linear magnification 1 + 2 kappa must be positive; "
                "set max_magnification or use mode='exponential'"
            )
    else:
        mu = torch.clamp(mu, min=1.0 / max_magnification, max=max_magnification)
    return mu


def magnify_flux(
    flux: Tensor,
    kappa: Tensor,
    *,
    mode: str = "linear",
    max_magnification: float | None = None,
) -> Tensor:
    """Magnify unresolved-source flux, ``F -> mu F``.

    Args:
        flux: Fluxes broadcastable against ``kappa``'s shape.
        kappa: Convergence, ``[N]``, ``[H,W]``, ``[B,H,W]`` or ``[B,1,H,W]``.
        mode: ``"linear"`` (``mu = 1 + 2 kappa``) or ``"exponential"``
            (``mu = exp(2 kappa)``).
        max_magnification: ``None`` applies the law exactly; a finite bound
            greater than one clamps ``mu`` to its reciprocal and itself.

    Returns:
        Magnified flux.

    Raises:
        ValueError: On a bad shape, non-finite ``kappa``, unknown ``mode`` or
            bound, or ``1 + 2 kappa <= 0`` under the unbounded linear law.
    """
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
    """Dilate resolved-source sizes, ``sigma -> sqrt(mu) sigma``.

    Surface brightness is conserved, so linear sizes scale with ``sqrt(mu)``.

    Args:
        sigma: Source sizes broadcastable against ``kappa``'s shape.
        kappa: Convergence, ``[N]``, ``[H,W]``, ``[B,H,W]`` or ``[B,1,H,W]``.
        mode: ``"linear"`` or ``"exponential"`` magnification law.
        max_magnification: ``None`` applies the law exactly; a finite bound
            greater than one clamps ``mu``.

    Returns:
        Dilated sizes.

    Raises:
        ValueError: As :func:`magnify_flux`.
    """
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
    """Modulate a number density by magnification bias, ``n -> mu^(alpha - 1) n``.

    Args:
        density: Number densities broadcastable against ``kappa``'s shape.
        kappa: Convergence, ``[N]``, ``[H,W]``, ``[B,H,W]`` or ``[B,1,H,W]``.
        count_slope: Cumulative count slope ``alpha = 2.5 dlog10 N/dm`` of the
            caller's selection.
        mode: ``"linear"`` or ``"exponential"`` magnification law.
        max_magnification: ``None`` applies the law exactly; a finite bound
            greater than one clamps ``mu``.

    Returns:
        Modulated densities.

    Raises:
        ValueError: As :func:`magnify_flux`.
    """
    kappa_map, original = _as_kappa_map(kappa)
    mu = _magnification_from_kappa(
        kappa_map, mode=mode, max_magnification=max_magnification
    )
    mu = _restore_shape(mu, original)
    exponent = _as_tensor(count_slope, dtype=mu.dtype, device=mu.device) - 1.0
    return density * mu.pow(exponent)


class LearnedConvergenceCorrection(nn.Module):
    """Learned bounded correction added to ``kappa``, gated at zero initially.

    A two-layer convolutional correction on the convergence map, bounded by
    ``tanh`` to ``|delta kappa| <= 0.5`` before the gate. The gate starts at
    zero, so the correction is exactly the identity at initialisation; the
    caller regularises the gate toward zero with :meth:`gate_penalty` so the
    correction only opens when the data demand it.

    Args:
        channels: Hidden convolution channels (positive).
        kernel_size: Odd convolution kernel size.

    Raises:
        ValueError: If ``channels`` is not positive or ``kernel_size`` is not a
            positive odd integer.
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
        """Return the corrected ``[B,1,H,W]`` convergence."""
        correction = torch.tanh(self.conv(kappa)) * self.max_abs_delta_kappa
        return kappa + self.gate * correction

    def gate_penalty(self) -> Tensor:
        """Return the squared gate, the L2 penalty toward the physics law."""
        return torch.square(self.gate).sum()


class ConvergenceMagnification(nn.Module):
    """Map convergence to magnification, optionally with a learned correction.

    ``forward`` returns ``mu`` with the same shape as ``kappa``. A learned
    correction can push the effective ``kappa`` below ``-1/2``; with the
    default unbounded linear law that fails closed, so training loops that
    enable ``residual`` usually pass a finite ``max_magnification`` or use
    ``mode="exponential"``.

    Args:
        mode: ``"linear"`` (``mu = 1 + 2 kappa``) or ``"exponential"``
            (``mu = exp(2 kappa)``).
        max_magnification: ``None`` (default, as in the functions) applies the
            law exactly; a finite bound greater than one clamps ``mu``.
        residual: Add a :class:`LearnedConvergenceCorrection` before the law.
        residual_channels: Hidden channels of the correction.
        residual_kernel_size: Odd kernel size of the correction.

    Raises:
        ValueError: On an unknown ``mode`` or an invalid bound.
    """

    def __init__(
        self,
        *,
        mode: str = "linear",
        max_magnification: float | None = None,
        residual: bool = False,
        residual_channels: int = 8,
        residual_kernel_size: int = 3,
    ) -> None:
        super().__init__()
        if mode not in _MAGNIFICATION_MODES:
            raise ValueError(
                f"mode must be one of {_MAGNIFICATION_MODES}, got {mode!r}"
            )
        _validate_bound(max_magnification)
        self.mode = mode
        self.max_magnification = max_magnification
        self.residual = (
            LearnedConvergenceCorrection(
                channels=residual_channels, kernel_size=residual_kernel_size
            )
            if residual
            else None
        )

    def effective_kappa(self, kappa: Tensor) -> Tensor:
        """Return ``kappa`` as ``[B,1,H,W]`` after the optional correction."""
        kappa_map, _ = _as_kappa_map(kappa)
        if self.residual is not None:
            kappa_map = self.residual(kappa_map)
        return kappa_map

    def forward(self, kappa: Tensor) -> Tensor:
        """Return the magnification ``mu`` with the shape of ``kappa``."""
        kappa_map, original = _as_kappa_map(kappa)
        effective = self.effective_kappa(kappa_map)
        mu = _magnification_from_kappa(
            effective, mode=self.mode, max_magnification=self.max_magnification
        )
        return _restore_shape(mu, original)

    def regularization(self) -> Tensor:
        """Return the correction's gate penalty (zero without a correction)."""
        if self.residual is None:
            return torch.zeros((), dtype=torch.get_default_dtype())
        return self.residual.gate_penalty()
