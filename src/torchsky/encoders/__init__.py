"""Coordinate-aware encoders for implicit sky models.

Note:
This module is experimental; its API may change between releases.
"""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from .._dtype import work_dtype
from ..sphere.healpix import lonlat_to_xyz, spherical_fourier_features
from .photometry import (
    BinnedRedshiftHead,
    MaskedSetFusion,
    PhotometryEncoder,
    marginalize_association_logits,
)
from ..device import _to


def _sky_encoder_fourier_min_frequency(*, patch_extent_deg: float) -> float:
    """Minimum ``min_frequency`` for :func:`~torchsky.sphere.healpix.spherical_fourier_features`.

    Maps approximately one oscillation across an angular patch whose typical extent is
    ``patch_extent_deg`` (degrees). Use ``FieldDomain.spatial_scale_deg`` from
    :func:`torchsky.map.field.estimate_field_domain` so multi-scale Fourier bands align
    with the fitted footprint rather than defaulting to whole-sphere periods.

    For a ~360° domain this returns ~1.0 (global modes). For arcminute fields it increases
    so mid-level ``base**k`` bands resolve structure inside the patch.
    """
    theta = math.radians(max(float(patch_extent_deg), 1.0e-9))
    return (2.0 * math.pi) / max(theta, 1.0e-12)


def harmonic_sky_encoding(
    ra_deg: Tensor,
    dec_deg: Tensor,
    *,
    include_constant: bool = True,
    include_degree1: bool = True,
    include_degree2: bool = True,
) -> Tensor:
    """Low-order **real polynomial basis** on the unit sphere (not normalized SH).

    Uses `(x,y,z) = lonlat_to_xyz(ra_deg, dec_deg)` on the unit sphere, then optional
    terms: constant; linear `x,y,z`; quadratic `xy, yz, zx, x²−y², 3z²−1`. The degree-2
    pieces match (unnormalized) real spherical harmonic combinations in Cartesian form,
    but this is a **feature basis for implicit models**, not a complete orthonormal Y_lm stack.
    """
    x, y, z = lonlat_to_xyz(ra_deg, dec_deg)
    feats: list[Tensor] = []

    if include_constant:
        feats.append(torch.ones_like(x))
    if include_degree1:
        feats.extend([x, y, z])
    if include_degree2:
        feats.extend(
            [
                x * y,
                y * z,
                z * x,
                x.square() - y.square(),
                3.0 * z.square() - 1.0,
            ]
        )
    if not feats:
        raise ValueError("At least one harmonic component must be enabled")
    return torch.stack(feats, dim=-1)


def log_spectral_encoding(
    *,
    wavelength: Optional[Tensor] = None,
    frequency: Optional[Tensor] = None,
    num_frequencies: int = 4,
    base: float = 2.0,
) -> Tensor:
    """Multi-scale Fourier features for log wavelength or log frequency."""
    if (wavelength is None) == (frequency is None):
        raise ValueError("Provide exactly one of wavelength or frequency")
    if num_frequencies <= 0:
        raise ValueError("num_frequencies must be positive")

    raw = torch.as_tensor(wavelength if wavelength is not None else frequency)
    if not raw.is_floating_point():
        raw = raw.to(work_dtype(raw.device))
    coord = torch.log(torch.clamp(raw, min=torch.finfo(raw.dtype).tiny))
    scales = base ** torch.arange(
        num_frequencies, dtype=coord.dtype, device=coord.device
    )
    # Vectorize the scale multiplication and sin/cos application using broadcasting
    scaled = coord.unsqueeze(-1) * scales
    fourier_feats = torch.stack([torch.sin(scaled), torch.cos(scaled)], dim=-1).flatten(
        start_dim=-2
    )
    return torch.cat([coord.unsqueeze(-1), fourier_feats], dim=-1)


class _SineLayer(nn.Module):
    """SIREN sine layer with the reference ``U(-bound, bound)`` init."""

    def __init__(self, in_features: int, out_features: int, *, omega_0: float = 30.0):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.omega_0 = omega_0
        self.reset_parameters()

    def reset_parameters(self) -> None:
        bound = math.sqrt(6.0 / self.linear.in_features) / self.omega_0
        nn.init.uniform_(self.linear.weight, -bound, bound)
        nn.init.zeros_(self.linear.bias)

    def forward(self, x: Tensor) -> Tensor:
        return torch.sin(self.omega_0 * self.linear(x))


class SirenImplicitField(nn.Module):
    """SIREN multilayer perceptron over precomputed features."""

    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be at least one")
        if min(in_features, hidden_features, out_features) < 1:
            raise ValueError("layer widths must be positive")
        layers: list[nn.Module] = [_SineLayer(in_features, hidden_features)]
        for _ in range(hidden_layers - 1):
            layers.append(_SineLayer(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SkyCoordinateEncoder(nn.Module):
    """Spherical polynomial + Fourier features, optionally with log-spectral ones."""

    def __init__(
        self,
        *,
        num_fourier_frequencies: int = 4,
        spectral_frequencies: int = 4,
        include_degree2: bool = True,
        min_frequency: float = 1.0,
    ) -> None:
        super().__init__()
        if num_fourier_frequencies <= 0:
            raise ValueError("num_fourier_frequencies must be positive")
        if spectral_frequencies < 0:
            raise ValueError("spectral_frequencies must be non-negative")
        self.num_fourier_frequencies = num_fourier_frequencies
        self.spectral_frequencies = spectral_frequencies
        self.include_degree2 = include_degree2
        if float(min_frequency) <= 0.0:
            raise ValueError("min_frequency must be positive")
        self.min_frequency = float(min_frequency)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        *,
        wavelength: Optional[Tensor] = None,
        frequency: Optional[Tensor] = None,
    ) -> Tensor:
        sky = harmonic_sky_encoding(
            ra_deg,
            dec_deg,
            include_degree2=self.include_degree2,
        )
        fourier = spherical_fourier_features(
            ra_deg,
            dec_deg,
            num_frequencies=self.num_fourier_frequencies,
            min_frequency=self.min_frequency,
            include_xyz=True,
            include_lonlat=True,
        )
        feats = [sky, fourier]
        if wavelength is not None or frequency is not None:
            feats.append(
                _to(
                    log_spectral_encoding(
                        wavelength=wavelength,
                        frequency=frequency,
                        num_frequencies=self.spectral_frequencies,
                    ),
                    device=sky.device,
                    dtype=sky.dtype,
                )
            )
        return torch.cat(feats, dim=-1)


__all__ = [
    "BinnedRedshiftHead",
    "MaskedSetFusion",
    "PhotometryEncoder",
    "SkyCoordinateEncoder",
    "SirenImplicitField",
    "harmonic_sky_encoding",
    "log_spectral_encoding",
    "marginalize_association_logits",
]
