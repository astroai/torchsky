"""Coordinate-aware encoders for implicit sky models."""

from __future__ import annotations

import math
from typing import Optional

import torch
from torch import Tensor, nn

from ..coords import spherical_fourier_features
from ..wcs.healpix import lonlat_to_xyz


def harmonic_sky_encoding(
    ra_deg: Tensor,
    dec_deg: Tensor,
    *,
    include_constant: bool = True,
    include_degree1: bool = True,
    include_degree2: bool = True,
) -> Tensor:
    """Low-order real spherical-harmonic-style basis on the unit sphere."""
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

    raw = wavelength if wavelength is not None else frequency
    assert raw is not None
    raw = torch.as_tensor(raw, dtype=torch.float64)
    coord = torch.log(torch.clamp(raw, min=torch.finfo(raw.dtype).tiny))
    scales = base ** torch.arange(
        num_frequencies, dtype=coord.dtype, device=coord.device
    )
    feats = [coord]
    for scale in scales:
        feats.extend([torch.sin(scale * coord), torch.cos(scale * coord)])
    return torch.stack(feats, dim=-1)


class SineLayer(nn.Module):
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
    def __init__(
        self,
        in_features: int,
        hidden_features: int = 128,
        hidden_layers: int = 3,
        out_features: int = 1,
    ) -> None:
        super().__init__()
        layers: list[nn.Module] = [SineLayer(in_features, hidden_features)]
        for _ in range(hidden_layers - 1):
            layers.append(SineLayer(hidden_features, hidden_features))
        layers.append(nn.Linear(hidden_features, out_features))
        self.net = nn.Sequential(*layers)

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class SkyCoordinateEncoder(nn.Module):
    def __init__(
        self,
        *,
        num_fourier_frequencies: int = 4,
        spectral_frequencies: int = 4,
        include_degree2: bool = True,
    ) -> None:
        super().__init__()
        self.num_fourier_frequencies = num_fourier_frequencies
        self.spectral_frequencies = spectral_frequencies
        self.include_degree2 = include_degree2

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
            include_xyz=True,
            include_lonlat=True,
        )
        feats = [sky, fourier]
        if wavelength is not None or frequency is not None:
            feats.append(
                log_spectral_encoding(
                    wavelength=wavelength,
                    frequency=frequency,
                    num_frequencies=self.spectral_frequencies,
                )
            )
        return torch.cat(feats, dim=-1)


__all__ = [
    "SkyCoordinateEncoder",
    "SineLayer",
    "SirenImplicitField",
    "harmonic_sky_encoding",
    "log_spectral_encoding",
]
