"""Sky-coordinate encoder construction helpers for field models."""

from __future__ import annotations

import torch

from ...encoders import SkyCoordinateEncoder, _sky_encoder_fourier_min_frequency

from .types import FieldDomain


def make_sky_coordinate_encoder(
    domain: FieldDomain,
    *,
    num_fourier_frequencies: int,
    spectral_frequencies: int,
    include_degree2: bool = True,
    scale_fourier_to_domain: bool = True,
) -> SkyCoordinateEncoder:
    min_freq = 1.0
    if scale_fourier_to_domain:
        min_freq = _sky_encoder_fourier_min_frequency(
            patch_extent_deg=float(domain.spatial_scale_deg),
        )
    return SkyCoordinateEncoder(
        num_fourier_frequencies=num_fourier_frequencies,
        spectral_frequencies=spectral_frequencies,
        include_degree2=include_degree2,
        min_frequency=min_freq,
    )


def sky_encoder_output_dim(encoder: SkyCoordinateEncoder) -> int:
    with torch.no_grad():
        tensor = encoder(
            torch.zeros(1, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
            wavelength=torch.ones(1, dtype=torch.float64),
        )
    return int(tensor.shape[-1])


__all__ = [
    "make_sky_coordinate_encoder",
    "sky_encoder_output_dim",
]
