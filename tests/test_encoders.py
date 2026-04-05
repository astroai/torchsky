import torch
import pytest

from torchsky.encoders import (
    SkyCoordinateEncoder,
    harmonic_sky_encoding,
    log_spectral_encoding,
    sky_encoder_fourier_min_frequency,
)


def test_harmonic_sky_encoding_shape_and_grad() -> None:
    ra = torch.linspace(0.0, 360.0, steps=32, dtype=torch.float64, requires_grad=True)
    dec = torch.linspace(-60.0, 60.0, steps=32, dtype=torch.float64, requires_grad=True)
    feats = harmonic_sky_encoding(ra, dec)
    assert feats.shape == (32, 9)
    feats.square().mean().backward()
    assert ra.grad is not None
    assert dec.grad is not None


def test_log_spectral_encoding_shape() -> None:
    wavelength = torch.logspace(-1, 1, steps=16, dtype=torch.float64)
    feats = log_spectral_encoding(wavelength=wavelength, num_frequencies=3)
    assert feats.shape == (16, 7)


def test_sky_coordinate_encoder_combines_spatial_and_spectral_features() -> None:
    enc = SkyCoordinateEncoder(num_fourier_frequencies=2, spectral_frequencies=2)
    ra = torch.linspace(0.0, 90.0, steps=8, dtype=torch.float64)
    dec = torch.linspace(-10.0, 10.0, steps=8, dtype=torch.float64)
    wavelength = torch.linspace(0.4, 1.0, steps=8, dtype=torch.float64)
    feats = enc(ra, dec, wavelength=wavelength)
    assert feats.shape[0] == 8
    assert feats.shape[1] > 10


def test_sky_encoder_fourier_min_frequency_scales_with_patch() -> None:
    small = sky_encoder_fourier_min_frequency(patch_extent_deg=1.0)
    large = sky_encoder_fourier_min_frequency(patch_extent_deg=10.0)
    assert small > large > 0.0


def test_sky_coordinate_encoder_min_frequency_changes_fourier_features() -> None:
    ra = torch.tensor([10.0, 20.0], dtype=torch.float64)
    dec = torch.tensor([0.0, 1.0], dtype=torch.float64)
    a = SkyCoordinateEncoder(
        num_fourier_frequencies=2, spectral_frequencies=0, min_frequency=1.0
    )(ra, dec)
    b = SkyCoordinateEncoder(
        num_fourier_frequencies=2, spectral_frequencies=0, min_frequency=5.0
    )(ra, dec)
    assert a.shape == b.shape
    assert not torch.allclose(a, b)


def test_harmonic_sky_encoding_requires_enabled_components() -> None:
    with pytest.raises(ValueError, match="At least one harmonic component"):
        harmonic_sky_encoding(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            include_constant=False,
            include_degree1=False,
            include_degree2=False,
        )
