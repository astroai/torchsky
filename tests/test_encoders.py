import torch

from torchsky.encoders import (
    SkyCoordinateEncoder,
    harmonic_sky_encoding,
    log_spectral_encoding,
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
