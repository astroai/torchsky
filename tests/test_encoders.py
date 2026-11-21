import torch
import pytest

from torchsky.encoders import (
    SkyCoordinateEncoder,
    harmonic_sky_encoding,
    log_spectral_encoding,
    _sky_encoder_fourier_min_frequency,
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
    small = _sky_encoder_fourier_min_frequency(patch_extent_deg=1.0)
    large = _sky_encoder_fourier_min_frequency(patch_extent_deg=10.0)
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


@pytest.mark.parametrize("min_frequency", [1.0, 2.7, 57.29577951308232])
def test_sky_coordinate_encoder_is_periodic_in_right_ascension(
    min_frequency: float,
) -> None:
    # The encoding is a function of sky position: RA and RA +- 360 deg are the
    # same point, also for the non-integer frequencies of domain-scaled encoders.
    enc = SkyCoordinateEncoder(min_frequency=min_frequency)
    ra = torch.tensor([0.0, 10.0, 179.0, 359.99], dtype=torch.float64)
    dec = torch.tensor([1.0, -20.0, 45.0, 3.0], dtype=torch.float64)
    reference = enc(ra, dec)
    torch.testing.assert_close(enc(ra - 360.0, dec), reference, rtol=0, atol=1e-9)
    torch.testing.assert_close(enc(ra + 360.0, dec), reference, rtol=0, atol=1e-9)


def test_domain_scaled_sky_encoder_is_continuous_across_ra_zero() -> None:
    from torchsky.map.field.sky_encoding import make_sky_coordinate_encoder
    from torchsky.map.field.types import FieldDomain

    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=0.1,
        wavelength_center=5000.0,
        wavelength_scale=1000.0,
    )
    enc = make_sky_coordinate_encoder(
        domain, num_fourier_frequencies=4, spectral_frequencies=2
    )
    dec = torch.zeros(2, dtype=torch.float64)
    half_step = 1.0e-6
    across = enc(torch.tensor([360.0 - half_step, half_step], dtype=torch.float64), dec)
    inside = enc(
        torch.tensor([0.05 - half_step, 0.05 + half_step], dtype=torch.float64), dec
    )
    jump_across = float((across[1] - across[0]).abs().max())
    jump_inside = float((inside[1] - inside[0]).abs().max())
    # A seam at RA = 0 inside the patch was an O(1) jump; the step across RA = 0
    # must be as small as an ordinary step of the same size inside the patch.
    assert jump_across < 1.0e-2
    assert jump_across <= 2.0 * jump_inside


def test_sky_coordinate_encoder_accepts_zero_spectral_frequencies() -> None:
    enc = SkyCoordinateEncoder(num_fourier_frequencies=2, spectral_frequencies=0)
    ra = torch.tensor([10.0, 20.0], dtype=torch.float64)
    dec = torch.tensor([0.0, 1.0], dtype=torch.float64)
    wavelength = torch.tensor([5000.0, 6000.0], dtype=torch.float64)
    feats = enc(ra, dec, wavelength=wavelength)
    spatial = enc(ra, dec)
    assert feats.shape == (2, spatial.shape[-1] + 1)
    torch.testing.assert_close(feats[:, -1], torch.log(wavelength))
    torch.testing.assert_close(
        log_spectral_encoding(wavelength=wavelength, num_frequencies=0),
        torch.log(wavelength).unsqueeze(-1),
    )
    with pytest.raises(ValueError, match="non-negative"):
        log_spectral_encoding(wavelength=wavelength, num_frequencies=-1)


def test_harmonic_sky_encoding_requires_enabled_components() -> None:
    with pytest.raises(ValueError, match="At least one harmonic component"):
        harmonic_sky_encoding(
            torch.tensor([0.0]),
            torch.tensor([0.0]),
            include_constant=False,
            include_degree1=False,
            include_degree2=False,
        )
