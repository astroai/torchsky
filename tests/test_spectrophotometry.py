"""Analytic checks for the torch-native spectrophotometry contract."""

from __future__ import annotations

import pytest
import torch

from torchsky.spectrophotometry import (
    Passband,
    flux_to_magnitude,
    magnitude_to_flux,
    synthetic_photometry,
    synthetic_photometry_bands,
)


def test_constant_f_lambda_has_exact_photon_weighted_mean() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    result = synthetic_photometry(wavelength, torch.full_like(wavelength, 3.5), band)
    assert result.value.item() == pytest.approx(3.5)
    assert result.coverage.item() == pytest.approx(1.0)
    assert result.valid.item()


def test_constant_f_nu_is_preserved_and_ab_conversion_is_explicit() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 101)
    f_nu_microjy = 12.0
    f_nu_cgs = f_nu_microjy * 1.0e-29
    flux_lambda = f_nu_cgs * 2.99792458e18 / wavelength.square()
    result = synthetic_photometry(wavelength, flux_lambda, band, output="f_nu_microjy")
    assert result.value.item() == pytest.approx(f_nu_microjy, rel=1.0e-4)
    ab = synthetic_photometry(wavelength, flux_lambda, band, output="ab_mag")
    assert ab.value.item() == pytest.approx(
        -2.5 * torch.log10(torch.tensor(f_nu_microjy / 3.631e9)).item()
    )


def test_partial_filter_coverage_fails_closed() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(4500.0, 5500.0, 11), torch.ones(11), band
    )
    assert result.coverage.item() < 0.99
    assert not result.valid.item()
    assert torch.isnan(result.value)


def test_disjoint_filter_stays_invalid_even_with_zero_threshold() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(7000.0, 8000.0, 11), torch.ones(11), band, min_coverage=0.0
    )
    assert result.coverage.item() == 0.0
    assert not result.valid.item()
    assert torch.isnan(result.value)


def test_overlap_without_a_filter_sample_uses_clipped_boundaries() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    result = synthetic_photometry(
        torch.linspace(4500.0, 4900.0, 11), torch.ones(11), band
    )
    assert result.coverage.item() > 0.0
    assert not result.valid.item()


def test_unsorted_nm_curve_and_magnitude_round_trip() -> None:
    band = Passband.from_samples(
        torch.tensor([600.0, 400.0, 500.0]),
        torch.tensor([0.0, 0.0, 1.0]),
        wavelength_unit="nm",
    )
    assert band.wavelength.tolist() == [4000.0, 5000.0, 6000.0]
    flux = magnitude_to_flux(torch.tensor([22.0]), 3631.0)
    assert flux_to_magnitude(flux, 3631.0).item() == pytest.approx(22.0)


def test_batched_wavelength_matches_looped_1d() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wave_a = torch.linspace(4000.0, 6000.0, 21)
    wave_b = torch.linspace(4100.0, 5900.0, 21)
    flux_a = torch.full_like(wave_a, 3.5)
    flux_b = torch.full_like(wave_b, 2.0)
    stacked_wave = torch.stack((wave_a, wave_b))
    stacked_flux = torch.stack((flux_a, flux_b))
    batched = synthetic_photometry(stacked_wave, stacked_flux, band)
    looped = torch.stack(
        (
            synthetic_photometry(wave_a, flux_a, band).value,
            synthetic_photometry(wave_b, flux_b, band).value,
        )
    )
    torch.testing.assert_close(batched.value, looped)


def test_synthetic_photometry_bands_trailing_axis() -> None:
    g = Passband(torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0]))
    r = Passband(torch.tensor([5500.0, 6500.0, 7500.0]), torch.tensor([0.0, 1.0, 0.0]))
    wave = torch.linspace(4000.0, 7500.0, 71)
    flux = torch.full_like(wave, 4.0)
    result = synthetic_photometry_bands(wave, flux, [g, r])
    assert result.value.shape == (2,)
    assert result.value[0].item() == pytest.approx(4.0)
    assert result.value[1].item() == pytest.approx(4.0)


def test_mask_does_not_count_as_filter_coverage() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.zeros_like(wavelength, dtype=torch.bool)
    mask[:3] = True
    result = synthetic_photometry(wavelength, flux, band, mask=mask)
    assert result.coverage.item() < 0.99
    assert not result.valid.item()
    covered = synthetic_photometry(wavelength, flux, band, mask=torch.ones_like(mask))
    assert covered.valid.item()


def test_mask_allows_nan_on_dropped_samples() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.ones_like(wavelength, dtype=torch.bool)
    mask[10] = False
    flux[10] = torch.nan
    result = synthetic_photometry(wavelength, flux, band, mask=mask, min_coverage=0.0)
    assert torch.isfinite(result.value).item()


def test_mask_hole_is_not_interpolated_as_coverage() -> None:
    band = Passband(
        torch.tensor([4000.0, 5000.0, 6000.0]), torch.tensor([0.0, 1.0, 0.0])
    )
    wavelength = torch.linspace(4000.0, 6000.0, 21)
    flux = torch.ones_like(wavelength)
    mask = torch.ones_like(wavelength, dtype=torch.bool)
    mask[8:13] = False
    holed = synthetic_photometry(wavelength, flux, band, mask=mask, min_coverage=0.0)
    full = synthetic_photometry(wavelength, flux, band, min_coverage=0.0)
    assert holed.coverage.item() < full.coverage.item()
    assert holed.value.item() == pytest.approx(1.0)
    compacted = synthetic_photometry(
        torch.cat((wavelength[:8], wavelength[13:])),
        torch.ones(wavelength.numel() - 5),
        band,
        min_coverage=0.0,
    )
    assert compacted.coverage.item() > holed.coverage.item()


def test_evaluate_throughput_lvf_and_gaussian() -> None:
    from torchsky.spectrophotometry import evaluate_throughput

    wave = torch.tensor([500.0, 510.0, 520.0])
    gauss = evaluate_throughput(wave, center=510.0, sigma=5.0, wavelength_unit="nm")
    assert gauss[1].item() == pytest.approx(1.0)
    assert gauss[0].item() < gauss[1].item()
    lvf = evaluate_throughput(
        wave.unsqueeze(0).expand(2, -1),
        sigma=4.0,
        lambda_center_map=torch.tensor([500.0, 520.0]),
        wavelength_unit="nm",
    )
    assert lvf[0, 0].item() == pytest.approx(1.0)
    assert lvf[1, 2].item() == pytest.approx(1.0)

