"""Analytic checks for the torch-native spectrophotometry contract."""

from __future__ import annotations

import pytest
import torch

from torchsky.spectrophotometry import (
    Passband,
    flux_to_magnitude,
    magnitude_to_flux,
    synthetic_photometry,
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
