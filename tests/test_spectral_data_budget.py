"""Tests for observation-derived spectral quadrature and capacity hints."""

from __future__ import annotations

import torch

from torchsky.map.field import (
    FieldObservation,
    spectral_data_budget,
    suggest_num_spectral_samples,
    suggest_spectral_model_capacity,
)


def _obs(
    *,
    lo: float,
    hi: float,
    sigma: float | None = None,
    band: str | None = "b",
) -> FieldObservation:
    return FieldObservation(
        image=torch.zeros(2, 2, dtype=torch.float32),
        header={"NAXIS1": 2, "NAXIS2": 2},
        spectral_min=lo,
        spectral_max=hi,
        transmission_sigma=sigma,
        band_name=band,
    )


def test_spectral_data_budget_three_wide_bands() -> None:
    obs = [
        _obs(lo=4000.0, hi=5000.0, sigma=800.0, band="g"),
        _obs(lo=5000.0, hi=6000.0, sigma=800.0, band="r"),
        _obs(lo=6000.0, hi=7000.0, sigma=800.0, band="i"),
    ]
    b = spectral_data_budget(obs)
    assert b.wavelength_min == 4000.0
    assert b.wavelength_max == 7000.0
    assert b.wavelength_span == 3000.0
    assert b.n_distinct_band_specs == 3
    assert all(s.variation_scale == min(800.0, s.span) for s in b.observations)


def test_suggest_n_increases_with_narrower_sigma() -> None:
    wide = [_obs(lo=4000.0, hi=5000.0, sigma=500.0)]
    narrow = [_obs(lo=4000.0, hi=5000.0, sigma=50.0)]
    bw = spectral_data_budget(wide)
    bn = spectral_data_budget(narrow)
    assert suggest_num_spectral_samples(bn) >= suggest_num_spectral_samples(bw)


def test_delta_wavelength_observation_uses_floor() -> None:
    point = FieldObservation(
        image=torch.zeros(1, 1),
        header={"NAXIS1": 1, "NAXIS2": 1},
        wavelength=5500.0,
        band_name="mono",
    )
    b = spectral_data_budget([point])
    assert b.observations[0].span == 0.0
    n = suggest_num_spectral_samples(b, floor=3)
    assert n == 3


def test_many_narrow_bands_high_spectroscopicity_profile() -> None:
    obs = [
        _obs(lo=4000.0 + 20.0 * i, hi=4010.0 + 20.0 * i, sigma=2.0, band=str(i))
        for i in range(80)
    ]
    b = spectral_data_budget(obs)
    h = suggest_spectral_model_capacity(b)
    assert h.spectroscopicity > 100.0
    assert h.profile == "spectroscopic"
    assert h.spectral_frequencies >= 8


def test_cap_limits_suggested_samples() -> None:
    obs = [_obs(lo=4000.0, hi=9000.0, sigma=1.0)]
    b = spectral_data_budget(obs)
    n_uncapped = suggest_num_spectral_samples(b, cap=10_000)
    n_capped = suggest_num_spectral_samples(b, cap=12)
    assert n_capped == 12
    assert n_uncapped > n_capped
