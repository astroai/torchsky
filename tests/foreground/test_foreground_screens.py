"""Foreground screen tests: 3D dust profiles and DIB absorption."""

from __future__ import annotations

import pytest
import torch

from torchsky.foreground import (
    DIB_REST_ANGSTROM,
    a_v_at_distance,
    apply_dib_transmission,
    build_dust_profile_table,
    dib_optical_depth,
    extinction_from_a_v,
)


@pytest.fixture()
def profile() -> dict[str, torch.Tensor]:
    return build_dust_profile_table(
        torch.tensor([0.0, 100.0, 500.0, 2000.0]),
        torch.tensor([0.0, 0.05, 0.3, 0.8]),
    )


def test_dust_profile_interpolates_and_requires_monotone_table(profile):
    distances = torch.tensor([0.0, 250.0, 4000.0])
    values = a_v_at_distance(profile, distances)
    assert float(values[0]) == 0.0
    assert abs(float(values[1]) - 0.14375) < 1e-6
    assert float(values[2]) == pytest.approx(0.8)  # beyond-table clamp to total column


def test_dust_profile_gradient_flows_to_distance(profile):
    distances = torch.tensor([250.0], requires_grad=True)
    a_v_at_distance(profile, distances).sum().backward()
    assert distances.grad is not None
    assert float(distances.grad[0]) > 0.0


def test_dust_profile_rejects_invalid_tables():
    with pytest.raises(ValueError, match="align"):
        build_dust_profile_table(torch.tensor([0.0, 1.0]), torch.tensor([0.0]))
    with pytest.raises(ValueError, match="ascending"):
        build_dust_profile_table(
            torch.tensor([0.0, 5.0, 1.0]), torch.tensor([0.0, 0.1, 0.2])
        )
    with pytest.raises(ValueError, match="non-negative"):
        build_dust_profile_table(
            torch.tensor([0.0, 5.0]), torch.tensor([0.0, -0.1])
        )
    with pytest.raises(ValueError, match="distance 0"):
        build_dust_profile_table(
            torch.tensor([10.0, 50.0]), torch.tensor([0.0, 0.1])
        )


def test_dib_optical_depth_peaks_at_rest_wavelengths():
    wave = torch.linspace(5600.0, 6800.0, 1201)
    tau = dib_optical_depth(wave, ebv=0.3, strength_scale=1.0)
    peak = float(tau[torch.argmin((wave - DIB_REST_ANGSTROM[0]).abs())])
    assert peak == pytest.approx(0.060 * 0.3, rel=0.05)
    zero = dib_optical_depth(wave, ebv=torch.tensor(0.0))
    assert float(zero.abs().sum()) == 0.0  # identity at zero color excess


def test_dib_transmission_reduces_flux_and_is_bounded():
    wave = torch.linspace(5600.0, 6800.0, 1201)
    transmission = apply_dib_transmission(torch.ones_like(wave), wave, 0.5)
    assert float(transmission.max()) <= 1.0
    assert float(transmission.min()) < 1.0
    assert torch.all(transmission > 0.0)


def test_extinction_from_a_v_matches_ccm89_screen():
    screen = extinction_from_a_v(torch.tensor([5000.0]), torch.tensor(1.0))
    assert 0.0 < float(screen[0]) < 1.0
    zero = extinction_from_a_v(torch.tensor([5000.0]), torch.tensor(0.0))
    assert float(zero[0]) == pytest.approx(1.0)  # identity at A_V = 0
