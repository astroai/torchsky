"""Tests for :mod:`torchsky.map.field.observation_builders`."""

from __future__ import annotations

import torch
from pytest import approx

from torchsky.map.field import (
    field_observations_from_bin_centers,
    field_observations_from_bin_edges,
    field_observations_from_ifu_cube,
    ifu_cube_spectral_to_leading,
    make_tangent_plane_header,
    spectral_data_budget,
)


def test_bin_edges_1d_spectrum_uses_1x1_images() -> None:
    edges = torch.tensor([4000.0, 4100.0, 4200.0, 4300.0])
    flux = torch.tensor([1.0, 2.0, 3.0])
    obs = field_observations_from_bin_edges(
        edges,
        flux,
        spectral_quantity="wavelength",
    )
    assert len(obs) == 3
    for o in obs:
        assert o.image.shape == (1, 1)
        assert o.spectral_quantity == "wavelength"
        assert o.spectral_max > o.spectral_min


def test_bin_edges_ifu_stack_requires_header() -> None:
    edges = torch.linspace(5000.0, 5100.0, 5)
    cube = torch.randn(4, 3, 2)
    hdr = make_tangent_plane_header(3, 2)
    obs = field_observations_from_bin_edges(edges, cube, spatial_header=hdr)
    assert len(obs) == 4
    assert obs[0].image.shape == (3, 2)


def test_ifu_spectral_axis_last() -> None:
    edges = torch.tensor([0.5, 1.0, 1.5, 2.0])
    # (H, W, L)
    cube_hwl = torch.randn(2, 2, 3)
    hdr = make_tangent_plane_header(2, 2)
    obs = field_observations_from_ifu_cube(
        cube_hwl,
        edges,
        hdr,
        spectral_axis=2,
        spectral_quantity="energy",
    )
    assert len(obs) == 3
    assert obs[0].spectral_quantity == "energy"


def test_bin_edges_carries_shared_and_per_bin_background_maps() -> None:
    edges = torch.tensor([5000.0, 5100.0, 5200.0])
    cube = torch.zeros((2, 3, 2), dtype=torch.float32)
    hdr = make_tangent_plane_header(3, 2)
    sky = torch.ones((3, 2), dtype=torch.float32)
    detector = torch.stack(
        [
            torch.full((3, 2), 2.0, dtype=torch.float32),
            torch.full((3, 2), 3.0, dtype=torch.float32),
        ]
    )

    obs = field_observations_from_bin_edges(
        edges,
        cube,
        spatial_header=hdr,
        sky_background=sky,
        detector_background=detector,
    )

    torch.testing.assert_close(obs[0].sky_background, sky)
    torch.testing.assert_close(obs[0].detector_background, detector[0])
    torch.testing.assert_close(obs[1].detector_background, detector[1])


def test_bin_edges_preserves_image_dtype_for_weight_and_backgrounds() -> None:
    edges = torch.tensor([5000.0, 5100.0, 5200.0])
    cube = torch.zeros((2, 3, 2), dtype=torch.float64)
    hdr = make_tangent_plane_header(3, 2)
    weight = torch.ones((3, 2), dtype=torch.float32)
    sky = torch.ones((3, 2), dtype=torch.float32)

    obs = field_observations_from_bin_edges(
        edges,
        cube,
        spatial_header=hdr,
        weight=weight,
        sky_background=sky,
    )

    assert obs[0].image.dtype == torch.float64
    assert obs[0].weight.dtype == torch.float64
    assert obs[0].sky_background.dtype == torch.float64


def test_bin_edges_rejects_background_shape_mismatch() -> None:
    edges = torch.tensor([5000.0, 5100.0, 5200.0])
    cube = torch.zeros((2, 3, 2), dtype=torch.float32)
    hdr = make_tangent_plane_header(3, 2)

    try:
        field_observations_from_bin_edges(
            edges,
            cube,
            spatial_header=hdr,
            instrument_background=torch.zeros((2, 2), dtype=torch.float32),
        )
    except ValueError as exc:
        assert "instrument_background spatial shape" in str(exc)
    else:
        raise AssertionError("expected background shape mismatch to raise")


def test_bin_centers_non_contiguous_intervals() -> None:
    centers = torch.tensor([1.0, 10.0])
    widths = torch.tensor([0.2, 0.2])
    values = torch.tensor([0.5, 1.5])
    obs = field_observations_from_bin_centers(centers, widths, values)
    assert obs[0].spectral_min == approx(0.9)
    assert obs[0].spectral_max == approx(1.1)
    assert obs[1].spectral_min == approx(9.9)
    assert obs[1].spectral_max == approx(10.1)


def test_spectral_budget_on_built_observations() -> None:
    edges = torch.linspace(1.0, 3.0, 11)
    flux = torch.ones(10)
    obs = field_observations_from_bin_edges(edges, flux)
    b = spectral_data_budget(obs)
    assert b.wavelength_min == 1.0
    assert b.wavelength_max == 3.0
    assert b.n_distinct_band_specs == 10


def test_ifu_cube_spectral_to_leading_is_noop_when_already_first() -> None:
    x = torch.randn(5, 2, 2)
    y = ifu_cube_spectral_to_leading(x, spectral_axis=0)
    assert y.shape == x.shape
    assert torch.equal(y, x)
