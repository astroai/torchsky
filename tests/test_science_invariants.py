"""Cross-namespace scientific invariants.

Each test pins a property that must hold for any reasonable input: index
self-consistency, integral conservation, round trips, and physical limits.
They complement the reference-comparison gates in torchsky-harness.
"""

from __future__ import annotations

import math

import pytest
import torch

import torchsky.catalogs as catalogs
import torchsky.cosmology as cosmology
import torchsky.extinction as extinction
import torchsky.spectral as spectral
import torchsky.spectrophotometry as spectrophotometry
from torchsky.sphere import MOC


def _random_sky(n: int, seed: int) -> tuple[torch.Tensor, torch.Tensor]:
    g = torch.Generator().manual_seed(seed)
    lon = torch.rand(n, generator=g, dtype=torch.float64) * 360.0 - 180.0
    lat = torch.asin(
        torch.rand(n, generator=g, dtype=torch.float64) * 2.0 - 1.0
    ).rad2deg()
    return lon, lat


def test_catalog_self_crossmatch_recovers_identity() -> None:
    lon, lat = _random_sky(200, seed=0)
    catalog = catalogs.Catalog(lon, lat)
    result = catalogs.crossmatch(catalog, catalog, radius_arcsec=0.01)

    assert torch.equal(result.left_index, result.right_index)
    assert int(result.separation_arcsec.abs().max() * 3600.0**2) < 1


def test_crossmatch_all_pairs_is_symmetric() -> None:
    g = torch.Generator().manual_seed(1)
    lon = torch.rand(50, generator=g, dtype=torch.float64) * 10.0
    lat = torch.zeros(50, dtype=torch.float64)
    left = catalogs.Catalog(lon, lat)
    right = catalogs.Catalog(lon + 0.0005, lat)

    ab = catalogs.crossmatch(left, right, 60.0, find="all")
    ba = catalogs.crossmatch(right, left, 60.0, find="all")
    assert len(ab.left_index) == len(ba.left_index) > 0


def test_flux_conserving_rebin_preserves_a_constant_flux() -> None:
    flux = torch.ones(5, dtype=torch.float64)
    input_edges = torch.linspace(5000.0, 6000.0, 6, dtype=torch.float64)
    output_edges = torch.linspace(5000.0, 6000.0, 3, dtype=torch.float64)

    result = spectral.flux_conserving_rebin(flux, input_edges, output_edges)
    assert torch.allclose(result.flux, torch.ones(2, dtype=torch.float64), atol=1e-12)


def test_flux_conserving_rebin_preserves_the_integral_on_a_nonuniform_grid() -> None:
    g = torch.Generator().manual_seed(2)
    edges = torch.sort(
        torch.rand(9, generator=g, dtype=torch.float64) * 4000.0 + 5000.0
    ).values
    flux = torch.rand(8, generator=g, dtype=torch.float64) + 0.5
    output_edges = torch.linspace(
        float(edges[0]), float(edges[-1]), 5, dtype=torch.float64
    )

    result = spectral.flux_conserving_rebin(flux, edges, output_edges)
    total_in = float((flux * (edges[1:] - edges[:-1])).sum())
    total_out = float(
        (result.flux * (result.spectral_edges[1:] - result.spectral_edges[:-1])).sum()
    )
    assert math.isclose(total_out, total_in, rel_tol=1e-12)


def test_magnitude_flux_roundtrip() -> None:
    flux = torch.tensor([1.0, 10.0, 1e-6], dtype=torch.float64)
    zero_point = 3631.0
    magnitudes = spectrophotometry.flux_to_magnitude(flux, zero_point)
    recovered = spectrophotometry.magnitude_to_flux(magnitudes, zero_point)
    assert torch.allclose(recovered, flux, rtol=1e-12)


def test_extinction_curve_normalisation_and_limits() -> None:
    wavelength = torch.linspace(3000.0, 8000.0, 11, dtype=torch.float64)
    k = extinction.k_lambda(wavelength)
    assert bool(torch.isfinite(k).all())
    assert float(k.min()) > 0.0
    assert float(
        extinction.k_lambda(torch.tensor([5500.0], dtype=torch.float64))
    ) == pytest.approx(1.0, abs=0.01)


def test_flat_wcdm_distance_relations() -> None:
    model = cosmology.FlatWCDM(H0=70.0, Om0=0.3)
    z = torch.tensor([0.1, 1.0, 2.0], dtype=torch.float64)

    distance_modulus = model.distance_modulus(z)
    luminosity_distance = model.luminosity_distance(z)
    assert torch.allclose(
        distance_modulus, 5.0 * torch.log10(luminosity_distance * 1e5), atol=1e-10
    )
    assert torch.allclose(
        model.distance_modulus(z),
        cosmology.distance_modulus(z, H0=70.0, om0=0.3),
        atol=1e-10,
    )
    angular = model.angular_diameter_distance(z)
    comoving = model.comoving_distance(z)
    assert torch.allclose(angular, comoving / (1.0 + z), rtol=1e-10)
    assert float(model.e(torch.tensor([0.0], dtype=torch.float64))) == pytest.approx(
        1.0
    )
    assert bool(torch.all(torch.diff(comoving) > 0))


def test_moc_circle_contains_its_centre_and_not_the_antipode() -> None:
    moc = MOC.from_circle(0.0, 0.0, radius_deg=10.0)
    assert moc.contains(0.0, 0.0)
    assert not moc.contains(180.0, 0.0)
    assert moc.area_sq_deg == pytest.approx(math.pi * 100.0, rel=0.02)


def test_healpix_interpolation_is_linear_in_map_values() -> None:
    from torchsky.sphere import get_interp_val

    nside = 64
    g = torch.Generator().manual_seed(3)
    map_a = torch.rand(12 * nside * nside, generator=g, dtype=torch.float64)
    map_b = torch.rand(12 * nside * nside, generator=g, dtype=torch.float64)
    lon = torch.rand(100, generator=g, dtype=torch.float64) * 360.0 - 180.0
    lat = torch.asin(
        torch.rand(100, generator=g, dtype=torch.float64) * 2.0 - 1.0
    ).rad2deg()

    value_a = get_interp_val(map_a, lon, lat, nside, lonlat=True)
    value_b = get_interp_val(map_b, lon, lat, nside, lonlat=True)
    value_sum = get_interp_val(map_a + map_b, lon, lat, nside, lonlat=True)
    assert torch.allclose(value_a + value_b, value_sum, atol=1e-6)


def test_moc_algebra_contains_operands() -> None:
    a = MOC.from_circle(0.0, 0.0, radius_deg=20.0)
    b = MOC.from_circle(180.0, 0.0, radius_deg=20.0)
    union = a.union(b)
    intersection = a.intersection(b)
    difference = a.difference(b)

    assert union.contains(0.0, 0.0) and union.contains(180.0, 0.0)
    assert not intersection.contains(0.0, 0.0)
    assert difference.contains(0.0, 0.0) and not difference.contains(180.0, 0.0)


def test_quality_bit_names_roundtrip() -> None:
    from torchsky.quality import QualityBit, quality_bit_names, quality_bits_from_names

    bits = QualityBit(0) | QualityBit(1) | QualityBit(2)
    names = quality_bit_names(bits)
    assert quality_bits_from_names(names) == bits
