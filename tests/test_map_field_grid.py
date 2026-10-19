import math

import pytest
import torch

from torchsky import WCS
from torchsky.map.field.core import ExplicitSpectralGridField, FieldDomain
from torchsky.map.product import BandpassSpec, ProductResult, ProductSpec


def test_float32_grid_nodes_roundtrip_at_smacs_angular_scale():
    model = ExplicitSpectralGridField(FieldDomain(
        center_ra_deg=110.8375, center_dec_deg=-73.4392,
        spatial_scale_deg=8.0 / 3600,
        wavelength_center=20000.0, wavelength_scale=10000.0,
    ), spatial_resolution=256, spectral_resolution=4)
    ra, dec, wavelengths = model._grid_world_coords()
    actual = model._coords_unit(ra, dec, wavelengths[1])
    nodes = torch.linspace(0, 1, 256)
    yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
    torch.testing.assert_close(actual[..., 0], xx, atol=2e-7, rtol=0)
    torch.testing.assert_close(actual[..., 1], yy, atol=2e-7, rtol=0)


def test_explicit_spatial_bounds_grid_nodes_roundtrip():
    bounds = (-0.004, 0.006, -0.003, 0.005)
    model = ExplicitSpectralGridField(
        FieldDomain(
            center_ra_deg=110.8375,
            center_dec_deg=-73.4392,
            spatial_scale_deg=8.0 / 3600,
            wavelength_center=20000.0,
            wavelength_scale=10000.0,
        ),
        spatial_resolution=5,
        spectral_resolution=4,
        spatial_bounds_local_deg=bounds,
    )
    ra, dec, wavelengths = model._grid_world_coords()
    actual = model._coords_unit(ra, dec, wavelengths[1])
    nodes = torch.linspace(0, 1, 5)
    yy, xx = torch.meshgrid(nodes, nodes, indexing="ij")
    torch.testing.assert_close(actual[..., 0], xx, atol=2e-7, rtol=0)
    torch.testing.assert_close(actual[..., 1], yy, atol=2e-7, rtol=0)


@pytest.mark.parametrize(
    "bounds",
    [
        (0.0, 0.0, -1.0, 1.0),
        (-1.0, 1.0, 2.0, 1.0),
        (-1.0, float("nan"), -1.0, 1.0),
    ],
)
def test_explicit_spatial_bounds_must_be_finite_and_increasing(bounds):
    with pytest.raises(ValueError, match="spatial_bounds_local_deg"):
        ExplicitSpectralGridField(
            FieldDomain(
                center_ra_deg=10.0,
                center_dec_deg=0.0,
                spatial_scale_deg=0.001,
                wavelength_center=5000.0,
                wavelength_scale=1000.0,
            ),
            spatial_bounds_local_deg=bounds,
        )


def test_explicit_spatial_bounds_keep_border_sampling() -> None:
    model = ExplicitSpectralGridField(
        FieldDomain(
            center_ra_deg=10.0,
            center_dec_deg=0.0,
            spatial_scale_deg=0.001,
            wavelength_center=5000.0,
            wavelength_scale=1000.0,
        ),
        spatial_resolution=2,
        spectral_resolution=2,
        spatial_bounds_local_deg=(-0.01, 0.01, -0.01, 0.01),
    )
    with torch.no_grad():
        model.grid.copy_(
            torch.tensor(
                [[[[0.0, 1.0], [2.0, 3.0]], [[4.0, 5.0], [6.0, 7.0]]]]
            )
        )
    wavelength = torch.tensor([model.domain.wavelength_center], dtype=torch.float64)
    inside = model(
        torch.tensor([10.01], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        wavelength,
    )
    outside = model(
        torch.tensor([10.02], dtype=torch.float64),
        torch.tensor([0.0], dtype=torch.float64),
        wavelength,
    )
    torch.testing.assert_close(outside, inside)


def test_explicit_spatial_bounds_resolve_narrow_source_at_equal_grid_size() -> None:
    domain = FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=0.0,
        spatial_scale_deg=0.02,
        wavelength_center=5000.0,
        wavelength_scale=1000.0,
    )
    bounds = (-0.002, 0.002, -0.002, 0.002)

    def source(ra: torch.Tensor, dec: torch.Tensor) -> torch.Tensor:
        x = (ra - domain.center_ra_deg) * math.cos(
            math.radians(domain.center_dec_deg)
        )
        y = dec - domain.center_dec_deg
        return torch.exp(-0.5 * ((x / 0.0004) ** 2 + (y / 0.0004) ** 2))

    def initialize(model: ExplicitSpectralGridField) -> None:
        ra, dec, _ = model._grid_world_coords()
        values = source(ra, dec).clamp_min(1.0001e-4) - 1.0e-4
        raw = torch.log(torch.expm1(values.clamp_min(1.0e-8)))
        with torch.no_grad():
            model.grid.copy_(
                raw.unsqueeze(0).expand(1, model.spectral_resolution, -1, -1)
            )

    legacy = ExplicitSpectralGridField(
        domain,
        spatial_resolution=17,
        spectral_resolution=2,
    )
    explicit = ExplicitSpectralGridField(
        domain,
        spatial_resolution=17,
        spectral_resolution=2,
        spatial_bounds_local_deg=bounds,
    )
    initialize(legacy)
    initialize(explicit)

    x = torch.linspace(bounds[0], bounds[1], 81, dtype=torch.float64)
    y = torch.linspace(bounds[2], bounds[3], 81, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    ra = domain.center_ra_deg + xx
    dec = domain.center_dec_deg + yy
    wavelength = torch.full_like(ra, domain.wavelength_center)
    expected = source(ra, dec)
    legacy_rmse = torch.sqrt(torch.mean((legacy(ra, dec, wavelength) - expected) ** 2))
    explicit_rmse = torch.sqrt(
        torch.mean((explicit(ra, dec, wavelength) - expected) ** 2)
    )

    assert explicit_rmse < 0.25 * legacy_rmse


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_grid_initializer_recovers_nonconstant_product_at_nodes(dtype):
    header = {
        "NAXIS": 2, "NAXIS1": 101, "NAXIS2": 101,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0, "CRVAL2": 0.0, "CRPIX1": 51.0, "CRPIX2": 51.0,
        "CDELT1": -0.001, "CDELT2": 0.001,
    }
    domain = FieldDomain(
        center_ra_deg=10.0, center_dec_deg=0.0, spatial_scale_deg=0.001,
        wavelength_center=5000.0, wavelength_scale=1000.0,
    )
    model = ExplicitSpectralGridField(
        domain, spatial_resolution=5, spectral_resolution=5,
    ).to(dtype=dtype)
    yy, xx = torch.meshgrid(
        torch.arange(101, dtype=torch.float64),
        torch.arange(101, dtype=torch.float64), indexing="ij",
    )
    centers = (4000.0, 6000.0)
    data = torch.stack([1.0 + 0.01 * xx + 0.02 * yy + lam / 10000 for lam in centers])
    product = ProductResult(
        product=ProductSpec(header, tuple(
            BandpassSpec(transmission_center=lam) for lam in centers
        )), data=data, weight=None, header=header,
    )
    model.initialize_from_product(product)
    ra, dec, wavelengths = model._grid_world_coords()
    px, py = WCS(header).world_to_pixel(ra, dec)
    expected = (
        1.0 + 0.01 * px + 0.02 * py
        + wavelengths.clamp(*centers)[:, None, None] / 10000
    )
    actual = model(ra.expand(5, -1, -1), dec.expand(5, -1, -1), wavelengths[:, None, None])
    # The forward model adds its existing positive floor after softplus.
    tolerance = 1e-4 if dtype == torch.float32 else 1e-8
    torch.testing.assert_close(actual, (expected + 1e-4).to(dtype), atol=tolerance, rtol=tolerance)


def test_grid_initializer_stays_finite_for_bright_surface_brightness() -> None:
    header = {
        "NAXIS": 2, "NAXIS1": 3, "NAXIS2": 3,
        "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0, "CRVAL2": 0.0, "CRPIX1": 2.0, "CRPIX2": 2.0,
        "CDELT1": -0.001, "CDELT2": 0.001,
    }
    domain = FieldDomain(
        center_ra_deg=10.0, center_dec_deg=0.0, spatial_scale_deg=0.001,
        wavelength_center=5000.0, wavelength_scale=1000.0,
    )
    model = ExplicitSpectralGridField(
        domain, spatial_resolution=3, spectral_resolution=2,
    )
    product = ProductResult(
        product=ProductSpec(
            header,
            (
                BandpassSpec(transmission_center=4000.0),
                BandpassSpec(transmission_center=6000.0),
            ),
        ),
        data=torch.full((2, 3, 3), 1000.0),
        weight=None,
        header=header,
    )

    model.initialize_from_product(product)
    ra, dec, wavelengths = model._grid_world_coords()
    actual = model(
        ra.expand(2, -1, -1),
        dec.expand(2, -1, -1),
        wavelengths[:, None, None],
    )

    assert torch.isfinite(model.grid).all()
    torch.testing.assert_close(actual, torch.full_like(actual, 1000.0001))
