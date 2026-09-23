import pytest
import torch

from torchsky import WCS
from torchsky.map.field.core import ExplicitSpectralGridField, FieldDomain
from torchsky.map.product import MapBandpassSpec, MapProductResult, MapProductSpec


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
    product = MapProductResult(
        product=MapProductSpec(header, tuple(
            MapBandpassSpec(transmission_center=lam) for lam in centers
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
