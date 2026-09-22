import math

import pytest
import torch
import torch.nn.functional as F

from torchsky.map.field.core import FieldDomain, GaussianBasisField
from torchsky.map.product import MapBandpassSpec, MapProductResult, MapProductSpec


def domain():
    return FieldDomain(center_ra_deg=10.0, center_dec_deg=0.0,
                       spatial_scale_deg=0.01, wavelength_center=5000.0,
                       wavelength_scale=1000.0)


def test_splat_flux_spectrum_chunking_and_gradients():
    model = GaussianBasisField(domain(), splat=True, num_basis=1,
                               spectral_resolution=2, query_chunk_size=4096).double()
    with torch.no_grad():
        model.centers.zero_()
        model.log_scales.copy_(torch.tensor([[0.18, 0.25]]).log())
        model.shear.fill_(0.4)
        model.amplitudes.copy_(torch.tensor([[1.0, 3.0]]).expm1().log())
    axis = torch.linspace(-2, 2, 201, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    ra, dec = 10 + xx * 0.01, yy * 0.01
    with torch.no_grad():
        values = model(ra, dec, 5000.0)
        torch.testing.assert_close(values.sum() * (axis[1] - axis[0]).square(),
                                   torch.tensor(2.0, dtype=torch.float64), atol=2e-6, rtol=2e-6)
        model.query_chunk_size = 501
        torch.testing.assert_close(model(ra, dec, 5000.0), values)
        torch.testing.assert_close(model(ra, dec, 8000.0), 1.5 * values)
    pred = model(ra[::20, ::20] + 0.001, dec[::20, ::20], 5200.0)
    pred.square().sum().backward()
    for parameter in model.parameters():
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
        assert parameter.grad.abs().sum() > 0


def test_splat_basic_recovery():
    torch.manual_seed(1)
    model = GaussianBasisField(domain(), splat=True, num_basis=1,
                               spectral_resolution=2, query_chunk_size=512).double()
    axis = torch.linspace(-0.6, 0.6, 21, dtype=torch.float64)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    target = torch.exp(-0.5 * (((xx - 0.1) / 0.16).square() + ((yy + 0.08) / 0.23).square()))
    with torch.no_grad():
        model.centers.zero_()
        model.log_scales.fill_(math.log(0.25))
        model.amplitudes.fill_(math.log(math.expm1(0.15)))
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)
    losses = []
    for _ in range(100):
        optimizer.zero_grad()
        loss = F.mse_loss(model(10 + 0.01 * xx, 0.01 * yy, 5000.0), target)
        losses.append(float(loss.detach()))
        loss.backward()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.001


def test_splat_product_seed_and_checkpoint():
    header = {"NAXIS": 2, "NAXIS1": 16, "NAXIS2": 16,
              "CTYPE1": "RA---TAN", "CTYPE2": "DEC--TAN",
              "CRVAL1": 10.0, "CRVAL2": 0.0, "CRPIX1": 8.5, "CRPIX2": 8.5,
              "CDELT1": -0.001, "CDELT2": 0.001}
    plane = 1 + torch.arange(16).float()[None, :].expand(16, 16) / 16
    product = MapProductResult(product=MapProductSpec(header, (
        MapBandpassSpec(transmission_center=6000), MapBandpassSpec(transmission_center=4000))),
        data=torch.stack((3 * plane, plane)), weight=None, header=header)
    model = GaussianBasisField(domain(), splat=True, num_basis=16, spectral_resolution=3)
    model.initialize_from_product(product)
    assert torch.unique(model.centers, dim=0).shape[0] == 16
    assert model.amplitudes[:, 0].std() > 0
    torch.testing.assert_close(F.softplus(model.amplitudes[:, -1]), 3 * F.softplus(model.amplitudes[:, 0]))
    restored = GaussianBasisField(domain(), splat=True, num_basis=16, spectral_resolution=3)
    restored.load_state_dict(model.state_dict())
    ra, dec = torch.tensor([10.0]), torch.tensor([0.0])
    torch.testing.assert_close(restored(ra, dec, 5000.0), model(ra, dec, 5000.0))
    legacy = GaussianBasisField(domain())
    assert set(legacy.state_dict()) == {"centers", "log_scales", "amplitudes", "bias"}


def test_splat_invalid_inputs():
    with pytest.raises(ValueError):
        GaussianBasisField(domain(), splat=True, spectral_resolution=1)
    with pytest.raises(ValueError):
        GaussianBasisField(domain(), splat=True, init_scale=0)
