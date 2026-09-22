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


def _splat_reference(model, coords):
    scales = model.log_scales.exp()
    diff = coords[:, None, :2] - model.centers
    x = diff[..., 0] / scales[:, 0]
    y = (diff[..., 1] - model.shear * diff[..., 0]) / scales[:, 1]
    basis = torch.exp(-0.5 * (x.square() + y.square())) / (
        2 * math.pi * scales.prod(dim=-1)
    )
    lam = coords[:, 2].clamp(model.spectral_knots[0], model.spectral_knots[-1])
    upper = torch.searchsorted(model.spectral_knots, lam.contiguous()).clamp(
        1, model.spectral_knots.numel() - 1
    )
    alpha = (lam - model.spectral_knots[upper - 1]) / (
        model.spectral_knots[upper] - model.spectral_knots[upper - 1]
    )
    flux = F.softplus(model.amplitudes)
    amplitude = flux[:, upper - 1].T * (1 - alpha[:, None]) + flux[:, upper].T * alpha[:, None]
    return (basis * amplitude).sum(dim=-1)


@pytest.mark.parametrize("shared_wavelength", [False, True])
@pytest.mark.parametrize("chunk_size", [1, 7, 100])
def test_splat_interpolation_order_values_and_gradients(shared_wavelength, chunk_size):
    torch.manual_seed(42)
    model = GaussianBasisField(domain(), splat=True, num_basis=11,
                               spectral_resolution=4, query_chunk_size=chunk_size).double()
    with torch.no_grad():
        model.log_scales.uniform_(-1.5, -0.5)
        model.shear.uniform_(-0.8, 0.8)
        model.amplitudes.uniform_(-2, 2)
        model.spectral_knots.copy_(torch.tensor([-1.0, -0.4, 0.2, 1.0]))
    coords = torch.randn(15, 3, dtype=torch.float64, requires_grad=True)
    with torch.no_grad():
        coords[:, 2] = 0.1 if shared_wavelength else torch.tensor(
            [-2, -1, -0.7, -0.4, -0.1, 0, 0.1, 0.2, 0.5, 0.8, 1, 2, -2, 0.1, 2]
        )
    weights = torch.randn(15, dtype=torch.float64)
    variables = (*model.parameters(), coords)
    actual, expected = model._splat_values(coords), _splat_reference(model, coords)
    torch.testing.assert_close(actual, expected, rtol=1e-12, atol=1e-12)
    actual_grads = torch.autograd.grad((actual * weights).sum(), variables)
    expected_grads = torch.autograd.grad((expected * weights).sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)

    ra = (10 + coords.detach()[:, 0].reshape(3, 5) * 0.01).requires_grad_()
    dec = (coords.detach()[:, 1].reshape(3, 5) * 0.01).requires_grad_()
    wavelength = (5000 + coords.detach()[:, 2].reshape(3, 5) * 1000).requires_grad_()
    actual = model(ra, dec, wavelength)
    expected = _splat_reference(model, model._encoded_coords(ra, dec, wavelength).reshape(-1, 3))
    torch.testing.assert_close(actual.flatten(), expected, rtol=1e-12, atol=1e-12)
    variables = (*model.parameters(), ra, dec, wavelength)
    actual_grads = torch.autograd.grad((actual.flatten() * weights).sum(), variables)
    expected_grads = torch.autograd.grad((expected * weights).sum(), variables)
    for actual_grad, expected_grad in zip(actual_grads, expected_grads):
        torch.testing.assert_close(actual_grad, expected_grad, rtol=1e-11, atol=1e-12)


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
