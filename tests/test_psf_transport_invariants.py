"""Independent source-scatter references for detector-grid PSF transport."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.field.inverse import (
    bind_linear_band_grid_observation_operator,
    bind_linear_band_grid_observations_to_operator_set,
)
from torchsky.map.field.types import FieldDomain, FieldObservation, PSFSpec
from torchsky.map.operators import (
    ConditionedKernelPSF,
    ConstrainedKernelPSF,
    ExtendedResidualKernelPSF,
    PSFCalibrationTerm,
    SpatioSpectralTemporalEncoder,
)
from torchsky.map.operators.psf import apply_kernel2d, apply_kernel2d_variance


def _asymmetric_kernel() -> torch.Tensor:
    return (
        torch.tensor(
            [[1.0, 2.0, 3.0], [4.0, 8.0, 16.0], [5.0, 6.0, 7.0]],
            dtype=torch.float64,
        )
        / 52.0
    )


def _scatter(image: torch.Tensor, kernels: torch.Tensor) -> torch.Tensor:
    """Deposit each source's stamp at output = source + stamp offset."""
    height, width = image.shape
    ky, kx = kernels.shape[-2:]
    kernels = kernels.expand(height, width, ky, kx)
    out = torch.zeros_like(image)
    for dy in range(ky):
        for dx in range(kx):
            oy, ox = dy - ky // 2, dx - kx // 2
            sy0, sy1 = max(0, -oy), min(height, height - oy)
            sx0, sx1 = max(0, -ox), min(width, width - ox)
            out[sy0 + oy : sy1 + oy, sx0 + ox : sx1 + ox] += (
                image[sy0:sy1, sx0:sx1] * kernels[sy0:sy1, sx0:sx1, dy, dx]
            )
    return out


def _coordinates(side=7):
    y, x = torch.meshgrid(
        torch.linspace(-1, 1, side, dtype=torch.float64),
        torch.linspace(-1, 1, side, dtype=torch.float64),
        indexing="ij",
    )
    return x, y


def _set_basis(model, kernels):
    basis = model.kernel_basis if hasattr(model, "kernel_basis") else model
    with torch.no_grad():
        basis.raw_kernels.copy_(torch.log(torch.expm1(torch.stack(kernels))))


def _conditioned():
    model = ConditionedKernelPSF(
        SpatioSpectralTemporalEncoder(
            spatial_frequencies=1, spectral_frequencies=1, temporal_frequencies=1
        ),
        n_basis=2,
        kernel_size=3,
        hidden_dim=4,
    ).double()
    _set_basis(model, [_asymmetric_kernel(), _asymmetric_kernel().flip(-1)])
    with torch.no_grad():
        model.mixer[0].weight.fill_(0.2)
        model.mixer[-1].weight[0].fill_(0.8)
        model.mixer[-1].weight[1].fill_(-0.8)
    return model


@pytest.mark.parametrize(
    "family", ["explicit", "constrained", "conditioned", "extended"]
)
def test_asymmetric_stamp_is_the_rendered_impulse_and_calibration_response(family):
    kernel = _asymmetric_kernel()
    image = torch.zeros(7, 7, dtype=kernel.dtype)
    image[3, 3] = 1.0
    x, y = _coordinates()
    if family == "explicit":
        rendered = apply_kernel2d(image, kernel)
        expected = kernel
    elif family == "constrained":
        model = ConstrainedKernelPSF(n_basis=1, kernel_size=3).double()
        _set_basis(model, [kernel])
        rendered = model(image[None, None])[0, 0]
        expected = model.effective_kernels(None, 1)[0]
    else:
        model = (
            _conditioned()
            if family == "conditioned"
            else ExtendedResidualKernelPSF(
                SpatioSpectralTemporalEncoder(),
                core_kernel=kernel,
                wing_kernel=kernel,
                n_residual_basis=1,
            ).double()
        )
        rendered = model(image, spatial_x=x, spatial_y=y, wavelength=0.0)
        expected = model.effective_kernels(x[3:4, 3], y[3:4, 3], 0.0)
        term = PSFCalibrationTerm(
            model,
            x[3:4, 3],
            y[3:4, 3],
            torch.zeros(1),
            torch.zeros(1),
            expected.detach().reshape(1, 3, 3),
        )
        assert term.raw_loss().item() < 1e-25
        expected = term.predicted_kernels()[0]
    torch.testing.assert_close(rendered[2:5, 2:5], expected, atol=1e-14, rtol=1e-14)
    torch.testing.assert_close(rendered, _scatter(image, expected))


def test_source_conditioning_preserves_interior_flux_without_flat_sky_guarantee():
    model = _conditioned()
    x, y = _coordinates()
    kernels = model.effective_kernels(x, y, 0.0)
    image = torch.zeros_like(x)
    image[2, 2], image[4, 4] = 2.0, 5.0
    rendered = model(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(rendered, _scatter(image, kernels))
    torch.testing.assert_close(rendered.sum(), image.sum())

    flat = torch.ones_like(image)
    response = model(flat, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(response, _scatter(flat, kernels))
    assert (response[1:-1, 1:-1] - 1.0).abs().max() > 1e-4
    # Output-side weighting has unit interior row response, a different operator.
    mixture = model.mixture_weights(x, y, 0.0)
    output_mixed = sum(
        mixture[b] * _scatter(flat, basis)
        for b, basis in enumerate(model.kernel_basis.basis_kernels())
    )
    torch.testing.assert_close(output_mixed[1:-1, 1:-1], flat[1:-1, 1:-1])
    assert not torch.allclose(response, output_mixed)


def test_boundary_response_is_lost_not_renormalized():
    kernel = _asymmetric_kernel()
    image = torch.zeros(7, 7, dtype=kernel.dtype)
    image[0, 0] = 3.0
    rendered = apply_kernel2d(image, kernel)
    torch.testing.assert_close(rendered, _scatter(image, kernel))
    torch.testing.assert_close(rendered.sum(), 3.0 * kernel[1:, 1:].sum())
    assert rendered.sum() < image.sum()


def test_conditioned_adjoint_and_parameter_gradients_match_source_scatter():
    model = _conditioned()
    x, y = _coordinates(5)
    image = torch.linspace(-0.3, 1.0, 25, dtype=x.dtype).reshape(5, 5).requires_grad_()
    dual = torch.linspace(0.5, -0.4, 25, dtype=x.dtype).reshape(5, 5)
    expected = _scatter(image, model.effective_kernels(x, y, 0.0))
    rendered = model(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    torch.testing.assert_close(rendered, expected)
    inputs = (image, model.kernel_basis.raw_kernels, *model.mixer.parameters())
    actual_grads = torch.autograd.grad((rendered * dual).sum(), inputs)
    expected_grads = torch.autograd.grad((expected * dual).sum(), inputs)
    for actual, reference in zip(actual_grads, expected_grads, strict=True):
        torch.testing.assert_close(actual, reference, atol=1e-13, rtol=1e-12)
    torch.testing.assert_close((image * actual_grads[0]).sum(), (rendered * dual).sum())


def test_explicit_variance_matches_independent_dense_transport():
    kernel = _asymmetric_kernel()
    identity = torch.eye(25, dtype=kernel.dtype).reshape(25, 5, 5)
    matrix = torch.stack([_scatter(delta, kernel).flatten() for delta in identity], 1)
    variance = torch.linspace(0.2, 2.0, 25, dtype=kernel.dtype).reshape(5, 5)
    expected = (matrix.square() @ variance.flatten()).reshape(5, 5)
    torch.testing.assert_close(apply_kernel2d_variance(variance, kernel), expected)


def test_signed_kernel_helper_keeps_its_existing_normalization_contract():
    # The low-level helper accepts signed kernels; physical PSFSpec does not.
    kernel = _asymmetric_kernel().clone()
    kernel[0, 0] = -0.1
    image = torch.arange(25, dtype=kernel.dtype).reshape(5, 5)
    torch.testing.assert_close(
        apply_kernel2d(image, kernel), _scatter(image, kernel / kernel.sum())
    )


@pytest.mark.parametrize(
    "device,dtype",
    [
        ("cpu", torch.float32),
        ("cpu", torch.float64),
        pytest.param(
            "cuda",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA unavailable"
            ),
        ),
        pytest.param(
            "mps",
            torch.float32,
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS unavailable"
            ),
        ),
    ],
)
def test_explicit_and_conditioned_transport_preserve_device_and_dtype(device, dtype):
    model = _conditioned().to(device=device, dtype=dtype)
    x, y = (coord.to(device=device, dtype=dtype) for coord in _coordinates(5))
    image = torch.linspace(-0.3, 1.0, 25, device=device, dtype=dtype).reshape(5, 5)
    rendered = model(image, spatial_x=x, spatial_y=y, wavelength=0.0)
    assert rendered.device == image.device and rendered.dtype == image.dtype
    torch.testing.assert_close(
        rendered, _scatter(image, model.effective_kernels(x, y, 0.0))
    )
    kernel = _asymmetric_kernel().to(device=device, dtype=dtype)
    explicit = apply_kernel2d(image, kernel)
    assert explicit.device == image.device and explicit.dtype == image.dtype
    torch.testing.assert_close(explicit, _scatter(image, kernel))


class _IdentityDetector:
    def pixel_to_world(self, x, y):
        return x, y


@pytest.mark.parametrize("tiled", [False, True])
def test_explicit_inverse_forward_and_adjoint_match_dense_source_scatter(tiled):
    kernel = _asymmetric_kernel()
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=8.0,
        wavelength_center=0.0,
        wavelength_scale=1.0,
    )
    spec = PSFSpec(mode="kernel2d", kernel2d=kernel, pixel_response_included=True)
    observations = tuple(
        FieldObservation(
            image=torch.zeros(5, 5, dtype=kernel.dtype),
            header={"NAXIS1": 5, "NAXIS2": 5},
            weight=torch.ones(5, 5, dtype=kernel.dtype),
            psf_spec=spec,
            detector_to_sky=_IdentityDetector(),
            band_name="test",
            value_semantics="surface_brightness",
        )
        for _ in range(2)
    )
    kwargs = dict(
        spatial_resolution=5,
        spatial_bounds_local_deg=(0.0, 4.0, 0.0, 4.0),
        spatial_tile_hw=(2, 3) if tiled else None,
    )
    single = bind_linear_band_grid_observation_operator(
        observations[0], domain, **kwargs
    )
    joint = bind_linear_band_grid_observations_to_operator_set(
        observations, domain, **kwargs
    )
    image = torch.linspace(-0.3, 1.0, 25, dtype=kernel.dtype).reshape(5, 5)
    dual = image.flip(-1)
    identity = torch.eye(25, dtype=kernel.dtype).reshape(25, 5, 5)
    matrix = torch.stack([_scatter(delta, kernel).flatten() for delta in identity], 1)
    expected = (matrix @ image.flatten()).reshape(5, 5)
    adjoint = (matrix.T @ dual.flatten()).reshape(5, 5)
    torch.testing.assert_close(single.apply(image), expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(single.adjoint(dual), adjoint, atol=1e-12, rtol=1e-12)
    for output in joint.apply(image):
        torch.testing.assert_close(output, expected, atol=1e-12, rtol=1e-12)
    torch.testing.assert_close(
        joint.adjoint((dual, dual)), 2 * adjoint, atol=1e-12, rtol=1e-12
    )
