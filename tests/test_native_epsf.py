"""Bounded independent reference and native-pixel ePSF inference contracts."""

from __future__ import annotations

import numpy as np
import pytest
import torch
from scipy.special import ndtr

from torchsky.map.operators.epsf import (
    GaussianPixelResponse,
    NativePixelStamp,
    native_pixel_stamp,
    profile_native_star,
)


def _stamp(shape=(9, 10), dtype=torch.float64, device="cpu"):
    image = torch.zeros(shape, dtype=dtype, device=device)
    return native_pixel_stamp(
        image,
        torch.ones_like(image),
        x_slice=slice(0, shape[1]),
        y_slice=slice(0, shape[0]),
    )


def _reference(pixel_xy, centroid, sigma):
    offsets = pixel_xy.detach().cpu().numpy() - np.asarray(centroid)
    intervals = ndtr((offsets + 0.5) / sigma) - ndtr((offsets - 0.5) / sigma)
    return torch.from_numpy(intervals.prod(axis=-1))


@pytest.mark.parametrize(
    "centroid", [(4.0, 3.0), (4.17, 3.43), (4.4999, 3.0001), (-0.3, 0.14)]
)
def test_native_phases_match_independent_gaussian_integral(centroid):
    stamp = _stamp()
    sigma = torch.tensor([0.38, 0.71], dtype=stamp.data.dtype)
    center = torch.tensor(centroid, dtype=stamp.data.dtype)
    response = GaussianPixelResponse(sigma)
    expected = _reference(stamp.pixel_xy, centroid, sigma.numpy())
    torch.testing.assert_close(
        response(stamp.pixel_xy - center), expected, atol=3e-16, rtol=1e-12
    )
    assert bool((response(stamp.pixel_xy - center) >= 0).all())


def test_tail_fraction_is_not_lost_to_cdf_cancellation():
    response = GaussianPixelResponse(torch.tensor(1.0, dtype=torch.float64))
    offsets = torch.tensor([[9.0, 0.0], [-9.0, 0.0]], dtype=torch.float64)
    values = response(offsets)
    assert values[0] > 0
    torch.testing.assert_close(values[0], values[1], atol=0, rtol=0)


@pytest.mark.parametrize("phase", [0.0, 0.13, 0.49])
@pytest.mark.parametrize("flux", [123.4, -2.7, 0.0])
def test_flux_and_affine_background_recovery_from_reference(phase, flux):
    stamp = _stamp(shape=(6, 7))
    center = torch.tensor([0.2 + phase, 1.23], dtype=stamp.data.dtype)
    sigma = torch.tensor([0.65, 0.9], dtype=stamp.data.dtype)
    reference = _reference(stamp.pixel_xy, center.numpy(), sigma.numpy())
    xy = stamp.pixel_xy
    basis = torch.stack((torch.ones_like(stamp.data), xy[..., 0], xy[..., 1]), dim=-1)
    bg = torch.tensor([3.1, -0.17, 0.23], dtype=stamp.data.dtype)
    stamp.data = flux * reference + basis @ bg
    stamp.variance = 0.3 + 0.07 * xy[..., 0]
    fit = profile_native_star(
        stamp, GaussianPixelResponse(sigma), center, background_basis=basis
    )
    torch.testing.assert_close(
        fit.flux, torch.tensor(flux, dtype=stamp.data.dtype), atol=1e-12, rtol=1e-12
    )
    torch.testing.assert_close(fit.background_coefficients, bg, atol=1e-13, rtol=1e-12)
    assert fit.chi2 < 1e-24
    assert fit.retained_response < 0.95
    assert fit.degrees_of_freedom == stamp.data.numel() - 4
    design = torch.cat((reference.unsqueeze(-1), basis), dim=-1).numpy().reshape(-1, 4)
    covariance = np.linalg.inv(
        design.T @ (design / stamp.variance.numpy().reshape(-1, 1))
    )
    torch.testing.assert_close(
        fit.conditional_covariance, torch.from_numpy(covariance), rtol=1e-12, atol=1e-12
    )


def test_no_crop_renormalization_and_masked_nonfinite_data():
    stamp = _stamp(shape=(3, 4))
    center = torch.tensor([-0.25, 0.17], dtype=stamp.data.dtype)
    sigma = torch.tensor(0.85, dtype=stamp.data.dtype)
    reference = _reference(stamp.pixel_xy, center.numpy(), sigma.numpy())
    stamp.data = 11.0 * reference + 1.5
    stamp.mask = torch.ones_like(stamp.data, dtype=torch.bool)
    stamp.mask[0, 0] = False
    stamp.data[0, 0] = torch.nan
    stamp.variance[0, 0] = torch.nan
    fit = profile_native_star(stamp, GaussianPixelResponse(sigma), center)
    torch.testing.assert_close(fit.flux, torch.tensor(11.0, dtype=stamp.data.dtype))
    torch.testing.assert_close(fit.retained_response, reference.sum())
    torch.testing.assert_close(fit.valid_response, reference[stamp.mask].sum())
    assert fit.valid_response < fit.retained_response < 0.6
    assert fit.residual[0, 0] == 0
    assert torch.isfinite(fit.chi2)


def test_lossy_response_retains_incident_flux_convention():
    stamp = _stamp(shape=(8, 8))
    center = torch.tensor([3.27, 4.16], dtype=stamp.data.dtype)
    sigma = torch.tensor([0.73, 0.91], dtype=stamp.data.dtype)
    expected = _reference(stamp.pixel_xy, center.numpy(), sigma.numpy())
    throughput = 0.64
    stamp.data = 45.0 * throughput * expected - 0.3

    def response(offset_xy):
        return throughput * GaussianPixelResponse(sigma)(offset_xy)

    fit = profile_native_star(stamp, response, center)
    torch.testing.assert_close(fit.flux, torch.tensor(45.0, dtype=stamp.data.dtype))
    torch.testing.assert_close(fit.retained_response, throughput * expected.sum())
    assert fit.retained_response < throughput


def test_zero_flux_signed_estimator_has_no_clipping_bias():
    stamp = _stamp(shape=(5, 6))
    center = torch.tensor([2.27, 1.41], dtype=stamp.data.dtype)
    sigma = torch.tensor([0.63, 0.81], dtype=stamp.data.dtype)
    noise = torch.linspace(
        -0.7, 0.9, stamp.data.numel(), dtype=stamp.data.dtype
    ).reshape(stamp.data.shape)
    stamp.data = 2.0 + noise
    plus = profile_native_star(stamp, GaussianPixelResponse(sigma), center)
    stamp.data = 2.0 - noise
    minus = profile_native_star(stamp, GaussianPixelResponse(sigma), center)
    assert plus.flux * minus.flux < 0
    torch.testing.assert_close(
        plus.flux + minus.flux,
        torch.tensor(0.0, dtype=stamp.data.dtype),
        atol=2e-14,
        rtol=0,
    )


def test_stamp_extraction_preserves_native_pixels_and_detector_coordinates():
    image = torch.arange(48, dtype=torch.float64).reshape(6, 8)
    variance = torch.ones_like(image)
    stamp = native_pixel_stamp(
        image, variance, x_slice=slice(3, 7), y_slice=slice(1, 4)
    )
    torch.testing.assert_close(stamp.data, image[1:4, 3:7])
    torch.testing.assert_close(
        stamp.pixel_xy[0, 0], torch.tensor([3.0, 1.0], dtype=image.dtype)
    )
    assert stamp.data.data_ptr() == image[1:4, 3:7].data_ptr()
    stamp.data[0, 0] = -7
    assert image[1, 3] == -7


def test_centroid_and_width_recovery_by_variable_projection():
    stamp = _stamp(shape=(9, 10))
    truth_center = np.array([4.27, 3.41])
    truth_sigma = np.array([0.53, 0.82])
    stamp.data = 74.0 * _reference(stamp.pixel_xy, truth_center, truth_sigma) + 2.3
    center = torch.tensor([4.0, 3.65], dtype=stamp.data.dtype, requires_grad=True)
    log_sigma = torch.tensor([0.7, 0.65], dtype=stamp.data.dtype).log().requires_grad_()
    optimizer = torch.optim.LBFGS(
        [center, log_sigma],
        max_iter=70,
        tolerance_grad=1e-11,
        tolerance_change=1e-14,
        line_search_fn="strong_wolfe",
    )

    def closure():
        optimizer.zero_grad()
        fit = profile_native_star(stamp, GaussianPixelResponse(log_sigma.exp()), center)
        loss = fit.chi2 / 2
        loss.backward()
        return loss

    optimizer.step(closure)
    torch.testing.assert_close(
        center, torch.from_numpy(truth_center), atol=2e-8, rtol=0
    )
    torch.testing.assert_close(
        log_sigma.exp(), torch.from_numpy(truth_sigma), atol=2e-8, rtol=0
    )
    fit = profile_native_star(stamp, GaussianPixelResponse(log_sigma.exp()), center)
    torch.testing.assert_close(
        fit.flux, torch.tensor(74.0, dtype=stamp.data.dtype), atol=2e-6, rtol=0
    )


def test_response_and_profile_gradients():
    stamp = _stamp(shape=(4, 5))
    stamp.data = _reference(stamp.pixel_xy, [1.2, 2.1], [0.6, 0.8]) * 12 + 0.2
    center = torch.tensor([1.31, 1.93], dtype=stamp.data.dtype, requires_grad=True)
    sigma = torch.tensor([0.57, 0.83], dtype=stamp.data.dtype, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda c, s: GaussianPixelResponse(s)(stamp.pixel_xy - c), (center, sigma)
    )
    assert torch.autograd.gradcheck(
        lambda c, s: profile_native_star(stamp, GaussianPixelResponse(s), c).chi2,
        (center, sigma),
    )
    # A symmetric pixel has zero first derivative but nonzero curvature.
    zero_offset = torch.zeros(2, dtype=stamp.data.dtype, requires_grad=True)
    assert torch.autograd.gradgradcheck(
        lambda xy: GaussianPixelResponse(sigma)(xy), (zero_offset,)
    )


@pytest.mark.parametrize("kind", ["identical", "masked", "zero", "underconstrained"])
def test_rank_deficiency_fails_closed(kind):
    stamp = _stamp(shape=(2, 2))
    center = torch.tensor([0.3, 0.4], dtype=stamp.data.dtype)
    response = GaussianPixelResponse(torch.tensor(0.8, dtype=stamp.data.dtype))
    basis = None
    if kind == "identical":
        basis = response(stamp.pixel_xy - center).unsqueeze(-1)
    if kind == "masked":
        stamp.mask = torch.zeros_like(stamp.data, dtype=torch.bool)
        stamp.mask[0, 0] = True
    if kind == "zero":

        def response(xy):
            return torch.zeros(xy.shape[:-1], dtype=xy.dtype, device=xy.device)

    if kind == "underconstrained":
        basis = torch.ones((*stamp.data.shape, 5), dtype=stamp.data.dtype)
    with pytest.raises(ValueError, match="rank-deficient"):
        profile_native_star(stamp, response, center, background_basis=basis)


@pytest.mark.parametrize("rcond", [-1, 1, float("nan"), float("inf")])
def test_invalid_rank_tolerance(rcond):
    stamp = _stamp()
    with pytest.raises(ValueError, match="rcond"):
        profile_native_star(
            stamp,
            GaussianPixelResponse(torch.tensor(1.0, dtype=stamp.data.dtype)),
            torch.tensor([3.1, 3.2], dtype=stamp.data.dtype),
            rcond=rcond,
        )


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_dtype_and_flux_only_fit(dtype):
    stamp = _stamp(dtype=dtype)
    center = torch.tensor([3.7, 4.2], dtype=dtype)
    response = GaussianPixelResponse(torch.tensor([0.5, 0.8], dtype=dtype))
    stamp.data = -3.0 * response(stamp.pixel_xy - center)
    fit = profile_native_star(
        stamp,
        response,
        center,
        background_basis=torch.empty((*stamp.data.shape, 0), dtype=dtype),
    )
    assert fit.flux.dtype == dtype
    assert fit.background_coefficients.shape == (0,)
    torch.testing.assert_close(fit.flux, torch.tensor(-3.0, dtype=dtype))


@pytest.mark.parametrize("device", ["cuda", "mps"])
def test_native_device_response(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    if device == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS unavailable")
    stamp = _stamp(dtype=torch.float32, device=device)
    center = torch.tensor([3.7, 4.2], device=device, requires_grad=True)
    sigma = torch.tensor([0.5, 0.8], device=device, requires_grad=True)
    response = GaussianPixelResponse(sigma)(stamp.pixel_xy - center)
    expected = GaussianPixelResponse(sigma.cpu())(stamp.pixel_xy.cpu() - center.cpu())
    torch.testing.assert_close(response.cpu(), expected)
    response.square().sum().backward()
    assert center.grad.device.type == device
    assert sigma.grad.device.type == device


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_cuda_profile_keeps_device_and_gradients():
    stamp = _stamp(dtype=torch.float32, device="cuda")
    center = torch.tensor([3.7, 4.2], device="cuda", requires_grad=True)
    sigma = torch.tensor([0.5, 0.8], device="cuda", requires_grad=True)
    stamp.data = (
        _reference(stamp.pixel_xy, [3.3, 4.1], [0.6, 0.8]).to("cuda", torch.float32) * 7
    )
    fit = profile_native_star(stamp, GaussianPixelResponse(sigma), center)
    fit.chi2.backward()
    assert fit.flux.device.type == "cuda"
    assert torch.isfinite(center.grad).all()
    assert torch.isfinite(sigma.grad).all()


def test_invalid_contracts_and_mutated_stamp_validation():
    stamp = _stamp()
    center = torch.tensor([3.7, 4.2], dtype=stamp.data.dtype)
    with pytest.raises(ValueError, match="dtype and device"):
        profile_native_star(stamp, GaussianPixelResponse(torch.tensor(1.0)), center)
    with pytest.raises(ValueError, match="positive"):
        GaussianPixelResponse(torch.tensor(0.0, dtype=stamp.data.dtype))(stamp.pixel_xy)
    with pytest.raises(ValueError, match="nonnegative"):
        profile_native_star(stamp, lambda xy: -torch.ones_like(xy[..., 0]), center)
    stamp.variance[0, 0] = 0
    with pytest.raises(ValueError, match="variance"):
        profile_native_star(
            stamp,
            GaussianPixelResponse(torch.tensor(1.0, dtype=stamp.data.dtype)),
            center,
        )
    with pytest.raises(ValueError, match="unit-step"):
        native_pixel_stamp(
            stamp.data, stamp.variance, x_slice=slice(0, 5, 2), y_slice=slice(0, 4)
        )
    with pytest.raises(ValueError, match="nonempty"):
        NativePixelStamp(
            torch.empty((0, 2)), torch.empty((0, 2)), torch.empty((0, 2, 2))
        )
