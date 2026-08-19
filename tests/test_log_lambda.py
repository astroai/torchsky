"""Unit-integral ln λ Gaussian convolution."""

from __future__ import annotations

import torch

from torchsky.log_lambda import convolve_log_lambda, sigma_R_kms
from torchsky.redshift import C_KM_S


def test_sigma_r_kms_matches_fwhm_definition() -> None:
    r = torch.tensor(2000.0)
    torch.testing.assert_close(sigma_R_kms(r), torch.tensor(C_KM_S / (2.355 * 2000.0)))


def test_tiny_sigma_is_grid_identity() -> None:
    flux = torch.randn(2, 64)
    out = convolve_log_lambda(flux, 1.0e-4, sigma_kms=0.0, kernel_radius=6)
    torch.testing.assert_close(out, flux)


def test_kernel_has_unit_sum_and_optional_zero_mean() -> None:
    flux = torch.ones(1, 128)
    dln = 1.0e-4
    sigma = 80.0
    out = convolve_log_lambda(flux, dln, sigma, kernel_radius=12, zero_mean=True)
    torch.testing.assert_close(out[:, 20:-20], flux[:, 20:-20], atol=1e-5, rtol=0)
    line = torch.zeros(1, 256)
    line[0, 128] = 1.0
    broadened = convolve_log_lambda(
        line, dln, sigma, kernel_radius=24, mean_kms=0.0, zero_mean=True
    )
    pixels = torch.arange(256, dtype=torch.float32)
    mean_px = (pixels * broadened[0]).sum()
    assert abs(float(mean_px) - 128.0) < 0.05


def test_convolve_log_lambda_adjoint() -> None:
    torch.manual_seed(0)
    flux = torch.randn(3, 96)
    residual = torch.randn(3, 96)
    dln = 2.0e-4
    sigma = torch.tensor([40.0, 80.0, 120.0])
    mean = 25.0
    forward = convolve_log_lambda(flux, dln, sigma, kernel_radius=10, mean_kms=mean)
    adjoint = convolve_log_lambda(
        residual, dln, sigma, kernel_radius=10, mean_kms=-mean
    )
    left = (forward * residual).sum()
    right = (flux * adjoint).sum()
    torch.testing.assert_close(left, right, atol=1e-5, rtol=1e-5)
