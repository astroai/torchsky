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


def test_per_pixel_sigma_matches_uniform_and_supports_varying_dispersion() -> None:
    torch.manual_seed(7)
    flux = torch.randn(2, 64, dtype=torch.float64)
    dln = 1.0e-4
    sigma_batch = torch.tensor([[60.0], [90.0]], dtype=torch.float64)
    sigma_pixel = sigma_batch.expand(2, 64).clone()

    out_uniform = convolve_log_lambda(flux, dln, sigma_batch, kernel_radius=8)
    out_per_pixel = convolve_log_lambda(flux, dln, sigma_pixel, kernel_radius=8)
    torch.testing.assert_close(out_per_pixel, out_uniform, atol=1e-12, rtol=1e-12)

    sigma_px = torch.linspace(0.0, 2.5, 64, dtype=torch.float64).unsqueeze(0).expand(2, 64).clone()
    sigma_px[:, :8] = 0.0
    sigma_px.requires_grad_(True)
    out_px = convolve_log_lambda(flux, sigma_px=sigma_px, kernel_radius=8)
    # Pixels with sigma_px == 0 must be exact identity
    torch.testing.assert_close(out_px[:, :8], flux[:, :8], atol=0.0, rtol=0.0)
    # Constant continuum must be preserved under spatially varying sigma_px
    flat = torch.ones_like(flux)
    out_flat = convolve_log_lambda(flat, sigma_px=sigma_px, kernel_radius=8)
    torch.testing.assert_close(out_flat[:, 10:-10], flat[:, 10:-10], atol=1e-12, rtol=1e-12)
    loss = out_px.square().sum()
    loss.backward()
    assert sigma_px.grad is not None
    assert torch.isfinite(sigma_px.grad).all()


def test_convolve_log_lambda_sigma_px_vectorization_identity_and_flux_conservation() -> None:
    torch.manual_seed(42)
    b, n_lam = 3, 128
    flux = torch.randn(b, n_lam, dtype=torch.float64)

    # 1. Scalar, [B], [B, 1], [B, L], [1, L]
    out_scalar = convolve_log_lambda(flux, sigma_px=1.5, kernel_radius=8)
    out_b1 = convolve_log_lambda(
        flux, sigma_px=torch.full((b, 1), 1.5, dtype=torch.float64), kernel_radius=8
    )
    out_bl = convolve_log_lambda(
        flux, sigma_px=torch.full((b, n_lam), 1.5, dtype=torch.float64), kernel_radius=8
    )
    torch.testing.assert_close(out_scalar, out_b1)
    torch.testing.assert_close(out_scalar, out_bl)

    # 2. Strict identity for non-positive sigma_px (zero and negative)
    neg_sigma_scalar = convolve_log_lambda(flux, sigma_px=-1.0, kernel_radius=8)
    torch.testing.assert_close(neg_sigma_scalar, flux, atol=0.0, rtol=0.0)

    zero_sigma_b = convolve_log_lambda(
        flux, sigma_px=torch.zeros(b, dtype=torch.float64), kernel_radius=8
    )
    torch.testing.assert_close(zero_sigma_b, flux, atol=0.0, rtol=0.0)

    # Mixed per-pixel [B, L] with negative, zero, and positive values
    mixed_sigma = torch.linspace(-1.0, 3.0, n_lam, dtype=torch.float64).unsqueeze(0).expand(b, n_lam)
    out_mixed = convolve_log_lambda(flux, sigma_px=mixed_sigma, kernel_radius=8)
    non_positive_mask = mixed_sigma <= 0.0
    torch.testing.assert_close(
        out_mixed[non_positive_mask], flux[non_positive_mask], atol=0.0, rtol=0.0
    )

    # 3. Flux conservation for localized emission lines
    x = torch.arange(n_lam, dtype=torch.float64)
    line = torch.exp(-0.5 * ((x - 64.0) / 4.0).square()).unsqueeze(0).expand(b, n_lam)
    line_total = line.sum(dim=-1)

    out_line_uniform = convolve_log_lambda(line, sigma_px=2.0, kernel_radius=12)
    torch.testing.assert_close(out_line_uniform.sum(dim=-1), line_total, atol=1e-12, rtol=1e-12)

    # Spatially varying sigma_px across line
    sigma_profile = torch.linspace(1.0, 3.0, n_lam, dtype=torch.float64).unsqueeze(0).expand(b, n_lam)
    out_line_varying = convolve_log_lambda(line, sigma_px=sigma_profile, kernel_radius=12)
    # Unit-integral kernel over smooth continuum/lines conserves integrated flux
    torch.testing.assert_close(out_line_varying.sum(dim=-1), line_total, atol=1e-2, rtol=1e-2)


