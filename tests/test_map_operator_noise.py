"""Focused checks for bounded structured detector-noise primitives."""

import pytest
import torch

from torchsky.map.operators.noise import (
    VAR_FLAT,
    VAR_POISSON,
    VAR_RNOISE,
    OneOverFRowNoise,
    PipelineVarianceCalibration,
    StructuredDetectorNoise,
    fft_whiten_rows,
    gaussian_student_t_mixture_nll,
    gaussian_student_t_mixture_responsibilities,
    one_over_f_row_psd,
    pipeline_variance,
    profile_amplifier_row_offsets,
)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_pipeline_components_and_err_fallback(dtype: torch.dtype) -> None:
    poisson = torch.full((2, 3, 4), 2.0, dtype=dtype)
    rnoise = torch.full((2, 3, 4), 3.0, dtype=dtype)
    flat = torch.full((2, 3, 4), 5.0, dtype=dtype)
    calibrated = pipeline_variance(
        {
            VAR_POISSON: poisson,
            VAR_RNOISE: rnoise,
            VAR_FLAT: flat,
        },
        poisson_scale=torch.tensor([1.0, 2.0], dtype=dtype),
        rnoise_scale=0.5,
        flat_scale=1.5,
        variance_floor=0.25,
    )
    expected_first = 1.0**2 * 2.0 + 0.5**2 * 3.0 + 1.5**2 * 5.0 + 0.25
    expected_second = 2.0**2 * 2.0 + 0.5**2 * 3.0 + 1.5**2 * 5.0 + 0.25
    torch.testing.assert_close(
        calibrated,
        torch.cat(
            (
                torch.full((1, 3, 4), expected_first, dtype=dtype),
                torch.full((1, 3, 4), expected_second, dtype=dtype),
            )
        ),
    )

    err = torch.full((3, 4), 2.0, dtype=dtype)
    fallback = pipeline_variance(err=err, err_scale=1.5, variance_floor=0.25)
    torch.testing.assert_close(fallback, torch.full_like(err, 9.25))


def test_pipeline_calibration_stays_positive_and_differentiable() -> None:
    model = PipelineVarianceCalibration(
        poisson_scale=1.2,
        rnoise_scale=0.8,
        flat_scale=1.1,
        variance_floor=1.0e-4,
    )
    output = model(
        {
            VAR_POISSON: torch.ones(4, 5),
            VAR_RNOISE: torch.ones(4, 5),
            VAR_FLAT: torch.ones(4, 5),
        }
    )
    assert bool(torch.all(output > 0.0))
    loss = output.mean()
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert bool(torch.isfinite(parameter.grad).all())
    assert bool(torch.all(model.poisson_scale > 0.0))
    assert bool(torch.all(model.variance_floor > 0.0))


def test_pipeline_rejects_invalid_variance_inputs() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        pipeline_variance({VAR_POISSON: torch.tensor([-1.0])})
    with pytest.raises(ValueError, match="positive"):
        pipeline_variance({VAR_POISSON: torch.ones(1)}, poisson_scale=0.0)
    with pytest.raises(ValueError, match="ERR"):
        pipeline_variance()


def test_one_over_f_psd_and_white_noise_identity_limit() -> None:
    psd = one_over_f_row_psd(
        32,
        white_psd=torch.tensor([1.0, 2.0], dtype=torch.float64),
        knee_frequency=torch.tensor([0.0, 0.05], dtype=torch.float64),
        sample_spacing=0.5,
    )
    assert psd.shape == (2, 17)
    assert bool(torch.isfinite(psd).all())
    assert bool(torch.all(psd > 0.0))
    torch.testing.assert_close(psd[0], torch.ones(17, dtype=torch.float64))

    rows = torch.randn(2, 32, dtype=torch.float64)
    constant_psd = torch.full((2, 17), 3.0, dtype=rows.dtype)
    whitened = fft_whiten_rows(rows, constant_psd)
    torch.testing.assert_close(whitened, rows, rtol=2.0e-6, atol=2.0e-6)


def test_one_over_f_parameters_have_finite_gradients_and_amp_broadcast() -> None:
    white = torch.tensor([1.0, 2.0, 3.0, 4.0], dtype=torch.float64, requires_grad=True)
    knee = torch.tensor(
        [0.01, 0.02, 0.03, 0.04], dtype=torch.float64, requires_grad=True
    )
    psd = one_over_f_row_psd(
        24,
        white_psd=white,
        knee_frequency=knee,
        exponent=1.0,
    )
    psd.mean().backward()
    assert white.grad is not None and bool(torch.isfinite(white.grad).all())
    assert knee.grad is not None and bool(torch.isfinite(knee.grad).all())

    model = OneOverFRowNoise((2, 4), white_psd=2.0, knee_frequency=0.02)
    rows = torch.randn(2, 4, 24)
    filtered = model(rows)
    assert filtered.shape == rows.shape
    filtered.square().mean().backward()
    for parameter in model.parameters():
        if parameter.grad is None:
            raise AssertionError("expected a finite gradient")
        assert bool(torch.isfinite(parameter.grad).all())
    assert bool(torch.all(model.white_psd > 0.0))
    assert bool(torch.all(model.knee_frequency > 0.0))
    assert bool(torch.all(model.exponent > 0.0))
    assert model(torch.empty(2, 4, 0)).shape == (2, 4, 0)


@pytest.mark.parametrize("dtype", (torch.float32, torch.float64))
def test_student_t_mixture_downweights_large_residuals(dtype: torch.dtype) -> None:
    residual = torch.tensor([0.0, 1.0, 12.0, -15.0], dtype=dtype)
    variance = torch.ones_like(residual)
    inlier, outlier = gaussian_student_t_mixture_responsibilities(
        residual,
        variance,
        outlier_scale=8.0,
        degrees_of_freedom=3.0,
        outlier_probability=0.01,
    )
    torch.testing.assert_close(inlier + outlier, torch.ones_like(residual))
    assert float(outlier[0]) < 0.01
    assert float(outlier[-1]) > 0.99
    prior = torch.tensor([0.5, 0.5, 0.5, 0.5], dtype=dtype)
    _, prior_outlier = gaussian_student_t_mixture_responsibilities(
        residual, variance, outlier_probability=0.01, prior=prior
    )
    assert float(prior_outlier[0]) > float(outlier[0])


def test_student_t_mixture_likelihood_and_gradients() -> None:
    residual = torch.tensor(
        [0.1, -0.4, 0.7, 1.2], dtype=torch.float64, requires_grad=True
    )
    variance = torch.full((4,), 1.5, dtype=torch.float64, requires_grad=True)
    scale = torch.tensor(7.0, dtype=torch.float64, requires_grad=True)
    nll = gaussian_student_t_mixture_nll(
        residual,
        variance,
        outlier_scale=scale,
        degrees_of_freedom=4.0,
        outlier_probability=0.01,
    )
    nll.backward()
    for value in (residual.grad, variance.grad, scale.grad):
        if value is None:
            raise AssertionError("expected a finite gradient")
        assert bool(torch.isfinite(value).all())

    # The injected inlier variance should be preferred to an over-dispersed
    # alternative when the residuals are ordinary inliers.
    true_nll = gaussian_student_t_mixture_nll(
        residual.detach(),
        torch.ones(4, dtype=torch.float64),
        outlier_probability=1.0e-5,
    )
    broad_nll = gaussian_student_t_mixture_nll(
        residual.detach(),
        torch.full((4,), 4.0, dtype=torch.float64),
        outlier_probability=1.0e-5,
    )
    assert float(true_nll) < float(broad_nll)


def test_student_t_mixture_edge_cases() -> None:
    empty = torch.empty(0)
    assert gaussian_student_t_mixture_nll(empty, empty, reduction="none").shape == (0,)
    with pytest.raises(ValueError, match="empty"):
        gaussian_student_t_mixture_nll(empty, empty)
    with pytest.raises(ValueError, match="prior"):
        gaussian_student_t_mixture_nll(
            torch.ones(2), torch.ones(2), prior=torch.tensor([-0.1, 0.1])
        )
    with pytest.raises(ValueError, match="variance"):
        gaussian_student_t_mixture_nll(torch.ones(2), torch.zeros(2))


def test_structured_detector_noise_profiles_amplifier_rows_and_scores_cosmics() -> None:
    height, width = 16, 20
    amplifier = (
        torch.arange(width)[None, :].expand(height, width) * 4 // width
    )
    injected_rows = torch.stack(
        [torch.linspace(-0.3 + 0.1 * amp, 0.2 + 0.1 * amp, height) for amp in range(4)]
    )
    residual = torch.zeros((height, width))
    for amp in range(4):
        residual = torch.where(
            amplifier == amp, injected_rows[amp, :, None], residual
        )
    variance = torch.ones_like(residual)
    offsets, corrected = profile_amplifier_row_offsets(
        residual, variance, amplifier
    )
    torch.testing.assert_close(offsets, injected_rows)
    torch.testing.assert_close(corrected, torch.zeros_like(corrected), atol=1e-6, rtol=0)

    residual[5, 11] = 15.0
    prior = torch.full_like(residual, 0.01)
    prior[5, 11] = 0.9
    model = StructuredDetectorNoise(num_amplifiers=4)
    nll = model.negative_log_likelihood(
        residual,
        err=torch.ones_like(residual),
        amplifier_index=amplifier,
        contamination_prior=prior,
    )
    _, outlier = model.contamination_responsibilities(
        residual,
        err=torch.ones_like(residual),
        amplifier_index=amplifier,
        contamination_prior=prior,
    )
    assert bool(torch.isfinite(nll))
    assert float(outlier[5, 11].detach()) > 0.9
    nll.backward()
    assert model.raw_outlier_scale.grad is not None
    assert model.row_noise.raw_knee_frequency.grad is not None
