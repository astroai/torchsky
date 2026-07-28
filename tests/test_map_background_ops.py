from __future__ import annotations

import torch

from torchsky.map import FieldObservation, render_field_observation
from torchsky.map.operators.background import (
    BackgroundOp,
    SpatialSkyBackgroundOp,
    apply_background_components,
    constant_background,
    fit_background_plane,
    observation_background_components,
    polynomial_background,
)
from torchsky.map.sky import MeshSkyEstimator


class _ZeroField(torch.nn.Module):
    def forward(self, ra, dec, wavelength):
        return torch.zeros_like(ra, dtype=torch.float32)


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 4,
        "NAXIS2": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 2.0,
        "CRPIX2": 2.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def test_constant_background_helper_and_combiner() -> None:
    image = torch.ones((3, 3), dtype=torch.float32)
    bg = constant_background((3, 3), value=2.5, device=image.device)
    out = apply_background_components(image, detector=bg)
    torch.testing.assert_close(out, torch.full((3, 3), 3.5))


def test_polynomial_background_uses_structured_coefficients() -> None:
    bg = polynomial_background(
        (3, 3),
        coeffs=torch.tensor([1.0, 2.0, -3.0], dtype=torch.float32),
        device=torch.device("cpu"),
        normalized_coordinates=False,
    )

    yy, xx = torch.meshgrid(
        torch.arange(3, dtype=torch.float32),
        torch.arange(3, dtype=torch.float32),
        indexing="ij",
    )
    torch.testing.assert_close(bg, 1.0 + 2.0 * xx - 3.0 * yy)


def test_fit_background_plane_recovers_structured_residual() -> None:
    yy, xx = torch.meshgrid(
        torch.arange(5, dtype=torch.float32),
        torch.arange(6, dtype=torch.float32),
        indexing="ij",
    )
    residual = 0.5 * xx - 0.25 * yy + 1.5
    mask = torch.ones_like(residual, dtype=torch.bool)

    summary = fit_background_plane(residual, mask)

    torch.testing.assert_close(summary.model, residual, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(
        summary.coeffs, torch.tensor([0.5, -0.25, 1.5]), atol=1e-5, rtol=1e-5
    )
    assert summary.detrended_rms < 1e-5
    assert summary.leakage_fraction > 0.99


def test_fit_background_plane_uses_weights_and_rejects_source_outlier() -> None:
    yy, xx = torch.meshgrid(
        torch.arange(9, dtype=torch.float32),
        torch.arange(10, dtype=torch.float32),
        indexing="ij",
    )
    truth = 0.02 * xx - 0.03 * yy + 1.2
    observed = truth.clone()
    observed[4, 5] = 100.0
    weight = torch.ones_like(observed)
    weight[:, :2] = 0.25

    summary = fit_background_plane(
        observed,
        torch.ones_like(observed, dtype=torch.bool),
        weight=weight,
        sigma_clip=4.0,
    )

    torch.testing.assert_close(summary.model, truth, atol=2e-5, rtol=2e-5)
    torch.testing.assert_close(
        summary.coeffs,
        torch.tensor([0.02, -0.03, 1.2]),
        atol=2e-5,
        rtol=2e-5,
    )


def test_render_field_observation_applies_background_offset() -> None:
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
        background_offset=1.25,
    )
    rendered = render_field_observation(_ZeroField(), obs, num_spectral_samples=1)
    torch.testing.assert_close(rendered, torch.full((4, 4), 1.25))


def test_observation_background_components_separates_structured_terms() -> None:
    sky = torch.full((4, 4), 0.5)
    instrument = torch.full((4, 4), 1.0)
    detector = torch.full((4, 4), 2.0)
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
        sky_background=sky,
        instrument_background=instrument,
        detector_background=detector,
        background_offset=0.25,
    )

    components = observation_background_components(
        obs,
        shape=(4, 4),
        device=torch.device("cpu"),
        dtype=torch.float32,
    )

    torch.testing.assert_close(components["sky"], sky)
    torch.testing.assert_close(components["instrument"], instrument)
    torch.testing.assert_close(components["detector"], detector + 0.25)


def test_render_field_observation_applies_structured_background_components() -> None:
    yy, xx = torch.meshgrid(
        torch.arange(4, dtype=torch.float32),
        torch.arange(4, dtype=torch.float32),
        indexing="ij",
    )
    obs = FieldObservation(
        image=torch.zeros((4, 4), dtype=torch.float32),
        header=_simple_header(),
        wavelength=5000.0,
        sky_background=0.1 * xx,
        instrument_background=0.2 * yy,
        detector_background=torch.full((4, 4), 0.3),
        background_offset=0.4,
    )

    rendered = render_field_observation(_ZeroField(), obs, num_spectral_samples=1)

    torch.testing.assert_close(rendered, 0.1 * xx + 0.2 * yy + 0.7)


def test_field_observation_rejects_mismatched_background_shape() -> None:
    try:
        FieldObservation(
            image=torch.zeros((4, 4), dtype=torch.float32),
            header=_simple_header(),
            wavelength=5000.0,
            sky_background=torch.zeros((3, 4), dtype=torch.float32),
        )
    except ValueError as exc:
        assert "sky_background shape must match image shape" in str(exc)
    else:
        raise AssertionError("expected mismatched background shape to raise")


def test_spatial_sky_background_op_is_background_op() -> None:
    est = MeshSkyEstimator(block_size=16, statistic="median")
    op = SpatialSkyBackgroundOp(est)
    assert isinstance(op, BackgroundOp)
    assert op.name == "spatial_sky_background"
    assert not op.learned
    assert op.enabled
    assert op.background_estimator is est


def test_spatial_sky_background_op_can_estimate_sky() -> None:
    image = torch.ones((32, 32), dtype=torch.float32)
    est = MeshSkyEstimator(block_size=16, statistic="median")
    op = SpatialSkyBackgroundOp(est)
    estimator = op.background_estimator
    assert estimator is not None
    sky = estimator(image, None)
    assert sky.shape == (32, 32)
    assert torch.allclose(sky, torch.ones_like(sky), atol=1e-4)


def test_spatial_sky_background_op_can_be_disabled() -> None:
    est = MeshSkyEstimator(block_size=16, statistic="median")
    op = SpatialSkyBackgroundOp(est, name="test_bg")
    op.enabled = False
    assert not op.enabled
