from __future__ import annotations

import torch
from torch import nn

from torchsky.map.operators import (
    BackgroundCalibrationTerm,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
    PSFCalibrationTerm,
    SpatioSpectralTemporalEncoder,
    operator_calibration_score_report,
    registered_image_stamps,
    set_field_training_stage,
)


def _encoder() -> SpatioSpectralTemporalEncoder:
    return SpatioSpectralTemporalEncoder(
        spatial_center=(100.0, 200.0),
        spatial_scale=(10.0, 10.0),
        wavelength_center=27_760.0,
        spectral_scale=1_000.0,
        time_center_mjd=60_000.0,
        temporal_scale_days=1.0,
        spatial_frequencies=1,
        spectral_frequencies=1,
        temporal_frequencies=1,
    )


def _coordinates() -> tuple[torch.Tensor, ...]:
    return (
        torch.tensor([100.0, 104.0, 108.0]),
        torch.tensor([201.0, 205.0, 207.0]),
        torch.tensor([27_000.0, 27_760.0, 28_000.0]),
        torch.tensor([60_000.0, 60_000.1, 60_000.2]),
    )


def test_psf_calibration_uses_normalized_effective_kernels() -> None:
    model = ConditionedKernelPSF(_encoder(), n_basis=2, kernel_size=5)
    x, y, wavelength, time = _coordinates()
    mixture = model.mixture_weights(x, y, wavelength, time).movedim(0, -1)
    target = torch.einsum(
        "nb,bhw->nhw", mixture, model.kernel_basis.basis_kernels()
    ).detach()
    term = PSFCalibrationTerm(
        model,
        x,
        y,
        wavelength,
        time,
        7.0 * target,
        sample_weight=torch.tensor([1.0, 2.0, 1.0]),
    )

    assert term.raw_loss().item() < 1.0e-12
    with torch.no_grad():
        model.kernel_basis.raw_kernels[0, 0, 0] += 1.0
    assert term.raw_loss().item() > 0.0
    term.penalty().backward()
    assert model.kernel_basis.raw_kernels.grad is not None


def test_psf_calibration_masks_unmeasured_pixels_and_reports_shape_errors() -> None:
    model = ConditionedKernelPSF(_encoder(), n_basis=2, kernel_size=5)
    x, y, wavelength, time = _coordinates()
    mixture = model.mixture_weights(x, y, wavelength, time).movedim(0, -1)
    target = torch.einsum(
        "nb,bhw->nhw", mixture, model.kernel_basis.basis_kernels()
    ).detach()
    pixel_weight = torch.ones_like(target)
    pixel_weight[:, 0, 0] = 0.0
    target[:, 0, 0] = 1.0e6
    term = PSFCalibrationTerm(
        model,
        x,
        y,
        wavelength,
        time,
        target,
        pixel_weight=pixel_weight,
    )

    diagnostics = term.diagnostics()
    assert torch.all(term.target_kernels[:, 0, 0] == 0.0)
    assert term.raw_loss().item() < 1.0e-12
    assert diagnostics["minimum_valid_pixel_fraction"] == 24 / 25
    assert float(diagnostics["centroid_rmse_pixels"]) >= 0.0
    assert float(diagnostics["maximum_encircled_energy_error"]) >= 0.0


def test_background_calibration_is_weighted_in_value_space() -> None:
    model = ConditionedBackgroundField(_encoder(), hidden_dim=4, hidden_layers=1)
    x, y, wavelength, time = _coordinates()
    term = BackgroundCalibrationTerm(
        model,
        x,
        y,
        wavelength,
        time,
        target_values=torch.tensor([1.0, 2.0, 3.0]),
        sample_weight=torch.tensor([1.0, 0.0, 3.0]),
        loss_weight=2.0,
    )

    assert term.raw_loss().item() == 7.0
    assert term.penalty().item() == 14.0
    term.penalty().backward()
    assert model.network[-1].bias.grad is not None
    diagnostics = term.diagnostics()
    assert diagnostics["weighted_bias"] == -2.5
    assert diagnostics["weighted_mae"] == 2.5


def test_operator_calibration_score_report_names_held_out_split() -> None:
    model = ConditionedBackgroundField(_encoder(), hidden_dim=4, hidden_layers=1)
    x, y, wavelength, time = _coordinates()
    term = BackgroundCalibrationTerm(
        model,
        x,
        y,
        wavelength,
        time,
        target_values=torch.tensor([1.0, 2.0, 3.0]),
    )

    report = operator_calibration_score_report([term], split="test")

    assert report["schema_version"] == 1
    assert report["split"] == "test"
    assert report["term_count"] == 1
    assert report["terms"] == [term.diagnostics()]


def test_registered_image_stamps_centres_fractional_sources() -> None:
    image = torch.zeros((15, 15), dtype=torch.float64)
    image[7, 8] = 1.0
    stamp = registered_image_stamps(
        image,
        torch.tensor([[7.5, 7.0]], dtype=torch.float64),
        radius_px=2,
    )[0]

    torch.testing.assert_close(stamp[2, 2], torch.tensor(0.5, dtype=stamp.dtype))
    torch.testing.assert_close(stamp[2, 3], torch.tensor(0.5, dtype=stamp.dtype))
    torch.testing.assert_close(stamp.sum(), torch.tensor(1.0, dtype=stamp.dtype))


def test_field_training_stages_toggle_scene_background_psf_and_joint() -> None:
    scene = nn.Linear(2, 1)
    psf = ConditionedKernelPSF(_encoder(), n_basis=2, kernel_size=5)
    background = ConditionedBackgroundField(
        _encoder(), hidden_dim=4, hidden_layers=1
    )
    bundles = [
        ObservationOperatorBundle(
            psf=LearnedPSFOp(psf),
            background=LearnedBackgroundOp(background, coordinate_frame="detector"),
        )
    ]

    for stage, expected in (
        ("scene", (True, False, False)),
        ("background", (False, True, False)),
        ("psf", (False, False, True)),
        ("joint", (True, True, True)),
    ):
        report = set_field_training_stage(scene, bundles, stage)
        actual = (
            all(parameter.requires_grad for parameter in scene.parameters()),
            all(parameter.requires_grad for parameter in background.parameters()),
            all(parameter.requires_grad for parameter in psf.parameters()),
        )
        assert actual == expected
        assert report["stage"] == stage
