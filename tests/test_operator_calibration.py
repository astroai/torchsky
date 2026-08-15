from __future__ import annotations

from typing import Any, cast

import torch
from torch import nn

from torchsky.map.operators import (
    AdaptiveSpatioSpectralTemporalEncoder,
    BackgroundCalibrationTerm,
    ConditionedBackgroundField,
    ConditionedKernelPSF,
    ExtendedResidualKernelPSF,
    FieldTrainingStage,
    LearnedBackgroundOp,
    LearnedPSFOp,
    ObservationOperatorBundle,
    PSFCalibrationTerm,
    SpatioSpectralTemporalEncoder,
    apply_operator_calibration_pruning,
    fit_operator_calibration_terms,
    operator_calibration_identifiability_report,
    operator_calibration_validation_pruning_report,
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


def _adaptive_encoder() -> AdaptiveSpatioSpectralTemporalEncoder:
    return AdaptiveSpatioSpectralTemporalEncoder(
        spatial_center=(104.0, 204.0),
        spatial_scale=(8.0, 8.0),
        spatial_length_bounds=((2.0, 16.0), (2.0, 16.0)),
        wavelength_center=27_500.0,
        spectral_scale=1_000.0,
        spectral_length_bounds=(500.0, 4_000.0),
        time_center_mjd=60_000.1,
        temporal_scale_days=0.2,
        temporal_length_bounds_days=(0.05, 0.8),
        spatial_bands=1,
        spectral_bands=1,
        temporal_bands=1,
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


def test_psf_calibration_accepts_extended_core_wing_model() -> None:
    model = ExtendedResidualKernelPSF(
        _encoder(), core_kernel_size=3, wing_kernel_size=5, n_residual_basis=2
    )
    x, y, wavelength, time = _coordinates()
    target = model.effective_kernels(x, y, wavelength, time).detach()
    term = PSFCalibrationTerm(model, x, y, wavelength, time, target)

    assert term.raw_loss().item() < 1.0e-12
    term.penalty().backward()
    assert model.raw_residual_fraction.grad is not None


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


def test_fit_operator_calibration_terms_reduces_training_loss_and_restores_mode() -> None:
    torch.manual_seed(4)
    model = ConditionedBackgroundField(_encoder(), hidden_dim=4, hidden_layers=1)
    model.eval()
    x, y, wavelength, time = _coordinates()
    term = BackgroundCalibrationTerm(
        model,
        x,
        y,
        wavelength,
        time,
        target_values=torch.tensor([1.5, 1.5, 1.5]),
    )

    result = fit_operator_calibration_terms([term], steps=80, lr=0.05)

    initial_report = result["initial_train_report"]
    final_report = result["final_train_report"]
    history = result["loss_history"]
    assert isinstance(initial_report, dict)
    assert isinstance(final_report, dict)
    assert isinstance(history, list)
    initial = initial_report["terms"][0]["raw_loss"]
    final = final_report["terms"][0]["raw_loss"]
    assert final < initial * 1.0e-3
    assert len(history) == 80
    trainable_parameter_count = result["trainable_parameter_count"]
    assert isinstance(trainable_parameter_count, int)
    assert trainable_parameter_count > 0
    assert model.training is False


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


def test_validation_pruning_report_can_be_applied_for_deployment() -> None:
    model = ConditionedBackgroundField(
        _adaptive_encoder(), hidden_dim=4, hidden_layers=1
    )
    report = {
        "schema_version": 1,
        "selection_split": "validation",
        "pruned_bands": [
            {"axis": "time_mjd", "band": 0},
            {"axis": "spatial_x", "band": 0},
        ],
    }

    applied = apply_operator_calibration_pruning(model, report)

    encoder = model.encoder
    assert isinstance(encoder, AdaptiveSpatioSpectralTemporalEncoder)
    assert applied["applied_band_count"] == 2
    assert float(encoder.temporal_axis.gates[0].detach()) < 1.0e-20
    assert float(encoder.x_axis.gates[0].detach()) < 1.0e-20
    assert float(encoder.y_axis.gates[0].detach()) > 0.0


def test_identifiability_report_restores_gates_and_scores_output_jacobians() -> None:
    model = ConditionedBackgroundField(
        _adaptive_encoder(), hidden_dim=4, hidden_layers=1
    )
    with torch.no_grad():
        for layer in model.network:
            if isinstance(layer, nn.Linear):
                layer.weight.fill_(0.1)
                layer.bias.fill_(0.01)
    x, y, wavelength, time = _coordinates()
    target = model(x, y, wavelength, time).detach() + 0.1
    term = BackgroundCalibrationTerm(model, x, y, wavelength, time, target)
    encoder = model.encoder
    assert isinstance(encoder, AdaptiveSpatioSpectralTemporalEncoder)
    saved = {
        name: axis.gate_logits.detach().clone()
        for name, axis in (
            ("x", encoder.x_axis),
            ("y", encoder.y_axis),
            ("spectral", encoder.spectral_axis),
            ("time", encoder.temporal_axis),
        )
    }
    saved_coordinates = (term.detector_x, term.detector_y, term.wavelength, term.time_mjd)
    model.eval()

    report = cast(
        dict[str, Any],
        operator_calibration_identifiability_report(
            term, split="validation", jacobian_probes=2
        ),
    )

    assert model.training is False
    assert len(report["band_ablations"]) == 4
    assert len(report["axis_center_ablations"]) == 4
    assert report["output_jacobian"]["method"] == "exact_scalar"
    assert report["output_jacobian"]["probes"] == 1
    assert (
        report["output_jacobian"]["axes"]["spatial_x"][
            "rms_output_jacobian_per_axis_unit"
        ]
        > 0.0
    )
    for name, axis in (
        ("x", encoder.x_axis),
        ("y", encoder.y_axis),
        ("spectral", encoder.spectral_axis),
        ("time", encoder.temporal_axis),
    ):
        torch.testing.assert_close(axis.gate_logits, saved[name])
    assert all(
        current is original
        for current, original in zip(
            (term.detector_x, term.detector_y, term.wavelength, term.time_mjd),
            saved_coordinates,
            strict=True,
        )
    )

    test_term = BackgroundCalibrationTerm(
        model, x, y, wavelength, time, target + 0.05
    )
    pruning = cast(
        dict[str, Any],
        operator_calibration_validation_pruning_report(term, test_term),
    )
    assert pruning["pruned_validation_raw_loss"] <= pruning[
        "initial_validation_raw_loss"
    ]
    for name, axis in (
        ("x", encoder.x_axis),
        ("y", encoder.y_axis),
        ("spectral", encoder.spectral_axis),
        ("time", encoder.temporal_axis),
    ):
        torch.testing.assert_close(axis.gate_logits, saved[name])

    psf = ConditionedKernelPSF(
        _adaptive_encoder(), n_basis=2, kernel_size=5, hidden_dim=4
    )
    mixer_output = psf.mixer[-1]
    assert isinstance(mixer_output, nn.Linear)
    with torch.no_grad():
        mixer_output.weight.normal_(generator=torch.Generator().manual_seed(7))
    mixture = psf.mixture_weights(x, y, wavelength, time).movedim(0, -1)
    kernels = torch.einsum(
        "nb,bhw->nhw", mixture, psf.kernel_basis.basis_kernels()
    ).detach()
    psf_report = cast(
        dict[str, Any],
        operator_calibration_identifiability_report(
            PSFCalibrationTerm(psf, x, y, wavelength, time, kernels),
            split="test",
            jacobian_probes=2,
        ),
    )
    assert psf_report["output_jacobian"]["method"] == "hutchinson_rademacher"
    assert psf_report["output_jacobian"]["probes"] == 2


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

    stages: tuple[
        tuple[FieldTrainingStage, tuple[bool, bool, bool]], ...
    ] = (
        ("scene", (True, False, False)),
        ("background", (False, True, False)),
        ("psf", (False, False, True)),
        ("joint", (True, True, True)),
    )
    for stage, expected in stages:
        report = set_field_training_stage(scene, bundles, stage)
        actual = (
            all(parameter.requires_grad for parameter in scene.parameters()),
            all(parameter.requires_grad for parameter in background.parameters()),
            all(parameter.requires_grad for parameter in psf.parameters()),
        )
        assert actual == expected
        assert report["stage"] == stage
