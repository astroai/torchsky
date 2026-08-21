"""Calibration losses and explicit training stages for learned operators."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from typing import Literal, Protocol

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from .adaptive_axes import AdaptiveSpatioSpectralTemporalEncoder
from .background import ConditionedBackgroundField
from .bundle import ObservationOperatorBundle
from .learned_kernel import (
    ConditionedKernelPSF,
    ExtendedResidualKernelPSF,
    LowRankChromaticPSF,
)

FieldTrainingStage = Literal["scene", "background", "psf", "joint"]


class OperatorCalibrationTerm(Protocol):
    """A differentiable, evidence-space calibration penalty."""

    @property
    def model(self) -> nn.Module: ...

    @property
    def name(self) -> str: ...

    def penalty(self) -> Tensor: ...

    def diagnostics(self) -> dict[str, float | int | str]: ...


def _as_vector(value: Tensor, *, name: str, length: int | None = None) -> Tensor:
    result = torch.as_tensor(value).detach().reshape(-1)
    if result.numel() < 1 or not bool(torch.isfinite(result).all()):
        raise ValueError(f"{name} must contain finite samples")
    if length is not None and result.numel() != length:
        raise ValueError(f"{name} must have {length} samples")
    return result


def _validated_loss_weight(value: float) -> float:
    weight = float(value)
    if not math.isfinite(weight) or weight < 0.0:
        raise ValueError("loss_weight must be finite and non-negative")
    return weight


def _sample_weight(value: Tensor | None, *, length: int) -> Tensor:
    if value is None:
        return torch.ones(length, dtype=torch.float64)
    weights = _as_vector(value, name="sample_weight", length=length).to(
        dtype=torch.float64
    )
    if bool((weights < 0.0).any()) or float(weights.sum()) <= 0.0:
        raise ValueError("sample_weight must be non-negative with positive sum")
    return weights


def registered_image_stamps(
    image: Tensor,
    source_xy: Tensor,
    *,
    radius_px: int,
) -> Tensor:
    """Bilinearly sample source-centred, phase-registered image stamps."""
    if image.ndim != 2 or not image.is_floating_point():
        raise ValueError("image must be a floating-point 2D tensor")
    positions = torch.as_tensor(source_xy, device=image.device, dtype=image.dtype)
    if (
        positions.ndim != 2
        or positions.shape[1] != 2
        or positions.shape[0] < 1
        or not bool(torch.isfinite(positions).all())
    ):
        raise ValueError("source_xy must contain finite (x, y) positions")
    if not isinstance(radius_px, int) or radius_px < 1:
        raise ValueError("radius_px must be a positive integer")
    height, width = image.shape
    if (
        bool((positions[:, 0] < radius_px).any())
        or bool((positions[:, 0] > width - 1 - radius_px).any())
        or bool((positions[:, 1] < radius_px).any())
        or bool((positions[:, 1] > height - 1 - radius_px).any())
    ):
        raise ValueError("registered stamps must lie fully inside the image")
    offsets = torch.arange(
        -radius_px,
        radius_px + 1,
        device=image.device,
        dtype=image.dtype,
    )
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    grids = torch.stack(
        (
            (positions[:, None, None, 0] + xx)
            / max(width - 1, 1)
            * 2.0
            - 1.0,
            (positions[:, None, None, 1] + yy)
            / max(height - 1, 1)
            * 2.0
            - 1.0,
        ),
        dim=-1,
    )
    # ponytail: bilinear registration is detector-sampling exact enough for the
    # initial CAL evidence; replace with an oversampled ePSF only if held-out
    # encircled-energy residuals show phase-dependent structure.
    return F.grid_sample(
        image[None, None].expand(positions.shape[0], -1, -1, -1),
        grids,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]


@dataclass
class PSFCalibrationTerm:
    """Anchor a conditioned PSF to measured normalized effective kernels."""

    model: ConditionedKernelPSF | ExtendedResidualKernelPSF | LowRankChromaticPSF
    detector_x: Tensor
    detector_y: Tensor
    wavelength: Tensor
    time_mjd: Tensor
    target_kernels: Tensor
    sample_weight: Tensor | None = None
    pixel_weight: Tensor | None = None
    loss_weight: float = 1.0
    name: str = "psf"

    def __post_init__(self) -> None:
        if not isinstance(
            self.model,
            (ConditionedKernelPSF, ExtendedResidualKernelPSF, LowRankChromaticPSF),
        ):
            raise TypeError("PSFCalibrationTerm requires a conditioned kernel PSF")
        count = int(torch.as_tensor(self.detector_x).numel())
        self.detector_x = _as_vector(self.detector_x, name="detector_x")
        self.detector_y = _as_vector(
            self.detector_y, name="detector_y", length=count
        )
        self.wavelength = _as_vector(
            self.wavelength, name="wavelength", length=count
        )
        self.time_mjd = _as_vector(self.time_mjd, name="time_mjd", length=count)
        kernels = torch.as_tensor(self.target_kernels).detach()
        expected = (count, self.model.kernel_size, self.model.kernel_size)
        if tuple(kernels.shape) != expected or not bool(torch.isfinite(kernels).all()):
            raise ValueError(f"target_kernels must be finite with shape {expected}")
        if self.pixel_weight is None:
            pixel_weight = torch.ones_like(kernels, dtype=torch.float64)
        else:
            pixel_weight = (
                torch.as_tensor(self.pixel_weight).detach().to(torch.float64)
            )
            if (
                tuple(pixel_weight.shape) != expected
                or not bool(torch.isfinite(pixel_weight).all())
                or bool((pixel_weight < 0.0).any())
                or bool((pixel_weight.sum(dim=(-2, -1)) <= 0.0).any())
            ):
                raise ValueError(
                    f"pixel_weight must be finite, non-negative, and have shape {expected}"
                )
        kernels = torch.where(pixel_weight > 0.0, kernels, torch.zeros_like(kernels))
        sums = kernels.sum(dim=(-2, -1))
        if bool((sums <= 0.0).any()):
            raise ValueError("each target PSF kernel must have positive total flux")
        self.target_kernels = kernels / sums[:, None, None]
        self.sample_weight = _sample_weight(self.sample_weight, length=count)
        self.pixel_weight = pixel_weight
        self.loss_weight = _validated_loss_weight(self.loss_weight)

    def predicted_kernels(self) -> Tensor:
        parameter = next(self.model.parameters())
        coordinates = [
            self.detector_x.to(device=parameter.device, dtype=parameter.dtype),
            self.detector_y.to(device=parameter.device, dtype=parameter.dtype),
            self.wavelength.to(device=parameter.device),
            self.time_mjd.to(device=parameter.device),
        ]
        return self.model.effective_kernels(*coordinates)

    @property
    def smoothness_weight(self) -> float:
        return float(getattr(self, "_smoothness_weight", 1.0e-3))

    @smoothness_weight.setter
    def smoothness_weight(self, value: float) -> None:
        self._smoothness_weight = float(value)

    def raw_loss(self) -> Tensor:
        predicted = self.predicted_kernels()
        target = self.target_kernels.to(
            device=predicted.device, dtype=predicted.dtype
        )
        assert self.pixel_weight is not None
        pixel_weight = self.pixel_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        valid = pixel_weight > 0.0
        predicted = torch.where(valid, predicted, torch.zeros_like(predicted))
        predicted = predicted / predicted.sum(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(predicted.dtype).tiny
        )
        per_sample = (
            torch.square(predicted - target) * pixel_weight
        ).sum(dim=(-2, -1)) / pixel_weight.sum(dim=(-2, -1))
        assert self.sample_weight is not None
        weights = self.sample_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        return (per_sample * weights).sum() / weights.sum()

    def penalty(self) -> Tensor:
        total = self.loss_weight * self.raw_loss()
        if isinstance(self.model, LowRankChromaticPSF):
            # Smoothness prior on phi(u, lambda); unit integral and zero
            # centroid are enforced by kernel construction (see
            # gaussian_kernels_from_params), so only the field prior is added.
            total = total + self.smoothness_weight * self.model.smoothness_penalty()
        return total

    def diagnostics(self) -> dict[str, float | int | str]:
        predicted = self.predicted_kernels().detach()
        target = self.target_kernels.to(
            device=predicted.device, dtype=predicted.dtype
        )
        assert self.pixel_weight is not None
        valid = self.pixel_weight.to(device=predicted.device) > 0.0
        predicted = torch.where(valid, predicted, torch.zeros_like(predicted))
        predicted = predicted / predicted.sum(dim=(-2, -1), keepdim=True).clamp_min(
            torch.finfo(predicted.dtype).tiny
        )
        target_shape = target.clamp_min(0.0)
        target_shape = target_shape / target_shape.sum(
            dim=(-2, -1), keepdim=True
        ).clamp_min(torch.finfo(target.dtype).tiny)
        radius = self.model.kernel_size // 2
        offsets = torch.arange(
            -radius,
            radius + 1,
            device=predicted.device,
            dtype=predicted.dtype,
        )
        yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
        predicted_centroid = torch.stack(
            (
                (predicted * xx).sum(dim=(-2, -1)),
                (predicted * yy).sum(dim=(-2, -1)),
            ),
            dim=-1,
        )
        target_centroid = torch.stack(
            (
                (target_shape * xx).sum(dim=(-2, -1)),
                (target_shape * yy).sum(dim=(-2, -1)),
            ),
            dim=-1,
        )
        centroid_error = torch.linalg.vector_norm(
            predicted_centroid - target_centroid, dim=-1
        )
        aperture_radii = torch.arange(
            1,
            radius + 1,
            device=predicted.device,
            dtype=predicted.dtype,
        )
        radial_distance = torch.sqrt(xx.square() + yy.square())
        if aperture_radii.numel():
            aperture_masks = radial_distance[None] <= aperture_radii[:, None, None]
            predicted_ee = torch.einsum(
                "nhw,rhw->nr", predicted, aperture_masks.to(predicted.dtype)
            )
            target_ee = torch.einsum(
                "nhw,rhw->nr", target_shape, aperture_masks.to(target.dtype)
            )
            maximum_ee_error = float(torch.max(torch.abs(predicted_ee - target_ee)))
        else:
            maximum_ee_error = 0.0
        return {
            "name": self.name,
            "kind": "normalized_effective_psf_mse",
            "sample_count": int(self.detector_x.numel()),
            "loss_weight": self.loss_weight,
            "raw_loss": float(self.raw_loss().detach().cpu()),
            "centroid_rmse_pixels": float(
                torch.sqrt(torch.mean(centroid_error.square())).cpu()
            ),
            "maximum_encircled_energy_error": maximum_ee_error,
            "minimum_valid_pixel_fraction": float(
                (self.pixel_weight > 0.0).to(torch.float64).mean(dim=(-2, -1)).min()
            ),
        }


@dataclass
class BackgroundCalibrationTerm:
    """Anchor a conditioned background to calibrated detector-space values."""

    model: ConditionedBackgroundField
    detector_x: Tensor
    detector_y: Tensor
    wavelength: Tensor
    time_mjd: Tensor
    target_values: Tensor
    sample_weight: Tensor | None = None
    loss_weight: float = 1.0
    name: str = "background"

    def __post_init__(self) -> None:
        if not isinstance(self.model, ConditionedBackgroundField):
            raise TypeError(
                "BackgroundCalibrationTerm requires ConditionedBackgroundField"
            )
        count = int(torch.as_tensor(self.detector_x).numel())
        self.detector_x = _as_vector(self.detector_x, name="detector_x")
        self.detector_y = _as_vector(
            self.detector_y, name="detector_y", length=count
        )
        self.wavelength = _as_vector(
            self.wavelength, name="wavelength", length=count
        )
        self.time_mjd = _as_vector(self.time_mjd, name="time_mjd", length=count)
        self.target_values = _as_vector(
            self.target_values, name="target_values", length=count
        )
        self.sample_weight = _sample_weight(self.sample_weight, length=count)
        self.loss_weight = _validated_loss_weight(self.loss_weight)

    def raw_loss(self) -> Tensor:
        parameter = next(self.model.parameters())
        predicted = self.model(
            self.detector_x.to(device=parameter.device, dtype=parameter.dtype),
            self.detector_y.to(device=parameter.device, dtype=parameter.dtype),
            self.wavelength.to(device=parameter.device),
            self.time_mjd.to(device=parameter.device),
        )
        target = self.target_values.to(
            device=predicted.device, dtype=predicted.dtype
        )
        assert self.sample_weight is not None
        weights = self.sample_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        return (torch.square(predicted - target) * weights).sum() / weights.sum()

    def penalty(self) -> Tensor:
        return self.loss_weight * self.raw_loss()

    def diagnostics(self) -> dict[str, float | int | str]:
        parameter = next(self.model.parameters())
        predicted = self.model(
            self.detector_x.to(device=parameter.device, dtype=parameter.dtype),
            self.detector_y.to(device=parameter.device, dtype=parameter.dtype),
            self.wavelength.to(device=parameter.device),
            self.time_mjd.to(device=parameter.device),
        ).detach()
        residual = predicted - self.target_values.to(
            device=predicted.device, dtype=predicted.dtype
        )
        assert self.sample_weight is not None
        weights = self.sample_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        return {
            "name": self.name,
            "kind": "background_value_mse",
            "sample_count": int(self.detector_x.numel()),
            "loss_weight": self.loss_weight,
            "raw_loss": float(
                ((residual.square() * weights).sum() / weights.sum()).cpu()
            ),
            "weighted_bias": float(
                ((residual * weights).sum() / weights.sum()).cpu()
            ),
            "weighted_mae": float(
                ((residual.abs() * weights).sum() / weights.sum()).cpu()
            ),
        }


def operator_calibration_score_report(
    terms: Sequence[OperatorCalibrationTerm],
    *,
    split: Literal["train", "validation", "test"],
) -> dict[str, object]:
    """Evaluate one explicitly named calibration split without gradients."""
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    if not terms:
        raise ValueError("at least one operator calibration term is required")
    with torch.no_grad():
        reports = [term.diagnostics() for term in terms]
    return {
        "schema_version": 1,
        "split": split,
        "term_count": len(reports),
        "terms": reports,
    }


def operator_calibration_identifiability_report(
    term: PSFCalibrationTerm | BackgroundCalibrationTerm,
    *,
    split: Literal["validation", "test"],
    jacobian_probes: int = 4,
    seed: int = 1729,
) -> dict[str, object]:
    """Measure held-out band utility and output sensitivity of adaptive axes.

    Each Fourier band is disabled without refitting and the fractional change
    in raw held-out loss is reported. Coordinate sensitivities are exact for a
    scalar background and use deterministic Rademacher projections for a PSF
    kernel, avoiding a full output Jacobian.
    """
    if split not in {"validation", "test"}:
        raise ValueError("split must be validation or test")
    probes = int(jacobian_probes)
    if probes < 1:
        raise ValueError("jacobian_probes must be positive")
    model = term.model
    encoder = model.encoder
    if not isinstance(encoder, AdaptiveSpatioSpectralTemporalEncoder):
        raise ValueError("identifiability diagnostics require an adaptive encoder")
    parameter = next(model.parameters())
    axes = (
        ("spatial_x", encoder.x_axis),
        ("spatial_y", encoder.y_axis),
        ("spectral", encoder.spectral_axis),
        ("time_mjd", encoder.temporal_axis),
    )
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            baseline = float(term.raw_loss().detach().cpu())
            centered_axes: list[dict[str, float | str]] = []
            coordinate_axes = (
                ("spatial_x", "detector_x", encoder.x_axis.center),
                ("spatial_y", "detector_y", encoder.y_axis.center),
                ("spectral", "wavelength", encoder.spectral_axis.center),
                ("time_mjd", "time_mjd", encoder.temporal_axis.center),
            )
            denominator = max(abs(baseline), torch.finfo(torch.float64).tiny)
            for axis_name, attribute, center in coordinate_axes:
                original = getattr(term, attribute)
                try:
                    setattr(term, attribute, torch.full_like(original, center))
                    ablated = float(term.raw_loss().detach().cpu())
                finally:
                    setattr(term, attribute, original)
                centered_axes.append(
                    {
                        "axis": axis_name,
                        "center": center,
                        "centered_raw_loss": ablated,
                        "loss_increase_fraction": (ablated - baseline)
                        / denominator,
                    }
                )
            ablations: list[dict[str, float | int | str]] = []
            for axis_name, axis in axes:
                saved_logits = axis.gate_logits.detach().clone()
                try:
                    for band in range(axis.num_bands):
                        axis.gate_logits[band] = -80.0
                        ablated = float(term.raw_loss().detach().cpu())
                        ablations.append(
                            {
                                "axis": axis_name,
                                "band": band,
                                "gate": float(torch.sigmoid(saved_logits[band]).cpu()),
                                "length_scale": float(
                                    axis.length_scales[band].detach().cpu()
                                ),
                                "ablated_raw_loss": ablated,
                                "loss_increase_fraction": (
                                    ablated - baseline
                                )
                                / denominator,
                            }
                        )
                        axis.gate_logits[band] = saved_logits[band]
                finally:
                    axis.gate_logits.copy_(saved_logits)

        coordinates = [
            term.detector_x.to(device=parameter.device, dtype=parameter.dtype)
            .detach()
            .requires_grad_(),
            term.detector_y.to(device=parameter.device, dtype=parameter.dtype)
            .detach()
            .requires_grad_(),
            # Preserve small offsets from large wavelength/MJD origins until
            # the encoders subtract their float64 physical-axis centres.
            term.wavelength.to(device=parameter.device)
            .detach()
            .requires_grad_(),
            term.time_mjd.to(device=parameter.device).detach().requires_grad_(),
        ]
        if isinstance(term, PSFCalibrationTerm):
            output = term.model.effective_kernels(*coordinates)
            effective_probes = probes
            method = "hutchinson_rademacher"
        else:
            output = term.model(*coordinates)
            effective_probes = 1
            method = "exact_scalar"
        squared_gradients = [torch.zeros_like(value) for value in coordinates]
        generator = torch.Generator(device="cpu").manual_seed(int(seed))
        for probe_index in range(effective_probes):
            if isinstance(term, PSFCalibrationTerm):
                projection = (
                    torch.randint(
                        0,
                        2,
                        output.shape,
                        generator=generator,
                        device="cpu",
                    )
                    .to(device=output.device, dtype=output.dtype)
                    .mul_(2.0)
                    .sub_(1.0)
                )
            else:
                projection = torch.ones_like(output)
            gradients = torch.autograd.grad(
                (output * projection).sum(),
                coordinates,
                retain_graph=probe_index + 1 < effective_probes,
            )
            squared_gradients = [
                total + gradient.detach().square()
                for total, gradient in zip(
                    squared_gradients, gradients, strict=True
                )
            ]
        sensitivities = {}
        for (axis_name, axis), squared in zip(axes, squared_gradients, strict=True):
            rms = float(torch.sqrt(squared.mean() / effective_probes).cpu())
            sensitivities[axis_name] = {
                "rms_output_jacobian_per_axis_unit": rms,
                "rms_output_change_per_reference_scale": rms
                * axis.reference_scale,
                "reference_scale": axis.reference_scale,
            }
    finally:
        model.train(training)
    return {
        "schema_version": 1,
        "split": split,
        "term": term.name,
        "baseline_raw_loss": baseline,
        "axis_center_ablations": centered_axes,
        "band_ablations": ablations,
        "output_jacobian": {
            "method": method,
            "probes": effective_probes,
            "seed": int(seed),
            "axes": sensitivities,
        },
    }


def operator_calibration_validation_pruning_report(
    validation_term: PSFCalibrationTerm | BackgroundCalibrationTerm,
    test_term: PSFCalibrationTerm | BackgroundCalibrationTerm,
    *,
    minimum_validation_improvement_fraction: float = 0.005,
) -> dict[str, object]:
    """Greedily prune adaptive Fourier bands using validation evidence only."""
    threshold = float(minimum_validation_improvement_fraction)
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError(
            "minimum_validation_improvement_fraction must be finite and non-negative"
        )
    if validation_term.model is not test_term.model:
        raise ValueError("validation and test terms must share one model")
    if type(validation_term) is not type(test_term):
        raise TypeError("validation and test calibration terms must have the same kind")
    model = validation_term.model
    encoder = model.encoder
    if not isinstance(encoder, AdaptiveSpatioSpectralTemporalEncoder):
        raise ValueError("validation pruning requires an adaptive encoder")
    axes = (
        ("spatial_x", encoder.x_axis),
        ("spatial_y", encoder.y_axis),
        ("spectral", encoder.spectral_axis),
        ("time_mjd", encoder.temporal_axis),
    )
    saved_logits = {name: axis.gate_logits.detach().clone() for name, axis in axes}
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            initial_validation = float(validation_term.raw_loss().detach().cpu())
            initial_test = float(test_term.raw_loss().detach().cpu())
            current_validation = initial_validation
            remaining = [
                (name, axis, band)
                for name, axis in axes
                for band in range(axis.num_bands)
            ]
            pruned: list[dict[str, float | int | str]] = []
            while remaining:
                trials = []
                for name, axis, band in remaining:
                    original = axis.gate_logits[band].detach().clone()
                    axis.gate_logits[band] = -80.0
                    loss = float(validation_term.raw_loss().detach().cpu())
                    axis.gate_logits[band] = original
                    trials.append((loss, name, axis, band))
                loss, name, axis, band = min(trials, key=lambda item: item[0])
                improvement = (current_validation - loss) / max(
                    abs(current_validation), torch.finfo(torch.float64).tiny
                )
                if improvement < threshold:
                    break
                gate = float(axis.gates[band].detach().cpu())
                axis.gate_logits[band] = -80.0
                pruned.append(
                    {
                        "axis": name,
                        "band": band,
                        "gate": gate,
                        "validation_improvement_fraction": improvement,
                    }
                )
                current_validation = loss
                remaining = [
                    item
                    for item in remaining
                    if not (item[0] == name and item[2] == band)
                ]
            final_test = float(test_term.raw_loss().detach().cpu())
    finally:
        with torch.no_grad():
            for name, axis in axes:
                axis.gate_logits.copy_(saved_logits[name])
        model.train(training)
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "evaluation_split": "test",
        "minimum_validation_improvement_fraction": threshold,
        "pruned_bands": pruned,
        "initial_validation_raw_loss": initial_validation,
        "pruned_validation_raw_loss": current_validation,
        "validation_improvement_fraction": 1.0
        - current_validation
        / max(abs(initial_validation), torch.finfo(torch.float64).tiny),
        "initial_test_raw_loss": initial_test,
        "pruned_test_raw_loss": final_test,
        "test_improvement_fraction": 1.0
        - final_test / max(abs(initial_test), torch.finfo(torch.float64).tiny),
    }


def apply_operator_calibration_pruning(
    model: ConditionedKernelPSF
    | ExtendedResidualKernelPSF
    | ConditionedBackgroundField,
    report: Mapping[str, object],
) -> dict[str, object]:
    """Apply a validation-selected Fourier pruning report to a fitted model."""
    if report.get("schema_version") != 1 or report.get("selection_split") != "validation":
        raise ValueError("expected a schema-1 validation pruning report")
    raw_items = report.get("pruned_bands")
    if not isinstance(raw_items, list):
        raise ValueError("pruning report must contain a pruned_bands list")
    encoder = model.encoder
    if not isinstance(encoder, AdaptiveSpatioSpectralTemporalEncoder):
        raise ValueError("operator pruning requires an adaptive encoder")
    axes = {
        "spatial_x": encoder.x_axis,
        "spatial_y": encoder.y_axis,
        "spectral": encoder.spectral_axis,
        "time_mjd": encoder.temporal_axis,
    }
    selected = []
    seen: set[tuple[str, int]] = set()
    for item in raw_items:
        if not isinstance(item, Mapping):
            raise ValueError("each pruned band must be a mapping")
        axis_name = item.get("axis")
        band = item.get("band")
        if axis_name not in axes or not isinstance(band, int) or isinstance(band, bool):
            raise ValueError("each pruned band needs a known axis and integer band")
        axis = axes[axis_name]
        if band < 0 or band >= axis.num_bands:
            raise ValueError(f"pruned band {axis_name}[{band}] is out of range")
        key = (axis_name, band)
        if key in seen:
            raise ValueError(f"duplicate pruned band {axis_name}[{band}]")
        seen.add(key)
        selected.append((axis_name, axis, band))
    with torch.no_grad():
        for _axis_name, axis, band in selected:
            axis.gate_logits[band] = -80.0
    return {
        "schema_version": 1,
        "selection_split": "validation",
        "applied_band_count": len(selected),
        "applied_bands": [
            {"axis": axis_name, "band": band}
            for axis_name, _axis, band in selected
        ],
    }


def fit_operator_calibration_terms(
    terms: Sequence[OperatorCalibrationTerm],
    *,
    steps: int = 200,
    lr: float = 1.0e-3,
    gate_weight: float = 0.0,
    scale_shift_weight: float = 0.0,
) -> dict[str, object]:
    """Fit train-split operator evidence without rendering a science scene."""
    if not terms:
        raise ValueError("at least one operator calibration term is required")
    steps = int(steps)
    lr = float(lr)
    gate_weight = float(gate_weight)
    scale_shift_weight = float(scale_shift_weight)
    if steps < 1:
        raise ValueError("steps must be positive")
    if not math.isfinite(lr) or lr <= 0.0:
        raise ValueError("lr must be finite and positive")
    if (
        not math.isfinite(gate_weight)
        or not math.isfinite(scale_shift_weight)
        or gate_weight < 0.0
        or scale_shift_weight < 0.0
    ):
        raise ValueError("conditioning regularization weights must be finite and non-negative")

    models: list[nn.Module] = []
    seen_models: set[int] = set()
    parameters: list[nn.Parameter] = []
    seen_parameters: set[int] = set()
    devices: set[torch.device] = set()
    for term in terms:
        if id(term.model) not in seen_models:
            models.append(term.model)
            seen_models.add(id(term.model))
        for parameter in term.model.parameters():
            if parameter.requires_grad and id(parameter) not in seen_parameters:
                parameters.append(parameter)
                seen_parameters.add(id(parameter))
                devices.add(parameter.device)
    if not parameters:
        raise ValueError("operator calibration has no trainable parameters")
    if len(devices) != 1:
        raise ValueError("operator calibration models must share one device")

    initial = operator_calibration_score_report(terms, split="train")
    training_states = [model.training for model in models]
    for model in models:
        model.train()
    optimizer = torch.optim.Adam(parameters, lr=lr)
    history: list[float] = []
    try:
        for _ in range(steps):
            optimizer.zero_grad(set_to_none=True)
            loss = torch.stack([term.penalty() for term in terms]).sum()
            if gate_weight or scale_shift_weight:
                penalties = []
                for model in models:
                    regularization = getattr(model, "conditioning_regularization", None)
                    if callable(regularization):
                        penalties.append(
                            regularization(
                                gate_weight=gate_weight,
                                scale_shift_weight=scale_shift_weight,
                            )
                        )
                if penalties:
                    loss = loss + torch.stack(penalties).sum()
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("operator calibration loss became non-finite")
            loss.backward()
            if any(
                parameter.grad is not None
                and not bool(torch.isfinite(parameter.grad).all())
                for parameter in parameters
            ):
                raise FloatingPointError("operator calibration gradient became non-finite")
            optimizer.step()
            history.append(float(loss.detach().cpu()))
    finally:
        for model, training in zip(models, training_states, strict=True):
            model.train(training)

    return {
        "schema_version": 1,
        "steps": steps,
        "learning_rate": lr,
        "trainable_parameter_count": sum(parameter.numel() for parameter in parameters),
        "loss_history": history,
        "initial_train_report": initial,
        "final_train_report": operator_calibration_score_report(terms, split="train"),
    }


def set_field_training_stage(
    scene_model: nn.Module,
    observation_operator_bundles: list[ObservationOperatorBundle]
    | tuple[ObservationOperatorBundle, ...],
    stage: FieldTrainingStage,
) -> dict[str, int | str]:
    """Select one explicit scene/operator optimization stage.

    Call this before each separate :func:`fit_field_model` invocation. Shared
    parameters are trainable if any selected role owns them.
    """
    if stage not in {"scene", "background", "psf", "joint"}:
        raise ValueError("stage must be scene, background, psf, or joint")
    role_modules: dict[str, list[nn.Module]] = {
        "scene": [scene_model],
        "background": [],
        "psf": [],
    }
    seen_models: set[int] = set()
    for bundle in observation_operator_bundles:
        for role, operator in (
            ("psf", bundle.psf),
            ("background", bundle.background),
        ):
            operator_model = getattr(operator, "model", None)
            if (
                isinstance(operator_model, nn.Module)
                and id(operator_model) not in seen_models
            ):
                role_modules[role].append(operator_model)
                seen_models.add(id(operator_model))
    selected = {
        "scene": stage in {"scene", "joint"},
        "background": stage in {"background", "joint"},
        "psf": stage in {"psf", "joint"},
    }
    if stage in {"background", "psf"} and not role_modules[stage]:
        raise ValueError(f"training stage {stage!r} requires a learned {stage} model")
    requested: dict[int, bool] = {}
    parameters: dict[int, nn.Parameter] = {}
    owners: dict[int, set[str]] = {}
    for role, modules in role_modules.items():
        for module in modules:
            for parameter in module.parameters():
                identifier = id(parameter)
                parameters[identifier] = parameter
                owners.setdefault(identifier, set()).add(role)
                requested[identifier] = requested.get(identifier, False) or selected[role]
    for identifier, parameter in parameters.items():
        parameter.requires_grad_(requested[identifier])
    return {
        "stage": stage,
        "scene_trainable_parameters": sum(
            parameter.numel()
            for identifier, parameter in parameters.items()
            if requested[identifier] and "scene" in owners[identifier]
        ),
        "background_trainable_parameters": sum(
            parameter.numel()
            for identifier, parameter in parameters.items()
            if requested[identifier] and "background" in owners[identifier]
        ),
        "psf_trainable_parameters": sum(
            parameter.numel()
            for identifier, parameter in parameters.items()
            if requested[identifier] and "psf" in owners[identifier]
        ),
    }


__all__ = [
    "BackgroundCalibrationTerm",
    "FieldTrainingStage",
    "OperatorCalibrationTerm",
    "PSFCalibrationTerm",
    "fit_operator_calibration_terms",
    "operator_calibration_identifiability_report",
    "operator_calibration_validation_pruning_report",
    "operator_calibration_score_report",
    "registered_image_stamps",
    "set_field_training_stage",
]
