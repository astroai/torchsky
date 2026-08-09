"""Calibration losses and explicit training stages for learned operators."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal, Protocol

import torch
from torch import Tensor, nn

from .background import ConditionedBackgroundField
from .bundle import ObservationOperatorBundle
from .learned_kernel import ConditionedKernelPSF

FieldTrainingStage = Literal["scene", "background", "psf", "joint"]


class OperatorCalibrationTerm(Protocol):
    """A differentiable, evidence-space calibration penalty."""

    model: nn.Module

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


@dataclass
class PSFCalibrationTerm:
    """Anchor a conditioned PSF to measured normalized effective kernels."""

    model: ConditionedKernelPSF
    detector_x: Tensor
    detector_y: Tensor
    wavelength: Tensor
    time_mjd: Tensor
    target_kernels: Tensor
    sample_weight: Tensor | None = None
    loss_weight: float = 1.0
    name: str = "psf"

    def __post_init__(self) -> None:
        if not isinstance(self.model, ConditionedKernelPSF):
            raise TypeError("PSFCalibrationTerm requires ConditionedKernelPSF")
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
        sums = kernels.sum(dim=(-2, -1))
        if bool((sums <= 0.0).any()):
            raise ValueError("each target PSF kernel must have positive total flux")
        self.target_kernels = kernels / sums[:, None, None]
        self.sample_weight = _sample_weight(self.sample_weight, length=count)
        self.loss_weight = _validated_loss_weight(self.loss_weight)

    def predicted_kernels(self) -> Tensor:
        parameter = next(self.model.parameters())
        coordinates = [
            self.detector_x.to(device=parameter.device, dtype=parameter.dtype),
            self.detector_y.to(device=parameter.device, dtype=parameter.dtype),
            self.wavelength.to(device=parameter.device),
            self.time_mjd.to(device=parameter.device),
        ]
        mixture = self.model.mixture_weights(*coordinates).movedim(0, -1)
        basis = self.model.kernel_basis.basis_kernels().to(
            device=parameter.device, dtype=parameter.dtype
        )
        return torch.einsum("nb,bhw->nhw", mixture, basis)

    def raw_loss(self) -> Tensor:
        predicted = self.predicted_kernels()
        target = self.target_kernels.to(
            device=predicted.device, dtype=predicted.dtype
        )
        per_sample = torch.square(predicted - target).mean(dim=(-2, -1))
        weights = self.sample_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        return (per_sample * weights).sum() / weights.sum()

    def penalty(self) -> Tensor:
        return self.loss_weight * self.raw_loss()

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "kind": "normalized_effective_psf_mse",
            "sample_count": int(self.detector_x.numel()),
            "loss_weight": self.loss_weight,
            "raw_loss": float(self.raw_loss().detach().cpu()),
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
        weights = self.sample_weight.to(
            device=predicted.device, dtype=predicted.dtype
        )
        return (torch.square(predicted - target) * weights).sum() / weights.sum()

    def penalty(self) -> Tensor:
        return self.loss_weight * self.raw_loss()

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "name": self.name,
            "kind": "background_value_mse",
            "sample_count": int(self.detector_x.numel()),
            "loss_weight": self.loss_weight,
            "raw_loss": float(self.raw_loss().detach().cpu()),
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
    "set_field_training_stage",
]
