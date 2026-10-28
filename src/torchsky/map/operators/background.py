"""Background operators shared across mapmaking and field rendering."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from enum import Enum
from typing import Literal, cast

import torch
from torch import Tensor, nn

from .adaptive_axes import AdaptiveSpatioSpectralTemporalEncoder
from .base import BackgroundOp
from .learned_kernel import SpatioSpectralTemporalEncoder


class AdditiveStage(str, Enum):
    """Placement in the response graph, including currently unsupported physics."""

    SAMPLED_RESPONSE_INPUT = "sampled_response_input"
    CALIBRATED_OUTPUT = "calibrated_output"
    PHOTON_ILLUMINATION = "photon_illumination"
    DARK_CHARGE = "dark_charge"
    READ_OFFSET = "read_offset"


class AdditiveFrame(str, Enum):
    SAMPLED_SKY = "sampled_sky"
    DETECTOR = "detector"


class AdditiveProcessingState(str, Enum):
    CALIBRATED_IMAGE = "calibrated_image"
    RAW_READS = "raw_reads"


@dataclass(frozen=True)
class AdditiveComponent:
    """A signed mean plane with explicit placement and radiometric convention.

    ``sampled_response_input`` is already sampled/integrated on the observation's
    native grid in the scene's band-integrated units. It receives the existing
    discrete, achromatic PSF, not a continuous photon/pixel response. No spectral,
    area, time, gain or flat conversion is inferred. ``calibrated_output`` is a
    residual in the observed detector product's units and bypasses that PSF.
    Both are deterministic mean terms: neither adds shot noise or read covariance.
    Physical illumination/charge/read stages are reserved and fail closed at use.
    """

    values: Tensor
    stage: AdditiveStage
    coordinate_frame: AdditiveFrame
    value_unit: str
    value_semantics: str
    processing_state: AdditiveProcessingState
    stochastic_contribution: Literal[
        "deterministic_mean", "photon_shot", "charge_shot", "read_process"
    ] = "deterministic_mean"

    def __post_init__(self) -> None:
        if not isinstance(self.values, Tensor) or self.values.ndim != 2:
            raise ValueError("additive values must be a 2D tensor")
        if not self.values.is_floating_point():
            raise TypeError("additive values must be floating point")
        if not bool(torch.isfinite(self.values).all()):
            raise ValueError("additive values must be finite")
        for name, enum_type in (
            ("stage", AdditiveStage),
            ("coordinate_frame", AdditiveFrame),
            ("processing_state", AdditiveProcessingState),
        ):
            object.__setattr__(self, name, enum_type(getattr(self, name)))
        for name in ("value_unit", "value_semantics"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"additive {name} must be a nonempty string")
        if self.stochastic_contribution not in {
            "deterministic_mean",
            "photon_shot",
            "charge_shot",
            "read_process",
        }:
            raise ValueError("unsupported additive stochastic_contribution")

    def to(
        self,
        device: torch.device | str | None = None,
        dtype: torch.dtype | None = None,
    ) -> AdditiveComponent:
        """Move the plane without detaching its gradient graph."""
        return replace(self, values=self.values.to(device=device, dtype=dtype))


def validate_additive_components(
    components: tuple[AdditiveComponent, ...],
    *,
    shape: tuple[int, int],
    value_unit: str | None,
    value_semantics: str | None,
    processing_state: AdditiveProcessingState | None,
) -> None:
    """Validate the bounded calibrated-image execution contract."""
    for component in components:
        if not isinstance(component, AdditiveComponent):
            raise TypeError(
                "additive_components must contain AdditiveComponent objects"
            )
        if component.stage not in {
            AdditiveStage.SAMPLED_RESPONSE_INPUT,
            AdditiveStage.CALIBRATED_OUTPUT,
        }:
            raise ValueError(
                f"unsupported physical additive stage {component.stage.value!r}"
            )
        expected_frame = (
            AdditiveFrame.SAMPLED_SKY
            if component.stage == AdditiveStage.SAMPLED_RESPONSE_INPUT
            else AdditiveFrame.DETECTOR
        )
        if component.coordinate_frame != expected_frame:
            raise ValueError("additive coordinate_frame does not match stage")
        if (
            processing_state != AdditiveProcessingState.CALIBRATED_IMAGE
            or component.processing_state != processing_state
        ):
            raise ValueError(
                "additive execution requires matching calibrated_image processing_state"
            )
        if (
            component.value_unit != value_unit
            or component.value_semantics != value_semantics
        ):
            raise ValueError(
                "additive units and value_semantics must match observation"
            )
        if component.stochastic_contribution != "deterministic_mean":
            raise ValueError("additive stochastic physics is not implemented")
        if tuple(component.values.shape) != shape:
            raise ValueError("additive values shape must match observation")
        if not bool(torch.isfinite(component.values).all()):
            raise ValueError("additive values must be finite")


def apply_additive_stage(
    image: Tensor,
    components: tuple[AdditiveComponent, ...],
    stage: AdditiveStage,
) -> Tensor:
    """Add validated native-grid means; preserve dtype, device and gradients."""
    for component in components:
        if component.stage == stage:
            image = image + component.values.to(device=image.device, dtype=image.dtype)
    return image


def apply_calibrated_additive_response(
    image: Tensor,
    components: tuple[AdditiveComponent, ...],
    *,
    response: Callable[[Tensor], Tensor],
    value_unit: str,
    value_semantics: str,
    processing_state: AdditiveProcessingState,
) -> Tensor:
    """Compose calibrated native-grid means around a discrete linear response.

    This exposure-level primitive needs no global field. The caller supplies an
    achromatic, unit-preserving response on the same native grid, for example
    ``lambda plane: apply_psf_spec(plane, psf_spec)`` with an explicit PSF. It must
    not stand in for photon collection, dark accumulation or raw readout. Mean
    terms carry no additional noise or nuisance covariance.
    """
    if image.ndim != 2 or not image.is_floating_point():
        raise ValueError("calibrated additive response requires a floating 2D image")
    validate_additive_components(
        components,
        shape=(int(image.shape[0]), int(image.shape[1])),
        value_unit=value_unit,
        value_semantics=value_semantics,
        processing_state=processing_state,
    )
    output = response(
        apply_additive_stage(image, components, AdditiveStage.SAMPLED_RESPONSE_INPUT)
    )
    if (
        output.shape != image.shape
        or output.dtype != image.dtype
        or output.device != image.device
    ):
        raise ValueError(
            "calibrated additive response must preserve shape, dtype and device"
        )
    return apply_additive_stage(output, components, AdditiveStage.CALIBRATED_OUTPUT)


class ConditionedBackgroundField(nn.Module):
    """Signed additive background with independently scaled conditioning axes."""

    def __init__(
        self,
        encoder: SpatioSpectralTemporalEncoder
        | AdaptiveSpatioSpectralTemporalEncoder,
        *,
        hidden_dim: int = 32,
        hidden_layers: int = 2,
    ) -> None:
        super().__init__()
        if hidden_dim < 1:
            raise ValueError("hidden_dim must be positive")
        if hidden_layers < 1:
            raise ValueError("hidden_layers must be positive")
        layers: list[nn.Module] = []
        width = encoder.output_dim
        for _ in range(int(hidden_layers)):
            layers.extend((nn.Linear(width, int(hidden_dim)), nn.GELU()))
            width = int(hidden_dim)
        output = nn.Linear(width, 1)
        nn.init.zeros_(output.weight)
        nn.init.zeros_(output.bias)
        layers.append(output)
        self.encoder = encoder
        self.network = nn.Sequential(*layers)

    def forward(
        self,
        spatial_x: Tensor,
        spatial_y: Tensor,
        wavelength: Tensor | float,
        time_mjd: Tensor | float | None = None,
    ) -> Tensor:
        encoded = self.encoder(spatial_x, spatial_y, wavelength, time_mjd)
        return self.network(encoded).squeeze(-1)

    def conditioning_regularization(
        self,
        *,
        gate_weight: float = 1.0,
        scale_shift_weight: float = 1.0,
    ) -> Tensor:
        regularization = getattr(self.encoder, "regularization", None)
        if not callable(regularization):
            return torch.sum(cast(Tensor, self.network[-1].weight)) * 0.0
        return cast(Callable[..., Tensor], regularization)(
            gate_weight=gate_weight,
            scale_shift_weight=scale_shift_weight,
        )


@dataclass
class ExplicitBackgroundOp(BackgroundOp):
    def __init__(self, *, name: str = "explicit_background") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedBackgroundOp(BackgroundOp):
    model: nn.Module | None = None
    coordinate_frame: str = "sky"

    def __init__(
        self,
        model: nn.Module | None = None,
        *,
        coordinate_frame: str = "sky",
        name: str = "learned_background",
    ) -> None:
        if coordinate_frame not in {"sky", "detector"}:
            raise ValueError("coordinate_frame must be 'sky' or 'detector'")
        super().__init__(name=name, learned=True)
        self.model = model
        self.coordinate_frame = coordinate_frame


@dataclass
class SkyBackgroundOp(ExplicitBackgroundOp):
    def __init__(self) -> None:
        super().__init__(name="sky_background")


@dataclass
class InstrumentBackgroundOp(ExplicitBackgroundOp):
    def __init__(self) -> None:
        super().__init__(name="instrument_background")


@dataclass
class DetectorBackgroundOp(ExplicitBackgroundOp):
    def __init__(self) -> None:
        super().__init__(name="detector_background")


@dataclass
class SpatialSkyBackgroundOp(ExplicitBackgroundOp):
    """Background operator wrapping a spatially-varying sky model estimator.

    The ``background_estimator`` callable is evaluated on each calibrated
    observation during ``prepare_image_observation_data``.  It receives
    ``(image, mask)`` and must return a sky-background tensor of the same
    shape.  The result is subtracted from the calibrated image before
    geometry projection.

    This op replaces the standalone ``background_estimator`` parameter
    formerly handled inline in ``coadd_image_plane``.
    """

    background_estimator: Callable[[Tensor, Tensor | None], Tensor] | None = None

    def __init__(
        self,
        background_estimator: Callable[[Tensor, Tensor | None], Tensor],
        *,
        name: str = "spatial_sky_background",
    ) -> None:
        super().__init__(name=name)
        self.background_estimator = background_estimator


@dataclass(frozen=True)
class BackgroundFitSummary:
    model: Tensor
    coeffs: Tensor
    model_rms: float
    detrended_rms: float
    residual_bias: float
    leakage_fraction: float


def constant_background(
    shape: tuple[int, int],
    *,
    value: float,
    device: torch.device,
    dtype: torch.dtype = torch.float32,
) -> Tensor:
    if value == 0.0:
        return torch.zeros(shape, dtype=dtype, device=device)
    return torch.full(shape, float(value), dtype=dtype, device=device)


def polynomial_background(
    shape: tuple[int, int],
    *,
    coeffs: Tensor,
    device: torch.device,
    dtype: torch.dtype | None = None,
    normalized_coordinates: bool = True,
) -> Tensor:
    """Evaluate a 2D polynomial background in increasing total-degree order."""
    dtype = dtype or coeffs.dtype
    height, width = int(shape[0]), int(shape[1])
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=dtype, device=device),
        torch.arange(width, dtype=dtype, device=device),
        indexing="ij",
    )
    if normalized_coordinates:
        xx = (xx / max(float(width - 1), 1.0)) * 2.0 - 1.0
        yy = (yy / max(float(height - 1), 1.0)) * 2.0 - 1.0
    coeffs = coeffs.to(device=device, dtype=dtype).reshape(-1)
    terms: list[Tensor] = []
    degree = 0
    while len(terms) < int(coeffs.numel()):
        for y_power in range(degree + 1):
            x_power = degree - y_power
            terms.append((xx**x_power) * (yy**y_power))
            if len(terms) == int(coeffs.numel()):
                break
        degree += 1
    out = torch.zeros(shape, dtype=dtype, device=device)
    for coeff, term in zip(coeffs, terms, strict=True):
        out = out + coeff * term
    return out


def apply_background_components(
    image: Tensor,
    *,
    sky: Tensor | None = None,
    instrument: Tensor | None = None,
    detector: Tensor | None = None,
) -> Tensor:
    out = image
    for component in (sky, instrument, detector):
        if component is not None:
            out = out + component.to(device=image.device, dtype=image.dtype)
    return out


def _coerce_background_component(
    component: Tensor | None,
    *,
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
    name: str,
) -> Tensor | None:
    if component is None:
        return None
    tensor = component.to(device=device, dtype=dtype)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"{name} background shape must match rendered image shape")
    return tensor


def observation_background_components(
    observation: object,
    *,
    shape: tuple[int, int],
    device: torch.device,
    dtype: torch.dtype,
) -> dict[str, Tensor | None]:
    """Normalize observation background metadata into sky/instrument/detector tensors."""
    sky = _coerce_background_component(
        getattr(observation, "sky_background", None),
        shape=shape,
        device=device,
        dtype=dtype,
        name="sky",
    )
    instrument = _coerce_background_component(
        getattr(observation, "instrument_background", None),
        shape=shape,
        device=device,
        dtype=dtype,
        name="instrument",
    )
    detector = _coerce_background_component(
        getattr(observation, "detector_background", None),
        shape=shape,
        device=device,
        dtype=dtype,
        name="detector",
    )
    offset = float(getattr(observation, "background_offset", 0.0))
    if offset != 0.0:
        offset_background = constant_background(
            shape,
            value=offset,
            device=device,
            dtype=dtype,
        )
        detector = (
            offset_background if detector is None else detector + offset_background
        )
    return {"sky": sky, "instrument": instrument, "detector": detector}


def fit_background_plane(
    residual: Tensor,
    mask: Tensor,
    *,
    weight: Tensor | None = None,
    sigma_clip: float | None = None,
    max_iterations: int = 3,
) -> BackgroundFitSummary:
    """Fit an optionally weighted, sigma-clipped plane to supported residuals."""
    residual = residual.to(dtype=torch.float32)
    mask = mask.to(device=residual.device, dtype=torch.bool)
    if residual.ndim != 2 or mask.shape != residual.shape:
        raise ValueError("fit_background_plane expects 2D residual and matching mask")
    if sigma_clip is not None and sigma_clip <= 0:
        raise ValueError("sigma_clip must be positive")
    if max_iterations < 0:
        raise ValueError("max_iterations must be non-negative")
    if weight is None:
        weight = torch.ones_like(residual)
    else:
        weight = weight.to(device=residual.device, dtype=torch.float32)
        if weight.shape != residual.shape:
            raise ValueError("background weight must match residual shape")
    mask = (
        mask
        & torch.isfinite(residual)
        & torch.isfinite(weight)
        & (weight > 0)
    )
    if not bool(mask.any()):
        zeros = torch.zeros_like(residual)
        return BackgroundFitSummary(
            model=zeros,
            coeffs=torch.zeros(3, dtype=torch.float32, device=residual.device),
            model_rms=0.0,
            detrended_rms=0.0,
            residual_bias=0.0,
            leakage_fraction=0.0,
        )
    yy, xx = torch.meshgrid(
        torch.arange(residual.shape[0], dtype=torch.float32, device=residual.device),
        torch.arange(residual.shape[1], dtype=torch.float32, device=residual.device),
        indexing="ij",
    )
    support = mask
    for iteration in range(max_iterations + 1):
        design = torch.stack(
            [
                xx[support].to(dtype=torch.float64),
                yy[support].to(dtype=torch.float64),
                torch.ones_like(xx[support], dtype=torch.float64),
            ],
            dim=1,
        )
        response = residual[support].to(dtype=torch.float64)
        sqrt_weight = torch.sqrt(weight[support].to(dtype=torch.float64))
        coeffs, *_ = torch.linalg.lstsq(
            design * sqrt_weight[:, None],
            (response * sqrt_weight).unsqueeze(1),
        )
        coeffs = coeffs[:, 0].to(dtype=torch.float32)
        model = coeffs[0] * xx + coeffs[1] * yy + coeffs[2]
        if sigma_clip is None or iteration == max_iterations:
            break
        detrended = residual - model
        center = torch.median(detrended[support])
        robust_sigma = 1.4826 * torch.median(
            torch.abs(detrended[support] - center)
        )
        tolerance = max(
            float(sigma_clip * robust_sigma),
            32.0 * torch.finfo(residual.dtype).eps,
        )
        clipped = mask & (torch.abs(detrended - center) <= tolerance)
        if int(clipped.sum()) < 3 or torch.equal(clipped, support):
            break
        support = clipped
    supported_model = model[support]
    supported_residual = residual[support]
    detrended = supported_residual - supported_model
    residual_rms = torch.sqrt(torch.mean(torch.square(supported_residual)))
    model_rms = torch.sqrt(torch.mean(torch.square(supported_model)))
    detrended_rms = torch.sqrt(torch.mean(torch.square(detrended)))
    return BackgroundFitSummary(
        model=model,
        coeffs=coeffs,
        model_rms=float(model_rms.item()),
        detrended_rms=float(detrended_rms.item()),
        residual_bias=float(torch.mean(supported_residual).item()),
        leakage_fraction=float(
            (model_rms / torch.clamp(residual_rms, min=1.0e-8)).item()
        ),
    )
