"""Opt-in native source-quadrature rendering and fixed Gaussian evidence.

This route uses caller-supplied source nodes and a calibrated response. It does
not sample the field at detector centers and convolve, infer quadrature rules,
or reuse the legacy normalized bandpass renderer.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import Tensor, nn

from ..evidence import EvidenceSnapshot
from ..likelihood import GaussianLikelihoodFactor
from ..operators.background import (
    AdditiveProcessingState,
    AdditiveStage,
    validate_additive_components,
)
from ..operators.bundle import ObservationOperatorBundle
from ..operators.continuous_response import NativeQuadratureResponseOp
from ..operators.pixel_calibration import PixelCalibrationOp
from ..scenes import DiscreteTargetImage
from ..solvers import BoundObservationOperator
from ..target import TargetRenderSpec
from .forwarding import call_scalar_field_forward
from .gaussian import _revision_signature
from .observation_ops import observation_support_mask
from .types import FieldObservation, ForwardContext


def _identifier(value: str, name: str) -> None:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be a nonempty declaration")


@dataclass(frozen=True, kw_only=True)
class NativeFieldResponse:
    """Fixed source-node response, sky registration and product calibration.

    ``source_ra_deg/dec_deg[N]`` register the quadrature's source nodes on the
    sky. ``geometry_id`` declares their relation to its common response chart;
    detector rows must already be in the image's row-major order. No transform
    or row reordering is inferred or certified. The field's
    monochromatic output has ``source_value_unit``. ``quadrature_output_unit``
    declares the units after source/spectral measures and response factors,
    and must match calibration input units. Labels never trigger conversions.

    Radiance uses ``source_measure_kind='solid_angle_sr'``; point spectral flux
    uses ``'point_flux'`` and source measures one. Density per Angstrom uses
    ``spectral_measure_kind='density_per_angstrom'``; already integrated bins
    use ``'integrated_bins'`` and wavelength measures one. Throughput/QE/area
    belong in the response or its explicit source weights; scalar commuting
    conversions can be in calibration. Include each factor exactly once.

    ``dependencies`` must explicitly declare response parameters and hidden
    state. Geometry/measure tensors and response bindings are tracked here;
    arbitrary callable internals are not inspected. Rebuild after changes.
    Field parameters are evaluated afresh and are not fixed response evidence.
    Signed field values are supported; physical photon/count laws impose their
    own nonnegative support. This is finite quadrature, not an accuracy or
    physical-optics certificate, exposure-time integral or default renderer.
    """

    quadrature: NativeQuadratureResponseOp
    calibration: PixelCalibrationOp
    source_ra_deg: Tensor
    source_dec_deg: Tensor
    source_value_unit: str
    quadrature_output_unit: str
    source_measure_kind: Literal["point_flux", "solid_angle_sr"]
    spectral_measure_kind: Literal["density_per_angstrom", "integrated_bins"]
    band_name: str
    geometry_id: str
    response_id: str
    dependencies: EvidenceSnapshot
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)
    _binding: tuple[object, ...] = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if not isinstance(self.quadrature, NativeQuadratureResponseOp):
            raise TypeError("quadrature must be a NativeQuadratureResponseOp")
        if not isinstance(self.calibration, PixelCalibrationOp):
            raise TypeError("calibration must be a PixelCalibrationOp")
        if not isinstance(self.dependencies, EvidenceSnapshot):
            raise TypeError("dependencies must explicitly declare response evidence")
        self.quadrature._validate()
        reference = self.quadrature.source_xy
        count = reference.shape[0]
        if count == 0:
            raise ValueError("a field response requires at least one source node")
        for name in ("source_ra_deg", "source_dec_deg"):
            value = getattr(self, name)
            if (
                not isinstance(value, Tensor)
                or value.shape != (count,)
                or value.dtype != reference.dtype
                or value.device != reference.device
                or not bool(torch.isfinite(value).all())
            ):
                raise ValueError(
                    f"{name} must be finite [N] with response dtype/device"
                )
        if bool((self.source_dec_deg.abs() > 90).any()):
            raise ValueError("source declination must be in [-90,90] degrees")
        if self.source_measure_kind not in ("point_flux", "solid_angle_sr"):
            raise ValueError("unsupported source_measure_kind")
        if self.spectral_measure_kind not in (
            "density_per_angstrom",
            "integrated_bins",
        ):
            raise ValueError("unsupported spectral_measure_kind")
        if self.source_measure_kind == "point_flux" and not bool(
            (self.quadrature.source_measure == 1).all()
        ):
            raise ValueError("point flux requires source measures one")
        if self.spectral_measure_kind == "integrated_bins" and not bool(
            (self.quadrature.wavelength_measure == 1).all()
        ):
            raise ValueError("integrated spectral bins require wavelength measures one")
        for name in (
            "source_value_unit",
            "quadrature_output_unit",
            "band_name",
            "geometry_id",
            "response_id",
        ):
            _identifier(getattr(self, name), name)
        if self.calibration.input_unit != self.quadrature_output_unit:
            raise ValueError("calibration input units must match quadrature output")
        if (
            self.calibration.scale.shape != (self.quadrature.detector_xy.shape[0],)
            or self.calibration.scale.dtype != reference.dtype
            or self.calibration.scale.device != reference.device
        ):
            raise ValueError(
                "calibration must match detector count and response dtype/device"
            )
        tensors = {
            "source_ra_deg": self.source_ra_deg,
            "source_dec_deg": self.source_dec_deg,
        }
        for name in (
            "source_xy",
            "detector_xy",
            "wavelength_angstrom",
            "source_measure",
            "wavelength_measure",
            "source_spectral_weight",
            "detector_footprint",
            "mask",
            "interaction_pairs",
        ):
            value = getattr(self.quadrature, name)
            if value is not None:
                tensors[f"quadrature.{name}"] = value
        object.__setattr__(self, "_snapshot", self.dependencies.with_tensors(tensors))
        object.__setattr__(self, "_binding", self._bindings())
        self.validate_evidence()

    def _bindings(self) -> tuple[object, ...]:
        return (
            tuple(
                id(getattr(self.quadrature, name))
                for name in (
                    "response",
                    "source_xy",
                    "detector_xy",
                    "wavelength_angstrom",
                    "source_measure",
                    "wavelength_measure",
                    "source_spectral_weight",
                    "detector_footprint",
                    "mask",
                    "interaction_pairs",
                )
            ),
            self.quadrature.interaction_support,
            self.quadrature.source_chunk_size,
            self.quadrature.detector_chunk_size,
            self.quadrature.wavelength_chunk_size,
            self.quadrature.pair_chunk_size,
        )

    def validate_evidence(self) -> None:
        self._snapshot.validate()
        self.calibration.validate_evidence()
        if self._bindings() != self._binding:
            raise RuntimeError(
                "stale evidence response binding; rebuild the native field response"
            )

    @property
    def state_shape(self) -> tuple[int, int]:
        return self.quadrature.source_xy.shape[
            0
        ], self.quadrature.wavelength_angstrom.numel()

    def forward(self, values: Tensor) -> Tensor:
        """Map sampled monochromatic node values [N,L] to product pixels [P]."""
        self.validate_evidence()
        if values.shape != self.state_shape:
            raise ValueError("source values must have shape [N,L]")
        result = self.calibration.forward(self.quadrature.forward(values))
        self.validate_evidence()
        return result

    def adjoint(self, values: Tensor) -> Tensor:
        """Exact transpose into sampled [N,L] coordinates, including measures."""
        self.validate_evidence()
        if values.shape != (self.quadrature.detector_xy.shape[0],):
            raise ValueError("detector values must have shape [P]")
        result = self.quadrature.adjoint(self.calibration.adjoint(values))
        self.validate_evidence()
        return result

    def sample_field(
        self,
        model: nn.Module,
        *,
        forward_context: ForwardContext | None = None,
        source_chunk_size: int = 4096,
    ) -> Tensor:
        """Evaluate source nodes at each scalar wavelength, with no bandpass call.

        Each model call contains at most source_chunk_size coordinates. The
        resulting [N,L] state and autograd graphs are resident; this is not a
        bounded total training-memory claim. Context can supply a scalar epoch
        to a time-aware field, but does not integrate within the exposure.
        """
        self.validate_evidence()
        if self.spectral_measure_kind != "density_per_angstrom":
            raise ValueError(
                "scalar fields supply spectral density; integrated bins require direct sampled states"
            )
        if not isinstance(model, nn.Module):
            raise TypeError("model must implement the nn.Module scalar field interface")
        if (
            isinstance(source_chunk_size, bool)
            or not isinstance(source_chunk_size, int)
            or source_chunk_size < 1
        ):
            raise ValueError("source_chunk_size must be a positive integer")
        columns = []
        for wavelength in self.quadrature.wavelength_angstrom:
            chunks = []
            for start in range(0, self.state_shape[0], source_chunk_size):
                ra = self.source_ra_deg[start : start + source_chunk_size]
                dec = self.source_dec_deg[start : start + source_chunk_size]
                values = call_scalar_field_forward(
                    model, ra, dec, wavelength, forward_context=forward_context
                )
                if (
                    not isinstance(values, Tensor)
                    or values.shape != ra.shape
                    or values.dtype != ra.dtype
                    or values.device != ra.device
                    or not bool(torch.isfinite(values).all())
                ):
                    raise ValueError(
                        "scalar field must return finite source-node values with matching shape/dtype/device"
                    )
                chunks.append(values)
            columns.append(torch.cat(chunks))
        result = torch.stack(columns, dim=-1)
        self.validate_evidence()
        return result


def _validate_observation(
    observation: FieldObservation, response: NativeFieldResponse
) -> None:
    response.validate_evidence()
    if (
        observation.image.numel() != response.quadrature.detector_xy.shape[0]
        or observation.image.dtype != response.calibration.scale.dtype
        or observation.image.device != response.calibration.scale.device
    ):
        raise ValueError(
            "observation must match native detector count, dtype and device"
        )
    if (
        observation.value_unit != response.calibration.output_unit
        or observation.band_name != response.band_name
    ):
        raise ValueError(
            "observation units and band must match the declared native response"
        )
    if (
        observation.psf_spec is not None and observation.psf_spec.mode != "none"
    ) or observation.psf_sigma_pix != 0:
        raise ValueError(
            "native response already owns PSF/pixel integration; a second PSF is invalid"
        )
    if observation.detector_law not in (None, "gaussian"):
        raise ValueError(
            "this calibrated native field route requires a fixed Gaussian law"
        )
    if observation.processing_state != AdditiveProcessingState.CALIBRATED_IMAGE:
        raise ValueError(
            "native field route requires an explicit calibrated_image processing state"
        )
    if any(
        component.stage != AdditiveStage.CALIBRATED_OUTPUT
        for component in observation.additive_components
    ):
        raise ValueError(
            "native field route supports only calibrated_output additive means"
        )
    validate_additive_components(
        observation.additive_components,
        shape=(int(observation.image.shape[0]), int(observation.image.shape[1])),
        value_unit=observation.value_unit,
        value_semantics=observation.value_semantics,
        processing_state=observation.processing_state,
    )


def _output_background(observation: FieldObservation) -> Tensor:
    # Legacy planes retain their documented detector-product residual meaning.
    mean = torch.full_like(observation.image, float(observation.background_offset))
    for name in ("sky_background", "instrument_background", "detector_background"):
        values = getattr(observation, name)
        if values is not None:
            if (
                not isinstance(values, Tensor)
                or values.shape not in (torch.Size([]), observation.image.shape)
                or values.dtype != mean.dtype
                or values.device != mean.device
            ):
                raise ValueError(
                    "output background must be scalar or image-shaped with product dtype/device"
                )
            mean = mean + values
    for component in observation.additive_components:
        if (
            component.values.dtype != mean.dtype
            or component.values.device != mean.device
        ):
            raise ValueError("output additive means must match product dtype/device")
        mean = mean + component.values
    if not bool(torch.isfinite(mean).all()):
        raise ValueError("declared output background must be finite")
    return mean


def render_native_field_observation(
    model: nn.Module,
    observation: FieldObservation,
    response: NativeFieldResponse,
    *,
    forward_context: ForwardContext | None = None,
    source_chunk_size: int = 4096,
) -> Tensor:
    """Opt-in source-node field rendering, calibration and output means.

    Observation bandpass/area/time/gain metadata is provenance, never applied
    again: the declared response/calibration owns those factors. Existing
    render/training defaults are unchanged. This route has no physical raw
    detector likelihood, stochastic additive process or nonlinear history.
    """
    _validate_observation(observation, response)
    values = response.sample_field(
        model, forward_context=forward_context, source_chunk_size=source_chunk_size
    )
    prediction = response.forward(values).reshape(
        observation.image.shape
    ) + _output_background(observation)
    if not bool(torch.isfinite(prediction).all()):
        raise ValueError("combined native prediction must be finite")
    return prediction


def bind_native_field_likelihood(
    observation: FieldObservation,
    response: NativeFieldResponse,
    *,
    measurement_ids: tuple[str, ...],
    provenance_id: str,
    revision: Callable[[], object] | None = None,
) -> GaussianLikelihoodFactor:
    """Bind sampled [N,L] values to conditional native Gaussian evidence.

    No nn.Module or global field is required. Precision is fixed diagonal
    inverse variance in observed product units, restricted to observation
    support intersected with the response mask. Density normalization counts
    that same support. Output means are centered once. Shared uncertain
    calibration/nuisances are NOT inferred here; use explicit joint treatment,
    never independently split their uncertainty or use photon moments as fixed
    precision. UQ conditions on the declared response/calibration.

    Original measurement dependencies and metadata are tracked. Explicit
    caller revisions are required for inference-mode evidence and alias writes.
    """
    _validate_observation(observation, response)
    if (
        observation.weight is None
        or observation.weight.shape != observation.image.shape
        or observation.weight.dtype != observation.image.dtype
        or observation.weight.device != observation.image.device
    ):
        raise ValueError(
            "native Gaussian binding requires image-shaped inverse-variance weights"
        )
    tensors = {"image": observation.image, "weight": observation.weight}
    for name in (
        "mask",
        "sky_background",
        "instrument_background",
        "detector_background",
        "transmission_wavelength",
        "transmission_throughput",
        "gain_electrons_per_adu",
    ):
        value = getattr(observation, name)
        if isinstance(value, Tensor):
            tensors[name] = value
    for index, component in enumerate(observation.additive_components):
        tensors[f"output_mean[{index}]"] = component.values
    if revision is None and (
        torch.is_inference_mode_enabled()
        or any(t.is_inference() for t in tensors.values())
    ):
        raise ValueError(
            "inference-mode evidence requires an explicit caller revision getter"
        )

    def current_revision() -> object:
        response.validate_evidence()
        return (
            _revision_signature(observation),
            observation.exposure_time_s,
            observation.collecting_area_m2,
            observation.gain_provenance_id,
            id(observation.gain_electrons_per_adu),
            None if revision is None else revision(),
        )

    snapshot = EvidenceSnapshot.capture(tensors, revision=current_revision)
    support = observation_support_mask(observation).reshape(-1)
    if response.quadrature.mask is not None:
        support = support & response.quadrature.mask
    data = torch.where(
        support, (observation.image - _output_background(observation)).reshape(-1), 0.0
    )
    weight = torch.where(support, observation.weight.reshape(-1), 0.0)
    normalization = 0.5 * (
        int(support.sum().item()) * math.log(2 * math.pi) - weight[support].log().sum()
    )

    def forward(scene, target) -> Tensor:
        del target
        snapshot.validate()
        if not isinstance(scene, DiscreteTargetImage):
            raise TypeError(
                "native linear likelihood expects sampled [N,L] tensor values"
            )
        result = response.forward(scene.data)
        snapshot.validate()
        return result

    def adjoint(value: Tensor) -> Tensor:
        snapshot.validate()
        result = response.adjoint(value)
        snapshot.validate()
        return result

    def precision(value: Tensor) -> Tensor:
        snapshot.validate()
        return weight * value

    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(
            header=dict(observation.header),
            shape=(int(observation.image.shape[0]), int(observation.image.shape[1])),
            band_label=response.band_name,
        ),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=precision,
    )
    return GaussianLikelihoodFactor(
        operator,
        data,
        measurement_ids,
        response.calibration.calibration_id,
        provenance_id,
        log_normalization=normalization,
        dependencies=snapshot,
    )


__all__ = [
    "NativeFieldResponse",
    "render_native_field_observation",
    "bind_native_field_likelihood",
]
