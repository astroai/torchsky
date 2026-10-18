"""Opt-in single-integrated-read native photon/dark/read likelihoods."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
import math
from typing import Literal

import torch
from torch import Tensor, nn
from torch.utils.checkpoint import checkpoint

from ..evidence import EvidenceSnapshot
from ..likelihood import _identifiers
from ..operators.background import (
    AdditiveComponent,
    AdditiveFrame,
    AdditiveProcessingState,
    AdditiveStage,
)
from ..operators.detector_response import LinearDetectorResponse
from ..operators.noise import poisson_gaussian_nll
from ..operators.pixel_calibration import PixelCalibrationOp
from .gaussian import _revision_signature
from .native_response import NativeFieldResponse
from .types import FieldObservation, ForwardContext


_LEGACY_PLANES = (
    "weight",
    "err",
    "var_poisson",
    "var_rnoise",
    "var_flat",
    "contamination_prior",
    "amplifier_index",
    "sky_background",
    "instrument_background",
    "detector_background",
)


def _matching(value: Tensor, reference: Tensor, name: str) -> None:
    if not isinstance(value, Tensor):
        raise TypeError(f"{name} must be a tensor")
    if value.dtype != reference.dtype or value.device != reference.device:
        raise ValueError(f"{name} must match observed dtype and device")


def _pixel_parameter(value: Tensor, reference: Tensor, name: str) -> Tensor:
    _matching(value, reference, name)
    if value.ndim != 0 and value.shape != (reference.numel(),):
        raise ValueError(f"{name} must be scalar or explicit [P]")
    return value.expand(reference.numel())


@dataclass(frozen=True, kw_only=True)
class NativePhysicalLikelihood:
    """Conditional normalized finite-mixture law for one integrated read.

    Bind with ``bind_native_physical_likelihood``. This route reuses an empty-
    additive ``FieldObservation`` with ``RAW_READS`` processing and the explicit
    ``single_integrated_read`` value semantics. Here that processing label means
    one continuous output per native pixel, not a nondestructive ramp, event
    stream, nonlinear/history response, resampled image or arbitrary pipeline
    residual. Fixed diagonal conversion and deterministic output offsets are
    declared explicitly, with their density Jacobian supplied by evaluating in
    observed units. ``CALIBRATED_OUTPUT`` components are allowed only in this
    route-local raw-read scope; legacy additive validators remain unchanged.

    The native photon response maps source spectral values to photon/s in
    already integrated detector pixels. Its calibration input/output and
    quadrature output must be photon/s, with the declared physical convention
    ``photon_routing_qe_excluded``. This is a caller contract for independent
    Poisson routing/thinning, including the fixed scale; labels cannot prove
    positivity/substochasticity of a callback or calibrate a sky-flat into a
    physical photon response. Remaining QE is an explicit [0,1] probability,
    with one electron at most per detected photon. Photon/charge/read mixing
    maps are not inferred from moments. Exposure rates are constant over the
    explicitly supplied positive duration; this does not integrate time history.

    External typed components use RAW_READS processing. PHOTON_ILLUMINATION is
    [N,L] nonnegative sampled-sky spectral density, in source_value_unit with
    value_semantics='density_per_angstrom' and photon_shot. DARK_CHARGE is [H,W]
    nonnegative detector electron/s, semantics='electron_rate' and charge_shot.
    READ_OFFSET is [H,W] signed detector electrons, semantics='electron_offset'
    and deterministic_mean. CALIBRATED_OUTPUT is [H,W] signed detector output
    units, semantics='single_integrated_read' and deterministic_mean. Physical
    components are passed separately; they are not stored on FieldObservation.

    Independent dark rate is added after optics, before readout. Signed read
    and output offsets change conditional means, never erase photon/dark shot
    noise. Read variance is electron^2, strictly positive on kept support. Zero
    read variance is a discrete law rather than this continuous density. Output
    calibration maps electron to observed units with no inferred gain/area/QE.

    ``sum`` is the normalized independent-pixel finite-mixture NLL; ``mean``
    is a reporting objective. The law, detached support and conservative first/
    second derivative-domain limits are exactly those of poisson_gaussian_nll.
    It is not the full untruncated physical law, a Gaussian b/H/c compression,
    nuisance marginalization, Fisher matrix or sampling-covariance guarantee.
    All rates/offsets/calibration are conditional fixed evidence. Original
    tensors, metadata and response/calibration are guarded before/after calls;
    inference tensors and alias/hidden state require explicit caller revisions.
    Rebind after evidence changes and for a new fixed-evidence autograd graph.
    By default NLL evaluates all P kept rows together, with O(P*K) mixture
    intermediates where K can reach max_terms. Opt-in mixture_chunk_size bounds
    each mixture call and checkpoints its reverse-AD intermediates. Source and
    detector response graphs remain resident: this is not total bounded-memory
    survey training. Checkpointed callbacks must be deterministic and evidence
    unchanged through backward; the standard per-call evidence guards apply.
    """

    observation: FieldObservation
    photon_response: NativeFieldResponse
    output_calibration: PixelCalibrationOp
    exposure_time_s: Tensor
    electrons_per_photon: Tensor
    read_variance_electrons2: Tensor
    measurement_ids: tuple[str, ...]
    calibration_id: str
    provenance_id: str
    additive_components: tuple[AdditiveComponent, ...] = ()
    dependencies: EvidenceSnapshot | None = None
    revision: Callable[[], object] | None = None
    tail_sigma: float = 12.0
    max_terms: int = 4096
    _snapshot: EvidenceSnapshot = field(init=False, repr=False, compare=False)
    _support: Tensor = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        observation, response, calibration = (
            self.observation,
            self.photon_response,
            self.output_calibration,
        )
        if (
            not isinstance(observation, FieldObservation)
            or not isinstance(response, NativeFieldResponse)
            or not isinstance(calibration, PixelCalibrationOp)
        ):
            raise TypeError(
                "binding requires FieldObservation, NativeFieldResponse and PixelCalibrationOp"
            )
        response.validate_evidence()
        calibration.validate_evidence()
        image = observation.image
        if (
            not isinstance(image, Tensor)
            or image.ndim != 2
            or image.dtype not in (torch.float32, torch.float64)
            or image.numel() == 0
        ):
            raise ValueError("observed image must be nonempty 2D real float32/float64")
        if (
            observation.processing_state != AdditiveProcessingState.RAW_READS
            or observation.value_semantics != "single_integrated_read"
            or observation.detector_law != "poisson_gaussian"
        ):
            raise ValueError(
                "physical binding requires raw_reads, single_integrated_read and poisson_gaussian"
            )
        if (
            observation.additive_components
            or observation.background_offset != 0
            or any(getattr(observation, name) is not None for name in _LEGACY_PLANES)
        ):
            raise ValueError(
                "physical binding rejects legacy additive/background/noise planes and weights"
            )
        if (
            observation.exposure_time_s is not None
            or observation.gain_electrons_per_adu is not None
            or observation.gain_provenance_id is not None
        ):
            raise ValueError(
                "exposure/gain conversion must be explicit only in this binding"
            )
        if observation.psf_sigma_pix != 0 or (
            observation.psf_spec is not None and observation.psf_spec.mode != "none"
        ):
            raise ValueError(
                "native photon response already owns PSF/pixel integration"
            )
        if (
            observation.band_name != response.band_name
            or observation.value_unit != calibration.output_unit
            or calibration.input_unit != "electron"
        ):
            raise ValueError("band and electron/output calibration units must match")
        photon_cal = response.calibration
        if (
            photon_cal.input_unit != "photon/s"
            or photon_cal.output_unit != "photon/s"
            or response.quadrature_output_unit != "photon/s"
            or photon_cal.flat_convention != "photon_routing_qe_excluded"
        ):
            raise ValueError(
                "photon response must declare physical photon/s routing with QE excluded"
            )
        if response.spectral_measure_kind != "density_per_angstrom":
            raise ValueError(
                "physical field binding requires source density per Angstrom"
            )
        _matching(response.quadrature.source_xy, image, "photon response")
        _matching(calibration.scale, image, "output calibration")
        if response.quadrature.detector_xy.shape[
            0
        ] != image.numel() or calibration.scale.shape != (image.numel(),):
            raise ValueError(
                "detector rows and calibration must match native image pixels"
            )
        _matching(self.exposure_time_s, image, "exposure_time_s")
        if (
            self.exposure_time_s.ndim != 0
            or not bool(torch.isfinite(self.exposure_time_s))
            or not bool(self.exposure_time_s > 0)
        ):
            raise ValueError("exposure_time_s must be finite positive scalar seconds")
        support = torch.ones(image.numel(), dtype=torch.bool, device=image.device)
        if observation.mask is not None:
            if (
                observation.mask.dtype != torch.bool
                or observation.mask.device != image.device
                or observation.mask.shape != image.shape
            ):
                raise ValueError(
                    "observation mask must be bool with observed shape/device"
                )
            support = support & observation.mask.reshape(-1)
        if response.quadrature.mask is not None:
            support = support & response.quadrature.mask
        if not bool(support.any()):
            raise ValueError(
                "physical likelihood requires at least one kept measurement"
            )
        if not bool(torch.isfinite(image.reshape(-1)[support]).all()):
            raise ValueError(
                "kept observations must be finite (signed reads are allowed)"
            )
        object.__setattr__(self, "_support", support)
        qe = _pixel_parameter(self.electrons_per_photon, image, "electrons_per_photon")[
            support
        ]
        read_var = _pixel_parameter(
            self.read_variance_electrons2, image, "read_variance_electrons2"
        )[support]
        if not bool(torch.isfinite(qe).all()) or bool(((qe < 0) | (qe > 1)).any()):
            raise ValueError(
                "kept electrons_per_photon must be finite probabilities in [0,1]"
            )
        if not bool(torch.isfinite(read_var).all()) or bool((read_var <= 0).any()):
            raise ValueError("kept read variance must be finite and strictly positive")
        _identifiers(self.measurement_ids, name="measurement_ids", required=True)
        for name in ("calibration_id", "provenance_id"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"{name} must be a nonempty identifier")
        if (
            not math.isfinite(self.tail_sigma)
            or self.tail_sigma <= 0
            or isinstance(self.max_terms, bool)
            or not isinstance(self.max_terms, int)
            or self.max_terms < 2
        ):
            raise ValueError(
                "tail_sigma must be finite positive and max_terms an integer >=2"
            )
        if not isinstance(self.additive_components, tuple):
            raise TypeError("physical additive_components must be a tuple")
        tensors = {
            "observed": image,
            "duration": self.exposure_time_s,
            "qe": self.electrons_per_photon,
            "read_variance": self.read_variance_electrons2,
        }
        if observation.mask is not None:
            tensors["observation_mask"] = observation.mask
        for index, component in enumerate(self.additive_components):
            self._validate_component(component)
            tensors[f"physical_component[{index}]"] = component.values
        if self.dependencies is not None and not isinstance(
            self.dependencies, EvidenceSnapshot
        ):
            raise TypeError("dependencies must be an EvidenceSnapshot")
        if self.revision is not None and not callable(self.revision):
            raise TypeError("revision must be callable")
        if self.revision is None and (
            torch.is_inference_mode_enabled()
            or any(value.is_inference() for value in tensors.values())
        ):
            raise ValueError(
                "inference physical evidence requires an explicit caller revision"
            )

        def current_revision() -> object:
            response.validate_evidence()
            calibration.validate_evidence()
            if self.dependencies is not None:
                self.dependencies.validate()
            return (
                _revision_signature(observation),
                tuple(
                    (name, id(getattr(observation, name))) for name in _LEGACY_PLANES
                ),
                observation.exposure_time_s,
                observation.collecting_area_m2,
                observation.psf_sigma_pix,
                id(observation.gain_electrons_per_adu),
                observation.gain_provenance_id,
                None if self.revision is None else self.revision(),
            )

        object.__setattr__(
            self,
            "_snapshot",
            EvidenceSnapshot.capture(tensors, revision=current_revision),
        )

    def _validate_component(self, component: AdditiveComponent) -> None:
        if not isinstance(component, AdditiveComponent):
            raise TypeError("physical components must be AdditiveComponent values")
        _matching(component.values, self.observation.image, "physical component")
        if component.processing_state != AdditiveProcessingState.RAW_READS:
            raise ValueError("physical components require raw_reads processing")
        source = component.stage == AdditiveStage.PHOTON_ILLUMINATION
        if source:
            contract = (
                AdditiveFrame.SAMPLED_SKY,
                self.photon_response.source_value_unit,
                "density_per_angstrom",
                "photon_shot",
            )
            shape = self.photon_response.state_shape
            values = component.values
        else:
            contracts = {
                AdditiveStage.DARK_CHARGE: (
                    "electron/s",
                    "electron_rate",
                    "charge_shot",
                ),
                AdditiveStage.READ_OFFSET: (
                    "electron",
                    "electron_offset",
                    "deterministic_mean",
                ),
                AdditiveStage.CALIBRATED_OUTPUT: (
                    self.observation.value_unit,
                    "single_integrated_read",
                    "deterministic_mean",
                ),
            }
            if component.stage not in contracts:
                raise ValueError("unsupported physical additive stage")
            unit, semantics, stochastic = contracts[component.stage]
            contract = (AdditiveFrame.DETECTOR, unit, semantics, stochastic)
            shape = self.observation.image.shape
            if component.values.shape != shape:
                raise ValueError(
                    "physical additive shape must match native detector image"
                )
            values = component.values.reshape(-1)[self._support]
        actual = (
            component.coordinate_frame,
            component.value_unit,
            component.value_semantics,
            component.stochastic_contribution,
        )
        if component.values.shape != shape or actual != contract:
            raise ValueError(
                "physical additive shape/frame/units/semantics/stochastic stage mismatch"
            )
        if not bool(torch.isfinite(values).all()):
            raise ValueError("physical component must be finite on kept support")
        if component.stage in (
            AdditiveStage.PHOTON_ILLUMINATION,
            AdditiveStage.DARK_CHARGE,
        ) and bool((values < 0).any()):
            raise ValueError("photon/dark physical contributions must be nonnegative")

    def validate_evidence(self) -> None:
        """Reject changes to original data/support/physics and parent calibration."""
        self._snapshot.validate()

    def _law_parameters(self, values: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
        self.validate_evidence()
        _matching(values, self.observation.image, "source state")
        if (
            values.shape != self.photon_response.state_shape
            or not bool(torch.isfinite(values).all())
            or bool((values < 0).any())
        ):
            raise ValueError("source state must be finite nonnegative [N,L]")
        illumination = values
        for component in self.additive_components:
            if component.stage == AdditiveStage.PHOTON_ILLUMINATION:
                illumination = illumination + component.values
        if not bool(torch.isfinite(illumination).all()):
            raise ValueError("total source illumination must be finite")
        photon_rate = self.photon_response.forward(illumination)[self._support]
        if bool((photon_rate < 0).any()):
            raise ValueError("physical routed photon rate must be nonnegative")
        image = self.observation.image
        qe = _pixel_parameter(self.electrons_per_photon, image, "qe")[self._support]
        rate = self.exposure_time_s * qe * photon_rate
        offset = torch.zeros_like(rate)
        output = torch.zeros_like(rate)
        for component in self.additive_components:
            if component.stage == AdditiveStage.PHOTON_ILLUMINATION:
                continue
            selected = component.values.reshape(-1)[self._support]
            if component.stage == AdditiveStage.DARK_CHARGE:
                rate = rate + self.exposure_time_s * selected
            elif component.stage == AdditiveStage.READ_OFFSET:
                offset = offset + selected
            else:
                output = output + selected
        gain = self.output_calibration.scale[self._support]
        read_var = _pixel_parameter(
            self.read_variance_electrons2, image, "read_variance"
        )[self._support]
        # This ordering avoids squaring a large/small unit conversion first.
        read_var = (read_var * gain) * gain
        background = gain * offset + output
        for value in (rate, gain, read_var, background):
            if not bool(torch.isfinite(value).all()):
                raise ValueError(
                    "combined physical likelihood parameters must be finite"
                )
        self.validate_evidence()
        return rate, gain, read_var, background

    def expected_output(self, values: Tensor) -> Tensor:
        """Full physical conditional mean; masked output rows are zero.

        This is the untruncated detector expectation, not the expectation of the
        helper's finite retained mixture, and it supplies no noise precision.
        """
        rate, gain, _, background = self._law_parameters(values)
        selected = gain * rate + background
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("expected output must be finite")
        result = (
            torch.zeros_like(self.observation.image)
            .reshape(-1)
            .masked_scatter(self._support, selected)
            .reshape(self.observation.image.shape)
        )
        self.validate_evidence()
        return result

    def nll(
        self,
        values: Tensor,
        *,
        reduction: Literal["sum", "mean", "none"] = "sum",
        mixture_chunk_size: int | None = None,
    ) -> Tensor:
        """Evaluate the normalized kept-pixel law, optionally in pixel chunks.

        ``none`` returns a masked-zero image. With mixture_chunk_size=C, each
        call has at most C kept rows. Non-reentrant checkpointing recomputes
        mixture intermediates in ordinary reverse AD, including double backward:
        O(C*K) first-backward workspace plus O(P) saved law inputs/results.
        Higher-order derivative graphs can retain recomputed intermediates.
        Field sampling and response graphs remain resident. No host transfer,
        change in finite support, detached score or parameter-gradient omission
        is introduced. Floating-point batching can change roundoff.

        The default is unchanged. Forward AD evaluates chunks directly, without
        checkpointing; simultaneous forward/reverse AD therefore retains its
        mixture graph. Evidence must remain fixed through recomputation, not
        only through forward evaluation.
        """
        if reduction not in ("sum", "mean", "none"):
            raise ValueError("reduction must be sum, mean or none")
        if mixture_chunk_size is not None and (
            isinstance(mixture_chunk_size, bool)
            or not isinstance(mixture_chunk_size, int)
            or mixture_chunk_size < 1
        ):
            raise ValueError("mixture_chunk_size must be a positive integer or None")
        rate, gain, read_var, background = self._law_parameters(values)
        observed = self.observation.image.reshape(-1)[self._support]

        def evaluate(y: Tensor, r: Tensor, v: Tensor, g: Tensor, b: Tensor) -> Tensor:
            self.validate_evidence()
            try:
                return poisson_gaussian_nll(
                    y,
                    r,
                    v,
                    gain=g,
                    background=b,
                    tail_sigma=self.tail_sigma,
                    max_terms=self.max_terms,
                    reduction="none",
                )
            finally:
                # Non-reentrant checkpoint can stop recomputation early.
                self.validate_evidence()

        parameters = (observed, rate, read_var, gain, background)
        if mixture_chunk_size is None:
            selected = evaluate(*parameters)
        else:
            forward_ad = any(
                torch.autograd.forward_ad.unpack_dual(p).tangent is not None
                for p in parameters
            )
            recompute = (
                torch.is_grad_enabled()
                and any(p.requires_grad for p in parameters)
                and not forward_ad
            )
            chunks = []
            for start in range(0, observed.numel(), mixture_chunk_size):
                block = tuple(p[start : start + mixture_chunk_size] for p in parameters)
                chunks.append(
                    checkpoint(evaluate, *block, use_reentrant=False)
                    if recompute
                    else evaluate(*block)
                )
            selected = torch.cat(chunks)
        if not bool(torch.isfinite(selected).all()):
            raise ValueError("physical kept-pixel NLL must be finite")
        if reduction == "none":
            result = (
                torch.zeros_like(self.observation.image)
                .reshape(-1)
                .masked_scatter(self._support, selected)
                .reshape(self.observation.image.shape)
            )
        else:
            result = selected.sum() if reduction == "sum" else selected.mean()
        if not bool(torch.isfinite(result).all()):
            raise ValueError("reduced physical NLL exceeds the destination dtype range")
        self.validate_evidence()
        return result

    def nll_field(
        self,
        model: nn.Module,
        *,
        forward_context: ForwardContext | None = None,
        source_chunk_size: int = 4096,
        reduction: Literal["sum", "mean", "none"] = "sum",
        mixture_chunk_size: int | None = None,
    ) -> Tensor:
        """Actual monochromatic field client on source nodes, then native NLL."""
        self.validate_evidence()
        values = self.photon_response.sample_field(
            model, forward_context=forward_context, source_chunk_size=source_chunk_size
        )
        result = self.nll(
            values, reduction=reduction, mixture_chunk_size=mixture_chunk_size
        )
        self.validate_evidence()
        return result


def bind_native_physical_likelihood(
    observation: FieldObservation,
    photon_response: NativeFieldResponse,
    output_calibration: PixelCalibrationOp,
    *,
    exposure_time_s: Tensor,
    electrons_per_photon: Tensor,
    read_variance_electrons2: Tensor,
    measurement_ids: tuple[str, ...],
    calibration_id: str,
    provenance_id: str,
    additive_components: tuple[AdditiveComponent, ...] = (),
    dependencies: EvidenceSnapshot | None = None,
    revision: Callable[[], object] | None = None,
    charge_response: LinearDetectorResponse | None = None,
    read_noise_response: LinearDetectorResponse | None = None,
    tail_sigma: float = 12.0,
    max_terms: int = 4096,
) -> NativePhysicalLikelihood:
    """Bind explicit independent physical collection/read law without a field.

    No legacy weight/noise/additive pipeline is used. Source-independent area/
    throughput conversion must already be declared inside photon_response;
    observation area metadata is provenance only. Exposure/gain metadata on the
    observation rejects, preventing a second ambiguous conversion. Duration is
    scalar; QE/read variance are scalar or explicit row-major [P], preserving
    dtype/device without coercion. General charge/read maps are rejected even
    when a caller claims identity: one test cannot establish independence.
    """
    if charge_response is not None or read_noise_response is not None:
        raise ValueError("independent pixel likelihood rejects charge/read mixing maps")
    return NativePhysicalLikelihood(
        observation=observation,
        photon_response=photon_response,
        output_calibration=output_calibration,
        exposure_time_s=exposure_time_s,
        electrons_per_photon=electrons_per_photon,
        read_variance_electrons2=read_variance_electrons2,
        measurement_ids=measurement_ids,
        calibration_id=calibration_id,
        provenance_id=provenance_id,
        additive_components=additive_components,
        dependencies=dependencies,
        revision=revision,
        tail_sigma=tail_sigma,
        max_terms=max_terms,
    )


__all__ = ["NativePhysicalLikelihood", "bind_native_physical_likelihood"]
