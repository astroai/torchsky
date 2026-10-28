"""Fixed-band field operators with explicit joint Gaussian nuisance evidence.

This bridge binds a bounded exposure/tile group. It does not fit a global field,
change its response convention, or make different bandpasses interchangeable.
Shared coefficients are eliminated once across the complete declared group.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import replace
import math

import torch
from torch import Tensor

from ..evidence import EvidenceSnapshot
from ..likelihood import GaussianLikelihoodFactor, NuisanceTreatment
from ..operators import ObservationOperatorBundle
from ..operators.background import apply_calibrated_additive_response
from ..operators.nuisance import CalibratedNuisancePrecision, ProfiledNuisancePrecision
from ..operators.psf import apply_kernel2d
from ..solvers import BoundObservationOperator
from .inverse import (
    bind_linear_band_grid_observation_operator,
    prepare_linear_band_grid_observation_data,
)
from .observation_ops import observation_support_mask
from .types import FieldDomain, FieldObservation


def _spectral_signature(observation: FieldObservation) -> tuple[object, ...]:
    return tuple(
        getattr(observation, name)
        for name in (
            "band_name",
            "wavelength",
            "spectral_min",
            "spectral_max",
            "transmission_center",
            "transmission_sigma",
            "spectral_quantity",
            "spectral_unit",
        )
    )


def _revision_signature(observation: FieldObservation) -> tuple[object, ...]:
    psf = observation.psf_spec
    return (
        _spectral_signature(observation),
        tuple(
            sorted((str(key), repr(value)) for key, value in observation.header.items())
        ),
        observation.background_offset,
        observation.detector_x0_pix,
        observation.detector_y0_pix,
        observation.value_unit,
        observation.value_semantics,
        observation.detector_law,
        observation.processing_state,
        tuple(
            (name, id(getattr(observation, name)))
            for name in (
                "image",
                "weight",
                "mask",
                "sky_background",
                "instrument_background",
                "detector_background",
                "transmission_wavelength",
                "transmission_throughput",
                "detector_to_sky",
                "passband",
                "psf_spec",
            )
        ),
        None
        if psf is None
        else (psf.mode, psf.frozen, psf.pixel_response_included, id(psf.kernel2d)),
        tuple(
            (
                id(component),
                id(component.values),
                component.stage.value,
                component.coordinate_frame.value,
                component.value_unit,
                component.value_semantics,
                component.processing_state.value,
                component.stochastic_contribution,
            )
            for component in observation.additive_components
        ),
    )


def bind_linear_band_grid_likelihood(
    observations: tuple[FieldObservation, ...],
    domain: FieldDomain,
    *,
    spatial_resolution: int,
    spatial_bounds_local_deg: Sequence[float],
    measurement_ids: tuple[tuple[str, ...], ...],
    calibration_id: str,
    provenance_id: str,
    nuisance_bases: tuple[Tensor, ...] | None = None,
    nuisance_ids: tuple[str, ...] = (),
    nuisance_covariance_factor: Tensor | None = None,
    nuisance_mean: Tensor | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
    spatial_tile_hw: tuple[int, int] | None = None,
    revision: Callable[[], object] | None = None,
) -> GaussianLikelihoodFactor:
    """Bind native images to one fixed band-grid Gaussian likelihood factor.

    Observations use the existing frozen discrete image-grid response and must
    share band metadata, product units and real float32/float64 dtype/device.
    This is not continuous/chromatic pre-pixel rendering. Fixed legacy output
    backgrounds and supported typed input/output means are subtracted in their
    declared detector stages. Physical raw additive/noise stages still reject.

    Each optional ``nuisance_bases[e]`` has shape ``(*image.shape, k)`` in that
    detector product's units, with the same ordered coefficients/identifiers.
    No covariance factor means fixed unknown coefficients profiled jointly.
    With an independently calibrated ``L``, covariance is ``L L.T`` and mean
    centering/determinant correction are applied jointly. All shared exposures
    must be in this factor; splitting shared coefficients across independent
    factors is invalid and their repeated IDs are rejected by the system.

    Noise is fixed independent Gaussian with the existing support/weight
    convention. Conditional/calibrated densities include the observed-support
    normalization. No normalized predictive density is claimed for profiling.
    Explicit non-Gaussian detector laws reject rather than being approximated.

    Original data, weights, masks, background/calibration tensors and response
    metadata are tracked by ``EvidenceSnapshot``. A custom detector-to-sky or
    passband object requires ``revision`` for its otherwise hidden state.
    Rebuild the complete binding after changing dependencies or autograd graphs.
    Ordinary Torch writes and object replacements are checked; NumPy/.data
    alias writes still require a truthful caller revision.

    Packing uses O(N) measurement storage and O(N k) nuisance storage for this
    bounded group, with no dense pixel covariance. This is not a streaming
    whole-survey shared-calibration solver. Use suitable exposure/tile groups
    and do not treat cropped pieces with shared errors as independent factors.
    """
    if not isinstance(observations, tuple) or not observations:
        raise ValueError("observations must be a nonempty tuple")
    if not isinstance(measurement_ids, tuple) or len(measurement_ids) != len(
        observations
    ):
        raise ValueError(
            "measurement_ids must have one native-group tuple per observation"
        )
    if any(not isinstance(ids, tuple) or not ids for ids in measurement_ids):
        raise ValueError(
            "each observation requires a nonempty native measurement-group tuple"
        )
    first = observations[0]
    if not isinstance(first, FieldObservation):
        raise TypeError("observations must contain FieldObservation objects")
    if first.image.dtype not in (torch.float32, torch.float64):
        raise TypeError("joint field likelihood requires real float32/float64 data")
    if not first.band_name or not first.value_unit:
        raise ValueError("a fixed band_name and calibrated value_unit are required")
    tracked: dict[str, Tensor] = {}
    for index, observation in enumerate(observations):
        if not isinstance(observation, FieldObservation):
            raise TypeError("observations must contain FieldObservation objects")
        if (
            observation.image.dtype != first.image.dtype
            or observation.image.device != first.image.device
        ):
            raise ValueError("observations must share dtype and device")
        if observation.value_unit != first.value_unit or _spectral_signature(
            observation
        ) != _spectral_signature(first):
            raise ValueError(
                "a fixed band-grid likelihood cannot combine different units or band responses"
            )
        if observation.detector_law not in (None, "gaussian"):
            raise ValueError(
                "joint field Gaussian likelihood requires a fixed Gaussian noise law"
            )
        if observation.detector_to_sky is not None or observation.passband is not None:
            if revision is None:
                raise ValueError(
                    "custom geometry/passband state requires an explicit revision getter"
                )
        for name in ("transmission_wavelength", "transmission_throughput"):
            current, reference = getattr(observation, name), getattr(first, name)
            if (current is None) != (reference is None) or (
                current is not None and not torch.equal(current, reference)
            ):
                raise ValueError(
                    "a fixed band-grid likelihood requires matching bandpass tables"
                )
        if observation.passband is not first.passband:
            raise ValueError("custom passband objects must be shared and revisioned")
        for name in (
            "image",
            "weight",
            "mask",
            "sky_background",
            "instrument_background",
            "detector_background",
            "transmission_wavelength",
            "transmission_throughput",
        ):
            value = getattr(observation, name)
            if isinstance(value, Tensor):
                tracked[f"observation[{index}].{name}"] = value
        if (
            observation.psf_spec is not None
            and observation.psf_spec.kernel2d is not None
        ):
            tracked[f"observation[{index}].kernel"] = observation.psf_spec.kernel2d
        for number, component in enumerate(observation.additive_components):
            tracked[f"observation[{index}].additive[{number}]"] = component.values
    if nuisance_bases is None:
        if (
            nuisance_ids
            or nuisance_covariance_factor is not None
            or nuisance_mean is not None
        ):
            raise ValueError("nuisance calibration/identifiers require explicit bases")
    elif not isinstance(nuisance_bases, tuple) or len(nuisance_bases) != len(
        observations
    ):
        raise ValueError("nuisance_bases must have one tensor per observation")
    else:
        if (
            not isinstance(nuisance_ids, tuple)
            or not nuisance_ids
            or any(
                not isinstance(name, str) or not name.strip() for name in nuisance_ids
            )
            or len(set(nuisance_ids)) != len(nuisance_ids)
        ):
            raise ValueError("joint nuisances require unique nonempty identifiers")
        for index, (basis, observation) in enumerate(
            zip(nuisance_bases, observations, strict=True)
        ):
            if (
                not isinstance(basis, Tensor)
                or basis.shape != (*observation.image.shape, len(nuisance_ids))
                or basis.dtype != first.image.dtype
                or basis.device != first.image.device
            ):
                raise ValueError(
                    "nuisance bases must match detector shape, mode count, dtype and device"
                )
            tracked[f"nuisance_basis[{index}]"] = basis
        if nuisance_covariance_factor is None and nuisance_mean is not None:
            raise ValueError(
                "a calibrated nuisance mean requires its covariance factor"
            )
    for name, value in (
        ("nuisance_factor", nuisance_covariance_factor),
        ("nuisance_mean", nuisance_mean),
    ):
        if value is not None:
            tracked[name] = value
    if revision is None and (
        torch.is_inference_mode_enabled()
        or any(tensor.is_inference() for tensor in tracked.values())
    ):
        raise ValueError(
            "inference-mode evidence requires an explicit caller revision getter"
        )

    def current_revision() -> object:
        return (
            tuple(_revision_signature(observation) for observation in observations),
            domain.coord_mode,
            domain.center_ra_deg,
            domain.center_dec_deg,
            None if revision is None else revision(),
        )

    snapshot = EvidenceSnapshot.capture(tracked, revision=current_revision)
    snapshots = []
    operators = []
    shapes = []
    support_parts = []
    weight_parts = []
    for observation in observations:
        plain = replace(observation, additive_components=())
        operator = bind_linear_band_grid_observation_operator(
            plain,
            domain,
            spatial_resolution=spatial_resolution,
            spatial_bounds_local_deg=spatial_bounds_local_deg,
            spatial_tile_hw=spatial_tile_hw,
        )
        data = prepare_linear_band_grid_observation_data(plain)
        if observation.additive_components:
            psf = observation.psf_spec
            assert observation.value_unit is not None
            assert observation.value_semantics is not None
            assert observation.processing_state is not None

            def response(image: Tensor) -> Tensor:
                if psf is None or psf.mode == "none":
                    return image
                assert psf.kernel2d is not None
                return apply_kernel2d(image, psf.kernel2d)

            additive = apply_calibrated_additive_response(
                torch.zeros_like(observation.image),
                observation.additive_components,
                response=response,
                value_unit=observation.value_unit,
                value_semantics=observation.value_semantics,
                processing_state=observation.processing_state,
            )
            data = data - torch.where(
                observation_support_mask(observation), additive, 0.0
            )
        support = observation_support_mask(observation)
        snapshots.append(data.reshape(-1))
        operators.append(operator)
        shapes.append(observation.image.shape)
        support_parts.append(support.reshape(-1))
        weight_parts.append(
            operator.apply_noise_model(torch.ones_like(observation.image)).reshape(-1)
        )
    sizes = tuple(math.prod(shape) for shape in shapes)
    data = torch.cat(snapshots)
    support = torch.cat(support_parts)
    weight = torch.cat(weight_parts)

    def base_precision(value: Tensor) -> Tensor:
        snapshot.validate()
        return weight * value

    treatment: NuisanceTreatment = "conditional"
    normalization: Tensor | None = 0.5 * (
        int(support.sum().item()) * math.log(2 * math.pi) - weight[support].log().sum()
    )
    precision: Callable[[Tensor], Tensor] = base_precision
    if nuisance_bases is not None:
        basis = torch.cat([b.reshape(-1, len(nuisance_ids)) for b in nuisance_bases])
        if nuisance_covariance_factor is None:
            profiled = ProfiledNuisancePrecision(
                basis,
                apply_precision=base_precision,
                valid=support,
                rank_rtol=rank_rtol,
                rank_atol=rank_atol,
            )
            precision = profiled.apply
            treatment = "profiled"
            normalization = None
        else:
            calibrated = CalibratedNuisancePrecision(
                basis,
                apply_precision=base_precision,
                valid=support,
                covariance_factor=nuisance_covariance_factor,
                mean=nuisance_mean,
            )
            data = calibrated.center(data)
            precision = calibrated.apply
            treatment = "calibrated"
            assert normalization is not None
            normalization = (
                normalization + 0.5 * calibrated.logdet_covariance_correction
            )

    def forward(scene, target) -> Tensor:
        del target
        snapshot.validate()
        predictions = [op.apply(scene).reshape(-1) for op in operators]
        snapshot.validate()
        return torch.cat(predictions)

    def adjoint(value: Tensor) -> Tensor:
        snapshot.validate()
        parts = value.split(sizes)
        result: Tensor | None = None
        for operator, part, shape in zip(operators, parts, shapes, strict=True):
            contribution = operator.adjoint(part.reshape(shape))
            result = contribution if result is None else result + contribution
        snapshot.validate()
        assert result is not None
        return result

    joint = BoundObservationOperator(
        ObservationOperatorBundle(),
        operators[0].target,
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=precision,
    )
    return GaussianLikelihoodFactor(
        joint,
        data,
        tuple(identifier for ids in measurement_ids for identifier in ids),
        calibration_id,
        provenance_id,
        treatment,
        nuisance_ids,
        normalization,
        dependencies=snapshot,
    )


__all__ = ["bind_linear_band_grid_likelihood"]
