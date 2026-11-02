"""Joint native-field Gaussian evidence with shared additive nuisance modes."""

from __future__ import annotations

from collections.abc import Callable
from typing import Literal

import torch
from torch import Tensor

from ..evidence import EvidenceSnapshot
from ..likelihood import GaussianLikelihoodFactor, _identifiers
from ..operators.bundle import ObservationOperatorBundle
from ..operators.nuisance import (
    CalibratedNuisancePrecision,
    ProfiledNuisancePrecision,
)
from ..solvers import BoundObservationOperator
from ..target import TargetRenderSpec
from .native_response import NativeFieldResponse, bind_native_field_likelihood
from .types import FieldObservation


def _shared_state(responses: tuple[NativeFieldResponse, ...]) -> None:
    reference = responses[0]
    for response in responses:
        if not isinstance(response, NativeFieldResponse):
            raise TypeError("responses must contain NativeFieldResponse objects")
        response.validate_evidence()
        if (
            response.state_shape != reference.state_shape
            or response.source_value_unit != reference.source_value_unit
            or response.source_measure_kind != reference.source_measure_kind
            or response.spectral_measure_kind != reference.spectral_measure_kind
        ):
            raise ValueError(
                "responses must declare the same sampled source state and units"
            )
        for name in ("source_ra_deg", "source_dec_deg"):
            value, first = getattr(response, name), getattr(reference, name)
            if (
                value.dtype != first.dtype
                or value.device != first.device
                or not torch.equal(value, first)
            ):
                raise ValueError(
                    "responses must share ordered source sky coordinates, dtype and device"
                )
        wave, first_wave = (
            response.quadrature.wavelength_angstrom,
            reference.quadrature.wavelength_angstrom,
        )
        if (
            wave.dtype != first_wave.dtype
            or wave.device != first_wave.device
            or not torch.equal(wave, first_wave)
        ):
            raise ValueError(
                "responses must share ordered source wavelengths, dtype and device"
            )


def bind_joint_native_likelihood(
    observations: tuple[FieldObservation, ...],
    responses: tuple[NativeFieldResponse, ...],
    *,
    measurement_ids: tuple[tuple[str, ...], ...],
    nuisance_bases: tuple[Tensor, ...],
    nuisance_basis_units: tuple[str, ...],
    nuisance_ids: tuple[str, ...],
    nuisance_treatment: Literal["profiled", "calibrated"],
    calibration_id: str,
    provenance_id: str,
    calibrated_mean: Tensor | None = None,
    covariance_factor: Tensor | None = None,
    rank_rtol: float | None = None,
    rank_atol: float = 0.0,
    dependencies: EvidenceSnapshot | None = None,
    revision: Callable[[], object] | None = None,
) -> GaussianLikelihoodFactor:
    """Eliminate shared nuisance coordinates once across native observations.

    Each observation is first bound by ``bind_native_field_likelihood`` as
    conditional diagonal Gaussian evidence. These conditional noise terms
    must be independent across observations; shared covariance is introduced
    only through the declared nuisance basis. Other cross-instrument correlated
    read noise requires a different joint base precision. Native detector
    vectors/actions are packed, without building a dense pixel covariance or
    homogenizing PSFs, sampling grids, coordinate charts or product units.

    All responses must use the same ordered source RA/Dec and Angstrom nodes,
    source value units, spatial/spectral measure kinds, shape, dtype and device.
    Source chart coordinates, integration weights and response products may
    differ as declared by the caller. Equality is exact; no coordinate
    tolerance, reordering, interpolation or unit conversion is inferred.

    ``nuisance_ids`` orders shared, standardized dimensionless coefficient
    coordinates. Each ``nuisance_bases[e]`` is explicitly [P_e,K], in native
    row-major pixel order, and carries observation e's product units per
    coefficient. ``nuisance_basis_units[e]`` must equal its observed units.
    The shared coefficient ``calibrated_mean[K]`` and covariance factor
    ``covariance_factor[K,R]`` use those same ordered coefficient coordinates;
    physical conversions between products belong in their supplied bases.
    An omitted mean in calibrated mode means an explicitly zero mean.

    Calibrated mode centers the additive mean once and uses one Woodbury
    action and its determinant correction, including cross-exposure terms.
    Calibration must be independent of conditional measurement noise. Profiled
    mode projects the single joint nuisance span; it does not supply a
    normalized predictive density or restore a removed gauge. Rank thresholds
    are numerical policies from ``ProfiledNuisancePrecision``; deficient bases
    and rank changes do not certify scientific identifiability. Gradients are
    supported on a fixed-rank stratum. The Gaussian factor's data is centered
    and masked; its response stays linear in sampled [N,L] states.

    Original parent evidence, response/calibration bindings, bases, mean and
    covariance factor guard both calls and cached normal/query reads. Declare
    hidden dependencies with ``dependencies``; supply ``revision`` for
    inference-mode inputs and for NumPy/.data alias writes or other state not
    covered by tensor mutation versions. Identifiers alone cannot verify
    independence or lineage. Rebuild after changes and for a new autograd
    graph. A previously retrieved plain tensor cannot intercept later use.

    Packed vectors/bases remain resident. Construction/storage and rank-space
    costs follow the reused nuisance primitive; this is not a bounded total
    training-memory claim or a performance measurement.
    """
    if not isinstance(observations, tuple) or not observations:
        raise ValueError("observations must be a nonempty tuple")
    count = len(observations)
    for value, name in (
        (responses, "responses"),
        (measurement_ids, "measurement_ids"),
        (nuisance_bases, "nuisance_bases"),
        (nuisance_basis_units, "nuisance_basis_units"),
    ):
        if not isinstance(value, tuple) or len(value) != count:
            raise ValueError(f"{name} must have one entry per observation")
    if any(not isinstance(value, FieldObservation) for value in observations):
        raise TypeError("observations must contain FieldObservation objects")
    if any(not isinstance(value, NativeFieldResponse) for value in responses):
        raise TypeError("responses must contain NativeFieldResponse objects")
    _shared_state(responses)
    _identifiers(nuisance_ids, name="nuisance_ids", required=True)
    for ids in measurement_ids:
        _identifiers(ids, name="measurement_ids", required=True)
    ids = tuple(identifier for group in measurement_ids for identifier in group)
    _identifiers(ids, name="joint measurement_ids", required=True)
    for name, value in (
        ("calibration_id", calibration_id),
        ("provenance_id", provenance_id),
    ):
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a nonempty identifier")
    if nuisance_treatment not in ("profiled", "calibrated"):
        raise ValueError("nuisance_treatment must be profiled or calibrated")
    if nuisance_treatment == "profiled" and (
        calibrated_mean is not None or covariance_factor is not None
    ):
        raise ValueError(
            "profiled nuisances do not accept a calibrated mean/covariance"
        )
    if nuisance_treatment == "calibrated":
        if covariance_factor is None:
            raise ValueError("calibrated nuisances require a covariance_factor")
        if rank_rtol is not None or rank_atol != 0:
            raise ValueError("rank tolerances apply only to profiled nuisances")
    if dependencies is not None and not isinstance(dependencies, EvidenceSnapshot):
        raise TypeError("dependencies must be an EvidenceSnapshot")
    if revision is not None and not callable(revision):
        raise TypeError("revision must be callable")
    tensors = {}
    for index, (basis, observation) in enumerate(
        zip(nuisance_bases, observations, strict=True)
    ):
        if not isinstance(basis, Tensor) or basis.shape != (
            observation.image.numel(),
            len(nuisance_ids),
        ):
            raise ValueError("nuisance bases must have explicit shape [P_e,K]")
        if (
            basis.dtype != observation.image.dtype
            or basis.device != observation.image.device
        ):
            raise ValueError("nuisance bases must match observed dtype and device")
        if nuisance_basis_units[index] != observation.value_unit:
            raise ValueError("nuisance basis units must match each observed product")
        tensors[f"nuisance_basis[{index}]"] = basis
    for name, value in (
        ("calibrated_mean", calibrated_mean),
        ("covariance_factor", covariance_factor),
    ):
        if value is not None:
            if not isinstance(value, Tensor):
                raise TypeError(f"{name} must be a tensor")
            tensors[name] = value
    if revision is None and (
        torch.is_inference_mode_enabled()
        or any(value.is_inference() for value in tensors.values())
    ):
        raise ValueError("inference-mode joint evidence requires an explicit revision")

    parents = tuple(
        bind_native_field_likelihood(
            observation,
            response,
            measurement_ids=group,
            provenance_id=f"{provenance_id}:parent:{index}",
            revision=revision,
        )
        for index, (observation, response, group) in enumerate(
            zip(observations, responses, measurement_ids, strict=True)
        )
    )

    def current_revision() -> object:
        if dependencies is not None:
            dependencies.validate()
        for parent, response in zip(parents, responses, strict=True):
            parent.validate_evidence()
            response.validate_evidence()
        return None if revision is None else revision()

    snapshot = EvidenceSnapshot.capture(tensors, revision=current_revision)
    sizes = tuple(parent.data.numel() for parent in parents)
    data = torch.cat(tuple(parent.data for parent in parents))

    def unpack(value: Tensor) -> tuple[Tensor, ...]:
        if (
            not isinstance(value, Tensor)
            or value.shape != data.shape
            or value.dtype != data.dtype
            or value.device != data.device
        ):
            raise ValueError(
                "joint vector must preserve packed shape, dtype and device"
            )
        return tuple(value.split(sizes))

    def base_precision(value: Tensor) -> Tensor:
        snapshot.validate()
        result = torch.cat(
            tuple(
                parent.precision(block)
                for parent, block in zip(parents, unpack(value), strict=True)
            )
        )
        snapshot.validate()
        return result

    valid = base_precision(torch.ones_like(data)) > 0
    basis = torch.cat(nuisance_bases, dim=0)
    if nuisance_treatment == "calibrated":
        assert covariance_factor is not None
        nuisance = CalibratedNuisancePrecision(
            basis,
            apply_precision=base_precision,
            covariance_factor=covariance_factor,
            mean=calibrated_mean,
            valid=valid,
        )
        data = nuisance.center(data)
        normalization = (
            sum(
                (
                    parent.log_normalization
                    for parent in parents
                    if parent.log_normalization is not None
                ),
                data.new_zeros(()),
            )
            + 0.5 * nuisance.logdet_covariance_correction
        )
    else:
        nuisance = ProfiledNuisancePrecision(
            basis,
            apply_precision=base_precision,
            valid=valid,
            rank_rtol=rank_rtol,
            rank_atol=rank_atol,
        )
        normalization = None

    def forward(scene, target) -> Tensor:
        del target
        snapshot.validate()
        result = torch.cat(tuple(parent.operator.apply(scene) for parent in parents))
        snapshot.validate()
        return result

    def adjoint(value: Tensor) -> Tensor:
        snapshot.validate()
        results = tuple(
            parent.operator.adjoint(block)
            for parent, block in zip(parents, unpack(value), strict=True)
        )
        result = sum(results[1:], results[0])
        snapshot.validate()
        return result

    def precision(value: Tensor) -> Tensor:
        snapshot.validate()
        result = nuisance.apply(value)
        snapshot.validate()
        return result

    snapshot.validate()
    operator = BoundObservationOperator(
        ObservationOperatorBundle(),
        TargetRenderSpec(
            header={}, shape=(1, sum(sizes)), band_label="joint_native_likelihood"
        ),
        apply_operator=forward,
        apply_adjoint=adjoint,
        apply_noise=precision,
    )
    return GaussianLikelihoodFactor(
        operator,
        data,
        ids,
        calibration_id,
        provenance_id,
        nuisance_treatment=nuisance_treatment,
        eliminated_nuisance_ids=nuisance_ids,
        log_normalization=normalization,
        dependencies=snapshot,
    )


__all__ = ["bind_joint_native_likelihood"]
