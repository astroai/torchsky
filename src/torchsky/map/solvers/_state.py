"""Internal solver hooks for field fitting and discrete inverse mapmaking."""

from __future__ import annotations

from ._core import (
    BoundObservationOperator,
    BoundObservationOperatorSet,
    _relative_norm,
    _subspace_leakage,
)
from ._types import ApplyNoiseFn

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor, nn

from ..operators.bundle import ObservationOperatorBundle
from ..scenes import (
    DiscreteTargetImage,
    HarmonicAlmScene,
    HealpixMapScene,
    SceneRepresentation,
    SpinHarmonicAlmScene,
    as_scene_representation,
)
from ..target import TargetRenderSpec
from ...device import _as_tensor, _to


def _scene_state_tensor(scene: SceneRepresentation) -> Tensor:
    if isinstance(scene, DiscreteTargetImage):
        return torch.as_tensor(scene.data)
    if isinstance(scene, HealpixMapScene):
        return torch.as_tensor(scene.data)
    if isinstance(scene, HarmonicAlmScene):
        return torch.as_tensor(scene.data)
    if isinstance(scene, SpinHarmonicAlmScene):
        return torch.stack(
            [torch.as_tensor(scene.alm_e), torch.as_tensor(scene.alm_b)],
            dim=0,
        )
    raise TypeError(
        "state-space identity operator requires a tensor-backed scene representation"
    )


def _flatten_state(tensor: Tensor) -> Tensor:
    return torch.as_tensor(tensor).reshape(-1)


def _reshape_state(flat: Tensor, reference: Tensor) -> Tensor:
    return torch.as_tensor(flat).reshape_as(reference)


def _canonical_subspace_indices(
    reference_state: Tensor,
    selector: int | slice | Sequence[int] | Tensor,
) -> tuple[Tensor, bool]:
    state = torch.as_tensor(reference_state)
    if state.ndim < 1:
        raise ValueError("reference_state must have at least one dimension")
    if isinstance(selector, int):
        idx = torch.tensor([int(selector)], dtype=torch.int64)
        squeeze = True
    elif isinstance(selector, slice):
        start, stop, step = selector.indices(int(state.shape[0]))
        idx = torch.arange(start, stop, step, dtype=torch.int64)
        squeeze = False
    else:
        idx = torch.as_tensor(selector, dtype=torch.int64).reshape(-1)
        squeeze = False
    if idx.numel() == 0:
        raise ValueError("subspace selector must select at least one index")
    if int(idx.min()) < 0 or int(idx.max()) >= int(state.shape[0]):
        raise ValueError("subspace selector is out of bounds for reference_state")
    return idx, squeeze


def build_state_identity_operator(
    reference_state: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    """Bind the identity map on a flattened scene state.

    Useful as a ridge or anchor term when composing with data operators via
    :func:`compose_bound_operators`.
    """
    state = torch.as_tensor(reference_state)
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.numel()),
            "NAXIS2": 1,
        },
        shape=(1, int(state.numel())),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "state-space identity operator received a scene with incompatible shape"
            )
        return tensor

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(mapped.shape) != tuple(state.shape):
            raise ValueError(
                "state-space identity adjoint received data with incompatible shape"
            )
        return mapped

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_subspace_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    selector: int | slice | Sequence[int] | Tensor,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    """Restrict ``base_operator`` to a subset of the leading state axis.

    ``selector`` picks rows of the state (index, slice, sequence, or tensor);
    the adjoint embeds results back into a zero state of the original shape.
    """
    state = torch.as_tensor(reference_state)
    indices, squeeze = _canonical_subspace_indices(state, selector)

    def _extract(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "subspace operator received a scene with incompatible shape"
            )
        subset = tensor.index_select(0, indices.to(device=tensor.device))
        if squeeze:
            return subset[0]
        return subset

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        out = torch.zeros_like(state)
        if squeeze:
            adj = torch.as_tensor(adjoint_tensor)
            if adj.ndim != state.ndim - 1 or tuple(adj.shape) != tuple(state.shape[1:]):
                raise ValueError("subspace adjoint returned an incompatible tensor")
            out[indices[0]] = _to(adj, dtype=state.dtype, device=state.device)
            return out
        adj = torch.as_tensor(adjoint_tensor)
        expected = (int(indices.numel()), *tuple(state.shape[1:]))
        if tuple(adj.shape) != expected:
            raise ValueError("subspace adjoint returned an incompatible tensor")
        out.index_copy_(
            0,
            indices.to(device=out.device),
            _to(adj, dtype=state.dtype, device=out.device),
        )
        return out

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_extract(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators),
            exposure_chunk_size=base_operator.exposure_chunk_size,
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


def build_state_component_mixture_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    components: Sequence[int] | Tensor,
    *,
    weights: Sequence[complex | float] | Tensor | None = None,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    state = torch.as_tensor(reference_state)
    if state.ndim < 2:
        raise ValueError(
            "component mixture operator requires a reference_state with a leading component axis"
        )
    component_idx = torch.as_tensor(components, dtype=torch.int64).reshape(-1)
    if int(component_idx.numel()) == 0:
        raise ValueError("components must select at least one latent component")
    if int(component_idx.min()) < 0 or int(component_idx.max()) >= int(state.shape[0]):
        raise ValueError("components are out of bounds for the reference_state")
    weights_t = (
        torch.ones(
            int(component_idx.numel()),
            dtype=state.dtype if state.is_complex() else torch.float32,
            device=state.device,
        )
        if weights is None
        else _as_tensor(
            weights,
            dtype=state.dtype if state.is_complex() else torch.float32,
            device=state.device,
        ).reshape(-1)
    )
    if int(weights_t.numel()) != int(component_idx.numel()):
        raise ValueError("weights length must match the number of selected components")

    def _mix(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "component mixture operator received a scene with incompatible shape"
            )
        subset = tensor.index_select(0, component_idx.to(device=tensor.device))
        coeffs = _to(weights_t, dtype=subset.dtype, device=subset.device).view(
            int(component_idx.numel()),
            *([1] * (subset.ndim - 1)),
        )
        return torch.sum(coeffs * subset, dim=0)

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        adj = torch.as_tensor(adjoint_tensor)
        expected = tuple(state.shape[1:])
        if tuple(adj.shape) != expected:
            raise ValueError(
                "component mixture adjoint returned an incompatible tensor"
            )
        out = torch.zeros_like(state)
        coeffs = torch.conj(_to(weights_t, dtype=adj.dtype, device=adj.device))
        for idx, weight in zip(component_idx.tolist(), coeffs, strict=True):
            out[idx] = out[idx] + _to(weight * adj, dtype=out.dtype, device=out.device)
        return out

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_mix(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators),
            exposure_chunk_size=base_operator.exposure_chunk_size,
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


class SurveyInstrumentComponentMixer(nn.Module):
    """Mechanistic learned context mixer over latent shared/private components.

    Each latent component is assigned a role:
    - ``shared``: seen by every context through one learned global weight
    - ``survey``: context-specific survey weight
    - ``instrument``: context-specific instrument weight
    - ``interaction``: context-specific survey-instrument interaction weight

    This keeps the decomposition explicit while still allowing the relative
    contribution of each component family to be learned from data.
    """

    _VALID_ROLES = ("shared", "survey", "instrument", "interaction")

    shared_mask: Tensor
    survey_mask: Tensor
    instrument_mask: Tensor
    interaction_mask: Tensor

    def __init__(
        self,
        *,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        roles = tuple(str(role) for role in component_roles)
        unknown = tuple(role for role in roles if role not in self._VALID_ROLES)
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        self.component_roles = roles
        self.surveys = tuple(str(name) for name in surveys)
        self.instruments = tuple(str(name) for name in instruments)
        ncomp = len(roles)
        self.shared_weights = nn.Parameter(
            torch.zeros(ncomp, dtype=dtype, device=device)
        )
        self.survey_weights = nn.Parameter(
            torch.zeros(len(self.surveys) + 1, ncomp, dtype=dtype, device=device)
        )
        self.instrument_weights = nn.Parameter(
            torch.zeros(len(self.instruments) + 1, ncomp, dtype=dtype, device=device)
        )
        self.interaction_weights = nn.Parameter(
            torch.zeros(
                len(self.surveys) + 1,
                len(self.instruments) + 1,
                ncomp,
                dtype=dtype,
                device=device,
            )
        )
        role_tensor = {
            role: torch.tensor(
                [1.0 if item == role else 0.0 for item in roles],
                dtype=dtype,
                device=device,
            )
            for role in self._VALID_ROLES
        }
        for role, tensor in role_tensor.items():
            self.register_buffer(f"{role}_mask", tensor)

    def _survey_index(self, survey: str | None) -> int:
        if survey is None:
            return 0
        try:
            return self.surveys.index(str(survey)) + 1
        except ValueError:
            return 0

    def _instrument_index(self, instrument: str | None) -> int:
        if instrument is None:
            return 0
        try:
            return self.instruments.index(str(instrument)) + 1
        except ValueError:
            return 0

    def forward(
        self,
        *,
        survey: str | None = None,
        instrument: str | None = None,
    ) -> Tensor:
        sidx = self._survey_index(survey)
        iidx = self._instrument_index(instrument)
        return (
            self.shared_mask * self.shared_weights
            + self.survey_mask * self.survey_weights[sidx]
            + self.instrument_mask * self.instrument_weights[iidx]
            + self.interaction_mask * self.interaction_weights[sidx, iidx]
        )

    def scale_component_columns_(
        self,
        scales: Tensor | Sequence[float],
    ) -> None:
        """Rescale role-matched mixer columns in-place for exact gauge preservation."""
        scales_t = _as_tensor(
            scales,
            dtype=self.shared_weights.dtype,
            device=self.shared_weights.device,
        ).reshape(-1)
        if int(scales_t.numel()) != len(self.component_roles):
            raise ValueError("scales length must match the number of components")
        if not torch.all(torch.isfinite(scales_t)):
            raise ValueError("scales must be finite")
        if torch.any(scales_t == 0.0):
            raise ValueError("scales must be non-zero")
        with torch.no_grad():
            for idx, role in enumerate(self.component_roles):
                scale = scales_t[idx]
                if role == "shared":
                    self.shared_weights[idx].mul_(scale)
                elif role == "survey":
                    self.survey_weights[:, idx].mul_(scale)
                elif role == "instrument":
                    self.instrument_weights[:, idx].mul_(scale)
                elif role == "interaction":
                    self.interaction_weights[:, :, idx].mul_(scale)
                else:  # pragma: no cover - guarded by __init__
                    raise ValueError(f"unsupported component role {role!r}")


@dataclass(frozen=True)
class ComponentDecompositionDiagnostics:
    """Diagnostics for explicit shared/private latent decompositions."""

    shared_to_private_leakage: float
    private_to_shared_leakage: float
    shared_relative_error: float
    private_relative_error: float
    calibration_transfer_stability: float | None = None


class HarmonicComponentDecomposition(nn.Module):
    """Learnable harmonic component state with explicit shared/private roles."""

    def __init__(
        self,
        *,
        lmax: int,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        init_scale: float = 1.0e-2,
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        super().__init__()
        ll = int(lmax)
        if ll < 0:
            raise ValueError("lmax must be non-negative")
        self.lmax = ll
        self.nalm = (ll + 1) * (ll + 2) // 2
        self.component_roles = tuple(str(role) for role in component_roles)
        ncomp = len(self.component_roles)
        self.coeff_real = nn.Parameter(
            float(init_scale)
            * torch.randn(ncomp, self.nalm, dtype=dtype, device=device)
        )
        self.coeff_imag = nn.Parameter(
            float(init_scale)
            * torch.randn(ncomp, self.nalm, dtype=dtype, device=device)
        )
        self.mixer = SurveyInstrumentComponentMixer(
            component_roles=self.component_roles,
            surveys=surveys,
            instruments=instruments,
            dtype=dtype,
            device=device,
        )

    def state(self) -> Tensor:
        return torch.complex(self.coeff_real, self.coeff_imag)

    def forward(self) -> Tensor:
        return self.state()

    def conditioned_operator(
        self,
        base_operator: BoundObservationOperator | BoundObservationOperatorSet,
        *,
        survey: str | None = None,
        instrument: str | None = None,
    ) -> BoundObservationOperator | BoundObservationOperatorSet:
        return build_conditioned_component_mixture_operator(
            self.state(),
            base_operator,
            self.mixer,
            survey=survey,
            instrument=instrument,
        )

    def component_mask(self, role: str) -> Tensor:
        if role not in SurveyInstrumentComponentMixer._VALID_ROLES:
            raise ValueError(f"unsupported component role {role!r}")
        return _to(
            getattr(self.mixer, f"{role}_mask"),
            device=self.coeff_real.device,
            dtype=self.coeff_real.dtype,
        )

    def component_state(
        self,
        roles: str | Sequence[str],
        *,
        state: Tensor | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        state_t = self.state() if state is None else torch.as_tensor(state)
        if state_t.ndim < 2 or int(state_t.shape[0]) != len(self.component_roles):
            raise ValueError(
                "state must have a leading component axis matching component_roles"
            )
        role_names = (
            (roles,) if isinstance(roles, str) else tuple(str(role) for role in roles)
        )
        unknown = tuple(
            role
            for role in role_names
            if role not in SurveyInstrumentComponentMixer._VALID_ROLES
        )
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        selected = [
            idx
            for idx, role in enumerate(self.component_roles)
            if role in set(role_names)
        ]
        if not selected:
            empty = state_t[:0]
            if aggregate:
                return torch.zeros_like(state_t[0])
            return empty
        subset = state_t.index_select(
            0,
            torch.tensor(selected, dtype=torch.int64, device=state_t.device),
        )
        if aggregate:
            return subset.sum(dim=0)
        return subset

    def component_norms(self) -> Tensor:
        state = self.state()
        return torch.linalg.vector_norm(state, dim=1).to(dtype=self.coeff_real.dtype)

    def fix_component_scale_gauge_(
        self,
        *,
        target_norm: float = 1.0,
        min_norm: float = 1.0e-12,
    ) -> Tensor:
        """Canonicalize multiplicative component gauge without changing predictions.

        Each component column is rescaled to ``target_norm`` in harmonic state space,
        and the matching mixer weights are scaled by the inverse factor so every
        conditioned operator remains exactly unchanged.
        """
        if target_norm <= 0.0:
            raise ValueError("target_norm must be positive")
        if min_norm <= 0.0:
            raise ValueError("min_norm must be positive")
        norms = self.component_norms()
        scale = torch.where(
            norms > float(min_norm),
            norms / float(target_norm),
            torch.ones_like(norms),
        )
        with torch.no_grad():
            self.coeff_real.div_(scale[:, None])
            self.coeff_imag.div_(scale[:, None])
        self.mixer.scale_component_columns_(scale)
        return self.component_norms()

    def decomposition_diagnostics(
        self,
        truth_state: Tensor,
        *,
        state: Tensor | None = None,
        calibration_anchor: Tensor | None = None,
        calibration_selector: int | slice | Sequence[int] | Tensor | None = None,
    ) -> ComponentDecompositionDiagnostics:
        estimate = self.state() if state is None else torch.as_tensor(state)
        truth = torch.as_tensor(truth_state)
        if tuple(estimate.shape) != tuple(truth.shape):
            raise ValueError("state and truth_state must have matching shape")
        if estimate.ndim < 2 or int(estimate.shape[0]) != len(self.component_roles):
            raise ValueError(
                "state and truth_state must have a leading component axis matching component_roles"
            )
        shared_truth = self.component_state("shared", state=truth, aggregate=True)
        private_truth = self.component_state(
            ("survey", "instrument", "interaction"),
            state=truth,
            aggregate=True,
        )
        private_truth_basis = self.component_state(
            ("survey", "instrument", "interaction"),
            state=truth,
            aggregate=False,
        )
        shared_estimate = self.component_state("shared", state=estimate, aggregate=True)
        private_estimate = self.component_state(
            ("survey", "instrument", "interaction"),
            state=estimate,
            aggregate=True,
        )

        shared_error = shared_estimate - shared_truth
        private_error = private_estimate - private_truth
        calibration_transfer_stability = None
        if calibration_anchor is not None:
            anchor = torch.as_tensor(calibration_anchor)
            if tuple(anchor.shape) != tuple(shared_truth.shape):
                raise ValueError(
                    "calibration_anchor must match the aggregated shared-state shape"
                )
            shared_for_anchor = shared_estimate
            anchor_ref = anchor
            if calibration_selector is not None:
                selector_idx, _ = _canonical_subspace_indices(
                    shared_truth, calibration_selector
                )
                shared_for_anchor = shared_for_anchor.index_select(0, selector_idx)
                anchor_ref = anchor_ref.index_select(
                    0, selector_idx.to(device=anchor_ref.device)
                )
            calibration_transfer_stability = _relative_norm(
                shared_for_anchor - anchor_ref,
                anchor_ref,
            )
        return ComponentDecompositionDiagnostics(
            shared_to_private_leakage=_subspace_leakage(
                shared_error,
                private_truth_basis,
                private_truth,
            ),
            private_to_shared_leakage=_subspace_leakage(
                private_error,
                shared_truth.unsqueeze(0),
                shared_truth,
            ),
            shared_relative_error=_relative_norm(shared_error, shared_truth),
            private_relative_error=_relative_norm(private_error, private_truth),
            calibration_transfer_stability=calibration_transfer_stability,
        )

    def regularization_loss(
        self,
        *,
        private_harmonic_diagonal: Tensor | None = None,
        private_weight: float = 1.0,
        mixer_weight: float = 1.0,
    ) -> Tensor:
        state = self.state()
        loss = torch.zeros(
            (), dtype=self.coeff_real.dtype, device=self.coeff_real.device
        )
        private_mask = (
            self.component_mask("survey")
            + self.component_mask("instrument")
            + self.component_mask("interaction")
        ).clamp(max=1.0)
        if private_harmonic_diagonal is not None:
            diag = _as_tensor(
                private_harmonic_diagonal,
                dtype=self.coeff_real.dtype,
                device=self.coeff_real.device,
            ).reshape(1, self.nalm)
            weighted = diag * torch.abs(state)
            loss = loss + float(private_weight) * torch.mean(
                private_mask.view(-1, 1) * torch.square(weighted)
            )
        loss = loss + float(mixer_weight) * (
            torch.mean(torch.square(self.mixer.survey_weights))
            + torch.mean(torch.square(self.mixer.instrument_weights))
            + torch.mean(torch.square(self.mixer.interaction_weights))
        )
        return loss


def build_conditioned_component_mixture_operator(
    reference_state: Tensor,
    base_operator: BoundObservationOperator | BoundObservationOperatorSet,
    mixer: SurveyInstrumentComponentMixer,
    *,
    survey: str | None = None,
    instrument: str | None = None,
) -> BoundObservationOperator | BoundObservationOperatorSet:
    state = torch.as_tensor(reference_state)
    if state.ndim < 2:
        raise ValueError(
            "conditioned component mixture requires a reference_state with a leading component axis"
        )
    ncomp = int(state.shape[0])
    if len(mixer.component_roles) != ncomp:
        raise ValueError(
            "mixer component_roles length must match reference_state leading dimension"
        )

    def _weights_for(dtype: torch.dtype, device: torch.device) -> Tensor:
        weights = mixer(survey=survey, instrument=instrument)
        return _to(weights, dtype=dtype, device=device)

    def _mix(scene: SceneRepresentation | nn.Module | Tensor) -> Tensor:
        tensor = _scene_state_tensor(as_scene_representation(scene))
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "conditioned component mixture received a scene with incompatible shape"
            )
        coeffs = _weights_for(tensor.dtype, tensor.device).view(
            ncomp,
            *([1] * (tensor.ndim - 1)),
        )
        return torch.sum(coeffs * tensor, dim=0)

    def _embed(adjoint_tensor: Tensor) -> Tensor:
        adj = torch.as_tensor(adjoint_tensor)
        expected = tuple(state.shape[1:])
        if tuple(adj.shape) != expected:
            raise ValueError(
                "conditioned component mixture adjoint returned an incompatible tensor"
            )
        coeffs = torch.conj(_weights_for(adj.dtype, adj.device)).view(
            ncomp,
            *([1] * adj.ndim),
        )
        return _to(coeffs * adj.unsqueeze(0), dtype=state.dtype, device=state.device)

    def _wrap_bound(operator: BoundObservationOperator) -> BoundObservationOperator:
        def _apply_operator(
            scene: SceneRepresentation, target_spec: TargetRenderSpec
        ) -> Tensor:
            del target_spec
            return operator.apply(_mix(scene))

        def _apply_adjoint(tensor: Tensor) -> Tensor:
            return _embed(operator.adjoint(tensor))

        return BoundObservationOperator(
            operators=operator.operators,
            target=operator.target,
            apply_operator=_apply_operator,
            apply_adjoint=_apply_adjoint,
            apply_noise=operator.apply_noise,
        )

    if isinstance(base_operator, BoundObservationOperator):
        return _wrap_bound(base_operator)
    if isinstance(base_operator, BoundObservationOperatorSet):
        return BoundObservationOperatorSet(
            tuple(_wrap_bound(op) for op in base_operator.operators),
            exposure_chunk_size=base_operator.exposure_chunk_size,
        )
    raise TypeError(
        "base_operator must be a BoundObservationOperator or BoundObservationOperatorSet"
    )


def build_state_diagonal_operator(
    reference_state: Tensor,
    diagonal: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    diag = torch.as_tensor(diagonal)
    if tuple(diag.shape) != tuple(state.shape):
        raise ValueError("diagonal shape must match reference_state shape")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": int(state.numel()),
            "NAXIS2": 1,
        },
        shape=(1, int(state.numel())),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "state-space diagonal operator received a scene with incompatible shape"
            )
        return _to(diag, dtype=tensor.dtype, device=tensor.device) * tensor

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(mapped.shape) != tuple(state.shape):
            raise ValueError(
                "state-space diagonal adjoint received data with incompatible shape"
            )
        return _to(torch.conj(diag), dtype=mapped.dtype, device=mapped.device) * mapped

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_matrix_operator(
    reference_state: Tensor,
    matrix: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    matrix_t = torch.as_tensor(matrix)
    nstate = int(state.numel())
    if (
        matrix_t.ndim != 2
        or int(matrix_t.shape[0]) != nstate
        or int(matrix_t.shape[1]) != nstate
    ):
        raise ValueError("matrix operator requires a square (nstate, nstate) matrix")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": nstate,
            "NAXIS2": 1,
        },
        shape=(1, nstate),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "state-space matrix operator received a scene with incompatible shape"
            )
        flat = _to(_flatten_state(tensor), dtype=matrix_t.dtype, device=matrix_t.device)
        mapped = matrix_t @ flat
        return _reshape_state(
            _to(mapped, dtype=tensor.dtype, device=tensor.device), state
        )

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if tuple(mapped.shape) != tuple(state.shape):
            raise ValueError(
                "state-space matrix adjoint received data with incompatible shape"
            )
        flat = _to(_flatten_state(mapped), dtype=matrix_t.dtype, device=matrix_t.device)
        adj = torch.conj(matrix_t.transpose(-2, -1)) @ flat
        return _reshape_state(_to(adj, dtype=mapped.dtype, device=mapped.device), state)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_low_rank_operator(
    reference_state: Tensor,
    basis: Tensor,
    *,
    weights: Tensor | None = None,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    basis_t = torch.as_tensor(basis)
    nstate = int(state.numel())
    if basis_t.ndim != 2 or int(basis_t.shape[0]) != nstate:
        raise ValueError("low-rank basis must have shape (nstate, rank)")
    rank = int(basis_t.shape[1])
    if weights is None:
        weights_t = torch.ones(rank, dtype=basis_t.dtype, device=basis_t.device)
    else:
        weights_t = _as_tensor(weights, dtype=basis_t.dtype, device=basis_t.device)
        if weights_t.ndim != 1 or int(weights_t.shape[0]) != rank:
            raise ValueError("weights must have shape (rank,)")
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": rank,
            "NAXIS2": 1,
        },
        shape=(1, rank),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "state-space low-rank operator received a scene with incompatible shape"
            )
        flat = _to(_flatten_state(tensor), dtype=basis_t.dtype, device=basis_t.device)
        coeffs = torch.conj(basis_t).transpose(-2, -1) @ flat
        return _to(weights_t * coeffs, dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if mapped.ndim != 1 or int(mapped.shape[0]) != rank:
            raise ValueError(
                "state-space low-rank adjoint received data with incompatible shape"
            )
        coeffs = _to(mapped, dtype=basis_t.dtype, device=basis_t.device)
        flat = basis_t @ (torch.conj(weights_t) * coeffs)
        return _reshape_state(_to(flat, dtype=state.dtype, device=state.device), state)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )


def build_state_kronecker_operator(
    reference_state: Tensor,
    left_factor: Tensor,
    right_factor: Tensor,
    *,
    apply_noise: ApplyNoiseFn | None = None,
) -> BoundObservationOperator:
    state = torch.as_tensor(reference_state)
    if state.ndim != 2:
        raise ValueError("Kronecker state operator requires a 2D reference_state")
    left_t = torch.as_tensor(left_factor)
    right_t = torch.as_tensor(right_factor)
    rows, cols = int(state.shape[0]), int(state.shape[1])
    if left_t.ndim != 2 or int(left_t.shape[1]) != rows:
        raise ValueError("left_factor must have shape (out_rows, state_rows)")
    if right_t.ndim != 2 or int(right_t.shape[1]) != cols:
        raise ValueError("right_factor must have shape (out_cols, state_cols)")
    out_rows, out_cols = int(left_t.shape[0]), int(right_t.shape[0])
    target = TargetRenderSpec(
        header={
            "NAXIS": 2,
            "NAXIS1": out_cols,
            "NAXIS2": out_rows,
        },
        shape=(out_rows, out_cols),
    )

    def _apply_operator(
        scene: SceneRepresentation, target_spec: TargetRenderSpec
    ) -> Tensor:
        del target_spec
        tensor = _scene_state_tensor(scene)
        if tuple(tensor.shape) != tuple(state.shape):
            raise ValueError(
                "Kronecker state operator received a scene with incompatible shape"
            )
        tensor_t = _to(tensor, dtype=left_t.dtype, device=left_t.device)
        mapped = left_t @ tensor_t @ torch.conj(right_t).transpose(-2, -1)
        return _to(mapped, dtype=tensor.dtype, device=tensor.device)

    def _apply_adjoint(tensor: Tensor) -> Tensor:
        mapped = torch.as_tensor(tensor)
        if mapped.ndim != 2 or tuple(mapped.shape) != (
            out_rows,
            out_cols,
        ):
            raise ValueError("Kronecker adjoint received data with incompatible shape")
        mapped_t = _to(mapped, dtype=left_t.dtype, device=left_t.device)
        adj = torch.conj(left_t).transpose(-2, -1) @ mapped_t @ right_t
        return _to(adj, dtype=state.dtype, device=state.device)

    return BoundObservationOperator(
        operators=ObservationOperatorBundle(),
        target=target,
        apply_operator=_apply_operator,
        apply_adjoint=_apply_adjoint,
        apply_noise=apply_noise,
    )
