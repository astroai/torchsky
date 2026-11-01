"""Neural field-model classes for :mod:`torchsky.map.field`."""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ._core_helpers import (
    _BaseFieldModel,
    _observation_without_background,
)
from .bandpass import (
    bandpass_mc_samples as _bandpass_mc_samples,
    bandpass_quadrature as _bandpass_quadrature,
    validate_override_quadrature as _validate_override_quadrature,
)
from .forwarding import call_scalar_field_forward as _call_scalar_field_forward
from .observation_ops import coordinate_grid as _coordinate_grid
from .rendering import render_field_observation
from .types import (
    FieldDomain,
    FieldObservation,
    FieldSpectralSampling,
    ForwardContext,
)
from ..operators.bundle import field_observation_operator_bundle
from ..operators.psf import apply_psf_spec
from ..solvers import (
    ComponentDecompositionDiagnostics,
    SurveyInstrumentComponentMixer,
    _relative_norm,
    _subspace_leakage,
)
from ...device import default_compute_device


if TYPE_CHECKING:
    pass

class FieldComponentDecomposition(_BaseFieldModel):
    """Explicit shared/private decomposition over multiple field models."""

    def __init__(
        self,
        component_models: Sequence[nn.Module],
        *,
        component_roles: Sequence[str],
        surveys: Sequence[str] = (),
        instruments: Sequence[str] = (),
        dtype: torch.dtype = torch.float32,
        device: torch.device | None = None,
    ) -> None:
        models = tuple(component_models)
        roles = tuple(str(role) for role in component_roles)
        if not models:
            raise ValueError("component_models must be non-empty")
        if len(models) != len(roles):
            raise ValueError("component_roles length must match component_models")
        domain = getattr(models[0], "domain", None)
        if not isinstance(domain, FieldDomain):
            raise ValueError(
                "component_models[0] must expose a FieldDomain via .domain"
            )
        for idx, model in enumerate(models[1:], start=1):
            if getattr(model, "domain", None) != domain:
                raise ValueError(
                    f"component_models[{idx}] domain must match component_models[0].domain"
                )
        super().__init__(domain)
        self.component_roles = roles
        self.component_models = nn.ModuleList(models)
        self.component_gains = nn.Parameter(
            torch.ones(len(self.component_roles), dtype=dtype, device=device)
        )
        self.mixer = SurveyInstrumentComponentMixer(
            component_roles=self.component_roles,
            surveys=surveys,
            instruments=instruments,
            dtype=dtype,
            device=device,
        )

    def _role_indices(self, roles: str | Sequence[str]) -> tuple[int, ...]:
        role_names = (
            (roles,) if isinstance(roles, str) else tuple(str(role) for role in roles)
        )
        valid_roles = set(SurveyInstrumentComponentMixer._VALID_ROLES)
        unknown = tuple(role for role in role_names if role not in valid_roles)
        if unknown:
            raise ValueError(f"unsupported component roles: {unknown!r}")
        role_set = set(role_names)
        return tuple(
            idx for idx, role in enumerate(self.component_roles) if role in role_set
        )

    def _context_labels(
        self,
        forward_context: ForwardContext | None,
    ) -> tuple[str | None, str | None]:
        if forward_context is None:
            return None, None
        return forward_context.survey, forward_context.instrument

    def fix_component_scale_gauge_(
        self,
        *,
        target_gain: float = 1.0,
        min_gain: float = 1.0e-12,
    ) -> Tensor:
        """Canonicalize explicit component gains without changing rendered predictions."""
        if target_gain <= 0.0:
            raise ValueError("target_gain must be positive")
        if min_gain <= 0.0:
            raise ValueError("min_gain must be positive")
        gains = self.component_gains.detach()
        safe = torch.abs(gains) > float(min_gain)
        scale = torch.where(
            safe,
            gains / float(target_gain),
            torch.ones_like(gains),
        )
        with torch.no_grad():
            self.component_gains.div_(scale)
        nonzero_scale = scale[safe]
        if int(nonzero_scale.numel()) > 0:
            full_scale = torch.where(safe, scale, torch.ones_like(scale))
            self.mixer.scale_component_columns_(full_scale)
        return self.component_gains.detach().clone()

    def component_prediction(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        roles: str | Sequence[str],
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        selected = self._role_indices(roles)
        if not selected:
            if aggregate:
                return torch.zeros_like(ra_deg)
            return torch.zeros(
                (0, *ra_deg.shape),
                dtype=ra_deg.dtype,
                device=ra_deg.device,
            )
        survey, instrument = self._context_labels(forward_context)
        weights = self.mixer(survey=survey, instrument=instrument)
        outputs = []
        for idx in selected:
            plane = _call_scalar_field_forward(
                self.component_models[idx],
                ra_deg,
                dec_deg,
                wavelength_angstrom,
                forward_context=forward_context,
            )
            gain = self.component_gains[idx].to(dtype=plane.dtype, device=plane.device)
            outputs.append(
                weights[idx].to(dtype=plane.dtype, device=plane.device) * gain * plane
            )
        stacked = torch.stack(outputs, dim=0)
        if aggregate:
            return stacked.sum(dim=0)
        return stacked

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
        *,
        forward_context: ForwardContext | None = None,
    ) -> Tensor:
        return self.component_prediction(
            ra_deg,
            dec_deg,
            wavelength_angstrom,
            roles=SurveyInstrumentComponentMixer._VALID_ROLES,
            forward_context=forward_context,
            aggregate=True,
        )

    def render_component_observation(
        self,
        observation: FieldObservation,
        *,
        roles: str | Sequence[str],
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        override_wavelengths: Tensor | None = None,
        override_weights: Tensor | None = None,
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        base_observation = _observation_without_background(observation)
        image = base_observation.image
        if image.ndim != 2:
            raise ValueError("FieldObservation.image must be 2D")
        device = next((p.device for p in self.parameters()), default_compute_device())
        ra_deg, dec_deg = _coordinate_grid(
            base_observation,
            shape=tuple(image.shape),
            device=device,
        )
        if override_wavelengths is not None or override_weights is not None:
            if override_wavelengths is None or override_weights is None:
                raise ValueError(
                    "override_wavelengths and override_weights must be provided together"
                )
            wavelengths, weights = _validate_override_quadrature(
                override_wavelengths,
                override_weights,
                device=device,
                dtype=torch.float32,
            )
        elif spectral_sampling == "deterministic_quadrature":
            wavelengths, weights = _bandpass_quadrature(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=device,
            )
        elif spectral_sampling == "bandpass_mc":
            wavelengths, weights = _bandpass_mc_samples(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=device,
                proposal_grid_size=spectral_sampling_grid_size,
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
        selected = self._role_indices(roles)
        if not selected:
            shape = (0, *image.shape) if not aggregate else tuple(image.shape)
            return torch.zeros(shape, dtype=image.dtype, device=device)
        rendered_shape = (
            tuple(image.shape) if aggregate else (len(selected), *image.shape)
        )
        rendered = torch.zeros(rendered_shape, device=device, dtype=torch.float32)
        ra32 = ra_deg.to(dtype=torch.float32)
        dec32 = dec_deg.to(dtype=torch.float32)
        for i, wavelength in enumerate(wavelengths):
            plane = self.component_prediction(
                ra32,
                dec32,
                wavelength,
                roles=roles,
                forward_context=forward_context,
                aggregate=aggregate,
            )
            weight = weights[i].to(dtype=plane.dtype, device=plane.device)
            if aggregate:
                rendered = rendered + plane * weight
            else:
                rendered = rendered + plane * weight.view(1, 1, 1)
        if (
            base_observation.psf_spec is not None
            and base_observation.psf_spec.mode != "none"
        ):
            if aggregate:
                rendered = apply_psf_spec(rendered, base_observation.psf_spec)
            else:
                rendered = torch.stack(
                    [
                        apply_psf_spec(rendered[idx], base_observation.psf_spec)
                        for idx in range(int(rendered.shape[0]))
                    ],
                    dim=0,
                )
        return rendered.to(dtype=image.dtype)

    def render_component_samples(
        self,
        observation: FieldObservation,
        *,
        ra_deg: Tensor,
        dec_deg: Tensor,
        roles: str | Sequence[str],
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        override_wavelengths: Tensor | None = None,
        override_weights: Tensor | None = None,
        forward_context: ForwardContext | None = None,
        aggregate: bool = True,
    ) -> Tensor:
        base_observation = _observation_without_background(observation)
        operator_bundle = field_observation_operator_bundle(base_observation)
        if operator_bundle.psf is not None:
            raise ValueError("component sample rendering does not support PSF blurring")
        if override_wavelengths is not None or override_weights is not None:
            if override_wavelengths is None or override_weights is None:
                raise ValueError(
                    "override_wavelengths and override_weights must be provided together"
                )
            wavelengths, weights = _validate_override_quadrature(
                override_wavelengths,
                override_weights,
                device=ra_deg.device,
                dtype=torch.float32,
            )
        elif spectral_sampling == "deterministic_quadrature":
            wavelengths, weights = _bandpass_quadrature(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=ra_deg.device,
            )
        elif spectral_sampling == "bandpass_mc":
            wavelengths, weights = _bandpass_mc_samples(
                base_observation,
                num_spectral_samples=num_spectral_samples,
                device=ra_deg.device,
                proposal_grid_size=spectral_sampling_grid_size,
                generator=generator,
            )
        else:
            raise ValueError(f"unsupported spectral_sampling {spectral_sampling!r}")
        selected = self._role_indices(roles)
        if not selected:
            shape = (0, *ra_deg.shape) if not aggregate else tuple(ra_deg.shape)
            return torch.zeros(shape, dtype=ra_deg.dtype, device=ra_deg.device)
        rendered_shape = (
            tuple(ra_deg.shape) if aggregate else (len(selected), *ra_deg.shape)
        )
        rendered = torch.zeros(
            rendered_shape, device=ra_deg.device, dtype=torch.float32
        )
        ra32 = ra_deg.to(dtype=torch.float32)
        dec32 = dec_deg.to(dtype=torch.float32)
        for i, wavelength in enumerate(wavelengths):
            sample = self.component_prediction(
                ra32,
                dec32,
                wavelength,
                roles=roles,
                forward_context=forward_context,
                aggregate=aggregate,
            )
            weight = weights[i].to(dtype=sample.dtype, device=sample.device)
            if aggregate:
                rendered = rendered + sample * weight
            else:
                rendered = rendered + sample * weight.view(
                    1, *([1] * (sample.ndim - 1))
                )
        return rendered

    def decomposition_diagnostics(
        self,
        observations: Sequence[FieldObservation],
        truth_model: "FieldComponentDecomposition",
        *,
        observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
        num_spectral_samples: int = 9,
        spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
        spectral_sampling_grid_size: int | None = None,
        generator: torch.Generator | None = None,
        calibration_anchor_model: nn.Module | None = None,
        calibration_anchor_roles: str | Sequence[str] = "shared",
    ) -> ComponentDecompositionDiagnostics:
        if not observations:
            raise ValueError("observations must be non-empty")
        if observation_forward_contexts is not None:
            if len(observation_forward_contexts) != len(observations):
                raise ValueError(
                    "observation_forward_contexts must match observations length"
                )
            contexts = tuple(observation_forward_contexts)
        else:
            contexts = tuple(None for _ in observations)
        private_basis_count = len(
            truth_model._role_indices(("survey", "instrument", "interaction"))
        )
        private_basis_chunks: list[list[Tensor]] = (
            [[] for _ in range(private_basis_count)] if private_basis_count > 0 else []
        )
        shared_est_chunks: list[Tensor] = []
        private_est_chunks: list[Tensor] = []
        shared_truth_chunks: list[Tensor] = []
        private_truth_chunks: list[Tensor] = []
        calibration_anchor_chunks: list[Tensor] = []

        for observation, forward_context in zip(observations, contexts, strict=True):
            render_kwargs = dict(
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=forward_context,
            )
            shared_est = self.render_component_observation(
                observation,
                roles="shared",
                **render_kwargs,
            )
            private_est = self.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                **render_kwargs,
            )
            shared_truth = truth_model.render_component_observation(
                observation,
                roles="shared",
                **render_kwargs,
            )
            private_truth = truth_model.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                **render_kwargs,
            )
            private_truth_basis = truth_model.render_component_observation(
                observation,
                roles=("survey", "instrument", "interaction"),
                aggregate=False,
                **render_kwargs,
            )
            shared_est_chunks.append(shared_est.reshape(-1))
            private_est_chunks.append(private_est.reshape(-1))
            # The truth/anchor models may live on a different device than the
            # (possibly device-moved) fitted model; compare on the estimate's device.
            est_device = shared_est.device
            shared_truth_chunks.append(shared_truth.reshape(-1).to(est_device))
            private_truth_chunks.append(private_truth.reshape(-1).to(est_device))
            for basis_idx in range(int(private_truth_basis.shape[0])):
                private_basis_chunks[basis_idx].append(
                    private_truth_basis[basis_idx].reshape(-1).to(est_device)
                )
            if calibration_anchor_model is not None:
                base_observation = _observation_without_background(observation)
                if isinstance(calibration_anchor_model, FieldComponentDecomposition):
                    anchor = calibration_anchor_model.render_component_observation(
                        base_observation,
                        roles=calibration_anchor_roles,
                        **render_kwargs,
                    )
                else:
                    anchor = render_field_observation(
                        calibration_anchor_model,
                        base_observation,
                        **render_kwargs,
                    )
                calibration_anchor_chunks.append(anchor.reshape(-1).to(est_device))

        shared_est_vec = torch.cat(shared_est_chunks, dim=0)
        private_est_vec = torch.cat(private_est_chunks, dim=0)
        shared_truth_vec = torch.cat(shared_truth_chunks, dim=0)
        private_truth_vec = torch.cat(private_truth_chunks, dim=0)
        if private_basis_chunks:
            private_truth_basis_vec = torch.stack(
                [torch.cat(chunks, dim=0) for chunks in private_basis_chunks],
                dim=0,
            )
        else:
            private_truth_basis_vec = torch.zeros(
                (0, int(shared_truth_vec.numel())),
                dtype=shared_truth_vec.dtype,
                device=shared_truth_vec.device,
            )
        calibration_transfer_stability = None
        if calibration_anchor_chunks:
            calibration_anchor_vec = torch.cat(calibration_anchor_chunks, dim=0)
            calibration_transfer_stability = _relative_norm(
                shared_est_vec - calibration_anchor_vec,
                calibration_anchor_vec,
            )
        return ComponentDecompositionDiagnostics(
            shared_to_private_leakage=_subspace_leakage(
                shared_est_vec - shared_truth_vec,
                private_truth_basis_vec,
                private_truth_vec,
            ),
            private_to_shared_leakage=_subspace_leakage(
                private_est_vec - private_truth_vec,
                shared_truth_vec.unsqueeze(0),
                shared_truth_vec,
            ),
            shared_relative_error=_relative_norm(
                shared_est_vec - shared_truth_vec,
                shared_truth_vec,
            ),
            private_relative_error=_relative_norm(
                private_est_vec - private_truth_vec,
                private_truth_vec,
            ),
            calibration_transfer_stability=calibration_transfer_stability,
        )
