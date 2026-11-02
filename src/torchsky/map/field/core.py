"""Experimental continuous field models for multiwavelength map fitting."""

from __future__ import annotations

from typing import Sequence

import torch
import torch.nn as nn
from torch import Tensor

from ...device import resolve_device

from ..solvers import SurveyInstrumentComponentMixer

from .losses import _field_residual_loss
from .observation_ops import (
    observation_support_mask as _observation_support_mask,
)
from .rendering import (
    render_field_observation,
    render_field_samples,
)

from .models_component import FieldComponentDecomposition
from .models_encoder import (
    ConditionedSpectralField,
    HashGridConditionedField,
    HashGridMLPField,
    HashGridResMLPField,
    HashGridSpectralField,
    SkyEncoderBandpassField,
    SkyEncoderMLPField,
    SkyEncoderResMLPField,
    StrainerSkyEncoderMLPField,
)
from .models_foam import (
    DiffuseFoamHybridField,
    FoamInitializedINRField,
    SpectralPowerFoamField,
)
from .models_implicit import (
    BandpassConditionedField,
    ExplicitSpectralGridField,
    FourierMLPField,
    GaussianBasisField,
    LearnedSpectralBasisField,
    SineImplicitField,
    SpectralBasisField,
    WireImplicitField,
)
from .models_skylambda import (
    OperatorConstrainedFeatureField,
    SkyLambdaAtlasField,
    SkyLambdaFeatureAtlasField,
    SkyLambdaFeatureField,
    SkyLambdaKPlanesField,
    SkyLambdaPlanesField,
)

from .types import (
    FieldComponentAnchorTerm,
    FieldCentroidAnchorTerm,
    FieldDomain,
    FieldLossKind,
    FieldModelKind,
    FieldSpectralSampling,
    ForwardContext,
)

from .bandpass import bandpass_bounds as _bandpass_bounds  # noqa: F401 - re-exported
from ._core_helpers import (  # noqa: F401 - re-exported helper layer
    _STRAINER_TRUNK_FORMAT,
    _BaseFieldModel,
    _atlas_expert_weights,
    _atlas_local_coords,
    _fourier_features_vectorized,
    _fourier_features_with_curriculum,
    _infer_context_axis_labels,
    _observation_footprint_world_deg,
    _observation_without_background,
    _pixel_coordinate_grids,
    _point_bandpass_features,
    _reset_xavier_mlp,
    _resolve_anchor_sample_world_coordinates,
    _resolve_centroid_target_pixel_coordinates,
    _sample_feature_volume,
    _smoothness_probe_grid,
    estimate_field_domain,
    load_strainer_trunk_checkpoint,
    pretrain_strainer_trunk_synthetic,
    save_strainer_trunk_checkpoint,
)


def _render_field_component_anchor_target(
    term: FieldComponentAnchorTerm,
    *,
    num_spectral_samples: int,
    spectral_sampling: FieldSpectralSampling,
    spectral_sampling_grid_size: int | None,
    generator: torch.Generator | None,
    device: torch.device,
    dtype: torch.dtype,
) -> Tensor:
    if term.target_image is not None:
        return term.target_image.to(device=device, dtype=dtype)
    if term.target_values is not None:
        return term.target_values.to(device=device, dtype=dtype)
    assert term.reference_model is not None
    base_observation = _observation_without_background(term.observation)
    if isinstance(term.reference_model, FieldComponentDecomposition):
        if term.sample_ra_deg is not None:
            return term.reference_model.render_component_samples(
                base_observation,
                ra_deg=term.sample_ra_deg.to(device=device, dtype=torch.float32),
                dec_deg=term.sample_dec_deg.to(device=device, dtype=torch.float32),
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            ).to(device=device, dtype=dtype)
        return term.reference_model.render_component_observation(
            base_observation,
            roles=term.roles,
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            override_wavelengths=term.override_wavelengths,
            override_weights=term.override_bandpass_weights,
            forward_context=term.forward_context,
        ).to(device=device, dtype=dtype)
    if term.sample_ra_deg is not None:
        return render_field_samples(
            term.reference_model,
            base_observation,
            ra_deg=term.sample_ra_deg.to(device=device, dtype=torch.float32),
            dec_deg=term.sample_dec_deg.to(device=device, dtype=torch.float32),
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            override_wavelengths=term.override_wavelengths,
            override_weights=term.override_bandpass_weights,
            forward_context=term.forward_context,
        ).to(device=device, dtype=dtype)
    return render_field_observation(
        term.reference_model,
        base_observation,
        num_spectral_samples=num_spectral_samples,
        spectral_sampling=spectral_sampling,
        spectral_sampling_grid_size=spectral_sampling_grid_size,
        generator=generator,
        override_wavelengths=term.override_wavelengths,
        override_weights=term.override_bandpass_weights,
        forward_context=term.forward_context,
    ).to(device=device, dtype=dtype)


def field_component_anchor_loss(
    model: nn.Module,
    component_anchor_terms: Sequence[FieldComponentAnchorTerm],
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
    loss_kind: FieldLossKind = "mse",
    charbonnier_eps: float = 1.0e-3,
    huber_delta: float = 5.0e-2,
) -> Tensor:
    """Weighted anchor loss for selected field decomposition roles."""
    if loss_kind == "gaussian_nll":
        raise ValueError("field component anchors do not support gaussian_nll")
    total = None
    for term in component_anchor_terms:
        if term.loss_weight == 0.0:
            continue
        base_observation = _observation_without_background(term.observation)
        if (
            term.sample_ra_deg is not None
            or term.sample_dec_deg is not None
            or term.sample_x_pix is not None
            or term.sample_y_pix is not None
        ):
            ra_anchor, dec_anchor = _resolve_anchor_sample_world_coordinates(
                term,
                device=resolve_device(None, model),
            )
            if isinstance(model, FieldComponentDecomposition):
                prediction = model.render_component_samples(
                    base_observation,
                    ra_deg=ra_anchor,
                    dec_deg=dec_anchor,
                    roles=term.roles,
                    num_spectral_samples=num_spectral_samples,
                    spectral_sampling=spectral_sampling,
                    spectral_sampling_grid_size=spectral_sampling_grid_size,
                    generator=generator,
                    override_wavelengths=term.override_wavelengths,
                    override_weights=term.override_bandpass_weights,
                    forward_context=term.forward_context,
                )
            else:
                if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                    raise ValueError(
                        "component anchor role selection requires FieldComponentDecomposition"
                    )
                prediction = render_field_samples(
                    model,
                    base_observation,
                    ra_deg=ra_anchor,
                    dec_deg=dec_anchor,
                    num_spectral_samples=num_spectral_samples,
                    spectral_sampling=spectral_sampling,
                    spectral_sampling_grid_size=spectral_sampling_grid_size,
                    generator=generator,
                    override_wavelengths=term.override_wavelengths,
                    override_weights=term.override_bandpass_weights,
                    forward_context=term.forward_context,
                )
        elif isinstance(model, FieldComponentDecomposition):
            prediction = model.render_component_observation(
                base_observation,
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            )
        else:
            if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                raise ValueError(
                    "component anchor role selection requires FieldComponentDecomposition"
                )
            prediction = render_field_observation(
                model,
                base_observation,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                override_wavelengths=term.override_wavelengths,
                override_weights=term.override_bandpass_weights,
                forward_context=term.forward_context,
            )
        target = _render_field_component_anchor_target(
            term,
            num_spectral_samples=num_spectral_samples,
            spectral_sampling=spectral_sampling,
            spectral_sampling_grid_size=spectral_sampling_grid_size,
            generator=generator,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        if (
            term.sample_ra_deg is not None
            or term.sample_dec_deg is not None
            or term.sample_x_pix is not None
            or term.sample_y_pix is not None
        ):
            residual = prediction - target
            weight = (
                term.sample_weight.to(device=prediction.device, dtype=prediction.dtype)
                if term.sample_weight is not None
                else torch.ones_like(prediction)
            )
        else:
            valid = _observation_support_mask(term.observation).to(
                device=prediction.device
            )
            target = torch.where(valid, target, torch.zeros_like(target))
            residual = torch.where(
                valid, prediction - target, torch.zeros_like(prediction)
            )
            weight = (
                term.observation.weight.to(
                    device=prediction.device, dtype=prediction.dtype
                )
                if term.observation.weight is not None
                else torch.ones_like(prediction)
            )
            weight = torch.where(valid, weight, torch.zeros_like(weight))
        weighted_loss = torch.sum(
            weight
            * _field_residual_loss(
                residual,
                loss_kind=loss_kind,
                charbonnier_eps=charbonnier_eps,
                huber_delta=huber_delta,
                logvar=None,
            )
        )
        term_loss = float(term.loss_weight) * (
            weighted_loss / torch.clamp(weight.sum(), min=1.0e-8)
        )
        total = term_loss if total is None else total + term_loss
    if total is None:
        return torch.zeros(
            (),
            device=resolve_device(None, model),
            dtype=torch.float32,
        )
    return total


def field_component_centroid_anchor_loss(
    model: nn.Module,
    centroid_anchor_terms: Sequence[FieldCentroidAnchorTerm],
    *,
    num_spectral_samples: int = 9,
    spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    spectral_sampling_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> Tensor:
    """Weighted centroid loss for geometry-aware role anchors."""
    total = None
    for term in centroid_anchor_terms:
        if term.loss_weight == 0.0:
            continue
        base_observation = _observation_without_background(term.observation)
        if isinstance(model, FieldComponentDecomposition):
            prediction = model.render_component_observation(
                base_observation,
                roles=term.roles,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=term.forward_context,
            )
        else:
            if term.roles != SurveyInstrumentComponentMixer._VALID_ROLES:
                raise ValueError(
                    "centroid anchor role selection requires FieldComponentDecomposition"
                )
            prediction = render_field_observation(
                model,
                base_observation,
                num_spectral_samples=num_spectral_samples,
                spectral_sampling=spectral_sampling,
                spectral_sampling_grid_size=spectral_sampling_grid_size,
                generator=generator,
                forward_context=term.forward_context,
            )
        target_x, target_y = _resolve_centroid_target_pixel_coordinates(
            term,
            device=prediction.device,
            dtype=prediction.dtype,
        )
        x_grid, y_grid = _pixel_coordinate_grids(
            tuple(prediction.shape),
            device=prediction.device,
            dtype=prediction.dtype,
        )
        sigma_pix = torch.tensor(
            float(term.window_sigma_pix),
            device=prediction.device,
            dtype=prediction.dtype,
        )
        valid = _observation_support_mask(term.observation).to(device=prediction.device)
        prediction_scale = torch.clamp(prediction.detach().abs().amax(), min=1.0)
        logits = 16.0 * (prediction / prediction_scale)
        logits = torch.where(valid, logits, torch.full_like(logits, -1.0e9))
        energy = torch.softmax(logits.reshape(-1), dim=0).reshape_as(prediction)
        centroid_x = (energy * x_grid).sum()
        centroid_y = (energy * y_grid).sum()
        term_loss = float(term.loss_weight) * (
            ((centroid_x - target_x) / sigma_pix).square()
            + ((centroid_y - target_y) / sigma_pix).square()
        )
        total = term_loss if total is None else total + term_loss
    if total is None:
        return torch.zeros(
            (),
            device=resolve_device(None, model),
            dtype=torch.float32,
        )
    return total


def make_field_component_decomposition(
    *,
    component_roles: Sequence[str],
    domain: FieldDomain | None = None,
    component_models: Sequence[nn.Module] | None = None,
    component_model_kind: FieldModelKind = "sine_implicit",
    component_model_kwargs: dict[str, object] | None = None,
    surveys: Sequence[str] | None = None,
    instruments: Sequence[str] | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | None = None,
) -> FieldComponentDecomposition:
    """Build an explicit field decomposition model from component specs or modules."""
    roles = tuple(str(role) for role in component_roles)
    if not roles:
        raise ValueError("component_roles must be non-empty")
    if component_models is not None:
        models = tuple(component_models)
        if len(models) != len(roles):
            raise ValueError("component_models length must match component_roles")
    else:
        if domain is None:
            raise ValueError(
                "domain is required when component_models are not provided"
            )
        model_kwargs = dict(component_model_kwargs or {})
        models = tuple(
            make_field_model(component_model_kind, domain, **model_kwargs)
            for _ in roles
        )
    inferred_surveys, inferred_instruments = _infer_context_axis_labels(
        observation_forward_contexts
    )
    return FieldComponentDecomposition(
        models,
        component_roles=roles,
        surveys=tuple(str(name) for name in surveys)
        if surveys is not None
        else inferred_surveys,
        instruments=tuple(str(name) for name in instruments)
        if instruments is not None
        else inferred_instruments,
        dtype=dtype,
        device=device,
    )


STRAINER_TRUNK_CHECKPOINT_FORMAT = _STRAINER_TRUNK_FORMAT


def make_field_model(
    kind: FieldModelKind,
    domain: FieldDomain,
    **kwargs: object,
) -> nn.Module:
    """Construct a field model of the requested kind on a fixed domain.

    Args:
        kind: Registered field-model kind.
        domain: Spatial and spectral normalization domain.
        **kwargs: Model-specific constructor options.

    Returns:
        An untrained PyTorch field module.

    Examples:
        >>> from torchsky.field import Domain, model
        >>> domain = Domain(0.0, 0.0, 1.0, 5000.0, 1000.0)
        >>> field = model("sine_implicit", domain)
        >>> field.domain is domain
        True
    """
    if kind == "spectral_power_foam":
        return SpectralPowerFoamField(domain, **kwargs)
    if kind == "foam_initialized_inr":
        return FoamInitializedINRField(domain, **kwargs)
    if kind == "diffuse_foam_hybrid":
        return DiffuseFoamHybridField(domain, **kwargs)
    if kind == "sine_implicit":
        return SineImplicitField(domain, **kwargs)
    if kind == "wire_implicit":
        return WireImplicitField(domain, **kwargs)
    if kind == "explicit_spectral_grid_field":
        return ExplicitSpectralGridField(domain, **kwargs)
    if kind == "sasnet":
        from .frequency_adaptive import SASNetField

        return SASNetField(domain, **kwargs)
    if kind == "bacon":
        from .frequency_adaptive import BaconField

        return BaconField(domain, **kwargs)
    if kind == "flair_bla":
        from .frequency_adaptive import FlairField

        return FlairField(domain, **kwargs)
    if kind == "gaussian_basis":
        return GaussianBasisField(domain, **kwargs)
    if kind == "fourier_mlp":
        return FourierMLPField(domain, **kwargs)
    if kind == "spectral_basis_field":
        return SpectralBasisField(domain, **kwargs)
    if kind == "bandpass_conditioned_field":
        return BandpassConditionedField(domain, **kwargs)
    if kind == "learned_spectral_basis_field":
        return LearnedSpectralBasisField(domain, **kwargs)
    if kind == "sky_encoder_bandpass_field":
        return SkyEncoderBandpassField(domain, **kwargs)
    if kind == "sky_lambda_planes_field":
        return SkyLambdaPlanesField(domain, **kwargs)
    if kind == "sky_lambda_kplanes_field":
        return SkyLambdaKPlanesField(domain, **kwargs)
    if kind == "sky_lambda_feature_field":
        return SkyLambdaFeatureField(domain, **kwargs)
    if kind == "operator_constrained_feature_field":
        return OperatorConstrainedFeatureField(domain, **kwargs)
    if kind == "sky_lambda_feature_field_hetero":
        return SkyLambdaFeatureField(domain, predict_variance=True, **kwargs)
    if kind == "sky_lambda_feature_atlas_field":
        return SkyLambdaFeatureAtlasField(domain, **kwargs)
    if kind == "sky_lambda_atlas_field":
        return SkyLambdaAtlasField(domain, **kwargs)
    if kind == "conditioned_spectral_field":
        return ConditionedSpectralField(domain, **kwargs)
    if kind == "hashgrid_conditioned_field":
        return HashGridConditionedField(domain, **kwargs)
    if kind == "sky_encoder_mlp":
        return SkyEncoderMLPField(domain, **kwargs)
    if kind == "strainer_sky_encoder_mlp":
        return StrainerSkyEncoderMLPField(domain, **kwargs)
    if kind == "sky_encoder_resmlp":
        return SkyEncoderResMLPField(domain, **kwargs)
    if kind == "hashgrid_mlp":
        return HashGridMLPField(domain, **kwargs)
    if kind == "hashgrid_spectral_field":
        return HashGridSpectralField(domain, **kwargs)
    if kind == "hashgrid_resmlp":
        return HashGridResMLPField(domain, **kwargs)
    raise ValueError(f"unknown field model kind {kind!r}")
