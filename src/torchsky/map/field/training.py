"""Training loops for continuous field models."""

from __future__ import annotations

import copy
from contextlib import nullcontext
import math
import os
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Any, cast

import torch
import torch.nn as nn
from torch import Tensor

from ...device import _as_tensor, resolve_device, resolve_fit_device, _to

from ..exposure_pose import ExposurePose, create_exposure_poses
from ..operators.bundle import (
    ObservationOperatorBundle,
    field_observation_operator_bundle,
    merge_operator_bundles,
)
from ..operators.calibration import OperatorCalibrationTerm
from ..operators.noise import StructuredDetectorNoise, StructuredNoiseOp

from .bandpass import (
    bandpass_bounds as _bandpass_bounds,
    mixture_mc_samples as _mixture_mc_samples,
)
from .core import (
    _BaseFieldModel,
    BandpassConditionedField,
    DiffuseFoamHybridField,
    FieldComponentDecomposition,
    FourierMLPField,
    HashGridConditionedField,
    HashGridMLPField,
    HashGridResMLPField,
    SineImplicitField,
    SkyEncoderBandpassField,
    SkyEncoderMLPField,
    SkyEncoderResMLPField,
    StrainerSkyEncoderMLPField,
    WireImplicitField,
    _smoothness_probe_grid,
    estimate_field_domain,
    field_component_anchor_loss,
    field_component_centroid_anchor_loss,
    make_field_component_decomposition,
)
from .losses import _field_gradient_residual_weighted_loss, _field_residual_loss
from .noise import _synthetic_noise_target
from .observation_ops import (
    avg_pool_map as _avg_pool_map,
    crop_field_observation_for_training as _crop_field_observation_for_training,
    crop_field_observation_to_shared_sky_patch as _crop_field_observation_to_shared_sky_patch,
    observation_support_mask as _observation_support_mask,
    observation_weight_sum_scalar as _observation_weight_sum_scalar,
    sample_observation_indices as _sample_observation_indices,
    sample_observation_pixels as _sample_observation_pixels,
    sample_patch_origin as _sample_patch_origin,
)
from .quality import _aggregate_quality_metrics, field_image_quality_metrics
from .rendering import (
    field_model_has_predictive_distribution,
    iter_field_observation_tiles,
    render_field_observation,
    render_field_observation_distribution,
    render_field_samples,
    render_field_samples_distribution,
)
from .types import (
    FieldAMPDtype,
    FieldCentroidAnchorTerm,
    FieldComponentAnchorTerm,
    FieldCurriculum,
    FieldDomain,
    FieldFitResult,
    FieldLossKind,
    FieldLossReduction,
    FieldLRSchedule,
    FieldModelKind,
    FieldNoisePretrainKind,
    FieldObservation,
    FieldObservationSampling,
    FieldSpatialSampling,
    FieldSpectralSampling,
    ForwardContext,
)

from ._training_losses import (  # noqa: F401
    _compute_full_observation_losses,
    _mixture_render_kwargs,
    _structured_noise_loss,
)


def _field_noise_pretrain_head_modules(
    model: nn.Module,
) -> tuple[nn.Module, ...] | None:
    if isinstance(
        model,
        (
            SineImplicitField,
            WireImplicitField,
            FourierMLPField,
            BandpassConditionedField,
            SkyEncoderBandpassField,
            SkyEncoderMLPField,
            SkyEncoderResMLPField,
            StrainerSkyEncoderMLPField,
            HashGridMLPField,
            HashGridResMLPField,
            HashGridConditionedField,
        ),
    ):
        output = getattr(model, "output", None)
        if isinstance(output, nn.Module):
            return (output,)
    return None


def _noise_pretrain_reference_wavelength(observation: FieldObservation) -> float:
    if observation.wavelength is not None:
        return float(observation.wavelength)
    if observation.transmission_center is not None:
        return float(observation.transmission_center)
    lower, upper = _bandpass_bounds(observation)
    return 0.5 * (lower + upper)


def _noise_pretrain_field_model(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    *,
    noise_kind: FieldNoisePretrainKind,
    steps: int,
    lr: float,
    observations_per_step: int,
    pixel_samples_per_observation: int,
    num_terms: int,
    max_frequency: float,
) -> None:
    if steps <= 0 or noise_kind == "none":
        return
    head_modules = _field_noise_pretrain_head_modules(model)
    if head_modules is None:
        raise ValueError(
            "noise pretraining is only supported for MLP-style field models with a separable output head"
        )
    preserved_heads = [copy.deepcopy(module.state_dict()) for module in head_modules]
    optimizer = torch.optim.Adam(model.parameters(), lr=float(lr))
    sampling_generator = torch.Generator(device="cpu")
    sampling_generator.manual_seed(314159)
    model.train()
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        selected_indices = _sample_observation_indices(
            observations,
            observations_per_step=max(int(observations_per_step), 1),
            observation_sampling="band_balanced",
            generator=sampling_generator,
        )
        total_loss = None
        for index in selected_indices:
            observation = observations[index]
            sampled = _sample_observation_pixels(
                observation,
                num_samples=max(int(pixel_samples_per_observation), 1),
                generator=sampling_generator,
                device=resolve_device(None, model),
            )
            if sampled is None:
                continue
            sample_ra_deg, sample_dec_deg, _image, sample_weight, _support_count, _ = (
                sampled
            )
            sample_wavelength = torch.full_like(
                sample_ra_deg,
                _noise_pretrain_reference_wavelength(observation),
            )
            assert isinstance(model, _BaseFieldModel)
            coords = model._unit_box_coords(
                sample_ra_deg.to(dtype=torch.float64),
                sample_dec_deg.to(dtype=torch.float64),
                sample_wavelength.to(dtype=torch.float64),
            )
            target = _synthetic_noise_target(
                coords,
                noise_kind=noise_kind,
                num_terms=num_terms,
                max_frequency=max_frequency,
                generator=sampling_generator,
            ).to(device=sample_ra_deg.device)
            pred = model(
                sample_ra_deg,
                sample_dec_deg,
                sample_wavelength,
            )
            weight = _to(sample_weight, device=pred.device, dtype=pred.dtype)
            # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
            diff_pt = pred - target.to(dtype=pred.dtype)
            loss = torch.sum(weight * (diff_pt * diff_pt))
            loss = loss / torch.clamp(weight.sum(), min=1.0e-8)
            total_loss = loss if total_loss is None else total_loss + loss
        if total_loss is None:
            continue
        total_loss.backward()
        optimizer.step()
    for module, state_dict in zip(head_modules, preserved_heads, strict=True):
        module.load_state_dict(state_dict)


def fit_field_component_decomposition(
    observations: Sequence[FieldObservation],
    *,
    component_roles: Sequence[str],
    model: FieldComponentDecomposition | None = None,
    domain: FieldDomain | None = None,
    component_models: Sequence[nn.Module] | None = None,
    component_model_kind: FieldModelKind = "sine_implicit",
    component_model_kwargs: dict[str, object] | None = None,
    surveys: Sequence[str] | None = None,
    instruments: Sequence[str] | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
    **fit_kwargs: Any,
) -> FieldFitResult:
    """Construct and fit a field decomposition model on context-labeled observations."""
    if model is not None:
        if component_models is not None:
            raise ValueError(
                "component_models cannot be provided when model is provided"
            )
        if domain is not None:
            raise ValueError("domain cannot be provided when model is provided")
        if component_model_kwargs is not None:
            raise ValueError(
                "component_model_kwargs cannot be provided when model is provided"
            )
    else:
        build_domain = (
            domain if domain is not None else estimate_field_domain(observations)
        )
        model = make_field_component_decomposition(
            component_roles=component_roles,
            domain=build_domain,
            component_models=component_models,
            component_model_kind=component_model_kind,
            component_model_kwargs=component_model_kwargs,
            surveys=surveys,
            instruments=instruments,
            observation_forward_contexts=observation_forward_contexts,
            dtype=dtype,
            device=torch.device(device) if isinstance(device, str) else device,
        )
    return fit_field_model(
        observations,
        model,
        device=device,
        observation_forward_contexts=observation_forward_contexts,
        **fit_kwargs,
    )


def compute_field_fit_diagnostics(
    model: nn.Module,
    observations: Sequence[FieldObservation],
    obs_forward_ctx: Sequence[ForwardContext | None],
    obs_exposure_poses: Sequence[ExposurePose | None],
    obs_operator_bundles: Sequence[ObservationOperatorBundle],
    num_spectral_samples: int,
    fit_duration_sec: float,
    steps: int,
    total_pixels: int,
    max_mem: int,
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
) -> dict[str, Any]:
    device = next((p.device for p in model.parameters()), torch.device("cpu"))
    domain = getattr(model, "domain", None)

    grad_corr = 0.0
    leakage_index = 0.0

    if domain is not None:
        try:
            n_samples = 512
            gen = torch.Generator(device=device)
            gen.manual_seed(1234)

            dra = (
                torch.rand(
                    n_samples,
                    generator=gen,
                    dtype=torch.float32 if device.type == "mps" else torch.float64,
                    device=device,
                )
                - 0.5
            ) * (2.0 * float(domain.spatial_scale_deg))
            ddec = (
                torch.rand(
                    n_samples,
                    generator=gen,
                    dtype=torch.float32 if device.type == "mps" else torch.float64,
                    device=device,
                )
                - 0.5
            ) * (2.0 * float(domain.spatial_scale_deg))
            ra = (
                torch.remainder(float(domain.center_ra_deg) + dra, 360.0)
                .to(dtype=torch.float32)
                .requires_grad_(True)
            )
            dec = (
                torch.clamp(
                    float(domain.center_dec_deg) + ddec,
                    min=-90.0,
                    max=90.0,
                )
                .to(dtype=torch.float32)
                .requires_grad_(True)
            )

            lam_offset = (
                torch.rand(
                    n_samples,
                    generator=gen,
                    dtype=torch.float32 if device.type == "mps" else torch.float64,
                    device=device,
                )
                - 0.5
            ) * (2.0 * float(domain.wavelength_scale))
            wavelength = (
                (float(domain.wavelength_center) + lam_offset)
                .to(dtype=torch.float32)
                .requires_grad_(True)
            )

            with torch.enable_grad():
                pred = model(ra, dec, wavelength)
                grad_outputs = torch.ones_like(pred)
                grads = torch.autograd.grad(
                    pred,
                    [ra, dec, wavelength],
                    grad_outputs=grad_outputs,
                    create_graph=True,
                    retain_graph=True,
                    allow_unused=True,
                )

                grad_ra, grad_dec, grad_lam = grads
                if (
                    grad_ra is not None
                    and grad_dec is not None
                    and grad_lam is not None
                ):
                    grad_spatial_mag = torch.sqrt(grad_ra**2 + grad_dec**2 + 1.0e-8)

                    mx = grad_spatial_mag.mean()
                    my = grad_lam.mean()
                    xm, ym = grad_spatial_mag - mx, grad_lam - my
                    r_num = torch.sum(xm * ym)
                    r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2) + 1.0e-8)
                    grad_corr = float(r_num.detach() / r_den.detach())

                    try:
                        cross_grads = torch.autograd.grad(
                            grad_lam,
                            [ra, dec],
                            grad_outputs=torch.ones_like(grad_lam),
                            retain_graph=True,
                            allow_unused=True,
                        )
                        grad_lam_ra, grad_lam_dec = cross_grads
                        if grad_lam_ra is not None and grad_lam_dec is not None:
                            cross_deriv_mean = 0.5 * (
                                grad_lam_ra.abs().mean() + grad_lam_dec.abs().mean()
                            )
                            lam_deriv_mean = grad_lam.abs().mean()
                            leakage_index = float(
                                cross_deriv_mean.detach()
                                / (lam_deriv_mean.detach() + 1.0e-8)
                            )
                    except Exception:
                        leakage_index = 0.0
        except Exception:
            grad_corr = 0.0
            leakage_index = 0.0

    reduced_chi2s = []
    psf_residual_correlations = []
    quality_metrics: list[dict[str, float]] = []

    support_uncs = []
    outer_uncs = []

    with torch.no_grad():
        for obs_idx, observation in enumerate(observations):
            fctx = obs_forward_ctx[obs_idx]
            pose = obs_exposure_poses[obs_idx]
            observation_bundle = obs_operator_bundles[obs_idx]

            pred = render_field_observation(
                model,
                observation,
                num_spectral_samples=num_spectral_samples,
                forward_context=fctx,
                exposure_pose=pose,
                operator_bundle=observation_bundle,
                spatial_sampling=spatial_sampling,
                spatial_tile_hw=spatial_tile_hw,
            )

            obs_img = _to(observation.image, device=pred.device, dtype=pred.dtype)
            weight = (
                _to(observation.weight, device=pred.device, dtype=pred.dtype)
                if observation.weight is not None
                else torch.ones_like(pred)
            )

            if observation.mask is not None:
                mask = observation.mask.to(device=pred.device) > 0
            else:
                mask = torch.ones_like(pred, dtype=torch.bool)

            residual = pred - obs_img
            valid_res = residual[mask]
            valid_weight = weight[mask]

            if valid_res.numel() > 0:
                weighted_sq_res = (valid_res**2) * valid_weight
                obs_chi2 = float(weighted_sq_res.mean())
                reduced_chi2s.append(obs_chi2)
                quality_metrics.append(
                    field_image_quality_metrics(
                        pred,
                        observation.image,
                        weight=observation.weight,
                        mask=observation.mask,
                    )
                )

                if observation_bundle.psf is not None:
                    from ..operators.psf import apply_psf_spec

                    res_convolved = apply_psf_spec(residual, observation.psf_spec)
                    valid_res_conv = res_convolved[mask]

                    mx = valid_res.mean()
                    my = valid_res_conv.mean()
                    xm, ym = valid_res - mx, valid_res_conv - my
                    r_num = torch.sum(xm * ym)
                    r_den = torch.sqrt(torch.sum(xm**2) * torch.sum(ym**2) + 1.0e-8)
                    psf_corr = float(r_num / r_den)
                    psf_residual_correlations.append(psf_corr)

            if (
                field_model_has_predictive_distribution(model)
                and spatial_tile_hw is None
            ):
                _pred_mean, pred_logvar = render_field_observation_distribution(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    forward_context=fctx,
                    exposure_pose=pose,
                    operator_bundle=observation_bundle,
                    spatial_sampling=spatial_sampling,
                )
                stddev = torch.sqrt(torch.clamp(torch.exp(pred_logvar), min=1.0e-8))

                if observation.mask is not None:
                    mask = observation.mask.to(device=stddev.device) > 0
                    support_unc = stddev[mask]
                    outer_unc = stddev[~mask]
                    if support_unc.numel() > 0:
                        support_uncs.append(float(support_unc.mean()))
                    if outer_unc.numel() > 0:
                        outer_uncs.append(float(outer_unc.mean()))
                else:
                    support_uncs.append(float(stddev.mean()))

    mean_reduced_chi2 = (
        sum(reduced_chi2s) / len(reduced_chi2s) if reduced_chi2s else float("nan")
    )
    mean_psf_residual_corr = (
        sum(psf_residual_correlations) / len(psf_residual_correlations)
        if psf_residual_correlations
        else 0.0
    )

    support_mean_uncertainty = (
        sum(support_uncs) / len(support_uncs) if support_uncs else 0.0
    )
    outer_mean_uncertainty = sum(outer_uncs) / len(outer_uncs) if outer_uncs else 0.0
    uncertainty_ratio = (
        outer_mean_uncertainty / (support_mean_uncertainty + 1.0e-8)
        if support_mean_uncertainty > 0.0
        else 1.0
    )
    quality_summary = _aggregate_quality_metrics(quality_metrics)

    step_duration_mean_ms = (fit_duration_sec / steps * 1000.0) if steps > 0 else 0.0
    throughput_pixels_per_sec = (
        total_pixels / fit_duration_sec if fit_duration_sec > 0.0 else 0.0
    )

    return {
        "spatial_spectral_leakage": leakage_index,
        "gradient_correlation": grad_corr,
        "reduced_chi2": mean_reduced_chi2,
        "psf_alignment": mean_psf_residual_corr,
        "quality_metrics": quality_summary,
        "uncertainty_stats": {
            "support_mean_uncertainty": support_mean_uncertainty,
            "outer_mean_uncertainty": outer_mean_uncertainty,
            "uncertainty_ratio": uncertainty_ratio,
        },
        "performance_profile": {
            "total_duration_sec": fit_duration_sec,
            "step_duration_mean_ms": step_duration_mean_ms,
            "throughput_pixels_per_sec": throughput_pixels_per_sec,
            "max_memory_allocated_bytes": max_mem,
        },
    }


def fit_structured_noise_crossfit(
    observations: Sequence[FieldObservation],
    cross_fitted_predictions: Sequence[Tensor],
    *,
    models: Sequence[StructuredDetectorNoise] | None = None,
    steps: int = 100,
    lr: float = 1.0e-2,
    device: torch.device | str | None = None,
) -> tuple[tuple[StructuredDetectorNoise, ...], tuple[float, ...]]:
    """Calibrate exposure noise only from leave-one-exposure-out predictions.

    ``cross_fitted_predictions[e]`` must come from a scene fit that excluded
    observation ``e``. The function cannot verify that provenance, so callers
    should record the exclusion split alongside the returned parameters.
    Models are frozen before return for the subsequent scene fit.
    """
    if not observations or len(observations) != len(cross_fitted_predictions):
        raise ValueError("observations and cross_fitted_predictions must match")
    if int(steps) < 1 or not math.isfinite(float(lr)) or float(lr) <= 0.0:
        raise ValueError("steps and lr must be positive")
    fitted = (
        tuple(StructuredDetectorNoise() for _ in observations)
        if models is None
        else tuple(models)
    )
    if len(fitted) != len(observations):
        raise ValueError("models must match observations")
    target_device = resolve_device(
        device, observations, cross_fitted_predictions, models
    )
    parameters = []
    for model in fitted:
        model.to(target_device)
        model.requires_grad_(True)
        parameters.extend(model.parameters())
    optimizer = torch.optim.Adam(parameters, lr=float(lr))
    final_losses: list[float] = []
    for _ in range(int(steps)):
        optimizer.zero_grad(set_to_none=True)
        losses = []
        for observation, prediction, model in zip(
            observations, cross_fitted_predictions, fitted, strict=True
        ):
            if prediction.shape != observation.image.shape:
                raise ValueError("cross-fitted prediction shape must match observation")
            pred = _to(prediction.detach(), target_device, dtype=torch.float32)
            image = _to(observation.image, target_device, dtype=pred.dtype)
            valid = _observation_support_mask(observation).to(target_device)
            residual = torch.where(valid, pred - image, torch.zeros_like(pred))
            losses.append(
                _structured_noise_loss(
                    observation,
                    ObservationOperatorBundle(
                        noise=StructuredNoiseOp(model, frozen=False)
                    ),
                    residual,
                    valid,
                )
            )
        total = torch.stack(losses).mean()
        total.backward()
        optimizer.step()
        final_losses = [float(loss.detach().cpu()) for loss in losses]
    for model in fitted:
        model.requires_grad_(False)
    return fitted, tuple(final_losses)


def fit_field_model(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    *,
    steps: int = 200,
    lr: float = 5.0e-3,
    weight_decay: float = 0.0,
    ema_decay: float = 0.0,
    num_spectral_samples: int = 9,
    loss_kind: FieldLossKind = "mse",
    charbonnier_eps: float = 1.0e-3,
    huber_delta: float = 5.0e-2,
    loss_reduction: FieldLossReduction = "per_observation",
    lr_schedule: FieldLRSchedule = "constant",
    grad_clip_norm: float | None = None,
    curriculum: FieldCurriculum = "none",
    training_spectral_sampling: FieldSpectralSampling = "deterministic_quadrature",
    training_spectral_sampling_grid_size: int | None = None,
    spatial_sampling: FieldSpatialSampling = "center",
    spectral_anchor_observations_per_step: int = 0,
    spectral_anchor_num_samples: int | None = None,
    spectral_anchor_loss_weight: float = 0.0,
    component_anchor_terms: Sequence[FieldComponentAnchorTerm] | None = None,
    component_anchor_num_spectral_samples: int | None = None,
    component_anchor_loss_kind: FieldLossKind | None = None,
    component_centroid_anchor_terms: Sequence[FieldCentroidAnchorTerm] | None = None,
    lr_warmup_steps: int = 0,
    spatial_smoothness_weight: float = 0.0,
    spatial_smoothness_grid_size: int = 8,
    gradient_residual_weight: float = 0.0,
    observations_per_step: int | None = None,
    observation_sampling: FieldObservationSampling = "uniform",
    training_patch_size: int | None = None,
    training_patches_per_observation: int = 1,
    pixel_samples_per_observation: int | None = None,
    shared_sky_patch_sampling: bool = False,
    streaming_accumulation: bool = False,
    full_observations_per_step: int = 0,
    context_observations_per_step: int = 0,
    context_downsample_factor: int = 1,
    context_loss_weight: float = 0.0,
    spectral_observability_weight: float = 0.0,
    spectral_observability_probe_grid_size: int = 8,
    residual_head_penalty_weight: float = 0.0,
    hybrid_foam_sparsity_weight: float = 0.0,
    hybrid_correlation_weight: float = 0.0,
    hybrid_probe_grid_size: int = 8,
    operator_scale_gate_weight: float = 0.0,
    operator_scale_shift_weight: float = 0.0,
    noise_pretrain_kind: FieldNoisePretrainKind = "none",
    noise_pretrain_steps: int = 0,
    noise_pretrain_lr: float | None = None,
    noise_pretrain_observations_per_step: int | None = None,
    noise_pretrain_pixel_samples_per_observation: int = 1024,
    noise_pretrain_num_terms: int = 48,
    noise_pretrain_max_frequency: float = 8.0,
    freeze_field_encoder: bool = False,
    freeze_field_body: bool = False,
    amp_dtype: FieldAMPDtype = "none",
    amp_scaler_state: Mapping[str, Any] | None = None,
    device: torch.device | str | None = None,
    observation_forward_contexts: Sequence[ForwardContext | None] | None = None,
    observation_operator_bundles: Sequence[ObservationOperatorBundle] | None = None,
    operator_calibration_terms: Sequence[OperatorCalibrationTerm] | None = None,
    observation_exposure_poses: Sequence[ExposurePose | None] | None = None,
    refine_poses: bool = False,
    anchor_first_pose: bool = True,
    render_spatial_tile_hw: tuple[int, int] | None = None,
    diagnostic_spatial_tile_hw: tuple[int, int] | None = None,
    optimizer_state: Mapping[str, Any] | None = None,
    sampling_rng_state: Tensor | None = None,
    _progress_callback: Callable[[int, float], None] | None = None,
) -> FieldFitResult:
    """Fit observations, optionally recomputing tiled structured-noise renders.

    With ``streaming_accumulation`` and ``render_spatial_tile_hw``, structured
    noise uses one full-exposure likelihood and replays each renderer tile for
    its gradient. Renderer activations are bounded by tile size; prediction,
    residual, and noise-model storage still scale with exposure size. As with
    activation checkpointing, rendering must not mutate model state.

    Args:
        observations: Non-empty sequence of field observations to fit.
        model: Field module whose parameters are optimized in place.
        steps: Number of optimizer steps.
        lr: Initial learning rate.
        device: Optional training device.
        loss_kind: Per-pixel fitting loss.
        num_spectral_samples: Spectral quadrature samples per observation.
        render_spatial_tile_hw: Optional renderer tile height and width.

    Returns:
        Fit diagnostics, optimizer state, and reproducibility state.

    Examples:
        >>> from torchsky.field import fit
        >>> callable(fit)
        True
    """
    if not observations:
        raise ValueError("fit_field_model requires at least one observation")
    fit_call_args = dict(locals())
    amp_name = str(amp_dtype)
    if amp_name in {"float16", "bfloat16"}:
        amp_device = resolve_fit_device(model, device)
        if amp_device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("field AMP is supported only on CUDA")
        if amp_name == "bfloat16" and not torch.cuda.is_bf16_supported():
            raise ValueError("CUDA device does not support bfloat16 AMP")
        if noise_pretrain_kind != "none" or int(noise_pretrain_steps) > 0:
            raise ValueError(
                "field AMP does not yet support noise pretraining; run it in FP32"
            )
        recursive_args = fit_call_args
        recursive_args["amp_dtype"] = f"_active_{amp_name}"
        autocast_dtype = torch.float16 if amp_name == "float16" else torch.bfloat16
        with torch.autocast(
            device_type="cuda",
            dtype=autocast_dtype,
        ):
            return fit_field_model(**recursive_args)
    if amp_name == "none":
        active_amp_name: str | None = None
    elif amp_name in {"_active_float16", "_active_bfloat16"}:
        active_amp_name = amp_name.removeprefix("_active_")
    else:
        raise ValueError("amp_dtype must be 'none', 'float16', or 'bfloat16'")
    train_device = resolve_fit_device(model, device)
    amp_active = active_amp_name is not None
    amp_scaler = (
        torch.amp.GradScaler(
            "cuda",
            enabled=active_amp_name == "float16",
        )
        if amp_active
        else None
    )
    if amp_scaler_state is not None:
        if amp_scaler is None:
            raise ValueError("amp_scaler_state requires amp_dtype")
        amp_scaler.load_state_dict(dict(amp_scaler_state))

    def _amp_disabled():
        if amp_active:
            return torch.autocast(device_type="cuda", enabled=False)
        return nullcontext()

    def _backward(loss: Tensor) -> None:
        with _amp_disabled():
            if amp_scaler is not None and amp_scaler.is_enabled():
                amp_scaler.scale(loss).backward()
            else:
                loss.backward()

    model = model.to(device=train_device)
    obs_operator_bundles = (
        tuple(observation_operator_bundles)
        if observation_operator_bundles is not None
        else tuple(
            field_observation_operator_bundle(observation)
            for observation in observations
        )
    )
    if len(obs_operator_bundles) != len(observations):
        raise ValueError(
            "observation_operator_bundles must have the same length as observations"
        )
    streamed_tiles = streaming_accumulation and render_spatial_tile_hw is not None
    recompute_structured = streamed_tiles and loss_kind == "structured_nll"
    pixel_sample_count = int(pixel_samples_per_observation or 0)
    if pixel_sample_count < 0:
        raise ValueError("pixel_samples_per_observation must be non-negative")
    if streamed_tiles and diagnostic_spatial_tile_hw is None:
        diagnostic_spatial_tile_hw = render_spatial_tile_hw
    if pixel_sample_count > 0 and (
        spatial_sampling != "center"
        or any(bundle.psf is not None for bundle in obs_operator_bundles)
    ):
        raise ValueError(
            "pixel-sampled fitting requires center sampling without a PSF; "
            "use streamed spatial tiles to retain pixel integration and PSF response"
        )
    if streamed_tiles and (
        loss_kind == "gaussian_nll"
        or gradient_residual_weight > 0
        or pixel_sample_count > 0
        or (context_observations_per_step > 0 and context_loss_weight > 0)
        or (
            spectral_anchor_observations_per_step > 0
            and spectral_anchor_loss_weight > 0
        )
        or bool(component_anchor_terms)
        or bool(component_centroid_anchor_terms)
    ):
        raise ValueError(
            "streamed spatial tiles support residual and structured_nll losses; "
            "learned gaussian_nll, gradient residuals, pixel sampling, "
            "context and anchor losses "
            "require the full-observation path"
        )
    if loss_kind == "structured_nll":
        incompatible = (
            pixel_samples_per_observation not in (None, 0)
            or training_patch_size is not None
            or shared_sky_patch_sampling
            or gradient_residual_weight > 0.0
            or spectral_anchor_observations_per_step > 0
            or context_observations_per_step > 0
            or bool(component_anchor_terms)
            or bool(component_centroid_anchor_terms)
        )
        if incompatible:
            raise ValueError(
                "structured_nll currently requires full native exposures without "
                "pixel/patch/context/anchor approximations"
            )
        for bundle in obs_operator_bundles:
            if not isinstance(bundle.noise, StructuredNoiseOp) or not isinstance(
                getattr(bundle.noise, "model", None), StructuredDetectorNoise
            ):
                raise ValueError(
                    "every observation needs StructuredNoiseOp for structured_nll"
                )
    operator_bundle = merge_operator_bundles(*obs_operator_bundles)
    calibration_terms = tuple(operator_calibration_terms or ())
    calibrated_operator_model_ids = {
        id(operator_model)
        for bundle in obs_operator_bundles
        for operator_model in (
            getattr(bundle.psf, "model", None),
            getattr(bundle.background, "model", None),
        )
        if isinstance(operator_model, nn.Module)
    }
    for term in calibration_terms:
        if not callable(getattr(term, "penalty", None)) or not callable(
            getattr(term, "diagnostics", None)
        ):
            raise TypeError(
                "operator_calibration_terms must provide penalty and diagnostics"
            )
        if id(getattr(term, "model", None)) not in calibrated_operator_model_ids:
            raise ValueError(
                "each operator calibration term model must belong to an observation bundle"
            )

    def training_psf_halo(
        bundle: ObservationOperatorBundle,
    ) -> tuple[int, int] | None:
        operator_model = getattr(bundle.psf, "model", None)
        kernel_size = getattr(operator_model, "kernel_size", None)
        if kernel_size is None:
            return None
        radius = int(kernel_size) // 2
        return radius, radius

    obs_forward_ctx: tuple[ForwardContext | None, ...] = (
        tuple(observation_forward_contexts)
        if observation_forward_contexts is not None
        else tuple(None for _ in observations)
    )
    if len(obs_forward_ctx) != len(observations):
        raise ValueError(
            "observation_forward_contexts must have the same length as observations"
        )

    if refine_poses:
        if observation_exposure_poses is not None:
            raise ValueError(
                "refine_poses cannot be combined with observation_exposure_poses"
            )
        observation_exposure_poses = create_exposure_poses(
            list(observations),
            anchor_first=bool(anchor_first_pose),
        )
    obs_exposure_poses: tuple[ExposurePose | None, ...] = (
        tuple(observation_exposure_poses)
        if observation_exposure_poses is not None
        else tuple(None for _ in observations)
    )
    if len(obs_exposure_poses) != len(observations):
        raise ValueError(
            "observation_exposure_poses must have the same length as observations"
        )
    for pose in obs_exposure_poses:
        if pose is not None:
            pose.to(device=train_device)
    if steps <= 0:
        raise ValueError("steps must be positive")
    pretrain_steps = max(int(noise_pretrain_steps), 0)
    if noise_pretrain_kind != "none" and pretrain_steps > 0:
        _noise_pretrain_field_model(
            observations,
            model,
            noise_kind=noise_pretrain_kind,
            steps=pretrain_steps,
            lr=(
                float(noise_pretrain_lr) if noise_pretrain_lr is not None else float(lr)
            ),
            observations_per_step=(
                int(noise_pretrain_observations_per_step)
                if noise_pretrain_observations_per_step is not None
                else min(len(observations), 4)
            ),
            pixel_samples_per_observation=noise_pretrain_pixel_samples_per_observation,
            num_terms=noise_pretrain_num_terms,
            max_frequency=noise_pretrain_max_frequency,
        )
    if freeze_field_encoder or freeze_field_body:
        for name, param in model.named_parameters():
            if freeze_field_encoder and name.startswith("encoder."):
                param.requires_grad = False
            if freeze_field_body and name.startswith("body."):
                param.requires_grad = False
    trainable_params = [p for p in model.parameters() if p.requires_grad]
    seen_param_ids = {id(parameter) for parameter in trainable_params}
    conditioning_models: list[tuple[str, nn.Module]] = []
    seen_conditioner_ids: set[int] = set()
    for bundle in obs_operator_bundles:
        for slot_name, operator in (
            ("psf", bundle.psf),
            ("background", bundle.background),
            ("noise", bundle.noise),
        ):
            operator_model = getattr(operator, "model", None)
            if not isinstance(operator_model, nn.Module):
                continue
            operator_model.to(device=train_device)
            if slot_name == "noise" and bool(getattr(operator, "frozen", True)):
                continue
            for parameter in operator_model.parameters():
                if parameter.requires_grad and id(parameter) not in seen_param_ids:
                    trainable_params.append(parameter)
                    seen_param_ids.add(id(parameter))
            conditioner = getattr(operator_model, "encoder", None)
            if (
                conditioner is not None
                and id(conditioner) not in seen_conditioner_ids
                and callable(
                    getattr(operator_model, "conditioning_regularization", None)
                )
            ):
                conditioning_models.append((slot_name, operator_model))
                seen_conditioner_ids.add(id(conditioner))
    seen_poses = set()
    for pose in obs_exposure_poses:
        if pose is not None and id(pose) not in seen_poses:
            trainable_params.extend([p for p in pose.parameters() if p.requires_grad])
            seen_poses.add(id(pose))

    if not trainable_params:
        raise ValueError(
            "fit_field_model: no trainable parameters (check freeze_field_encoder / freeze_field_body)"
        )
    base_lr = float(lr)
    optimizer = torch.optim.Adam(
        trainable_params,
        lr=base_lr,
        weight_decay=max(float(weight_decay), 0.0),
    )
    if optimizer_state is not None:
        optimizer.load_state_dict(dict(optimizer_state))
    gradient_weight = max(float(gradient_residual_weight), 0.0)

    # ── Pre-training loss on full observations ──────────────────────
    # Measure the loss BEFORE any optimizer steps, on full observation images
    # (same computation path as the post-training final_loss), so initial_loss
    # and final_loss are directly comparable regardless of which random patches
    # happen to be selected in the first training step.
    with torch.no_grad(), _amp_disabled():
        pre_train_loss = _compute_full_observation_losses(
            observations,
            model,
            obs_forward_ctx,
            obs_exposure_poses,
            obs_operator_bundles,
            num_spectral_samples=num_spectral_samples,
            loss_kind=loss_kind,
            charbonnier_eps=charbonnier_eps,
            huber_delta=huber_delta,
            loss_reduction=loss_reduction,
            gradient_residual_weight=gradient_weight,
            spatial_sampling=spatial_sampling,
            spatial_tile_hw=diagnostic_spatial_tile_hw,
        )

    model.train()
    loss_history: list[float] = []
    amp_overflow_steps = 0
    anchor_count = max(int(spectral_anchor_observations_per_step), 0)
    anchor_num_samples = (
        int(spectral_anchor_num_samples)
        if spectral_anchor_num_samples is not None
        else max(int(num_spectral_samples), 17)
    )
    anchor_loss_weight = max(float(spectral_anchor_loss_weight), 0.0)
    component_anchor_num = (
        int(component_anchor_num_spectral_samples)
        if component_anchor_num_spectral_samples is not None
        else int(num_spectral_samples)
    )
    component_anchor_kind = (
        loss_kind if component_anchor_loss_kind is None else component_anchor_loss_kind
    )
    warmup_steps = max(int(lr_warmup_steps), 0)
    smoothness_weight = max(float(spatial_smoothness_weight), 0.0)
    smoothness_grid_size = max(int(spatial_smoothness_grid_size), 2)
    patches_per_observation = max(int(training_patches_per_observation), 1)
    uncropped_count = max(int(full_observations_per_step), 0)
    context_count = max(int(context_observations_per_step), 0)
    context_pool = max(int(context_downsample_factor), 1)
    context_weight = max(float(context_loss_weight), 0.0)
    spectral_penalty_weight = max(float(spectral_observability_weight), 0.0)
    spectral_penalty_grid = max(int(spectral_observability_probe_grid_size), 2)
    residual_penalty_weight = max(float(residual_head_penalty_weight), 0.0)
    hybrid_sparsity_weight = float(hybrid_foam_sparsity_weight)
    hybrid_separation_weight = float(hybrid_correlation_weight)
    if (
        not math.isfinite(hybrid_sparsity_weight)
        or not math.isfinite(hybrid_separation_weight)
        or hybrid_sparsity_weight < 0.0
        or hybrid_separation_weight < 0.0
    ):
        raise ValueError(
            "hybrid regularization weights must be finite and non-negative"
        )
    if (
        hybrid_sparsity_weight > 0.0 or hybrid_separation_weight > 0.0
    ) and not isinstance(model, DiffuseFoamHybridField):
        raise ValueError("hybrid regularization requires DiffuseFoamHybridField")
    hybrid_grid_size = max(int(hybrid_probe_grid_size), 2)
    scale_gate_weight = max(float(operator_scale_gate_weight), 0.0)
    scale_shift_weight = max(float(operator_scale_shift_weight), 0.0)
    smoothness_domain = getattr(model, "domain", None)
    ema_rate = float(ema_decay)
    use_ema = 0.0 < ema_rate < 1.0
    sampling_generator = torch.Generator(device="cpu")
    if sampling_rng_state is None:
        sampling_generator.manual_seed(12345)
    else:
        sampling_generator.set_state(
            _as_tensor(sampling_rng_state, dtype=torch.uint8, device="cpu")
        )
    ema_params: list[Tensor] | None = None
    if use_ema:
        ema_params = [
            param.detach().clone()
            for param in model.parameters()
            if param.requires_grad
        ]

    total_pixels = 0
    if train_device.type == "cuda":
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    start_time = time.perf_counter()

    for step_idx in range(int(steps)):
        progress = float(step_idx + 1) / max(int(steps), 1)
        set_progress = getattr(model, "set_curriculum_progress", None)
        if callable(set_progress):
            if curriculum == "none":
                set_progress(1.0)
            elif curriculum in {"spectral_frequency_ramp", "feature_scale_ramp"}:
                set_progress(progress)
            else:
                raise ValueError(f"unsupported curriculum {curriculum!r}")
        if lr_schedule == "constant":
            step_lr = base_lr
        elif lr_schedule == "cosine":
            frac = float(step_idx) / max(int(steps) - 1, 1)
            step_lr = base_lr * (0.1 + 0.9 * 0.5 * (1.0 + math.cos(math.pi * frac)))
        else:
            raise ValueError(f"unsupported lr_schedule {lr_schedule!r}")
        if warmup_steps > 0 and step_idx < warmup_steps:
            warm_frac = float(step_idx + 1) / float(warmup_steps)
            step_lr = step_lr * max(warm_frac, 1.0e-3)
        for group in optimizer.param_groups:
            group["lr"] = step_lr
        optimizer.zero_grad(set_to_none=True)
        total_loss = None
        global_weighted_loss = None
        global_weight_sum = None
        band_weighted_loss = {}
        band_weight_sum = {}
        mixture_wavelengths = None
        mixture_probs = None
        if training_spectral_sampling == "mixture_mc":
            mixture_wavelengths, mixture_probs = _mixture_mc_samples(
                observations,
                num_spectral_samples=num_spectral_samples,
                device=resolve_device(None, model),
                proposal_grid_size=training_spectral_sampling_grid_size,
            )
        epoch_batch_size = (
            len(observations)
            if observations_per_step is None or int(observations_per_step) <= 0
            else min(int(observations_per_step), len(observations))
        )
        epoch_steps = max(math.ceil(len(observations) / epoch_batch_size), 1)
        sampling_epoch = step_idx // epoch_steps
        sampling_step_in_epoch = step_idx % epoch_steps
        selected_indices = _sample_observation_indices(
            observations,
            observations_per_step=observations_per_step,
            observation_sampling=observation_sampling,
            generator=sampling_generator,
            epoch=sampling_epoch,
            step_in_epoch=sampling_step_in_epoch,
        )
        selected_pairs: list[
            tuple[
                FieldObservation,
                ForwardContext | None,
                ExposurePose | None,
                ObservationOperatorBundle,
            ]
        ] = []
        if pixel_sample_count > 0:
            selected_pairs = [
                (
                    observations[idx],
                    obs_forward_ctx[idx],
                    obs_exposure_poses[idx],
                    obs_operator_bundles[idx],
                )
                for idx in selected_indices
            ]
        elif (
            shared_sky_patch_sampling
            and training_patch_size is not None
            and len(selected_indices) > uncropped_count
        ):
            uncropped_indices = selected_indices[:uncropped_count]
            cropped_indices = selected_indices[uncropped_count:]
            if uncropped_indices:
                selected_pairs.extend(
                    (
                        observations[i],
                        obs_forward_ctx[i],
                        obs_exposure_poses[i],
                        obs_operator_bundles[i],
                    )
                    for i in uncropped_indices
                )
            for _ in range(patches_per_observation):
                ref_position = int(
                    torch.randint(
                        len(cropped_indices),
                        (1,),
                        generator=sampling_generator,
                    ).item()
                )
                reference_observation = observations[cropped_indices[ref_position]]
                patch = min(
                    int(training_patch_size),
                    int(reference_observation.image.shape[-2]),
                    int(reference_observation.image.shape[-1]),
                )
                if patch <= 0:
                    continue
                ref_x0, ref_y0 = _sample_patch_origin(
                    reference_observation,
                    patch_size=patch,
                    generator=sampling_generator,
                )
                for observation_index in cropped_indices:
                    cropped = _crop_field_observation_to_shared_sky_patch(
                        observations[observation_index],
                        reference_observation=reference_observation,
                        x0=ref_x0,
                        y0=ref_y0,
                        patch_size=patch,
                        psf_halo=training_psf_halo(
                            obs_operator_bundles[observation_index],
                        ),
                    )
                    if cropped is not None:
                        selected_pairs.append(
                            (
                                cropped,
                                obs_forward_ctx[observation_index],
                                obs_exposure_poses[observation_index],
                                obs_operator_bundles[observation_index],
                            )
                        )
            if not selected_pairs:
                selected_pairs = [
                    (
                        observations[idx],
                        obs_forward_ctx[idx],
                        obs_exposure_poses[idx],
                        obs_operator_bundles[idx],
                    )
                    for idx in selected_indices
                ]
        else:
            for selected_rank, observation_index in enumerate(selected_indices):
                base_observation = observations[observation_index]
                base_ctx = obs_forward_ctx[observation_index]
                base_pose = obs_exposure_poses[observation_index]
                base_bundle = obs_operator_bundles[observation_index]
                if training_patch_size is None or selected_rank < uncropped_count:
                    selected_pairs.append(
                        (base_observation, base_ctx, base_pose, base_bundle)
                    )
                    continue
                patch = min(
                    int(training_patch_size),
                    int(base_observation.image.shape[-2]),
                    int(base_observation.image.shape[-1]),
                )
                if patch <= 0 or (
                    patch >= int(base_observation.image.shape[-2])
                    and patch >= int(base_observation.image.shape[-1])
                ):
                    selected_pairs.append(
                        (base_observation, base_ctx, base_pose, base_bundle)
                    )
                    continue
                for _ in range(patches_per_observation):
                    x0, y0 = _sample_patch_origin(
                        base_observation,
                        patch_size=patch,
                        generator=sampling_generator,
                    )
                    selected_pairs.append(
                        (
                            _crop_field_observation_for_training(
                                base_observation,
                                x0=x0,
                                y0=y0,
                                width=patch,
                                height=patch,
                                psf_halo=training_psf_halo(
                                    base_bundle,
                                ),
                            ),
                            base_ctx,
                            base_pose,
                            base_bundle,
                        )
                    )
        selected_observations = [o for o, _, _, _ in selected_pairs]
        for observation, _, _, observation_bundle in selected_pairs:
            if pixel_sample_count > 0 and observation_bundle.psf is None:
                total_pixels += pixel_sample_count
            else:
                total_pixels += observation.image.numel()
        context_pairs: list[
            tuple[
                FieldObservation,
                ForwardContext | None,
                ExposurePose | None,
                ObservationOperatorBundle,
            ]
        ] = []
        if context_count > 0 and context_weight > 0.0:
            context_epoch_steps = max(math.ceil(len(observations) / context_count), 1)
            context_indices = _sample_observation_indices(
                observations,
                observations_per_step=context_count,
                observation_sampling=observation_sampling,
                generator=sampling_generator,
                epoch=(
                    step_idx // context_epoch_steps
                    if observation_sampling == "epoch"
                    else 0
                ),
                step_in_epoch=(
                    step_idx % context_epoch_steps
                    if observation_sampling == "epoch"
                    else 0
                ),
            )
            context_pairs = [
                (
                    observations[idx],
                    obs_forward_ctx[idx],
                    obs_exposure_poses[idx],
                    obs_operator_bundles[idx],
                )
                for idx in context_indices
            ]

        streamed_total_loss = 0.0
        selected_global_weight_sum = None
        selected_band_weight_sum: dict[tuple[object, ...], float] = {}
        if streaming_accumulation:

            def observation_normalizer(observation: FieldObservation) -> float:
                if loss_kind == "structured_nll":
                    return float(_observation_support_mask(observation).sum())
                return _observation_weight_sum_scalar(observation)

            if loss_reduction in {"global", "band_global_mix"}:
                selected_global_weight_sum = sum(
                    observation_normalizer(observation)
                    for observation in selected_observations
                )
            if loss_reduction in {"per_band", "band_global_mix"}:
                for observation in selected_observations:
                    band_key = (
                        observation.spectral_min,
                        observation.spectral_max,
                        observation.wavelength,
                        observation.band_name,
                    )
                    selected_band_weight_sum[band_key] = selected_band_weight_sum.get(
                        band_key, 0.0
                    ) + observation_normalizer(observation)
        for (
            pair_index,
            (observation, forward_context, exposure_pose, observation_bundle),
        ) in enumerate(selected_pairs):
            if (
                os.environ.get("TORCHSKY_GPU_MEM_LOG") == "1"
                and train_device.type == "cuda"
                and pair_index % 25 == 0
            ):
                print(
                    f"[gpu-mem] fit-obs-{pair_index}: "
                    f"allocated={torch.cuda.memory_allocated() / 1024**3:.2f}GiB "
                    f"reserved={torch.cuda.memory_reserved() / 1024**3:.2f}GiB",
                    flush=True,
                )
            render_kwargs = {}
            if training_spectral_sampling == "mixture_mc":
                assert mixture_wavelengths is not None and mixture_probs is not None
                render_kwargs.update(
                    _mixture_render_kwargs(
                        observation,
                        mixture_wavelengths,
                        mixture_probs,
                    )
                )
            else:
                render_kwargs["spectral_sampling"] = training_spectral_sampling
                render_kwargs["spectral_sampling_grid_size"] = (
                    training_spectral_sampling_grid_size
                )
            render_kwargs["forward_context"] = forward_context
            render_kwargs["exposure_pose"] = exposure_pose
            render_kwargs["operator_bundle"] = observation_bundle
            # Full-frame renders through the field MLP explode on small MIG
            # slices; tile them when the caller asks for bounded activations.
            if render_spatial_tile_hw is not None:
                render_kwargs["spatial_tile_hw"] = render_spatial_tile_hw
            if streamed_tiles and not recompute_structured:
                band_key = (
                    observation.spectral_min,
                    observation.spectral_max,
                    observation.wavelength,
                    observation.band_name,
                )
                if loss_reduction == "per_observation":
                    inverse_denominator = 1.0 / max(
                        _observation_weight_sum_scalar(observation), 1e-8
                    )
                elif loss_reduction == "per_band":
                    inverse_denominator = 1.0 / max(
                        selected_band_weight_sum.get(band_key, 0.0), 1e-8
                    )
                elif loss_reduction == "global":
                    inverse_denominator = 1.0 / max(
                        float(selected_global_weight_sum or 0.0), 1e-8
                    )
                elif loss_reduction == "band_global_mix":
                    inverse_denominator = 0.5 * (
                        1.0 / max(selected_band_weight_sum.get(band_key, 0.0), 1e-8)
                        + 1.0 / max(float(selected_global_weight_sum or 0.0), 1e-8)
                    )
                else:
                    raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
                support = _observation_support_mask(observation)
                for ys, xs, prediction in iter_field_observation_tiles(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    spatial_sampling=spatial_sampling,
                    **render_kwargs,
                ):
                    valid_tile = support[ys, xs].to(device=prediction.device)
                    data_tile = _to(
                        observation.image[ys, xs], prediction.device, prediction.dtype
                    )
                    residual_tile = torch.where(valid_tile, prediction - data_tile, 0.0)
                    weight_tile = (
                        torch.ones_like(prediction)
                        if observation.weight is None
                        else _to(
                            observation.weight[ys, xs],
                            prediction.device,
                            prediction.dtype,
                        )
                    )
                    weight_tile = torch.where(valid_tile, weight_tile, 0.0)
                    tile_loss = (
                        weight_tile
                        * _field_residual_loss(
                            residual_tile,
                            loss_kind=loss_kind,
                            charbonnier_eps=charbonnier_eps,
                            huber_delta=huber_delta,
                        )
                    ).sum() * inverse_denominator
                    streamed_total_loss += float(tile_loss.detach().cpu())
                    _backward(tile_loss)
                    del prediction, tile_loss, residual_tile, data_tile, weight_tile
                continue
            if pixel_sample_count > 0 and observation_bundle.psf is None:
                sampled = _sample_observation_pixels(
                    observation,
                    num_samples=pixel_sample_count,
                    generator=sampling_generator,
                    device=resolve_device(None, model),
                )
                if sampled is None:
                    continue
                (
                    sample_ra_deg,
                    sample_dec_deg,
                    image,
                    weight,
                    support_count,
                    full_weight_sum,
                ) = sampled
                if loss_kind == "gaussian_nll":
                    pred, pred_logvar = render_field_samples_distribution(
                        model,
                        observation,
                        ra_deg=sample_ra_deg,
                        dec_deg=sample_dec_deg,
                        num_spectral_samples=num_spectral_samples,
                        **render_kwargs,
                    )
                else:
                    pred = render_field_samples(
                        model,
                        observation,
                        ra_deg=sample_ra_deg,
                        dec_deg=sample_dec_deg,
                        num_spectral_samples=num_spectral_samples,
                        **render_kwargs,
                    )
                    pred_logvar = None
                image = _to(image, device=pred.device, dtype=pred.dtype)
                weight = _to(weight, device=pred.device, dtype=pred.dtype)
                residual = pred - image
                sample_element_loss = _field_residual_loss(
                    residual,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                    logvar=pred_logvar,
                    observation_invvar=(
                        weight
                        if loss_kind == "gaussian_nll"
                        and observation.weight is not None
                        else None
                    ),
                    sample_weight=(weight if loss_kind != "gaussian_nll" else None),
                )
                weighted_loss = torch.mean(sample_element_loss) * float(
                    max(support_count, 1)
                )
                weight_sum = torch.as_tensor(
                    max(
                        float(support_count)
                        if loss_kind == "gaussian_nll"
                        else float(full_weight_sum),
                        1.0e-8,
                    ),
                    device=pred.device,
                    dtype=pred.dtype,
                )
            elif loss_kind == "structured_nll":
                if recompute_structured:
                    render_cpu_rng = torch.get_rng_state()
                    render_device_rng = (
                        None
                        if train_device.type == "cpu"
                        else torch.get_device_module(train_device).get_rng_state(
                            train_device
                        )
                    )
                with torch.no_grad() if recompute_structured else nullcontext():
                    pred = render_field_observation(
                        model,
                        observation,
                        num_spectral_samples=num_spectral_samples,
                        spatial_sampling=spatial_sampling,
                        **render_kwargs,
                    )
                if recompute_structured:
                    pred.requires_grad_()
                pred_logvar = None
                image = _to(observation.image, device=pred.device, dtype=pred.dtype)
                valid = _observation_support_mask(observation).to(device=pred.device)
                image = torch.where(valid, image, torch.zeros_like(image))
                residual = torch.where(valid, pred - image, torch.zeros_like(pred))
                weight = torch.where(
                    valid, torch.ones_like(pred), torch.zeros_like(pred)
                )
                weight_sum = weight.sum()
                weighted_loss = (
                    _structured_noise_loss(
                        observation, observation_bundle, residual, valid
                    )
                    * weight_sum
                )
            elif loss_kind == "gaussian_nll":
                pred, pred_logvar = render_field_observation_distribution(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    spatial_sampling=spatial_sampling,
                    **render_kwargs,
                )
                image = _to(observation.image, device=pred.device, dtype=pred.dtype)
                valid = _observation_support_mask(observation).to(device=pred.device)
                image = torch.where(valid, image, torch.zeros_like(image))
                residual = torch.where(valid, pred - image, torch.zeros_like(pred))
                weight = (
                    _to(observation.weight, device=pred.device, dtype=pred.dtype)
                    if observation.weight is not None
                    else torch.ones_like(pred)
                )
                weight = torch.where(valid, weight, torch.zeros_like(weight))
                weighted_loss = torch.sum(
                    _field_residual_loss(
                        residual,
                        loss_kind=loss_kind,
                        charbonnier_eps=charbonnier_eps,
                        huber_delta=huber_delta,
                        logvar=pred_logvar,
                        observation_invvar=(
                            weight if observation.weight is not None else None
                        ),
                        sample_weight=(
                            None
                            if observation.weight is not None
                            else valid.to(pred.dtype)
                        ),
                    )
                )
                weight_sum = valid.sum().to(dtype=pred.dtype)
            else:
                pred = render_field_observation(
                    model,
                    observation,
                    num_spectral_samples=num_spectral_samples,
                    spatial_sampling=spatial_sampling,
                    **render_kwargs,
                )
                pred_logvar = None
                image = _to(observation.image, device=pred.device, dtype=pred.dtype)
                valid = _observation_support_mask(observation).to(device=pred.device)
                image = torch.where(valid, image, torch.zeros_like(image))
                residual = torch.where(valid, pred - image, torch.zeros_like(pred))
                weight = (
                    _to(observation.weight, device=pred.device, dtype=pred.dtype)
                    if observation.weight is not None
                    else torch.ones_like(pred)
                )
                weight = torch.where(valid, weight, torch.zeros_like(weight))
                weighted_loss = torch.sum(
                    _field_residual_loss(
                        residual,
                        loss_kind=loss_kind,
                        charbonnier_eps=charbonnier_eps,
                        huber_delta=huber_delta,
                        logvar=pred_logvar,
                        sample_weight=weight,
                    )
                )
                weight_sum = weight.sum()
            if gradient_weight > 0.0 and pred.ndim == 2:
                grad_num, grad_den = _field_gradient_residual_weighted_loss(
                    pred,
                    image,
                    weight,
                    valid,
                    loss_kind=loss_kind,
                    charbonnier_eps=charbonnier_eps,
                    huber_delta=huber_delta,
                )
                grad_loss = grad_num / torch.clamp(grad_den, min=1.0e-8)
                weighted_loss = weighted_loss + gradient_weight * weight_sum * grad_loss
            band_key = (
                observation.spectral_min,
                observation.spectral_max,
                observation.wavelength,
                observation.band_name,
            )
            if streaming_accumulation:
                if loss_reduction == "per_observation":
                    obs_loss = weighted_loss / torch.clamp(weight_sum, min=1.0e-8)
                elif loss_reduction == "per_band":
                    denom = max(selected_band_weight_sum.get(band_key, 0.0), 1.0e-8)
                    obs_loss = weighted_loss / denom
                elif loss_reduction == "global":
                    denom = max(float(selected_global_weight_sum or 0.0), 1.0e-8)
                    obs_loss = weighted_loss / denom
                elif loss_reduction == "band_global_mix":
                    band_denom = max(
                        selected_band_weight_sum.get(band_key, 0.0), 1.0e-8
                    )
                    global_denom = max(float(selected_global_weight_sum or 0.0), 1.0e-8)
                    obs_loss = 0.5 * (
                        (weighted_loss / band_denom) + (weighted_loss / global_denom)
                    )
                else:
                    raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
                streamed_total_loss += float(obs_loss.detach().cpu())
                _backward(obs_loss)
                if recompute_structured:
                    # Checkpoint the renderer, not the likelihood: correlations
                    # and amplifier geometry still see the entire exposure.
                    # Full image/noise storage is O(pixels); field activations
                    # are released after each tile's vector-Jacobian product.
                    assert pred.grad is not None
                    with torch.random.fork_rng(
                        devices=[] if train_device.type == "cpu" else [train_device],
                        device_type=train_device.type,
                    ):
                        torch.set_rng_state(render_cpu_rng)
                        if render_device_rng is not None:
                            torch.get_device_module(train_device).set_rng_state(
                                render_device_rng, train_device
                            )
                        for ys, xs, prediction in iter_field_observation_tiles(
                            model,
                            observation,
                            num_spectral_samples=num_spectral_samples,
                            spatial_sampling=spatial_sampling,
                            **render_kwargs,
                        ):
                            # pred.grad already includes AMP loss scaling.
                            if prediction.requires_grad:
                                with _amp_disabled():
                                    prediction.backward(pred.grad[ys, xs])
                            del prediction
                if (
                    os.environ.get("TORCHSKY_GPU_MEM_LOG") == "1"
                    and train_device.type == "cuda"
                ):
                    # MIG slices are small; cached blocks from prior
                    # observations can starve the next render otherwise.
                    torch.cuda.empty_cache()
                    torch.cuda.ipc_collect()
            elif loss_reduction == "per_observation":
                denom = torch.clamp(weight_sum, min=1.0e-8)
                obs_loss = weighted_loss / denom
                total_loss = obs_loss if total_loss is None else total_loss + obs_loss
            elif loss_reduction == "per_band":
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
            elif loss_reduction == "global":
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            elif loss_reduction == "band_global_mix":
                band_weighted_loss[band_key] = (
                    weighted_loss
                    if band_key not in band_weighted_loss
                    else band_weighted_loss[band_key] + weighted_loss
                )
                band_weight_sum[band_key] = (
                    weight_sum
                    if band_key not in band_weight_sum
                    else band_weight_sum[band_key] + weight_sum
                )
                global_weighted_loss = (
                    weighted_loss
                    if global_weighted_loss is None
                    else global_weighted_loss + weighted_loss
                )
                global_weight_sum = (
                    weight_sum
                    if global_weight_sum is None
                    else global_weight_sum + weight_sum
                )
            else:
                raise ValueError(f"unsupported loss_reduction {loss_reduction!r}")
        if streaming_accumulation:
            total_loss = torch.tensor(
                streamed_total_loss,
                dtype=torch.float32,
                device=resolve_device(None, model),
            )
        elif loss_reduction == "global":
            assert global_weighted_loss is not None and global_weight_sum is not None
            total_loss = global_weighted_loss / torch.clamp(
                global_weight_sum, min=1.0e-8
            )
        elif loss_reduction == "per_band":
            total_loss = torch.stack(
                [
                    weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8)
                    for band_key, weighted in band_weighted_loss.items()
                ]
            ).sum()
        elif loss_reduction == "band_global_mix":
            assert global_weighted_loss is not None and global_weight_sum is not None
            band_loss = torch.stack(
                [
                    weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8)
                    for band_key, weighted in band_weighted_loss.items()
                ]
            ).sum()
            global_loss = global_weighted_loss / torch.clamp(
                global_weight_sum, min=1.0e-8
            )
            total_loss = 0.5 * (band_loss + global_loss)
        assert total_loss is not None  # every reduction above assigns it
        if (
            training_spectral_sampling == "mixture_mc"
            and anchor_count > 0
            and anchor_loss_weight > 0.0
        ):
            anchor_losses = []
            max_anchor_count = min(anchor_count, len(selected_observations))
            if max_anchor_count > 0:
                anchor_perm = torch.randperm(
                    len(selected_observations),
                    generator=sampling_generator,
                )[:max_anchor_count].tolist()
                for anchor_index in anchor_perm:
                    (
                        anchor_observation,
                        anchor_fctx,
                        anchor_pose,
                        anchor_bundle,
                    ) = selected_pairs[anchor_index]
                    anchor_pred = render_field_observation(
                        model,
                        anchor_observation,
                        num_spectral_samples=anchor_num_samples,
                        spectral_sampling="deterministic_quadrature",
                        spatial_sampling=spatial_sampling,
                        forward_context=anchor_fctx,
                        exposure_pose=anchor_pose,
                        operator_bundle=anchor_bundle,
                    )
                    anchor_image = _to(
                        anchor_observation.image,
                        device=anchor_pred.device,
                        dtype=anchor_pred.dtype,
                    )
                    anchor_weight = (
                        _to(
                            anchor_observation.weight,
                            device=anchor_pred.device,
                            dtype=anchor_pred.dtype,
                        )
                        if anchor_observation.weight is not None
                        else torch.ones_like(anchor_pred)
                    )
                    if anchor_observation.mask is not None:
                        anchor_valid = (
                            anchor_observation.mask.to(device=anchor_pred.device) > 0
                        )
                        anchor_image = torch.where(
                            anchor_valid, anchor_image, torch.zeros_like(anchor_image)
                        )
                        anchor_weight = torch.where(
                            anchor_valid, anchor_weight, torch.zeros_like(anchor_weight)
                        )
                    else:
                        anchor_valid = torch.ones_like(anchor_image, dtype=torch.bool)

                    anchor_residual = torch.where(
                        anchor_valid,
                        anchor_pred - anchor_image,
                        torch.zeros_like(anchor_pred),
                    )

                    anchor_weighted_loss = torch.sum(
                        anchor_weight
                        * _field_residual_loss(
                            anchor_residual,
                            loss_kind=loss_kind,
                            charbonnier_eps=charbonnier_eps,
                            huber_delta=huber_delta,
                            logvar=None,
                        )
                    )
                    anchor_losses.append(
                        anchor_weighted_loss
                        / torch.clamp(anchor_weight.sum(), min=1.0e-8)
                    )
            if anchor_losses:
                anchor_loss = torch.stack(anchor_losses).mean()
                if streaming_accumulation:
                    scaled_anchor_loss = anchor_loss_weight * anchor_loss
                    streamed_total_loss += float(scaled_anchor_loss.detach().cpu())
                    _backward(scaled_anchor_loss)
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=resolve_device(None, model),
                    )
                else:
                    total_loss = total_loss + anchor_loss_weight * anchor_loss
        if component_anchor_terms:
            anchor_component_loss = field_component_anchor_loss(
                model,
                component_anchor_terms,
                num_spectral_samples=component_anchor_num,
                spectral_sampling="deterministic_quadrature",
                loss_kind=component_anchor_kind,
                charbonnier_eps=charbonnier_eps,
                huber_delta=huber_delta,
            )
            if streaming_accumulation:
                streamed_total_loss += float(anchor_component_loss.detach().cpu())
                _backward(anchor_component_loss)
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=resolve_device(None, model),
                )
            else:
                total_loss = total_loss + anchor_component_loss
        if component_centroid_anchor_terms:
            centroid_anchor_loss = field_component_centroid_anchor_loss(
                model,
                component_centroid_anchor_terms,
                num_spectral_samples=component_anchor_num,
                spectral_sampling="deterministic_quadrature",
            )
            if streaming_accumulation:
                streamed_total_loss += float(centroid_anchor_loss.detach().cpu())
                _backward(centroid_anchor_loss)
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=resolve_device(None, model),
                )
            else:
                total_loss = total_loss + centroid_anchor_loss
        if context_pairs and context_weight > 0.0:
            scaled_context_weight = context_weight / max(len(context_pairs), 1)
            for context_observation, ctx_f, ctx_pose, ctx_bundle in context_pairs:
                context_render_kwargs = {}
                context_render_kwargs["exposure_pose"] = ctx_pose
                if training_spectral_sampling == "mixture_mc":
                    assert mixture_wavelengths is not None and mixture_probs is not None
                    context_render_kwargs.update(
                        _mixture_render_kwargs(
                            context_observation,
                            mixture_wavelengths,
                            mixture_probs,
                        )
                    )
                else:
                    context_render_kwargs["spectral_sampling"] = (
                        training_spectral_sampling
                    )
                    context_render_kwargs["spectral_sampling_grid_size"] = (
                        training_spectral_sampling_grid_size
                    )
                context_render_kwargs["forward_context"] = ctx_f
                context_render_kwargs["operator_bundle"] = ctx_bundle
                context_render_kwargs["spatial_sampling"] = spatial_sampling
                context_pred = render_field_observation(
                    model,
                    context_observation,
                    num_spectral_samples=num_spectral_samples,
                    **context_render_kwargs,
                )
                context_image = _to(
                    context_observation.image,
                    device=context_pred.device,
                    dtype=context_pred.dtype,
                )
                context_weight_map = (
                    _to(
                        context_observation.weight,
                        device=context_pred.device,
                        dtype=context_pred.dtype,
                    )
                    if context_observation.weight is not None
                    else torch.ones_like(context_pred)
                )
                if context_observation.mask is not None:
                    context_valid = (
                        context_observation.mask.to(device=context_pred.device) > 0
                    )
                    context_image = torch.where(
                        context_valid, context_image, torch.zeros_like(context_image)
                    )
                    context_weight_map = torch.where(
                        context_valid,
                        context_weight_map,
                        torch.zeros_like(context_weight_map),
                    )
                else:
                    context_valid = torch.ones_like(context_image, dtype=torch.bool)

                pooled_pred = _avg_pool_map(context_pred, factor=context_pool)
                pooled_image = _avg_pool_map(context_image, factor=context_pool)
                pooled_weight = _avg_pool_map(context_weight_map, factor=context_pool)

                # Use pooled validity approximation: if weight is > 0, it is somewhat valid
                pooled_valid = pooled_weight > 0.0
                context_residual = torch.where(
                    pooled_valid,
                    pooled_pred - pooled_image,
                    torch.zeros_like(pooled_pred),
                )

                context_loss = torch.sum(
                    pooled_weight
                    * _field_residual_loss(
                        context_residual,
                        loss_kind=loss_kind,
                        charbonnier_eps=charbonnier_eps,
                        huber_delta=huber_delta,
                        logvar=None,
                    )
                ) / torch.clamp(pooled_weight.sum(), min=1.0e-8)
                scaled_context_loss = scaled_context_weight * context_loss
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_context_loss.detach().cpu())
                    _backward(scaled_context_loss)
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=resolve_device(None, model),
                    )
                else:
                    total_loss = total_loss + scaled_context_loss
        if smoothness_weight > 0.0 and isinstance(smoothness_domain, FieldDomain):
            probe_ra, probe_dec = _smoothness_probe_grid(
                smoothness_domain,
                grid_size=smoothness_grid_size,
                device=resolve_device(None, model),
            )
            probe = model(
                probe_ra,
                probe_dec,
                torch.full_like(
                    probe_ra,
                    float(smoothness_domain.wavelength_center),
                ),
            )
            diff_x = probe[:, 1:] - probe[:, :-1]
            diff_y = probe[1:, :] - probe[:-1, :]
            # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
            smoothness_loss = 0.5 * (
                torch.mean(diff_x * diff_x) + torch.mean(diff_y * diff_y)
            )
            if streaming_accumulation:
                scaled_smoothness_loss = smoothness_weight * smoothness_loss
                streamed_total_loss += float(scaled_smoothness_loss.detach().cpu())
                _backward(scaled_smoothness_loss)
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=resolve_device(None, model),
                )
            else:
                total_loss = total_loss + smoothness_weight * smoothness_loss
        if spectral_penalty_weight > 0.0:
            spectral_penalty_fn = getattr(model, "spectral_observability_penalty", None)
            if callable(spectral_penalty_fn):
                spectral_penalty_fn = cast(Callable[..., Tensor], spectral_penalty_fn)
                spectral_penalty = spectral_penalty_fn(
                    observations,
                    probe_grid_size=spectral_penalty_grid,
                )
                scaled_spectral_penalty = spectral_penalty_weight * spectral_penalty
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_spectral_penalty.detach().cpu())
                    _backward(scaled_spectral_penalty)
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=resolve_device(None, model),
                    )
                else:
                    total_loss = total_loss + scaled_spectral_penalty
        if residual_penalty_weight > 0.0:
            residual_penalty_fn = getattr(model, "residual_head_penalty", None)
            if callable(residual_penalty_fn):
                residual_penalty_fn = cast(Callable[..., Tensor], residual_penalty_fn)

                residual_penalty = residual_penalty_fn()
                scaled_residual_penalty = residual_penalty_weight * residual_penalty
                if streaming_accumulation:
                    streamed_total_loss += float(scaled_residual_penalty.detach().cpu())
                    _backward(scaled_residual_penalty)
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=resolve_device(None, model),
                    )
                else:
                    total_loss = total_loss + scaled_residual_penalty
        if hybrid_sparsity_weight > 0.0 or hybrid_separation_weight > 0.0:
            assert isinstance(model, DiffuseFoamHybridField)
            hybrid_ra, hybrid_dec = _smoothness_probe_grid(
                model.domain,
                grid_size=hybrid_grid_size,
                device=train_device,
            )
            hybrid_penalty = model.component_regularization(
                hybrid_ra,
                hybrid_dec,
                torch.full_like(
                    hybrid_ra,
                    float(model.domain.wavelength_center),
                ),
                foam_sparsity_weight=hybrid_sparsity_weight,
                correlation_weight=hybrid_separation_weight,
            )
            if streaming_accumulation:
                streamed_total_loss += float(hybrid_penalty.detach().cpu())
                _backward(hybrid_penalty)
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=train_device,
                )
            else:
                total_loss = total_loss + hybrid_penalty
        if scale_gate_weight > 0.0 or scale_shift_weight > 0.0:
            conditioning_penalties = [
                operator_model.conditioning_regularization(  # type: ignore[attr-defined]
                    gate_weight=scale_gate_weight,
                    scale_shift_weight=scale_shift_weight,
                )
                for _, operator_model in conditioning_models
            ]
            if conditioning_penalties:
                conditioning_penalty = torch.stack(conditioning_penalties).mean()
                if streaming_accumulation:
                    streamed_total_loss += float(conditioning_penalty.detach().cpu())
                    _backward(conditioning_penalty)
                    total_loss = torch.tensor(
                        streamed_total_loss,
                        dtype=torch.float32,
                        device=resolve_device(None, model),
                    )
                else:
                    total_loss = total_loss + conditioning_penalty
        if calibration_terms:
            calibration_penalties = [term.penalty() for term in calibration_terms]
            if not all(bool(torch.isfinite(value)) for value in calibration_penalties):
                raise ValueError("operator calibration penalty must be finite")
            calibration_penalty = torch.stack(calibration_penalties).sum()
            if streaming_accumulation:
                streamed_total_loss += float(calibration_penalty.detach().cpu())
                if calibration_penalty.requires_grad:
                    _backward(calibration_penalty)
                total_loss = torch.tensor(
                    streamed_total_loss,
                    dtype=torch.float32,
                    device=resolve_device(None, model),
                )
            else:
                total_loss = total_loss + calibration_penalty
        assert total_loss is not None
        loss_history.append(float(total_loss.detach().cpu()))
        if not streaming_accumulation:
            _backward(total_loss)
        optimizer_step_applied = True
        with _amp_disabled():
            if amp_scaler is not None and amp_scaler.is_enabled():
                amp_scaler.unscale_(optimizer)
            if grad_clip_norm is not None:
                torch.nn.utils.clip_grad_norm_(trainable_params, float(grad_clip_norm))
            if amp_scaler is not None and amp_scaler.is_enabled():
                scale_before = float(amp_scaler.get_scale())
                amp_scaler.step(optimizer)
                amp_scaler.update()
                optimizer_step_applied = float(amp_scaler.get_scale()) >= scale_before
            else:
                optimizer.step()
        if not optimizer_step_applied:
            amp_overflow_steps += 1
        if optimizer_step_applied and use_ema and ema_params is not None:
            ema_index = 0
            with torch.no_grad():
                for param in model.parameters():
                    if not param.requires_grad:
                        continue
                    ema_params[ema_index].mul_(ema_rate).add_(
                        param.detach(),
                        alpha=1.0 - ema_rate,
                    )
                    ema_index += 1
        if _progress_callback is not None:
            _progress_callback(step_idx, float(loss_history[-1]))
    if use_ema and ema_params is not None:
        ema_index = 0
        with torch.no_grad():
            for param in model.parameters():
                if not param.requires_grad:
                    continue
                param.copy_(ema_params[ema_index])
                ema_index += 1
    model.eval()
    with torch.no_grad(), _amp_disabled():
        final_loss = _compute_full_observation_losses(
            observations,
            model,
            obs_forward_ctx,
            obs_exposure_poses,
            obs_operator_bundles,
            num_spectral_samples=num_spectral_samples,
            loss_kind=loss_kind,
            charbonnier_eps=charbonnier_eps,
            huber_delta=huber_delta,
            loss_reduction=loss_reduction,
            gradient_residual_weight=gradient_weight,
            spatial_sampling=spatial_sampling,
            spatial_tile_hw=diagnostic_spatial_tile_hw,
        )
    if train_device.type == "cuda":
        torch.cuda.synchronize()
    end_time = time.perf_counter()
    duration = end_time - start_time

    max_mem = 0
    if train_device.type == "cuda":
        max_mem = torch.cuda.max_memory_allocated()
        torch.cuda.reset_peak_memory_stats()

    with _amp_disabled():
        diagnostics = compute_field_fit_diagnostics(
            model=model,
            observations=observations,
            obs_forward_ctx=obs_forward_ctx,
            obs_exposure_poses=obs_exposure_poses,
            obs_operator_bundles=obs_operator_bundles,
            num_spectral_samples=num_spectral_samples,
            fit_duration_sec=duration,
            steps=int(steps),
            total_pixels=total_pixels,
            max_mem=max_mem,
            spatial_sampling=spatial_sampling,
            spatial_tile_hw=diagnostic_spatial_tile_hw,
        )
    diagnostics = dict(diagnostics or {})
    diagnostics["operator_bundle"] = operator_bundle.enabled_operator_names()
    diagnostics["amp_dtype"] = active_amp_name or "none"
    diagnostics["amp_scaler_enabled"] = bool(
        amp_scaler is not None and amp_scaler.is_enabled()
    )
    diagnostics["amp_overflow_steps"] = int(amp_overflow_steps)
    if isinstance(model, DiffuseFoamHybridField):
        diagnostics["hybrid"] = {
            "training_stage": model.training_stage,
            "component_identifiability": model.component_identifiability,
            "foam_sparsity_weight": hybrid_sparsity_weight,
            "correlation_weight": hybrid_separation_weight,
            "probe_grid_size": hybrid_grid_size,
        }
    scale_diagnostics = []
    for slot_name, operator_model in conditioning_models:
        report = getattr(operator_model.encoder, "scale_diagnostics", None)  # type: ignore[attr-defined]
        if callable(report):
            scale_diagnostics.append({"operator": slot_name, "axes": report()})
    if scale_diagnostics:
        diagnostics["operator_scale_diagnostics"] = scale_diagnostics
    if calibration_terms:
        diagnostics["operator_calibration_diagnostics"] = [
            term.diagnostics() for term in calibration_terms
        ]
    if any(pose is not None for pose in obs_exposure_poses):
        diagnostics["pose_refinement"] = {
            "enabled": bool(refine_poses or observation_exposure_poses is not None),
            "auto_created": bool(refine_poses),
            "anchor_first_pose": bool(anchor_first_pose) if refine_poses else None,
            "num_poses": sum(pose is not None for pose in obs_exposure_poses),
            "num_trainable_poses": sum(
                pose is not None
                and any(param.requires_grad for param in pose.parameters())
                for pose in obs_exposure_poses
            ),
            "parameters": [
                None
                if pose is None
                else {
                    "delta_ra_deg": float(pose.delta_ra_deg.detach().cpu()),
                    "delta_dec_deg": float(pose.delta_dec_deg.detach().cpu()),
                    "delta_theta_rad": float(pose.delta_theta_rad.detach().cpu()),
                    "delta_log_scale": float(pose.delta_log_scale.detach().cpu()),
                }
                for pose in obs_exposure_poses
            ],
        }

    return FieldFitResult(
        model=model,
        loss_history=tuple(loss_history),
        final_loss=final_loss,
        diagnostics=diagnostics,
        exposure_poses=(
            obs_exposure_poses
            if any(pose is not None for pose in obs_exposure_poses)
            else None
        ),
        operator_bundle=operator_bundle,
        observation_operator_bundles=obs_operator_bundles,
        optimizer_state=optimizer.state_dict(),
        amp_scaler_state=(amp_scaler.state_dict() if amp_scaler is not None else None),
        sampling_rng_state=sampling_generator.get_state(),
        completed_steps=int(steps),
        _pre_train_loss=pre_train_loss,
    )


def fit_diffuse_foam_hybrid(
    observations: Sequence[FieldObservation],
    model: DiffuseFoamHybridField,
    *,
    diffuse_steps: int,
    foam_steps: int,
    joint_steps: int,
    joint_foam_sparsity_weight: float = 0.0,
    joint_correlation_weight: float = 0.0,
    **fit_kwargs: Any,
) -> FieldFitResult:
    """Fit diffuse, localized, then joint hybrid stages on one operator path."""
    stage_steps = {
        "diffuse": int(diffuse_steps),
        "foam": int(foam_steps),
        "joint": int(joint_steps),
    }
    if any(steps < 1 for steps in stage_steps.values()):
        raise ValueError("all diffuse/Foam hybrid stages require at least one step")
    forbidden = {
        "steps",
        "optimizer_state",
        "amp_scaler_state",
        "hybrid_foam_sparsity_weight",
        "hybrid_correlation_weight",
    }.intersection(fit_kwargs)
    if forbidden:
        raise ValueError(
            "fit_diffuse_foam_hybrid owns stage state and regularization: "
            + ", ".join(sorted(forbidden))
        )
    sampling_rng_state = fit_kwargs.pop("sampling_rng_state", None)
    stage_results: list[FieldFitResult] = []
    for stage in ("diffuse", "foam", "joint"):
        model.set_training_stage(stage)  # type: ignore[arg-type]
        result = fit_field_model(
            observations,
            model,
            steps=stage_steps[stage],
            sampling_rng_state=sampling_rng_state,
            hybrid_foam_sparsity_weight=(
                float(joint_foam_sparsity_weight) if stage == "joint" else 0.0
            ),
            hybrid_correlation_weight=(
                float(joint_correlation_weight) if stage == "joint" else 0.0
            ),
            **fit_kwargs,
        )
        sampling_rng_state = result.sampling_rng_state
        stage_results.append(result)
    final = stage_results[-1]
    final.loss_history = tuple(
        loss for result in stage_results for loss in result.loss_history
    )
    final.completed_steps = sum(stage_steps.values())
    final._pre_train_loss = stage_results[0].initial_loss
    diagnostics = dict(final.diagnostics or {})
    diagnostics["hybrid_stages"] = [
        {
            "stage": stage,
            "steps": stage_steps[stage],
            "initial_loss": result.initial_loss,
            "final_loss": result.final_loss,
        }
        for stage, result in zip(
            ("diffuse", "foam", "joint"),
            stage_results,
            strict=True,
        )
    ]
    final.diagnostics = diagnostics
    return final


__all__ = [
    "fit_diffuse_foam_hybrid",
    "fit_field_component_decomposition",
    "fit_field_model",
    "_noise_pretrain_field_model",
]
