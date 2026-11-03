"""Full-observation loss evaluation and structured-noise loss helpers for field training."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
import torch.nn as nn
from torch import Tensor


from ..exposure_pose import ExposurePose
from ..operators.bundle import (
    ObservationOperatorBundle,
)
from ..operators.noise import StructuredDetectorNoise, StructuredNoiseOp

from .bandpass import (
    mixture_importance_weights as _mixture_importance_weights,
)
from .losses import _field_gradient_residual_weighted_loss, _field_residual_loss
from .observation_ops import (
    observation_support_mask as _observation_support_mask,
)
from .rendering import (
    render_field_observation,
)
from .types import (
    FieldLossKind,
    FieldLossReduction,
    FieldObservation,
    FieldSpatialSampling,
    ForwardContext,
)
from ...device import _to


def _mixture_render_kwargs(
    observation: FieldObservation,
    wavelengths: Tensor,
    proposal_probs: Tensor,
) -> dict[str, Any]:
    weights = _mixture_importance_weights(
        observation,
        wavelengths,
        proposal_probs,
    )
    if bool(weights.sum() > 0.0):
        return {
            "override_wavelengths": wavelengths,
            "override_weights": weights,
            "spectral_sampling": "deterministic_quadrature",
        }
    # ponytail: a finite shared Monte Carlo draw can miss a disjoint passband;
    # use its exact per-observation quadrature instead of increasing every draw.
    return {"spectral_sampling": "deterministic_quadrature"}


def _structured_noise_loss(
    observation: FieldObservation,
    bundle: ObservationOperatorBundle,
    residual: Tensor,
    valid: Tensor,
) -> Tensor:
    operator = bundle.noise
    model = getattr(operator, "model", None)
    if not isinstance(operator, StructuredNoiseOp) or not isinstance(
        model, StructuredDetectorNoise
    ):
        raise ValueError(
            "structured_nll requires StructuredNoiseOp(StructuredDetectorNoise(...))"
        )
    components = {}
    for key, value in (
        ("VAR_POISSON", observation.var_poisson),
        ("VAR_RNOISE", observation.var_rnoise),
        ("VAR_FLAT", observation.var_flat),
    ):
        if value is None:
            continue
        plane = _to(value, device=residual.device, dtype=residual.dtype)
        plane_valid = torch.isfinite(plane) & (plane >= 0.0)
        valid = valid & plane_valid
        components[key] = torch.where(plane_valid, plane, torch.zeros_like(plane))
    err = (
        None
        if observation.err is None
        else _to(observation.err, device=residual.device, dtype=residual.dtype)
    )
    if not components and err is None:
        if observation.weight is None:
            raise ValueError(
                "structured_nll requires variance planes, ERR, or inverse-variance weight"
            )
        weight = _to(observation.weight, device=residual.device, dtype=residual.dtype)
        err = torch.where(
            weight > 0.0,
            weight.clamp_min(torch.finfo(weight.dtype).tiny).rsqrt(),
            torch.ones_like(weight),
        )
    elif not components and err is not None:
        err_valid = torch.isfinite(err) & (err >= 0.0)
        valid = valid & err_valid
        err = torch.where(err_valid, err, torch.ones_like(err))
    return model.negative_log_likelihood(
        residual,
        components=components or None,
        err=err,
        valid=valid,
        amplifier_index=(
            None
            if observation.amplifier_index is None
            else observation.amplifier_index.to(device=residual.device)
        ),
        contamination_prior=(
            None
            if observation.contamination_prior is None
            else _to(
                observation.contamination_prior,
                device=residual.device,
                dtype=residual.dtype,
            )
        ),
    )


def _compute_full_observation_losses(
    observations: Sequence[FieldObservation],
    model: nn.Module,
    obs_forward_ctx: tuple[ForwardContext | None, ...],
    obs_exposure_poses: tuple[ExposurePose | None, ...],
    obs_operator_bundles: tuple[ObservationOperatorBundle, ...],
    *,
    num_spectral_samples: int,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
    loss_reduction: FieldLossReduction,
    gradient_residual_weight: float = 0.0,
    spatial_sampling: FieldSpatialSampling = "center",
    spatial_tile_hw: tuple[int, int] | None = None,
) -> float:
    """Compute per-observation or aggregated reconstruction loss on full images.

    Shared by the pre-training (initial) loss measurement and the post-training
    final loss measurement so both are computed identically.
    """
    gradient_weight = max(float(gradient_residual_weight), 0.0)
    losses: list[float] = []
    global_weighted_loss = None
    global_weight_sum = None
    band_weighted_loss: dict[tuple[object, ...], Tensor] = {}
    band_weight_sum: dict[tuple[object, ...], Tensor] = {}
    for obs_idx, observation in enumerate(observations):
        fctx = obs_forward_ctx[obs_idx]
        pose = obs_exposure_poses[obs_idx]
        observation_bundle = obs_operator_bundles[obs_idx]
        if loss_kind == "gaussian_nll":
            if spatial_tile_hw is not None:
                raise ValueError(
                    "diagnostic spatial tiling is not supported with gaussian_nll"
                )
            pred, pred_logvar = _core_mod.render_field_observation_distribution(
                model,
                observation,
                num_spectral_samples=num_spectral_samples,
                forward_context=fctx,
                exposure_pose=pose,
                operator_bundle=observation_bundle,
                spatial_sampling=spatial_sampling,
            )
        else:
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
        weight_sum = weight.sum()
        if loss_kind == "structured_nll":
            weight_sum = valid.sum().to(dtype=pred.dtype)
            weighted_loss = (
                _structured_noise_loss(observation, observation_bundle, residual, valid)
                * weight_sum
            )
        elif loss_kind == "gaussian_nll":
            weight_sum = valid.sum().to(dtype=pred.dtype)
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
                        None if observation.weight is not None else valid.to(pred.dtype)
                    ),
                )
            )
        else:
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
        if loss_reduction == "per_observation":
            denom = torch.clamp(weight_sum, min=1.0e-8)
            losses.append(float(weighted_loss / denom))
        elif loss_reduction == "per_band":
            band_key = (
                observation.spectral_min,
                observation.spectral_max,
                observation.wavelength,
                observation.band_name,
            )
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
            band_key = (
                observation.spectral_min,
                observation.spectral_max,
                observation.wavelength,
                observation.band_name,
            )
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
    if loss_reduction == "global":
        assert global_weighted_loss is not None and global_weight_sum is not None
        losses = [
            float(global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8))
        ]
    elif loss_reduction == "per_band":
        losses = [
            float(weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8))
            for band_key, weighted in band_weighted_loss.items()
        ]
    elif loss_reduction == "band_global_mix":
        assert global_weighted_loss is not None and global_weight_sum is not None
        band_losses = [
            float(weighted / torch.clamp(band_weight_sum[band_key], min=1.0e-8))
            for band_key, weighted in band_weighted_loss.items()
        ]
        global_loss = float(
            global_weighted_loss / torch.clamp(global_weight_sum, min=1.0e-8)
        )
        losses = [0.5 * (sum(band_losses) + global_loss)]
    return float(sum(losses)) if losses else 0.0


# Late-bound so monkeypatching ``training`` attributes still reaches this code.
from . import training as _core_mod  # noqa: E402
