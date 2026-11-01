"""Loss helpers shared by field fitting and anchor regularizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import FieldLossKind


# exp(+-20) spans ~1e-9..5e8 and stays finite in float32 (exp overflows at ~88).
# The map is the identity for |logvar| <= 10 (variances 4.5e-5..2.2e4, so ordinary
# data scales are untouched) and a C1 tanh saturation toward +-20 beyond, so the
# gradient never vanishes exactly, unlike a hard clamp.
LOGVAR_LINEAR_BOUND = 10.0
LOGVAR_SOFT_BOUND = 20.0


def soft_bound_logvar(logvar: Tensor) -> Tensor:
    """Identity inside ``|logvar| <= 10``; smooth tanh saturation toward ``+-20``."""
    span = LOGVAR_SOFT_BOUND - LOGVAR_LINEAR_BOUND
    magnitude = logvar.abs()
    excess = (magnitude - LOGVAR_LINEAR_BOUND).clamp_min(0.0)
    saturated = LOGVAR_LINEAR_BOUND + span * torch.tanh(excess / span)
    return torch.where(
        magnitude > LOGVAR_LINEAR_BOUND, torch.sign(logvar) * saturated, logvar
    )


def _field_residual_loss(
    residual: Tensor,
    *,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
    logvar: Tensor | None = None,
    observation_invvar: Tensor | None = None,
    sample_weight: Tensor | None = None,
) -> Tensor:
    valid_invvar = (
        None
        if observation_invvar is None
        else torch.isfinite(observation_invvar) & (observation_invvar > 0.0)
    )
    valid_weight = (
        None
        if sample_weight is None
        else torch.isfinite(sample_weight) & (sample_weight > 0.0)
    )
    valid = valid_invvar
    if valid_weight is not None:
        valid = valid_weight if valid is None else valid & valid_weight
    safe_residual = residual if valid is None else torch.where(valid, residual, 0.0)
    # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
    if loss_kind == "mse":
        loss = safe_residual * safe_residual
    elif loss_kind == "charbonnier":
        eps = torch.as_tensor(
            float(charbonnier_eps),
            device=safe_residual.device,
            dtype=safe_residual.dtype,
        )
        loss = torch.sqrt((safe_residual * safe_residual) + (eps * eps))
    elif loss_kind == "huber":
        loss = F.huber_loss(
            safe_residual,
            torch.zeros_like(safe_residual),
            reduction="none",
            delta=float(huber_delta),
        )
    elif loss_kind == "gaussian_nll":
        if logvar is None:
            raise ValueError("gaussian_nll requires predictive log-variance")
        safe_logvar = logvar if valid is None else torch.where(valid, logvar, 0.0)
        clipped = soft_bound_logvar(safe_logvar)
        total_variance = torch.exp(clipped)
        if observation_invvar is not None:
            assert valid_invvar is not None
            safe_invvar = torch.where(
                valid_invvar,
                observation_invvar,
                torch.ones_like(observation_invvar),
            )
            total_variance = total_variance + safe_invvar.reciprocal()
        loss = 0.5 * (
            (safe_residual * safe_residual) / total_variance
            + torch.log(total_variance)
        )
        if observation_invvar is not None:
            loss = torch.where(valid_invvar, loss, torch.zeros_like(loss))
    elif loss_kind == "structured_nll":
        raise ValueError(
            "structured_nll requires a full FieldObservation and StructuredNoiseOp"
        )
    else:
        raise ValueError(f"unsupported field loss kind {loss_kind!r}")
    if sample_weight is not None:
        assert valid_weight is not None
        safe_weight = torch.where(
            valid_weight, sample_weight, torch.zeros_like(sample_weight)
        )
        loss = torch.where(
            valid_weight,
            loss * safe_weight,
            torch.zeros_like(loss),
        )
    return loss


def _field_gradient_residual_weighted_loss(
    prediction: Tensor,
    target: Tensor,
    weight: Tensor,
    valid: Tensor,
    *,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
) -> tuple[Tensor, Tensor]:
    """Return weighted image-gradient residual numerator and denominator."""
    if prediction.ndim != 2 or target.ndim != 2:
        raise ValueError("gradient residual loss expects 2D prediction and target")
    if weight.shape != prediction.shape or valid.shape != prediction.shape:
        raise ValueError("weight and valid masks must match prediction shape")
    gradient_loss_kind: FieldLossKind = (
        "mse" if loss_kind in {"gaussian_nll", "structured_nll"} else loss_kind
    )

    grad_num = torch.zeros((), device=prediction.device, dtype=prediction.dtype)
    grad_den = torch.zeros((), device=prediction.device, dtype=prediction.dtype)

    if prediction.shape[-1] > 1:
        pred_dx = prediction[:, 1:] - prediction[:, :-1]
        target_dx = target[:, 1:] - target[:, :-1]
        valid_dx = valid[:, 1:] & valid[:, :-1]
        weight_dx = 0.5 * (weight[:, 1:] + weight[:, :-1])
        weight_dx = torch.where(valid_dx, weight_dx, torch.zeros_like(weight_dx))
        loss_dx = _field_residual_loss(
            pred_dx - target_dx,
            loss_kind=gradient_loss_kind,
            charbonnier_eps=charbonnier_eps,
            huber_delta=huber_delta,
            logvar=None,
        )
        grad_num = grad_num + torch.sum(weight_dx * loss_dx)
        grad_den = grad_den + weight_dx.sum()

    if prediction.shape[-2] > 1:
        pred_dy = prediction[1:, :] - prediction[:-1, :]
        target_dy = target[1:, :] - target[:-1, :]
        valid_dy = valid[1:, :] & valid[:-1, :]
        weight_dy = 0.5 * (weight[1:, :] + weight[:-1, :])
        weight_dy = torch.where(valid_dy, weight_dy, torch.zeros_like(weight_dy))
        loss_dy = _field_residual_loss(
            pred_dy - target_dy,
            loss_kind=gradient_loss_kind,
            charbonnier_eps=charbonnier_eps,
            huber_delta=huber_delta,
            logvar=None,
        )
        grad_num = grad_num + torch.sum(weight_dy * loss_dy)
        grad_den = grad_den + weight_dy.sum()

    return grad_num, grad_den


__all__ = ["_field_gradient_residual_weighted_loss", "_field_residual_loss"]
