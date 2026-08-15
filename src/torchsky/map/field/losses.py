"""Loss helpers shared by field fitting and anchor regularizers."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import Tensor

from .types import FieldLossKind


def _field_residual_loss(
    residual: Tensor,
    *,
    loss_kind: FieldLossKind,
    charbonnier_eps: float,
    huber_delta: float,
    logvar: Tensor | None = None,
) -> Tensor:
    # ⚡ Bolt: Element-wise multiplication is faster than torch.square or **2
    if loss_kind == "mse":
        return residual * residual
    if loss_kind == "charbonnier":
        eps = torch.as_tensor(
            float(charbonnier_eps),
            device=residual.device,
            dtype=residual.dtype,
        )
        return torch.sqrt((residual * residual) + (eps * eps))
    if loss_kind == "huber":
        return F.huber_loss(
            residual,
            torch.zeros_like(residual),
            reduction="none",
            delta=float(huber_delta),
        )
    if loss_kind == "gaussian_nll":
        if logvar is None:
            raise ValueError("gaussian_nll requires predictive log-variance")
        clipped = torch.clamp(logvar, min=-8.0, max=6.0)
        return 0.5 * (torch.exp(-clipped) * (residual * residual) + clipped)
    if loss_kind == "structured_nll":
        raise ValueError(
            "structured_nll requires a full FieldObservation and StructuredNoiseOp"
        )
    raise ValueError(f"unsupported field loss kind {loss_kind!r}")


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
