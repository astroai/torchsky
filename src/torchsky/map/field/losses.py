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
    raise ValueError(f"unsupported field loss kind {loss_kind!r}")

__all__ = ["_field_residual_loss"]
