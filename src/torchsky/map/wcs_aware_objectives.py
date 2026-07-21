"""Loss functions for the WCS-aware multi-wavelength INR."""

from __future__ import annotations

import math

import torch
from torch import Tensor


def err_whitened_robust_loss(
    prediction: Tensor,
    target: Tensor,
    err: Tensor,
    mask: Tensor,
    *,
    kind: str = "huber",
    delta: float = 3.0,
    max_relative_weight: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Apply an objective to residuals whitened by the per-pixel ERR map.

    The optimized residual is ``z = (prediction - target) / err``. Supported
    objectives are:

    - ``huber``: quadratic for ``|z| <= delta`` and linear outside;
    - ``charbonnier``: smooth robust penalty with scale ``delta``;
    - ``whitened_l2``: un-clipped ``0.5 * z**2``.

    A Huber threshold of ``delta=3`` has the direct interpretation of a
    three-sigma transition. No second inverse-variance weighting is applied
    after whitening.

    ``max_relative_weight`` is accepted only for compatibility with the previous
    training API and is intentionally ignored.
    """

    _ = max_relative_weight
    delta_value = float(delta)
    if kind in {"huber", "charbonnier"}:
        if not math.isfinite(delta_value) or delta_value <= 0.0:
            raise ValueError("delta must be finite and positive")

    valid = (
        mask.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & torch.isfinite(err)
        & (err > 0)
    )
    if not bool(valid.any()):
        raise RuntimeError("batch has no valid pixels")

    # Keep the statistical normalization in float32 even under CUDA autocast.
    residual = (prediction[valid] - target[valid]).float()
    whitened = residual / err[valid].float()
    if kind == "huber":
        absolute = whitened.abs()
        rho = torch.where(
            absolute <= delta_value,
            0.5 * whitened.square(),
            delta_value * (absolute - 0.5 * delta_value),
        )
    elif kind == "charbonnier":
        scaled = whitened / delta_value
        rho = delta_value * delta_value * (torch.sqrt(1.0 + scaled.square()) - 1.0)
    elif kind == "whitened_l2":
        rho = 0.5 * whitened.square()
    else:
        raise ValueError("kind must be huber, charbonnier, or whitened_l2")

    return rho.mean(), {
        "rmse": residual.square().mean().sqrt().detach(),
        "mae": residual.abs().mean().detach(),
        "whitened_rmse": whitened.square().mean().sqrt().detach(),
        "whitened_mae": whitened.abs().mean().detach(),
        "valid_fraction": valid.float().mean().detach(),
    }


__all__ = ["err_whitened_robust_loss"]
