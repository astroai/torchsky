#!/usr/bin/env python3
"""Controlled loss and Fourier-bandwidth ablations for the WCS-aware INR.

This diagnostic entry point preserves the production science wrapper. It reuses
its band-balanced sampler while exposing three objectives:

- ``huber``: current ERR-whitened Huber loss;
- ``whitened_l2``: pure L2 on ``(prediction - target) / ERR``;
- ``uniform_l2``: pure L2 on normalized residuals with no per-pixel ERR weight.
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.smacs0723_wcs_aware_multiband_inr as base  # noqa: E402
import torchsky.map.wcs_aware_training as training  # noqa: E402
from torchsky.map.wcs_aware_sampling import (  # noqa: E402
    sample_band_balanced_batch,
)


def diagnostic_loss(
    prediction: Tensor,
    target: Tensor,
    err: Tensor,
    mask: Tensor,
    *,
    kind: str = "huber",
    delta: float = 3.0,
    max_relative_weight: float | None = None,
) -> tuple[Tensor, dict[str, Tensor]]:
    """Compute one of the controlled diagnostic objectives."""

    del max_relative_weight

    valid = (
        mask.bool()
        & torch.isfinite(prediction)
        & torch.isfinite(target)
        & torch.isfinite(err)
        & (err > 0)
    )
    if not bool(valid.any()):
        raise RuntimeError("batch has no valid pixels")

    residual = (prediction[valid] - target[valid]).float()
    err_valid = err[valid].float()
    whitened = residual / err_valid

    if kind == "huber":
        delta_value = float(delta)
        if not math.isfinite(delta_value) or delta_value <= 0.0:
            raise ValueError("delta must be finite and positive")
        absolute = whitened.abs()
        rho = torch.where(
            absolute <= delta_value,
            0.5 * whitened.square(),
            delta_value * (absolute - 0.5 * delta_value),
        )
    elif kind == "whitened_l2":
        rho = 0.5 * whitened.square()
    elif kind == "uniform_l2":
        # A single detached scale keeps optimization numerically comparable while
        # removing all relative per-pixel ERR weighting.
        constant_scale = err_valid.median().detach().clamp_min(1.0e-12)
        rho = 0.5 * (residual / constant_scale).square()
    else:
        raise ValueError("kind must be huber, whitened_l2, or uniform_l2")

    return rho.mean(), {
        "rmse": residual.square().mean().sqrt().detach(),
        "mae": residual.abs().mean().detach(),
        "whitened_rmse": whitened.square().mean().sqrt().detach(),
        "whitened_mae": whitened.abs().mean().detach(),
        "valid_fraction": valid.float().mean().detach(),
    }


training.weighted_robust_loss = diagnostic_loss
training.sample_balanced_batch = sample_band_balanced_batch
base.train_model = training.train_model

_original_build_parser = base.build_parser


def build_parser() -> argparse.ArgumentParser:
    parser = _original_build_parser()
    for action in parser._actions:
        if action.dest == "batch_size":
            action.default = 1080
        elif action.dest == "robust_loss":
            action.choices = ("huber", "whitened_l2", "uniform_l2")
            action.default = "huber"
        elif action.dest == "robust_delta":
            action.default = 3.0
    return parser


base.build_parser = build_parser


if __name__ == "__main__":
    raise SystemExit(base.main())
