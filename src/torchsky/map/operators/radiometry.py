"""Radiometric calibration helpers shared across mapmaking paths."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from .base import RadiometryOp


@dataclass
class ExplicitRadiometryOp(RadiometryOp):
    def __init__(self, *, name: str = "explicit_radiometry") -> None:
        super().__init__(name=name, learned=False)


def apply_image_observation_calibration(
    *,
    photometric_scale: float,
    background_offset: float,
    image: Tensor,
    source_weight: Tensor,
) -> tuple[Tensor, Tensor]:
    scale = float(photometric_scale)
    background = float(background_offset)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("Observation photometric_scale must be finite and positive")
    if not math.isfinite(background):
        raise ValueError("Observation background_offset must be finite")
    calibrated_image = scale * (
        torch.nan_to_num(image.to(dtype=torch.float32), nan=0.0, posinf=0.0, neginf=0.0)
        - background
    )
    calibrated_weight = source_weight.to(dtype=torch.float32) / (scale * scale)
    return calibrated_image, calibrated_weight
