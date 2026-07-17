"""Shared types for WCS-aware bandpass and PSF calibration."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from torch import Tensor

from .wcs_aware_inr import PSFKernelBank

BandpassMode = Literal["official", "center", "top_hat", "gaussian"]
PSFMode = Literal["none", "gaussian", "npy", "stpsf"]


@dataclass(frozen=True)
class ExposureCalibrationInput:
    path: Path
    band: str
    detector: str
    header: dict[str, Any]
    shape: tuple[int, int]


@dataclass(frozen=True)
class BandpassCalibration:
    wavelength: Tensor  # [E, L]
    weight: Tensor  # [E, L]
    metadata: list[dict[str, Any]]


@dataclass(frozen=True)
class PSFCalibration:
    bank: PSFKernelBank | None
    metadata: list[dict[str, Any]]


__all__ = [
    "BandpassCalibration",
    "BandpassMode",
    "ExposureCalibrationInput",
    "PSFCalibration",
    "PSFMode",
]
