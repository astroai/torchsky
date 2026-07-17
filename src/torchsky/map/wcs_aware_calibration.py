"""Calibration helpers for the WCS-aware multi-wavelength INR.

The production bandpass path queries current JWST/NIRCam throughput curves from
SVO FPS through astroquery. The optional production PSF path uses
per-exposure/per-wavelength STPSF kernels. Simpler center, top-hat, Gaussian
bandpasses and no-PSF, Gaussian-PSF, and NPY-kernel paths remain available for
debugging and ablation studies.
"""

from __future__ import annotations

from dataclasses import dataclass
import importlib
import json
import math
from pathlib import Path
from typing import Any, Literal, Sequence

import numpy as np
import torch
from astropy import units as u
from torch import Tensor

from .field.types import FieldObservation
from .wcs_aware_inr import (
    PSFKernelBank,
    gaussian_psf_kernel,
    native_bandpass_quadrature_bank,
)

BandpassMode = Literal["official", "center", "top_hat", "gaussian"]
PSFMode = Literal["none", "gaussian", "npy", "stpsf"]
SVO_DEFAULT_COLLECTION = "JWST/NIRCam2025"


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
    metadata: list[dict[str, Any