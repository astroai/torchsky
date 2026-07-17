"""Calibration helpers for the WCS-aware multi-wavelength INR.

The public API is split into focused bandpass and PSF modules. The official
bandpass path queries JWST/NIRCam throughput curves from SVO FPS through
``astroquery.svo_fps`` and snapshots the returned curves for reproducibility.
"""

from .wcs_aware_calibration_bandpass import (
    SVO_DEFAULT_COLLECTION,
    build_bandpass_calibration,
    load_svo_transmission_curve,
)
from .wcs_aware_calibration_psf import build_psf_calibration, normalize_detector
from .wcs_aware_calibration_types import (
    BandpassCalibration,
    BandpassMode,
    ExposureCalibrationInput,
    PSFCalibration,
    PSFMode,
)

__all__ = [
    "BandpassCalibration",
    "BandpassMode",
    "ExposureCalibrationInput",
    "PSFCalibration",
    "PSFMode",
    "SVO_DEFAULT_COLLECTION",
    "build_bandpass_calibration",
    "build_psf_calibration",
    "load_svo_transmission_curve",
    "normalize_detector",
]
