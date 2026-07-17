"""PSF calibration for the WCS-aware multi-wavelength INR."""

from __future__ import annotations

import importlib
import json
import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from torch import Tensor

from .wcs_aware_calibration_types import (
    ExposureCalibrationInput,
    PSFCalibration,
    PSFMode,
)
from .wcs_aware_inr import PSFKernelBank, gaussian_psf_kernel


def normalize_detector(value: str | None) -> str:
    if not value:
        return ""
    token = str(value).strip().upper().replace(" ", "")
    return {"NRCALONG": "NRCA5", "NRCBLONG": "NRCB5"}.get(token, token)


def _load_kernel_file(
    path: Path,
    *,
    num_samples: int,
    expected_wavelength: Tensor,
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        with np.load(path) as payload:
            if "kernel" not in payload:
                raise ValueError(f"{path} must contain a 'kernel' array")
            array = np.asarray(payload["kernel"], dtype=np.float32)
            if "wavelength" in payload:
                saved = np.asarray(payload["wavelength"], dtype=np.float32)
                expected = expected_wavelength.detach().cpu().numpy()
                if saved.shape != expected.shape or not np.allclose(
                    saved, expected, rtol=1.0e-6, atol=1.0e-3
                ):
                    raise ValueError(f"{path} wavelength samples do not match bandpass")
    else:
        array = np.asarray(np.load(path), dtype=np.float32)
    if array.ndim == 2:
        array = np.repeat(array[None, :, :], num_samples, axis=0)
    if array.ndim != 3 or array.shape[0] != num_samples:
        raise ValueError(f"{path} must have shape [K,K] or [{num_samples},K,K]")
    return array


def _kernel_candidate(kernel_dir: Path, exposure: ExposureCalibrationInput) -> Path:
    candidates = [
        kernel_dir / f"{exposure.path.stem}.npz",
        kernel_dir / f"{exposure.path.stem}.npy",
        kernel_dir / f"{exposure.band}_{normalize_detector(exposure.detector)}.npz",
        kernel_dir / f"{exposure.band}_{normalize_detector(exposure.detector)}.npy",
        kernel_dir / f"{exposure.band}.npz",
        kernel_dir / f"{exposure.band}.npy",
    ]
    found = next((path for path in candidates if path.exists()), None)
    if found is None:
        raise FileNotFoundError(
            f"no PSF kernel for {exposure.path.name}; tried: {candidates}"
        )
    return found


def _import_stpsf_module():
    errors: list[Exception] = []
    for module_name in ("stpsf", "webbpsf"):
        try:
            return importlib.import_module(module_name)
        except ImportError as exc:
            errors.append(exc)
    raise RuntimeError(
        "STPSF/WebbPSF is required for psf_mode='stpsf'"
    ) from errors[-1]


def _static_nircam_simulator(module, exposure: ExposureCalibrationInput):
    simulator = module.NIRCam()
    simulator.filter = exposure.band
    detector = normalize_detector(exposure.detector)
    if detector:
        simulator.detector = detector
    return simulator


def _stpsf_simulator(
    exposure: ExposureCalibrationInput,
    *,
    allow_static_fallback: bool,
):
    module = _import_stpsf_module()
    setup = getattr(module, "setup_sim_to_match_file", None)
    if callable(setup):
        try:
            simulator = setup(str(exposure.path))
            return module, simulator, "matched_to_exposure"
        except Exception:
            if not allow_static_fallback:
                raise
    if not allow_static_fallback:
        raise RuntimeError(
            "could not construct an exposure-matched STPSF simulator; "
            "use allow_static_stpsf_fallback only when that approximation is acceptable"
        )
    return module, _static_nircam_simulator(module, exposure), "static_header_fallback"


def _stpsf_stack(
    exposure: ExposureCalibrationInput,
    wavelength: Tensor,
    *,
    kernel_size: int,
    extension: str,
    allow_static_fallback: bool,
) -> tuple[np.ndarray, dict[str, Any]]:
    module, simulator, setup_mode = _stpsf_simulator(
        exposure,
        allow_static_fallback=allow_static_fallback,
    )
    simulator.detector_position = (
        (exposure.shape[1] - 1) / 2.0,
        (exposure.shape[0] - 1) / 2.0,
    )
    kernels: list[np.ndarray] = []
    for value in wavelength.detach().cpu().tolist():
        result = simulator.calc_psf(
            monochromatic=float(value) * 1.0e-10,
            fov_pixels=int(kernel_size),
            oversample=1,
        )
        try:
            kernel = np.array(result[extension].data, dtype=np.float32, copy=True)
        except (KeyError, IndexError) as exc:
            names = [hdu.name for hdu in result]
            raise KeyError(
                f"STPSF extension {extension!r} unavailable; found {names}"
            ) from exc
        finally:
            result.close()
        kernel = np.clip(kernel, 0.0, None)
        mass = float(kernel.sum())
        if not math.isfinite(mass) or mass <= 0.0:
            raise ValueError("STPSF produced a non-positive kernel")
        kernels.append(kernel / mass)
    return np.stack(kernels), {
        "package": module.__name__,
        "package_version": str(getattr(module, "__version__", "unknown")),
        "setup_mode": setup_mode,
        "extension": extension,
        "detector_position": list(simulator.detector_position),
    }


def build_psf_calibration(
    exposures: Sequence[ExposureCalibrationInput],
    bandpass_wavelength: Tensor,
    *,
    mode: PSFMode,
    kernel_size: int,
    gaussian_fwhm_by_band: dict[str, float] | None = None,
    default_gaussian_fwhm: float | None = None,
    kernel_dir: Path | None = None,
    cache_dir: Path | None = None,
    stpsf_extension: str = "DET_DIST",
    allow_static_stpsf_fallback: bool = False,
) -> PSFCalibration:
    if mode == "none":
        return PSFCalibration(None, [])
    if bandpass_wavelength.ndim != 2 or bandpass_wavelength.shape[0] != len(exposures):
        raise ValueError("bandpass_wavelength must have shape [E,L]")
    if kernel_size <= 0 or kernel_size % 2 != 1:
        raise ValueError("kernel_size must be a positive odd integer")
    gaussian_fwhm_by_band = gaussian_fwhm_by_band or {}
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    stacks: list[Tensor] = []
    metadata: list[dict[str, Any]] = []
    for index, exposure in enumerate(exposures):
        wavelength = bandpass_wavelength[index]
        info: dict[str, Any] = {
            "path": str(exposure.path),
            "band": exposure.band,
            "detector": normalize_detector(exposure.detector),
            "mode": mode,
        }
        if mode == "gaussian":
            fwhm = gaussian_fwhm_by_band.get(exposure.band, default_gaussian_fwhm)
            if fwhm is None:
                raise ValueError(f"no Gaussian PSF FWHM supplied for {exposure.band}")
            base = gaussian_psf_kernel(float(fwhm), size=kernel_size)
            array = base[None].repeat(wavelength.numel(), 1, 1).numpy()
            info["fwhm_pixels"] = float(fwhm)
        elif mode == "npy":
            if kernel_dir is None:
                raise ValueError("kernel_dir is required for npy PSF mode")
            source = _kernel_candidate(kernel_dir, exposure)
            array = _load_kernel_file(
                source,
                num_samples=int(wavelength.numel()),
                expected_wavelength=wavelength,
            )
            info["source"] = str(source)
        elif mode == "stpsf":
            if cache_dir is None:
                raise ValueError("cache_dir is required for stpsf PSF mode")
            cache = cache_dir / (
                f"{exposure.path.stem}__L{wavelength.numel()}__K{kernel_size}__"
                f"{stpsf_extension}.npz"
            )
            expected = wavelength.detach().cpu().numpy().astype(np.float32)
            array: np.ndarray | None = None
            stpsf_info: dict[str, Any] = {}
            if cache.exists():
                with np.load(cache) as payload:
                    saved_wavelength = np.asarray(payload["wavelength"], dtype=np.float32)
                    saved_kernel = np.asarray(payload["kernel"], dtype=np.float32)
                if saved_wavelength.shape == expected.shape and np.allclose(
                    saved_wavelength, expected, rtol=1.0e-6, atol=1.0e-3
                ):
                    array = saved_kernel
                    stpsf_info["cache_hit"] = True
            if array is None:
                array, generated = _stpsf_stack(
                    exposure,
                    wavelength,
                    kernel_size=kernel_size,
                    extension=stpsf_extension,
                    allow_static_fallback=allow_static_stpsf_fallback,
                )
                stpsf_info.update(generated)
                stpsf_info["cache_hit"] = False
                np.savez_compressed(
                    cache,
                    wavelength=expected,
                    kernel=array,
                    metadata=json.dumps(generated),
                )
            info.update(stpsf_info)
            info["source"] = str(cache)
        else:
            raise ValueError(f"unsupported PSF mode: {mode}")

        array = np.asarray(array, dtype=np.float32)
        if array.ndim != 3 or tuple(array.shape[-2:]) != (kernel_size, kernel_size):
            raise ValueError(
                f"PSF stack for {exposure.path.name} must have shape "
                f"[L,{kernel_size},{kernel_size}], got {array.shape}"
            )
        array = np.flip(array, axis=(-2, -1)).copy()
        info["stored_kernel_orientation"] = "flipped_for_convolution"
        stacks.append(torch.as_tensor(array, dtype=torch.float32))
        metadata.append(info)

    bank = PSFKernelBank(
        torch.stack(stacks, dim=0),
        wavelength=bandpass_wavelength.detach().cpu().float(),
    ).normalized()
    return PSFCalibration(bank, metadata)


__all__ = ["build_psf_calibration", "normalize_detector"]
