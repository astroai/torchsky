"""Calibration helpers for the WCS-aware multi-wavelength INR.

The production paths use official tabulated throughput curves and
per-exposure/per-wavelength STPSF kernels. The simple center, top-hat,
Gaussian-bandpass, no-PSF, Gaussian-PSF, and NPY-kernel paths remain available
for debugging and ablation studies.
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
from torch import Tensor

from .field.types import FieldObservation
from .wcs_aware_inr import (
    PSFKernelBank,
    gaussian_psf_kernel,
    native_bandpass_quadrature_bank,
)

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


def normalize_detector(value: str | None) -> str:
    if not value:
        return ""
    token = str(value).strip().upper().replace(" ", "")
    return {"NRCALONG": "NRCA5", "NRCBLONG": "NRCB5"}.get(token, token)


def _detector_aliases(detector: str) -> tuple[str, ...]:
    detector = normalize_detector(detector)
    aliases = {detector}
    if detector.startswith("NRC") and len(detector) >= 5:
        aliases.add(detector[3:])
    if detector.endswith("5") and detector.startswith("NRCA"):
        aliases.add("NRCALONG")
        aliases.add("ALONG")
    if detector.endswith("5") and detector.startswith("NRCB"):
        aliases.add("NRCBLONG")
        aliases.add("BLONG")
    return tuple(sorted(alias for alias in aliases if alias))


def _read_two_column_curve(path: Path) -> tuple[np.ndarray, np.ndarray]:
    array = np.genfromtxt(path, comments="#", dtype=np.float64)
    if getattr(array.dtype, "names", None) is not None:
        names = list(array.dtype.names or ())
        if len(names) < 2:
            raise ValueError(f"{path} must contain at least two columns")
        wavelength = np.asarray(array[names[0]], dtype=np.float64)
        throughput = np.asarray(array[names[1]], dtype=np.float64)
    else:
        array = np.atleast_2d(array)
        if array.shape[1] < 2:
            raise ValueError(f"{path} must contain at least two columns")
        wavelength = np.asarray(array[:, 0], dtype=np.float64)
        throughput = np.asarray(array[:, 1], dtype=np.float64)
    finite = np.isfinite(wavelength) & np.isfinite(throughput)
    wavelength = wavelength[finite]
    throughput = np.maximum(throughput[finite], 0.0)
    if wavelength.size < 2:
        raise ValueError(f"{path} has fewer than two valid samples")
    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    throughput = throughput[order]
    if float(wavelength.max()) < 100.0:  # official NIRCam tables are usually micron
        wavelength = wavelength * 1.0e4
    if float(throughput.max()) <= 0.0:
        raise ValueError(f"{path} has no positive throughput")
    return wavelength, throughput


def _load_manifest(path: Path | None) -> dict[str, Path]:
    if path is None:
        return {}
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("throughput manifest must be a JSON object")
    out: dict[str, Path] = {}
    for key, value in raw.items():
        candidate = Path(str(value)).expanduser()
        if not candidate.is_absolute():
            candidate = (path.parent / candidate).resolve()
        out[str(key).upper()] = candidate
    return out


def _manifest_curve(
    manifest: dict[str, Path],
    *,
    band: str,
    detector: str,
    allow_mean: bool,
) -> Path | None:
    aliases = _detector_aliases(detector)
    keys: list[str] = []
    for alias in aliases:
        keys.extend(
            [
                f"{alias}/{band}",
                f"{alias}:{band}",
                f"{band}/{alias}",
                f"{band}:{alias}",
            ]
        )
    if allow_mean:
        keys.extend([band, f"MEAN/{band}", f"{band}/MEAN"])
    for key in keys:
        if key in manifest:
            return manifest[key]
    return None


def _discover_curve(
    root: Path,
    *,
    band: str,
    detector: str,
    allow_mean: bool,
) -> Path:
    detector = normalize_detector(detector)
    aliases = _detector_aliases(detector)
    scored: list[tuple[int, str, Path]] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name.upper()
        if band not in name:
            continue
        detector_match = any(alias in name for alias in aliases)
        mean_match = "MEAN" in name or "AVERAGE" in name
        if not detector_match and not (allow_mean and mean_match):
            continue
        score = 100 if detector_match else 10
        score += 3 if "THROUGHPUT" in name else 0
        score += 1 if path.suffix.lower() in {".txt", ".dat", ".csv", ".ecsv"} else 0
        scored.append((score, name, path))
    if not scored:
        detail = "detector-specific or mean" if allow_mean else "detector-specific"
        raise FileNotFoundError(
            f"no {detail} throughput curve for {band}/{detector} under {root}"
        )
    scored.sort(key=lambda item: (-item[0], item[1]))
    return scored[0][2]


def resolve_throughput_curve(
    exposure: ExposureCalibrationInput,
    *,
    throughput_dir: Path,
    throughput_manifest: Path | None = None,
    allow_mean: bool = False,
) -> Path:
    manifest = _load_manifest(throughput_manifest)
    selected = _manifest_curve(
        manifest,
        band=exposure.band,
        detector=exposure.detector,
        allow_mean=allow_mean,
    )
    if selected is not None:
        if not selected.exists():
            raise FileNotFoundError(selected)
        return selected
    return _discover_curve(
        throughput_dir,
        band=exposure.band,
        detector=exposure.detector,
        allow_mean=allow_mean,
    )


def _positive_support(
    wavelength: np.ndarray,
    throughput: np.ndarray,
    *,
    relative_threshold: float = 1.0e-6,
) -> tuple[float, float]:
    threshold = float(throughput.max()) * relative_threshold
    positive = np.flatnonzero(throughput > threshold)
    if positive.size == 0:
        positive = np.flatnonzero(throughput > 0.0)
    first = max(int(positive[0]) - 1, 0)
    last = min(int(positive[-1]) + 1, wavelength.size - 1)
    return float(wavelength[first]), float(wavelength[last])


def _bandpass_observation(
    exposure: ExposureCalibrationInput,
    *,
    mode: BandpassMode,
    band_specs: dict[str, tuple[float, float, float]],
    throughput_dir: Path | None,
    throughput_manifest: Path | None,
    allow_mean_throughput: bool,
) -> tuple[FieldObservation, dict[str, Any]]:
    center, lower, upper = band_specs[exposure.band]
    kwargs: dict[str, Any] = {
        "image": torch.zeros((1, 1), dtype=torch.float32),
        "header": exposure.header,
        "wavelength": center,
        "band_name": exposure.band,
    }
    metadata: dict[str, Any] = {
        "band": exposure.band,
        "detector": normalize_detector(exposure.detector),
        "mode": mode,
    }
    if mode == "official":
        if throughput_dir is None:
            raise ValueError("throughput_dir is required for official bandpass mode")
        source = resolve_throughput_curve(
            exposure,
            throughput_dir=throughput_dir,
            throughput_manifest=throughput_manifest,
            allow_mean=allow_mean_throughput,
        )
        wavelength, throughput = _read_two_column_curve(source)
        support_min, support_max = _positive_support(wavelength, throughput)
        kwargs.update(
            spectral_min=support_min,
            spectral_max=support_max,
            transmission_wavelength=torch.as_tensor(wavelength, dtype=torch.float32),
            transmission_throughput=torch.as_tensor(throughput, dtype=torch.float32),
        )
        metadata.update(
            source=str(source),
            support=[support_min, support_max],
            table_samples=int(wavelength.size),
        )
    elif mode == "center":
        pass
    elif mode == "top_hat":
        kwargs.update(
            spectral_min=lower,
            spectral_max=upper,
            transmission_center=center,
        )
    elif mode == "gaussian":
        kwargs.update(
            spectral_min=lower,
            spectral_max=upper,
            transmission_center=center,
            transmission_sigma=max((upper - lower) / 2.354820045, 1.0e-6),
        )
    else:
        raise ValueError(f"unsupported bandpass mode: {mode}")
    return FieldObservation(**kwargs), metadata


def build_bandpass_calibration(
    exposures: Sequence[ExposureCalibrationInput],
    *,
    mode: BandpassMode,
    num_samples: int,
    band_specs: dict[str, tuple[float, float, float]],
    throughput_dir: Path | None = None,
    throughput_manifest: Path | None = None,
    allow_mean_throughput: bool = False,
) -> BandpassCalibration:
    if not exposures:
        raise ValueError("at least one exposure is required")
    observations: list[FieldObservation] = []
    metadata: list[dict[str, Any]] = []
    for exposure in exposures:
        observation, info = _bandpass_observation(
            exposure,
            mode=mode,
            band_specs=band_specs,
            throughput_dir=throughput_dir,
            throughput_manifest=throughput_manifest,
            allow_mean_throughput=allow_mean_throughput,
        )
        observations.append(observation)
        metadata.append(info)
    samples = 1 if mode == "center" else int(num_samples)
    wavelength, weight = native_bandpass_quadrature_bank(
        observations,
        num_spectral_samples=samples,
    )
    for index, info in enumerate(metadata):
        info["quadrature_wavelength"] = wavelength[index].tolist()
        info["quadrature_weight"] = weight[index].tolist()
    return BandpassCalibration(wavelength, weight, metadata)


def _load_kernel_file(
    path: Path,
    *,
    num_samples: int,
    expected_wavelength: Tensor,
) -> np.ndarray:
    if path.suffix.lower() == ".npz":
        payload = np.load(path)
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
            monochromatic=float(value) * 1.0e-10,  # Angstrom -> metre
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
            fwhm = gaussian_fwhm_by_band.get(
                exposure.band, default_gaussian_fwhm
            )
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
                payload = np.load(cache)
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
        # The core renderer evaluates neighboring scene samples with a
        # correlation-style gather. Flip the optical impulse response so the
        # resulting operation is the mathematical convolution K * S, including
        # for asymmetric STPSF kernels.
        array = np.flip(np.asarray(array, dtype=np.float32), axis=(-2, -1)).copy()
        info["stored_kernel_orientation"] = "flipped_for_convolution"
        stacks.append(torch.as_tensor(array, dtype=torch.float32))
        metadata.append(info)
    bank = PSFKernelBank(
        torch.stack(stacks, dim=0),
        wavelength=bandpass_wavelength.detach().cpu().float(),
    ).normalized()
    return PSFCalibration(bank, metadata)


__all__ = [
    "BandpassCalibration",
    "BandpassMode",
    "ExposureCalibrationInput",
    "PSFCalibration",
    "PSFMode",
    "build_bandpass_calibration",
    "build_psf_calibration",
    "normalize_detector",
    "resolve_throughput_curve",
]
