"""Bandpass calibration for the WCS-aware multi-wavelength INR."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from astropy import units as u

from .field.types import FieldObservation
from .wcs_aware_calibration_types import (
    BandpassCalibration,
    BandpassMode,
    ExposureCalibrationInput,
)
from .wcs_aware_inr import native_bandpass_quadrature_bank

SVO_DEFAULT_COLLECTION = "JWST/NIRCam2025"


def _normalise_svo_collection(value: str) -> str:
    collection = str(value).strip().rstrip(".")
    if "/" not in collection:
        raise ValueError("SVO collection must look like 'JWST/NIRCam2025'")
    return collection


def _svo_filter_id(band: str, *, collection: str) -> str:
    token = str(band).strip().upper()
    if not token:
        raise ValueError("band must be non-empty")
    return f"{_normalise_svo_collection(collection)}.{token}"


def _svo_cache_path(cache_dir: Path, filter_id: str) -> Path:
    safe = "".join(
        ch if ch.isalnum() or ch in "._-" else "_" for ch in filter_id
    )
    return cache_dir / f"{safe}.npz"


def _as_float64(column: Any) -> np.ndarray:
    return np.asarray(np.ma.filled(np.ma.asarray(column), np.nan), dtype=np.float64)


def _clean_curve(
    wavelength_angstrom: np.ndarray,
    transmission: np.ndarray,
    *,
    filter_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    wavelength = np.asarray(wavelength_angstrom, dtype=np.float64).reshape(-1)
    throughput = np.asarray(transmission, dtype=np.float64).reshape(-1)
    if wavelength.shape != throughput.shape:
        raise ValueError(f"SVO curve {filter_id} has mismatched array shapes")

    finite = np.isfinite(wavelength) & np.isfinite(throughput)
    wavelength = wavelength[finite]
    throughput = throughput[finite]
    if wavelength.size < 2:
        raise ValueError(f"SVO curve {filter_id} has fewer than two finite samples")
    if bool((wavelength <= 0.0).any()):
        raise ValueError(f"SVO curve {filter_id} contains non-positive wavelengths")
    if float(throughput.min()) < -1.0e-6:
        raise ValueError(f"SVO curve {filter_id} contains negative transmission")
    if float(throughput.max()) > 1.0 + 1.0e-3:
        raise ValueError(f"SVO curve {filter_id} has transmission above one")
    throughput = np.clip(throughput, 0.0, 1.0)

    order = np.argsort(wavelength)
    wavelength = wavelength[order]
    throughput = throughput[order]
    unique, inverse = np.unique(wavelength, return_inverse=True)
    if unique.size != wavelength.size:
        summed = np.zeros(unique.size, dtype=np.float64)
        counts = np.zeros(unique.size, dtype=np.int64)
        np.add.at(summed, inverse, throughput)
        np.add.at(counts, inverse, 1)
        wavelength = unique
        throughput = summed / counts

    if not bool(np.all(np.diff(wavelength) > 0.0)):
        raise ValueError(f"SVO curve {filter_id} is not strictly increasing")
    if float(throughput.max()) <= 0.0:
        raise ValueError(f"SVO curve {filter_id} has no positive transmission")
    return wavelength, throughput


def _extract_svo_table_curve(
    table: Any,
    *,
    filter_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    names = {str(name).lower(): str(name) for name in table.colnames}
    try:
        wavelength_name = names["wavelength"]
        transmission_name = names["transmission"]
    except KeyError as exc:
        raise ValueError(
            f"SVO response for {filter_id} must contain Wavelength and Transmission"
        ) from exc

    wavelength_column = table[wavelength_name]
    wavelength_unit = getattr(wavelength_column, "unit", None)
    if wavelength_unit is None:
        raise ValueError(f"SVO wavelength column for {filter_id} has no unit")
    wavelength = (_as_float64(wavelength_column) * u.Unit(wavelength_unit)).to_value(u.AA)
    transmission = _as_float64(table[transmission_name])
    return _clean_curve(wavelength, transmission, filter_id=filter_id)


def _download_svo_curve(
    filter_id: str,
    *,
    timeout: float,
) -> tuple[np.ndarray, np.ndarray]:
    try:
        from astroquery.svo_fps import SvoFps
    except ImportError as exc:
        raise RuntimeError(
            "astroquery is required for bandpass_mode='official'; "
            "install requirements/wcs-aware-inr.txt"
        ) from exc
    try:
        table = SvoFps.get_transmission_data(
            filter_id,
            cache=True,
            timeout=float(timeout),
        )
    except Exception as exc:
        raise RuntimeError(f"failed to query SVO FPS curve {filter_id}") from exc
    return _extract_svo_table_curve(table, filter_id=filter_id)


def _save_svo_curve(
    path: Path,
    *,
    filter_id: str,
    wavelength: np.ndarray,
    transmission: np.ndarray,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            filter_id=np.asarray(filter_id),
            wavelength_angstrom=np.asarray(wavelength, dtype=np.float64),
            transmission=np.asarray(transmission, dtype=np.float64),
        )
    temporary.replace(path)


def _load_svo_curve(
    path: Path,
    *,
    expected_filter_id: str,
) -> tuple[np.ndarray, np.ndarray]:
    with np.load(path, allow_pickle=False) as payload:
        saved_filter_id = str(np.asarray(payload["filter_id"]).item())
        if saved_filter_id != expected_filter_id:
            raise ValueError(
                f"SVO cache {path} stores {saved_filter_id}, expected {expected_filter_id}"
            )
        wavelength = np.asarray(payload["wavelength_angstrom"], dtype=np.float64)
        transmission = np.asarray(payload["transmission"], dtype=np.float64)
    return _clean_curve(wavelength, transmission, filter_id=expected_filter_id)


def load_svo_transmission_curve(
    band: str,
    *,
    collection: str = SVO_DEFAULT_COLLECTION,
    cache_dir: Path | None = None,
    timeout: float = 60.0,
    force_refresh: bool = False,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    if not math.isfinite(float(timeout)) or timeout <= 0.0:
        raise ValueError("SVO timeout must be finite and positive")
    filter_id = _svo_filter_id(band, collection=collection)
    cache_path = _svo_cache_path(cache_dir, filter_id) if cache_dir is not None else None

    if cache_path is not None and cache_path.exists() and not force_refresh:
        wavelength, transmission = _load_svo_curve(
            cache_path,
            expected_filter_id=filter_id,
        )
        cache_hit = True
    else:
        wavelength, transmission = _download_svo_curve(filter_id, timeout=timeout)
        cache_hit = False
        if cache_path is not None:
            _save_svo_curve(
                cache_path,
                filter_id=filter_id,
                wavelength=wavelength,
                transmission=transmission,
            )

    metadata: dict[str, Any] = {
        "source": "SVO Filter Profile Service via astroquery.svo_fps",
        "svo_filter_id": filter_id,
        "svo_collection": _normalise_svo_collection(collection),
        "svo_cache_path": None if cache_path is None else str(cache_path),
        "svo_cache_hit": cache_hit,
        "wavelength_unit": "Angstrom",
        "detector_specific": False,
    }
    return wavelength, transmission, metadata


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
    svo_collection: str,
    svo_cache_dir: Path | None,
    svo_timeout: float,
    svo_force_refresh: bool,
    svo_curve_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]],
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
        "detector": exposure.detector,
        "mode": mode,
    }

    if mode == "official":
        filter_id = _svo_filter_id(exposure.band, collection=svo_collection)
        cached = svo_curve_cache.get(filter_id)
        if cached is None:
            cached = load_svo_transmission_curve(
                exposure.band,
                collection=svo_collection,
                cache_dir=svo_cache_dir,
                timeout=svo_timeout,
                force_refresh=svo_force_refresh,
            )
            svo_curve_cache[filter_id] = cached
        wavelength, throughput, source_metadata = cached
        support_min, support_max = _positive_support(wavelength, throughput)
        kwargs.update(
            spectral_min=support_min,
            spectral_max=support_max,
            transmission_wavelength=torch.as_tensor(wavelength, dtype=torch.float32),
            transmission_throughput=torch.as_tensor(throughput, dtype=torch.float32),
        )
        metadata.update(dict(source_metadata))
        metadata.update(
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
    svo_collection: str = SVO_DEFAULT_COLLECTION,
    svo_cache_dir: Path | None = None,
    svo_timeout: float = 60.0,
    svo_force_refresh: bool = False,
) -> BandpassCalibration:
    if not exposures:
        raise ValueError("at least one exposure is required")
    observations: list[FieldObservation] = []
    metadata: list[dict[str, Any]] = []
    svo_curve_cache: dict[str, tuple[np.ndarray, np.ndarray, dict[str, Any]]] = {}
    for exposure in exposures:
        observation, info = _bandpass_observation(
            exposure,
            mode=mode,
            band_specs=band_specs,
            svo_collection=svo_collection,
            svo_cache_dir=svo_cache_dir,
            svo_timeout=svo_timeout,
            svo_force_refresh=svo_force_refresh,
            svo_curve_cache=svo_curve_cache,
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


__all__ = [
    "SVO_DEFAULT_COLLECTION",
    "build_bandpass_calibration",
    "load_svo_transmission_curve",
]
