from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
import os
import zipfile
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

NIRCAM_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _npy_bytes(value: np.ndarray) -> bytes:
    stream = io.BytesIO()
    np.lib.format.write_array(stream, value, allow_pickle=False)
    return stream.getvalue()


def _deterministic_npz_bytes(payload: dict[str, np.ndarray]) -> bytes:
    stream = io.BytesIO()
    with zipfile.ZipFile(stream, "w", compression=zipfile.ZIP_STORED) as archive:
        for name in ("wavelength_angstrom", "transmission", "filter_id"):
            info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, _npy_bytes(payload[name]))
    return stream.getvalue()


def _write_immutable(path: Path, data: bytes) -> None:
    if path.exists():
        if path.read_bytes() != data:
            raise ValueError(f"{path} conflicts with fetched SVO content")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    try:
        temporary.write_bytes(data)
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.read_bytes() != data:
                raise ValueError(f"{path} conflicts with fetched SVO content") from None
    finally:
        if temporary.exists():
            temporary.unlink()


def _fetch_svo_curve(*, filter_id: str, **kwargs: Any) -> dict[str, Any]:
    from astroquery.svo_fps import SvoFps

    data = SvoFps.get_transmission_data(filter_id)
    wavelength = np.asarray(
        data["Wavelength"].value
        if hasattr(data["Wavelength"], "value")
        else data["Wavelength"],
        dtype=np.float64,
    )
    transmission = np.asarray(
        data["Transmission"].value
        if hasattr(data["Transmission"], "value")
        else data["Transmission"],
        dtype=np.float64,
    )
    return {
        "svo_filter_id": filter_id,
        "transmission_wavelength": wavelength,
        "transmission_throughput": transmission,
    }


def _validated_curve(spec: dict[str, Any], filter_id: str) -> tuple[np.ndarray, np.ndarray]:
    if str(spec.get("svo_filter_id")) != filter_id:
        raise ValueError(f"SVO returned {spec.get('svo_filter_id')!r}, expected {filter_id!r}")
    wavelength = np.asarray(spec["transmission_wavelength"], dtype=np.float64)
    transmission = np.asarray(spec["transmission_throughput"], dtype=np.float64)
    if wavelength.ndim != 1 or wavelength.size < 2 or wavelength.shape != transmission.shape:
        raise ValueError(f"{filter_id} must provide matching one-dimensional arrays")
    if not np.all(np.isfinite(wavelength)) or not np.all(wavelength > 0.0):
        raise ValueError(f"{filter_id} wavelengths must be finite and positive")
    if not np.all(np.diff(wavelength) > 0.0):
        raise ValueError(f"{filter_id} wavelengths must be strictly increasing")
    if not np.all(np.isfinite(transmission)) or np.any(transmission < 0.0):
        raise ValueError(f"{filter_id} transmission must be finite and non-negative")
    if not math.isfinite(float(transmission.max())) or float(transmission.max()) <= 0.0:
        raise ValueError(f"{filter_id} transmission must have positive support")
    return wavelength, transmission


def bootstrap_svo_passbands(
    output_dir: Path,
    *,
    bands: Sequence[str] = NIRCAM_BANDS,
    cache_dir: Path | None = None,
    fetcher: Callable[..., dict[str, Any]] = _fetch_svo_curve,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    normalized_bands = tuple(str(band).strip().upper() for band in bands)
    if not normalized_bands or len(set(normalized_bands)) != len(normalized_bands):
        raise ValueError("bands must be non-empty and unique")
    manifest_path = output_dir / "manifest.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("schema_version") != 1:
            raise ValueError("existing passband manifest has unsupported schema")
        recorded_bands = tuple(entry["band"] for entry in manifest["passbands"])
        if recorded_bands != normalized_bands:
            raise ValueError("existing passband manifest has different bands")
        for entry in manifest["passbands"]:
            expected_filter_id = f"JWST/NIRCam2025.{entry['band']}"
            expected_filename = f"JWST_NIRCam2025.{entry['band']}.npz"
            if (
                entry.get("filter_id") != expected_filter_id
                or entry.get("filename") != expected_filename
            ):
                raise ValueError("existing passband manifest has invalid identity")
            path = output_dir / entry["filename"]
            if not path.is_file() or _sha256(path.read_bytes()) != entry["sha256"]:
                raise ValueError(f"{path} hash mismatch")
        return manifest
    entries = []
    for band in normalized_bands:
        if band not in NIRCAM_BANDS:
            raise ValueError(f"unsupported NIRCam band {band!r}")
        filter_id = f"JWST/NIRCam2025.{band}"
        spec = fetcher(
            band_name=band,
            facility="JWST",
            instrument="NIRCAM",
            filter_id=filter_id,
            prefer="2025",
            cache_dir=cache_dir,
            use_cache=True,
        )
        wavelength, transmission = _validated_curve(spec, filter_id)
        filename = f"JWST_NIRCam2025.{band}.npz"
        data = _deterministic_npz_bytes({
            "wavelength_angstrom": wavelength,
            "transmission": transmission,
            "filter_id": np.asarray(filter_id),
        })
        _write_immutable(output_dir / filename, data)
        entries.append({
            "band": band,
            "filter_id": filter_id,
            "filename": filename,
            "samples": int(wavelength.size),
            "wavelength_min_angstrom": float(wavelength.min()),
            "wavelength_max_angstrom": float(wavelength.max()),
            "sha256": _sha256(data),
            "size": len(data),
        })
    manifest = {
        "schema_version": 1,
        "source": "SVO Filter Profile Service",
        "wavelength_unit": "angstrom",
        "transmission_unit": "dimensionless",
        "passbands": entries,
    }
    manifest_data = (json.dumps(manifest, indent=2, sort_keys=True) + "\n").encode()
    _write_immutable(manifest_path, manifest_data)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--bands", default=",".join(NIRCAM_BANDS))
    parser.add_argument("--cache-dir", type=Path, default=None)
    args = parser.parse_args()
    bands = tuple(part.strip() for part in str(args.bands).split(",") if part.strip())
    manifest = bootstrap_svo_passbands(
        args.output_dir,
        bands=bands,
        cache_dir=args.cache_dir,
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
