"""Runtime loader for validated operator-calibration evidence."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .instrument import nircam_imaging_band
from torchfits import read

_ROLES = {"psf", "background"}

def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_operator_calibration_split(
    manifest: Path | dict[str, Any],
    *,
    split: str,
    role: str,
    detector: str,
    exposure_id: str | None = None,
    band: str | None = None,
    verify_hashes: bool = True,
) -> dict[str, Tensor]:
    """Load detector evidence, optionally restricted to an exact exposure and band."""
    if exposure_id is not None and (
        not exposure_id.strip() or not band or not band.strip()
    ):
        raise ValueError("exposure selection requires nonempty exposure_id and band")
    if split not in {"train", "validation", "test"}:
        raise ValueError("split must be train, validation, or test")
    role = role.lower()
    if role not in _ROLES:
        raise ValueError("role must be psf or background")
    if isinstance(manifest, Path):
        payload = json.loads(manifest.read_text(encoding="utf-8"))
    else:
        payload = manifest
    products = payload.get("extraction", {}).get("products", {})
    rows = [
        row
        for row in payload.get("evidence", [])
        if row.get("calibration_split" if exposure_id is not None else "split") == split
        and row.get("role") == role
        and str(row.get("detector", "")).upper() == detector.upper()
        and (exposure_id is None or calibration_exposure_id(row) == exposure_id)
        and (band is None or str(row.get("band", "")).upper() == band.upper())
    ]
    if not rows:
        raise ValueError(
            f"no {split} {role} evidence for detector {detector}"
            + (
                f", exposure {exposure_id!r}, band {band}; independent split evidence is required"
                if exposure_id is not None
                else ""
            )
        )

    def product_path(name: str) -> Path:
        try:
            descriptor = products[name]
            path = Path(descriptor["path"])
        except (KeyError, TypeError) as exc:
            raise ValueError(f"manifest is missing {name}") from exc
        if verify_hashes and _file_sha256(path) != descriptor.get("sha256"):
            raise ValueError(f"{name} SHA-256 does not match the manifest")
        return path

    result = {
        "detector_x": torch.tensor(
            [row["detector_x"] for row in rows], dtype=torch.float64
        ),
        "detector_y": torch.tensor(
            [row["detector_y"] for row in rows], dtype=torch.float64
        ),
        "wavelength": torch.tensor(
            [
                nircam_imaging_band(row["band"]).pivot_wavelength_angstrom
                for row in rows
            ],
            dtype=torch.float64,
        ),
        "time_mjd": torch.tensor(
            [row["time_mjd_utc"] for row in rows], dtype=torch.float64
        ),
        "sample_weight": torch.tensor(
            [row.get("sample_weight", 1.0) for row in rows], dtype=torch.float64
        ),
    }
    if role == "psf":
        indices = torch.tensor([row["target_index"] for row in rows], dtype=torch.long)
        result["target_kernels"] = read(str(product_path("psf_kernels"))).index_select(
            0, indices
        )
        result["pixel_weight"] = read(
            str(product_path("psf_pixel_weights"))
        ).index_select(0, indices)
    else:
        result["target_values"] = torch.tensor(
            [row["target_value"] for row in rows], dtype=torch.float64
        )
    return result


def calibration_exposure_id(row: dict[str, Any]) -> str:
    """Return the explicit exposure identity; legacy evidence stores it as exposure."""
    value = row.get("exposure_id", row.get("exposure"))
    if not isinstance(value, str) or not value.strip():
        raise ValueError(
            "exposure calibration requires a nonempty exposure_id or exposure provenance field"
        )
    return value


__all__ = [
    "calibration_exposure_id",
    "load_operator_calibration_split",
]
