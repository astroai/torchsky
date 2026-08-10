"""Observed CAL-image targets for learned PSF and background operators."""

from __future__ import annotations

import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any

import torch
from torch import Tensor

from .instrument import nircam_imaging_band
from .io import get_header, read, write
from .map.operators import fit_background_plane, registered_image_stamps
from .wcs import WCS, propagate_proper_motion

_ROLES = {"psf", "background"}
_TRUE = {"1", "true", "yes", "y"}


def _finite_float(row: dict[str, str], name: str, *, default: float | None = None) -> float:
    value = row.get(name, "").strip()
    if not value and default is not None:
        return default
    try:
        result = float(value)
    except ValueError as exc:
        raise ValueError(f"catalog {name} must be numeric") from exc
    if not math.isfinite(result):
        raise ValueError(f"catalog {name} must be finite")
    return result


def read_reviewed_sky_catalog(path: Path) -> list[dict[str, Any]]:
    """Read the explicit catalog trust boundary used by evidence extraction."""
    if not path.is_file():
        raise FileNotFoundError(f"reviewed catalog not found: {path}")
    with path.open("r", encoding="utf-8", newline="") as handle:
        raw_rows = list(csv.DictReader(handle))
    if not raw_rows:
        raise ValueError("reviewed catalog must contain at least one row")
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for index, raw in enumerate(raw_rows):
        source_id = raw.get("source_id", "").strip()
        if not source_id or source_id in seen:
            raise ValueError(f"catalog row {index}: source_id must be nonempty and unique")
        seen.add(source_id)
        role = raw.get("role", "").strip().lower()
        if role not in _ROLES:
            raise ValueError(f"catalog row {index}: role must be psf or background")
        if raw.get("reviewed", "").strip().lower() not in _TRUE:
            raise ValueError(f"catalog row {index}: reviewed must be true")
        weight = _finite_float(raw, "sample_weight", default=1.0)
        if weight <= 0.0:
            raise ValueError(f"catalog row {index}: sample_weight must be positive")
        rows.append(
            {
                "source_id": source_id,
                "role": role,
                "ra_deg": _finite_float(raw, "ra_deg"),
                "dec_deg": _finite_float(raw, "dec_deg"),
                "pmra_mas_yr": _finite_float(raw, "pmra_mas_yr", default=0.0),
                "pmdec_mas_yr": _finite_float(raw, "pmdec_mas_yr", default=0.0),
                "ref_epoch_jyear": _finite_float(
                    raw, "ref_epoch_jyear", default=2016.0
                ),
                "sample_weight": weight,
            }
        )
    return rows


def _registered_arrays(
    science: Tensor,
    error: Tensor,
    valid: Tensor,
    positions: Tensor,
    *,
    radius_px: int,
) -> tuple[Tensor, Tensor, Tensor]:
    clean_science = torch.where(valid, science, torch.zeros_like(science))
    clean_error = torch.where(valid, error, torch.ones_like(error))
    stamps = registered_image_stamps(
        clean_science, positions, radius_px=radius_px
    )
    error_stamps = registered_image_stamps(
        clean_error, positions, radius_px=radius_px
    )
    support = registered_image_stamps(
        valid.to(science.dtype), positions, radius_px=radius_px
    ).clamp(0.0, 1.0)
    weights = support / error_stamps.square().clamp_min(
        torch.finfo(error_stamps.dtype).tiny
    )
    return stamps, weights, support


def _psf_target(
    stamp: Tensor,
    weight: Tensor,
    support: Tensor,
    *,
    minimum_snr: float,
) -> tuple[Tensor, Tensor, float] | str:
    size = stamp.shape[0]
    yy, xx = torch.meshgrid(
        torch.arange(size, dtype=stamp.dtype),
        torch.arange(size, dtype=stamp.dtype),
        indexing="ij",
    )
    center = (size - 1) / 2.0
    radius = torch.sqrt((xx - center).square() + (yy - center).square())
    annulus = (radius >= 0.7 * center) & (support >= 0.999)
    if int(annulus.sum()) < max(12, size):
        return "insufficient_background_annulus"
    background = fit_background_plane(
        stamp,
        annulus,
        weight=weight,
        sigma_clip=3.0,
        max_iterations=3,
    ).model
    target = torch.where(support >= 0.999, stamp - background, torch.zeros_like(stamp))
    pixel_weight = torch.where(support >= 0.999, weight, torch.zeros_like(weight))
    valid_fraction = float((pixel_weight > 0).to(torch.float32).mean())
    if valid_fraction < 0.8:
        return "insufficient_valid_pixels"
    flux = target.sum()
    if not bool(torch.isfinite(flux)) or float(flux) <= 0.0:
        return "nonpositive_flux"
    variance = torch.where(pixel_weight > 0, pixel_weight.reciprocal(), 0.0).sum()
    snr = float(flux / torch.sqrt(variance.clamp_min(torch.finfo(stamp.dtype).tiny)))
    if not math.isfinite(snr) or snr < minimum_snr:
        return "low_snr"
    target = target / flux
    positive_weight = pixel_weight[pixel_weight > 0]
    pixel_weight = pixel_weight / torch.median(positive_weight)
    return target, pixel_weight, snr


def _background_target(
    stamp: Tensor,
    weight: Tensor,
    support: Tensor,
) -> tuple[float, float, float] | str:
    mask = support >= 0.999
    if float(mask.to(torch.float32).mean()) < 0.8:
        return "insufficient_valid_pixels"
    fit = fit_background_plane(
        stamp,
        mask,
        weight=weight,
        sigma_clip=3.0,
        max_iterations=3,
    )
    radius = stamp.shape[0] // 2
    value = float(fit.model[radius, radius])
    rms = float(fit.detrended_rms)
    if not math.isfinite(value) or not math.isfinite(rms):
        return "nonfinite_background_fit"
    return value, 1.0 / max(rms * rms, 1.0e-12), rms


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def extract_reviewed_calibration_evidence(
    exposure_manifest: dict[str, Any],
    *,
    catalog_path: Path,
    output_dir: Path,
    psf_radius_px: int = 9,
    background_radius_px: int = 16,
    minimum_psf_snr: float = 20.0,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Extract split-labelled observational targets from JWST CAL exposures."""
    if psf_radius_px < 2 or background_radius_px < 2:
        raise ValueError("evidence radii must be at least two pixels")
    if not math.isfinite(minimum_psf_snr) or minimum_psf_snr <= 0.0:
        raise ValueError("minimum_psf_snr must be finite and positive")
    catalog = read_reviewed_sky_catalog(catalog_path)
    destination = output_dir.expanduser().absolute()
    destination.mkdir(parents=True, exist_ok=True)
    product_paths = {
        "psf_kernels": destination / "psf_kernels.fits",
        "psf_pixel_weights": destination / "psf_pixel_weights.fits",
        "background_targets": destination / "background_targets.csv",
        "detector_evidence": destination / "detector_evidence.csv",
    }
    existing = [path for path in product_paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            f"refusing to overwrite {existing[0]}; pass --overwrite"
        )

    psf_kernels: list[Tensor] = []
    psf_weights: list[Tensor] = []
    evidence: list[dict[str, Any]] = []
    background_rows: list[dict[str, Any]] = []
    rejected: list[dict[str, Any]] = []
    for exposure in exposure_manifest["exposures"]:
        path = Path(exposure["resolved_path"])
        science_header = dict(get_header(str(path), "SCI"))
        science = read(str(path), hdu="SCI").detach().to(torch.float32)
        error = read(str(path), hdu="ERR").detach().to(torch.float32)
        dq = read(str(path), hdu="DQ").detach()
        if science.ndim != 2 or error.shape != science.shape or dq.shape != science.shape:
            raise ValueError(f"{path}: SCI, ERR, and DQ must be matching 2D images")
        valid = (
            torch.isfinite(science)
            & torch.isfinite(error)
            & (error > 0)
            & (dq == 0)
        )
        epoch = 2000.0 + (
            float(exposure["timing"]["midpoint_mjd_utc"]) - 51544.5
        ) / 365.25
        lon, lat = propagate_proper_motion(
            [row["ra_deg"] for row in catalog],
            [row["dec_deg"] for row in catalog],
            [row["pmra_mas_yr"] for row in catalog],
            [row["pmdec_mas_yr"] for row in catalog],
            source_epoch_jyear=[row["ref_epoch_jyear"] for row in catalog],
            target_epoch_jyear=epoch,
        )
        local_x, local_y = WCS(science_header).world_to_pixel(lon, lat, origin=0)
        for role, radius_px in (
            ("psf", psf_radius_px),
            ("background", background_radius_px),
        ):
            indices = [
                index
                for index, row in enumerate(catalog)
                if row["role"] == role
                and radius_px <= float(local_x[index]) <= science.shape[1] - 1 - radius_px
                and radius_px <= float(local_y[index]) <= science.shape[0] - 1 - radius_px
            ]
            if not indices:
                continue
            positions = torch.stack((local_x[indices], local_y[indices]), dim=-1).to(
                dtype=science.dtype
            )
            stamps, weights, support = _registered_arrays(
                science, error, valid, positions, radius_px=radius_px
            )
            x_offset = float(exposure["detector_bounds_xy"][0])
            y_offset = float(exposure["detector_bounds_xy"][2])
            for local_index, catalog_index in enumerate(indices):
                source = catalog[catalog_index]
                common = {
                    "exposure": exposure["path"],
                    "source_id": source["source_id"],
                    "role": role,
                    "detector_x": float(local_x[catalog_index]) + x_offset,
                    "detector_y": float(local_y[catalog_index]) + y_offset,
                    "sample_weight": source["sample_weight"],
                }
                if role == "psf":
                    target = _psf_target(
                        stamps[local_index],
                        weights[local_index],
                        support[local_index],
                        minimum_snr=minimum_psf_snr,
                    )
                    if isinstance(target, str):
                        rejected.append({**common, "reason": target})
                        continue
                    kernel, pixel_weight, snr = target
                    common.update(
                        {
                            "target_index": len(psf_kernels),
                            "target_kind": "normalized_effective_psf",
                            "snr": snr,
                        }
                    )
                    psf_kernels.append(kernel.cpu())
                    psf_weights.append(pixel_weight.cpu())
                else:
                    target = _background_target(
                        stamps[local_index], weights[local_index], support[local_index]
                    )
                    if isinstance(target, str):
                        rejected.append({**common, "reason": target})
                        continue
                    value, reliability, rms = target
                    common.update(
                        {
                            "target_index": len(background_rows),
                            "target_kind": "local_sigma_clipped_plane",
                            "target_value": value,
                            "sample_weight": source["sample_weight"] * reliability,
                            "detrended_rms": rms,
                        }
                    )
                    background_rows.append(common.copy())
                evidence.append(common)

    if not evidence:
        raise ValueError("no reviewed catalog positions produced valid evidence")
    generated_paths = {
        "detector_evidence": product_paths["detector_evidence"],
    }
    if psf_kernels:
        write(
            str(product_paths["psf_kernels"]),
            torch.stack(psf_kernels),
            header={"BUNIT": "NORMALIZED", "CONTENT": "PSF KERNEL"},
            overwrite=overwrite,
        )
        generated_paths["psf_kernels"] = product_paths["psf_kernels"]
        generated_paths["psf_pixel_weights"] = product_paths[
            "psf_pixel_weights"
        ]
        write(
            str(product_paths["psf_pixel_weights"]),
            torch.stack(psf_weights),
            header={"BUNIT": "RELATIVE", "CONTENT": "PIXEL WEIGHT"},
            overwrite=overwrite,
        )
    fieldnames = sorted({key for row in evidence for key in row})
    with product_paths["detector_evidence"].open(
        "w", encoding="utf-8", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(evidence)
    if background_rows:
        with product_paths["background_targets"].open(
            "w", encoding="utf-8", newline=""
        ) as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(background_rows)
        generated_paths["background_targets"] = product_paths[
            "background_targets"
        ]

    products = {
        name: {
            "path": str(path),
            "sha256": _file_sha256(path),
            "size_bytes": path.stat().st_size,
        }
        for name, path in generated_paths.items()
    }
    bands = {
        exposure["band"]: nircam_imaging_band(exposure["band"]).__dict__
        for exposure in exposure_manifest["exposures"]
    }
    return {
        "schema_version": 1,
        "catalog_path": str(catalog_path.absolute()),
        "catalog_sha256": _file_sha256(catalog_path),
        "catalog_row_count": len(catalog),
        "evidence": evidence,
        "rejected": rejected,
        "products": products,
        "instrument_bands": bands,
        "settings": {
            "psf_radius_px": psf_radius_px,
            "background_radius_px": background_radius_px,
            "minimum_psf_snr": minimum_psf_snr,
            "dq_policy": "DQ == 0",
            "proper_motion_epoch_approximation": "2000 + (MJD_UTC - 51544.5) / 365.25",
        },
    }


def load_operator_calibration_split(
    manifest: Path | dict[str, Any],
    *,
    split: str,
    role: str,
    detector: str,
    verify_hashes: bool = True,
) -> dict[str, Tensor]:
    """Load one detector/role/split as tensors ready for a calibration term."""
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
        if row.get("split") == split
        and row.get("role") == role
        and str(row.get("detector", "")).upper() == detector.upper()
    ]
    if not rows:
        raise ValueError(f"no {split} {role} evidence for detector {detector}")

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
            [nircam_imaging_band(row["band"]).pivot_wavelength_angstrom for row in rows],
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
        result["target_kernels"] = read(
            str(product_path("psf_kernels"))
        ).index_select(0, indices)
        result["pixel_weight"] = read(
            str(product_path("psf_pixel_weights"))
        ).index_select(0, indices)
    else:
        result["target_values"] = torch.tensor(
            [row["target_value"] for row in rows], dtype=torch.float64
        )
    return result


__all__ = [
    "extract_reviewed_calibration_evidence",
    "load_operator_calibration_split",
    "read_reviewed_sky_catalog",
]
