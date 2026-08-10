"""Build split-safe JWST/NIRCam calibration-evidence manifests."""

from __future__ import annotations

import argparse
import csv
import glob
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Any, Iterable

import torch

from ..instrument import (
    NIRCAM_METADATA_SOURCES,
    jwst_exposure_timing,
    nircam_imaging_band,
)
from ..io import get_header
from ..map.operators import spatial_fill_distance_bounds

_PATT_SPLIT = {
    1: "train",
    2: "train",
    3: "validation",
    4: "train",
    5: "train",
    6: "test",
    7: "train",
    8: "validation",
    9: "test",
}
_EVIDENCE_ROLES = {"psf", "background"}


def _expand_exposure_inputs(inputs: Iterable[str]) -> list[Path]:
    paths: list[Path] = []
    for value in inputs:
        candidate = Path(value).expanduser()
        if candidate.is_dir():
            paths.extend(candidate.rglob("*.fits"))
            paths.extend(candidate.rglob("*.fits.fz"))
        elif any(token in value for token in "*?[]"):
            paths.extend(Path(match) for match in glob.glob(value, recursive=True))
        else:
            paths.append(candidate)
    unique: dict[str, Path] = {}
    for path in paths:
        if path.is_file():
            unique.setdefault(str(path.absolute()), path.absolute())
    if not unique:
        raise FileNotFoundError("no FITS exposure files matched")
    return sorted(unique.values(), key=lambda path: str(path))


def _header_value(
    primary: dict[str, Any], science: dict[str, Any], *names: str
) -> Any:
    for name in names:
        if name in science and science[name] not in (None, ""):
            return science[name]
        if name in primary and primary[name] not in (None, ""):
            return primary[name]
    return None


def _exposure_row(path: Path) -> dict[str, Any]:
    primary = {
        str(key).upper(): value
        for key, value in dict(get_header(str(path), 0)).items()
    }
    science = {
        str(key).upper(): value
        for key, value in dict(get_header(str(path), "SCI")).items()
    }
    instrument = str(_header_value(primary, science, "INSTRUME") or "").upper()
    telescope = str(_header_value(primary, science, "TELESCOP") or "").upper()
    if instrument != "NIRCAM" or (telescope and telescope != "JWST"):
        raise ValueError(f"{path}: expected a JWST/NIRCam exposure")
    band = str(_header_value(primary, science, "FILTER", "PUPIL") or "").upper()
    metadata = nircam_imaging_band(band)
    try:
        patt_num = int(primary["PATT_NUM"])
        split = _PATT_SPLIT[patt_num]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"{path}: PATT_NUM must be an integer from 1 through 9") from exc
    detector = str(_header_value(primary, science, "DETECTOR") or "").upper()
    if not detector:
        raise ValueError(f"{path}: missing DETECTOR")
    width = int(science.get("ZNAXIS1", science.get("NAXIS1", 0)))
    height = int(science.get("ZNAXIS2", science.get("NAXIS2", 0)))
    if width < 2 or height < 2:
        raise ValueError(f"{path}: invalid SCI image shape")
    try:
        x_start = int(primary.get("SUBSTRT1", 1)) - 1
        y_start = int(primary.get("SUBSTRT2", 1)) - 1
        subsize_x = int(primary.get("SUBSIZE1", width))
        subsize_y = int(primary.get("SUBSIZE2", height))
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{path}: invalid JWST subarray geometry") from exc
    if x_start < 0 or y_start < 0 or subsize_x != width or subsize_y != height:
        raise ValueError(
            f"{path}: SCI shape must match positive JWST SUBSTRT/SUBSIZE geometry"
        )
    detector_bounds_xy = [
        x_start,
        x_start + width - 1,
        y_start,
        y_start + height - 1,
    ]
    timing = jwst_exposure_timing(primary, science)
    resolved = path.resolve()
    return {
        "path": str(path),
        "resolved_path": str(resolved),
        "filename": path.name,
        "size_bytes": path.stat().st_size,
        "band": band,
        "detector": detector,
        "patt_num": patt_num,
        "split": split,
        "shape_yx": [height, width],
        "detector_bounds_xy": detector_bounds_xy,
        "timing": timing.as_dict(),
        "program": _header_value(primary, science, "PROGRAM"),
        "observation": _header_value(primary, science, "OBSERVTN"),
        "visit": _header_value(primary, science, "VISIT"),
        "exposure": _header_value(primary, science, "EXPOSURE"),
        "instrument": {
            "channel": metadata.channel,
            "detector_pixel_scale_arcsec": metadata.detector_pixel_scale_arcsec,
            "detector_nyquist_period_arcsec": (
                metadata.detector_nyquist_period_arcsec
            ),
            "pivot_wavelength_angstrom": metadata.pivot_wavelength_angstrom,
            "half_power_bounds_angstrom": [
                metadata.half_power_min_angstrom,
                metadata.half_power_max_angstrom,
            ],
            "simulated_psf_fwhm_arcsec": metadata.simulated_psf_fwhm_arcsec,
            "empirical_psf_fwhm_arcsec": metadata.empirical_psf_fwhm_arcsec,
            "empirical_psf_fwhm_pixels": metadata.empirical_psf_fwhm_pixels,
        },
    }


def _read_evidence(path: Path) -> list[dict[str, Any]]:
    if not path.is_file():
        raise FileNotFoundError(f"evidence table not found: {path}")
    if path.suffix.lower() == ".json":
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows = payload.get("evidence", []) if isinstance(payload, dict) else payload
        if not isinstance(rows, list):
            raise ValueError("JSON evidence must be a list or contain an evidence list")
        return [dict(row) for row in rows]
    with path.open("r", encoding="utf-8", newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def _attach_evidence(
    rows: list[dict[str, Any]],
    exposures: list[dict[str, Any]],
    *,
    coverage_rtol: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    by_path: dict[str, dict[str, Any]] = {}
    by_name: dict[str, list[dict[str, Any]]] = {}
    for exposure in exposures:
        by_path[str(Path(exposure["path"]).absolute())] = exposure
        by_path[str(Path(exposure["resolved_path"]))] = exposure
        by_name.setdefault(str(exposure["filename"]), []).append(exposure)

    attached: list[dict[str, Any]] = []
    for index, raw in enumerate(rows):
        exposure_token = str(raw.get("exposure", "")).strip()
        if not exposure_token:
            raise ValueError(f"evidence row {index}: missing exposure")
        exposure = by_path.get(str(Path(exposure_token).expanduser().absolute()))
        if exposure is None:
            matches = by_name.get(Path(exposure_token).name, [])
            if len(matches) != 1:
                raise ValueError(
                    f"evidence row {index}: exposure must match one manifest file"
                )
            exposure = matches[0]
        role = str(raw.get("role", "")).strip().lower()
        if role not in _EVIDENCE_ROLES:
            raise ValueError(f"evidence row {index}: role must be psf or background")
        try:
            x = float(raw["detector_x"])
            y = float(raw["detector_y"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"evidence row {index}: detector_x and detector_y must be numeric"
            ) from exc
        xmin, xmax, ymin, ymax = map(float, exposure["detector_bounds_xy"])
        if not math.isfinite(x) or not math.isfinite(y) or not (
            xmin <= x <= xmax and ymin <= y <= ymax
        ):
            raise ValueError(
                f"evidence row {index}: detector coordinates lie outside the exposure"
            )
        attached.append(
            {
                "source_id": str(raw.get("source_id", index)),
                "role": role,
                "detector_x": x,
                "detector_y": y,
                "exposure": exposure["path"],
                "band": exposure["band"],
                "detector": exposure["detector"],
                "patt_num": exposure["patt_num"],
                "split": exposure["split"],
                "time_mjd_utc": exposure["timing"]["midpoint_mjd_utc"],
            }
        )

    grouped: dict[tuple[str, str, str, str], list[dict[str, Any]]] = {}
    for row in attached:
        key = (row["role"], row["band"], row["detector"], row["split"])
        grouped.setdefault(key, []).append(row)
    coverage: list[dict[str, Any]] = []
    for (role, band, detector, split), group in sorted(grouped.items()):
        matching_exposures = [
            exposure
            for exposure in exposures
            if exposure["band"] == band and exposure["detector"] == detector
        ]
        domains = {
            tuple(exposure["detector_bounds_xy"])
            for exposure in matching_exposures
        }
        if len(domains) != 1:
            raise ValueError(
                f"{band}/{detector}: inconsistent detector footprints prevent coverage audit"
            )
        domain = next(iter(domains))
        points = torch.tensor(
            [[row["detector_x"], row["detector_y"]] for row in group],
            dtype=torch.float64,
        )
        bounds = spatial_fill_distance_bounds(
            points,
            tuple(map(float, domain)),
            rtol=coverage_rtol,
        )
        coverage.append(
            {
                "role": role,
                "band": band,
                "detector": detector,
                "split": split,
                **bounds.as_dict(),
            }
        )
    return attached, coverage


def build_calibration_manifest(
    inputs: Iterable[str],
    *,
    evidence_path: Path | None = None,
    coverage_rtol: float = 0.01,
) -> dict[str, Any]:
    exposures = [_exposure_row(path) for path in _expand_exposure_inputs(inputs)]
    evidence: list[dict[str, Any]] = []
    coverage: list[dict[str, Any]] = []
    if evidence_path is not None:
        evidence, coverage = _attach_evidence(
            _read_evidence(evidence_path),
            exposures,
            coverage_rtol=coverage_rtol,
        )
    payload: dict[str, Any] = {
        "schema_version": 1,
        "profile": "jwst_nircam_updec_calibration_evidence",
        "time_scale": "MJD UTC",
        "coordinate_frame": "zero-indexed NIRCam detector pixels",
        "patt_num_split": {
            split: [number for number, value in _PATT_SPLIT.items() if value == split]
            for split in ("train", "validation", "test")
        },
        "instrument_metadata_sources": NIRCAM_METADATA_SOURCES,
        "exposures": exposures,
        "evidence": evidence,
        "coverage": coverage,
    }
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(canonical.encode("utf-8")).hexdigest()
    return payload


def _write_json_atomic(path: Path, payload: dict[str, Any], *, overwrite: bool) -> None:
    destination = path.expanduser().absolute()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"refusing to overwrite {destination}; pass --overwrite")
    temporary: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            dir=destination.parent,
            prefix=f".{destination.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary = handle.name
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, destination)
    except Exception:
        if temporary is not None:
            try:
                os.unlink(temporary)
            except FileNotFoundError:
                pass
        raise


def _calibration_manifest_handler(args: argparse.Namespace) -> int:
    payload = build_calibration_manifest(
        args.inputs,
        evidence_path=args.evidence,
        coverage_rtol=args.coverage_rtol,
    )
    _write_json_atomic(args.output, payload, overwrite=args.overwrite)
    print(
        f"wrote {len(payload['exposures'])} exposures, "
        f"{len(payload['evidence'])} evidence rows, and "
        f"{len(payload['coverage'])} coverage groups to {args.output}"
    )
    return 0


def _calibration_extract_handler(args: argparse.Namespace) -> int:
    from ..calibration_evidence import extract_calibration_evidence

    base = build_calibration_manifest(args.inputs)
    extraction = extract_calibration_evidence(
        base,
        catalog_path=args.catalog,
        output_dir=args.output_dir,
        psf_radius_px=args.psf_radius,
        background_radius_px=args.background_radius,
        minimum_psf_snr=args.minimum_psf_snr,
        detection_sigma=args.detection_sigma,
        minimum_psf_correlation=args.minimum_psf_correlation,
        background_exclusion_sigma=args.background_exclusion_sigma,
        max_psf_candidates=args.max_psf_candidates,
        max_background_candidates=args.max_background_candidates,
        overwrite=args.overwrite,
    )
    evidence_path = Path(extraction["products"]["detector_evidence"]["path"])
    payload = build_calibration_manifest(
        args.inputs,
        evidence_path=evidence_path,
        coverage_rtol=args.coverage_rtol,
    )
    if len(payload["evidence"]) != len(extraction["evidence"]):
        raise RuntimeError("extracted and attached evidence row counts disagree")
    for attached, target in zip(
        payload["evidence"], extraction["evidence"], strict=True
    ):
        attached.update(
            {
                key: value
                for key, value in target.items()
                if key not in attached
            }
        )
    payload["extraction"] = {
        key: value for key, value in extraction.items() if key != "evidence"
    }
    payload.pop("manifest_sha256", None)
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["manifest_sha256"] = hashlib.sha256(
        canonical.encode("utf-8")
    ).hexdigest()
    output = args.output_dir / "calibration_evidence_manifest.json"
    _write_json_atomic(output, payload, overwrite=args.overwrite)
    print(
        f"wrote {len(payload['evidence'])} accepted targets and "
        f"{len(extraction['rejected'])} rejected measurements "
        f"({extraction['selection_mode']}) to {output}"
    )
    return 0


def register_calibration_parser(subparsers: argparse.Action) -> None:
    parser = subparsers.add_parser(
        "calibration-manifest",
        help="Build a split-safe JWST/NIRCam calibration-evidence manifest.",
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        help="FITS files, directories, or glob patterns (quote shell globs).",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--evidence",
        type=Path,
        default=None,
        help=(
            "Optional CSV/JSON rows with exposure,detector_x,detector_y,role and "
            "optional source_id; role is psf or background."
        ),
    )
    parser.add_argument("--coverage-rtol", type=float, default=0.01)
    parser.add_argument("--overwrite", action="store_true")
    parser.set_defaults(handler=_calibration_manifest_handler)

    extract = subparsers.add_parser(
        "calibration-extract",
        help="Extract PSF/background targets from JWST/NIRCam detector files.",
    )
    extract.add_argument(
        "inputs",
        nargs="+",
        help="FITS files, directories, or glob patterns (quote shell globs).",
    )
    extract.add_argument(
        "--catalog",
        type=Path,
        default=None,
        help=(
            "Optional reviewed sky catalog. Without one, select conservative "
            "PSF and source-excluded background targets from SCI/ERR/DQ."
        ),
    )
    extract.add_argument("--output-dir", type=Path, required=True)
    extract.add_argument("--psf-radius", type=int, default=9)
    extract.add_argument("--background-radius", type=int, default=16)
    extract.add_argument("--minimum-psf-snr", type=float, default=20.0)
    extract.add_argument("--detection-sigma", type=float, default=8.0)
    extract.add_argument("--minimum-psf-correlation", type=float, default=0.65)
    extract.add_argument("--background-exclusion-sigma", type=float, default=5.0)
    extract.add_argument("--max-psf-candidates", type=int, default=32)
    extract.add_argument("--max-background-candidates", type=int, default=36)
    extract.add_argument("--coverage-rtol", type=float, default=0.01)
    extract.add_argument("--overwrite", action="store_true")
    extract.set_defaults(handler=_calibration_extract_handler)
