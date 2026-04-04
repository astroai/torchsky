#!/usr/bin/env python3
"""Benchmark torchsky image-plane mapmaking against external reprojection libraries."""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import astropy  # noqa: E402
from astropy.io import fits  # noqa: E402
from astropy.wcs import WCS  # noqa: E402

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    ImageObservation,
    coadd_image_plane,
    plan_image_plane_target,
)

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover - native bench env requirement
    raise RuntimeError(
        "bench_map_compat.py requires torchsky.cpp; "
        "run it from a native-extension environment"
    ) from exc

try:  # noqa: E402
    from reproject import reproject_interp  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    reproject_interp = None

try:  # noqa: E402
    from dfreproject import calculate_reprojection  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    calculate_reprojection = None

try:  # noqa: E402
    from drizzle.resample import Drizzle as DrizzleResampler  # type: ignore
    from drizzle.utils import calc_pixmap, estimate_pixel_scale_ratio  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    DrizzleResampler = None
    calc_pixmap = None
    estimate_pixel_scale_ratio = None

try:  # noqa: E402
    import drizzlepac  # type: ignore  # noqa: F401
except Exception as exc:  # pragma: no cover - optional comparator
    drizzlepac = None
    DRIZZLEPAC_IMPORT_ERROR = str(exc)
else:
    DRIZZLEPAC_IMPORT_ERROR = None

try:  # noqa: E402
    from MontagePy import main as MontageMain  # type: ignore
except Exception as exc:  # pragma: no cover - optional comparator
    MontageMain = None
    MONTAGEPY_IMPORT_ERROR = str(exc)
else:
    MONTAGEPY_IMPORT_ERROR = None


_COMPARATORS = (
    "reproject_interp",
    "dfreproject",
    "drizzle",
    "drizzlepac",
    "montage",
    "swarp",
    "casutools",
)


def _compat_row_key(row: dict[str, Any]) -> str:
    if row.get("library") == "pairwise":
        return f"{row.get('case')}|pairwise|{row.get('pair')}"
    return "|".join(
        [
            str(row.get("case", "")),
            str(row.get("library", "")),
            str(row.get("variant", "")),
        ]
    )


def _build_compat_markdown(rows: list[dict[str, Any]]) -> str:
    cases = sorted({str(row["case"]) for row in rows if "case" in row})
    lines = [
        "| case | fastest backend | closest external to torchsky | largest pairwise disagreement |",
        "| --- | --- | --- | --- |",
    ]
    for case in cases:
        case_rows = [row for row in rows if str(row.get("case")) == case]
        ok_rows = [
            row
            for row in case_rows
            if row.get("status") == "OK" and row.get("library") != "pairwise"
        ]
        pairwise_rows = [
            row
            for row in case_rows
            if row.get("status") == "OK" and row.get("library") == "pairwise"
        ]
        fastest = min(ok_rows, key=lambda row: float(row["time_ms"])) if ok_rows else None
        external_rows = [
            row
            for row in ok_rows
            if row.get("library") != "torchsky"
            and row.get("max_abs_data_diff_vs_torchsky") is not None
        ]
        closest = (
            min(
                external_rows,
                key=lambda row: float(row["max_abs_data_diff_vs_torchsky"]),
            )
            if external_rows
            else None
        )
        largest_pairwise = (
            max(
                pairwise_rows,
                key=lambda row: float(row.get("max_abs_data_diff", 0.0)),
            )
            if pairwise_rows
            else None
        )

        def _fmt_backend(row: dict[str, Any] | None) -> str:
            if row is None:
                return "n/a"
            variant = row.get("variant")
            name = str(row.get("library"))
            if variant not in (None, "", "default"):
                name = f"{name}:{variant}"
            return f"{name} ({float(row['time_ms']):.3f} ms)"

        def _fmt_closest(row: dict[str, Any] | None) -> str:
            if row is None:
                return "n/a"
            variant = row.get("variant")
            name = str(row.get("library"))
            if variant not in (None, "", "default"):
                name = f"{name}:{variant}"
            return f"{name} ({float(row['max_abs_data_diff_vs_torchsky']):.4g})"

        def _fmt_pairwise(row: dict[str, Any] | None) -> str:
            if row is None:
                return "n/a"
            return f"{row['pair']} ({float(row['max_abs_data_diff']):.4g})"

        lines.append(
            "| "
            + " | ".join(
                [
                    case,
                    _fmt_backend(fastest),
                    _fmt_closest(closest),
                    _fmt_pairwise(largest_pairwise),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def _write_compat_artifacts(
    rows: list[dict[str, Any]],
    *,
    json_out: Path | None,
    markdown_out: Path | None,
) -> str:
    markdown = _build_compat_markdown(rows)
    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if markdown_out is not None:
        markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown_out.write_text(markdown, encoding="utf-8")
    return markdown


def _check_compat_regressions(
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    data_diff_tolerance: float = 0.10,
    weight_diff_tolerance: float = 0.10,
) -> list[str]:
    baseline_by_key = {_compat_row_key(row): row for row in baseline_rows}
    errors: list[str] = []
    for row in rows:
        key = _compat_row_key(row)
        baseline = baseline_by_key.get(key)
        if baseline is None:
            continue
        current_status = str(row.get("status", "OK"))
        baseline_status = str(baseline.get("status", "OK"))
        if baseline_status == "OK" and current_status != "OK":
            errors.append(f"{key}: status regressed from OK to {current_status}")
            continue
        if baseline_status != "OK":
            continue
        for metric, tolerance in (
            ("max_abs_data_diff_vs_torchsky", data_diff_tolerance),
            ("max_abs_weight_diff_vs_torchsky", weight_diff_tolerance),
            ("max_abs_data_diff", data_diff_tolerance),
            ("max_abs_weight_diff", weight_diff_tolerance),
        ):
            base_value = baseline.get(metric)
            current_value = row.get(metric)
            if base_value is None or current_value is None:
                continue
            allowed = float(base_value) * (1.0 + tolerance) + 1.0e-12
            if float(current_value) > allowed:
                errors.append(
                    f"{key}: {metric} regressed {float(current_value):.6g} > {allowed:.6g}"
                )
    return errors
_TORCHSKY_KERNELS = (
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
    "lanczos5",
    "mitchell_netravali",
    "catmull_rom",
)
_REPROJECT_ORDERS = (
    ("nearest-neighbor", "nearest"),
    ("bilinear", "bilinear"),
    ("bicubic", "bicubic"),
)
_DFREPROJECT_ORDERS = (
    ("nearest", "nearest"),
    ("bilinear", "bilinear"),
    ("bicubic", "bicubic"),
)
_DRIZZLE_KERNELS = ("point", "square")
_SWARP_RESAMPLING_TYPES = ("NEAREST", "BILINEAR", "LANCZOS3")


def _external_tool_prefix(tool: str) -> Path:
    return ROOT / ".benchmarks" / "external-tools" / "prefix" / tool


def _find_swarp_binary() -> Path | None:
    candidate = _external_tool_prefix("swarp") / "bin" / "swarp"
    if candidate.exists():
        return candidate
    source_candidate = ROOT / ".benchmarks" / "external-tools" / "src" / "swarp" / "src" / "swarp"
    if source_candidate.exists():
        return source_candidate
    path = shutil.which("swarp")
    return Path(path) if path is not None else None


def _find_montage_binary(name: str) -> Path | None:
    candidate = _external_tool_prefix("montage") / "bin" / name
    if candidate.exists():
        return candidate
    source_candidate = ROOT / ".benchmarks" / "external-tools" / "src" / "Montage" / "bin" / name
    if source_candidate.exists():
        return source_candidate
    return None


def _find_casutools_binary(name: str) -> Path | None:
    candidate = _external_tool_prefix("casutools") / "bin" / name
    if candidate.exists():
        return candidate
    source_candidate = ROOT / ".benchmarks" / "external-tools" / "src" / "casutools" / "src" / "mosaic" / name
    if source_candidate.exists():
        return source_candidate
    path = shutil.which(name)
    return Path(path) if path is not None else None


def _external_tool_env() -> dict[str, str]:
    env = dict(os.environ)
    lib_dirs: list[str] = []
    conda_prefix = env.get("CONDA_PREFIX")
    if conda_prefix:
        lib_dirs.append(str(Path(conda_prefix) / "lib"))
    map_compat_lib = ROOT / ".pixi" / "envs" / "map-compat" / "lib"
    if map_compat_lib.exists():
        lib_dirs.append(str(map_compat_lib))
    casutools_shim_lib = ROOT / ".benchmarks" / "external-tools" / "shims" / "casutools" / "lib"
    if casutools_shim_lib.exists():
        lib_dirs.append(str(casutools_shim_lib))
    for key in ("DYLD_LIBRARY_PATH", "LD_LIBRARY_PATH"):
        existing = env.get(key)
        parts = [part for part in lib_dirs if part]
        if existing:
            parts.append(existing)
        if parts:
            env[key] = ":".join(parts)
    return env


def _simple_header(size: int) -> dict[str, Any]:
    scale_deg = 0.01
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(size // 2),
        "CRPIX2": float(size // 2),
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def _rotated_cd(scale_deg: float, rotation_deg: float) -> tuple[float, float, float, float]:
    theta = np.deg2rad(float(rotation_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return (
        -scale_deg * cos_t,
        scale_deg * sin_t,
        scale_deg * sin_t,
        scale_deg * cos_t,
    )


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    rotation_deg: float = 0.0,
    scale_deg: float = 0.01,
) -> fits.Header:
    h = fits.Header()
    h["NAXIS"] = 2
    h["NAXIS1"] = size
    h["NAXIS2"] = size
    h["CTYPE1"] = "RA---TPV"
    h["CTYPE2"] = "DEC--TPV"
    h["CRPIX1"] = size / 2.0
    h["CRPIX2"] = size / 2.0
    h["CRVAL1"] = crval1
    h["CRVAL2"] = crval2
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    h["CD1_1"] = cd11
    h["CD1_2"] = cd12
    h["CD2_1"] = cd21
    h["CD2_2"] = cd22
    h["PV1_0"] = 0.0
    h["PV1_1"] = 1.0
    h["PV1_4"] = 2.0e-4
    h["PV1_5"] = -3.0e-4
    h["PV1_6"] = 1.5e-4
    h["PV1_7"] = 2.0e-6
    h["PV1_39"] = 2.0e-11
    h["PV2_0"] = 0.0
    h["PV2_1"] = 0.0
    h["PV2_2"] = 1.0
    h["PV2_4"] = -1.0e-4
    h["PV2_5"] = 2.5e-4
    h["PV2_6"] = -2.0e-4
    h["PV2_7"] = -1.5e-6
    h["PV2_39"] = -1.0e-11
    return h


def _astropy_sip_fixture() -> Path:
    return (
        Path(astropy.__file__).resolve().parent
        / "nddata"
        / "tests"
        / "data"
        / "sip-wcs.fits"
    )


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _sanitize_fits_header_value(value: Any) -> Any:
    if isinstance(value, str):
        cleaned = value.replace("\r", " ").replace("\n", " ")
        cleaned = "".join(ch if 32 <= ord(ch) <= 126 else " " for ch in cleaned)
        cleaned = " ".join(cleaned.split())
        return cleaned if cleaned else None
    return value


def _as_fits_header(header: dict[str, Any]) -> fits.Header:
    out = fits.Header()
    for key, value in header.items():
        if key.endswith("_COMMENT"):
            continue
        if key in {"", "COMMENT", "HISTORY"}:
            continue
        cleaned = _sanitize_fits_header_value(value)
        if cleaned is None:
            continue
        out[key] = cleaned
    return out


def _write_header_template(path: Path, header: dict[str, Any]) -> None:
    header_astropy = _as_fits_header(header)
    header_astropy.setdefault("SIMPLE", True)
    header_astropy.setdefault("BITPIX", -32)
    header_astropy.setdefault("NAXIS", 2)
    header_astropy.setdefault("NAXIS1", int(header["NAXIS1"]))
    header_astropy.setdefault("NAXIS2", int(header["NAXIS2"]))
    header_astropy.totextfile(str(path), endcard=True, overwrite=True)


_GRID_EQUIVALENT_IGNORE_KEYS = {
    "COMMENT",
    "HISTORY",
    "SIMPLE",
    "BITPIX",
    "EXTEND",
    "XTENSION",
    "EXTNAME",
    "EXTVER",
    "PCOUNT",
    "GCOUNT",
    "CHECKSUM",
    "DATASUM",
}


def _headers_equivalent_for_direct_copy(source: dict[str, Any], target: dict[str, Any]) -> bool:
    def normalize(header: dict[str, Any]) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for key, value in header.items():
            if str(key).endswith("_COMMENT"):
                continue
            if key in _GRID_EQUIVALENT_IGNORE_KEYS:
                continue
            out[key] = value
        return out

    return normalize(source) == normalize(target)


def _target_projection(header: dict[str, Any]) -> str:
    ctype1 = str(header.get("CTYPE1", ""))
    if len(ctype1) < 8:
        raise RuntimeError(f"cannot infer projection from CTYPE1={ctype1!r}")
    return ctype1[5:8]


def _validate_dfreproject_headers(
    source_header: dict[str, Any],
    target_header: dict[str, Any],
) -> None:
    for role, header in (("source", source_header), ("target", target_header)):
        ctype1 = str(header.get("CTYPE1", ""))
        ctype2 = str(header.get("CTYPE2", ""))
        if not ctype1.startswith("RA---") or not ctype2.startswith("DEC--"):
            raise RuntimeError(
                "dfreproject benchmark path currently supports only equatorial celestial WCS "
                f"(got {role} {ctype1!r}, {ctype2!r})"
            )
        if any(str(key).startswith("PV") for key in header):
            raise RuntimeError(
                f"dfreproject benchmark path currently skips PV/TPV-distorted {role} grids"
            )
        proj = _target_projection(header)
        if proj != "TAN":
            raise RuntimeError(
                "dfreproject benchmark path currently supports only TAN/SIP WCS "
                f"(got {role} projection {proj!r})"
            )


def _swarp_target_spec(target_header: dict[str, Any]) -> dict[str, Any]:
    header_astropy = _as_fits_header(target_header)
    wcs = WCS(header_astropy)
    ctype1 = str(target_header.get("CTYPE1", ""))
    ctype2 = str(target_header.get("CTYPE2", ""))
    if wcs.sip is not None or "SIP" in ctype1 or "SIP" in ctype2:
        raise RuntimeError("SWarp output-grid benchmark path does not support SIP target distortions")
    if not ctype1.startswith("RA---") or not ctype2.startswith("DEC--"):
        raise RuntimeError(
            "SWarp benchmark path currently supports only equatorial celestial targets "
            f"(got {ctype1!r}, {ctype2!r})"
        )
    if any(key.startswith("PV") for key in target_header):
        raise RuntimeError("SWarp benchmark path currently skips PV-distorted target grids")
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
    if matrix.shape != (2, 2):
        raise RuntimeError(f"unexpected pixel scale matrix shape {matrix.shape!r}")
    offdiag = matrix.copy()
    offdiag[0, 0] = 0.0
    offdiag[1, 1] = 0.0
    if float(np.abs(offdiag).max(initial=0.0)) > 1e-12:
        raise RuntimeError("SWarp benchmark path skips rotated/sheared target grids")
    scale_x = float(abs(matrix[0, 0]))
    scale_y = float(abs(matrix[1, 1]))
    if scale_x <= 0.0 or scale_y <= 0.0:
        raise RuntimeError("invalid target pixel scale")
    rel_diff = abs(scale_x - scale_y) / max(scale_x, scale_y)
    if rel_diff > 1e-8:
        raise RuntimeError("SWarp benchmark path currently requires square output pixels")
    width = int(target_header["NAXIS1"])
    height = int(target_header["NAXIS2"])
    swarp_crpix = np.array([[(width + 1.0) / 2.0, (height + 1.0) / 2.0]], dtype=np.float64)
    center_world = wcs.all_pix2world(swarp_crpix, 1)[0]
    return {
        "projection": _target_projection(target_header),
        "center_ra_deg": float(center_world[0] % 360.0),
        "center_dec_deg": float(center_world[1]),
        "pixel_scale_arcsec": scale_y * 3600.0,
        "image_size": (width, height),
    }


def _write_swarp_config(
    path: Path,
    *,
    target_header: dict[str, Any],
    imageout_path: Path,
    weightout_path: Path,
    combine_type: str,
    resampling_type: str,
    weight_type: str = "NONE",
    weight_images: Sequence[str] | None = None,
    fscalastro_type: str = "FIXED",
) -> None:
    if fscalastro_type not in {"NONE", "FIXED", "VARIABLE"}:
        raise ValueError("fscalastro_type must be one of {'NONE', 'FIXED', 'VARIABLE'}")
    spec = _swarp_target_spec(target_header)
    lines = [
        f"IMAGEOUT_NAME          {imageout_path}",
        f"WEIGHTOUT_NAME         {weightout_path}",
        f"WEIGHT_TYPE            {weight_type}",
        "COMBINE                Y",
        f"COMBINE_TYPE           {combine_type}",
        "CELESTIAL_TYPE         EQUATORIAL",
        f"PROJECTION_TYPE        {spec['projection']}",
        "PROJECTION_ERR         0.0",
        "CENTER_TYPE            MANUAL",
        f"CENTER                 {spec['center_ra_deg']:.10f}, {spec['center_dec_deg']:.10f}",
        "PIXELSCALE_TYPE        MANUAL",
        f"PIXEL_SCALE            {spec['pixel_scale_arcsec']:.10f}",
        f"IMAGE_SIZE             {spec['image_size'][0]},{spec['image_size'][1]}",
        "RESAMPLE               Y",
        f"RESAMPLING_TYPE        {resampling_type}",
        "INTERPOLATE            N",
        f"FSCALASTRO_TYPE        {fscalastro_type}",
        "SUBTRACT_BACK          N",
        "WRITE_XML              N",
        "VERBOSE_TYPE           QUIET",
        "NTHREADS               0",
        "DELETE_TMPFILES        Y",
    ]
    if weight_images is not None:
        lines.append(f"WEIGHT_IMAGE           {','.join(weight_images)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_mef(
    path: Path,
    science_data: list[np.ndarray],
    science_headers: list[fits.Header],
    *,
    weight_data: list[np.ndarray] | None = None,
    mask_data: list[np.ndarray] | None = None,
) -> None:
    hdus: list[fits.hdu.base.ExtensionHDU | fits.PrimaryHDU] = [fits.PrimaryHDU()]
    for idx, (data, header) in enumerate(zip(science_data, science_headers, strict=True), start=1):
        sci_header = header.copy()
        sci_header["EXTNAME"] = "SCI"
        sci_header["EXTVER"] = idx
        hdus.append(fits.ImageHDU(data=np.asarray(data, dtype=np.float32), header=sci_header, name="SCI"))
        if weight_data is not None:
            err_header = header.copy()
            err_header["EXTNAME"] = "ERR"
            err_header["EXTVER"] = idx
            hdus.append(
                fits.ImageHDU(
                    data=np.asarray(weight_data[idx - 1], dtype=np.float32),
                    header=err_header,
                    name="ERR",
                )
            )
        if mask_data is not None:
            dq_header = header.copy()
            dq_header["EXTNAME"] = "DQ"
            dq_header["EXTVER"] = idx
            hdus.append(
                fits.ImageHDU(
                    data=np.asarray(mask_data[idx - 1], dtype=np.float32),
                    header=dq_header,
                    name="DQ",
                )
            )
    fits.HDUList(hdus).writeto(path, overwrite=True)


def _load_weight_map(obs: ImageObservation, image: torch.Tensor) -> torch.Tensor:
    weight = torch.isfinite(image).to(dtype=torch.float32)
    if obs.weight_path is not None:
        source = torchsky.read(obs.weight_path, hdu=obs.weight_hdu).to(dtype=torch.float32)
        source = torch.where(
            torch.isfinite(source) & (source > 0.0),
            source,
            torch.zeros_like(source),
        )
        weight = weight * source
    if obs.mask_path is not None:
        mask = torchsky.read(obs.mask_path, hdu=obs.mask_hdu).to(dtype=torch.float32)
        mask = torch.where(
            torch.isfinite(mask) & (mask > 0.0),
            torch.ones_like(mask),
            torch.zeros_like(mask),
        )
        weight = weight * mask
    return weight * float(obs.weight)


def _reproject_weighted_observation(
    obs: ImageObservation,
    *,
    target_header: dict[str, Any],
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    if reproject_interp is None:  # pragma: no cover - checked by caller
        raise RuntimeError("reproject is unavailable")

    image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
    image = image.to(dtype=torch.float32)
    weight = _load_weight_map(obs, image)
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))

    if _headers_equivalent_for_direct_copy(dict(header), target_header) and tuple(int(v) for v in image.shape) == target_shape:
        numer = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
        return numer.cpu().numpy(), weight.cpu().numpy()

    source_header = _as_fits_header(dict(header))
    target_header_astropy = _as_fits_header(target_header)
    numer, _ = reproject_interp(
        (
            (torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight)
            .cpu()
            .numpy(),
            source_header,
        ),
        target_header_astropy,
        shape_out=target_shape,
        return_footprint=True,
        order=order,
    )
    denom, _ = reproject_interp(
        (weight.cpu().numpy(), source_header),
        target_header_astropy,
        shape_out=target_shape,
        return_footprint=True,
        order=order,
    )
    numer = np.nan_to_num(numer, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
    numer = np.where(denom > 0.0, numer, 0.0)
    return numer.astype(np.float32, copy=False), denom.astype(np.float32, copy=False)


def _dfreproject_weighted_observation(
    obs: ImageObservation,
    *,
    target_header: dict[str, Any],
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    if calculate_reprojection is None:  # pragma: no cover - checked by caller
        raise RuntimeError("dfreproject is unavailable")

    image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
    image = image.to(dtype=torch.float32)
    weight = _load_weight_map(obs, image)
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))

    if _headers_equivalent_for_direct_copy(dict(header), target_header) and tuple(int(v) for v in image.shape) == target_shape:
        numer = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
        return numer.cpu().numpy(), weight.cpu().numpy()

    _validate_dfreproject_headers(dict(header), target_header)
    source_header = _as_fits_header(dict(header))
    target_header_astropy = _as_fits_header(target_header)
    numer = calculate_reprojection(
        (
            (torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight)
            .cpu()
            .numpy(),
            source_header,
        ),
        target_header_astropy,
        shape_out=target_shape,
        order=order,
        conserve_flux=False,
        compute_jacobian=False,
        show_chunk_progress=False,
        show_log=False,
    )
    denom = calculate_reprojection(
        (weight.cpu().numpy(), source_header),
        target_header_astropy,
        shape_out=target_shape,
        order=order,
        conserve_flux=False,
        compute_jacobian=False,
        show_chunk_progress=False,
        show_log=False,
    )
    numer = np.asarray(numer, dtype=np.float32)
    denom = np.asarray(denom, dtype=np.float32)
    numer = np.nan_to_num(numer, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
    numer = np.where(denom > 0.0, numer, 0.0)
    return numer.astype(np.float32, copy=False), denom.astype(np.float32, copy=False)


def _coadd_with_reproject(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer = np.zeros(target_shape, dtype=np.float32)
    denom = np.zeros(target_shape, dtype=np.float32)
    for obs in observations:
        obs_numer, obs_denom = _reproject_weighted_observation(
            obs,
            target_header=target_header,
            order=order,
        )
        numer += obs_numer
        denom += obs_denom
    data = numer / np.clip(denom, 1e-8, None)
    return data, denom


def _coadd_with_dfreproject(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    order: str,
) -> tuple[np.ndarray, np.ndarray]:
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer = np.zeros(target_shape, dtype=np.float32)
    denom = np.zeros(target_shape, dtype=np.float32)
    for obs in observations:
        obs_numer, obs_denom = _dfreproject_weighted_observation(
            obs,
            target_header=target_header,
            order=order,
        )
        numer += obs_numer
        denom += obs_denom
    data = numer / np.clip(denom, 1e-8, None)
    return data, denom


def _coadd_with_drizzle(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    kernel: str,
) -> tuple[np.ndarray, np.ndarray]:
    if DrizzleResampler is None or calc_pixmap is None or estimate_pixel_scale_ratio is None:
        raise RuntimeError("drizzle is unavailable")

    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    target_wcs = WCS(_as_fits_header(target_header))
    target_wcs.array_shape = target_shape
    drizzle = DrizzleResampler(
        kernel=kernel,
        fillval=0.0,
        out_shape=target_shape,
        disable_ctx=True,
    )
    for obs in observations:
        image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
        image = image.to(dtype=torch.float32)
        weight = _load_weight_map(obs, image)
        source_shape = tuple(int(v) for v in image.shape)
        source_wcs = WCS(_as_fits_header(dict(header)))
        source_wcs.array_shape = source_shape
        pixmap = calc_pixmap(source_wcs, target_wcs, shape=source_shape)
        pscale_ratio = estimate_pixel_scale_ratio(
            source_wcs,
            target_wcs,
            refpix_from=source_wcs.wcs.crpix,
            refpix_to=target_wcs.wcs.crpix,
        )
        drizzle.add_image(
            torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy(),
            exptime=1.0,
            pixmap=pixmap,
            weight_map=weight.cpu().numpy(),
            iscale=pscale_ratio**2,
            pixel_scale_ratio=pscale_ratio,
            in_units="cps",
        )
    data = np.asarray(drizzle.out_img, dtype=np.float32)
    weight = np.asarray(drizzle.out_wht, dtype=np.float32)
    data = np.where(weight > 0.0, data, 0.0)
    return data, weight


def _project_with_montage(
    montage_bin: Path,
    input_path: Path,
    output_path: Path,
    template_path: Path,
) -> bool:
    completed = subprocess.run(
        [str(montage_bin), str(input_path), str(output_path), str(template_path)],
        cwd=ROOT,
        env=_external_tool_env(),
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode == 0:
        return True
    if "No overlap" in completed.stdout:
        return False
    if completed.returncode != 0:
        raise RuntimeError(
            f"Montage command failed ({montage_bin.name}): "
            f"{completed.stdout.strip()} {completed.stderr.strip()}".strip()
        )
    return True


def _supports_montage_fast_project(
    source_header: dict[str, Any],
    target_header: dict[str, Any],
) -> bool:
    allowed = {"TAN", "SIN", "ZEA", "STG", "ARC"}
    try:
        source_proj = _target_projection(source_header)
        target_proj = _target_projection(target_header)
    except RuntimeError:
        return False
    if source_proj not in allowed or target_proj not in allowed:
        return False
    if any("SIP" in str(source_header.get(key, "")) for key in ("CTYPE1", "CTYPE2")):
        return False
    if any("SIP" in str(target_header.get(key, "")) for key in ("CTYPE1", "CTYPE2")):
        return False
    if any(key.startswith("PV") for key in source_header) or any(key.startswith("PV") for key in target_header):
        return False
    return (
        str(source_header.get("CTYPE1", "")).startswith("RA---")
        and str(source_header.get("CTYPE2", "")).startswith("DEC--")
        and str(target_header.get("CTYPE1", "")).startswith("RA---")
        and str(target_header.get("CTYPE2", "")).startswith("DEC--")
    )


def _embed_on_target_grid(
    array: torch.Tensor,
    *,
    header: dict[str, Any],
    target_header: dict[str, Any],
    target_shape: tuple[int, int],
) -> np.ndarray:
    if tuple(int(v) for v in array.shape) == target_shape:
        return array.to(dtype=torch.float32).cpu().numpy()
    out = np.zeros(target_shape, dtype=np.float32)
    x0 = int(round(float(target_header["CRPIX1"]) - float(header["CRPIX1"])))
    y0 = int(round(float(target_header["CRPIX2"]) - float(header["CRPIX2"])))
    y1 = y0 + int(array.shape[0])
    x1 = x0 + int(array.shape[1])
    if x0 < 0 or y0 < 0 or x1 > target_shape[1] or y1 > target_shape[0]:
        raise RuntimeError(
            "Montage output footprint could not be embedded on the target grid "
            f"(shape={tuple(int(v) for v in array.shape)}, target={target_shape})"
        )
    out[y0:y1, x0:x1] = array.to(dtype=torch.float32).cpu().numpy()
    return out


def _masked_max(values: torch.Tensor, mask: torch.Tensor) -> float:
    if not bool(mask.any()):
        return 0.0
    return float(values[mask].max().item())


def _coadd_with_montage(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str, str]:
    default_bin = _find_montage_binary("mProject")
    fast_bin = _find_montage_binary("mProjectPP")
    if default_bin is None:
        raise RuntimeError("montage is unavailable")

    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer = np.zeros(target_shape, dtype=np.float32)
    denom = np.zeros(target_shape, dtype=np.float32)
    mode = "mProject"
    backend_path = str(default_bin)
    with tempfile.TemporaryDirectory(prefix="torchsky_montage_") as tmp:
        tmpdir = Path(tmp)
        template_path = tmpdir / "target.hdr"
        _write_header_template(template_path, target_header)
        for idx, obs in enumerate(observations):
            image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
            image = image.to(dtype=torch.float32)
            weight = _load_weight_map(obs, image)
            montage_bin = default_bin
            if fast_bin is not None and _supports_montage_fast_project(dict(header), target_header):
                montage_bin = fast_bin
                mode = "mProjectPP"
                backend_path = str(fast_bin)
            else:
                mode = "mProject"
                backend_path = str(default_bin)
            if _headers_equivalent_for_direct_copy(dict(header), target_header) and tuple(int(v) for v in image.shape) == target_shape:
                obs_numer = torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
                numer += obs_numer.cpu().numpy()
                denom += weight.cpu().numpy()
                continue

            numer_in = tmpdir / f"montage_{idx}_numer_in.fits"
            denom_in = tmpdir / f"montage_{idx}_denom_in.fits"
            numer_out = tmpdir / f"montage_{idx}_numer_out.fits"
            denom_out = tmpdir / f"montage_{idx}_denom_out.fits"
            torchsky.write(
                str(numer_in),
                torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight,
                header=dict(header),
                overwrite=True,
            )
            torchsky.write(str(denom_in), weight, header=dict(header), overwrite=True)
            numer_ok = _project_with_montage(montage_bin, numer_in, numer_out, template_path)
            denom_ok = _project_with_montage(montage_bin, denom_in, denom_out, template_path)
            if not numer_ok or not denom_ok:
                continue
            numer_proj, numer_header = torchsky.read(str(numer_out), return_header=True)
            denom_proj, denom_header = torchsky.read(str(denom_out), return_header=True)
            numer_proj = torch.nan_to_num(numer_proj, nan=0.0, posinf=0.0, neginf=0.0)
            denom_proj = torch.nan_to_num(denom_proj, nan=0.0, posinf=0.0, neginf=0.0)
            numer += _embed_on_target_grid(
                numer_proj,
                header=dict(numer_header),
                target_header=target_header,
                target_shape=target_shape,
            )
            denom += _embed_on_target_grid(
                denom_proj,
                header=dict(denom_header),
                target_header=target_header,
                target_shape=target_shape,
            )

    numer = np.nan_to_num(numer, nan=0.0, posinf=0.0, neginf=0.0)
    denom = np.nan_to_num(denom, nan=0.0, posinf=0.0, neginf=0.0)
    data = numer / np.clip(denom, 1e-8, None)
    data = np.where(denom > 0.0, data, 0.0)
    return data.astype(np.float32, copy=False), denom.astype(np.float32, copy=False), mode, backend_path


def _write_single_extension_mef(path: Path, data: np.ndarray, header: fits.Header) -> None:
    ext_header = header.copy()
    ext_header["EXTNAME"] = "SCI"
    ext_header["EXTVER"] = 1
    ext_header.setdefault("EXPTIME", 1.0)
    primary_header = fits.Header()
    primary_header["EXPTIME"] = float(ext_header["EXPTIME"])
    fits.HDUList(
        [
            fits.PrimaryHDU(header=primary_header),
            fits.ImageHDU(data=np.asarray(data), header=ext_header, name="SCI"),
        ]
    ).writeto(path, overwrite=True)


def _validate_casutools_header(header: dict[str, Any]) -> None:
    ctype1 = str(header.get("CTYPE1", ""))
    ctype2 = str(header.get("CTYPE2", ""))
    if "SIP" in ctype1 or "SIP" in ctype2:
        raise RuntimeError("casutools imstack currently skips SIP-distorted inputs")
    if any(str(key).startswith("PV") for key in header):
        raise RuntimeError("casutools imstack currently skips PV/TPV-distorted inputs")
    if not ctype1.startswith("RA---") or not ctype2.startswith("DEC--"):
        raise RuntimeError(
            "casutools imstack currently supports only equatorial celestial WCS "
            f"(got {ctype1!r}, {ctype2!r})"
        )
    proj = _target_projection(header)
    if proj not in {"TAN", "ZPN"}:
        raise RuntimeError(f"casutools imstack currently supports only TAN/ZPN inputs (got {proj!r})")


def _align_array_to_target(
    array: np.ndarray,
    *,
    header: dict[str, Any],
    target_header: dict[str, Any],
    order: str = "bilinear",
) -> np.ndarray:
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    if _headers_equivalent_for_direct_copy(header, target_header) and tuple(int(v) for v in array.shape) == target_shape:
        return np.asarray(array, dtype=np.float32)
    if reproject_interp is not None:
        aligned, _ = reproject_interp(
            (np.asarray(array, dtype=np.float32), _as_fits_header(header)),
            _as_fits_header(target_header),
            shape_out=target_shape,
            return_footprint=True,
            order=order,
        )
        return np.nan_to_num(np.asarray(aligned, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)

    # Fall back to torchsky's own image-plane mapper so projection-changing
    # legacy backends remain benchmarkable in lean environments.
    kernel = str(order).lower()
    if kernel == "nearest-neighbor":
        kernel = "nearest"
    with tempfile.TemporaryDirectory(prefix="torchsky_align_fallback_") as tmp:
        tmpdir = Path(tmp)
        image_path = tmpdir / "input.fits"
        _write_single_extension_mef(image_path, np.asarray(array, dtype=np.float32), _as_fits_header(header))
        result = coadd_image_plane(
            [ImageObservation(str(image_path), hdu=1)],
            target_header=dict(target_header),
            resampling_kernel=kernel,
            value_semantics="surface_brightness",
            stacking_method="weighted_mean",
        )
        aligned = result.data.cpu().numpy()
    return np.nan_to_num(np.asarray(aligned, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _coadd_with_casutools(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, str]:
    imstack_bin = _find_casutools_binary("imstack")
    if imstack_bin is None:
        raise RuntimeError("casutools imstack is unavailable")

    loaded: list[tuple[torch.Tensor, dict[str, Any], torch.Tensor]] = []
    max_weight = 0.0
    for obs in observations:
        image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
        image = image.to(dtype=torch.float32)
        _validate_casutools_header(dict(header))
        weight = _load_weight_map(obs, image)
        max_weight = max(max_weight, float(weight.max().item()))
        loaded.append((image, dict(header), weight))
    if max_weight <= 0.0:
        raise RuntimeError("casutools comparator requires at least one positive-weight pixel")

    with tempfile.TemporaryDirectory(prefix="torchsky_casutools_") as tmp:
        tmpdir = Path(tmp)
        image_list_path = tmpdir / "images.txt"
        conf_list_path = tmpdir / "confs.txt"
        image_paths: list[str] = []
        conf_paths: list[str] = []
        for idx, (image, header, weight) in enumerate(loaded):
            image_path = tmpdir / f"casu_image_{idx}.fits"
            conf_path = tmpdir / f"casu_conf_{idx}.fits"
            conf = torch.round((1000.0 * weight / max_weight).clamp(min=0.0, max=1000.0)).to(dtype=torch.int16)
            _write_single_extension_mef(
                image_path,
                torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0).cpu().numpy().astype(np.float32, copy=False),
                _as_fits_header(header),
            )
            _write_single_extension_mef(conf_path, conf.cpu().numpy().astype(np.int16, copy=False), _as_fits_header(header))
            image_paths.append(str(image_path))
            conf_paths.append(str(conf_path))
        image_list_path.write_text("\n".join(image_paths) + "\n", encoding="utf-8")
        conf_list_path.write_text("\n".join(conf_paths) + "\n", encoding="utf-8")

        out_path = tmpdir / "casu_imstack.fits"
        conf_out_path = tmpdir / "casu_imstack_conf.fits"
        completed = subprocess.run(
            [
                str(imstack_bin),
                f"@{image_list_path}",
                f"@{conf_list_path}",
                "",
                str(out_path),
                str(conf_out_path),
                "--method=1",
                "--lthr=1000000",
                "--hthr=1000000",
            ],
            cwd=tmpdir,
            env=_external_tool_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(f"casutools imstack failed: {completed.stdout.strip()} {completed.stderr.strip()}".strip())

        data, out_header = torchsky.read(str(out_path), hdu=1, return_header=True)
        conf, conf_header = torchsky.read(str(conf_out_path), hdu=1, return_header=True)
        data_np = _align_array_to_target(
            data.to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False),
            header=dict(out_header),
            target_header=target_header,
            order="bilinear",
        )
        coverage_np = _align_array_to_target(
            (conf.to(dtype=torch.float32) > 0.0).cpu().numpy().astype(np.float32, copy=False),
            header=dict(conf_header),
            target_header=target_header,
            order="nearest-neighbor",
        )
        coverage_np = (coverage_np > 0.0).astype(np.float32, copy=False)
        data_np = np.where(coverage_np > 0.0, data_np, 0.0)
        return data_np, coverage_np, str(imstack_bin)


def _coadd_with_swarp(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    resampling_type: str,
    fscalastro_type: str = "NONE",
) -> tuple[np.ndarray, np.ndarray]:
    swarp_bin = _find_swarp_binary()
    if swarp_bin is None:
        raise RuntimeError("SWarp is unavailable")
    _swarp_target_spec(target_header)
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))

    if observations:
        direct_numer = torch.zeros(target_shape, dtype=torch.float32)
        direct_weight = torch.zeros(target_shape, dtype=torch.float32)
        all_direct_copy = True
        for obs in observations:
            image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
            image = image.to(dtype=torch.float32)
            if not (
                _headers_equivalent_for_direct_copy(dict(header), target_header)
                and tuple(int(v) for v in image.shape) == target_shape
            ):
                all_direct_copy = False
                break
            weight = _load_weight_map(obs, image)
            direct_numer += torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
            direct_weight += weight
        if all_direct_copy:
            direct_data = torch.where(
                direct_weight > 0.0,
                direct_numer / torch.clamp(direct_weight, min=1e-8),
                torch.zeros_like(direct_weight),
            )
            return direct_data.cpu().numpy(), direct_weight.cpu().numpy()

    with tempfile.TemporaryDirectory(prefix="torchsky_swarp_") as tmp:
        tmpdir = Path(tmp)
        numer_paths: list[str] = []
        denom_paths: list[str] = []
        for idx, obs in enumerate(observations):
            image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
            image = image.to(dtype=torch.float32)
            weight = _load_weight_map(obs, image)
            numer_path = tmpdir / f"swarp_{idx}_numer.fits"
            denom_path = tmpdir / f"swarp_{idx}_denom.fits"
            torchsky.write(
                str(numer_path),
                torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight,
                header=dict(header),
                overwrite=True,
            )
            torchsky.write(str(denom_path), weight, header=dict(header), overwrite=True)
            numer_paths.append(str(numer_path))
            denom_paths.append(str(denom_path))

        numer_imageout_path = tmpdir / "swarp_numer.fits"
        numer_weightout_path = tmpdir / "swarp_numer.weight.fits"
        denom_imageout_path = tmpdir / "swarp_denom.fits"
        denom_weightout_path = tmpdir / "swarp_denom.weight.fits"
        numer_config_path = tmpdir / "swarp_numer.conf"
        denom_config_path = tmpdir / "swarp_denom.conf"
        _write_swarp_config(
            numer_config_path,
            target_header=target_header,
            imageout_path=numer_imageout_path,
            weightout_path=numer_weightout_path,
            combine_type="SUM",
            resampling_type=resampling_type,
            fscalastro_type=fscalastro_type,
        )
        _write_swarp_config(
            denom_config_path,
            target_header=target_header,
            imageout_path=denom_imageout_path,
            weightout_path=denom_weightout_path,
            combine_type="SUM",
            resampling_type=resampling_type,
            fscalastro_type=fscalastro_type,
        )
        numer_completed = subprocess.run(
            [str(swarp_bin), *numer_paths, "-c", str(numer_config_path)],
            cwd=tmpdir,
            env=_external_tool_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        denom_completed = subprocess.run(
            [str(swarp_bin), *denom_paths, "-c", str(denom_config_path)],
            cwd=tmpdir,
            env=_external_tool_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if numer_completed.returncode != 0:
            raise RuntimeError(
                f"SWarp numerator pass failed: "
                f"{numer_completed.stdout.strip()} {numer_completed.stderr.strip()}".strip()
            )
        if denom_completed.returncode != 0:
            raise RuntimeError(
                f"SWarp denominator pass failed: "
                f"{denom_completed.stdout.strip()} {denom_completed.stderr.strip()}".strip()
            )
        numer = torchsky.read(str(numer_imageout_path)).to(dtype=torch.float32)
        weight = torchsky.read(str(denom_imageout_path)).to(dtype=torch.float32)
        data = numer / torch.clamp(weight, min=1e-8)
        if tuple(int(v) for v in data.shape) != target_shape:
            raise RuntimeError(
                "SWarp output shape did not match requested target grid "
                f"(got {tuple(int(v) for v in data.shape)})"
            )
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        data = torch.where(weight > 0.0, data, torch.zeros_like(data))
        return data.cpu().numpy(), weight.cpu().numpy()


def _coadd_with_swarp_direct_stack(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    resampling_type: str,
    combine_type: str,
    fscalastro_type: str = "NONE",
) -> tuple[np.ndarray, np.ndarray]:
    swarp_bin = _find_swarp_binary()
    if swarp_bin is None:
        raise RuntimeError("SWarp is unavailable")
    _swarp_target_spec(target_header)
    with tempfile.TemporaryDirectory(prefix="torchsky_swarp_direct_") as tmp:
        tmpdir = Path(tmp)
        image_paths: list[str] = []
        weight_paths: list[str] = []
        for idx, obs in enumerate(observations):
            image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
            image = image.to(dtype=torch.float32)
            support = (_load_weight_map(obs, image) > 0.0).to(dtype=torch.float32)
            scale = float(obs.photometric_scale)
            background = float(obs.background_offset)
            if not math.isfinite(scale) or scale <= 0.0:
                raise RuntimeError("SWarp direct combine benchmark path requires finite positive photometric_scale")
            if not math.isfinite(background):
                raise RuntimeError("SWarp direct combine benchmark path requires finite background_offset")
            image_path = tmpdir / f"swarp_direct_{idx}.fits"
            weight_path = tmpdir / f"swarp_direct_{idx}.weight.fits"
            torchsky.write(
                str(image_path),
                scale
                * (
                    torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
                    - background
                ),
                header=dict(header),
                overwrite=True,
            )
            torchsky.write(
                str(weight_path),
                support,
                header=dict(header),
                overwrite=True,
            )
            image_paths.append(str(image_path))
            weight_paths.append(str(weight_path))

        imageout_path = tmpdir / "swarp_direct.fits"
        weightout_path = tmpdir / "swarp_direct.weight.fits"
        config_path = tmpdir / "swarp_direct.conf"
        _write_swarp_config(
            config_path,
            target_header=target_header,
            imageout_path=imageout_path,
            weightout_path=weightout_path,
            combine_type=combine_type,
            resampling_type=resampling_type,
            weight_type="MAP_WEIGHT",
            weight_images=weight_paths,
            fscalastro_type=fscalastro_type,
        )
        completed = subprocess.run(
            [str(swarp_bin), *image_paths, "-c", str(config_path)],
            cwd=tmpdir,
            env=_external_tool_env(),
            capture_output=True,
            text=True,
            check=False,
        )
        if completed.returncode != 0:
            raise RuntimeError(
                f"SWarp direct pass failed: "
                f"{completed.stdout.strip()} {completed.stderr.strip()}".strip()
            )
        data = torchsky.read(str(imageout_path)).to(dtype=torch.float32)
        if weightout_path.exists():
            weight = torchsky.read(str(weightout_path)).to(dtype=torch.float32)
        else:
            weight = torch.isfinite(data).to(dtype=torch.float32)
        if tuple(int(v) for v in data.shape) != (int(target_header["NAXIS2"]), int(target_header["NAXIS1"])):
            raise RuntimeError(
                "SWarp output shape did not match requested target grid "
                f"(got {tuple(int(v) for v in data.shape)})"
            )
        data = torch.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        weight = torch.nan_to_num(weight, nan=0.0, posinf=0.0, neginf=0.0)
        data = torch.where(weight > 0.0, data, torch.zeros_like(data))
        return data.cpu().numpy(), weight.cpu().numpy()


def _diff_summary(
    data: torch.Tensor,
    weight: torch.Tensor,
    ref_data: torch.Tensor,
    ref_weight: torch.Tensor,
    *,
    suffix: str,
    data_mask: torch.Tensor | None = None,
    compare_weight_as_mask: bool = False,
) -> dict[str, float]:
    if compare_weight_as_mask:
        weight = (weight > 0.0).to(dtype=torch.float32)
        ref_weight = (ref_weight > 0.0).to(dtype=torch.float32)
    data_diff = torch.abs(data - ref_data.to(dtype=torch.float32))
    weight_diff = torch.abs(weight - ref_weight.to(dtype=torch.float32))
    edge = torch.zeros_like(weight_diff, dtype=torch.bool)
    edge[0, :] = True
    edge[-1, :] = True
    edge[:, 0] = True
    edge[:, -1] = True
    if data_mask is None:
        data_mask = torch.ones_like(edge)
    else:
        data_mask = data_mask.to(dtype=torch.bool)
    edge_data_mask = edge & data_mask
    interior_data_mask = (~edge) & data_mask
    if bool(data_mask.any()):
        masked_data_diff = data_diff[data_mask]
        data_rmse = float(torch.sqrt(torch.mean(torch.square(masked_data_diff))).item())
        data_mae = float(torch.mean(masked_data_diff).item())
        compare_frac = float(data_mask.to(dtype=torch.float32).mean().item())
    else:
        data_rmse = 0.0
        data_mae = 0.0
        compare_frac = 0.0
    weight_rmse = float(torch.sqrt(torch.mean(torch.square(weight_diff))).item())
    weight_mae = float(torch.mean(weight_diff).item())
    return {
        f"rmse_data_diff{suffix}": data_rmse,
        f"mae_data_diff{suffix}": data_mae,
        f"rmse_weight_diff{suffix}": weight_rmse,
        f"mae_weight_diff{suffix}": weight_mae,
        f"comparison_support_frac{suffix}": compare_frac,
        f"max_abs_data_diff{suffix}": _masked_max(data_diff, data_mask),
        f"max_abs_weight_diff{suffix}": float(weight_diff.max().item()),
        f"max_abs_data_diff_edge{suffix}": _masked_max(data_diff, edge_data_mask),
        f"max_abs_weight_diff_edge{suffix}": float(weight_diff[edge].max().item()),
        f"max_abs_data_diff_interior{suffix}": _masked_max(data_diff, interior_data_mask),
        f"max_abs_weight_diff_interior{suffix}": float(weight_diff[~edge].max().item())
        if bool((~edge).any())
        else 0.0,
    }


def _pairwise_backend_row(
    case: str,
    left_row: dict[str, Any],
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_row: dict[str, Any],
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
) -> dict[str, Any]:
    left = str(left_row["library"])
    right = str(right_row["library"])
    left_variant = str(left_row.get("variant", "default"))
    right_variant = str(right_row.get("variant", "default"))
    compare_weight_as_mask = (
        left_row.get("comparison_weight_mode") == "coverage_mask"
        or right_row.get("comparison_weight_mode") == "coverage_mask"
    )
    data_mask: torch.Tensor | None = None
    if (
        left_row.get("comparison_data_domain") == "common_support"
        or right_row.get("comparison_data_domain") == "common_support"
    ):
        data_mask = (left_weight > 0.0) & (right_weight > 0.0)
    return {
        "case": case,
        "library": "pairwise",
        "status": "OK",
        "comparison_type": "external_backend_pair",
        "left_library": left,
        "right_library": right,
        "left_variant": left_variant,
        "right_variant": right_variant,
        "pair": f"{left}:{left_variant}_vs_{right}:{right_variant}",
        "left_time_ms": float(left_row["time_ms"]),
        "right_time_ms": float(right_row["time_ms"]),
        "speed_ratio_left_vs_right": float(left_row["time_ms"]) / max(float(right_row["time_ms"]), 1e-12),
        "comparison_weight_mode": "coverage_mask" if compare_weight_as_mask else "absolute",
        "comparison_data_domain": "common_support" if data_mask is not None else "full_grid",
        **_diff_summary(
            left_data,
            left_weight,
            right_data,
            right_weight,
            suffix="",
            data_mask=data_mask,
            compare_weight_as_mask=compare_weight_as_mask,
        ),
    }


def _prepare_synthetic_shifted_case(
    tmpdir: Path,
    *,
    size: int,
    n_obs: int,
) -> list[ImageObservation]:
    header = _simple_header(size)
    base = torch.linspace(0.0, 1.0, size * size, dtype=torch.float32).reshape(size, size)
    observations: list[ImageObservation] = []
    for idx in range(n_obs):
        image_path = tmpdir / f"shifted_{idx}.fits"
        weight_path = tmpdir / f"shifted_{idx}_wht.fits"
        obs_header = dict(header)
        obs_header["CRVAL1"] = float(obs_header["CRVAL1"]) + 0.0025 * idx
        obs_header["CRVAL2"] = float(obs_header["CRVAL2"]) - 0.0015 * idx
        image = base + float(idx)
        weight = torch.ones_like(image)
        weight[:, : size // 4] = 2.0 + float(idx)
        torchsky.write(str(image_path), image, header=obs_header, overwrite=True)
        torchsky.write(str(weight_path), weight, header=obs_header, overwrite=True)
        observations.append(ImageObservation(str(image_path), weight_path=str(weight_path)))
    return observations


def _prepare_upstream_drizzle_translation_case(tmpdir: Path, *, size: int) -> list[ImageObservation]:
    header = _simple_header(size)
    translated_header = dict(header)
    translated_header["CRPIX1"] = float(translated_header["CRPIX1"]) - 2.0
    translated_header["CRPIX2"] = float(translated_header["CRPIX2"]) - 2.0
    image = torch.zeros((size, size), dtype=torch.float32)
    image[size // 4, size // 3] = 3.0
    image[size // 2, size // 2] = 7.0
    image[(3 * size) // 4, (2 * size) // 3] = 11.0
    weight = torch.ones_like(image)

    base_path = tmpdir / "drizzle_translation_base.fits"
    translated_path = tmpdir / "drizzle_translation_shifted.fits"
    weight_path = tmpdir / "drizzle_translation_wht.fits"
    torchsky.write(str(base_path), image, header=header, overwrite=True)
    torchsky.write(str(translated_path), image, header=translated_header, overwrite=True)
    torchsky.write(str(weight_path), weight, header=header, overwrite=True)
    return [
        ImageObservation(str(base_path), weight_path=str(weight_path)),
        ImageObservation(str(translated_path)),
    ]


def _prepare_real_case(tmpdir: Path) -> list[ImageObservation]:
    image = torchsky.read(str(_astropy_sip_fixture())).to(dtype=torch.float32)
    mask_a = tmpdir / "mask_a.fits"
    mask_b = tmpdir / "mask_b.fits"
    left = torch.zeros_like(image)
    left[:, : image.shape[1] // 2] = 1.0
    right = torch.zeros_like(image)
    right[:, image.shape[1] // 2 :] = 1.0
    torchsky.write(str(mask_a), left, overwrite=True)
    torchsky.write(str(mask_b), right, overwrite=True)
    return [
        ImageObservation(str(_astropy_sip_fixture()), mask_path=str(mask_a)),
        ImageObservation(str(_astropy_sip_fixture()), mask_path=str(mask_b)),
    ]


def _prepare_real_shifted_sip_case(tmpdir: Path) -> list[ImageObservation]:
    image, header = torchsky.read(str(_astropy_sip_fixture()), return_header=True)
    image = image.to(dtype=torch.float32)
    shifted_header = dict(header)
    shifted_header["CRVAL1"] = float(shifted_header["CRVAL1"]) + 0.01
    shifted_header["CRVAL2"] = float(shifted_header["CRVAL2"]) - 0.01
    shifted_path = tmpdir / "sip_shifted.fits"
    torchsky.write(str(shifted_path), image, header=shifted_header, overwrite=True)
    return [
        ImageObservation(str(_astropy_sip_fixture())),
        ImageObservation(str(shifted_path)),
    ]


def _prepare_real_sip_mef_case(
    tmpdir: Path,
    *,
    shifted: bool,
) -> list[ImageObservation]:
    image, header = torchsky.read(str(_astropy_sip_fixture()), return_header=True)
    data_np = image.to(dtype=torch.float32).cpu().numpy().astype(np.float32, copy=False)
    header_1 = _as_fits_header(dict(header))
    header_2 = _as_fits_header(dict(header))
    if shifted:
        header_2["CRVAL1"] = float(header_2["CRVAL1"]) + 0.01
        header_2["CRVAL2"] = float(header_2["CRVAL2"]) - 0.01

    weight_1 = np.ones_like(data_np, dtype=np.float32)
    weight_2 = np.ones_like(data_np, dtype=np.float32)
    weight_1[:, : data_np.shape[1] // 3] = 2.0
    weight_2[:, (2 * data_np.shape[1]) // 3 :] = 3.0
    mask_1 = np.ones_like(data_np, dtype=np.float32)
    mask_2 = np.ones_like(data_np, dtype=np.float32)
    mask_1[:, (2 * data_np.shape[1]) // 3 :] = 0.0
    mask_2[:, : data_np.shape[1] // 3] = 0.0

    mef_path = tmpdir / ("real_sip_shifted_mef.fits" if shifted else "real_sip_mef.fits")
    _write_mef(
        mef_path,
        [data_np, data_np],
        [header_1, header_2],
        weight_data=[weight_1, weight_2],
        mask_data=[mask_1, mask_2],
    )
    return [
        ImageObservation(
            str(mef_path),
            hdu=1,
            weight_path=str(mef_path),
            weight_hdu=2,
            mask_path=str(mef_path),
            mask_hdu=3,
        ),
        ImageObservation(
            str(mef_path),
            hdu=4,
            weight_path=str(mef_path),
            weight_hdu=5,
            mask_path=str(mef_path),
            mask_hdu=6,
        ),
    ]


def _prepare_synthetic_tpv_mef_case(tmpdir: Path, *, size: int) -> list[ImageObservation]:
    base_header = _as_fits_header(_simple_header(size))
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    image = (
        0.7 * torch.exp(-((xx + 0.2) ** 2 + (yy - 0.15) ** 2) / 0.08)
        + 0.9 * torch.exp(-((xx - 0.35) ** 2 + (yy + 0.25) ** 2) / 0.03)
        + 0.2 * torch.sin(5.0 * xx)
        + 0.1 * torch.cos(4.0 * yy)
    ).to(dtype=torch.float32)
    sci1 = base_header.copy()
    sci1["EXTNAME"] = "SCI"
    sci1["EXTVER"] = 1
    sci2 = _tpv_header(
        size,
        crval1=float(base_header["CRVAL1"]) + 0.42,
        crval2=float(base_header["CRVAL2"]) - 0.18,
        rotation_deg=14.0,
        scale_deg=0.01,
    )
    sci2["EXTNAME"] = "SCI"
    sci2["EXTVER"] = 2
    err1 = sci1.copy()
    err1["EXTNAME"] = "ERR"
    err2 = sci2.copy()
    err2["EXTNAME"] = "ERR"
    dq1 = sci1.copy()
    dq1["EXTNAME"] = "DQ"
    dq2 = sci2.copy()
    dq2["EXTNAME"] = "DQ"

    weight1 = (0.7 + 1.3 * (xx + 1.0) / 2.0).to(dtype=torch.float32)
    weight2 = (0.6 + 1.6 * (yy + 1.0) / 2.0).to(dtype=torch.float32)
    mask1 = torch.ones_like(image, dtype=torch.float32)
    mask2 = torch.ones_like(image, dtype=torch.float32)
    mask1[:, size // 3 : size // 3 + max(2, size // 16)] = 0.0
    mask2[size // 2 - max(2, size // 20) : size // 2 + max(2, size // 20), :] = 0.0
    mef_path = tmpdir / f"synthetic_tpv_mef_{size}.fits"
    _write_mef(
        mef_path,
        [image.cpu().numpy(), image.cpu().numpy()],
        [sci1, sci2],
        weight_data=[weight1.cpu().numpy(), weight2.cpu().numpy()],
        mask_data=[mask1.cpu().numpy(), mask2.cpu().numpy()],
    )
    return [
        ImageObservation(str(mef_path), hdu=1, weight_path=str(mef_path), weight_hdu=2, mask_path=str(mef_path), mask_hdu=3),
        ImageObservation(str(mef_path), hdu=4, weight_path=str(mef_path), weight_hdu=5, mask_path=str(mef_path), mask_hdu=6),
    ]


def _torchsky_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    resampling_kernel: str,
) -> tuple[dict[str, Any], torch.Tensor, torch.Tensor]:
    dt = _time_many(
        lambda: coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel=resampling_kernel,
        ),
        runs,
    )
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel=resampling_kernel,
    )
    row = {
        "case": case,
        "library": "torchsky",
        "method": "torchsky",
        "variant": resampling_kernel,
        "status": "OK",
        "shape": list(result.data.shape),
        "n_obs": len(observations),
        "time_ms": dt * 1000.0,
        "mpix_s": (float(result.data.numel()) / dt) / 1e6,
        "coverage_frac": float((result.weight > 0.0).to(torch.float32).mean().item()),
        "mean_weight": float(result.weight.mean().item()),
        "interpolation_kernel": resampling_kernel,
        "resampling_method": "resample_weighted_numerator_and_denominator",
        "stacking_method": "weighted_mean",
    }
    return row, result.data, result.weight


def _reproject_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
    order: str,
    variant: str,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    if reproject_interp is None:
        return (
            {
                "case": case,
                "library": "reproject_interp",
                "status": "SKIPPED",
                "reason": "optional dependency reproject is unavailable",
            },
            None,
            None,
        )

    try:
        dt = _time_many(
            lambda: _coadd_with_reproject(
                observations,
                target_header=target_header,
                order=order,
            ),
            runs,
        )
        data_np, weight_np = _coadd_with_reproject(
            observations,
            target_header=target_header,
            order=order,
        )
    except Exception as exc:
        return _skipped_row(case, "reproject_interp", f"reproject failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    weight = torch.from_numpy(weight_np)
    return (
        {
            "case": case,
            "library": "reproject_interp",
            "method": "reproject_interp",
            "variant": variant,
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float((weight > 0.0).to(torch.float32).mean().item()),
            "mean_weight": float(weight.mean().item()),
            "interpolation_kernel": order,
            "comparison_weight_mode": "absolute",
            "comparison_data_domain": "full_grid",
            **_diff_summary(
                data,
                weight,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
            ),
        },
        data,
        weight,
    )


def _dfreproject_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
    order: str,
    variant: str,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    if calculate_reprojection is None:
        return _skipped_row(case, "dfreproject", "optional dependency dfreproject is unavailable"), None, None

    try:
        dt = _time_many(
            lambda: _coadd_with_dfreproject(
                observations,
                target_header=target_header,
                order=order,
            ),
            runs,
        )
        data_np, weight_np = _coadd_with_dfreproject(
            observations,
            target_header=target_header,
            order=order,
        )
    except Exception as exc:
        return _skipped_row(case, "dfreproject", f"dfreproject failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    weight = torch.from_numpy(weight_np)
    return (
        {
            "case": case,
            "library": "dfreproject",
            "method": "dfreproject",
            "variant": variant,
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float((weight > 0.0).to(torch.float32).mean().item()),
            "mean_weight": float(weight.mean().item()),
            "interpolation_kernel": order,
            "comparison_weight_mode": "absolute",
            "comparison_data_domain": "full_grid",
            **_diff_summary(
                data,
                weight,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
            ),
        },
        data,
        weight,
    )


def _drizzle_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
    kernel: str,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    if DrizzleResampler is None or calc_pixmap is None or estimate_pixel_scale_ratio is None:
        return _skipped_row(case, "drizzle", "optional dependency drizzle is unavailable"), None, None

    try:
        dt = _time_many(
            lambda: _coadd_with_drizzle(
                observations,
                target_header=target_header,
                kernel=kernel,
            ),
            runs,
        )
        data_np, weight_np = _coadd_with_drizzle(
            observations,
            target_header=target_header,
            kernel=kernel,
        )
    except Exception as exc:
        return _skipped_row(case, "drizzle", f"drizzle failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    weight = torch.from_numpy(weight_np)
    return (
        {
            "case": case,
            "library": "drizzle",
            "method": "drizzle",
            "variant": kernel,
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float((weight > 0.0).to(torch.float32).mean().item()),
            "mean_weight": float(weight.mean().item()),
            "kernel": kernel,
            "comparison_weight_mode": "absolute",
            "comparison_data_domain": "full_grid",
            **_diff_summary(
                data,
                weight,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
            ),
        },
        data,
        weight,
    )


def _montage_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    montage_bin = _find_montage_binary("mProject")
    if montage_bin is None:
        if MontageMain is None:
            return (
                _skipped_row(
                    case,
                    "montage",
                    f"optional dependency MontagePy is unavailable: {MONTAGEPY_IMPORT_ERROR}",
                ),
                None,
                None,
            )
        return _skipped_row(case, "montage", "Montage backend is unavailable"), None, None

    try:
        dt = _time_many(
            lambda: _coadd_with_montage(observations, target_header=target_header),
            runs,
        )
        data_np, weight_np, mode, backend_path = _coadd_with_montage(observations, target_header=target_header)
    except Exception as exc:
        return _skipped_row(case, "montage", f"Montage failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    weight = torch.from_numpy(weight_np)
    coverage = (weight > 0.0).to(dtype=torch.float32)
    common_support = (coverage > 0.0) & (torchsky_weight > 0.0)
    return (
        {
            "case": case,
            "library": "montage",
            "method": "montage",
            "variant": mode,
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float(coverage.mean().item()),
            "mean_weight": float(coverage.mean().item()),
            "backend": backend_path,
            "mode": mode,
            "comparison_weight_mode": "coverage_mask",
            "comparison_data_domain": "common_support",
            "weight_semantics": "coverage_mask_from_projected_denom",
            **_diff_summary(
                data,
                coverage,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
                data_mask=common_support,
                compare_weight_as_mask=True,
            ),
        },
        data,
        coverage,
    )


def _swarp_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
    resampling_type: str,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    swarp_bin = _find_swarp_binary()
    if swarp_bin is None:
        return _skipped_row(case, "swarp", "SWarp CLI is unavailable"), None, None
    try:
        spec = _swarp_target_spec(target_header)
    except RuntimeError as exc:
        return _skipped_row(case, "swarp", str(exc)), None, None

    try:
        dt = _time_many(
            lambda: _coadd_with_swarp(
                observations,
                target_header=target_header,
                resampling_type=resampling_type,
            ),
            runs,
        )
        data_np, weight_np = _coadd_with_swarp(
            observations,
            target_header=target_header,
            resampling_type=resampling_type,
        )
    except Exception as exc:
        return _skipped_row(case, "swarp", f"SWarp failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    weight = torch.from_numpy(weight_np)
    return (
        {
            "case": case,
            "library": "swarp",
            "method": "swarp",
            "variant": resampling_type.lower(),
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float((weight > 0.0).to(torch.float32).mean().item()),
            "mean_weight": float(weight.mean().item()),
            "backend": str(swarp_bin),
            "resampling_type": resampling_type,
            "combine_type": "SUM(num/denom)",
            "projection_type": spec["projection"],
            "comparison_weight_mode": "absolute",
            "comparison_data_domain": "full_grid",
            **_diff_summary(
                data,
                weight,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
            ),
        },
        data,
        weight,
    )


def _casutools_row(
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    imstack_bin = _find_casutools_binary("imstack")
    if imstack_bin is None:
        return _skipped_row(case, "casutools", "casutools imstack is unavailable"), None, None

    try:
        dt = _time_many(
            lambda: _coadd_with_casutools(observations, target_header=target_header),
            runs,
        )
        data_np, coverage_np, backend_path = _coadd_with_casutools(observations, target_header=target_header)
    except Exception as exc:
        return _skipped_row(case, "casutools", f"casutools imstack failed: {exc}"), None, None
    data = torch.from_numpy(data_np)
    coverage = torch.from_numpy(coverage_np)
    common_support = (coverage > 0.0) & (torchsky_weight > 0.0)
    return (
        {
            "case": case,
            "library": "casutools",
            "method": "casutools",
            "variant": "imstack",
            "status": "OK",
            "shape": list(data.shape),
            "n_obs": len(observations),
            "time_ms": dt * 1000.0,
            "mpix_s": (float(data.numel()) / dt) / 1e6,
            "coverage_frac": float(coverage.mean().item()),
            "mean_weight": float(coverage.mean().item()),
            "backend": backend_path,
            "comparison_weight_mode": "coverage_mask",
            "comparison_data_domain": "common_support",
            "output_alignment": "reproject_interp_to_target",
            "weight_semantics": "coverage_mask_from_confidence_map",
            **_diff_summary(
                data,
                coverage,
                torchsky_data,
                torchsky_weight,
                suffix="_vs_torchsky",
                data_mask=common_support,
                compare_weight_as_mask=True,
            ),
        },
        data,
        coverage,
    )


def _skipped_row(case: str, library: str, reason: str) -> dict[str, Any]:
    return {
        "case": case,
        "library": library,
        "status": "SKIPPED",
        "reason": reason,
    }


def _comparator_row(
    library: str,
    case: str,
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    runs: int,
    torchsky_row: dict[str, Any],
    torchsky_data: torch.Tensor,
    torchsky_weight: torch.Tensor,
    variant: str | None = None,
) -> tuple[dict[str, Any], torch.Tensor | None, torch.Tensor | None]:
    if library == "reproject_interp":
        order = next((order for order, name in _REPROJECT_ORDERS if name == variant), None)
        if order is None:
            return _skipped_row(case, library, f"unknown variant {variant!r}"), None, None
        if reproject_interp is None:
            return (
                _skipped_row(case, library, "optional dependency reproject is unavailable"),
                None,
                None,
            )
        row, data, weight = _reproject_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
            order=order,
            variant=str(variant),
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_reproject"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    if library == "dfreproject":
        order = next((order for order, name in _DFREPROJECT_ORDERS if name == variant), None)
        if order is None:
            return _skipped_row(case, library, f"unknown variant {variant!r}"), None, None
        row, data, weight = _dfreproject_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
            order=order,
            variant=str(variant),
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_dfreproject"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    if library == "drizzle":
        if variant not in _DRIZZLE_KERNELS:
            return _skipped_row(case, library, f"unknown variant {variant!r}"), None, None
        row, data, weight = _drizzle_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
            kernel=str(variant),
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_drizzle"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    if library == "drizzlepac":
        if drizzlepac is None:
            return (
                _skipped_row(
                    case,
                    library,
                    f"optional dependency drizzlepac is unavailable: {DRIZZLEPAC_IMPORT_ERROR}",
                ),
                None,
                None,
            )
        return (
            _skipped_row(
                case,
                library,
                "AstroDrizzle/DrizzlePac backend wiring not yet implemented",
            ),
            None,
            None,
        )
    if library == "montage":
        row, data, weight = _montage_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_montage"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    if library == "swarp":
        if variant is None:
            return _skipped_row(case, library, "missing resampling variant"), None, None
        row, data, weight = _swarp_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
            resampling_type=str(variant).upper(),
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_swarp"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    if library == "casutools":
        row, data, weight = _casutools_row(
            case,
            observations,
            target_header=target_header,
            runs=runs,
            torchsky_data=torchsky_data,
            torchsky_weight=torchsky_weight,
        )
        if row.get("status") == "OK":
            row["speed_ratio_torchsky_vs_casutools"] = row["time_ms"] / torchsky_row["time_ms"]
        return row, data, weight
    return _skipped_row(case, library, "unknown comparator"), None, None


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=256)
    parser.add_argument("--n-obs", type=int, default=3)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--cases",
        default="upstream_drizzle_crpix_translation,synthetic_shifted_wcs,synthetic_tpv_mef_low_overlap,real_astropy_sip_fixture,real_astropy_sip_shifted_wcs,real_astropy_sip_mef,real_astropy_sip_mef_shifted_wcs",
        help="Comma-separated case list",
    )
    parser.add_argument(
        "--torchsky-kernels",
        default=",".join(_TORCHSKY_KERNELS),
        help="Comma-separated torchsky resampling kernels",
    )
    parser.add_argument(
        "--reproject-kernels",
        default="nearest,bilinear,bicubic",
        help="Comma-separated reproject/dfreproject kernels",
    )
    parser.add_argument(
        "--drizzle-kernels",
        default="point,square",
        help="Comma-separated drizzle kernels",
    )
    parser.add_argument(
        "--swarp-kernels",
        default="nearest,bilinear,lanczos3",
        help="Comma-separated SWarp resampling kernels",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_map_compat_") as tmp:
        tmpdir = Path(tmp)
        torchsky_kernels = [part.strip() for part in args.torchsky_kernels.split(",") if part.strip()]
        reproject_kernels = [part.strip() for part in args.reproject_kernels.split(",") if part.strip()]
        drizzle_kernels = [part.strip() for part in args.drizzle_kernels.split(",") if part.strip()]
        swarp_kernels = [part.strip() for part in args.swarp_kernels.split(",") if part.strip()]
        all_cases = [
            (
                "upstream_drizzle_crpix_translation",
                _prepare_upstream_drizzle_translation_case(tmpdir, size=args.size),
            ),
            ("synthetic_shifted_wcs", _prepare_synthetic_shifted_case(tmpdir, size=args.size, n_obs=args.n_obs)),
            ("synthetic_tpv_mef_low_overlap", _prepare_synthetic_tpv_mef_case(tmpdir, size=args.size)),
            ("real_astropy_sip_fixture", _prepare_real_case(tmpdir)),
            ("real_astropy_sip_shifted_wcs", _prepare_real_shifted_sip_case(tmpdir)),
            ("real_astropy_sip_mef", _prepare_real_sip_mef_case(tmpdir, shifted=False)),
            ("real_astropy_sip_mef_shifted_wcs", _prepare_real_sip_mef_case(tmpdir, shifted=True)),
        ]
        case_filter = {
            part.strip() for part in args.cases.split(",") if part.strip()
        }
        cases = [item for item in all_cases if item[0] in case_filter]

        for case_name, observations in cases:
            target = plan_image_plane_target(observations, reference_index=0)
            torchsky_outputs: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = []
            for kernel in torchsky_kernels:
                row, data, weight = _torchsky_row(
                    case_name,
                    observations,
                    target_header=target.header,
                    runs=args.runs,
                    resampling_kernel=kernel,
                )
                rows.append(row)
                torchsky_outputs.append((row, data, weight))

            baseline_row, baseline_data, baseline_weight = next(
                (item for item in torchsky_outputs if item[0].get("variant") == "bilinear"),
                torchsky_outputs[0],
            )
            ok_outputs: list[tuple[dict[str, Any], torch.Tensor, torch.Tensor]] = list(torchsky_outputs)

            comparator_variants: dict[str, list[str | None]] = {
                "reproject_interp": reproject_kernels,
                "dfreproject": reproject_kernels,
                "drizzle": drizzle_kernels,
                "drizzlepac": [None],
                "montage": [None],
                "swarp": swarp_kernels,
                "casutools": [None],
            }
            for library in _COMPARATORS:
                for variant in comparator_variants[library]:
                    row, data, weight = _comparator_row(
                        library,
                        case_name,
                        observations,
                        target_header=target.header,
                        runs=max(2, args.runs),
                        torchsky_row=baseline_row,
                        torchsky_data=baseline_data,
                        torchsky_weight=baseline_weight,
                        variant=variant,
                    )
                    if variant is not None and "variant" not in row:
                        row["variant"] = variant
                    rows.append(row)
                    if row.get("status") == "OK" and data is not None and weight is not None:
                        ok_outputs.append((row, data, weight))

            for idx, (left_row, left_data, left_weight) in enumerate(ok_outputs):
                for right_row, right_data, right_weight in ok_outputs[idx + 1 :]:
                    rows.append(
                        _pairwise_backend_row(
                            case_name,
                            left_row,
                            left_data,
                            left_weight,
                            right_row,
                            right_data,
                            right_weight,
                        )
                    )

    markdown_path = args.markdown_out
    if markdown_path is None and args.json_out is not None:
        markdown_path = args.json_out.with_suffix(".md")
    markdown = _write_compat_artifacts(
        rows,
        json_out=args.json_out,
        markdown_out=markdown_path,
    )
    print(markdown)
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    if markdown_path is not None:
        print(f"Markdown: {markdown_path}")
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errors = _check_compat_regressions(rows, baseline_rows)
        if errors:
            raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
