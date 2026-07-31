#!/usr/bin/env python3
"""A1 NeRF: all-valid epoch sampling, 400/400 batches, PSF, and tracking.

The baseline manifest supplies only the completed run2 exposure selection,
frozen WCS, bandpass quadrature, normalization, loss, and optimizer settings.
This launcher forces A1 NeRF, all valid native pixels, epoch sampling without
replacement, 400/400 SW/LW samples, and a wavelength-dependent PSF bank.

During training it can save one diagnostic snapshot per completed global epoch.
Each snapshot contains native observation residuals, intrinsic band fields,
chromatic-PSF-observed band fields, and both A1-minus-UPDC residual families.
After the final training step it writes one final snapshot and, when enabled,
runs the existing source detector band by band on the final PSF-observed fields.
The detector never consumes the intrinsic fields.

F090W UPDC is used as the internal high-resolution training canvas by default.
The final six fields are not forced onto that grid: every band is rendered on
its own UPDC FITS template, preserving the SW/LW output-grid difference.
"""

from __future__ import annotations

import argparse
import copy
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
import json
import math
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any, Sequence

import numpy as np
import torch
from astropy.io import fits
from astropy.table import Table
from astropy.wcs import WCS as AstropyWCS
from astropy.wcs.utils import proj_plane_pixel_scales
from scipy.ndimage import zoom
from scipy.signal import oaconvolve

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
for path in (ROOT, SRC):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import scripts.smacs0723_wcs_aware_literature_ablation as a1  # noqa: E402
from torchsky.map.wcs_aware_calibration import build_psf_calibration  # noqa: E402
import torchsky.map.wcs_aware_training as training_mod  # noqa: E402

BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")
DEFAULT_UPDC_DIR = Path("/arc/projects/ots/jwst_smacs0723/data")
DEFAULT_UPDC_PATTERN = "{band}b_UPDC-v2_iter10_i2d.fits.fz"
DEFAULT_STPSF_DATA_DIR = Path("/arc/projects/ots/jwst_smacs0723/stpsf-data")
NIRCAM_SHORT_MIN_AA = 6000.0
NIRCAM_CHANNEL_SPLIT_AA = 23500.0
NIRCAM_LONG_MAX_AA = 53000.0
NIRCAM_BOUNDARY_EPSILON_AA = 1.0


def jsonable(value: Any) -> Any:
    if is_dataclass(value):
        return jsonable(asdict(value))
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return value


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2), encoding="utf-8")


def now_utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def parse_band_float_map(text: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for token in str(text).split(","):
        token = token.strip()
        if not token:
            continue
        if "=" not in token:
            raise ValueError(f"expected BAND=value, got {token!r}")
        band, value = token.split("=", 1)
        result[band.strip().upper()] = float(value)
    return result


def parse_band_path_map(values: Sequence[str]) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for token in values:
        if "=" not in token:
            raise ValueError(f"expected BAND=FITS, got {token!r}")
        band, raw = token.split("=", 1)
        band = band.strip().upper()
        if band not in BANDS:
            raise ValueError(f"unsupported band {band!r}")
        result[band] = Path(raw).expanduser().resolve()
    return result


def launcher_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--batch-size", type=int, nargs="+", default=[400, 400])
    parser.add_argument(
        "--psf-mode", choices=("none", "gaussian", "npy", "stpsf"), default="stpsf"
    )
    parser.add_argument("--psf-size", type=int, default=15)
    parser.add_argument("--psf-fwhm-pix", default="")
    parser.add_argument("--default-psf-fwhm-pix", type=float, default=None)
    parser.add_argument("--psf-kernel-dir", type=Path, default=None)
    parser.add_argument("--psf-cache-dir", type=Path, default=None)
    parser.add_argument("--stpsf-extension", default="DET_DIST")
    parser.add_argument(
        "--stpsf-data-dir",
        type=Path,
        default=DEFAULT_STPSF_DATA_DIR,
        help="STPSF reference-data directory exported as STPSF_PATH.",
    )
    parser.add_argument("--allow-static-stpsf-fallback", action="store_true")
    parser.add_argument("--psf-offset-chunk", type=int, default=32)
    parser.add_argument("--no-checkpoint-psf", action="store_true")

    parser.add_argument("--updc-dir", type=Path, default=DEFAULT_UPDC_DIR)
    parser.add_argument("--updc-pattern", default=DEFAULT_UPDC_PATTERN)
    parser.add_argument(
        "--updc-reference", action="append", default=[], metavar="BAND=FITS"
    )
    parser.add_argument(
        "--updc-zero-is-invalid", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--use-baseline-canvas", action="store_true")
    parser.add_argument("--also-render-union", action="store_true")
    parser.add_argument(
        "--diagnostics", action=argparse.BooleanOptionalAction, default=True
    )
    parser.add_argument("--diagnostic-tile-pixels", type=int, default=65536)
    parser.add_argument("--native-residual-samples-per-exposure", type=int, default=512)
    parser.add_argument(
        "--save-median-matched-residual",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    parser.add_argument(
        "--save-every-global-epoch",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Save native residuals plus intrinsic and PSF-observed band fields "
            "whenever every exposure has completed another full pixel pass."
        ),
    )
    parser.add_argument(
        "--epoch-save-every",
        type=int,
        default=1,
        help="Save every N completed global epochs.",
    )
    parser.add_argument(
        "--max-epoch-snapshots",
        type=int,
        default=0,
        help="Maximum epoch snapshots; 0 means unlimited.",
    )
    parser.add_argument(
        "--run-detector",
        action=argparse.BooleanOptionalAction,
        default=False,
        help=(
            "Run compare_wcs_aware_source_catalogs.py once after the final "
            "training snapshot. Detector inputs are the final PSF-observed "
            "band fields, never the intrinsic fields."
        ),
    )
    parser.add_argument(
        "--detector-script",
        type=Path,
        default=ROOT / "scripts" / "compare_wcs_aware_source_catalogs.py",
    )
    parser.add_argument("--detector-threshold-sigma", type=float, default=6.0)
    parser.add_argument("--detector-npixels", type=int, default=5)
    parser.add_argument(
        "--detector-connectivity",
        type=int,
        choices=(4, 8),
        default=8,
    )
    parser.add_argument("--detector-background-box-size", type=int, default=60)
    parser.add_argument("--detector-background-filter-size", type=int, default=5)
    parser.add_argument("--detector-sigma-clip", type=float, default=3.0)
    parser.add_argument("--detector-sigma-clip-maxiters", type=int, default=10)
    parser.add_argument(
        "--detector-deblend",
        action=argparse.BooleanOptionalAction,
        default=False,
    )
    parser.add_argument("--detector-deblend-nlevels", type=int, default=32)
    parser.add_argument("--detector-deblend-contrast", type=float, default=0.001)
    parser.add_argument("--detector-match-radius-arcsec", type=float, default=0.05)
    parser.add_argument(
        "--detector-match-radius-grid-arcsec",
        default="0.05,0.075,0.10,0.15",
    )
    parser.add_argument(
        "--detector-snr-bins",
        default="6,8,10,20,inf",
    )
    return parser


def has_option(arguments: Sequence[str], name: str) -> bool:
    return any(arg == name or arg.startswith(name + "=") for arg in arguments)


def option_value(arguments: Sequence[str], name: str) -> str | None:
    for i, arg in enumerate(arguments):
        if arg.startswith(name + "="):
            return arg.split("=", 1)[1]
        if arg == name and i + 1 < len(arguments):
            return arguments[i + 1]
    return None


def sanitize_nircam_channel_quadrature(
    manifest: dict[str, Any],
    wavelength: torch.Tensor,
    weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, list[dict[str, Any]]]:
    """Remove quadrature weight outside the detector's NIRCam channel.

    Baseline SVO curves are filter profiles and are not detector-specific. Tiny
    transmission tails can therefore place quadrature samples outside the
    wavelength range accepted by the detector selected in the exposure header.
    Invalid samples retain a boundary wavelength only so the rectangular [E,L]
    PSF bank can still be built; their integration weights are set to zero.
    """
    exposures = sorted(manifest["exposures"], key=lambda row: int(row["index"]))
    if len(exposures) != wavelength.shape[0]:
        raise ValueError("exposure metadata and bandpass rows have different lengths")

    safe_wave = wavelength.detach().clone()
    safe_weight = weight.detach().clone()
    records: list[dict[str, Any]] = []

    for index, exposure in enumerate(exposures):
        detector = str(exposure.get("detector", "")).strip().upper().replace(" ", "")
        is_long = detector.endswith("5") or detector.endswith("LONG")
        if is_long:
            lower = NIRCAM_CHANNEL_SPLIT_AA + NIRCAM_BOUNDARY_EPSILON_AA
            upper = NIRCAM_LONG_MAX_AA - NIRCAM_BOUNDARY_EPSILON_AA
            channel = "long"
        else:
            lower = NIRCAM_SHORT_MIN_AA + NIRCAM_BOUNDARY_EPSILON_AA
            upper = NIRCAM_CHANNEL_SPLIT_AA - NIRCAM_BOUNDARY_EPSILON_AA
            channel = "short"

        row_wave = safe_wave[index]
        row_weight = safe_weight[index]
        invalid = (row_wave < lower) | (row_wave > upper)
        removed_weight = float(row_weight[invalid].sum().item())
        invalid_count = int(invalid.sum().item())

        if invalid_count:
            row_weight[invalid] = 0.0
            row_wave.clamp_(min=lower, max=upper)
            remaining = row_weight.sum()
            if not bool(torch.isfinite(remaining)) or float(remaining.item()) <= 0.0:
                raise ValueError(
                    f"all bandpass weight lies outside the NIRCam {channel} channel "
                    f"for exposure {index} ({exposure.get('band')}, {detector})"
                )
            row_weight /= remaining
            records.append(
                {
                    "exposure_index": index,
                    "band": str(exposure.get("band", "")),
                    "detector": detector,
                    "channel": channel,
                    "valid_wavelength_angstrom": [lower, upper],
                    "invalid_sample_count": invalid_count,
                    "removed_pre_normalization_weight": removed_weight,
                    "requested_wavelength_angstrom": wavelength[index].tolist(),
                    "sanitized_wavelength_angstrom": row_wave.tolist(),
                    "sanitized_weight": row_weight.tolist(),
                }
            )

    return safe_wave, safe_weight, records


def normalize_batch(values: Sequence[int]) -> int | list[int]:
    parsed = [int(v) for v in values]
    if len(parsed) not in (1, 2) or min(parsed) <= 0:
        raise ValueError("--batch-size accepts one total or two positive SW/LW values")
    return parsed[0] if len(parsed) == 1 else parsed


def batch_label(value: int | Sequence[int]) -> str:
    return f"total{value}" if isinstance(value, int) else "_".join(map(str, value))


def resolve_references(args: argparse.Namespace) -> dict[str, Path]:
    overrides = parse_band_path_map(args.updc_reference)
    root = args.updc_dir.expanduser().resolve()
    return {
        band: overrides.get(band, root / args.updc_pattern.format(band=band)).resolve()
        for band in BANDS
    }


def read_reference(
    path: Path, *, zero_is_invalid: bool
) -> tuple[np.ndarray, fits.Header, np.ndarray, dict[str, Any]]:
    with fits.open(path, memmap=False) as hdul:
        image_hdu = None
        for hdu in hdul:
            if hdu.data is None or np.ndim(hdu.data) != 2:
                continue
            if str(hdu.header.get("EXTNAME", "")).upper() == "SCI":
                image_hdu = hdu
                break
            if image_hdu is None:
                image_hdu = hdu
        if image_hdu is None:
            raise ValueError(f"no 2D image HDU in {path}")
        data = np.asarray(image_hdu.data, dtype=np.float32)
        header = image_hdu.header.copy()
        valid = np.isfinite(data)
        coverage_name = None
        for name in ("WHT", "WEIGHT", "COVERAGE", "NEXP"):
            if name in hdul and hdul[name].data is not None:
                coverage = np.asarray(hdul[name].data)
                if coverage.shape == data.shape:
                    valid &= np.isfinite(coverage) & (coverage > 0)
                    coverage_name = name
                    break
        if "UPDCMASK" in hdul and hdul["UPDCMASK"].data is not None:
            mask = np.asarray(hdul["UPDCMASK"].data)
            if mask.shape == data.shape:
                valid &= np.isfinite(mask) & (mask == 0)
        if zero_is_invalid:
            valid &= data != 0.0
    return data, header, valid, {
        "path": str(path),
        "shape": list(data.shape),
        "bunit": header.get("BUNIT"),
        "coverage_extension": coverage_name,
        "zero_is_invalid": zero_is_invalid,
        "valid_fraction": float(valid.mean()),
    }


def write_image(
    path: Path,
    data: np.ndarray,
    header: fits.Header,
    *,
    band: str,
    stage: str,
    reference: Path | None,
) -> None:
    out_header = header.copy()
    out_header["BAND"] = band
    out_header["MODEL"] = "A1_NERF"
    out_header["STAGE"] = stage
    out_header["HISTORY"] = "Generated by A1 all-valid 400/400 PSF launcher"
    if reference is not None:
        out_header["HISTORY"] = f"Reference: {reference}"
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.PrimaryHDU(np.asarray(data), header=out_header).writeto(path, overwrite=False)


def bandpass_for_band(
    pools: Sequence[Any], wavelength: torch.Tensor, weight: torch.Tensor, band: str
) -> tuple[int, torch.Tensor, torch.Tensor, dict[str, Any]]:
    indices = [i for i, pool in enumerate(pools) if str(pool.band).upper() == band]
    if not indices:
        raise ValueError(f"no exposure for {band}")
    representative = indices[0]
    wave = wavelength[representative].detach().cpu().float()
    band_weight = weight[representative].detach().cpu().float()
    band_weight /= band_weight.sum().clamp_min(1.0e-20)
    max_dw = max(float((wavelength[i].cpu() - wave).abs().max()) for i in indices)
    max_dwt = max(float((weight[i].cpu() - band_weight).abs().max()) for i in indices)
    metadata = {
        "representative_exposure": representative,
        "exposure_count": len(indices),
        "quadrature_wavelength": wave.tolist(),
        "quadrature_weight": band_weight.tolist(),
        "max_same_band_wavelength_difference": max_dw,
        "max_same_band_weight_difference": max_dwt,
    }
    return representative, wave, band_weight, metadata


def render_intrinsic_field(
    *,
    model: torch.nn.Module,
    canvas: Any,
    template_header: fits.Header,
    representative: int,
    wavelength: torch.Tensor,
    weight: torch.Tensor,
    intensity_scale: float,
    device: torch.device,
    tile_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    height = int(template_header["NAXIS2"])
    width = int(template_header["NAXIS1"])
    template_wcs = AstropyWCS(template_header).celestial
    union_wcs = AstropyWCS(canvas.header).celestial
    field = np.full((height, width), np.nan, dtype=np.float32)
    union_valid = np.zeros((height, width), dtype=bool)
    field_flat = field.reshape(-1)
    valid_flat = union_valid.reshape(-1)
    wave_values = wavelength.numpy()
    weight_values = weight.numpy()

    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, height * width, tile_pixels):
            end = min(start + tile_pixels, height * width)
            linear = np.arange(start, end, dtype=np.int64)
            y = (linear // width).astype(np.float64)
            x = (linear % width).astype(np.float64)
            ra, dec = template_wcs.pixel_to_world_values(x, y)
            ux, uy = union_wcs.world_to_pixel_values(ra, dec)
            ux, uy = np.asarray(ux), np.asarray(uy)
            valid = (
                np.isfinite(ux)
                & np.isfinite(uy)
                & (ux >= 0)
                & (ux <= canvas.shape[1] - 1)
                & (uy >= 0)
                & (uy <= canvas.shape[0] - 1)
            )
            tile = np.full(end - start, np.nan, dtype=np.float32)
            if valid.any():
                xy = torch.from_numpy(
                    np.stack((ux[valid], uy[valid]), axis=-1).astype(np.float32)
                ).to(device)
                exposure_id = torch.full(
                    (xy.shape[0],), representative, dtype=torch.long, device=device
                )
                raw = model.position_encoder(xy)
                latent = model.trunk(model.feature_alignment(raw, exposure_id))
                inside = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
                prediction = torch.zeros(xy.shape[0], dtype=xy.dtype, device=device)
                for wave_value, weight_value in zip(
                    wave_values, weight_values, strict=True
                ):
                    wave_tensor = torch.full_like(prediction, float(wave_value))
                    prediction += float(weight_value) * model.decode(
                        latent, wave_tensor, inside
                    )
                tile[valid] = (
                    prediction.detach().float().cpu().numpy() * intensity_scale
                )
            field_flat[start:end] = tile
            valid_flat[start:end] = valid
    return field, union_valid


def render_monochromatic_field(
    *,
    model: torch.nn.Module,
    canvas: Any,
    template_header: fits.Header,
    representative: int,
    wavelength: float,
    intensity_scale: float,
    device: torch.device,
    tile_pixels: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Render one wavelength slice of the intrinsic A1 scene on a FITS grid."""
    height = int(template_header["NAXIS2"])
    width = int(template_header["NAXIS1"])
    template_wcs = AstropyWCS(template_header).celestial
    union_wcs = AstropyWCS(canvas.header).celestial
    field = np.full((height, width), np.nan, dtype=np.float32)
    union_valid = np.zeros((height, width), dtype=bool)
    field_flat = field.reshape(-1)
    valid_flat = union_valid.reshape(-1)

    model.to(device).eval()
    with torch.inference_mode():
        for start in range(0, height * width, tile_pixels):
            end = min(start + tile_pixels, height * width)
            linear = np.arange(start, end, dtype=np.int64)
            y = (linear // width).astype(np.float64)
            x = (linear % width).astype(np.float64)
            ra, dec = template_wcs.pixel_to_world_values(x, y)
            ux, uy = union_wcs.world_to_pixel_values(ra, dec)
            ux, uy = np.asarray(ux), np.asarray(uy)
            valid = (
                np.isfinite(ux)
                & np.isfinite(uy)
                & (ux >= 0)
                & (ux <= canvas.shape[1] - 1)
                & (uy >= 0)
                & (uy <= canvas.shape[0] - 1)
            )
            tile = np.full(end - start, np.nan, dtype=np.float32)
            if valid.any():
                xy = torch.from_numpy(
                    np.stack((ux[valid], uy[valid]), axis=-1).astype(np.float32)
                ).to(device)
                exposure_id = torch.full(
                    (xy.shape[0],), representative, dtype=torch.long, device=device
                )
                raw = model.position_encoder(xy)
                latent = model.trunk(model.feature_alignment(raw, exposure_id))
                inside = torch.ones(xy.shape[0], dtype=torch.bool, device=device)
                wave_tensor = torch.full(
                    (xy.shape[0],),
                    float(wavelength),
                    dtype=xy.dtype,
                    device=device,
                )
                prediction = model.decode(latent, wave_tensor, inside)
                tile[valid] = (
                    prediction.detach().float().cpu().numpy() * intensity_scale
                )
            field_flat[start:end] = tile
            valid_flat[start:end] = valid
    return field, union_valid


def pixel_scale_arcsec(header: fits.Header | dict[str, Any]) -> float:
    scale = float(
        np.sqrt(
            np.prod(
                proj_plane_pixel_scales(
                    AstropyWCS(header).celestial
                )[:2]
            )
        )
        * 3600.0
    )
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("could not infer a positive pixel scale")
    return scale


def center_fit_square(array: np.ndarray, size: int) -> np.ndarray:
    """Centrally crop or pad a square kernel to an exact odd size."""
    if size <= 0 or size % 2 != 1:
        raise ValueError("target kernel size must be a positive odd integer")
    source = np.asarray(array, dtype=np.float32)
    output = np.zeros((size, size), dtype=np.float32)
    copy_y = min(source.shape[0], size)
    copy_x = min(source.shape[1], size)
    source_y = max((source.shape[0] - copy_y) // 2, 0)
    source_x = max((source.shape[1] - copy_x) // 2, 0)
    target_y = max((size - copy_y) // 2, 0)
    target_x = max((size - copy_x) // 2, 0)
    output[
        target_y : target_y + copy_y,
        target_x : target_x + copy_x,
    ] = source[
        source_y : source_y + copy_y,
        source_x : source_x + copy_x,
    ]
    return output


def resample_psf_kernel(
    kernel: np.ndarray,
    *,
    native_pixel_scale_arcsec: float,
    target_pixel_scale_arcsec: float,
) -> np.ndarray:
    """Resample a detector-pixel PSF to the final FITS pixel scale."""
    source = np.asarray(kernel, dtype=np.float32)
    ratio = float(native_pixel_scale_arcsec) / float(target_pixel_scale_arcsec)
    half_size = max(1, int(round(((source.shape[0] - 1) / 2.0) * ratio)))
    target_size = 2 * half_size + 1
    resized = zoom(
        source,
        zoom=target_size / source.shape[0],
        order=1,
        prefilter=False,
    )
    resized = center_fit_square(np.asarray(resized, dtype=np.float32), target_size)
    resized = np.clip(resized, 0.0, None)
    total = float(resized.sum())
    if not np.isfinite(total) or total <= 0:
        raise ValueError("resampled PSF has non-positive mass")
    return resized / total


def band_psf_kernels(
    *,
    pools: Sequence[Any],
    psf_bank: Any,
    band: str,
    target_header: fits.Header,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Average same-band exposure PSFs separately at every quadrature wavelength."""
    indices = [i for i, pool in enumerate(pools) if str(pool.band).upper() == band]
    if not indices:
        raise ValueError(f"no exposure for {band}")
    if psf_bank is None:
        raise ValueError("PSF bank is unavailable")

    kernels = psf_bank.kernel.detach().cpu().float()[indices]
    target_scale = pixel_scale_arcsec(target_header)
    native_scales = [pixel_scale_arcsec(pools[i].header) for i in indices]
    native_scale = float(np.mean(native_scales))

    output: list[np.ndarray] = []
    for sample_index in range(kernels.shape[1]):
        mean_kernel = kernels[:, sample_index].mean(dim=0).numpy()
        mean_kernel = np.clip(mean_kernel, 0.0, None)
        mean_kernel /= max(float(mean_kernel.sum()), np.finfo(np.float32).tiny)
        output.append(
            resample_psf_kernel(
                mean_kernel,
                native_pixel_scale_arcsec=native_scale,
                target_pixel_scale_arcsec=target_scale,
            )
        )

    metadata = {
        "band": band,
        "exposure_indices": indices,
        "exposure_count": len(indices),
        "quadrature_samples": len(output),
        "native_kernel_size": int(kernels.shape[-1]),
        "target_kernel_size": int(output[0].shape[-1]),
        "mean_native_pixel_scale_arcsec": native_scale,
        "target_pixel_scale_arcsec": target_scale,
        "definition": (
            "For each quadrature wavelength, average the normalized detector-plane "
            "PSFs over same-band exposures, then resample that kernel to the exact "
            "UPDC output pixel scale."
        ),
    }
    return output, metadata


def convolve_field_with_psf(
    field: np.ndarray,
    kernel: np.ndarray,
    *,
    minimum_support: float = 0.999,
) -> tuple[np.ndarray, np.ndarray]:
    """PSF-convolve a field while rejecting target-grid boundary truncation."""
    valid = np.isfinite(field)
    numerator = oaconvolve(
        np.where(valid, field, 0.0).astype(np.float32),
        kernel.astype(np.float32),
        mode="same",
    ).astype(np.float32)
    support = oaconvolve(
        valid.astype(np.float32),
        kernel.astype(np.float32),
        mode="same",
    ).astype(np.float32)
    output_valid = support >= float(minimum_support)
    output = np.full(field.shape, np.nan, dtype=np.float32)
    output[output_valid] = numerator[output_valid] / np.maximum(
        support[output_valid], np.finfo(np.float32).tiny
    )
    return output, output_valid


def render_intrinsic_and_psf_band_fields(
    *,
    model: torch.nn.Module,
    canvas: Any,
    template_header: fits.Header,
    representative: int,
    wavelength: torch.Tensor,
    weight: torch.Tensor,
    psf_kernels: Sequence[np.ndarray],
    intensity_scale: float,
    device: torch.device,
    tile_pixels: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Render intrinsic and chromatic-PSF band fields.

    The observed field follows the training operator order:

        sum_j weight_j * (PSF_j * scene_j)

    rather than convolving the already band-integrated image with one mean PSF.
    """
    wave_values = wavelength.detach().cpu().numpy()
    weight_values = weight.detach().cpu().numpy()
    if len(psf_kernels) != len(wave_values):
        raise ValueError("PSF kernel count does not match quadrature samples")

    shape = (
        int(template_header["NAXIS2"]),
        int(template_header["NAXIS1"]),
    )
    intrinsic = np.zeros(shape, dtype=np.float32)
    observed = np.zeros(shape, dtype=np.float32)
    intrinsic_valid: np.ndarray | None = None
    observed_valid: np.ndarray | None = None

    for wave_value, weight_value, kernel in zip(
        wave_values,
        weight_values,
        psf_kernels,
        strict=True,
    ):
        monochromatic, mono_valid = render_monochromatic_field(
            model=model,
            canvas=canvas,
            template_header=template_header,
            representative=representative,
            wavelength=float(wave_value),
            intensity_scale=intensity_scale,
            device=device,
            tile_pixels=tile_pixels,
        )
        convolved, conv_valid = convolve_field_with_psf(
            monochromatic,
            kernel,
        )
        intrinsic += float(weight_value) * np.nan_to_num(
            monochromatic,
            nan=0.0,
        )
        observed += float(weight_value) * np.nan_to_num(
            convolved,
            nan=0.0,
        )
        intrinsic_valid = (
            mono_valid
            if intrinsic_valid is None
            else intrinsic_valid & mono_valid
        )
        observed_valid = (
            conv_valid
            if observed_valid is None
            else observed_valid & conv_valid
        )
        del monochromatic, convolved

    if intrinsic_valid is None or observed_valid is None:
        raise ValueError("empty bandpass quadrature")
    intrinsic[~intrinsic_valid] = np.nan
    observed[~observed_valid] = np.nan
    return intrinsic, intrinsic_valid, observed, observed_valid


def write_psf_cube(
    path: Path,
    kernels: Sequence[np.ndarray],
    wavelength: torch.Tensor,
    *,
    band: str,
    metadata: dict[str, Any],
) -> None:
    cube = np.stack([np.asarray(kernel, dtype=np.float32) for kernel in kernels])
    header = fits.Header()
    header["BAND"] = band
    header["STAGE"] = "PSF_BY_WAVELENGTH"
    header["NEXP"] = int(metadata["exposure_count"])
    header["NATIVEPX"] = float(metadata["mean_native_pixel_scale_arcsec"])
    header["TARGETPX"] = float(metadata["target_pixel_scale_arcsec"])
    primary = fits.PrimaryHDU(cube, header=header)
    wave_hdu = fits.ImageHDU(
        np.asarray(wavelength.detach().cpu(), dtype=np.float32),
        name="WAVELENGTH",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fits.HDUList([primary, wave_hdu]).writeto(path, overwrite=False)


def comparison_metrics(
    field: np.ndarray,
    reference: np.ndarray,
    valid: np.ndarray,
    *,
    chunk_pixels: int = 2_000_000,
    quantile_sample: int = 500_000,
) -> dict[str, Any]:
    field_flat = field.reshape(-1)
    ref_flat = reference.reshape(-1)
    valid_flat = valid.reshape(-1)
    chunks = max(1, math.ceil(field_flat.size / chunk_pixels))
    sample_each = max(1, quantile_sample // chunks)
    n = 0
    sum_f = sum_r = sum_f2 = sum_r2 = sum_fr = 0.0
    sum_d = sum_d2 = sum_abs_d = 0.0
    samples: list[np.ndarray] = []
    for start in range(0, field_flat.size, chunk_pixels):
        end = min(start + chunk_pixels, field_flat.size)
        choose = valid_flat[start:end]
        if not choose.any():
            continue
        fv = field_flat[start:end][choose].astype(np.float64, copy=False)
        rv = ref_flat[start:end][choose].astype(np.float64, copy=False)
        dv = fv - rv
        n += fv.size
        sum_f += float(fv.sum())
        sum_r += float(rv.sum())
        sum_f2 += float(np.dot(fv, fv))
        sum_r2 += float(np.dot(rv, rv))
        sum_fr += float(np.dot(fv, rv))
        sum_d += float(dv.sum())
        sum_d2 += float(np.dot(dv, dv))
        sum_abs_d += float(np.abs(dv).sum())
        take = min(sample_each, dv.size)
        samples.append(dv[np.linspace(0, dv.size - 1, take, dtype=np.int64)])
    if n == 0:
        raise ValueError("no common valid pixels")
    sample = np.concatenate(samples)
    mean_f, mean_r, mean_d = sum_f / n, sum_r / n, sum_d / n
    var_f = max(sum_f2 / n - mean_f**2, 0.0)
    var_r = max(sum_r2 / n - mean_r**2, 0.0)
    covariance = sum_fr / n - mean_f * mean_r
    corr = covariance / math.sqrt(var_f * var_r) if var_f > 0 and var_r > 0 else np.nan
    median = float(np.median(sample))
    return {
        "valid_pixels": int(n),
        "valid_fraction": float(n / field_flat.size),
        "field_mean": mean_f,
        "updc_mean": mean_r,
        "residual_mean": mean_d,
        "residual_median_approx": median,
        "residual_std": math.sqrt(max(sum_d2 / n - mean_d**2, 0.0)),
        "residual_mad_sigma_approx": float(1.4826 * np.median(np.abs(sample - median))),
        "rmse": math.sqrt(sum_d2 / n),
        "mae": sum_abs_d / n,
        "pearson_correlation": float(corr),
        "residual_quantiles_approx": {
            str(q): float(np.quantile(sample, q))
            for q in (0.01, 0.05, 0.16, 0.50, 0.84, 0.95, 0.99)
        },
        "quantile_sample_size": int(sample.size),
    }


def native_observation_residuals(
    *,
    model: torch.nn.Module,
    pools: Sequence[Any],
    wavelength: torch.Tensor,
    weight: torch.Tensor,
    psf_bank: Any,
    intensity_scale: float,
    device: torch.device,
    samples_per_exposure: int,
    seed: int,
    psf_offset_chunk: int,
    output_dir: Path,
) -> dict[str, Any]:
    if samples_per_exposure <= 0:
        return {"enabled": False}
    rng = np.random.default_rng(seed + 909)
    model.to(device).eval()
    psf = psf_bank.to(device) if psf_bank is not None else None
    wave = wavelength.to(device)
    band_weight = weight.to(device)
    rows: list[dict[str, Any]] = []
    with torch.inference_mode():
        for exposure_index, pool in enumerate(pools):
            count = min(samples_per_exposure, pool.size)
            chosen = torch.from_numpy(
                rng.choice(pool.size, size=count, replace=False).astype(np.int64)
            )
            uv = pool.uv[chosen].to(device)
            exposure_id = torch.full(
                (count,), exposure_index, dtype=torch.long, device=device
            )
            prediction, valid = model.predict_observation(
                uv,
                exposure_id,
                wave,
                band_weight,
                psf=psf,
                psf_offset_chunk=psf_offset_chunk,
                checkpoint_psf_chunks=False,
            )
            observed = pool.value[chosen].to(device)
            error = pool.err[chosen].to(device)
            valid &= torch.isfinite(prediction) & torch.isfinite(observed)
            valid &= torch.isfinite(error) & (error > 0)
            residual = (prediction - observed)[valid]
            z = residual / error[valid]
            residual_np = residual.detach().cpu().numpy() * intensity_scale
            z_np = z.detach().cpu().numpy()
            row = {
                "exposure_index": exposure_index,
                "path": str(pool.path),
                "band": str(pool.band),
                "detector": str(pool.detector),
                "sampled": count,
                "valid": int(valid.sum().item()),
                "valid_fraction": float(valid.float().mean().item()),
            }
            if residual_np.size:
                row.update(
                    {
                        "residual_median": float(np.median(residual_np)),
                        "rmse": float(np.sqrt(np.mean(residual_np**2))),
                        "mae": float(np.mean(np.abs(residual_np))),
                        "abs_z_q50": float(np.quantile(np.abs(z_np), 0.50)),
                        "abs_z_q68": float(np.quantile(np.abs(z_np), 0.68)),
                        "abs_z_q90": float(np.quantile(np.abs(z_np), 0.90)),
                        "abs_z_q95": float(np.quantile(np.abs(z_np), 0.95)),
                        "abs_z_q99": float(np.quantile(np.abs(z_np), 0.99)),
                    }
                )
            rows.append(row)
    output_dir.mkdir(parents=True, exist_ok=True)
    table_path = output_dir / "native_observation_residuals.ecsv"
    Table(rows=rows).write(table_path, format="ascii.ecsv", overwrite=False)
    summary = {
        "enabled": True,
        "samples_per_exposure": samples_per_exposure,
        "PSF_applied": psf_bank is not None,
        "bandpass_applied": True,
        "learned_background_applied": True,
        "residual_sign": "prediction_minus_native_observation",
        "table": str(table_path),
    }
    write_json(output_dir / "native_observation_residuals.json", summary)
    return summary


def diagnose_band(
    *,
    band: str,
    reference_path: Path,
    context: dict[str, Any],
    args: argparse.Namespace,
    output_root: Path,
) -> dict[str, Any]:
    band_dir = output_root / band
    band_dir.mkdir(parents=True, exist_ok=True)
    reference, header, reference_valid, reference_info = read_reference(
        reference_path,
        zero_is_invalid=args.updc_zero_is_invalid,
    )
    representative, wave, weight, bandpass_info = bandpass_for_band(
        context["pools"],
        context["wavelength"],
        context["weight"],
        band,
    )
    kernels, psf_info = band_psf_kernels(
        pools=context["pools"],
        psf_bank=context["psf_bank"],
        band=band,
        target_header=header,
    )
    intrinsic, intrinsic_valid, psf_field, psf_valid = (
        render_intrinsic_and_psf_band_fields(
            model=context["model"],
            canvas=context["canvas"],
            template_header=header,
            representative=representative,
            wavelength=wave,
            weight=weight,
            psf_kernels=kernels,
            intensity_scale=context["intensity_scale"],
            device=context["device"],
            tile_pixels=args.diagnostic_tile_pixels,
        )
    )

    intrinsic_common = (
        intrinsic_valid
        & reference_valid
        & np.isfinite(intrinsic)
    )
    psf_common = (
        psf_valid
        & reference_valid
        & np.isfinite(psf_field)
    )

    intrinsic_path = (
        band_dir / f"{band}_A1_intrinsic_bandpass_field.fits"
    )
    psf_path = (
        band_dir / f"{band}_A1_psf_observed_bandpass_field.fits"
    )
    psf_cube_path = (
        band_dir / f"{band}_A1_effective_psf_by_wavelength.fits"
    )
    intrinsic_mask_path = (
        band_dir / f"{band}_A1_intrinsic_UPDC_common_valid_mask.fits"
    )
    psf_mask_path = (
        band_dir / f"{band}_A1_psf_UPDC_common_valid_mask.fits"
    )
    intrinsic_residual_path = (
        band_dir / f"{band}_A1_intrinsic_minus_UPDC_residual.fits"
    )
    psf_residual_path = (
        band_dir / f"{band}_A1_psf_minus_UPDC_residual.fits"
    )

    write_image(
        intrinsic_path,
        intrinsic,
        header,
        band=band,
        stage="INTRINSIC_BANDPASS_FIELD",
        reference=reference_path,
    )
    write_image(
        psf_path,
        psf_field,
        header,
        band=band,
        stage="PSF_OBSERVED_BANDPASS_FIELD",
        reference=reference_path,
    )
    write_psf_cube(
        psf_cube_path,
        kernels,
        wave,
        band=band,
        metadata=psf_info,
    )
    write_image(
        intrinsic_mask_path,
        intrinsic_common.astype(np.uint8),
        header,
        band=band,
        stage="INTRINSIC_COMMON_VALID_MASK",
        reference=reference_path,
    )
    write_image(
        psf_mask_path,
        psf_common.astype(np.uint8),
        header,
        band=band,
        stage="PSF_COMMON_VALID_MASK",
        reference=reference_path,
    )

    intrinsic_metrics = comparison_metrics(
        intrinsic,
        reference,
        intrinsic_common,
    )
    psf_metrics = comparison_metrics(
        psf_field,
        reference,
        psf_common,
    )

    intrinsic_residual = intrinsic - reference
    intrinsic_residual[~intrinsic_common] = np.nan
    psf_residual = psf_field - reference
    psf_residual[~psf_common] = np.nan

    write_image(
        intrinsic_residual_path,
        intrinsic_residual.astype(np.float32),
        header,
        band=band,
        stage="A1_INTRINSIC_MINUS_UPDC",
        reference=reference_path,
    )
    write_image(
        psf_residual_path,
        psf_residual.astype(np.float32),
        header,
        band=band,
        stage="A1_PSF_MINUS_UPDC",
        reference=reference_path,
    )

    intrinsic_median_path = None
    psf_median_path = None
    if args.save_median_matched_residual:
        intrinsic_median = (
            intrinsic_residual
            - intrinsic_metrics["residual_median_approx"]
        )
        intrinsic_median_path = (
            band_dir
            / f"{band}_A1_intrinsic_minus_UPDC_median_matched.fits"
        )
        write_image(
            intrinsic_median_path,
            intrinsic_median.astype(np.float32),
            header,
            band=band,
            stage="A1_INTRINSIC_MINUS_UPDC_MEDIAN_MATCHED",
            reference=reference_path,
        )

        psf_median = (
            psf_residual
            - psf_metrics["residual_median_approx"]
        )
        psf_median_path = (
            band_dir
            / f"{band}_A1_psf_minus_UPDC_median_matched.fits"
        )
        write_image(
            psf_median_path,
            psf_median.astype(np.float32),
            header,
            band=band,
            stage="A1_PSF_MINUS_UPDC_MEDIAN_MATCHED",
            reference=reference_path,
        )

    record = {
        "band": band,
        "reference": reference_info,
        "pixel_scale_arcsec": pixel_scale_arcsec(header),
        "bandpass": bandpass_info,
        "PSF": psf_info,
        "products": {
            "intrinsic_bandpass_field": str(intrinsic_path),
            "psf_observed_bandpass_field": str(psf_path),
            "effective_psf_by_wavelength": str(psf_cube_path),
            "intrinsic_common_valid_mask": str(intrinsic_mask_path),
            "psf_common_valid_mask": str(psf_mask_path),
            "intrinsic_residual_A1_minus_UPDC": str(
                intrinsic_residual_path
            ),
            "psf_residual_A1_minus_UPDC": str(psf_residual_path),
            "intrinsic_median_matched_residual": (
                None
                if intrinsic_median_path is None
                else str(intrinsic_median_path)
            ),
            "psf_median_matched_residual": (
                None
                if psf_median_path is None
                else str(psf_median_path)
            ),
        },
        "metrics": {
            "intrinsic_vs_UPDC": intrinsic_metrics,
            "psf_observed_vs_UPDC": psf_metrics,
        },
        "stage_definition": {
            "intrinsic_field": (
                "sum_j w_j S_theta(x,y,lambda_j); no final-grid PSF"
            ),
            "psf_observed_field": (
                "sum_j w_j [K_bar_j * S_theta(x,y,lambda_j)], "
                "where K_bar_j is the same-band exposure-mean PSF "
                "at quadrature wavelength j, resampled to the exact "
                "UPDC pixel scale"
            ),
            "detector_recommended_input": str(psf_path),
        },
    }
    write_json(band_dir / f"{band}_diagnostics.json", record)
    return record



def patch_json(path: Path, additions: dict[str, Any]) -> None:
    if not path.exists():
        return
    payload = json.loads(path.read_text(encoding="utf-8"))
    payload.update(jsonable(additions))
    write_json(path, payload)


def update_diagnostics_index(
    path: Path,
    snapshot_record: dict[str, Any],
) -> None:
    if path.exists():
        payload = json.loads(path.read_text(encoding="utf-8"))
    else:
        payload = {"snapshots": []}
    payload.setdefault("snapshots", []).append(
        jsonable(snapshot_record)
    )
    write_json(path, payload)


def run_diagnostics(
    *,
    context: dict[str, Any],
    references: dict[str, Path],
    args: argparse.Namespace,
    baseline_path: Path,
    output_dir: Path,
    underlying_argv: Sequence[str],
    snapshot_name: str,
    training_step: int,
    global_epoch: int | None,
    snapshot_kind: str,
) -> dict[str, Any]:
    checkpoint_path = Path(context["checkpoint_path"])
    prefix = checkpoint_path.stem.removesuffix("_checkpoint")
    diagnostics_root = output_dir / f"{prefix}_diagnostics"
    diagnostics_dir = diagnostics_root / snapshot_name
    diagnostics_dir.mkdir(parents=True, exist_ok=False)

    snapshot = diagnostics_root / "baseline_manifest_snapshot.json"
    if not snapshot.exists():
        snapshot.write_text(
            baseline_path.read_text(encoding="utf-8"),
            encoding="utf-8",
        )

    was_training = bool(context["model"].training)
    try:
        native = native_observation_residuals(
            model=context["model"],
            pools=context["pools"],
            wavelength=context["wavelength"],
            weight=context["weight"],
            psf_bank=context["psf_bank"],
            intensity_scale=context["intensity_scale"],
            device=context["device"],
            samples_per_exposure=(
                args.native_residual_samples_per_exposure
            ),
            seed=context["seed"] + int(training_step),
            psf_offset_chunk=args.psf_offset_chunk,
            output_dir=diagnostics_dir
            / "native_observation_stage",
        )
        band_records: dict[str, Any] = {}
        for band in BANDS:
            try:
                band_records[band] = diagnose_band(
                    band=band,
                    reference_path=references[band],
                    context=context,
                    args=args,
                    output_root=(
                        diagnostics_dir / "updc_field_stage"
                    ),
                )
            except Exception as exc:
                band_records[band] = {
                    "status": "failed",
                    "error": repr(exc),
                    "traceback": traceback.format_exc(),
                }
                write_json(
                    diagnostics_dir
                    / "updc_field_stage"
                    / band
                    / "failure.json",
                    band_records[band],
                )
    finally:
        if was_training:
            context["model"].train()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    tracking = {
        "status": "complete",
        "snapshot_kind": snapshot_kind,
        "snapshot_name": snapshot_name,
        "training_step": int(training_step),
        "global_epoch": (
            None if global_epoch is None else int(global_epoch)
        ),
        "step_loss": context.get("current_step_loss"),
        "completed_utc": now_utc(),
        "baseline_manifest": str(baseline_path),
        "baseline_snapshot": str(snapshot),
        "underlying_argv": list(underlying_argv),
        "launcher_args": vars(args),
        "checkpoint": str(checkpoint_path),
        "loss_log": str(context["loss_log_path"]),
        "PSF": context["psf_record"],
        "training_history_tail": context.get(
            "training_history_tail",
            [],
        ),
        "native_observation_stage": native,
        "updc_field_stage": band_records,
        "residual_signs": {
            "native": "prediction_minus_native_observation",
            "intrinsic_UPDC": (
                "A1_intrinsic_bandpass_field_minus_UPDC"
            ),
            "psf_UPDC": (
                "A1_psf_observed_bandpass_field_minus_UPDC"
            ),
        },
    }
    tracking_path = diagnostics_dir / "tracking_manifest.json"
    tracking["tracking_manifest"] = str(tracking_path)
    tracking["diagnostics_directory"] = str(diagnostics_dir)
    write_json(tracking_path, tracking)

    index_path = diagnostics_root / "diagnostics_index.json"
    update_diagnostics_index(
        index_path,
        {
            "snapshot_kind": snapshot_kind,
            "snapshot_name": snapshot_name,
            "training_step": int(training_step),
            "global_epoch": global_epoch,
            "tracking_manifest": str(tracking_path),
            "diagnostics_directory": str(diagnostics_dir),
        },
    )

    additions = {
        "launcher": {
            "kind": (
                "A1_all_valid_epoch_400_400_psf_tracking"
            ),
            "batch_size": context["batch_spec"],
            "PSF": context["psf_record"],
        },
        "diagnostics": {
            "directory": str(diagnostics_root),
            "index": str(index_path),
            "latest_tracking_manifest": str(tracking_path),
        },
    }
    patch_json(
        output_dir / f"{prefix}_manifest.json",
        additions,
    )
    patch_json(
        output_dir / f"{prefix}_summary.json",
        additions,
    )
    return tracking


def completed_global_epochs(
    *,
    step: int,
    pools: Sequence[Any],
    batch_spec: int | Sequence[int],
) -> int:
    """Number of full passes completed by every exposure after `step` updates."""
    if isinstance(batch_spec, int):
        raise ValueError(
            "exact global-epoch tracking in this launcher requires "
            "two-value per-exposure SW/LW batch size"
        )
    short_count, long_count = [int(value) for value in batch_spec]
    completed: list[int] = []
    short_bands = training_mod.SHORT_WAVE_BANDS
    for pool in pools:
        count = (
            short_count
            if str(pool.band).upper() in short_bands
            else long_count
        )
        completed.append(
            (int(step) * count) // int(pool.size)
        )
    return min(completed)


def run_detector_once(
    *,
    args: argparse.Namespace,
    tracking: dict[str, Any],
    references: dict[str, Path],
) -> dict[str, Any]:
    """Run the existing detector once, band by band, on final PSF fields only."""
    if not args.run_detector:
        return {
            "enabled": False,
            "stage": "final_only",
            "input_stage": "PSF_OBSERVED_BANDPASS_FIELD",
        }

    script = args.detector_script.expanduser().resolve()
    if not script.exists():
        raise FileNotFoundError(script)

    if tracking.get("snapshot_kind") != "final":
        raise ValueError(
            "detector invocation is restricted to the final training snapshot"
        )

    field_records = tracking.get("updc_field_stage", {})
    detector_root = (
        Path(tracking["diagnostics_directory"])
        / "detector_stage"
    )
    detector_root.mkdir(parents=True, exist_ok=True)

    band_records: dict[str, Any] = {}
    failed_bands: list[str] = []

    for band in BANDS:
        record = field_records.get(band)
        if not isinstance(record, dict) or record.get("status") == "failed":
            band_records[band] = {
                "status": "skipped",
                "reason": "final band diagnostics are unavailable",
            }
            failed_bands.append(band)
            continue

        products = record.get("products", {})
        psf_field_raw = products.get("psf_observed_bandpass_field")
        common_mask_raw = products.get("psf_common_valid_mask")
        if psf_field_raw is None or common_mask_raw is None:
            band_records[band] = {
                "status": "skipped",
                "reason": (
                    "final PSF-observed field or common-valid mask is missing"
                ),
            }
            failed_bands.append(band)
            continue

        psf_field = Path(psf_field_raw).expanduser().resolve()
        common_mask = Path(common_mask_raw).expanduser().resolve()
        reference = references[band].expanduser().resolve()
        for required in (psf_field, common_mask, reference):
            if not required.exists():
                raise FileNotFoundError(required)

        band_dir = detector_root / band
        band_dir.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(script),
            "--image",
            f"A1={psf_field}",
            "--image",
            f"UPDC={reference}",
            "--coverage",
            f"A1={common_mask}",
            "--coverage",
            f"UPDC={common_mask}",
            "--reference-label",
            "UPDC",
            "--band",
            band,
            "--output-dir",
            str(band_dir),
            "--threshold-sigma",
            str(args.detector_threshold_sigma),
            "--npixels",
            str(args.detector_npixels),
            "--connectivity",
            str(args.detector_connectivity),
            "--background-box-size",
            str(args.detector_background_box_size),
            "--background-filter-size",
            str(args.detector_background_filter_size),
            "--sigma-clip",
            str(args.detector_sigma_clip),
            "--sigma-clip-maxiters",
            str(args.detector_sigma_clip_maxiters),
            "--match-radius-arcsec",
            str(args.detector_match_radius_arcsec),
            "--match-radius-grid-arcsec",
            str(args.detector_match_radius_grid_arcsec),
            "--snr-bins",
            str(args.detector_snr_bins),
        ]
        if args.detector_deblend:
            command.extend(
                [
                    "--deblend",
                    "--deblend-nlevels",
                    str(args.detector_deblend_nlevels),
                    "--deblend-contrast",
                    str(args.detector_deblend_contrast),
                ]
            )

        completed = subprocess.run(
            command,
            text=True,
            capture_output=True,
            check=False,
        )
        stdout_path = band_dir / "detector_stdout.log"
        stderr_path = band_dir / "detector_stderr.log"
        stdout_path.write_text(completed.stdout, encoding="utf-8")
        stderr_path.write_text(completed.stderr, encoding="utf-8")

        band_record = {
            "status": "complete" if completed.returncode == 0 else "failed",
            "band": band,
            "input_stage": "PSF_OBSERVED_BANDPASS_FIELD",
            "A1_image": str(psf_field),
            "UPDC_image": str(reference),
            "shared_coverage_mask": str(common_mask),
            "output_dir": str(band_dir),
            "command": command,
            "returncode": int(completed.returncode),
            "stdout": str(stdout_path),
            "stderr": str(stderr_path),
            "summary": str(band_dir / "catalog_comparison_summary.json"),
        }
        band_records[band] = band_record
        write_json(band_dir / "detector_invocation.json", band_record)
        if completed.returncode != 0:
            failed_bands.append(band)

    detector_record = {
        "enabled": True,
        "stage": "final_only",
        "final_snapshot": tracking["snapshot_name"],
        "final_training_step": tracking["training_step"],
        "input_stage": "PSF_OBSERVED_BANDPASS_FIELD",
        "intrinsic_fields_used": False,
        "script": str(script),
        "output_dir": str(detector_root),
        "configuration": {
            "threshold_sigma": args.detector_threshold_sigma,
            "npixels": args.detector_npixels,
            "connectivity": args.detector_connectivity,
            "background_box_size": args.detector_background_box_size,
            "background_filter_size": args.detector_background_filter_size,
            "sigma_clip": args.detector_sigma_clip,
            "sigma_clip_maxiters": args.detector_sigma_clip_maxiters,
            "deblend": args.detector_deblend,
            "deblend_nlevels": args.detector_deblend_nlevels,
            "deblend_contrast": args.detector_deblend_contrast,
            "match_radius_arcsec": args.detector_match_radius_arcsec,
            "match_radius_grid_arcsec": args.detector_match_radius_grid_arcsec,
            "snr_bins": args.detector_snr_bins,
            "coverage_policy": (
                "Use the final PSF-field/UPDC common-valid mask for both inputs"
            ),
        },
        "bands": band_records,
        "failed_bands": failed_bands,
        "status": "complete" if not failed_bands else "partial_failure",
    }
    write_json(detector_root / "detector_run_manifest.json", detector_record)

    if failed_bands:
        print(
            json.dumps(
                {
                    "warning": "detector did not complete for every band",
                    "failed_bands": failed_bands,
                    "detector_manifest": str(
                        detector_root / "detector_run_manifest.json"
                    ),
                },
                indent=2,
            ),
            flush=True,
        )
    return detector_record



def main() -> int:
    launcher_args, remaining = launcher_parser().parse_known_args(sys.argv[1:])
    stpsf_data_dir = launcher_args.stpsf_data_dir.expanduser().resolve()
    if not stpsf_data_dir.exists():
        raise FileNotFoundError(
            f"STPSF data directory does not exist: {stpsf_data_dir}. "
            "Copy the existing $HOME/data/stpsf-data directory there first."
        )
    os.environ["STPSF_PATH"] = str(stpsf_data_dir)
    print(json.dumps({"STPSF_PATH": str(stpsf_data_dir)}, indent=2), flush=True)
    batch = normalize_batch(launcher_args.batch_size)
    references = resolve_references(launcher_args)

    architecture = option_value(remaining, "--architecture")
    if architecture not in (None, "A1_nerf"):
        raise SystemExit("this launcher only supports --architecture A1_nerf")
    if architecture is None:
        remaining += ["--architecture", "A1_nerf"]
    if not has_option(remaining, "--max-pixels-per-exposure"):
        remaining += ["--max-pixels-per-exposure", "0"]
    if not has_option(remaining, "--sampling-mode"):
        remaining += ["--sampling-mode", "epoch_without_replacement"]
    if not launcher_args.use_baseline_canvas and not has_option(
        remaining, "--target-canvas-fits"
    ):
        remaining += ["--target-canvas-fits", str(references["F090W"])]
    if not launcher_args.also_render_union and not has_option(remaining, "--skip-render"):
        remaining += ["--skip-render"]

    baseline_raw = option_value(remaining, "--baseline-manifest")
    output_raw = option_value(remaining, "--output-dir")
    if baseline_raw is None or output_raw is None:
        raise SystemExit("--baseline-manifest and --output-dir are required")
    baseline_path = Path(baseline_raw).expanduser().resolve()
    output_dir = Path(output_raw).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    if launcher_args.diagnostics:
        missing = [str(path) for path in references.values() if not path.exists()]
        if missing:
            raise FileNotFoundError("missing UPDC references: " + ", ".join(missing))
    if launcher_args.psf_size <= 0 or launcher_args.psf_size % 2 != 1:
        raise SystemExit("--psf-size must be a positive odd integer")
    if launcher_args.diagnostic_tile_pixels <= 0:
        raise SystemExit("--diagnostic-tile-pixels must be positive")
    if launcher_args.epoch_save_every <= 0:
        raise SystemExit("--epoch-save-every must be positive")
    if launcher_args.max_epoch_snapshots < 0:
        raise SystemExit("--max-epoch-snapshots must be non-negative")
    if launcher_args.save_every_global_epoch and not launcher_args.diagnostics:
        raise SystemExit(
            "--save-every-global-epoch requires --diagnostics"
        )
    if launcher_args.save_every_global_epoch and isinstance(batch, int):
        raise SystemExit(
            "per-epoch snapshots require --batch-size SW LW, "
            "for example --batch-size 400 400"
        )
    if launcher_args.run_detector:
        if not launcher_args.diagnostics:
            raise SystemExit("--run-detector requires --diagnostics")
        if launcher_args.detector_script is None:
            raise SystemExit(
                "--run-detector requires --detector-script"
            )
        if (
            not math.isfinite(launcher_args.detector_threshold_sigma)
            or launcher_args.detector_threshold_sigma <= 0
        ):
            raise SystemExit(
                "--detector-threshold-sigma must be finite and positive"
            )
        if launcher_args.detector_npixels <= 0:
            raise SystemExit("--detector-npixels must be positive")
        if launcher_args.detector_background_box_size <= 0:
            raise SystemExit(
                "--detector-background-box-size must be positive"
            )
        if launcher_args.detector_background_filter_size <= 0:
            raise SystemExit(
                "--detector-background-filter-size must be positive"
            )
        if (
            not math.isfinite(launcher_args.detector_sigma_clip)
            or launcher_args.detector_sigma_clip <= 0
        ):
            raise SystemExit(
                "--detector-sigma-clip must be finite and positive"
            )
        if launcher_args.detector_sigma_clip_maxiters <= 0:
            raise SystemExit(
                "--detector-sigma-clip-maxiters must be positive"
            )
        if (
            not math.isfinite(launcher_args.detector_match_radius_arcsec)
            or launcher_args.detector_match_radius_arcsec <= 0
        ):
            raise SystemExit(
                "--detector-match-radius-arcsec must be finite and positive"
            )
        if launcher_args.detector_deblend_nlevels <= 0:
            raise SystemExit(
                "--detector-deblend-nlevels must be positive"
            )
        if not 0 <= launcher_args.detector_deblend_contrast <= 1:
            raise SystemExit(
                "--detector-deblend-contrast must lie in [0, 1]"
            )

    original_load_manifest = a1._load_manifest
    original_bandpass_from_manifest = a1._bandpass_from_manifest
    original_artifact_prefix = a1._artifact_prefix
    original_train_model = a1.train_model
    context: dict[str, Any] = {
        "batch_spec": batch,
        "last_saved_global_epoch": 0,
        "epoch_snapshot_count": 0,
        "last_training_step": 0,
        "current_step_loss": None,
    }

    def load_manifest(path: Path) -> tuple[Path, dict[str, Any]]:
        resolved, payload = original_load_manifest(path)
        payload = copy.deepcopy(payload)
        payload.setdefault("args", {})["batch_size"] = batch
        payload["effective_total_batch_size"] = None
        return resolved, payload

    def bandpass_from_manifest(
        manifest: dict[str, Any],
        *,
        num_exposures: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        wavelength, weight = original_bandpass_from_manifest(
            manifest,
            num_exposures=num_exposures,
        )
        wavelength, weight, trim_records = sanitize_nircam_channel_quadrature(
            manifest,
            wavelength,
            weight,
        )
        context["bandpass_channel_trim"] = trim_records
        if trim_records:
            print(
                json.dumps(
                    {
                        "bandpass_channel_trim": {
                            "reason": (
                                "SVO filter profile is not detector-specific; "
                                "zeroed quadrature weight outside the selected "
                                "NIRCam detector channel"
                            ),
                            "affected_exposures": len(trim_records),
                            "records": trim_records,
                        }
                    },
                    indent=2,
                ),
                flush=True,
            )
        return wavelength, weight


    def artifact_prefix(
        architecture: str,
        *,
        target_canvas: bool,
        sampling_mode: str,
        max_pixels_override: int | None,
        resolved_max_pixels: int,
    ) -> str:
        base = original_artifact_prefix(
            architecture,
            target_canvas=target_canvas,
            sampling_mode=sampling_mode,
            max_pixels_override=max_pixels_override,
            resolved_max_pixels=resolved_max_pixels,
        )
        psf_label = (
            "no_psf"
            if launcher_args.psf_mode == "none"
            else f"{launcher_args.psf_mode}_k{launcher_args.psf_size}"
        )
        return f"{base}_batch{batch_label(batch)}_{psf_label}"

    def train_model(
        model: torch.nn.Module,
        pools: Sequence[Any],
        wavelength: torch.Tensor,
        weight: torch.Tensor,
        *,
        psf: Any,
        **kwargs: Any,
    ) -> Any:
        del psf
        checkpoint_path = Path(kwargs["checkpoint_path"])
        cache_dir = (
            launcher_args.psf_cache_dir.expanduser().resolve()
            if launcher_args.psf_cache_dir is not None
            else checkpoint_path.parent / "calibration_cache" / "psf"
        )
        kernel_dir = (
            launcher_args.psf_kernel_dir.expanduser().resolve()
            if launcher_args.psf_kernel_dir is not None
            else None
        )
        calibration = build_psf_calibration(
            [pool.calibration_input() for pool in pools],
            wavelength,
            mode=launcher_args.psf_mode,
            kernel_size=launcher_args.psf_size,
            gaussian_fwhm_by_band=parse_band_float_map(launcher_args.psf_fwhm_pix),
            default_gaussian_fwhm=launcher_args.default_psf_fwhm_pix,
            kernel_dir=kernel_dir,
            cache_dir=cache_dir,
            stpsf_extension=launcher_args.stpsf_extension,
            allow_static_stpsf_fallback=launcher_args.allow_static_stpsf_fallback,
        )
        psf_record = {
            "mode": launcher_args.psf_mode,
            "kernel_size": launcher_args.psf_size,
            "offset_chunk": launcher_args.psf_offset_chunk,
            "checkpoint_chunks": not launcher_args.no_checkpoint_psf,
            "cache_dir": str(cache_dir),
            "kernel_dir": None if kernel_dir is None else str(kernel_dir),
            "stpsf_extension": launcher_args.stpsf_extension,
            "stpsf_data_dir": str(stpsf_data_dir),
            "bandpass_channel_trim": context.get("bandpass_channel_trim", []),
            "metadata": calibration.metadata,
        }
        psf_path = checkpoint_path.with_name(
            checkpoint_path.stem.removesuffix("_checkpoint") + "_psf_calibration.json"
        )
        write_json(psf_path, psf_record)
        checkpoint_args = dict(kwargs["checkpoint_args"])
        checkpoint_args["launcher"] = {
            "kind": "A1_all_valid_epoch_400_400_psf_tracking",
            "batch_size": batch,
            "PSF": psf_record,
        }
        kwargs["checkpoint_args"] = checkpoint_args
        kwargs["psf_offset_chunk"] = launcher_args.psf_offset_chunk
        kwargs["checkpoint_psf_chunks"] = not launcher_args.no_checkpoint_psf
        context.update(
            {
                "model": model,
                "pools": pools,
                "wavelength": wavelength.detach().cpu(),
                "weight": weight.detach().cpu(),
                "psf_bank": calibration.bank,
                "psf_record": psf_record,
                "intensity_scale": float(kwargs["intensity_scale"]),
                "canvas": kwargs["canvas"],
                "device": kwargs["device"],
                "checkpoint_path": checkpoint_path,
                "loss_log_path": Path(kwargs["loss_log_path"]),
                "seed": int(kwargs["seed"]),
            }
        )
        print(
            json.dumps(
                {
                    "launcher": "A1_all_valid_epoch_400_400_psf_tracking",
                    "batch_size": batch,
                    "PSF": {k: v for k, v in psf_record.items() if k != "metadata"},
                    "UPDC_references": {k: str(v) for k, v in references.items()},
                },
                indent=2,
            ),
            flush=True,
        )
        original_loss_append = training_mod.LossLogWriter.append

        def append_loss_and_maybe_snapshot(
            writer: Any,
            step: int,
            loss: float,
        ) -> None:
            original_loss_append(writer, step, loss)
            context["last_training_step"] = int(step)
            context["current_step_loss"] = float(loss)

            if not (
                launcher_args.diagnostics
                and launcher_args.save_every_global_epoch
            ):
                return

            completed = completed_global_epochs(
                step=int(step),
                pools=pools,
                batch_spec=batch,
            )
            previous = int(
                context["last_saved_global_epoch"]
            )
            if completed <= previous:
                return

            for global_epoch in range(
                previous + 1,
                completed + 1,
            ):
                context["last_saved_global_epoch"] = (
                    global_epoch
                )
                if (
                    global_epoch
                    % launcher_args.epoch_save_every
                    != 0
                ):
                    continue
                if (
                    launcher_args.max_epoch_snapshots > 0
                    and int(context["epoch_snapshot_count"])
                    >= launcher_args.max_epoch_snapshots
                ):
                    continue

                snapshot_name = (
                    f"epoch_{global_epoch:04d}_"
                    f"step_{int(step):06d}"
                )
                try:
                    run_diagnostics(
                        context=context,
                        references=references,
                        args=launcher_args,
                        baseline_path=baseline_path,
                        output_dir=output_dir,
                        underlying_argv=remaining,
                        snapshot_name=snapshot_name,
                        training_step=int(step),
                        global_epoch=global_epoch,
                        snapshot_kind="global_epoch",
                    )
                    context["epoch_snapshot_count"] = (
                        int(context["epoch_snapshot_count"])
                        + 1
                    )
                except Exception as snapshot_error:
                    failure_dir = (
                        output_dir
                        / (
                            checkpoint_path.stem.removesuffix(
                                "_checkpoint"
                            )
                            + "_diagnostics"
                        )
                        / snapshot_name
                    )
                    failure_dir.mkdir(
                        parents=True,
                        exist_ok=True,
                    )
                    write_json(
                        failure_dir
                        / "epoch_snapshot_failure.json",
                        {
                            "training_step": int(step),
                            "global_epoch": global_epoch,
                            "error": repr(snapshot_error),
                            "traceback": traceback.format_exc(),
                        },
                    )
                    print(
                        json.dumps(
                            {
                                "warning": (
                                    "epoch diagnostic snapshot "
                                    "failed; training continues"
                                ),
                                "training_step": int(step),
                                "global_epoch": global_epoch,
                                "error": repr(snapshot_error),
                            },
                            indent=2,
                        ),
                        flush=True,
                    )

        training_mod.LossLogWriter.append = (
            append_loss_and_maybe_snapshot
        )
        try:
            history = original_train_model(
                model,
                pools,
                wavelength,
                weight,
                psf=calibration.bank,
                **kwargs,
            )
        finally:
            training_mod.LossLogWriter.append = (
                original_loss_append
            )

        context["training_history_tail"] = [
            jsonable(row)
            for row in history[-20:]
        ]
        return history

    a1._load_manifest = load_manifest
    a1._bandpass_from_manifest = bandpass_from_manifest
    a1._artifact_prefix = artifact_prefix
    a1.train_model = train_model
    sys.argv = [sys.argv[0], *remaining]

    try:
        status = int(a1.main())
        if status == 0 and launcher_args.diagnostics:
            final_step = int(
                context.get("last_training_step", 0)
            )
            final_tracking = run_diagnostics(
                context=context,
                references=references,
                args=launcher_args,
                baseline_path=baseline_path,
                output_dir=output_dir,
                underlying_argv=remaining,
                snapshot_name=(
                    f"final_step_{final_step:06d}"
                ),
                training_step=final_step,
                global_epoch=(
                    None
                    if isinstance(batch, int)
                    else completed_global_epochs(
                        step=final_step,
                        pools=context["pools"],
                        batch_spec=batch,
                    )
                ),
                snapshot_kind="final",
            )
            prefix = Path(
                context["checkpoint_path"]
            ).stem.removesuffix("_checkpoint")
            detector_record = run_detector_once(
                args=launcher_args,
                tracking=final_tracking,
                references=references,
            )
            patch_json(
                output_dir / f"{prefix}_summary.json",
                {"detector": detector_record},
            )
            patch_json(
                output_dir / f"{prefix}_manifest.json",
                {"detector": detector_record},
            )
        return status
    except BaseException as exc:
        write_json(
            output_dir / "A1_launcher_failure.json",
            {
                "failed_utc": now_utc(),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
                "launcher_args": vars(launcher_args),
                "underlying_argv": remaining,
            },
        )
        raise
    finally:
        a1._load_manifest = original_load_manifest
        a1._bandpass_from_manifest = original_bandpass_from_manifest
        a1._artifact_prefix = original_artifact_prefix
        a1.train_model = original_train_model


if __name__ == "__main__":
    raise SystemExit(main())
