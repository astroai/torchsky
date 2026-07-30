#!/usr/bin/env python3
"""Real-exposure SMACS J0723 smoke benchmark against a published F277W drizzle product."""

from __future__ import annotations

import argparse
import json
import os
import sys
import tempfile
import time
import math
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"


def _smacs_data_root() -> Path:
    """Root directory for JWST/UPdec inputs and benchmark outputs.

    Set ``TORCHSKY_DATA`` to use scratch or shared storage; expected layout::

        $TORCHSKY_DATA/jwst_smacs0723/
        $TORCHSKY_DATA/updec_webb/
        $TORCHSKY_DATA/benchmark_results/   (created as needed)
    """
    raw = os.environ.get("TORCHSKY_DATA", "").strip()
    if raw:
        return Path(raw).expanduser().resolve()
    return (ROOT / "data").resolve()


DATA = _smacs_data_root()
DEFAULT_REFERENCE_PATH = DATA / "updec_webb" / "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
DEFAULT_EXPOSURE_DIR = DATA / "jwst_smacs0723"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_real_smoke.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_real_smoke.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402


SMACS0723_BAND_SPECS: dict[str, dict[str, Any]] = {
    "F090W": {
        "reference_filename": "F090Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 9000.0,
        "spectral_min_angstrom": 8000.0,
        "spectral_max_angstrom": 10000.0,
    },
    "F150W": {
        "reference_filename": "F150Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 15000.0,
        "spectral_min_angstrom": 13000.0,
        "spectral_max_angstrom": 17000.0,
    },
    "F200W": {
        "reference_filename": "F200Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 20000.0,
        "spectral_min_angstrom": 17000.0,
        "spectral_max_angstrom": 23000.0,
    },
    "F277W": {
        "reference_filename": "F277Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 27700.0,
        "spectral_min_angstrom": 24000.0,
        "spectral_max_angstrom": 32000.0,
    },
    "F356W": {
        "reference_filename": "F356Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 35600.0,
        "spectral_min_angstrom": 31000.0,
        "spectral_max_angstrom": 40000.0,
    },
    "F444W": {
        "reference_filename": "F444Wb_UPDC-v2_iter0_i2d.fits.fz",
        "wavelength_angstrom": 44400.0,
        "spectral_min_angstrom": 38000.0,
        "spectral_max_angstrom": 50000.0,
    },
}


def _band_spec(band: str) -> dict[str, Any]:
    key = str(band).upper()
    if key not in SMACS0723_BAND_SPECS:
        raise KeyError(f"unsupported SMACS0723 band {band!r}")
    return dict(SMACS0723_BAND_SPECS[key])


def _reference_path_for_band(band: str) -> Path:
    return DATA / "updec_webb" / str(_band_spec(band)["reference_filename"])


def _field_observation_kwargs_for_band(
    band: str,
    *,
    observation_mode: str = "bandpass",
) -> dict[str, Any]:
    spec = _band_spec(band)
    mode = str(observation_mode).lower()
    if mode not in {"bandpass", "exact_wavelength"}:
        raise ValueError(
            "observation_mode must be one of {'bandpass', 'exact_wavelength'}"
        )
    if mode == "exact_wavelength":
        return {
            "wavelength": float(spec["wavelength_angstrom"]),
            "spectral_min": None,
            "spectral_max": None,
            "transmission_center": None,
            "transmission_sigma": None,
            "band_name": str(band).upper(),
        }
    return {
        "wavelength": float(spec["wavelength_angstrom"]),
        "spectral_min": float(spec["spectral_min_angstrom"]),
        "spectral_max": float(spec["spectral_max_angstrom"]),
        "transmission_center": float(spec["wavelength_angstrom"]),
        "transmission_sigma": None,
        "band_name": str(band).upper(),
    }


def _module_detector_tokens(band: str, module: str | None) -> tuple[str, ...]:
    if module is None:
        return ()
    band_upper = str(band).upper()
    module_lower = str(module).lower()
    if module_lower not in {"a", "b", "any"}:
        raise ValueError("module must be one of {'a', 'b', 'any'}")
    if module_lower == "any":
        return ()
    is_long = band_upper in {"F277W", "F356W", "F444W"}
    if is_long:
        return (f"nrc{module_lower}long",)
    return tuple(f"nrc{module_lower}{idx}" for idx in range(1, 5))


def _exposure_filter_band(path: Path) -> str | None:
    try:
        primary = dict(torchsky.get_header(str(path), 0))
        try:
            science = dict(torchsky.get_header(str(path), "SCI"))
        except Exception:
            science = primary
        for key in ("FILTER", "PUPIL"):
            value = science.get(key) or primary.get(key)
            if value:
                text = str(value).strip().upper()
                if text.startswith("F") and text.endswith("W"):
                    return text
    except Exception:
        return None
    return None


def _sci_header(path: Path) -> dict[str, Any]:
    header = dict(torchsky.get_header(str(path), "SCI"))
    for axis in (1, 2):
        compressed_key = f"ZNAXIS{axis}"
        if compressed_key in header:
            header[f"NAXIS{axis}"] = int(header[compressed_key])
    return header


def _read_ext(path: Path, ext: str) -> np.ndarray:
    image = torchsky.read(str(path), hdu=ext)
    return image.detach().cpu().numpy().astype(np.float32, copy=False)


def _discover_exposure_paths(
    exposure_dir: Path,
    *,
    input_product: str = "auto",
    band: str | None = None,
    module: str | None = None,
    filename_contains: str | None = None,
) -> list[Path]:
    product = str(input_product).lower()
    if product not in {"auto", "crf", "cal", "aligned"}:
        raise ValueError(
            "input_product must be one of {'auto', 'crf', 'cal', 'aligned'}"
        )
    crf_paths = sorted(exposure_dir.glob("*_crf.fits"))
    cal_paths = sorted(exposure_dir.glob("*_cal.fits"))
    aligned_paths = sorted(exposure_dir.glob("*_rshift_tweakreg.fits"))
    if product == "crf":
        paths = crf_paths
    elif product == "cal":
        paths = cal_paths
    elif product == "aligned":
        paths = aligned_paths
    elif crf_paths:
        paths = crf_paths
    else:
        paths = cal_paths
    if band is not None:
        band_upper = str(band).upper()
        paths = [path for path in paths if _exposure_filter_band(path) == band_upper]
    detector_tokens = (
        _module_detector_tokens(band or "F277W", module) if module is not None else ()
    )
    if detector_tokens:
        paths = [
            path
            for path in paths
            if any(token in path.name.lower() for token in detector_tokens)
        ]
    if filename_contains:
        needle = str(filename_contains).lower()
        paths = [path for path in paths if needle in path.name.lower()]
    return paths


def _linearize_sip_header(header: dict[str, Any]) -> dict[str, Any]:
    out = header.copy()
    for key in list(out.keys()):
        if key in {"A_ORDER", "B_ORDER", "AP_ORDER", "BP_ORDER"}:
            del out[key]
            continue
        if key.startswith(("A_", "B_", "AP_", "BP_")):
            del out[key]
    if out.get("CTYPE1", "").endswith("-SIP"):
        out["CTYPE1"] = str(out["CTYPE1"]).replace("-SIP", "")
    if out.get("CTYPE2", "").endswith("-SIP"):
        out["CTYPE2"] = str(out["CTYPE2"]).replace("-SIP", "")
    return out


def _cutout_header(
    header: dict[str, Any], x0: int, y0: int, size: int
) -> dict[str, Any]:
    out = header.copy()
    out["NAXIS1"] = size
    out["NAXIS2"] = size
    out["CRPIX1"] = float(out["CRPIX1"]) - float(x0)
    out["CRPIX2"] = float(out["CRPIX2"]) - float(y0)
    return out


def _center_cutout(array: np.ndarray, size: int) -> tuple[np.ndarray, int, int]:
    h, w = array.shape
    x0 = max(0, (w - size) // 2)
    y0 = max(0, (h - size) // 2)
    return array[y0 : y0 + size, x0 : x0 + size], x0, y0


def _integral_image(array: np.ndarray) -> np.ndarray:
    out = np.pad(array.astype(np.float64), ((1, 0), (1, 0)), mode="constant")
    return out.cumsum(axis=0).cumsum(axis=1)


def _best_overlap_cutout(
    reference: np.ndarray,
    overlap_count: np.ndarray,
    size: int,
) -> tuple[np.ndarray, int, int]:
    h, w = reference.shape
    if size >= h or size >= w:
        return reference.copy(), 0, 0
    overlap_score = np.where(
        np.isfinite(reference), np.maximum(overlap_count, 0), 0.0
    ).astype(np.float32)
    integral = _integral_image(overlap_score)
    window_sums = integral[size:, size:] - integral[:-size, size:]
    window_sums -= integral[size:, :-size]
    window_sums += integral[:-size, :-size]
    best_y0, best_x0 = np.unravel_index(
        int(np.argmax(window_sums)), window_sums.shape
    )
    return (
        reference[best_y0 : best_y0 + size, best_x0 : best_x0 + size],
        int(best_x0),
        int(best_y0),
    )


def _header_samples(header: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    width = int(header["NAXIS1"])
    height = int(header["NAXIS2"])
    xs = np.array(
        [
            0.0,
            float(width - 1),
            float(width - 1),
            0.0,
            0.5 * float(width - 1),
            0.5 * float(width - 1),
            float(width - 1),
            0.0,
            0.5 * float(width - 1),
        ],
        dtype=np.float32,
    )
    ys = np.array(
        [
            0.0,
            0.0,
            float(height - 1),
            float(height - 1),
            0.0,
            float(height - 1),
            0.5 * float(height - 1),
            0.5 * float(height - 1),
            0.5 * float(height - 1),
        ],
        dtype=np.float32,
    )
    return xs, ys


def _approx_overlap_grid(
    headers: list[dict[str, Any]],
    target_header: dict[str, Any],
    *,
    coarse_step: int = 8,
) -> np.ndarray:
    coarse_step = max(int(coarse_step), 1)
    target_wcs = torchsky.WCS(dict(target_header))
    full_height = int(target_header["NAXIS2"])
    full_width = int(target_header["NAXIS1"])
    coarse_height = max(1, int(math.ceil(full_height / coarse_step)))
    coarse_width = max(1, int(math.ceil(full_width / coarse_step)))
    overlap = np.zeros((coarse_height, coarse_width), dtype=np.float32)

    for header in headers:
        source_wcs = torchsky.WCS(dict(header))
        sx, sy = _header_samples(header)
        ra, dec = source_wcs.pixel_to_world(torch.from_numpy(sx), torch.from_numpy(sy))
        tx, ty = target_wcs.world_to_pixel(ra, dec)
        tx_np = tx.detach().cpu().numpy()
        ty_np = ty.detach().cpu().numpy()
        valid = np.isfinite(tx_np) & np.isfinite(ty_np)
        if not np.any(valid):
            continue
        x0 = max(int(np.floor(np.min(tx_np[valid]) / coarse_step)), 0)
        y0 = max(int(np.floor(np.min(ty_np[valid]) / coarse_step)), 0)
        x1 = min(int(np.ceil(np.max(tx_np[valid]) / coarse_step)) + 1, coarse_width)
        y1 = min(int(np.ceil(np.max(ty_np[valid]) / coarse_step)) + 1, coarse_height)
        if x1 <= x0 or y1 <= y0:
            continue
        overlap[y0:y1, x0:x1] += 1.0
    return overlap


def _downsample_header(header: dict[str, Any], step: int) -> dict[str, Any]:
    if step <= 1:
        return header.copy()
    out = header.copy()
    out["NAXIS1"] = max(1, int(math.ceil(int(header["NAXIS1"]) / step)))
    out["NAXIS2"] = max(1, int(math.ceil(int(header["NAXIS2"]) / step)))
    if "CRPIX1" in out:
        out["CRPIX1"] = (float(out["CRPIX1"]) - 1.0) / float(step) + 1.0
    if "CRPIX2" in out:
        out["CRPIX2"] = (float(out["CRPIX2"]) - 1.0) / float(step) + 1.0
    for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "CDELT1", "CDELT2"):
        if key in out:
            out[key] = float(out[key]) * float(step)
    return out


def _best_overlap_cutout_from_headers(
    reference: np.ndarray,
    headers: list[dict[str, Any]],
    target_header: dict[str, Any],
    *,
    size: int,
    coarse_step: int = 8,
) -> tuple[np.ndarray, int, int]:
    overlap_coarse = _approx_overlap_grid(
        headers, target_header, coarse_step=coarse_step
    )
    ref_coarse = reference[::coarse_step, ::coarse_step]
    coarse_size = max(1, int(math.ceil(size / coarse_step)))
    _, coarse_x0, coarse_y0 = _best_overlap_cutout(
        ref_coarse, overlap_coarse, coarse_size
    )
    x0 = min(max(coarse_x0 * coarse_step, 0), max(reference.shape[1] - size, 0))
    y0 = min(max(coarse_y0 * coarse_step, 0), max(reference.shape[0] - size, 0))
    return reference[y0 : y0 + size, x0 : x0 + size], x0, y0


def _best_overlap_cutout_from_observations(
    reference: np.ndarray,
    observations: list[ImageObservation],
    target_header: dict[str, Any],
    *,
    size: int,
    coarse_step: int = 8,
    resampling_kernel: str = "bilinear",
    value_semantics: str = "flux",
    stacking_method: str = "weighted_mean",
) -> tuple[np.ndarray, int, int]:
    coarse_step = max(int(coarse_step), 1)
    coarse_header = _downsample_header(target_header, coarse_step)
    overlap_probe = coadd_image_plane(
        observations,
        target_header=dict(coarse_header),
        resampling_kernel=resampling_kernel,
        value_semantics=value_semantics,
        stacking_method=stacking_method,
    )
    overlap_grid = overlap_probe.overlap_count.cpu().numpy().astype(np.float32)
    ref_coarse = reference[::coarse_step, ::coarse_step]
    coarse_size = max(1, int(math.ceil(size / coarse_step)))
    _, coarse_x0, coarse_y0 = _best_overlap_cutout(
        ref_coarse, overlap_grid, coarse_size
    )
    x0 = min(max(coarse_x0 * coarse_step, 0), max(reference.shape[1] - size, 0))
    y0 = min(max(coarse_y0 * coarse_step, 0), max(reference.shape[0] - size, 0))
    return reference[y0 : y0 + size, x0 : x0 + size], x0, y0


def _metric_mask(
    pred: np.ndarray, reference: np.ndarray, support: np.ndarray
) -> np.ndarray:
    return support & np.isfinite(pred) & np.isfinite(reference)


def _prepare_aux_files(
    exposure_path: Path,
    tmpdir: Path,
    *,
    linearize_sip: bool,
    science_override: np.ndarray | None = None,
) -> tuple[Path, Path, Path]:
    header = _sci_header(exposure_path)
    if linearize_sip:
        header = _linearize_sip_header(header)
    science = (
        _read_ext(exposure_path, "SCI")
        if science_override is None
        else np.asarray(science_override)
    )
    err = _read_ext(exposure_path, "ERR")
    dq = _read_ext(exposure_path, "DQ")
    if science.shape != err.shape or science.shape != dq.shape:
        raise ValueError("science override must match the exposure ERR/DQ shape")
    valid = np.isfinite(err) & (err > 0) & np.isfinite(dq) & (dq == 0)
    weight = np.zeros_like(err, dtype=np.float32)
    weight[valid] = 1.0 / np.square(err[valid])
    mask = valid.astype(np.float32)

    stem = exposure_path.stem
    science_path = tmpdir / f"{stem}_sci.fits"
    weight_path = tmpdir / f"{stem}_wht.fits"
    mask_path = tmpdir / f"{stem}_mask.fits"
    torchsky.write(
        str(science_path),
        torch.from_numpy(np.asarray(science, dtype=np.float32)),
        header=header,
        overwrite=True,
    )
    torchsky.write(
        str(weight_path),
        torch.from_numpy(np.asarray(weight, dtype=np.float32)),
        header=header,
        overwrite=True,
    )
    torchsky.write(
        str(mask_path),
        torch.from_numpy(np.asarray(mask, dtype=np.float32)),
        header=header,
        overwrite=True,
    )
    return science_path, weight_path, mask_path


def _time_many(fn, runs: int) -> float:
    if runs <= 1:
        t0 = time.perf_counter()
        fn()
        return float(time.perf_counter() - t0)
    fn()
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _markdown(row: dict[str, Any]) -> str:
    return "\n".join(
        [
            "# SMACS J0723 Real Exposure Smoke",
            "",
            "| n_exp | crop | support rmse | support mae | support flux ratio | coverage frac | max overlap | time ms |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            (
                f"| {row['n_exposures']} | {row['crop_size']} | {row['support_rmse']:.6f} | "
                f"{row['support_mae']:.6f} | {row['support_flux_ratio']:.4f} | "
                f"{row['coverage_frac']:.4f} | {row['max_overlap']} | {row['time_ms']:.2f} |"
            ),
            "",
        ]
    )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--reference-path",
        type=Path,
        default=DEFAULT_REFERENCE_PATH,
        help="Published F277W iter0 drizzle product",
    )
    parser.add_argument(
        "--exposures",
        default="",
        help="Comma-separated local calibrated exposure paths",
    )
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument(
        "--crop-mode",
        choices=("center", "best_overlap"),
        default="center",
        help="How to choose the target crop from the published reference product",
    )
    parser.add_argument(
        "--overlap-step",
        type=int,
        default=8,
        help="Coarse pixel step used by footprint-based best-overlap crop selection",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--linearize-sip",
        action="store_true",
        help="Strip SIP terms from the calibrated exposure headers as a fallback",
    )
    parser.add_argument("--resampling-kernel", default="bilinear")
    parser.add_argument("--value-semantics", default="surface_brightness")
    parser.add_argument("--stacking-method", default="weighted_mean")
    parser.add_argument("--drizzle-pixfrac", type=float, default=1.0)
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal"),
        default="auto",
        help="Which local exposure product family to use when --exposures is not given",
    )
    parser.add_argument(
        "--filename-contains",
        default="",
        help="Optional case-insensitive filename substring filter applied to auto-discovered exposures",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    if args.exposures.strip():
        exposure_paths = [
            Path(part.strip()) for part in args.exposures.split(",") if part.strip()
        ]
    else:
        exposure_paths = _discover_exposure_paths(
            DEFAULT_EXPOSURE_DIR,
            input_product=args.input_product,
            filename_contains=args.filename_contains or None,
        )
    if not exposure_paths:
        raise FileNotFoundError(
            "No calibrated exposures found. Pass --exposures or populate "
            f"{DEFAULT_EXPOSURE_DIR}"
        )
    ref_data = _read_ext(args.reference_path, "SCI")
    ref_header = _sci_header(args.reference_path)
    with tempfile.TemporaryDirectory(prefix="torchsky_smacs0723_real_") as tmp:
        tmpdir = Path(tmp)
        observations: list[ImageObservation] = []
        for exposure_path in exposure_paths:
            science_path, weight_path, mask_path = _prepare_aux_files(
                exposure_path,
                tmpdir,
                linearize_sip=args.linearize_sip,
            )
            observations.append(
                ImageObservation(
                    str(science_path),
                    hdu=0,
                    weight_path=str(weight_path),
                    weight_hdu=0,
                    mask_path=str(mask_path),
                    mask_hdu=0,
                )
            )

        if args.crop_mode == "best_overlap":
            ref_crop, x0, y0 = _best_overlap_cutout_from_observations(
                ref_data,
                observations,
                ref_header,
                size=args.crop_size,
                coarse_step=args.overlap_step,
                resampling_kernel=args.resampling_kernel,
                value_semantics=args.value_semantics,
                stacking_method=args.stacking_method,
            )
        else:
            ref_crop, x0, y0 = _center_cutout(ref_data, args.crop_size)
        target_header = _cutout_header(ref_header, x0, y0, args.crop_size)

        dt = _time_many(
            lambda: coadd_image_plane(
                observations,
                target_header=dict(target_header),
                resampling_kernel=args.resampling_kernel,
                value_semantics=args.value_semantics,
                stacking_method=args.stacking_method,
                drizzle_pixfrac=args.drizzle_pixfrac,
            ),
            args.runs,
        )
        result = coadd_image_plane(
            observations,
            target_header=dict(target_header),
            resampling_kernel=args.resampling_kernel,
            value_semantics=args.value_semantics,
            stacking_method=args.stacking_method,
            drizzle_pixfrac=args.drizzle_pixfrac,
        )
        pred = result.data.cpu().numpy().astype(np.float32)
        support = result.weight.cpu().numpy() > 0
        metric_mask = _metric_mask(pred, ref_crop, support)
        diff = pred[metric_mask] - ref_crop[metric_mask]
        row = {
            "n_exposures": len(exposure_paths),
            "crop_size": args.crop_size,
            "support_rmse": float(np.sqrt(np.mean(diff * diff)))
            if diff.size
            else float("nan"),
            "support_mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
            "support_flux_ratio": (
                float(np.sum(pred[metric_mask]) / np.sum(ref_crop[metric_mask]))
                if diff.size
                else float("nan")
            ),
            "coverage_frac": float(metric_mask.mean()),
            "max_overlap": int(result.overlap_count.max().item())
            if result.overlap_count is not None
            else 0,
            "time_ms": dt * 1000.0,
            "reference_path": str(args.reference_path),
            "exposure_paths": [str(path) for path in exposure_paths],
            "linearize_sip": bool(args.linearize_sip),
            "resampling_kernel": args.resampling_kernel,
            "value_semantics": args.value_semantics,
            "stacking_method": args.stacking_method,
            "drizzle_pixfrac": float(args.drizzle_pixfrac),
            "crop_mode": args.crop_mode,
            "crop_x0": int(x0),
            "crop_y0": int(y0),
            "reference_finite_frac": float(np.isfinite(ref_crop).mean()),
            "input_product": args.input_product,
            "filename_contains": args.filename_contains,
        }

    payload = {"rows": [row]}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(row))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
