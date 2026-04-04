#!/usr/bin/env python3
"""Reproduction sweep against the published SMACS J0723 F277W drizzle product."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
DATA = ROOT / "data"
DEFAULT_REFERENCE_PATH = DATA / "updec_webb" / "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
DEFAULT_EXPOSURE_DIR = DATA / "jwst_smacs0723"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_reproduce_smoke.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_reproduce_smoke.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_smacs0723_real import (  # noqa: E402
    _best_overlap_cutout_from_observations,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _sci_header,
)

import torchsky  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402

try:  # noqa: E402
    from astropy.io import fits  # type: ignore
    from astropy.wcs import WCS  # type: ignore
except Exception:  # pragma: no cover - optional real-data helper
    fits = None
    WCS = None

try:  # noqa: E402
    from drizzle.resample import Drizzle as DrizzleResampler  # type: ignore
    from drizzle.utils import calc_pixmap, estimate_pixel_scale_ratio  # type: ignore
except Exception:  # pragma: no cover - optional real-data helper
    DrizzleResampler = None
    calc_pixmap = None
    estimate_pixel_scale_ratio = None


VARIANTS: tuple[tuple[str, str, str], ...] = (
    ("bilinear", "surface_brightness", "weighted_mean"),
    ("bicubic", "surface_brightness", "weighted_mean"),
    ("lanczos3", "surface_brightness", "weighted_mean"),
    ("lanczos5", "surface_brightness", "weighted_mean"),
    ("mitchell_netravali", "surface_brightness", "weighted_mean"),
    ("catmull_rom", "surface_brightness", "weighted_mean"),
    ("drizzle_point", "surface_brightness", "weighted_mean"),
    ("drizzle_square", "surface_brightness", "weighted_mean"),
    ("bilinear", "flux", "weighted_mean"),
    ("lanczos3", "flux", "weighted_mean"),
)

def _variant_specs(
    *,
    include_drizzle: bool,
    drizzle_kernels: tuple[str, ...],
    drizzle_pixfrac: float,
    only_drizzle: bool,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not only_drizzle:
        for kernel, value_semantics, stacking_method in VARIANTS:
            variant = f"{kernel}:{value_semantics}:{stacking_method}"
            if kernel in {"drizzle_point", "drizzle_square"}:
                variant = f"{variant}:pixfrac={drizzle_pixfrac:.3f}"
            rows.append(
                {
                    "backend": "torchsky",
                    "variant": variant,
                    "kernel": kernel,
                    "value_semantics": value_semantics,
                    "stacking_method": stacking_method,
                    "drizzle_pixfrac": float(drizzle_pixfrac),
                }
            )
    if include_drizzle:
        for kernel in drizzle_kernels:
            rows.append(
                {
                    "backend": "drizzle",
                    "variant": f"drizzle:{kernel}:pixfrac={drizzle_pixfrac:.3f}",
                    "kernel": kernel,
                    "pixfrac": float(drizzle_pixfrac),
                    "value_semantics": "surface_brightness",
                }
            )
    return rows


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


def _row_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Reproduction Sweep",
        "",
        "| variant | support rmse | support mae | support flux ratio | coverage frac | max overlap | time ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['variant']} | {row['support_rmse']:.6f} | {row['support_mae']:.6f} | "
            f"{row['support_flux_ratio']:.4f} | {row['coverage_frac']:.4f} | "
            f"{row['max_overlap']} | {row['time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _as_fits_header(header: dict[str, Any]) -> "fits.Header":
    if fits is None:  # pragma: no cover - checked by caller
        raise RuntimeError("astropy is unavailable")
    if isinstance(header, fits.Header):
        return header.copy()
    out = fits.Header()
    for key, value in header.items():
        if str(key).endswith("_COMMENT"):
            continue
        if key in {"", "COMMENT", "HISTORY"}:
            continue
        out[key] = value
    return out


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


def _reference_pixfrac(reference_path: Path) -> float:
    if fits is None:  # pragma: no cover - checked by caller
        return 1.0
    with fits.open(reference_path) as hdul:
        return float(hdul[0].header.get("PIXFRAC", 1.0))


def _coadd_with_drizzle(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
    kernel: str,
    pixfrac: float,
    value_semantics: str = "surface_brightness",
) -> tuple[np.ndarray, np.ndarray]:
    if (
        DrizzleResampler is None
        or calc_pixmap is None
        or estimate_pixel_scale_ratio is None
        or WCS is None
        or fits is None
    ):
        raise RuntimeError("drizzle benchmark path requires astropy + drizzle")

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
            iscale=1.0 if value_semantics == "surface_brightness" else pscale_ratio**2,
            pixel_scale_ratio=pscale_ratio,
            pixfrac=float(pixfrac),
            in_units="cps",
        )
    data = np.asarray(drizzle.out_img, dtype=np.float32)
    weight = np.asarray(drizzle.out_wht, dtype=np.float32)
    data = np.where(weight > 0.0, data, 0.0)
    return data, weight


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--exposures", default="")
    parser.add_argument("--crop-size", type=int, default=512)
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
    parser.add_argument(
        "--include-drizzle",
        action="store_true",
        help="Also benchmark the external drizzle resampler on the same crop",
    )
    parser.add_argument(
        "--drizzle-kernels",
        default="point,square",
        help="Comma-separated drizzle kernels to compare when --include-drizzle is set",
    )
    parser.add_argument(
        "--only-drizzle",
        action="store_true",
        help="When set with --include-drizzle, skip the torchsky rows and run only drizzle variants",
    )
    parser.add_argument(
        "--drizzle-pixfrac",
        type=float,
        default=None,
        help="Override drizzle pixfrac; defaults to the published reference PIXFRAC",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--linearize-sip",
        action="store_true",
        help="Strip SIP terms from the calibrated exposure headers as a fallback",
    )
    args = parser.parse_args()

    if args.exposures.strip():
        exposure_paths = [Path(part.strip()) for part in args.exposures.split(",") if part.strip()]
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
    drizzle_pixfrac = (
        float(args.drizzle_pixfrac)
        if args.drizzle_pixfrac is not None
        else _reference_pixfrac(args.reference_path)
    )
    drizzle_kernels = tuple(
        part.strip() for part in args.drizzle_kernels.split(",") if part.strip()
    )
    variant_specs = _variant_specs(
        include_drizzle=args.include_drizzle,
        drizzle_kernels=drizzle_kernels,
        drizzle_pixfrac=drizzle_pixfrac,
        only_drizzle=args.only_drizzle,
    )
    ref_data = _read_ext(args.reference_path, "SCI")
    ref_header = _sci_header(args.reference_path)
    with tempfile.TemporaryDirectory(prefix="torchsky_smacs0723_repro_") as tmp:
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
                resampling_kernel="bilinear",
                value_semantics="flux",
                stacking_method="weighted_mean",
            )
        else:
            ref_crop, x0, y0 = _center_cutout(ref_data, args.crop_size)
        target_header = _cutout_header(ref_header, x0, y0, args.crop_size)

        rows: list[dict[str, Any]] = []
        for spec in variant_specs:
            if spec["backend"] == "drizzle":
                dt = _time_many(
                    lambda: _coadd_with_drizzle(
                        observations,
                        target_header=dict(target_header),
                        kernel=str(spec["kernel"]),
                        pixfrac=float(spec["pixfrac"]),
                        value_semantics=str(spec.get("value_semantics", "surface_brightness")),
                    ),
                    args.runs,
                )
                pred, weight = _coadd_with_drizzle(
                    observations,
                    target_header=dict(target_header),
                    kernel=str(spec["kernel"]),
                    pixfrac=float(spec["pixfrac"]),
                    value_semantics=str(spec.get("value_semantics", "surface_brightness")),
                )
                support = weight > 0.0
                max_overlap = 0
            else:
                dt = _time_many(
                    lambda: coadd_image_plane(
                        observations,
                        target_header=dict(target_header),
                        resampling_kernel=str(spec["kernel"]),
                        value_semantics=str(spec["value_semantics"]),
                        stacking_method=str(spec["stacking_method"]),
                        drizzle_pixfrac=float(spec.get("drizzle_pixfrac", drizzle_pixfrac)),
                    ),
                    args.runs,
                )
                result = coadd_image_plane(
                    observations,
                    target_header=dict(target_header),
                    resampling_kernel=str(spec["kernel"]),
                    value_semantics=str(spec["value_semantics"]),
                    stacking_method=str(spec["stacking_method"]),
                    drizzle_pixfrac=float(spec.get("drizzle_pixfrac", drizzle_pixfrac)),
                )
                pred = result.data.cpu().numpy().astype(np.float32)
                support = result.weight.cpu().numpy() > 0
                max_overlap = (
                    int(result.overlap_count.max().item()) if result.overlap_count is not None else 0
                )
            metric_mask = _metric_mask(pred, ref_crop, support)
            diff = pred[metric_mask] - ref_crop[metric_mask]
            rows.append(
                {
                    "variant": str(spec["variant"]),
                    "support_rmse": float(np.sqrt(np.mean(diff * diff))) if diff.size else float("nan"),
                    "support_mae": float(np.mean(np.abs(diff))) if diff.size else float("nan"),
                    "support_flux_ratio": (
                        float(np.sum(pred[metric_mask]) / np.sum(ref_crop[metric_mask])) if diff.size else float("nan")
                    ),
                    "coverage_frac": float(metric_mask.mean()),
                    "max_overlap": max_overlap,
                    "time_ms": dt * 1000.0,
                    "crop_mode": args.crop_mode,
                    "crop_x0": int(x0),
                    "crop_y0": int(y0),
                    "input_product": args.input_product,
                    "filename_contains": args.filename_contains,
                }
            )

    rows.sort(key=lambda row: row["support_rmse"])
    payload = {"rows": rows}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_row_markdown(rows))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
