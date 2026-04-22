#!/usr/bin/env python3
"""Exposure-ladder reproduction benchmark on real JWST SMACS J0723 data."""

from __future__ import annotations

import argparse
import glob
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    _best_overlap_cutout_from_observations,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _sci_header,
)

DEFAULT_REFERENCE_PATH = DATA / "updec_webb" / "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
DEFAULT_EXPOSURE_GLOB = str(DATA / "jwst_smacs0723" / "*_crf.fits")
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_ladder.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_ladder.md"

from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402


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


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Exposure Ladder",
        "",
        "| n_exp | support rmse | support mae | support flux ratio | coverage frac | max overlap | crop mode | crop x0 | crop y0 | time ms |",
        "| ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['n_exposures']} | {row['support_rmse']:.6f} | {row['support_mae']:.6f} | "
            f"{row['support_flux_ratio']:.4f} | {row['coverage_frac']:.4f} | {row['max_overlap']} | "
            f"{row['crop_mode']} | {row['crop_x0']} | {row['crop_y0']} | {row['time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument(
        "--exposure-glob",
        default=DEFAULT_EXPOSURE_GLOB,
        help="Glob pattern for calibrated exposure paths",
    )
    parser.add_argument("--tiers", default="2,4,6")
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal"),
        default="auto",
        help="Which local exposure product family to use when the default exposure glob is used",
    )
    parser.add_argument(
        "--filename-contains",
        default="",
        help="Optional case-insensitive filename substring filter applied to auto-discovered exposures",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--crop-mode", choices=("center", "best_overlap"), default="best_overlap"
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
    parser.add_argument("--value-semantics", default="flux")
    parser.add_argument("--stacking-method", default="weighted_mean")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    exposure_paths = [Path(path) for path in sorted(glob.glob(args.exposure_glob))]
    if not exposure_paths and args.exposure_glob == DEFAULT_EXPOSURE_GLOB:
        exposure_paths = _discover_exposure_paths(
            DATA / "jwst_smacs0723",
            input_product=args.input_product,
            filename_contains=args.filename_contains or None,
        )
    if not exposure_paths:
        raise SystemExit(f"No exposures matched: {args.exposure_glob}")
    tiers = [int(part.strip()) for part in args.tiers.split(",") if part.strip()]

    ref_data = _read_ext(args.reference_path, "SCI")
    ref_header = _sci_header(args.reference_path)

    rows: list[dict[str, Any]] = []
    for n_exposures in tiers:
        subset = exposure_paths[:n_exposures]
        if len(subset) < n_exposures:
            continue
        with tempfile.TemporaryDirectory(prefix="torchsky_smacs0723_ladder_") as tmp:
            tmpdir = Path(tmp)
            observations: list[ImageObservation] = []
            for exposure_path in subset:
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
                ),
                args.runs,
            )
            result = coadd_image_plane(
                observations,
                target_header=dict(target_header),
                resampling_kernel=args.resampling_kernel,
                value_semantics=args.value_semantics,
                stacking_method=args.stacking_method,
            )
            pred = result.data.cpu().numpy().astype(np.float32)
            support = result.weight.cpu().numpy() > 0
            metric_mask = _metric_mask(pred, ref_crop, support)
            diff = pred[metric_mask] - ref_crop[metric_mask]
            rows.append(
                {
                    "n_exposures": len(subset),
                    "support_rmse": float(np.sqrt(np.mean(diff * diff)))
                    if diff.size
                    else float("nan"),
                    "support_mae": float(np.mean(np.abs(diff)))
                    if diff.size
                    else float("nan"),
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
                    "crop_mode": args.crop_mode,
                    "crop_x0": int(x0),
                    "crop_y0": int(y0),
                    "exposure_paths": [str(path) for path in subset],
                }
            )

    payload = {"rows": rows}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_markdown(rows))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
