#!/usr/bin/env python3
"""Smoke benchmark: synthetic radio-style TP pair + optional ``reproject`` comparator."""

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
from astropy.io import fits

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
for p in (SRC, BENCHMARKS):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    ImageObservation,
    check_map_benchmark_regressions,
    coadd_image_plane,
    plan_image_plane_target,
)

import bench_map_compat as map_compat  # noqa: E402

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "bench_map_radio_tp_smoke.py requires torchsky.cpp"
    ) from exc


def _radio_header(*, size: int, crval2: float) -> dict[str, Any]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 120.0,
        "CRVAL2": crval2,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "BUNIT": "JY/BEAM",
        "BMAJ": 0.02,
        "BMIN": 0.018,
        "BPA": 10.0,
    }


def _write_pair(tmp: Path) -> tuple[list[ImageObservation], dict[str, Any]]:
    size = 28
    yy, xx = np.mgrid[0:size, 0:size]
    g = np.exp(
        -((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / 10.0
    ).astype(np.float32)
    h1 = _radio_header(size=size, crval2=-5.0)
    h2 = _radio_header(size=size, crval2=-5.008)
    p1 = tmp / "a.fits"
    p2 = tmp / "b.fits"
    fits.writeto(p1, g, fits.Header(h1))
    fits.writeto(p2, g * 0.88, fits.Header(h2))
    obs = [
        ImageObservation(path=str(p1), hdu=None, weight=1.0),
        ImageObservation(path=str(p2), hdu=None, weight=1.0),
    ]
    plan = plan_image_plane_target(obs)
    return obs, dict(plan.header)


def _time_torchsky(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
    *,
    runs: int,
) -> tuple[torch.Tensor, float]:
    t0 = time.perf_counter()
    for _ in range(runs):
        res = coadd_image_plane(
            observations,
            target_header=target_header,
            resampling_kernel="bilinear",
            value_semantics="surface_brightness",
            stacking_method="weighted_mean",
        )
        data = res.data
    dt = (time.perf_counter() - t0) / float(runs)
    return data, dt


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
        help="Write benchmark rows JSON",
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline for regression check",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    case = "synthetic_radio_tp_pair"

    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        observations, target_header = _write_pair(tmp)
        data, dt = _time_torchsky(observations, target_header, runs=max(1, args.runs))
        ts_np = data.detach().cpu().numpy()
        shape = list(ts_np.shape)
        rows.append(
            {
                "case": case,
                "library": "torchsky",
                "method": "coadd_image_plane",
                "variant": "bilinear",
                "status": "OK",
                "shape": shape,
                "n_obs": len(observations),
                "time_ms": dt * 1000.0,
                "skip_time_regression": True,
                "value_semantics": "surface_brightness",
                "note": "Synthetic JY/BEAM TP-style headers; not a CASA simulator",
            }
        )

        if map_compat.reproject_interp is not None:
            t0 = time.perf_counter()
            rp_data, rp_wt = map_compat._coadd_with_reproject(
                observations,
                target_header=target_header,
                order="bilinear",
            )
            rp_dt = time.perf_counter() - t0
            mask = rp_wt > 1e-6
            if bool(mask.any()):
                diff = np.abs(ts_np[mask] - rp_data[mask])
                max_abs = float(diff.max())
            else:
                max_abs = float("nan")
            rows.append(
                {
                    "case": case,
                    "library": "reproject_interp",
                    "method": "reproject_interp",
                    "variant": "bilinear",
                    "status": "OK",
                    "shape": shape,
                    "n_obs": len(observations),
                    "time_ms": rp_dt * 1000.0,
                    "skip_time_regression": True,
                    "max_abs_data_diff_vs_torchsky": max_abs,
                    "comparison_data_domain": "rp_weight_positive",
                }
            )
        else:
            rows.append(
                {
                    "case": case,
                    "library": "reproject_interp",
                    "method": "reproject_interp",
                    "variant": "bilinear",
                    "status": "SKIP",
                    "reason": "reproject not installed",
                }
            )

    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")

    if args.baseline_json is not None and args.baseline_json.is_file():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errs = check_map_benchmark_regressions(rows, baseline)
        if errs:
            for e in errs:
                print(e, file=sys.stderr)
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
