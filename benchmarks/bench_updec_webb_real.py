#!/usr/bin/env python3
"""Real-data smoke benchmark on public UPdec-Webb products."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
SCRIPTS = ROOT / "scripts"
DATA = ROOT / "data"
DEFAULT_INPUT_DIR = DATA / "updec_webb"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "updec_webb_real_smoke.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "updec_webb_real_smoke.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

import torchsky  # noqa: E402
from download_updec_webb import UPDEC_WEBB_FILES, download_spec  # noqa: E402


def _center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0 or crop_size >= min(array.shape[-2:]):
        return array
    h, w = array.shape[-2:]
    hh = crop_size // 2
    cy = h // 2
    cx = w // 2
    return array[cy - hh : cy - hh + crop_size, cx - hh : cx - hh + crop_size]


def _laplacian_std(array: np.ndarray) -> float:
    center = array[1:-1, 1:-1]
    lap = (
        4.0 * center
        - array[:-2, 1:-1]
        - array[2:, 1:-1]
        - array[1:-1, :-2]
        - array[1:-1, 2:]
    )
    finite = lap[np.isfinite(lap)]
    return float(finite.std()) if finite.size else float("nan")


def _gradient_mean(array: np.ndarray) -> float:
    gy, gx = np.gradient(array)
    grad = np.sqrt(gx * gx + gy * gy)
    finite = grad[np.isfinite(grad)]
    return float(finite.mean()) if finite.size else float("nan")


def _time_torchsky_read(path: Path, *, hdu: int) -> tuple[float, tuple[int, ...]]:
    t0 = time.perf_counter()
    image = torchsky.read(str(path), hdu=hdu)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, tuple(int(v) for v in image.shape)


def _open_sci(path: Path) -> tuple[np.ndarray, dict[str, Any], int]:
    with torchsky.open(str(path)) as hdul:
        n_hdus = len(hdul)
    data, header = torchsky.read(str(path), hdu="SCI", return_header=True)
    return (
        data.detach().cpu().numpy().astype(np.float32, copy=False),
        dict(header),
        n_hdus,
    )


def _compare_pair(
    iter0_path: Path,
    iter10_path: Path,
    *,
    crop_size: int,
) -> dict[str, Any]:
    drizzle, hdr0, n_hdus0 = _open_sci(iter0_path)
    updc, hdr1, n_hdus1 = _open_sci(iter10_path)

    mask = np.isfinite(drizzle) & np.isfinite(updc)
    diff = updc[mask] - drizzle[mask]
    crop0 = _center_crop(drizzle, crop_size)
    crop1 = _center_crop(updc, crop_size)
    crop_mask = np.isfinite(crop0) & np.isfinite(crop1)
    crop_diff = crop1[crop_mask] - crop0[crop_mask]

    read_ms0, read_shape0 = _time_torchsky_read(iter0_path, hdu=1)
    read_ms1, read_shape1 = _time_torchsky_read(iter10_path, hdu=1)

    wcs_keys = [
        "CTYPE1",
        "CTYPE2",
        "CRVAL1",
        "CRVAL2",
        "CRPIX1",
        "CRPIX2",
        "CDELT1",
        "CDELT2",
    ]
    same_wcs = all(hdr0.get(key) == hdr1.get(key) for key in wcs_keys)

    return {
        "shape": list(drizzle.shape),
        "crop_size": crop_size,
        "iter0_hdus": n_hdus0,
        "iter10_hdus": n_hdus1,
        "same_wcs_keys": same_wcs,
        "iter0_bunit": hdr0.get("BUNIT"),
        "iter10_bunit": hdr1.get("BUNIT"),
        "iter0_filter": hdr0.get("FILTER"),
        "iter10_filter": hdr1.get("FILTER"),
        "common_rmse": float(np.sqrt(np.mean(diff * diff))),
        "common_mae": float(np.mean(np.abs(diff))),
        "common_flux_ratio": float(np.sum(updc[mask]) / np.sum(drizzle[mask])),
        "crop_rmse": float(np.sqrt(np.mean(crop_diff * crop_diff))),
        "crop_mae": float(np.mean(np.abs(crop_diff))),
        "crop_flux_ratio": float(np.sum(crop1[crop_mask]) / np.sum(crop0[crop_mask])),
        "crop_gradient_mean_ratio": float(
            _gradient_mean(crop1) / _gradient_mean(crop0)
        ),
        "crop_laplacian_std_ratio": float(
            _laplacian_std(crop1) / _laplacian_std(crop0)
        ),
        "torchsky_iter0_read_ms": read_ms0,
        "torchsky_iter10_read_ms": read_ms1,
        "torchsky_iter0_shape": list(read_shape0),
        "torchsky_iter10_shape": list(read_shape1),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# UPdec-Webb Real Smoke",
        "",
        "| band | iter0 hdus | iter10 hdus | rmse | crop rmse | crop grad ratio | crop lap ratio | flux ratio |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        lines.append(
            "| {band} | {iter0_hdus} | {iter10_hdus} | {common_rmse:.6f} | {crop_rmse:.6f} | "
            "{crop_gradient_mean_ratio:.4f} | {crop_laplacian_std_ratio:.4f} | {common_flux_ratio:.4f} |".format(
                **row
            )
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F277W", help="Comma-separated band list")
    parser.add_argument("--input-dir", type=Path, default=DEFAULT_INPUT_DIR)
    parser.add_argument(
        "--download", action="store_true", help="Download missing files into input-dir"
    )
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--crop-size", type=int, default=768)
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    bands = [band.strip().upper() for band in args.bands.split(",") if band.strip()]
    rows: list[dict[str, Any]] = []
    for band in bands:
        iter0_spec = UPDEC_WEBB_FILES[(band, 0)]
        iter10_spec = UPDEC_WEBB_FILES[(band, 10)]
        iter0_path = args.input_dir / iter0_spec.filename
        iter10_path = args.input_dir / iter10_spec.filename
        if args.download:
            iter0_path = download_spec(
                iter0_spec, args.input_dir, overwrite=args.overwrite
            )
            iter10_path = download_spec(
                iter10_spec, args.input_dir, overwrite=args.overwrite
            )
        if not iter0_path.exists() or not iter10_path.exists():
            raise FileNotFoundError(
                f"Missing files for {band}: {iter0_path} and {iter10_path}. "
                "Use --download or prepopulate --input-dir."
            )
        row = {"band": band}
        row.update(_compare_pair(iter0_path, iter10_path, crop_size=args.crop_size))
        rows.append(row)

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
