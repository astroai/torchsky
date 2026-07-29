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
import torch
import torch.nn.functional as F

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

import torchfits  # noqa: E402
from download_updec_webb import UPDEC_WEBB_FILES, download_spec  # noqa: E402


def _center_crop(array: np.ndarray, crop_size: int) -> np.ndarray:
    if crop_size <= 0 or crop_size >= min(array.shape[-2:]):
        return array
    h, w = array.shape[-2:]
    hh = crop_size // 2
    cy = h // 2
    cx = w // 2
    return array[cy - hh : cy - hh + crop_size, cx - hh : cx - hh + crop_size]


def _aligned_tan_center_crops(
    target: np.ndarray,
    target_header: dict[str, Any],
    source: np.ndarray,
    source_header: dict[str, Any],
    *,
    crop_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Bilinearly sample a shared central TAN crop on the target pixel grid."""
    for key in ("CTYPE1", "CTYPE2", "CRVAL1", "CRVAL2"):
        if target_header.get(key) != source_header.get(key):
            raise ValueError(f"UPdec pair has incompatible {key}")
    if target_header.get("CTYPE1") != "RA---TAN" or target_header.get(
        "CTYPE2"
    ) != "DEC--TAN":
        raise ValueError("UPdec pair alignment currently requires RA---TAN/DEC--TAN")
    for header in (target_header, source_header):
        if any(
            key in header
            for key in ("CD1_1", "CD1_2", "CD2_1", "CD2_2", "PC1_2", "PC2_1")
        ):
            raise ValueError("UPdec pair alignment requires axis-aligned CDELT WCS")

    size = min(int(crop_size), int(target.shape[-2]), int(target.shape[-1]))
    if size <= 0:
        raise ValueError("crop_size must be positive")
    target_cx = float(target_header["CRPIX1"]) - 1.0
    target_cy = float(target_header["CRPIX2"]) - 1.0
    x0 = int(round(target_cx - 0.5 * (size - 1)))
    y0 = int(round(target_cy - 0.5 * (size - 1)))
    x0 = min(max(x0, 0), int(target.shape[-1]) - size)
    y0 = min(max(y0, 0), int(target.shape[-2]) - size)
    target_crop = target[y0 : y0 + size, x0 : x0 + size]

    target_x = torch.arange(x0, x0 + size, dtype=torch.float64)
    target_y = torch.arange(y0, y0 + size, dtype=torch.float64)
    yy, xx = torch.meshgrid(target_y, target_x, indexing="ij")
    source_x = (
        (xx + 1.0 - float(target_header["CRPIX1"]))
        * float(target_header["CDELT1"])
        / float(source_header["CDELT1"])
        + float(source_header["CRPIX1"])
        - 1.0
    )
    source_y = (
        (yy + 1.0 - float(target_header["CRPIX2"]))
        * float(target_header["CDELT2"])
        / float(source_header["CDELT2"])
        + float(source_header["CRPIX2"])
        - 1.0
    )
    grid = torch.stack(
        (
            2.0 * source_x / max(int(source.shape[-1]) - 1, 1) - 1.0,
            2.0 * source_y / max(int(source.shape[-2]) - 1, 1) - 1.0,
        ),
        dim=-1,
    ).to(dtype=torch.float32)
    source_tensor = torch.from_numpy(source.astype(np.float32, copy=False))
    finite = torch.isfinite(source_tensor)
    sampled = F.grid_sample(
        torch.where(finite, source_tensor, 0.0)[None, None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    sampled_support = F.grid_sample(
        finite.to(dtype=torch.float32)[None, None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0, 0]
    aligned_source = sampled.numpy()
    valid = (
        np.isfinite(target_crop)
        & np.isfinite(aligned_source)
        & (sampled_support.numpy() >= 1.0 - 1.0e-6)
    )
    return target_crop, aligned_source, valid


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


def _time_torchfits_read(path: Path, *, hdu: int) -> tuple[float, tuple[int, ...]]:
    t0 = time.perf_counter()
    image = torchfits.read(str(path), hdu=hdu)
    dt = (time.perf_counter() - t0) * 1000.0
    return dt, tuple(int(v) for v in image.shape)


def _open_sci(path: Path) -> tuple[np.ndarray, dict[str, Any], int]:
    with torchfits.open(str(path)) as hdul:
        n_hdus = len(hdul)
    data, header = torchfits.read(str(path), hdu="SCI", return_header=True)
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

    crop0, crop1, crop_mask = _aligned_tan_center_crops(
        drizzle,
        hdr0,
        updc,
        hdr1,
        crop_size=crop_size,
    )
    crop_diff = crop1[crop_mask] - crop0[crop_mask]

    read_ms0, read_shape0 = _time_torchfits_read(iter0_path, hdu=1)
    read_ms1, read_shape1 = _time_torchfits_read(iter10_path, hdu=1)

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
        "iter0_shape": list(drizzle.shape),
        "iter10_shape": list(updc.shape),
        "crop_size": crop_size,
        "alignment": "analytic_axis_aligned_tan_bilinear",
        "aligned_valid_pixels": int(crop_mask.sum()),
        "iter0_hdus": n_hdus0,
        "iter10_hdus": n_hdus1,
        "same_wcs_keys": same_wcs,
        "iter0_bunit": hdr0.get("BUNIT"),
        "iter10_bunit": hdr1.get("BUNIT"),
        "iter0_filter": hdr0.get("FILTER"),
        "iter10_filter": hdr1.get("FILTER"),
        "common_rmse": float(np.sqrt(np.mean(crop_diff * crop_diff))),
        "common_mae": float(np.mean(np.abs(crop_diff))),
        "common_flux_ratio": float(
            np.sum(crop1[crop_mask]) / np.sum(crop0[crop_mask])
        ),
        "crop_rmse": float(np.sqrt(np.mean(crop_diff * crop_diff))),
        "crop_mae": float(np.mean(np.abs(crop_diff))),
        "crop_flux_ratio": float(np.sum(crop1[crop_mask]) / np.sum(crop0[crop_mask])),
        "crop_gradient_mean_ratio": float(
            _gradient_mean(crop1) / _gradient_mean(crop0)
        ),
        "crop_laplacian_std_ratio": float(
            _laplacian_std(crop1) / _laplacian_std(crop0)
        ),
        "torchfits_iter0_read_ms": read_ms0,
        "torchfits_iter10_read_ms": read_ms1,
        "torchfits_iter0_shape": list(read_shape0),
        "torchfits_iter10_shape": list(read_shape1),
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
