#!/usr/bin/env python3
"""End-to-end multi-exposure operator benchmark: time, memory, deterministic accumulation.

Creates N synthetic exposures (TAN projection, Gaussian sources with position
offsets to simulate dithering) and runs the full coadd_image_plane pipeline,
optionally through exposure-major output tiles.

Metrics tracked:
  - wall time (median of warm runs)
  - peak memory (tracemalloc for CPU, torch.cuda.max_memory_allocated for GPU)
  - throughput (exposures/s, megapixels/s)
  - overlap count statistics
  - deterministic accumulation: assert bit-identical results on repeat
"""

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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402

import torchsky  # noqa: E402
from torchsky.map import ImageObservation, MapResult, coadd_image_plane  # noqa: E402


def _gaussian_source(
    size: int,
    *,
    sigma: float = 12.0,
    amplitude: float = 1.0,
    cx: float | None = None,
    cy: float | None = None,
) -> torch.Tensor:
    yy, xx = torch.meshgrid(
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        torch.linspace(-1.0, 1.0, size, dtype=torch.float32),
        indexing="ij",
    )
    cx = cx or 0.0
    cy = cy or 0.0
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    return amplitude * torch.exp(-r2 / (2.0 * sigma ** 2 / size ** 2))


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    cdelt: float,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    theta = float(np.deg2rad(rotation_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(size // 2),
        "CRPIX2": float(size // 2),
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": -cdelt * cos_t,
        "CD1_2": cdelt * sin_t,
        "CD2_1": cdelt * sin_t,
        "CD2_2": cdelt * cos_t,
    }


def _create_synthetic_exposures(
    tmpdir: Path,
    *,
    size: int,
    n_obs: int,
    center_ra: float,
    center_dec: float,
    cdelt: float,
    dither_arcsec: float,
    seed: int,
) -> list[ImageObservation]:
    rng = np.random.default_rng(seed)
    obs: list[ImageObservation] = []
    for i in range(n_obs):
        ra = center_ra + rng.normal(0.0, dither_arcsec / 3600.0)
        dec = center_dec + rng.normal(0.0, dither_arcsec / 3600.0)
        rot = rng.uniform(-5.0, 5.0)
        header = _tan_header(size, crval1=ra, crval2=dec, cdelt=cdelt, rotation_deg=rot)

        image = _gaussian_source(size, cx=-0.15, cy=0.1, amplitude=1.0) + 0.3 * _gaussian_source(
            size, cx=0.25, cy=-0.2, amplitude=0.5, sigma=8.0
        )
        path = tmpdir / f"exp_{i:04d}.fits"
        torchsky.write(str(path), image, header=header, overwrite=True)
        obs.append(ImageObservation(str(path)))
    return obs


def _median_time(fn, *, warmup: int = 1, runs: int) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _peak_memory_mb(*, device: torch.device | None) -> float | None:
    if device is not None and device.type == "cuda":
        return float(torch.cuda.max_memory_allocated(device)) / (1024 * 1024)
    return None


def run_bench(
    *,
    size: int,
    n_obs: int,
    dither_arcsec: float,
    runs: int,
    device: torch.device | None,
    tile_size: int | None = None,
) -> dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmpdir_str:
        tmpdir = Path(tmpdir_str)
        observations = _create_synthetic_exposures(
            tmpdir,
            size=size,
            n_obs=n_obs,
            center_ra=12.0,
            center_dec=-3.0,
            cdelt=-0.01,
            dither_arcsec=dither_arcsec,
            seed=42,
        )

        def run_once() -> MapResult:
            return coadd_image_plane(
                observations,
                device=device,
                spatial_tile_hw=(tile_size, tile_size) if tile_size else None,
            )

        # Warmup + timing
        dt = _median_time(run_once, warmup=1, runs=runs)

        # Deterministic accumulation check: run twice, compare data tensors
        result_a = run_once()
        result_b = run_once()
        deterministic = bool(
            torch.equal(result_a.data, result_b.data)
            and torch.equal(result_a.weight, result_b.weight)
            and (
                result_a.overlap_count is None
                or result_b.overlap_count is None
                or torch.equal(result_a.overlap_count, result_b.overlap_count)
            )
        )

        # Peak memory (CUDA only; CPU torch memory not separable via tracemalloc)
        peak_mem: float | None = None
        if device is not None and device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
            run_once()
            peak_mem_val = _peak_memory_mb(device=device)
            peak_mem = round(peak_mem_val, 1) if peak_mem_val is not None else None

        mpix = size * size / 1e6
        return {
            "case": f"size={size}_nobs={n_obs}_tile={tile_size or 'none'}",
            "size": size,
            "n_obs": n_obs,
            "tile_size": tile_size,
            "dither_arcsec": dither_arcsec,
            "time_s": round(dt, 4),
            "exposures_per_second": round(n_obs / dt, 1),
            "megapixels_per_second": round(n_obs * mpix / dt, 1),
            "peak_memory_mb": round(peak_mem, 1) if peak_mem is not None else None,
            "result_shape": list(result_a.data.shape),
            "overlap_min": float(result_a.overlap_count.min().item()) if result_a.overlap_count is not None else None,
            "overlap_max": float(result_a.overlap_count.max().item()) if result_a.overlap_count is not None else None,
            "overlap_mean": float(result_a.overlap_count.to(torch.float32).mean().item()) if result_a.overlap_count is not None else None,
            "data_min": float(result_a.data.min().item()),
            "data_max": float(result_a.data.max().item()),
            "deterministic": deterministic,
            "device": str(device) if device is not None else "cpu",
        }


def run_operator_domain(
    *,
    sizes: list[int] | None = None,
    n_obs_list: list[int] | None = None,
    runs: int = 3,
    device: torch.device | None = None,
    json_out: Path | None = None,
    tile_size: int | None = None,
) -> list[dict[str, Any]]:
    sizes = sizes or [256, 512]
    n_obs_list = n_obs_list or [2, 5, 10]
    rows: list[dict[str, Any]] = []

    for size in sizes:
        for n_obs in n_obs_list:
            row = run_bench(
                size=size,
                n_obs=n_obs,
                dither_arcsec=1.0,
                runs=runs,
                device=device,
                tile_size=tile_size,
            )
            rows.append(row)
            mem_str = f"{row['peak_memory_mb']:.0f} MB" if row['peak_memory_mb'] is not None else "N/A MB"
            print(
                f"  {row['case']:30s}  "
                f"{row['time_s']:.3f}s  "
                f"{row['exposures_per_second']:.0f} exp/s  "
                f"{mem_str:>8s}  "
                f"deterministic={row['deterministic']}"
            )

    if json_out is not None:
        json_out.parent.mkdir(parents=True, exist_ok=True)
        json_out.write_text(json.dumps(rows, indent=2))
        print(f"Wrote {json_out}")

    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sizes", type=int, nargs="*", default=[256, 512])
    parser.add_argument("--n-obs", type=int, nargs="*", default=[2, 5, 10])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument(
        "--tile-size",
        type=int,
        default=None,
        help="Optional square output tile size for exposure-major coaddition",
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=DEFAULT_OUTPUT_DIR / "operator_multiexposure.json",
    )
    args = parser.parse_args()

    device = torch.device(args.device) if args.device else None
    if device is not None and device.type == "cuda" and not torch.cuda.is_available():
        print("CUDA requested but not available, falling back to CPU")
        device = None

    print("Operator multi-exposure benchmark")
    print(f"  sizes={args.sizes}  n_obs={args.n_obs}  runs={args.runs}  device={device}")

    run_operator_domain(
        sizes=args.sizes,
        n_obs_list=args.n_obs,
        runs=args.runs,
        device=device,
        json_out=args.json_out,
        tile_size=args.tile_size,
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
