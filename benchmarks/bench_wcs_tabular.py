#!/usr/bin/env python3
"""Benchmark torch-native FITS -TAB lookup WCS paths."""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Callable

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.wcs.core import WCS as TorchWCS  # noqa: E402
from torchsky.wcs.tabular import CoupledTabularLookupWCS  # noqa: E402


def _time_median(fn: Callable[[], Any], runs: int) -> tuple[float, Any]:
    result = fn()
    timings: list[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        result = fn()
        timings.append(time.perf_counter() - start)
    return statistics.median(timings), result


def _make_coupled_lookup(grid_size: int) -> CoupledTabularLookupWCS:
    axis = torch.linspace(0.0, float(grid_size - 1), grid_size, dtype=torch.float64)
    x = axis[:, None].expand(grid_size, grid_size)
    y = axis[None, :].expand(grid_size, grid_size)
    world = torch.stack(
        (
            400.0 + 2.0 * x + 0.125 * y,
            59000.0 + 0.03 * x + 0.5 * y,
        ),
        dim=-1,
    )
    return CoupledTabularLookupWCS(
        (axis, axis),
        world,
        names=("WAVE", "TIME"),
        units=("nm", "d"),
    )


def _make_coupled_wcs(grid_size: int) -> TorchWCS:
    axis = torch.linspace(0.0, float(grid_size - 1), grid_size, dtype=torch.float64)
    x = axis[:, None].expand(grid_size, grid_size)
    y = axis[None, :].expand(grid_size, grid_size)
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CUNIT1": "nm",
        "CUNIT2": "d",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "PS1_1": "WAVE",
        "PV1_3": 1,
        "PS2_1": "TIME",
        "PV2_3": 2,
    }
    tables = {
        "WAVE": 400.0 + 2.0 * x + 0.125 * y,
        "TIME": 59000.0 + 0.03 * x + 0.5 * y,
    }
    return TorchWCS(header, tables=tables)


def _sample_pixels(grid_size: int, n_points: int) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(12345)
    return torch.rand((n_points, 2), generator=generator, dtype=torch.float64) * float(
        grid_size - 1
    )


def _row(
    *,
    surface: str,
    direction: str,
    n_points: int,
    grid_size: int,
    seconds: float,
    max_abs_error: float,
) -> dict[str, Any]:
    return {
        "domain": "wcs_tabular",
        "surface": surface,
        "case": "coupled_2d_affine",
        "direction": direction,
        "grid_size": grid_size,
        "n_points": n_points,
        "torchsky_s": seconds,
        "torchsky_kpts_s": n_points / seconds / 1.0e3,
        "max_abs_error": max_abs_error,
        "error_unit": "pixel" if direction == "world_to_pixel" else "world",
    }


def _run_surface(
    surface: str,
    transform: Any,
    pixels: torch.Tensor,
    runs: int,
    grid_size: int,
) -> list[dict[str, Any]]:
    forward_s, world = _time_median(lambda: transform.pixel_to_world(pixels), runs)
    inverse_s, pixels_back = _time_median(lambda: transform.world_to_pixel(world), runs)
    pixel_error = float(torch.max(torch.abs(pixels_back - pixels)).item())
    return [
        _row(
            surface=surface,
            direction="pixel_to_world",
            n_points=int(pixels.shape[0]),
            grid_size=grid_size,
            seconds=forward_s,
            max_abs_error=0.0,
        ),
        _row(
            surface=surface,
            direction="world_to_pixel",
            n_points=int(pixels.shape[0]),
            grid_size=grid_size,
            seconds=inverse_s,
            max_abs_error=pixel_error,
        ),
    ]


def _print_summary(rows: list[dict[str, Any]]) -> None:
    print("| Surface | Direction | Grid | N | torchsky | Throughput | Max error |")
    print("|---|---|---:|---:|---:|---:|---:|")
    for row in rows:
        print(
            "| {surface} | {direction} | {grid_size} | {n_points} | "
            "{torchsky_s:.6g}s | {torchsky_kpts_s:.3g} kpts/s | "
            "{max_abs_error:.3g} {error_unit} |".format(**row)
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--grid-size", type=int, default=8)
    parser.add_argument("--n-points", type=int, default=64)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if args.grid_size < 2:
        raise ValueError("--grid-size must be >= 2")
    if args.n_points < 1:
        raise ValueError("--n-points must be >= 1")
    if args.runs < 1:
        raise ValueError("--runs must be >= 1")

    pixels = _sample_pixels(args.grid_size, args.n_points)
    rows = []
    rows.extend(
        _run_surface(
            "lookup",
            _make_coupled_lookup(args.grid_size),
            pixels,
            args.runs,
            args.grid_size,
        )
    )
    rows.extend(
        _run_surface(
            "wcs",
            _make_coupled_wcs(args.grid_size),
            pixels,
            args.runs,
            args.grid_size,
        )
    )

    _print_summary(rows)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
