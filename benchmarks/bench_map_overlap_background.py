#!/usr/bin/env python3
"""Focused runtime/RSS/scratch scorecard for overlap-background mapmaking."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402
from torchsky.map.operators.execution import (  # noqa: E402
    overlap_background_scratch_bytes,
)
from torchsky.map.sky import (  # noqa: E402
    CrossExposureSkyCorrector,
    PolynomialSkyModel,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _header(size: int) -> dict[str, Any]:
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }


def _worker(*, size: int, n_obs: int, tile: int) -> dict[str, Any]:
    torch.set_num_threads(1)
    axis = torch.linspace(-1.0, 1.0, size, dtype=torch.float32)
    yy, xx = torch.meshgrid(axis, axis, indexing="ij")
    scene = (
        0.3 * torch.sin(5.0 * xx)
        + 0.2 * torch.cos(4.0 * yy)
        + torch.exp(-8.0 * (xx.square() + yy.square()))
    )
    with tempfile.TemporaryDirectory(prefix="torchsky-overlap-bench-") as work:
        root = Path(work)
        scratch = root / "scratch"
        observations: list[ImageObservation] = []
        for index in range(n_obs):
            fraction = float(index) / max(n_obs - 1, 1)
            background = (
                fraction * 3.0
                + fraction * 0.4 * xx
                - fraction * 0.25 * yy
            )
            path = root / f"observation-{index}.fits"
            torchsky.write(
                str(path),
                scene + background,
                header=_header(size),
                overwrite=True,
            )
            observations.append(
                ImageObservation(
                    str(path),
                    weight=2.0,
                    weight_is_inverse_variance=True,
                )
            )
        del xx, yy, axis, background
        gc.collect()
        rss_before = _peak_rss_bytes()
        start = time.perf_counter()
        result = coadd_image_plane(
            observations,
            spatial_tile_hw=(tile, tile),
            scratch_dir=scratch,
            cross_exposure_sky_corrector=CrossExposureSkyCorrector(
                estimator=PolynomialSkyModel(order=1, statistic="huber")
            ),
        )
        elapsed = time.perf_counter() - start
        peak_rss = _peak_rss_bytes()
        rmse = torch.sqrt(torch.mean(torch.square(result.data.cpu() - scene))).item()
        expected_ivar = float(2 * n_obs)
        ivar_error = (
            None
            if result.inverse_variance is None
            else float(
                torch.max(
                    torch.abs(result.inverse_variance.cpu() - expected_ivar)
                ).item()
            )
        )
        scratch_clean = scratch.is_dir() and not any(scratch.iterdir())
    return {
        "size": size,
        "n_observations": n_obs,
        "tile": tile,
        "output_pixels": size * size,
        "elapsed_seconds": elapsed,
        "projected_megapixels_per_second": (
            n_obs * size * size / 1.0e6 / elapsed
        ),
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak_rss,
        "peak_rss_delta_bytes": max(peak_rss - rss_before, 0),
        "scratch_required_bytes": overlap_background_scratch_bytes(
            n_obs,
            (size, size),
            stacking_method="weighted_mean",
        ),
        "scratch_clean": scratch_clean,
        "scene_rmse": rmse,
        "inverse_variance_max_abs_error": ivar_error,
        "checksum": float(result.data.sum().item()),
    }


def _metadata(row: dict[str, Any]) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": "torchsky.map_overlap_background_scorecard/v1",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchsky": torchsky.__version__,
        "git_commit": commit or None,
        "result": row,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--n-observations", type=int, default=4)
    parser.add_argument("--tile", type=int, default=256)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.size <= 0 or args.n_observations < 2 or args.tile <= 0:
        parser.error("size/tile must be positive and n-observations must be at least 2")
    if args.worker:
        print(
            json.dumps(
                _worker(
                    size=args.size,
                    n_obs=args.n_observations,
                    tile=args.tile,
                )
            )
        )
        return 0
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--size",
        str(args.size),
        "--n-observations",
        str(args.n_observations),
        "--tile",
        str(args.tile),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    scorecard = _metadata(json.loads(completed.stdout))
    rendered = json.dumps(scorecard, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
