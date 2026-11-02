#!/usr/bin/env python3
"""Measure torchsky-owned hot paths for the reference benchmark page.

This is a bounded, single-machine snapshot of operations torchsky owns. It is
not the scientific evaluation suite: upstream parity, survey-scale validation,
and real-data campaigns live in ``torchsky-harness``.

    pixi run docs-bench                 # write docs/data/reference_benchmarks.json
    pixi run -e test python scripts/bench_docs.py --device cuda --write
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import statistics
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs" / "data" / "reference_benchmarks.json"


def _median_seconds(fn, *, warmup: int = 1, iters: int = 3) -> float:
    for _ in range(warmup):
        fn()
    samples: list[float] = []
    for _ in range(iters):
        start = time.perf_counter()
        result = fn()
        if (
            torch.cuda.is_available()
            and torch.cuda.is_current_stream_capturing() is False
        ):
            torch.cuda.synchronize()
        samples.append(time.perf_counter() - start)
        del result
    return float(statistics.median(samples))


def _git_sha() -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _cpu_name() -> str:
    if sys.platform == "darwin":
        try:
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
            ).strip()
        except (OSError, subprocess.CalledProcessError):
            pass
    if Path("/proc/cpuinfo").is_file():
        for line in Path("/proc/cpuinfo").read_text(encoding="utf-8").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[1].strip()
    return platform.processor() or platform.machine()


def _environment(device: torch.device) -> dict[str, object]:
    import torchfits
    import torchsky

    gpu = None
    if device.type == "cuda":
        gpu = torch.cuda.get_device_name(device)
    return {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "machine": platform.machine(),
        "cpu": _cpu_name(),
        "gpu": gpu,
        "device": str(device),
        "torch": torch.__version__,
        "torchfits": torchfits.__version__,
        "torchsky": torchsky.__version__,
        "git_sha": _git_sha(),
        "threads": torch.get_num_threads(),
    }


def _tan_header(
    shape: tuple[int, int], crval=(150.0, 2.0), cdelt=0.001
) -> dict[str, object]:
    height, width = shape
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": width / 2.0 + 0.5,
        "CRPIX2": height / 2.0 + 0.5,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CDELT1": -cdelt,
        "CDELT2": cdelt,
    }


def _wcs_cases(device: torch.device) -> list[dict[str, object]]:
    import torchsky

    header = _tan_header((4096, 4096), cdelt=0.0002777778)
    wcs = torchsky.WCS(header)
    cases: list[dict[str, object]] = []
    for n in (10_000, 200_000):
        x = torch.linspace(1500.0, 2500.0, n, dtype=torch.float64, device=device)
        y = torch.linspace(1500.0, 2500.0, n, dtype=torch.float64, device=device)
        pixels = torch.stack([x, y], dim=-1)
        world = wcs.pixel_to_world(pixels)
        for case, fn in (
            ("wcs_pixel_to_world_tan", lambda: wcs.pixel_to_world(pixels)),
            ("wcs_world_to_pixel_tan", lambda: wcs.world_to_pixel(world)),
        ):
            seconds = _median_seconds(fn)
            cases.append(
                {
                    "group": "WCS",
                    "case": case,
                    "parameters": f"n={n:,}, float64, TAN",
                    "n": n,
                    "median_s": round(seconds, 6),
                    "throughput": round(n / seconds / 1e6, 3),
                    "throughput_unit": "Mpoints/s",
                }
            )
    return cases


def _healpix_cases(device: torch.device) -> list[dict[str, object]]:
    import torchsky.sphere as sphere

    cases: list[dict[str, object]] = []
    for n in (10_000, 200_000):
        lon = torch.linspace(0.0, 359.0, n, dtype=torch.float64, device=device)
        lat = torch.linspace(-60.0, 60.0, n, dtype=torch.float64, device=device)
        seconds = _median_seconds(
            lambda: sphere.ang2pix(1024, lon, lat, nest=True, lonlat=True)
        )
        cases.append(
            {
                "group": "HEALPix",
                "case": "ang2pix",
                "parameters": f"n={n:,}, nside=1024, NESTED",
                "n": n,
                "median_s": round(seconds, 6),
                "throughput": round(n / seconds / 1e6, 3),
                "throughput_unit": "Mpoints/s",
            }
        )

    n = 50_000
    lon = torch.linspace(10.0, 20.0, n, dtype=torch.float64, device=device)
    lat = torch.linspace(-5.0, 5.0, n, dtype=torch.float64, device=device)
    seconds = _median_seconds(
        lambda: sphere.get_interp_weights(512, lon, lat, nest=True, lonlat=True)
    )
    cases.append(
        {
            "group": "HEALPix",
            "case": "get_interp_weights",
            "parameters": f"n={n:,}, nside=512, bilinear",
            "n": n,
            "median_s": round(seconds, 6),
            "throughput": round(n / seconds / 1e6, 3),
            "throughput_unit": "Mpoints/s",
        }
    )

    center = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64, device=device)
    disc = sphere.query_disc(512, center, radius=0.5, nest=True)
    seconds = _median_seconds(
        lambda: sphere.query_disc(512, center, radius=0.5, nest=True)
    )
    npix = int(disc.numel())
    cases.append(
        {
            "group": "HEALPix",
            "case": "query_disc",
            "parameters": "nside=512, radius=0.5 deg",
            "n": npix,
            "median_s": round(seconds, 6),
            "throughput": round(npix / seconds / 1e6, 3),
            "throughput_unit": "Mpixels/s",
        }
    )
    return cases


def _sht_cases(device: torch.device) -> list[dict[str, object]]:
    import torchsky.sphere as sphere

    cases: list[dict[str, object]] = []
    for nside in (32, 128):
        npix = 12 * nside**2
        healpix_map = torch.randn(npix, dtype=torch.float64, device=device)
        lmax = 2 * nside
        seconds = _median_seconds(
            lambda: sphere.anafast(healpix_map, lmax=lmax, pol=False), iters=2
        )
        cases.append(
            {
                "group": "Spherical harmonics",
                "case": "anafast",
                "parameters": f"nside={nside}, lmax={lmax}",
                "n": npix,
                "median_s": round(seconds, 6),
                "throughput": round(npix / seconds / 1e6, 3),
                "throughput_unit": "Mpixels/s",
            }
        )
    return cases


def _catalog_cases(device: torch.device) -> list[dict[str, object]]:
    from torchsky.catalogs import crossmatch_sky

    n = 20_000
    generator = torch.Generator(device="cpu").manual_seed(7)
    left_ra = torch.rand(n, generator=generator, dtype=torch.float64) * 10.0
    left_dec = torch.rand(n, generator=generator, dtype=torch.float64) * 10.0
    right_ra = left_ra + torch.randn(n, generator=generator, dtype=torch.float64) * 1e-4
    right_dec = (
        left_dec + torch.randn(n, generator=generator, dtype=torch.float64) * 1e-4
    )
    if device.type != "cpu":
        left_ra, left_dec = left_ra.to(device), left_dec.to(device)
        right_ra, right_dec = right_ra.to(device), right_dec.to(device)
    seconds = _median_seconds(
        lambda: crossmatch_sky(
            left_ra,
            left_dec,
            right_ra,
            right_dec,
            radius_arcsec=2.0,
            find="best",
        ),
        iters=2,
    )
    return [
        {
            "group": "Catalogs",
            "case": "crossmatch_sky (best)",
            "parameters": f"left={n:,}, right={n:,}, radius=2 arcsec",
            "n": n,
            "median_s": round(seconds, 6),
            "throughput": round(n / seconds / 1e6, 3),
            "throughput_unit": "Mleft rows/s",
        }
    ]


def _map_cases(device: torch.device) -> list[dict[str, object]]:
    from torchsky.map import Observation, coadd, plan

    cases: list[dict[str, object]] = []
    for size in (128, 512):
        header = _tan_header((size, size), crval=(150.0, 2.0), cdelt=0.001)
        observations = [
            Observation(
                data=torch.full((size, size), float(index + 1), dtype=torch.float32),
                wcs={**header, "CRVAL1": 150.0 + 0.0004 * index},
            )
            for index in range(4)
        ]
        target = plan(observations, projection="TAN")
        seconds = _median_seconds(
            lambda: coadd(observations, target, kernel="bilinear", device=str(device)),
            iters=2,
        )
        pixels = size * size * len(observations)
        cases.append(
            {
                "group": "Mapmaking",
                "case": "coadd (bilinear)",
                "parameters": f"{len(observations)} x {size}x{size}, TAN",
                "n": pixels,
                "median_s": round(seconds, 6),
                "throughput": round(pixels / seconds / 1e6, 3),
                "throughput_unit": "Minput pixels/s",
            }
        )
    return cases


def collect(device: torch.device) -> dict[str, object]:
    torch.manual_seed(0)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(0)
    cases: list[dict[str, object]] = []
    for group in (_wcs_cases, _healpix_cases, _sht_cases, _catalog_cases, _map_cases):
        print(f"measuring {group.__name__.strip('_')} ...", flush=True)
        cases.extend(group(device))
    return {
        "schema": 1,
        "generated_at": datetime.now(UTC).isoformat(timespec="seconds"),
        "environment": _environment(device),
        "cases": cases,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", default="cpu", help="cpu, cuda, or mps")
    parser.add_argument(
        "--write", action="store_true", help=f"Write {OUT.relative_to(ROOT)}"
    )
    args = parser.parse_args(argv)

    if str(ROOT / "src") not in sys.path:
        sys.path.insert(0, str(ROOT / "src"))
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    device = torch.device(args.device)
    payload = collect(device)
    text = json.dumps(payload, indent=2, sort_keys=False) + "\n"
    if args.write:
        OUT.parent.mkdir(parents=True, exist_ok=True)
        OUT.write_text(text, encoding="utf-8")
        print(f"wrote {OUT.relative_to(ROOT)} ({len(payload['cases'])} cases)")
    else:
        print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
