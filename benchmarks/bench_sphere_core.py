#!/usr/bin/env python3
"""Benchmark torchsky spherical-core primitives."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in sys.path so we can import benchmarks.config
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

from torchsky.sphere.core import pairwise_angular_distance, sample_multiband_healpix  # noqa: E402
from torchsky.sphere.geom import query_ellipse  # noqa: E402

try:
    import healpy
except Exception:  # pragma: no cover - optional comparator
    healpy = None


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but unavailable")
    if choice == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but unavailable")
    return torch.device(choice)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _bench(
    fn: Callable[[], Any], runs: int, device: torch.device | None = None
) -> float:
    fn()
    if device is not None:
        _sync(device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if device is not None:
            _sync(device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _sample_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon, lat


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=256)
    parser.add_argument("--n-points", type=int, default=100_000)
    parser.add_argument("--n-bands", type=int, default=8)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="cpu"
    )
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "sphere_core.json"
    )
    args = parser.parse_args()

    device = _resolve_device(args.device)
    f_dtype = torch.float32 if device.type == "mps" else torch.float64

    npix = 12 * args.nside * args.nside
    lon_np, lat_np = _sample_lonlat(args.n_points, args.seed)
    lon_t = torch.from_numpy(lon_np).to(device=device, dtype=f_dtype)
    lat_t = torch.from_numpy(lat_np).to(device=device, dtype=f_dtype)

    rng = np.random.default_rng(args.seed + 1)
    cube_np = rng.normal(size=(args.n_bands, npix)).astype(np.float64)
    cube_t = torch.from_numpy(cube_np).to(device=device, dtype=f_dtype)

    results: list[dict[str, Any]] = []

    t_bilinear = _bench(
        lambda: sample_multiband_healpix(
            cube_t,
            args.nside,
            lon_t,
            lat_t,
            interpolation="bilinear",
        ),
        runs=args.runs,
        device=device,
    )
    results.append(
        {
            "operation": "sample_multiband_bilinear",
            "device": device.type,
            "nside": args.nside,
            "n_points": args.n_points,
            "n_bands": args.n_bands,
            "m_samples_s": (args.n_points * args.n_bands) / t_bilinear / 1e6,
        }
    )

    t_nearest = _bench(
        lambda: sample_multiband_healpix(
            cube_t,
            args.nside,
            lon_t,
            lat_t,
            interpolation="nearest",
        ),
        runs=args.runs,
        device=device,
    )
    results.append(
        {
            "operation": "sample_multiband_nearest",
            "device": device.type,
            "nside": args.nside,
            "n_points": args.n_points,
            "n_bands": args.n_bands,
            "m_samples_s": (args.n_points * args.n_bands) / t_nearest / 1e6,
        }
    )

    # Pairwise is O(N^2); keep N small for benchmark sanity.
    pair_n = min(8_000, args.n_points)
    lon_small = lon_t[:pair_n]
    lat_small = lat_t[:pair_n]
    t_pairwise = _bench(
        lambda: pairwise_angular_distance(lon_small, lat_small),
        runs=max(2, args.runs // 2),
        device=device,
    )
    results.append(
        {
            "operation": "pairwise_angular_distance",
            "device": device.type,
            "n_points": pair_n,
            "m_pairwise_s": (pair_n * pair_n) / t_pairwise / 1e6,
        }
    )

    t_query = _bench(
        lambda: query_ellipse(
            args.nside, 125.0, -30.0, 7.0, 2.0, pa_deg=15.0, nest=False
        ),
        runs=args.runs,
        device=torch.device("cpu"),
    )
    results.append(
        {
            "operation": "query_ellipse",
            "device": "cpu",
            "nside": args.nside,
            "queries_s": 1.0 / t_query,
        }
    )

    if healpy is not None and device.type == "cpu":

        def _healpy_loop_interp() -> None:
            for i in range(args.n_bands):
                healpy.get_interp_val(
                    cube_np[i], lon_np, lat_np, lonlat=True, nest=False
                )

        t_hp = _bench(_healpy_loop_interp, runs=max(2, args.runs // 2), device=None)
        results.append(
            {
                "operation": "healpy_loop_interp_baseline",
                "device": "cpu",
                "nside": args.nside,
                "n_points": args.n_points,
                "n_bands": args.n_bands,
                "m_samples_s": (args.n_points * args.n_bands) / t_hp / 1e6,
                "ratio_torchsky_bilinear_vs_healpy_loop": t_hp / t_bilinear,
            }
        )

    try:
        import astropy.units as u
        from mocpy import MOC as MocpyMOC
        from torchsky.sphere.moc import MOC as TorchMOC
    except Exception:
        MocpyMOC = None
        TorchMOC = None
        u = None
    if MocpyMOC is not None and TorchMOC is not None and u is not None:
        moc = TorchMOC.from_circle(20.0, -10.0, 5.0, max_order=6)
        moc_lon = lon_t[: min(50_000, args.n_points)].cpu()
        moc_lat = lat_t[: min(50_000, args.n_points)].cpu()
        t_ts = _bench(
            lambda: moc.contains(moc_lon, moc_lat),
            runs=args.runs,
            device=torch.device("cpu"),
        )
        results.append(
            {
                "operation": "moc_contains",
                "library": "torchsky",
                "n_points": int(moc_lon.numel()),
                "contains_s": t_ts,
            }
        )
        if hasattr(MocpyMOC, "from_cone"):
            ref = MocpyMOC.from_cone(
                lon=20.0 * u.deg, lat=-10.0 * u.deg, radius=5.0 * u.deg, max_depth=6
            )
            lon_u = moc_lon.numpy() * u.deg
            lat_u = moc_lat.numpy() * u.deg
            contains_fn = getattr(ref, "contains_lonlat", getattr(ref, "contains", None))
            if contains_fn is not None:
                t_ref = _bench(
                    lambda: contains_fn(lon_u, lat_u),
                    runs=args.runs,
                    device=None,
                )
                results.append(
                    {
                        "operation": "moc_contains",
                        "library": "mocpy",
                        "n_points": int(moc_lon.numel()),
                        "contains_s": t_ref,
                        "ratio_torchsky_vs_mocpy": t_ref / t_ts,
                    }
                )

    for row in results:
        line = ", ".join(f"{k}={v}" for k, v in row.items())
        print(line)

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
