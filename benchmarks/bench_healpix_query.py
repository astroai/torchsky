#!/usr/bin/env python3
"""Benchmark HEALPix pixel queries (query_circle and query_polygon) vs healpy.

Timed sections measure torchsky **tensor** work only (device sync included); NumPy
materialization for parity checks happens **outside** the timed loop.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in sys.path
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR, ensure_src_on_path

ensure_src_on_path()

import argparse
import json
import time
from typing import Any, Callable

import healpy
import numpy as np
import torch

from torchsky.sphere.healpix import query_circle, query_polygon


def _to_numpy(x: Any) -> np.ndarray:
    if isinstance(x, torch.Tensor):
        return x.detach().cpu().numpy()
    return np.asarray(x)


def _resolve_device(choice: str) -> torch.device:
    if choice == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if choice == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but no CUDA device is available")
    if choice == "mps" and (
        not hasattr(torch.backends, "mps") or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS requested but no MPS device is available")
    return torch.device(choice)


def _sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device=device)
    elif device.type == "mps":
        torch.mps.synchronize()


def _time_many(
    fn: Callable[[], Any], runs: int, sync_device: torch.device | None = None
) -> float:
    fn()
    if sync_device is not None:
        _sync(sync_device)
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        if sync_device is not None:
            _sync(sync_device)
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=1024)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument(
        "--device", choices=["auto", "cpu", "cuda", "mps"], default="auto"
    )
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "healpix_query.json"
    )
    args = parser.parse_args()

    nside = args.nside
    device = _resolve_device(args.device)
    print(f"Device: {device.type} (NSIDE={nside})")

    # Parameters for circular query
    lon, lat = 120.0, 30.0
    radius = 1.0  # degree

    dtype = torch.float32 if device.type == "mps" else torch.float64
    poly_vertices = torch.tensor(
        [[120.0, 30.0], [121.0, 30.0], [120.5, 31.0]],
        dtype=dtype,
        device=device,
    )
    poly_vertices_np = poly_vertices.cpu().numpy()

    # Functions to benchmark
    # 1. query_circle (inclusive=True/False)
    def circle_tf():
        return query_circle(
            nside, lon, lat, radius, nest=True, degrees=True, inclusive=True
        )

    def circle_hp():
        return healpy.query_disc(
            nside,
            healpy.ang2vec(lon, lat, lonlat=True),
            np.radians(radius),
            nest=True,
            inclusive=True,
        )

    # 2. query_polygon
    def poly_tf():
        return query_polygon(nside, poly_vertices, nest=True, lonlat=True)

    # healpy query_polygon uses vectors
    poly_hp_vecs = [healpy.ang2vec(v[0], v[1], lonlat=True) for v in poly_vertices_np]

    def poly_hp():
        return healpy.query_polygon(nside, poly_hp_vecs, nest=True)

    ops = [
        ("query_circle", circle_tf, circle_hp),
        ("query_polygon", poly_tf, poly_hp),
    ]

    rows = []
    print(
        f"{'operation':>22s} {'ms (torchsky)':>15s} {'ms (healpy)':>12s} {'ratio':>8s} {'mismatches':>10s}"
    )
    for name, tf_fn, hp_fn in ops:
        t_tf = _time_many(tf_fn, runs=args.runs, sync_device=device)
        t_hp = _time_many(hp_fn, runs=args.runs, sync_device=None)

        # Correctness check
        res_tf = _to_numpy(tf_fn())
        res_hp = _to_numpy(hp_fn())
        mismatches = int(
            np.sum(~np.isin(res_tf, res_hp)) + np.sum(~np.isin(res_hp, res_tf))
        )

        ratio = t_hp / t_tf
        print(
            f"{name:>22s} {t_tf * 1000:15.3f} {t_hp * 1000:12.3f} {ratio:8.3f} {mismatches:10d}"
        )

        rows.append(
            {
                "operation": name,
                "nside": nside,
                "device": device.type,
                "time_ms_torchsky": t_tf * 1000,
                "time_ms_healpy": t_hp * 1000,
                "ratio_torchsky_vs_healpy": ratio,
                "mismatches": mismatches,
            }
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nJSON results written to {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
