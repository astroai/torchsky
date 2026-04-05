#!/usr/bin/env python3
"""Benchmark bulk ``(lon, lat)`` containment: ``ExactSphericalRegion`` vs tensor path.

**Comparators**

- **`spherical_geometry`**: same vectorized ``contains_lonlat`` / ``contains_radec`` bulk
  path used in ``bench_sphere_polygons`` (implemented as
  ``torchsky.sphere.geom._numpy_sg_contains_lonlat_bulk``, mirroring upstream
  `spherical_geometry.polygon.SphericalPolygon` semantics).
- **torchsky** ``spherical_polygon_contains``: differentiable / native planar pipeline
  (already covered in ``bench_sphere_polygons``; repeated here for ratio vs exact backend).

**Other ecosystem libs** (`hpgeom`, `healpy`) generally target HEALPix indexing rather than
identical N-point lon/lat-in-polygon queries; this harness emits explicit **SKIPPED** rows
when no comparable API is found so optimization loops can track “not applicable” vs
“missing install”.

**Regression gate:** pass ``--baseline-json`` (see ``benchmarks/baselines/sphere_geom_exact_contains_gate.json``).
With ``spherical-geometry`` installed, parity (**zero mismatches**) is always enforced when
present in the baseline. Throughput floors (**min_median_points_s**) are **opt-in** per
baseline row via ``skip_points_s_regression: false`` so CI stays quiet until you pin hardware.
If ``spherical-geometry`` is missing, the script exits 0 and the gate is a no-op.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky.sphere.geom as geom  # noqa: E402
from torchsky.sphere.geom import (  # noqa: E402
    SphericalPolygon,
    _numpy_sg_contains_lonlat_bulk,
    spherical_polygon_contains,
    to_exact_region,
)


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _star_polygon(
    center_lon: float, center_lat: float, n: int = 9
) -> tuple[np.ndarray, np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n, endpoint=False)
    radii = np.where((np.arange(n) % 2) == 0, 12.0, 5.0)
    lon = center_lon + radii * np.cos(angles) / max(
        np.cos(np.deg2rad(center_lat)), 0.25
    )
    lat = center_lat + radii * np.sin(angles)
    lon = (lon + 360.0) % 360.0
    lat = np.clip(lat, -85.0, 85.0)
    return lon.astype(np.float64), lat.astype(np.float64)


def _random_lonlat(n: int, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    lon = rng.uniform(0.0, 360.0, size=n)
    lat = np.degrees(np.arcsin(rng.uniform(-1.0, 1.0, size=n)))
    return lon, lat


def _optional_import(name: str) -> Any:
    try:
        return importlib.import_module(name)
    except Exception:  # pragma: no cover - bench-only
        return None


def _bench_row_key(row: Mapping[str, Any]) -> str:
    return f"{row.get('operation')}|{row.get('library')}"


def check_sphere_geom_exact_contains_baseline(
    current_rows: Sequence[dict[str, Any]],
    baseline_rows: Sequence[dict[str, Any]],
    *,
    points_s_tolerance: float = 0.35,
) -> list[str]:
    """Return human-readable regression errors; empty list means pass."""
    errors: list[str] = []
    if not any(
        r.get("operation") == "exact_contains_parity_numpy_bulk" and r.get("status") == "OK"
        for r in current_rows
    ):
        return errors

    cur = {_bench_row_key(r): r for r in current_rows}
    for base in baseline_rows:
        op = str(base.get("operation", ""))
        lib = str(base.get("library", ""))
        key = f"{op}|{lib}"
        row = cur.get(key)
        if row is None:
            errors.append(f"baseline expects row {key!r} but it is missing from current output")
            continue
        if str(row.get("status", "")) != str(base.get("status", "OK")):
            errors.append(
                f"{key}: status {row.get('status')!r} != baseline {base.get('status')!r}"
            )
            continue
        max_mm = base.get("max_mismatches")
        if max_mm is not None and op == "exact_contains_parity_numpy_bulk":
            got = int(row.get("mismatches", -1))
            if got > int(max_mm):
                errors.append(
                    f"{key}: mismatches {got} exceed baseline max_mismatches {max_mm}"
                )
        if bool(base.get("skip_points_s_regression", True)):
            continue
        floor = base.get("min_median_points_s")
        if floor is None:
            continue
        got_ps = row.get("points_s")
        if got_ps is None:
            errors.append(f"{key}: baseline expects points_s but current row has none")
            continue
        got_ps_f = float(got_ps)
        floor_f = float(floor) * (1.0 - float(points_s_tolerance))
        if got_ps_f < floor_f:
            errors.append(
                f"{key}: points/s {got_ps_f:.6g} below floor "
                f"{floor_f:.6g} (baseline min {float(floor):.6g} with tolerance {points_s_tolerance:.0%})"
            )
    return errors


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=None,
    )
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline list JSON; enforces parity and optional throughput floors",
    )
    parser.add_argument(
        "--points-s-tolerance",
        type=float,
        default=0.35,
        help="Allowed fractional drop vs min_median_points_s when throughput regression is on",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []

    lon_poly, lat_poly = _star_polygon(120.0, -20.0, 9)
    lon_q, lat_q = _random_lonlat(args.n_points, args.seed)
    lon_t = torch.from_numpy(lon_poly).to(torch.float64)
    lat_t = torch.from_numpy(lat_poly).to(torch.float64)
    lon_q_t = torch.from_numpy(lon_q).to(torch.float64)
    lat_q_t = torch.from_numpy(lat_q).to(torch.float64)

    if not geom._spherical_geometry_available():
        rows.append(
            {
                "operation": "exact_region_contains",
                "library": "torchsky",
                "status": "SKIP",
                "skip_reason": "spherical-geometry not installed",
                "n_points": args.n_points,
            }
        )
        _write_out(args.json_out, rows)
        return 0

    exact = to_exact_region(
        SphericalPolygon(lon_t, lat_t),
        backend="spherical-geometry",
        cap_steps=256,
    )
    sg = exact._sg_poly

    # Parity: ExactSphericalRegion vs raw numpy bulk (must agree).
    mask_exact = exact.contains(lon_q_t, lat_q_t).numpy().reshape(-1)
    mask_np = _numpy_sg_contains_lonlat_bulk(sg, lon_q, lat_q)
    n_mismatch = int(np.sum(mask_exact != mask_np))
    rows.append(
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "n_points": args.n_points,
            "mismatches": n_mismatch,
            "status": "OK" if n_mismatch == 0 else "FAILED",
        }
    )
    if n_mismatch != 0:
        _write_out(args.json_out, rows)
        return 1

    t_exact = _time_many(
        lambda: exact.contains(lon_q_t, lat_q_t),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "n_points": args.n_points,
            "points_s": args.n_points / t_exact,
            "status": "OK",
        }
    )

    t_sg = _time_many(
        lambda: _numpy_sg_contains_lonlat_bulk(sg, lon_q, lat_q),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "exact_region_contains_numpy_bulk_reference",
            "library": "spherical_geometry",
            "n_points": args.n_points,
            "points_s": args.n_points / t_sg,
            "ratio_reference_vs_torchsky_wrap": t_exact / t_sg if t_sg > 0 else None,
            "status": "OK",
        }
    )

    t_tensor = _time_many(
        lambda: spherical_polygon_contains(
            lon_q_t,
            lat_q_t,
            lon_t,
            lat_t,
            inclusive=False,
        ),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "spherical_polygon_contains_nonconvex",
            "library": "torchsky",
            "n_points": args.n_points,
            "points_s": args.n_points / t_tensor,
            "ratio_tensor_contains_vs_exact_region": t_exact / t_tensor
            if t_tensor > 0
            else None,
            "status": "OK",
        }
    )

    hpgeom = _optional_import("hpgeom")
    if hpgeom is None:
        rows.append(
            {
                "operation": "polygon_lonlat_contains_N_points",
                "library": "hpgeom",
                "status": "SKIP",
                "skip_reason": "hpgeom not installed",
            }
        )
    else:
        rows.append(
            {
                "operation": "polygon_lonlat_contains_N_points",
                "library": "hpgeom",
                "status": "SKIP",
                "skip_reason": "no comparable bulk lon/lat-in-polygon entry point in this harness",
            }
        )

    healpy = _optional_import("healpy")
    if healpy is None:
        rows.append(
            {
                "operation": "polygon_lonlat_contains_N_points",
                "library": "healpy",
                "status": "SKIP",
                "skip_reason": "healpy not installed",
            }
        )
    else:
        rows.append(
            {
                "operation": "polygon_lonlat_contains_N_points",
                "library": "healpy",
                "status": "SKIP",
                "skip_reason": "healpy is HEALPix-centric; use bench_sphere_geometry.py for ang2pix parity",
            }
        )

    _write_out(args.json_out, rows)

    if args.baseline_json is not None and args.baseline_json.is_file():
        baseline = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        if not isinstance(baseline, list):
            print("baseline-json must be a JSON list", file=sys.stderr)
            return 1
        errs = check_sphere_geom_exact_contains_baseline(
            rows,
            baseline,
            points_s_tolerance=args.points_s_tolerance,
        )
        for msg in errs:
            print(msg, file=sys.stderr)
        if errs:
            return 1
    elif args.baseline_json is not None:
        print(f"baseline-json not found: {args.baseline_json}", file=sys.stderr)
        return 1

    return 0


def _write_out(path: Path | None, rows: list[dict[str, Any]]) -> None:
    if path is None:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(rows, indent=2), encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
