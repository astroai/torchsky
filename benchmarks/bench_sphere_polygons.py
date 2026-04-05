#!/usr/bin/env python3
"""Benchmark spherical polygon primitives and optional comparator parity."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in sys.path so we can import benchmarks.config
repo_root = str(Path(__file__).resolve().parent.parent)
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
src_root = str(Path(__file__).resolve().parent.parent / "src")
if src_root not in sys.path:
    sys.path.insert(0, src_root)

from benchmarks.config import DEFAULT_OUTPUT_DIR  # noqa: E402


import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any, Callable  # noqa: E402

import numpy as np  # noqa: E402
import torch  # noqa: E402

import torchsky.sphere.geom as geom  # noqa: E402
from torchsky.sphere.geom import (  # noqa: E402
    SphericalPolygon,
    _numpy_sg_contains_lonlat_bulk,
    query_polygon_general,
    spherical_polygon_contains,
    spherical_polygons_intersect,
)

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    SGP = None

try:
    from spherical_geometry.tests.helpers import ROOT_DIR as SG_ROOT_DIR  # type: ignore
except Exception:  # pragma: no cover - optional comparator fixtures
    SG_ROOT_DIR = None


def _time_many(fn: Callable[[], Any], runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    return float(np.median(samples))


def _has_cpp_geom() -> bool:
    return getattr(geom, "_cpp", None) is not None and hasattr(
        geom._cpp, "_sphere_arc_intersections_matrix_cpu"
    )


def _with_cpp_mode(force_cpp: bool | None, fn: Callable[[], Any]) -> Any:
    saved_cpp = getattr(geom, "_cpp", None)
    saved_planar_min_ops = getattr(geom, "_PLANAR_PIP_CPP_MIN_OPS", None)
    try:
        if force_cpp is True and saved_planar_min_ops is not None:
            geom._PLANAR_PIP_CPP_MIN_OPS = 0
        elif force_cpp is False:
            geom._cpp = None
        return fn()
    finally:
        geom._cpp = saved_cpp
        if saved_planar_min_ops is not None:
            geom._PLANAR_PIP_CPP_MIN_OPS = saved_planar_min_ops


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


def _sg_polygon_from_lonlat(lon: np.ndarray, lat: np.ndarray):
    if SGP is None:
        return None
    if hasattr(SGP, "from_lonlat"):
        return SGP.from_lonlat(lon, lat, degrees=True)
    if hasattr(SGP, "from_radec"):
        return SGP.from_radec(lon, lat, degrees=True)
    return None


def _sg_contains(poly: Any, lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    if poly is None:
        raise RuntimeError("spherical_geometry comparator unavailable")
    try:
        return _numpy_sg_contains_lonlat_bulk(poly, lon, lat)
    except TypeError as exc:
        raise RuntimeError("unsupported spherical_geometry API for contains") from exc


def _xy_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lon = np.remainder(np.degrees(np.arctan2(y, x)), 360.0)
    lat = np.degrees(np.arcsin(np.clip(z / np.linalg.norm(xyz, axis=1), -1.0, 1.0)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _strip_closed(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _load_one_difficult_pair() -> tuple[torch.Tensor, ...] | None:
    if SG_ROOT_DIR is None:
        return None
    path = Path(SG_ROOT_DIR) / "difficult_intersections.txt"
    if not path.exists():
        return None
    raw = path.read_bytes().splitlines()
    if len(raw) < 4:
        return None

    def _decode(line: bytes) -> np.ndarray:
        return np.frombuffer(bytes.fromhex(line.decode("ascii")), dtype="<f8").reshape(
            (-1, 3)
        )

    a_points = _decode(raw[0])
    a_inside = _decode(raw[1])
    b_points = _decode(raw[2])
    b_inside = _decode(raw[3])
    lon_a, lat_a = _strip_closed(*_xy_to_lonlat(a_points))
    lon_b, lat_b = _strip_closed(*_xy_to_lonlat(b_points))
    inside_a_lon, inside_a_lat = _xy_to_lonlat(a_inside[:1])
    inside_b_lon, inside_b_lat = _xy_to_lonlat(b_inside[:1])
    return (
        torch.from_numpy(lon_a).to(torch.float64),
        torch.from_numpy(lat_a).to(torch.float64),
        torch.from_numpy(lon_b).to(torch.float64),
        torch.from_numpy(lat_b).to(torch.float64),
        torch.tensor(float(inside_a_lon[0]), dtype=torch.float64),
        torch.tensor(float(inside_a_lat[0]), dtype=torch.float64),
        torch.tensor(float(inside_b_lon[0]), dtype=torch.float64),
        torch.tensor(float(inside_b_lat[0]), dtype=torch.float64),
    )


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for row in rows:
        op = row.get("operation")
        if op == "contains_nonconvex_medium_default_vs_forced":
            best_variant = "cpp" if row["cpp_points_s"] > row["python_points_s"] else "python"
            out.append(
                {
                    "operation": "contains_nonconvex_recommendation",
                    "library": "torchsky_recommendation",
                    "profile": "medium_contains",
                    "recommended_variant": best_variant,
                    "default_variant": "threshold_gated",
                    "default_points_s": row["default_points_s"],
                    "recommended_points_s": row[f"{best_variant}_points_s"],
                    "cpp_threshold_ops": row["cpp_threshold_ops"],
                    "note": "Best explicit contains backend below the threshold-sensitive large-N regime",
                }
            )
        elif op == "contains_nonconvex_threshold_default_vs_cpp":
            out.append(
                {
                    "operation": "contains_nonconvex_recommendation",
                    "library": "torchsky_recommendation",
                    "profile": "large_contains",
                    "recommended_variant": "cpp",
                    "default_variant": "threshold_gated",
                    "default_points_s": row["default_points_s"],
                    "recommended_points_s": row["cpp_points_s"],
                    "cpp_threshold_ops": row["cpp_threshold_ops"],
                    "note": "Large-N contains should use the native planar point-in-polygon path",
                }
            )
        elif op == "intersects_nonconvex_cpp_vs_python":
            best_variant = "cpp" if row["cpp_calls_s"] >= row["python_calls_s"] else "python"
            out.append(
                {
                    "operation": "intersects_nonconvex_recommendation",
                    "library": "torchsky_recommendation",
                    "profile": "default_intersection",
                    "recommended_variant": best_variant,
                    "recommended_calls_s": row[f"{best_variant}_calls_s"],
                    "speedup_cpp_vs_python": row["speedup_cpp_vs_python"],
                    "note": "Preferred backend for polygon intersection checks on this machine",
                }
            )
        elif op == "intersects_upstream_difficult_cpp_vs_python":
            best_variant = "cpp" if row["cpp_calls_s"] >= row["python_calls_s"] else "python"
            out.append(
                {
                    "operation": "intersects_upstream_difficult_recommendation",
                    "library": "torchsky_recommendation",
                    "profile": "difficult_intersection",
                    "recommended_variant": best_variant,
                    "recommended_calls_s": row[f"{best_variant}_calls_s"],
                    "speedup_cpp_vs_python": row["speedup_cpp_vs_python"],
                    "note": "Preferred backend for upstream difficult polygon intersection cases",
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nside", type=int, default=512)
    parser.add_argument("--n-points", type=int, default=200_000)
    parser.add_argument("--runs", type=int, default=5)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument(
        "--json-out", type=Path, default=DEFAULT_OUTPUT_DIR / "sphere_polygons.json"
    )
    args = parser.parse_args()

    lon_poly, lat_poly = _star_polygon(center_lon=120.0, center_lat=-20.0, n=9)
    lon_poly_overlap, lat_poly_overlap = _star_polygon(
        center_lon=128.0, center_lat=-18.0, n=9
    )
    lon_poly_far, lat_poly_far = _star_polygon(center_lon=210.0, center_lat=25.0, n=9)
    lon_q, lat_q = _random_lonlat(args.n_points, args.seed)

    lon_q_t = torch.from_numpy(lon_q).to(torch.float64)
    lat_q_t = torch.from_numpy(lat_q).to(torch.float64)
    lon_poly_t = torch.from_numpy(lon_poly).to(torch.float64)
    lat_poly_t = torch.from_numpy(lat_poly).to(torch.float64)
    lon_poly_overlap_t = torch.from_numpy(lon_poly_overlap).to(torch.float64)
    lat_poly_overlap_t = torch.from_numpy(lat_poly_overlap).to(torch.float64)
    lon_poly_far_t = torch.from_numpy(lon_poly_far).to(torch.float64)
    lat_poly_far_t = torch.from_numpy(lat_poly_far).to(torch.float64)

    poly = SphericalPolygon(lon_poly_t, lat_poly_t)

    rows: list[dict[str, Any]] = []
    medium_points = min(args.n_points, 50_000)
    medium_lon_q_t = lon_q_t[:medium_points]
    medium_lat_q_t = lat_q_t[:medium_points]
    poly_ops_threshold = getattr(geom, "_PLANAR_PIP_CPP_MIN_OPS", 2_000_000)
    threshold_points = max(
        args.n_points,
        int(np.ceil(float(poly_ops_threshold) / float(lon_poly_t.numel()))) + 1,
    )
    if threshold_points != args.n_points:
        lon_q_large, lat_q_large = _random_lonlat(threshold_points, args.seed + 17)
        large_lon_q_t = torch.from_numpy(lon_q_large).to(torch.float64)
        large_lat_q_t = torch.from_numpy(lat_q_large).to(torch.float64)
    else:
        large_lon_q_t = lon_q_t
        large_lat_q_t = lat_q_t

    t_contains = _time_many(
        lambda: spherical_polygon_contains(
            lon_q_t,
            lat_q_t,
            lon_poly_t,
            lat_poly_t,
            inclusive=False,
        ),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "contains_nonconvex",
            "library": "torchsky",
            "n_points": args.n_points,
            "points_s": args.n_points / t_contains,
        }
    )
    if _has_cpp_geom():
        t_contains_medium_default = _time_many(
            lambda: spherical_polygon_contains(
                medium_lon_q_t,
                medium_lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=False,
            ),
            runs=args.runs,
        )
        t_contains_medium_cpp = _time_many(
            lambda: _with_cpp_mode(
                True,
                lambda: spherical_polygon_contains(
                    medium_lon_q_t,
                    medium_lat_q_t,
                    lon_poly_t,
                    lat_poly_t,
                    inclusive=False,
                ),
            ),
            runs=args.runs,
        )
        t_contains_medium_py = _time_many(
            lambda: _with_cpp_mode(
                False,
                lambda: spherical_polygon_contains(
                    medium_lon_q_t,
                    medium_lat_q_t,
                    lon_poly_t,
                    lat_poly_t,
                    inclusive=False,
                ),
            ),
            runs=args.runs,
        )
        rows.append(
            {
                "operation": "contains_nonconvex_medium_default_vs_forced",
                "library": "torchsky",
                "n_points": medium_points,
                "ops": int(medium_points * lon_poly_t.numel()),
                "cpp_threshold_ops": int(poly_ops_threshold),
                "default_points_s": medium_points / t_contains_medium_default,
                "cpp_points_s": medium_points / t_contains_medium_cpp,
                "python_points_s": medium_points / t_contains_medium_py,
                "default_vs_cpp_ratio": t_contains_medium_cpp / t_contains_medium_default,
                "default_vs_python_ratio": t_contains_medium_py / t_contains_medium_default,
            }
        )
        t_contains_large_default = _time_many(
            lambda: spherical_polygon_contains(
                large_lon_q_t,
                large_lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=False,
            ),
            runs=args.runs,
        )
        t_contains_large_cpp = _time_many(
            lambda: _with_cpp_mode(
                True,
                lambda: spherical_polygon_contains(
                    large_lon_q_t,
                    large_lat_q_t,
                    lon_poly_t,
                    lat_poly_t,
                    inclusive=False,
                ),
            ),
            runs=args.runs,
        )
        rows.append(
            {
                "operation": "contains_nonconvex_threshold_default_vs_cpp",
                "library": "torchsky",
                "n_points": int(large_lon_q_t.numel()),
                "ops": int(large_lon_q_t.numel() * lon_poly_t.numel()),
                "cpp_threshold_ops": int(poly_ops_threshold),
                "default_points_s": large_lon_q_t.numel() / t_contains_large_default,
                "cpp_points_s": large_lon_q_t.numel() / t_contains_large_cpp,
                "default_vs_cpp_ratio": t_contains_large_cpp / t_contains_large_default,
            }
        )
    if _has_cpp_geom():
        t_contains_cpp = _time_many(
            lambda: _with_cpp_mode(
                True,
                lambda: spherical_polygon_contains(
                    lon_q_t,
                    lat_q_t,
                    lon_poly_t,
                    lat_poly_t,
                    inclusive=False,
                ),
            ),
            runs=args.runs,
        )
        t_contains_py = _time_many(
            lambda: _with_cpp_mode(
                False,
                lambda: spherical_polygon_contains(
                    lon_q_t,
                    lat_q_t,
                    lon_poly_t,
                    lat_poly_t,
                    inclusive=False,
                ),
            ),
            runs=args.runs,
        )
        rows.append(
            {
                "operation": "contains_nonconvex_cpp_vs_python",
                "library": "torchsky",
                "n_points": args.n_points,
                "cpp_points_s": args.n_points / t_contains_cpp,
                "python_points_s": args.n_points / t_contains_py,
                "speedup_cpp_vs_python": t_contains_py / t_contains_cpp,
            }
        )

    t_pixel_query = _time_many(
        lambda: query_polygon_general(args.nside, lon_poly_t, lat_poly_t, nest=False),
        runs=args.runs,
    )
    rows.append(
        {
            "operation": "query_polygon_nonconvex",
            "library": "torchsky",
            "nside": args.nside,
            "queries_s": 1.0 / t_pixel_query,
        }
    )

    def _run_intersections() -> tuple[bool, bool]:
        overlap = False
        far = False
        for _ in range(32):
            overlap = spherical_polygons_intersect(
                lon_poly_t,
                lat_poly_t,
                lon_poly_overlap_t,
                lat_poly_overlap_t,
            )
            far = spherical_polygons_intersect(
                lon_poly_t,
                lat_poly_t,
                lon_poly_far_t,
                lat_poly_far_t,
            )
        return overlap, far

    t_intersects = _time_many(_run_intersections, runs=args.runs)
    rows.append(
        {
            "operation": "intersects_nonconvex",
            "library": "torchsky",
            "calls_s": 64.0 / t_intersects,
            "overlap_expected": True,
            "far_expected": False,
        }
    )
    if _has_cpp_geom():
        t_intersects_cpp = _time_many(
            lambda: _with_cpp_mode(True, _run_intersections),
            runs=args.runs,
        )
        t_intersects_py = _time_many(
            lambda: _with_cpp_mode(False, _run_intersections),
            runs=args.runs,
        )
        rows.append(
            {
                "operation": "intersects_nonconvex_cpp_vs_python",
                "library": "torchsky",
                "cpp_calls_s": 64.0 / t_intersects_cpp,
                "python_calls_s": 64.0 / t_intersects_py,
                "speedup_cpp_vs_python": t_intersects_py / t_intersects_cpp,
            }
        )

    difficult_pair = _load_one_difficult_pair()
    if difficult_pair is not None:
        (
            lon_a_t,
            lat_a_t,
            lon_b_t,
            lat_b_t,
            inside_a_lon_t,
            inside_a_lat_t,
            inside_b_lon_t,
            inside_b_lat_t,
        ) = difficult_pair

        def _run_difficult_intersection() -> bool:
            out = False
            for _ in range(32):
                out = spherical_polygons_intersect(
                    lon_a_t,
                    lat_a_t,
                    lon_b_t,
                    lat_b_t,
                    inside1_lon_deg=float(inside_a_lon_t),
                    inside1_lat_deg=float(inside_a_lat_t),
                    inside2_lon_deg=float(inside_b_lon_t),
                    inside2_lat_deg=float(inside_b_lat_t),
                )
            return out

        t_difficult = _time_many(_run_difficult_intersection, runs=args.runs)
        rows.append(
            {
                "operation": "intersects_upstream_difficult",
                "library": "torchsky",
                "calls_s": 32.0 / t_difficult,
                "expected_nonempty": True,
            }
        )
        if _has_cpp_geom():
            t_difficult_cpp = _time_many(
                lambda: _with_cpp_mode(True, _run_difficult_intersection),
                runs=args.runs,
            )
            t_difficult_py = _time_many(
                lambda: _with_cpp_mode(False, _run_difficult_intersection),
                runs=args.runs,
            )
            rows.append(
                {
                    "operation": "intersects_upstream_difficult_cpp_vs_python",
                    "library": "torchsky",
                    "cpp_calls_s": 32.0 / t_difficult_cpp,
                    "python_calls_s": 32.0 / t_difficult_py,
                    "speedup_cpp_vs_python": t_difficult_py / t_difficult_cpp,
                }
            )

    t_area = _time_many(lambda: poly.area(degrees=False), runs=args.runs)
    rows.append(
        {
            "operation": "area_nonconvex",
            "library": "torchsky",
            "calls_s": 1.0 / t_area,
            "area_sr": float(poly.area(degrees=False)),
        }
    )

    sg_poly = _sg_polygon_from_lonlat(lon_poly, lat_poly)
    if sg_poly is not None:
        t_sg_contains = _time_many(
            lambda: _sg_contains(sg_poly, lon_q, lat_q), runs=max(2, args.runs // 2)
        )
        tf_contains = (
            spherical_polygon_contains(
                lon_q_t,
                lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=False,
            )
            .cpu()
            .numpy()
        )
        tf_contains_inclusive = (
            spherical_polygon_contains(
                lon_q_t,
                lat_q_t,
                lon_poly_t,
                lat_poly_t,
                inclusive=True,
                atol_deg=1e-5,
            )
            .cpu()
            .numpy()
        )
        sg_contains = _sg_contains(sg_poly, lon_q, lat_q)
        mismatches = int(np.sum(tf_contains != sg_contains))
        boundary_like = tf_contains_inclusive != tf_contains
        mismatches_nonboundary = int(
            np.sum((tf_contains != sg_contains) & (~boundary_like))
        )

        sg_area = float(sg_poly.area()) if hasattr(sg_poly, "area") else float("nan")
        rel_area_err = abs(float(poly.area(degrees=False)) - sg_area) / max(
            abs(sg_area), 1e-15
        )

        rows.append(
            {
                "operation": "contains_nonconvex",
                "library": "spherical-geometry",
                "n_points": args.n_points,
                "points_s": args.n_points / t_sg_contains,
                "ratio_torchsky_vs_spherical_geometry": t_sg_contains / t_contains,
                "mismatches_vs_spherical_geometry": mismatches,
                "mismatches_vs_spherical_geometry_nonboundary": mismatches_nonboundary,
            }
        )
        rows.append(
            {
                "operation": "area_nonconvex",
                "library": "spherical-geometry",
                "area_sr": sg_area,
                "relative_error_vs_torchsky": rel_area_err,
            }
        )

    rows.extend(_recommendation_rows(rows))

    for row in rows:
        print(", ".join(f"{k}={v}" for k, v in row.items()))

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    print(f"\nJSON: {args.json_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
