"""Unit tests for ``bench_sphere_geom_exact_contains`` baseline matching (no optional deps)."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from benchmarks.bench_sphere_geom_exact_contains import (  # noqa: E402
    check_sphere_geom_exact_contains_baseline,
)


def test_baseline_check_noop_when_bench_did_not_run_exact_path() -> None:
    current = [
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "status": "SKIP",
            "skip_reason": "spherical-geometry not installed",
        }
    ]
    baseline = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "max_mismatches": 0,
        }
    ]
    assert check_sphere_geom_exact_contains_baseline(current, baseline) == []


def test_baseline_parity_mismatch_errors() -> None:
    current = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "mismatches": 3,
        }
    ]
    baseline = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "max_mismatches": 0,
        }
    ]
    errs = check_sphere_geom_exact_contains_baseline(current, baseline)
    assert len(errs) == 1
    assert "mismatches" in errs[0]


def test_baseline_points_s_floor_when_enabled() -> None:
    current = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "mismatches": 0,
        },
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "status": "OK",
            "points_s": 100_000.0,
        },
    ]
    baseline = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "max_mismatches": 0,
        },
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "status": "OK",
            "min_median_points_s": 200_000.0,
            "skip_points_s_regression": False,
        },
    ]
    errs = check_sphere_geom_exact_contains_baseline(
        current, baseline, points_s_tolerance=0.0
    )
    assert len(errs) == 1
    assert "points/s" in errs[0]


def test_baseline_points_s_floor_passes_with_tolerance() -> None:
    current = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "mismatches": 0,
        },
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "status": "OK",
            "points_s": 100_000.0,
        },
    ]
    baseline = [
        {
            "operation": "exact_contains_parity_numpy_bulk",
            "library": "torchsky",
            "status": "OK",
            "max_mismatches": 0,
        },
        {
            "operation": "exact_region_contains",
            "library": "torchsky",
            "status": "OK",
            "min_median_points_s": 100_000.0,
            "skip_points_s_regression": False,
        },
    ]
    assert (
        check_sphere_geom_exact_contains_baseline(
            current, baseline, points_s_tolerance=0.35
        )
        == []
    )
