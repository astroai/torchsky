from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_reproduce import _variant_specs  # noqa: E402


def test_variant_specs_prefers_torchsky_only_by_default() -> None:
    rows = _variant_specs(
        include_drizzle=False,
        drizzle_kernels=("point", "square"),
        drizzle_pixfrac=1.0,
        only_drizzle=False,
    )
    assert rows
    assert all(row["backend"] == "torchsky" for row in rows)
    assert any(row["variant"] == "bilinear:flux:weighted_mean" for row in rows)
    assert any(row["variant"] == "drizzle_square:surface_brightness:weighted_mean:pixfrac=1.000" for row in rows)


def test_variant_specs_can_add_drizzle_rows() -> None:
    rows = _variant_specs(
        include_drizzle=True,
        drizzle_kernels=("point", "square"),
        drizzle_pixfrac=0.8,
        only_drizzle=False,
    )
    labels = [row["variant"] for row in rows]
    assert "drizzle:point:pixfrac=0.800" in labels
    assert "drizzle:square:pixfrac=0.800" in labels
    drizzle_rows = [row for row in rows if row["backend"] == "drizzle"]
    assert all(row["value_semantics"] == "surface_brightness" for row in drizzle_rows)


def test_variant_specs_can_run_drizzle_only() -> None:
    rows = _variant_specs(
        include_drizzle=True,
        drizzle_kernels=("point",),
        drizzle_pixfrac=1.0,
        only_drizzle=True,
    )
    assert [row["variant"] for row in rows] == ["drizzle:point:pixfrac=1.000"]
