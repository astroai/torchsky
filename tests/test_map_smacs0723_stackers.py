from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_compat import _as_fits_header
from bench_map_smacs0723_stackers import _build_method_specs, _compute_region_metrics


def test_build_method_specs_includes_linearized_legacy_targets() -> None:
    specs = _build_method_specs(include_inr=True)
    by_name = {spec["name"]: spec for spec in specs}
    assert by_name["swarp_bilinear"]["target_mode"] == "published_swarp_tan"
    assert by_name["montage"]["target_mode"] == "published_linearized"
    assert by_name["montage_tan"]["target_mode"] == "published_swarp_tan"
    assert by_name["torchsky_drizzle_square"]["target_mode"] == "published_native"
    assert by_name["torchsky_drizzle_point"]["target_mode"] == "published_native"
    assert by_name["reproject_exact"]["family"] == "reproject"
    assert by_name["reproject_adaptive"]["family"] == "reproject"
    assert by_name["inr_operator_constrained_feature_field"]["family"] == "inr"


def test_compute_region_metrics_reports_affine_and_local_terms() -> None:
    reference = np.ones((8, 8), dtype=np.float32)
    prediction = reference.copy()
    prediction[0, 0] = 2.0
    support = np.ones_like(reference, dtype=bool)

    metrics = _compute_region_metrics(prediction, reference, support)

    assert metrics["support_rmse"] > 0.0
    assert metrics["uniform_rmse"] > 0.0
    assert "center_support_rmse" in metrics
    assert "edge_support_rmse" in metrics
    assert "support_rmse_after_affine_fit" in metrics


def test_as_fits_header_sanitizes_multiline_non_ascii_strings() -> None:
    header = _as_fits_header(
        {
            "NAXIS": 2,
            "NAXIS1": 4,
            "NAXIS2": 4,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 2.0,
            "CRPIX2": 2.0,
            "BADTXT": "line1\nline2\u2603",
            "COMMENT": "\nignore me",
        }
    )
    assert header["BADTXT"] == "line1 line2"
    assert "COMMENT" not in header
