from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_reducer_streaming as reducer_bench  # noqa: E402


def test_reducer_scorecard_worker_reports_bounded_exact_scratch() -> None:
    for method in ("median", "sigma_clipped"):
        row = reducer_bench._worker(
            method=method,
            size=16,
            n_observations=5,
            working_memory_mib=1,
        )

        assert row["method"] == method
        assert row["device"] == "cpu"
        assert row["dtype"] == "float32"
        assert row["seed"] is None
        assert row["input_identity"] == "analytic_linear_plane_v1"
        assert row["output_pixels"] == 256
        assert row["scratch_required_bytes"] == 5 * 256 * (
            5 if method == "median" else 12
        )
        assert row["working_memory_limit_bytes"] == 1024 * 1024
        assert row["scratch_clean"] is True
        assert row["maximum_absolute_error"] < 1.0e-6
        assert row["peak_rss_bytes"] >= row["rss_before_bytes"]
