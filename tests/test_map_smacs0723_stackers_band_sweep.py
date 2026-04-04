from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_stackers_band_sweep import (  # noqa: E402
    _format_markdown,
    _recommended_filename_contains,
)


def test_recommended_filename_contains_matches_long_band_module_b() -> None:
    assert _recommended_filename_contains("F277W", "b") == "nrcblong"
    assert _recommended_filename_contains("F150W", "b") == ""


def test_multiband_stacker_markdown_lists_band_and_variant() -> None:
    text = _format_markdown(
        [
            {
                "band": "F277W",
                "variant": "torchsky:drizzle_square:surface_brightness:pixfrac=1.000",
                "family": "torchsky",
                "status": "OK",
                "n_exposures": 9,
                "support_rmse": 0.1,
                "uniform_rmse": 0.2,
                "support_flux_ratio": 1.01,
                "time_ms": 12.3,
            }
        ]
    )
    assert "F277W" in text
    assert "torchsky:drizzle_square" in text
