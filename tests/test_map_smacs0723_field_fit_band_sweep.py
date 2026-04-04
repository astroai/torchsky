from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_field_fit_band_sweep import _format_markdown  # noqa: E402


def test_band_sweep_markdown_lists_band_and_model() -> None:
    text = _format_markdown(
        [
            {
                "band": "F277W",
                "model": "sky_encoder_resmlp",
                "status": "OK",
                "support_rmse": 0.1,
                "aligned_rmse": 0.09,
                "affine_slope": 0.95,
                "uniform_rmse": 0.2,
                "support_flux_ratio": 1.01,
                "support_frac": 0.9,
                "fit_time_ms": 12.3,
            }
        ]
    )
    assert "F277W" in text
    assert "sky_encoder_resmlp" in text


def test_band_sweep_source_mentions_best_overlap_strategy() -> None:
    source = (BENCHMARKS / "bench_map_smacs0723_field_fit_band_sweep.py").read_text()
    assert "--crop-strategy" in source
    assert "best_overlap" in source
