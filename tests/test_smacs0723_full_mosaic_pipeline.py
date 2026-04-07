from __future__ import annotations

import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))

from smacs0723_full_mosaic_pipeline import _band_filename_contains, build_parser  # noqa: E402


def test_band_filename_contains_matches_long_wave_only() -> None:
    assert _band_filename_contains("F277W", "b") == "nrcblong"
    assert _band_filename_contains("F356W", "a") == "nrcalong"
    assert _band_filename_contains("F200W", "b") is None


def test_parser_defaults_are_simple() -> None:
    args = build_parser().parse_args([])
    assert str(args.root).endswith("/torchsky/data")
    assert args.input_product == "cal"
    assert args.train_model == "spectral_basis_field"
    assert args.train_steps == 12
