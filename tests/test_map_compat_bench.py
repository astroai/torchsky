from __future__ import annotations

import sys
from pathlib import Path

from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_compat as bench_map_compat  # noqa: E402
import run_ground_truth_competitive_large_union as competitive_runner  # noqa: E402


def test_montage_header_template_includes_core_fits_cards(tmp_path):
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = 32
    header["NAXIS2"] = 24
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CRPIX1"] = 16.0
    header["CRPIX2"] = 12.0
    header["CRVAL1"] = 11.0
    header["CRVAL2"] = -4.0
    header["CD1_1"] = -0.01
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.01

    path = tmp_path / "target.hdr"
    bench_map_compat._write_header_template(path, dict(header))
    text = path.read_text()

    assert "SIMPLE" in text
    assert "BITPIX" in text
    assert "NAXIS   =                    2" in text


def test_competitive_runner_uses_rectangular_extent_for_case_size() -> None:
    assert competitive_runner._effective_case_size(512, None, None) == 512
    assert competitive_runner._effective_case_size(512, 1024, 768) == 1024
    assert competitive_runner._effective_case_size(512, 2048, 1536) == 2048
