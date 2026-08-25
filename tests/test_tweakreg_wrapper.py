"""Tests for scripts/tweakreg_align_exposures.py pure helpers and guards."""

from __future__ import annotations

import fnmatch
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
from astropy.io import fits

SCRIPTS_DIR = Path(__file__).resolve().parents[1] / "scripts"
SCRIPT_PATH = SCRIPTS_DIR / "tweakreg_align_exposures.py"

_spec = importlib.util.spec_from_file_location("tweakreg_align_exposures", SCRIPT_PATH)
assert _spec is not None and _spec.loader is not None
tweakreg_script = importlib.util.module_from_spec(_spec)
sys.modules.setdefault("tweakreg_align_exposures", tweakreg_script)
_spec.loader.exec_module(tweakreg_script)


def test_output_name_matches_consumer_glob() -> None:
    name = tweakreg_script._output_name(3, "jw02731001001_04101_00001_nrcb1_cal", "rshift")
    assert name == "003_jw02731001001_04101_00001_nrcb1_cal_rshift_tweakreg.fits"
    assert fnmatch.fnmatch(name, "*_rshift_tweakreg.fits")


def test_invalid_fitgeometry_raises_value_error_before_jwst_import() -> None:
    exposures = [
        (Path("a.fits"), "F277W", "nrcb1"),
        (Path("b.fits"), "F277W", "nrcb1"),
    ]
    with pytest.raises(ValueError, match="invalid TweakReg fitgeometry"):
        tweakreg_script.run_tweakreg_alignment(exposures, output_dir=Path("/tmp/x"), fitgeometry="bogus")


@pytest.mark.skipif(
    importlib.util.find_spec("jwst") is not None,
    reason="jwst installed; missing-dependency path unreachable",
)
def test_missing_jwst_raises_runtime_error_with_hint(tmp_path: Path) -> None:
    exposures = [
        (Path("a.fits"), "F277W", "nrcb1"),
        (Path("b.fits"), "F277W", "nrcb1"),
    ]
    with pytest.raises(
        RuntimeError,
        match="tweakreg alignment requires the jwst pipeline environment",
    ):
        tweakreg_script.run_tweakreg_alignment(exposures, output_dir=tmp_path)


def test_single_exposure_rejected(tmp_path: Path) -> None:
    exposures = [(Path("a.fits"), "F277W", "nrcb1")]
    with pytest.raises(ValueError, match="at least two exposures"):
        tweakreg_script.run_tweakreg_alignment(exposures, output_dir=tmp_path)


def test_validate_sci_wcs_rejects_nonfinite_footprint(tmp_path: Path) -> None:
    header = fits.Header(
        {
            "NAXIS": 2,
            "NAXIS1": 8,
            "NAXIS2": 8,
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 4.5,
            "CRPIX2": 4.5,
            "CRVAL1": 150.0,
            "CRVAL2": 999.0,  # invalid declination -> non-finite footprint
            "CD1_1": -3e-5,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 3e-5,
        }
    )
    destination = tmp_path / "broken_tweakreg.fits"
    fits.PrimaryHDU().writeto(destination, overwrite=True)
    with fits.open(destination, mode="update") as hdul:
        hdul.append(fits.ImageHDU(data=np.zeros((8, 8), np.float32), name="SCI"))
        hdul["SCI"].header.update(header)
    with pytest.raises(RuntimeError, match=f"invalid saved TweakReg WCS in {destination}"):
        tweakreg_script._validate_sci_wcs(destination)
