from __future__ import annotations

import csv
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest

from torchsky.cli.calibration_command import (
    _write_json_atomic,
    build_calibration_manifest,
)


def _write_jwst_exposure(
    path: Path,
    *,
    band: str = "F277W",
    detector: str = "NRCALONG",
    patt_num: int = 3,
) -> None:
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "JWST"
    primary.header["INSTRUME"] = "NIRCAM"
    primary.header["FILTER"] = band
    primary.header["DETECTOR"] = detector
    primary.header["PATT_NUM"] = patt_num
    primary.header["SUBSTRT1"] = 101
    primary.header["SUBSTRT2"] = 201
    primary.header["SUBSIZE1"] = 10
    primary.header["SUBSIZE2"] = 8
    primary.header["EXPSTART"] = 60000.0
    primary.header["EXPMID"] = 60000.0005
    primary.header["EXPEND"] = 60000.001
    science = fits.ImageHDU(data=np.zeros((8, 10), dtype=np.float32), name="SCI")
    science.header["MJD-BEG"] = 60000.0
    science.header["MJD-MID"] = 60000.0005
    science.header["MJD-END"] = 60000.001
    fits.HDUList([primary, science]).writeto(path)


def test_calibration_manifest_reads_torchfits_headers_and_evidence(tmp_path: Path) -> None:
    exposure = tmp_path / "jw_test_nrcalong_cal.fits"
    _write_jwst_exposure(exposure)
    evidence = tmp_path / "evidence.csv"
    with evidence.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "exposure",
                "source_id",
                "detector_x",
                "detector_y",
                "role",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "exposure": exposure.name,
                "source_id": "star-1",
                "detector_x": 104.0,
                "detector_y": 203.0,
                "role": "psf",
            }
        )

    manifest = build_calibration_manifest(
        [str(exposure)], evidence_path=evidence, coverage_rtol=0.05
    )

    row = manifest["exposures"][0]
    assert row["band"] == "F277W"
    assert row["split"] == "validation"
    assert row["detector_bounds_xy"] == [100, 109, 200, 207]
    assert row["timing"]["midpoint_source"] == "SCI:MJD-MID"
    assert row["instrument"]["empirical_psf_fwhm_arcsec"] == 0.092
    assert manifest["evidence"][0]["split"] == "validation"
    assert manifest["coverage"][0]["role"] == "psf"
    assert len(manifest["manifest_sha256"]) == 64


def test_calibration_manifest_rejects_out_of_bounds_evidence(tmp_path: Path) -> None:
    exposure = tmp_path / "jw_test_nrcalong_cal.fits"
    _write_jwst_exposure(exposure)
    evidence = tmp_path / "evidence.json"
    evidence.write_text(
        json.dumps(
            [
                {
                    "exposure": exposure.name,
                    "detector_x": 120,
                    "detector_y": 203,
                    "role": "background",
                }
            ]
        ),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="outside"):
        build_calibration_manifest([str(exposure)], evidence_path=evidence)


def test_calibration_manifest_atomic_write_requires_overwrite(tmp_path: Path) -> None:
    output = tmp_path / "manifest.json"
    _write_json_atomic(output, {"ok": True}, overwrite=False)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        _write_json_atomic(output, {"ok": False}, overwrite=False)
    _write_json_atomic(output, {"ok": False}, overwrite=True)
    assert json.loads(output.read_text(encoding="utf-8")) == {"ok": False}
