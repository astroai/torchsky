from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path

from astropy.io import fits
import numpy as np
import pytest
import torch

import torchsky
from torchsky.calibration_evidence import (
    discover_detector_calibration_positions,
    extract_calibration_evidence,
    extract_reviewed_calibration_evidence,
    load_operator_calibration_split,
    read_reviewed_sky_catalog,
)
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


def _write_calibration_image(path: Path) -> tuple[dict[str, object], np.ndarray]:
    size = 25
    yy, xx = np.mgrid[:size, :size]
    science_data = 10.0 + 100.0 * np.exp(
        -0.5 * ((xx - 12.0) ** 2 + (yy - 12.0) ** 2)
    )
    primary = fits.PrimaryHDU()
    primary.header["TELESCOP"] = "JWST"
    primary.header["INSTRUME"] = "NIRCAM"
    primary.header["FILTER"] = "F277W"
    primary.header["DETECTOR"] = "NRCALONG"
    primary.header["PATT_NUM"] = 1
    primary.header["SUBSTRT1"] = 1
    primary.header["SUBSTRT2"] = 1
    primary.header["SUBSIZE1"] = size
    primary.header["SUBSIZE2"] = size
    primary.header["EXPSTART"] = 60000.0
    primary.header["EXPMID"] = 60000.0005
    primary.header["EXPEND"] = 60000.001
    science = fits.ImageHDU(data=science_data.astype(np.float32), name="SCI")
    science.header.update(
        {
            "CTYPE1": "RA---TAN",
            "CTYPE2": "DEC--TAN",
            "CRPIX1": 13.0,
            "CRPIX2": 13.0,
            "CRVAL1": 110.82,
            "CRVAL2": -73.45,
            "CD1_1": -0.00001,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.00001,
            "MJD-MID": 60000.0005,
            "BUNIT": "MJy/sr",
        }
    )
    error = fits.ImageHDU(data=np.ones((size, size), dtype=np.float32), name="ERR")
    dq = fits.ImageHDU(data=np.zeros((size, size), dtype=np.int16), name="DQ")
    fits.HDUList([primary, science, error, dq]).writeto(path)
    return dict(science.header), science_data


def test_extract_reviewed_calibration_evidence_persists_split_targets(
    tmp_path: Path,
) -> None:
    exposure = tmp_path / "jw_cal.fits"
    header, _ = _write_calibration_image(exposure)
    wcs = torchsky.WCS(header)
    world = wcs.pixel_to_world(
        torch.tensor([[12.0, 12.0], [5.0, 5.0]], dtype=torch.float64), origin=0
    )
    catalog = tmp_path / "reviewed.csv"
    with catalog.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=("source_id", "role", "ra_deg", "dec_deg", "reviewed"),
        )
        writer.writeheader()
        writer.writerow(
            {
                "source_id": "star-1",
                "role": "psf",
                "ra_deg": float(world[0, 0]),
                "dec_deg": float(world[0, 1]),
                "reviewed": "true",
            }
        )
        writer.writerow(
            {
                "source_id": "blank-1",
                "role": "background",
                "ra_deg": float(world[1, 0]),
                "dec_deg": float(world[1, 1]),
                "reviewed": "true",
            }
        )
    base = build_calibration_manifest([str(exposure)])

    result = extract_reviewed_calibration_evidence(
        base,
        catalog_path=catalog,
        output_dir=tmp_path / "evidence",
        psf_radius_px=3,
        background_radius_px=3,
        minimum_psf_snr=2.0,
    )

    assert {row["role"] for row in result["evidence"]} == {"psf", "background"}
    assert result["rejected"] == []
    psf_row = next(row for row in result["evidence"] if row["role"] == "psf")
    assert psf_row["pixel_response_included"] is True
    assert psf_row["pixel_response_provenance"] == "registered detector SCI stamp"
    kernels = torchsky.read(result["products"]["psf_kernels"]["path"])
    torch.testing.assert_close(kernels.sum(dim=(-2, -1)), torch.ones(1))
    assert result["products"]["detector_evidence"]["size_bytes"] > 0


def test_psf_selection_checks_neighbors_across_extracted_kernel() -> None:
    yy, xx = torch.meshgrid(torch.arange(64.), torch.arange(64.), indexing="ij")
    star = 100 * torch.exp(-((xx - 24).square() + (yy - 24).square()) / 2)
    neighbor = 100 * torch.exp(-((xx - 30).square() + (yy - 30).square()) / 2)
    error = torch.ones_like(star)
    dq = torch.zeros_like(star, dtype=torch.int16)
    kwargs = dict(psf_fwhm_px=2.354820045, psf_radius_px=9,
                  max_background_candidates=0)
    isolated = discover_detector_calibration_positions(star, error, dq, **kwargs)
    assert len(isolated) == 1
    blended = discover_detector_calibration_positions(star + neighbor, error, dq, **kwargs)
    assert blended == []


def test_automatic_calibration_evidence_finds_star_and_blank_sky(
    tmp_path: Path,
) -> None:
    exposure = tmp_path / "jw_crf_rshift_tweakreg.fits"
    _, science = _write_calibration_image(exposure)
    candidates = discover_detector_calibration_positions(
        torch.from_numpy(science).to(torch.float32),
        torch.ones_like(torch.from_numpy(science).to(torch.float32)),
        torch.zeros_like(torch.from_numpy(science), dtype=torch.int16),
        psf_fwhm_px=2.354820045,
        psf_radius_px=3,
        background_radius_px=3,
        detection_sigma=5.0,
        max_psf_candidates=4,
        max_background_candidates=4,
    )

    psf = [row for row in candidates if row["role"] == "psf"]
    background = [row for row in candidates if row["role"] == "background"]
    assert len(psf) == 1
    assert abs(psf[0]["detector_x"] - 12.0) < 0.1
    assert abs(psf[0]["detector_y"] - 12.0) < 0.1
    assert len(background) == 4

    result = extract_calibration_evidence(
        build_calibration_manifest([str(exposure)]),
        output_dir=tmp_path / "automatic-evidence",
        psf_radius_px=3,
        background_radius_px=3,
        minimum_psf_snr=2.0,
        detection_sigma=5.0,
        max_psf_candidates=4,
        max_background_candidates=4,
    )

    assert result["selection_mode"] == "automatic_detector"
    assert {row["role"] for row in result["evidence"]} == {"psf", "background"}
    assert "catalog_path" not in result


def test_load_operator_calibration_split_verifies_and_selects_targets(
    tmp_path: Path,
) -> None:
    kernels = tmp_path / "kernels.fits"
    weights = tmp_path / "weights.fits"
    torchsky.write(str(kernels), torch.ones((2, 3, 3)) / 9.0)
    torchsky.write(str(weights), torch.ones((2, 3, 3)))
    def sha(path: Path) -> str:
        return hashlib.sha256(path.read_bytes()).hexdigest()

    manifest = {
        "extraction": {
            "products": {
                "psf_kernels": {"path": str(kernels), "sha256": sha(kernels)},
                "psf_pixel_weights": {
                    "path": str(weights),
                    "sha256": sha(weights),
                },
            }
        },
        "evidence": [
            {
                "role": "psf",
                "split": "test",
                "detector": "NRCALONG",
                "band": "F277W",
                "detector_x": 1.0,
                "detector_y": 2.0,
                "time_mjd_utc": 60000.0,
                "sample_weight": 3.0,
                "target_index": 1,
            }
        ],
    }

    arrays = load_operator_calibration_split(
        manifest, split="test", role="psf", detector="NRCALONG"
    )

    assert arrays["target_kernels"].shape == (1, 3, 3)
    assert arrays["wavelength"].item() == 27760.0
    assert arrays["sample_weight"].item() == 3.0


def test_reviewed_catalog_fails_closed_on_unreviewed_rows(tmp_path: Path) -> None:
    catalog = tmp_path / "candidates.csv"
    catalog.write_text(
        "source_id,role,ra_deg,dec_deg,reviewed\nstar-1,psf,1,2,false\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reviewed must be true"):
        read_reviewed_sky_catalog(catalog)
