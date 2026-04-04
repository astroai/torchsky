from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from astropy.io import fits


ROOT = Path(__file__).resolve().parents[1]
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_real import (  # noqa: E402
    _band_spec,
    _approx_overlap_grid,
    _best_overlap_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _reference_path_for_band,
    _linearize_sip_header,
    _metric_mask,
)


def test_cutout_header_shifts_crpix_and_shape() -> None:
    header = fits.Header()
    header["NAXIS1"] = 100
    header["NAXIS2"] = 80
    header["CRPIX1"] = 50.5
    header["CRPIX2"] = 40.5

    cut = _cutout_header(header, x0=10, y0=7, size=32)
    assert cut["NAXIS1"] == 32
    assert cut["NAXIS2"] == 32
    assert cut["CRPIX1"] == 40.5
    assert cut["CRPIX2"] == 33.5


def test_linearize_sip_header_drops_sip_terms() -> None:
    header = fits.Header()
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["A_ORDER"] = 2
    header["A_0_2"] = 1.0e-6
    header["BP_ORDER"] = 2
    header["BP_0_2"] = -1.0e-6

    out = _linearize_sip_header(header)
    assert out["CTYPE1"] == "RA---TAN"
    assert out["CTYPE2"] == "DEC--TAN"
    assert "A_ORDER" not in out
    assert "A_0_2" not in out
    assert "BP_ORDER" not in out
    assert "BP_0_2" not in out


def test_best_overlap_cutout_prefers_supported_window() -> None:
    reference = np.arange(36, dtype=np.float32).reshape(6, 6)
    overlap = np.zeros((6, 6), dtype=np.float32)
    overlap[2:5, 1:4] = 1.0

    crop, x0, y0 = _best_overlap_cutout(reference, overlap, size=3)
    assert (x0, y0) == (1, 2)
    assert crop.shape == (3, 3)
    assert np.array_equal(crop, reference[2:5, 1:4])


def test_metric_mask_excludes_non_finite_reference() -> None:
    pred = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32)
    reference = np.array([[1.0, np.nan], [np.inf, 4.0]], dtype=np.float32)
    support = np.array([[True, True], [True, False]])

    mask = _metric_mask(pred, reference, support)
    expected = np.array([[True, False], [False, False]])
    assert np.array_equal(mask, expected)


def test_discover_exposure_paths_prefers_crf_in_auto_mode(tmp_path: Path) -> None:
    (tmp_path / "a_cal.fits").write_bytes(b"")
    (tmp_path / "b_crf.fits").write_bytes(b"")
    (tmp_path / "c_crf.fits").write_bytes(b"")

    paths = _discover_exposure_paths(tmp_path)
    assert [path.name for path in paths] == ["b_crf.fits", "c_crf.fits"]


def test_discover_exposure_paths_can_force_cal_or_crf(tmp_path: Path) -> None:
    (tmp_path / "a_cal.fits").write_bytes(b"")
    (tmp_path / "b_crf.fits").write_bytes(b"")
    (tmp_path / "c_crf.fits").write_bytes(b"")

    cal_paths = _discover_exposure_paths(tmp_path, input_product="cal")
    crf_paths = _discover_exposure_paths(tmp_path, input_product="crf")

    assert [path.name for path in cal_paths] == ["a_cal.fits"]
    assert [path.name for path in crf_paths] == ["b_crf.fits", "c_crf.fits"]


def test_discover_exposure_paths_can_filter_by_filename(tmp_path: Path) -> None:
    (tmp_path / "a_nrcalong_cal.fits").write_bytes(b"")
    (tmp_path / "b_nrcblong_cal.fits").write_bytes(b"")

    paths = _discover_exposure_paths(
        tmp_path,
        input_product="cal",
        filename_contains="nrcblong",
    )
    assert [path.name for path in paths] == ["b_nrcblong_cal.fits"]


def test_band_helpers_resolve_reference_and_wavelength() -> None:
    spec = _band_spec("F277W")
    path = _reference_path_for_band("F277W")

    assert spec["wavelength_angstrom"] == 27700.0
    assert path.name == "F277Wb_UPDC-v2_iter0_i2d.fits.fz"


def test_discover_exposure_paths_can_filter_by_band_and_module(tmp_path: Path) -> None:
    def _write_fake(path: Path, *, band: str) -> None:
        primary = fits.PrimaryHDU()
        sci = fits.ImageHDU(data=np.zeros((2, 2), dtype=np.float32), name="SCI")
        sci.header["FILTER"] = band
        fits.HDUList([primary, sci]).writeto(path)

    _write_fake(tmp_path / "a_nrcb1_cal.fits", band="F200W")
    _write_fake(tmp_path / "b_nrca1_cal.fits", band="F200W")
    _write_fake(tmp_path / "c_nrcblong_cal.fits", band="F277W")

    paths = _discover_exposure_paths(
        tmp_path,
        input_product="cal",
        band="F200W",
        module="b",
    )
    assert [path.name for path in paths] == ["a_nrcb1_cal.fits"]


def test_approx_overlap_grid_counts_projected_headers() -> None:
    target = fits.Header()
    target["NAXIS1"] = 16
    target["NAXIS2"] = 16
    target["CTYPE1"] = "RA---TAN"
    target["CTYPE2"] = "DEC--TAN"
    target["CRPIX1"] = 8.0
    target["CRPIX2"] = 8.0
    target["CRVAL1"] = 0.0
    target["CRVAL2"] = 0.0
    target["CDELT1"] = -1.0 / 3600.0
    target["CDELT2"] = 1.0 / 3600.0

    shifted = target.copy()
    shifted["CRPIX1"] = 10.0

    overlap = _approx_overlap_grid([target, shifted], target, coarse_step=4)
    assert overlap.shape == (4, 4)
    assert float(overlap.max()) >= 1.0
    assert float(overlap.sum()) > 0.0
