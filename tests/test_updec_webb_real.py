from __future__ import annotations

import sys
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / "scripts"
BENCHMARKS = ROOT / "benchmarks"
if str(SCRIPTS) not in sys.path:
    sys.path.insert(0, str(SCRIPTS))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from download_updec_webb import UPDEC_WEBB_FILES, resolve_specs  # noqa: E402
from bench_updec_webb_real import (  # noqa: E402
    _aligned_tan_center_crops,
    _markdown,
    _open_sci,
)


def test_updec_webb_public_manifest_contains_expected_sparse_smoke_band() -> None:
    spec = UPDEC_WEBB_FILES[("F277W", "b", 0)]
    assert spec.filename == "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
    assert spec.download_id == 47980
    assert spec.url.endswith("47980")


def test_module_a_manifest_resolved_from_release() -> None:
    spec = UPDEC_WEBB_FILES[("F277W", "a", 0)]
    assert spec.filename == "JWST_NIRCAM_F277Wa_UPDC-v1.1_SCI_clear_iter00.fits"
    assert spec.download_id == 46474
    assert UPDEC_WEBB_FILES[("F090W", "a", 10)].download_id == 46491


def test_resolve_specs_rejects_unknown_module() -> None:
    import pytest

    with pytest.raises(ValueError):
        resolve_specs(["F277W"], [0], modules=("c",))


def test_resolve_specs_supports_public_iter0_iter10_pair() -> None:
    specs = resolve_specs(["F277W"], [0, 10])
    assert [spec.iteration for spec in specs] == [0, 10]
    assert specs[1].filename == "F277Wb_UPDC-v2_iter10_i2d.fits.fz"


def test_updec_webb_markdown_renders_expected_columns() -> None:
    text = _markdown(
        [
            {
                "band": "F277W",
                "iter0_hdus": 10,
                "iter10_hdus": 2,
                "common_rmse": 0.1,
                "crop_rmse": 0.2,
                "crop_gradient_mean_ratio": 1.5,
                "crop_laplacian_std_ratio": 2.5,
                "common_flux_ratio": 1.01,
            }
        ]
    )
    assert "| band | iter0 hdus | iter10 hdus |" in text
    assert "| F277W | 10 | 2 | 0.100000 | 0.200000 | 1.5000 | 2.5000 | 1.0100 |" in text


def test_open_sci_uses_torchfits_boundary(tmp_path: Path) -> None:
    from astropy.io import fits

    path = tmp_path / "updec_pair.fits"
    expected = np.arange(12, dtype=np.float32).reshape(3, 4)
    fits.HDUList(
        [
            fits.PrimaryHDU(),
            fits.ImageHDU(expected, name="SCI"),
        ]
    ).writeto(path)

    data, header, n_hdus = _open_sci(path)

    assert n_hdus == 2
    assert header["EXTNAME"] == "SCI"
    np.testing.assert_array_equal(data, expected)


def test_tan_pair_alignment_handles_shape_scale_and_crpix_changes() -> None:
    target = np.arange(25, dtype=np.float32).reshape(5, 5)
    source = np.zeros((7, 7), dtype=np.float32)
    source[1:6, 1:6] = target
    target_header = {
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CRPIX1": 3.0,
        "CRPIX2": 3.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "CD1_1": None,
        "CD1_2": None,
        "CD2_1": None,
        "CD2_2": None,
        "PC1_1": 0.0,
        "PC1_2": -1.0,
        "PC2_1": 1.0,
        "PC2_2": 0.0,
    }
    source_header = {
        **target_header,
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
    }

    target_crop, aligned, valid = _aligned_tan_center_crops(
        target,
        target_header,
        source,
        source_header,
        crop_size=5,
    )

    assert valid.all()
    np.testing.assert_allclose(aligned, target_crop, rtol=0.0, atol=3.0e-6)
