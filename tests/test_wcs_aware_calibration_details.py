from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from astropy import units as u
from astropy.table import Table

import torchsky.map.wcs_aware_calibration_bandpass as bandpass_calibration
from torchsky.map.wcs_aware_calibration import (
    ExposureCalibrationInput,
    build_bandpass_calibration,
    build_psf_calibration,
)


BAND_SPECS = {"F090W": (9000.0, 8000.0, 10000.0)}


def _header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": 110.0,
        "CRVAL2": -73.0,
        "CRPIX1": 4.5,
        "CRPIX2": 4.5,
        "CD1_1": -1.0e-5,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-5,
    }


def _exposure(tmp_path: Path) -> ExposureCalibrationInput:
    return ExposureCalibrationInput(
        path=tmp_path / "sample_crf.fits",
        band="F090W",
        detector="NRCB1",
        header=_header(),
        shape=(8, 8),
    )


def test_svo_table_curve_is_converted_to_angstrom() -> None:
    table = Table()
    table["Wavelength"] = np.array([0.8, 0.9, 1.0]) * u.micron
    table["Transmission"] = np.array([0.0, 0.8, 0.0])

    wavelength, transmission = bandpass_calibration._extract_svo_table_curve(
        table,
        filter_id="JWST/NIRCam2025.F090W",
    )
    np.testing.assert_allclose(wavelength, [8000.0, 9000.0, 10000.0])
    np.testing.assert_allclose(transmission, [0.0, 0.8, 0.0])


def test_official_bandpass_queries_once_and_reuses_snapshot(
    monkeypatch,
    tmp_path: Path,
) -> None:
    calls: list[tuple[str, float]] = []

    def fake_download(
        filter_id: str,
        *,
        timeout: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        calls.append((filter_id, timeout))
        return (
            np.array([8000.0, 9000.0, 10000.0]),
            np.array([0.0, 1.0, 0.0]),
        )

    monkeypatch.setattr(bandpass_calibration, "_download_svo_curve", fake_download)
    second = ExposureCalibrationInput(
        path=tmp_path / "second_crf.fits",
        band="F090W",
        detector="NRCB2",
        header=_header(),
        shape=(8, 8),
    )
    cache_dir = tmp_path / "svo-cache"
    result = build_bandpass_calibration(
        [_exposure(tmp_path), second],
        mode="official",
        num_samples=5,
        band_specs=BAND_SPECS,
        svo_collection="JWST/NIRCam2025",
        svo_cache_dir=cache_dir,
        svo_timeout=12.0,
    )

    assert calls == [("JWST/NIRCam2025.F090W", 12.0)]
    assert result.metadata[0]["svo_filter_id"] == "JWST/NIRCam2025.F090W"
    assert result.metadata[0]["detector_specific"] is False
    torch.testing.assert_close(result.weight.sum(-1), torch.ones(2))
    assert len(list(cache_dir.glob("*.npz"))) == 1

    def fail_download(
        filter_id: str,
        *,
        timeout: float,
    ) -> tuple[np.ndarray, np.ndarray]:
        raise AssertionError(f"unexpected SVO query for {filter_id}")

    monkeypatch.setattr(bandpass_calibration, "_download_svo_curve", fail_download)
    cached = build_bandpass_calibration(
        [_exposure(tmp_path)],
        mode="official",
        num_samples=5,
        band_specs=BAND_SPECS,
        svo_collection="JWST/NIRCam2025",
        svo_cache_dir=cache_dir,
        svo_timeout=12.0,
    )
    assert cached.metadata[0]["svo_cache_hit"] is True
    torch.testing.assert_close(cached.weight.sum(-1), torch.ones(1))


def test_npy_psf_is_flipped_for_mathematical_convolution(tmp_path: Path) -> None:
    kernel_dir = tmp_path / "kernels"
    kernel_dir.mkdir()
    optical_kernel = np.array(
        [[0.0, 0.0, 0.0], [0.0, 1.0, 2.0], [0.0, 0.0, 0.0]],
        dtype=np.float32,
    )
    np.save(kernel_dir / "F090W.npy", optical_kernel)
    wavelength = torch.tensor([[8500.0, 9500.0]], dtype=torch.float32)
    result = build_psf_calibration(
        [_exposure(tmp_path)],
        wavelength,
        mode="npy",
        kernel_size=3,
        kernel_dir=kernel_dir,
    )
    assert result.bank is not None
    expected = np.flip(optical_kernel, axis=(-2, -1)).copy()
    expected /= expected.sum()
    torch.testing.assert_close(result.bank.kernel[0, 0], torch.from_numpy(expected))
    assert result.metadata[0]["stored_kernel_orientation"] == (
        "flipped_for_convolution"
    )
