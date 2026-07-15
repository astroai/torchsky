from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

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


def test_short_detector_alias_selects_throughput_file(tmp_path: Path) -> None:
    root = tmp_path / "throughput"
    root.mkdir()
    curve = root / "F090W_B1_system_throughput.txt"
    np.savetxt(
        curve,
        np.array([[0.8, 0.0], [0.9, 1.0], [1.0, 0.0]], dtype=np.float64),
    )
    result = build_bandpass_calibration(
        [_exposure(tmp_path)],
        mode="official",
        num_samples=5,
        band_specs=BAND_SPECS,
        throughput_dir=root,
    )
    assert result.metadata[0]["source"] == str(curve)
    torch.testing.assert_close(result.weight.sum(-1), torch.ones(1))


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
    torch.testing.assert_close(
        result.bank.kernel[0, 0],
        torch.from_numpy(expected),
    )
    assert result.metadata[0]["stored_kernel_orientation"] == (
        "flipped_for_convolution"
    )
