"""Round-trip tests for torchsky.spectral.write_cube_fits."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch

from torchsky.spectral import SpectralAxis, SpectralReader, write_cube_fits


def _spatial_header(*, h: int = 8, w: int = 12) -> dict[str, float | int | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": w,
        "NAXIS2": h,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": (w + 1) / 2,
        "CRPIX2": (h + 1) / 2,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.05 / 3600.0,
        "CDELT2": 0.05 / 3600.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }


def test_write_cube_fits_roundtrip_defaults() -> None:
    L, H, W = 5, 8, 12
    cube = torch.randn(L, H, W, dtype=torch.float32)
    wave = torch.linspace(4800.0, 5200.0, L, dtype=torch.float64)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cube.fits"
        header_out = write_cube_fits(
            str(path),
            cube,
            spatial_header=_spatial_header(h=H, w=W),
            spectral_axis=axis,
            overwrite=True,
        )
        assert int(header_out["NAXIS"]) == 3
        assert int(header_out["NAXIS1"]) == W
        assert int(header_out["NAXIS2"]) == H
        assert int(header_out["NAXIS3"]) == L
        assert float(header_out["CRVAL3"]) == pytest.approx(float(wave[0].item()))

        dc = SpectralReader.read_data_cube(str(path), hdu=0)
        assert dc.data.shape == cube.shape
        assert dc.spectral_axis_index == 0
        torch.testing.assert_close(dc.data, cube)
        torch.testing.assert_close(
            dc.spectral_axis.values.to(dtype=torch.float64),
            wave,
            rtol=1e-5,
            atol=1e-5,
        )
        assert dc.header is not None
        assert dc.header.get("CRVAL1") is not None


def test_write_cube_fits_permuted_spectral_axis() -> None:
    L, H, W = 4, 6, 8
    cube_chw = torch.randn(H, W, L, dtype=torch.float32)
    wave = torch.linspace(1.0, 4.0, L, dtype=torch.float64)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cube.fits"
        write_cube_fits(
            str(path),
            cube_chw,
            spatial_header=_spatial_header(h=H, w=W),
            spectral_axis=axis,
            spectral_axis_index=2,
            overwrite=True,
        )
        dc = SpectralReader.read_data_cube(str(path), hdu=0)
        expected = cube_chw.movedim(2, 0)
        torch.testing.assert_close(dc.data, expected)


def test_write_cube_fits_rejects_nonlinear_grid() -> None:
    L = 3
    cube = torch.zeros(L, 4, 4)
    wave = torch.tensor([4000.0, 4100.0, 4300.0], dtype=torch.float64)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "bad.fits"
        with pytest.raises(ValueError, match="uniformly spaced"):
            write_cube_fits(
                str(path),
                cube,
                spatial_header=_spatial_header(h=4, w=4),
                spectral_axis=axis,
                overwrite=True,
            )
