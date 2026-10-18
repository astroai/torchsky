"""Tests for non-uniform spectral axis writing and reading."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest
import torch
import torchfits

from torchsky.spectral import (
    SpectralAxis,
    SpectralReader,
    write_cube_fits,
    spectral_axis_from_header,
    spectral_bin_edges_from_header,
)


def _spatial_header(*, h: int = 8, w: int = 8) -> dict[str, float | int | str]:
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


def test_write_cube_fits_nonuniform_custom_cards() -> None:
    L, H, W = 4, 8, 8
    cube = torch.randn(L, H, W, dtype=torch.float32)
    # Explicitly non-uniform wavelengths
    wave = torch.tensor([4000.0, 4150.0, 4220.0, 4400.0], dtype=torch.float64)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cube_custom_cards.fits"

        # Should raise error if allow_nonuniform is False (default)
        with pytest.raises(ValueError, match="uniformly spaced"):
            write_cube_fits(
                str(path),
                cube,
                spatial_header=_spatial_header(h=H, w=W),
                spectral_axis=axis,
                overwrite=True,
            )

        header_out = write_cube_fits(
            str(path),
            cube,
            spatial_header=_spatial_header(h=H, w=W),
            spectral_axis=axis,
            overwrite=True,
            allow_nonuniform=True,
            nonuniform_fallback="custom_cards",
        )

        assert int(header_out["NAXIS"]) == 3
        assert header_out["WAVE0"] == 4000.0
        assert header_out["WAVE1"] == 4150.0
        assert header_out["WAVE2"] == 4220.0
        assert header_out["WAVE3"] == 4400.0

        dc = SpectralReader.read_data_cube(str(path), hdu=0)
        assert dc.data.shape == cube.shape
        torch.testing.assert_close(dc.data, cube)
        torch.testing.assert_close(
            dc.spectral_axis.values.to(dtype=torch.float64),
            wave,
            rtol=1e-5,
            atol=1e-5,
        )

        # Test bin edges computation
        edges, _ = spectral_bin_edges_from_header(dc.header)
        expected_edges = torch.tensor(
            [3925.0, 4075.0, 4185.0, 4310.0, 4490.0], dtype=torch.float64
        )
        torch.testing.assert_close(edges, expected_edges)


def test_write_cube_fits_nonuniform_table() -> None:
    L, H, W = 4, 8, 8
    cube = torch.randn(L, H, W, dtype=torch.float32)
    # Explicitly non-uniform wavelengths
    wave = torch.tensor([4000.0, 4150.0, 4220.0, 4400.0], dtype=torch.float64)
    axis = SpectralAxis(values=wave, unit="Angstrom", type="WAVE")

    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "cube_table.fits"
        header_out = write_cube_fits(
            str(path),
            cube,
            spatial_header=_spatial_header(h=H, w=W),
            spectral_axis=axis,
            overwrite=True,
            allow_nonuniform=True,
            nonuniform_fallback="table",
        )

        assert int(header_out["NAXIS"]) == 3
        assert header_out["CTYPE3"] == "WAVE-TAB"
        assert header_out["PS3_0"] == "WCS-TAB"
        assert header_out["PS3_1"] == "WAVE"


        with torchfits.open(str(path)) as hdul:
            assert len(hdul) == 2
            assert hdul[1].header["EXTNAME"] == "WCS-TAB"
            # Verify coordinates are stored in the second HDU
            table_hdu = hdul[1]
            torch.testing.assert_close(
                table_hdu["WAVE"].squeeze().to(dtype=torch.float64),
                wave,
            )

        dc = SpectralReader.read_data_cube(str(path), hdu=0)
        assert dc.data.shape == cube.shape
        torch.testing.assert_close(dc.data, cube)
        torch.testing.assert_close(
            dc.spectral_axis.values.to(dtype=torch.float64),
            wave,
            rtol=1e-5,
            atol=1e-5,
        )

        # Test bin edges computation with tables provided

        with torchfits.open(str(path)) as hdul:
            edges, _ = spectral_bin_edges_from_header(dc.header, tables=hdul)
            expected_edges = torch.tensor(
                [3925.0, 4075.0, 4185.0, 4310.0, 4490.0], dtype=torch.float64
            )
            torch.testing.assert_close(edges, expected_edges)


def test_tabular_spectral_axis_requires_tables() -> None:
    header = {
        "NAXIS": 3,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "NAXIS3": 4,
        "CTYPE3": "WAVE-TAB",
        "PS3_0": "WCS-TAB",
        "PS3_1": "WAVE",
    }

    with pytest.raises(ValueError, match="requires coordinate tables"):
        spectral_axis_from_header(header, data_shape=(4, 8, 8))
