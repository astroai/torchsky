from __future__ import annotations

import pytest
import torch

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane


def _simple_header() -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": 8,
        "NAXIS2": 8,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 4.0,
        "CRPIX2": 4.0,
        "CRVAL1": 12.0,
        "CRVAL2": -3.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
    }


def test_coadd_image_plane_identical_wcs(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    data_b = data_a + 2.0
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )
    expected = (data_a + data_b) / 2.0
    torch.testing.assert_close(result.data, expected, atol=1e-4, rtol=1e-4)
    assert result.weight.shape == expected.shape


def test_coadd_image_plane_respects_target_header_shape(tmp_path) -> None:
    path = tmp_path / "a.fits"
    header = _simple_header()
    data = torch.arange(64, dtype=torch.float32).reshape(8, 8)
    torchsky.write(str(path), data, header=header, overwrite=True)

    target_header = dict(header)
    target_header["NAXIS1"] = 6
    target_header["NAXIS2"] = 5

    result = coadd_image_plane(
        [ImageObservation(str(path))],
        target_header=target_header,
    )

    assert tuple(result.data.shape) == (5, 6)
    assert tuple(result.weight.shape) == (5, 6)


def test_coadd_image_plane_ignores_non_finite_pixels(tmp_path) -> None:
    path_a = tmp_path / "a.fits"
    path_b = tmp_path / "b.fits"
    header = _simple_header()
    data_a = torch.ones((8, 8), dtype=torch.float32)
    data_b = 3.0 * torch.ones((8, 8), dtype=torch.float32)
    data_a[2, 3] = torch.nan
    torchsky.write(str(path_a), data_a, header=header, overwrite=True)
    torchsky.write(str(path_b), data_b, header=header, overwrite=True)

    result = coadd_image_plane(
        [ImageObservation(str(path_a)), ImageObservation(str(path_b))]
    )

    assert torch.isfinite(result.data).all()
    assert result.data[2, 3].item() == 3.0
    assert result.weight[2, 3].item() == 1.0


def test_coadd_image_plane_rejects_non_2d_images(tmp_path) -> None:
    path = tmp_path / "cube.fits"
    header = _simple_header()
    header["NAXIS"] = 3
    header["NAXIS3"] = 2
    data = torch.ones((2, 8, 8), dtype=torch.float32)
    torchsky.write(str(path), data, header=header, overwrite=True)

    with pytest.raises(ValueError, match="2D images only"):
        coadd_image_plane([ImageObservation(str(path))])
