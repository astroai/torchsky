from pathlib import Path

import numpy as np
import torch
from astropy.io import fits

from torchsky.map.wcs_aware_target_canvas import (
    filter_sampled_pools_to_canvas,
    load_target_canvas_from_fits,
)
from torchsky.map.wcs_aware_training import ExposurePool


def _tan_header(width: int, height: int) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = width
    header["NAXIS2"] = height
    header["CTYPE1"] = "RA---TAN"
    header["CTYPE2"] = "DEC--TAN"
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"
    header["CRVAL1"] = 110.8
    header["CRVAL2"] = -73.45
    header["CRPIX1"] = (width + 1) / 2.0
    header["CRPIX2"] = (height + 1) / 2.0
    header["CD1_1"] = -0.031 / 3600.0
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = 0.031 / 3600.0
    return header


def test_load_target_canvas_from_fits_preserves_grid(tmp_path: Path) -> None:
    path = tmp_path / "template.fits"
    header = _tan_header(11, 7)
    fits.PrimaryHDU(np.zeros((7, 11), dtype=np.float32), header=header).writeto(path)

    canvas, metadata = load_target_canvas_from_fits(path)

    assert canvas.shape == (7, 11)
    assert canvas.header["NAXIS1"] == 11
    assert canvas.header["NAXIS2"] == 7
    assert abs(canvas.pixel_scale_arcsec - 0.031) < 1.0e-6
    assert metadata["mode"] == "fits_template"
    assert metadata["shape"] == [7, 11]


def test_filter_sampled_pools_to_canvas_removes_outside_samples(tmp_path: Path) -> None:
    template_path = tmp_path / "template.fits"
    header = _tan_header(5, 5)
    fits.PrimaryHDU(np.zeros((5, 5), dtype=np.float32), header=header).writeto(template_path)
    canvas, _ = load_target_canvas_from_fits(template_path)

    # Same WCS as the target. Pixel (2,2) is inside; (20,20) is outside.
    pool = ExposurePool(
        path=tmp_path / "exposure.fits",
        band="F090W",
        detector="NRCB1",
        header=dict(header),
        shape=(32, 32),
        uv=torch.tensor([[2.0, 2.0], [20.0, 20.0]], dtype=torch.float32),
        value=torch.tensor([1.0, 2.0]),
        err=torch.tensor([0.1, 0.1]),
    )

    filtered, report = filter_sampled_pools_to_canvas([pool], canvas)

    assert filtered[0].size == 1
    assert torch.allclose(filtered[0].uv[0], torch.tensor([2.0, 2.0]))
    assert report["sampled_pixels"] == 2
    assert report["inside_canvas"] == 1
    assert report["outside_canvas"] == 1
    assert report["filter_applied"] is True
