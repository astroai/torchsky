import os
import tempfile

import numpy as np
import torch
from astropy.io import fits

import torchsky
from torchsky.wcs import WCS


def test_wcs_2d_creation():
    header = {  # type: ignore
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CRPIX1": 1024,
        "CRPIX2": 1024,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,  # type: ignore
        "CDELT2": 0.1,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
    }
    wcs = WCS(header)  # type: ignore
    assert wcs.naxis == 2


def test_wcs_2d_pixel_to_world():
    header = {  # type: ignore
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CRPIX1": 1,
        "CRPIX2": 1,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,  # type: ignore
        "CDELT2": 0.1,
        "NAXIS1": 2,
        "NAXIS2": 2,
    }
    wcs = WCS(header)  # type: ignore
    # torchsky uses Python-style 0-based pixel coordinates.  # type: ignore
    pixels = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels)
    assert torch.allclose(  # type: ignore
        world, torch.tensor([[180.0, 0.0]], dtype=torch.float64), atol=1e-6  # type: ignore
    )


def test_wcs_2d_world_to_pixel():
    header = {  # type: ignore
        "WCSAXES": 2,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CRPIX1": 1,
        "CRPIX2": 1,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -0.1,  # type: ignore
        "CDELT2": 0.1,
        "NAXIS1": 2,
        "NAXIS2": 2,
    }
    wcs = WCS(header)  # type: ignore
    world = torch.tensor([[180.0, 0.0]], dtype=torch.float32)
    pixels = wcs.world_to_pixel(world)
    assert torch.allclose(  # type: ignore
        pixels, torch.tensor([[0.0, 0.0]], dtype=torch.float64), atol=1e-6  # type: ignore
    )


def test_get_wcs_from_file_auto_hdu():
    image = np.random.normal(size=(32, 32)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        hdu = fits.PrimaryHDU(image)
        hdu.header["CTYPE1"] = "RA---TAN"  # type: ignore
        hdu.header["CTYPE2"] = "DEC--TAN"  # type: ignore
        hdu.header["CRPIX1"] = 1.0  # type: ignore
        hdu.header["CRPIX2"] = 1.0  # type: ignore
        hdu.header["CRVAL1"] = 180.0  # type: ignore
        hdu.header["CRVAL2"] = 0.0  # type: ignore
        hdu.header["CDELT1"] = -0.1  # type: ignore
        hdu.header["CDELT2"] = 0.1  # type: ignore
        hdu.writeto(f.name, overwrite=True)
        path = f.name

    try:
        wcs = torchsky.get_wcs(path, hdu="auto")
        assert isinstance(wcs, WCS)
        world = wcs.pixel_to_world(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        assert torch.allclose(  # type: ignore
            world, torch.tensor([[180.0, 0.0]], dtype=torch.float64), atol=1e-6  # type: ignore
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_get_wcs_auto_hdu_skips_empty_primary():
    image = np.random.normal(size=(16, 16)).astype(np.float32)
    with tempfile.NamedTemporaryFile(suffix=".fits", delete=False) as f:
        primary = fits.PrimaryHDU()
        ext = fits.ImageHDU(image)
        ext.header["EXTNAME"] = "SCI"  # type: ignore
        ext.header["CTYPE1"] = "RA---TAN"  # type: ignore
        ext.header["CTYPE2"] = "DEC--TAN"  # type: ignore
        ext.header["CRPIX1"] = 1.0  # type: ignore
        ext.header["CRPIX2"] = 1.0  # type: ignore
        ext.header["CRVAL1"] = 10.0  # type: ignore
        ext.header["CRVAL2"] = -5.0  # type: ignore
        ext.header["CDELT1"] = -0.01  # type: ignore
        ext.header["CDELT2"] = 0.01  # type: ignore
        fits.HDUList([primary, ext]).writeto(f.name, overwrite=True)
        path = f.name

    try:
        wcs = torchsky.get_wcs(path, hdu="auto")
        world = wcs.pixel_to_world(torch.tensor([[0.0, 0.0]], dtype=torch.float64))
        assert torch.allclose(  # type: ignore
            world, torch.tensor([[10.0, -5.0]], dtype=torch.float64), atol=1e-6  # type: ignore
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)
