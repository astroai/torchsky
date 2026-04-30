import os
import tempfile

import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

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
        world,
        torch.tensor([[180.0, 0.0]], dtype=torch.float64),
        atol=1e-6,  # type: ignore
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
        pixels,
        torch.tensor([[0.0, 0.0]], dtype=torch.float64),
        atol=1e-6,  # type: ignore
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
            world,
            torch.tensor([[180.0, 0.0]], dtype=torch.float64),
            atol=1e-6,  # type: ignore
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
            world,
            torch.tensor([[10.0, -5.0]], dtype=torch.float64),
            atol=1e-6,  # type: ignore
        )
    finally:
        if os.path.exists(path):
            os.unlink(path)


def test_wcs_alternate_key_matches_astropy():
    header = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 50.0,
        "CRPIX2": 50.0,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CTYPE1A": "RA---TAN",
        "CTYPE2A": "DEC--TAN",
        "CRPIX1A": 25.0,
        "CRPIX2A": 25.0,
        "CRVAL1A": 120.0,
        "CRVAL2A": -30.0,
        "CDELT1A": -0.02,
        "CDELT2A": 0.02,
        "CUNIT1A": "deg",
        "CUNIT2A": "deg",
    }
    assert WCS.available_wcs_keys(header) == ("", "A")

    twcs = WCS(header, key="A")
    awcs = AstropyWCS(fits.Header(header), key="A")
    xy = np.array([[0.0, 0.0], [24.0, 24.0], [70.0, 20.0]], dtype=np.float64)

    world_t = twcs.pixel_to_world(torch.from_numpy(xy), origin=0)
    world_a = awcs.all_pix2world(xy, 0)

    np.testing.assert_allclose(world_t.cpu().numpy(), world_a, atol=2e-8)
    assert twcs.wcs_key == "A"
    assert twcs.world_axis_units == ("deg", "deg")
    assert twcs.pixel_shape == (100, 100)
    assert twcs.array_shape == (100, 100)
    assert twcs.pixel_bounds == [(0.0, 99.0), (0.0, 99.0)]


def test_wcs_axis_metadata_and_footprint():
    header = {
        "NAXIS": 3,
        "NAXIS1": 10,
        "NAXIS2": 12,
        "NAXIS3": 5,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CUNIT3": "Angstrom",
        "CRPIX1": 5.0,
        "CRPIX2": 6.0,
        "CRPIX3": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 5000.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CDELT3": 2.0,
    }
    wcs = WCS(header)

    assert wcs.pixel_n_dim == 3
    assert wcs.world_n_dim == 3
    assert wcs.world_axis_names == ("RA---TAN", "DEC--TAN", "WAVE")
    assert wcs.world_axis_units == ("deg", "deg", "Angstrom")
    assert wcs.world_axis_physical_types == ("pos.eq.ra", "pos.eq.dec", "em.wave")
    np.testing.assert_array_equal(
        wcs.axis_correlation_matrix,
        np.array([[True, True, False], [True, True, False], [False, False, True]]),
    )
    footprint = wcs.footprint()
    assert footprint.shape == (4, 2)
    assert torch.isfinite(footprint).all()


def test_wcs_mixed_pixel_world_solves_single_spatial_axis():
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 64.0,
        "CRPIX2": 64.0,
        "CRVAL1": 180.0,
        "CRVAL2": 10.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }
    wcs = WCS(header)
    pixels_true = torch.tensor([[60.0, 61.0], [63.0, 65.0]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)

    mixed_pixel = torch.stack(
        [pixels_true[:, 0], torch.full((2,), float("nan"), dtype=torch.float64)],
        dim=-1,
    )
    mixed_world = torch.stack(
        [
            torch.full((2,), float("nan"), dtype=torch.float64),
            world_true[:, 1],
        ],
        dim=-1,
    )

    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )

    torch.testing.assert_close(solved_pixel, pixels_true, atol=2e-6, rtol=0.0)
    torch.testing.assert_close(solved_world, world_true, atol=2e-8, rtol=0.0)


def test_wcs_differentiable_pixel_to_world_supports_autograd_and_jacobian():
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 64.0,
        "CRPIX2": 64.0,
        "CRVAL1": 30.0,
        "CRVAL2": -15.0,
        "CDELT1": -0.002,
        "CDELT2": 0.002,
    }
    wcs = WCS(header)
    pixels = torch.tensor([[63.5, 64.25]], dtype=torch.float64, requires_grad=True)

    world = wcs.pixel_to_world_differentiable(pixels)
    loss = (world[..., 0] * 0.1 + world[..., 1]).sum()
    loss.backward()

    assert pixels.grad is not None
    assert torch.isfinite(pixels.grad).all()

    jac = wcs.pixel_to_world_jacobian(pixels.detach())
    eps = 1.0e-4
    delta_x = torch.tensor([[eps, 0.0]], dtype=torch.float64)
    finite_dx = (
        wcs.pixel_to_world_differentiable(pixels.detach() + delta_x)
        - wcs.pixel_to_world_differentiable(pixels.detach() - delta_x)
    ) / (2.0 * eps)

    torch.testing.assert_close(jac[0, :, 0], finite_dx[0], atol=2e-6, rtol=2e-6)


def test_wcs_differentiable_world_to_pixel_jacobian_inverts_forward_jacobian():
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRVAL1": 120.0,
        "CRVAL2": 35.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }
    wcs = WCS(header)
    pixels = torch.tensor([[31.25, 33.5]], dtype=torch.float64)
    world = wcs.pixel_to_world_differentiable(pixels)

    j_fw = wcs.pixel_to_world_jacobian(pixels)[0]
    j_inv = wcs.world_to_pixel_jacobian(world)[0]
    ident = j_inv @ j_fw

    torch.testing.assert_close(
        ident, torch.eye(2, dtype=torch.float64), atol=2e-5, rtol=2e-5
    )


def test_wcs_value_aliases_and_array_index_helpers_match_pixel_world():
    header = {
        "NAXIS": 2,
        "NAXIS1": 128,
        "NAXIS2": 96,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 64.0,
        "CRPIX2": 48.0,
        "CRVAL1": 30.0,
        "CRVAL2": -15.0,
        "CDELT1": -0.002,
        "CDELT2": 0.002,
    }
    wcs = WCS(header)
    x = torch.tensor([62.2, 65.8], dtype=torch.float64)
    y = torch.tensor([47.4, 51.1], dtype=torch.float64)

    ra, dec = wcs.pixel_to_world_values(x, y)
    pixels = wcs.world_to_pixel(torch.stack([ra, dec], dim=-1))
    y_index, x_index = wcs.world_to_array_index_values(ra, dec)
    ra_arr, dec_arr = wcs.array_index_to_world_values(y, x)

    torch.testing.assert_close(pixels[:, 0], x, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(pixels[:, 1], y, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(ra_arr, ra)
    torch.testing.assert_close(dec_arr, dec)
    torch.testing.assert_close(x_index, torch.round(x).to(dtype=torch.long))
    torch.testing.assert_close(y_index, torch.round(y).to(dtype=torch.long))


def test_wcs_astropy_all_pix_world_aliases_accept_stacked_and_separate_inputs():
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 64.0,
        "CRPIX2": 64.0,
        "CRVAL1": 30.0,
        "CRVAL2": -15.0,
        "CDELT1": -0.002,
        "CDELT2": 0.002,
    }
    wcs = WCS(header)
    pixels = torch.tensor([[62.0, 63.0], [65.0, 66.0]], dtype=torch.float64)

    world_stacked = wcs.all_pix2world(pixels, 0)
    world_kw = wcs.all_pix2world(pixels, origin=0)
    ra, dec = wcs.all_pix2world(pixels[:, 0], pixels[:, 1], 0)
    pixels_back = wcs.all_world2pix(world_stacked, 0)
    x_back, y_back = wcs.all_world2pix(ra, dec, 0)

    assert isinstance(world_stacked, torch.Tensor)
    torch.testing.assert_close(world_stacked, world_kw)
    torch.testing.assert_close(world_stacked[:, 0], ra)
    torch.testing.assert_close(world_stacked[:, 1], dec)
    torch.testing.assert_close(pixels_back, pixels, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(x_back, pixels[:, 0], atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(y_back, pixels[:, 1], atol=2e-7, rtol=1e-9)


def test_wcs_array_index_helpers_use_numpy_order_for_3d_linear_wcs():
    header = {
        "NAXIS": 3,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "WAVE",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRPIX3": 1.0,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CRVAL3": 400.0,
        "CDELT1": 2.0,
        "CDELT2": 3.0,
        "CDELT3": 5.0,
    }
    wcs = WCS(header)
    z = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([1.0, 3.0], dtype=torch.float64)
    x = torch.tensor([4.0, 6.0], dtype=torch.float64)

    wx, wy, ww = wcs.array_index_to_world_values(z, y, x)
    zi, yi, xi = wcs.world_to_array_index_values(wx, wy, ww)

    torch.testing.assert_close(wx, torch.tensor([18.0, 22.0], dtype=torch.float64))
    torch.testing.assert_close(wy, torch.tensor([23.0, 29.0], dtype=torch.float64))
    torch.testing.assert_close(ww, torch.tensor([400.0, 410.0], dtype=torch.float64))
    torch.testing.assert_close(zi, z.to(dtype=torch.long))
    torch.testing.assert_close(yi, y.to(dtype=torch.long))
    torch.testing.assert_close(xi, x.to(dtype=torch.long))


def test_wcs_slice_updates_crpix_for_numpy_order_cutout():
    header = {
        "NAXIS": 2,
        "NAXIS1": 120,
        "NAXIS2": 80,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 60.0,
        "CRPIX2": 40.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CDELT1": -0.001,
        "CDELT2": 0.001,
    }
    wcs = WCS(header)
    cutout = wcs.slice((slice(10, 50), slice(20, 90)))
    cutout_pixels = torch.tensor([[0.0, 0.0], [15.0, 20.0]], dtype=torch.float64)
    original_pixels = cutout_pixels + torch.tensor([20.0, 10.0], dtype=torch.float64)

    torch.testing.assert_close(
        cutout.pixel_to_world(cutout_pixels),
        wcs.pixel_to_world(original_pixels),
        atol=2e-10,
        rtol=0.0,
    )
    assert cutout.pixel_shape == (70, 40)


def test_wcs_slice_updates_cd_for_strided_view():
    header = {
        "NAXIS": 2,
        "NAXIS1": 100,
        "NAXIS2": 100,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 2.0,
        "CDELT2": 3.0,
    }
    wcs = WCS(header)
    sliced = wcs.slice((slice(4, 24, 2), slice(6, 36, 3)))
    sliced_pixels = torch.tensor([[0.0, 0.0], [2.0, 5.0]], dtype=torch.float64)
    original_pixels = torch.stack(
        [6.0 + 3.0 * sliced_pixels[:, 0], 4.0 + 2.0 * sliced_pixels[:, 1]],
        dim=-1,
    )

    torch.testing.assert_close(
        sliced.pixel_to_world(sliced_pixels),
        wcs.pixel_to_world(original_pixels),
    )
    assert sliced.pixel_shape == (10, 10)


def test_wcs_celestial_extracts_spatial_wcs_from_cube():
    header = {
        "NAXIS": 3,
        "NAXIS1": 80,
        "NAXIS2": 60,
        "NAXIS3": 12,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 40.0,
        "CRPIX2": 30.0,
        "CRPIX3": 1.0,
        "CRVAL1": 12.0,
        "CRVAL2": -5.0,
        "CRVAL3": 500.0,
        "CDELT1": -0.002,
        "CDELT2": 0.002,
        "CDELT3": 10.0,
    }
    cube_wcs = WCS(header)
    sky_wcs = cube_wcs.celestial
    pixels_2d = torch.tensor([[39.0, 29.0], [42.5, 31.0]], dtype=torch.float64)
    pixels_3d = torch.cat(
        [pixels_2d, torch.zeros((pixels_2d.shape[0], 1), dtype=torch.float64)],
        dim=-1,
    )

    torch.testing.assert_close(
        sky_wcs.pixel_to_world(pixels_2d),
        cube_wcs.pixel_to_world(pixels_3d)[:, :2],
        atol=2e-10,
        rtol=0.0,
    )
    assert sky_wcs.pixel_n_dim == 2
    assert sky_wcs.pixel_shape == (80, 60)


def test_wcs_dropaxis_removes_uncoupled_linear_axis():
    header = {
        "NAXIS": 3,
        "NAXIS1": 10,
        "NAXIS2": 20,
        "NAXIS3": 30,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "Z",
        "CRPIX1": 1.0,
        "CRPIX2": 2.0,
        "CRPIX3": 3.0,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CRVAL3": 30.0,
        "CDELT1": 2.0,
        "CDELT2": 3.0,
        "CDELT3": 4.0,
    }
    wcs = WCS(header)
    reduced = wcs.dropaxis(1)
    pixels_2d = torch.tensor([[0.0, 2.0], [4.0, 5.0]], dtype=torch.float64)
    pixels_3d = torch.stack(
        [
            pixels_2d[:, 0],
            torch.full((pixels_2d.shape[0],), 1.0, dtype=torch.float64),
            pixels_2d[:, 1],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        reduced.pixel_to_world(pixels_2d),
        wcs.pixel_to_world(pixels_3d)[:, [0, 2]],
    )
    assert reduced.world_axis_names == ("X", "Z")
    assert reduced.pixel_shape == (10, 30)


def test_wcs_dropaxis_rejects_coupled_axis():
    header = {
        "NAXIS": 3,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "Z",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRPIX3": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRVAL3": 0.0,
        "CD1_1": 1.0,
        "CD1_2": 0.1,
        "CD1_3": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
        "CD2_3": 0.0,
        "CD3_1": 0.0,
        "CD3_2": 0.0,
        "CD3_3": 1.0,
    }
    wcs = WCS(header)

    with pytest.raises(NotImplementedError, match="coupled"):
        wcs.dropaxis(1)
