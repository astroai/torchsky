import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import DistortionLookupTable, WCS


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
    awcs = AstropyWCS(header, key="A")
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


def test_wcs_mixed_pixel_world_solves_coupled_linear_system():
    header = {
        "NAXIS": 3,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CTYPE3": "Z",
        "CRPIX1": 1.0,
        "CRPIX2": 2.0,
        "CRPIX3": 3.0,
        "CRVAL1": 10.0,
        "CRVAL2": 20.0,
        "CRVAL3": 30.0,
        "CD1_1": 2.0,
        "CD1_2": 0.5,
        "CD1_3": -0.25,
        "CD2_1": -0.75,
        "CD2_2": 3.0,
        "CD2_3": 0.5,
        "CD3_1": 0.25,
        "CD3_2": -0.5,
        "CD3_3": 4.0,
    }
    wcs = WCS(header)
    pixels_true = torch.tensor([[4.0, 3.0, 7.0], [6.0, 5.0, 2.0]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)
    mixed_pixel = torch.stack(
        [
            pixels_true[:, 0],
            torch.full((2,), float("nan"), dtype=torch.float64),
            torch.full((2,), float("nan"), dtype=torch.float64),
        ],
        dim=-1,
    )
    mixed_world = torch.stack(
        [
            torch.full((2,), float("nan"), dtype=torch.float64),
            world_true[:, 1],
            world_true[:, 2],
        ],
        dim=-1,
    )

    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )

    torch.testing.assert_close(solved_pixel, pixels_true, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(solved_world, world_true, atol=1e-10, rtol=1e-10)


def test_wcs_mixed_pixel_world_solves_spatial_cube_coupled_extra_axes():
    header = {
        "NAXIS": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CTYPE4": "TIME",
        "CRPIX1": 40.0,
        "CRPIX2": 30.0,
        "CRPIX3": 5.0,
        "CRPIX4": 2.0,
        "CRVAL1": 12.0,
        "CRVAL2": -5.0,
        "CRVAL3": 500.0,
        "CRVAL4": 60000.0,
        "CD1_1": -0.002,
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD1_4": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.002,
        "CD2_3": 0.0,
        "CD2_4": 0.0,
        "CD3_1": 0.1,
        "CD3_2": 0.0,
        "CD3_3": 10.0,
        "CD3_4": 0.25,
        "CD4_1": 0.0,
        "CD4_2": -0.01,
        "CD4_3": 0.05,
        "CD4_4": 0.5,
    }
    wcs = WCS(header)
    pixels_true = torch.tensor(
        [[39.0, 29.0, 7.0, 3.0], [41.5, 31.0, 2.0, 5.0]],
        dtype=torch.float64,
    )
    world_true = wcs.pixel_to_world(pixels_true)

    mixed_pixel = pixels_true.clone()
    mixed_pixel[:, 2:] = float("nan")
    mixed_world = torch.full_like(world_true, float("nan"))
    mixed_world[:, 2:] = world_true[:, 2:]

    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )

    torch.testing.assert_close(solved_pixel, pixels_true, atol=1e-10, rtol=1e-10)
    torch.testing.assert_close(solved_world, world_true, atol=2e-10, rtol=0.0)


def test_wcs_mixed_pixel_world_solves_spatial_and_coupled_extra_axes_together():
    header = {
        "NAXIS": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CTYPE4": "TIME",
        "CRPIX1": 40.0,
        "CRPIX2": 30.0,
        "CRPIX3": 5.0,
        "CRPIX4": 2.0,
        "CRVAL1": 12.0,
        "CRVAL2": -5.0,
        "CRVAL3": 500.0,
        "CRVAL4": 60000.0,
        "CD1_1": -0.002,
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD1_4": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.002,
        "CD2_3": 0.0,
        "CD2_4": 0.0,
        "CD3_1": 0.1,
        "CD3_2": 0.0,
        "CD3_3": 10.0,
        "CD3_4": 0.25,
        "CD4_1": 0.0,
        "CD4_2": -0.01,
        "CD4_3": 0.05,
        "CD4_4": 0.5,
    }
    wcs = WCS(header)
    pixels_true = torch.tensor(
        [[39.0, 29.0, 7.0, 3.0], [41.5, 31.0, 2.0, 5.0]],
        dtype=torch.float64,
    )
    world_true = wcs.pixel_to_world(pixels_true)

    mixed_pixel = torch.full_like(pixels_true, float("nan"))
    mixed_pixel[:, 0] = pixels_true[:, 0]
    mixed_world = torch.full_like(world_true, float("nan"))
    mixed_world[:, 1] = world_true[:, 1]
    mixed_world[:, 2:] = world_true[:, 2:]

    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )

    torch.testing.assert_close(solved_pixel, pixels_true, atol=3e-6, rtol=0.0)
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


def test_wcs_tan_center_pixel_to_world_jacobian_is_finite():
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
    pixels = torch.tensor([[31.0, 31.0]], dtype=torch.float64)

    jac = wcs.pixel_to_world_jacobian(pixels)[0]
    expected = torch.tensor(
        [
            [-0.001 / np.cos(np.deg2rad(35.0)), 0.0],
            [0.0, 0.001],
        ],
        dtype=torch.float64,
    )

    assert torch.isfinite(jac).all()
    torch.testing.assert_close(jac, expected, atol=1e-10, rtol=1e-10)


def test_wcs_spatial_cube_world_to_pixel_differentiable_roundtrip():
    header = {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 120.0,
        "CRVAL2": 35.0,
        "CRVAL3": 500.0,
        "CD1_1": -0.001,
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.001,
        "CD2_3": 0.0,
        "CD3_1": 0.25,
        "CD3_2": -0.5,
        "CD3_3": 10.0,
    }
    wcs = WCS(header)
    pixels = torch.tensor([[31.4, 30.6, 2.5], [32.2, 31.3, 4.0]], dtype=torch.float64)

    world = wcs.pixel_to_world_differentiable(pixels)
    pixels_back = wcs.world_to_pixel_differentiable(world)
    j_fw = wcs.pixel_to_world_jacobian(pixels)
    j_inv = wcs.world_to_pixel_jacobian(world)
    ident = torch.matmul(j_inv, j_fw)

    torch.testing.assert_close(pixels_back, pixels, atol=2e-8, rtol=2e-8)
    assert torch.isfinite(j_inv).all()
    torch.testing.assert_close(
        ident,
        torch.eye(3, dtype=torch.float64).expand_as(ident),
        atol=2e-5,
        rtol=2e-5,
    )


def test_wcs_axis_correlation_matrix_reports_extra_axis_cd_coupling():
    header = {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 120.0,
        "CRVAL2": 35.0,
        "CRVAL3": 500.0,
        "CD1_1": -0.001,
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.001,
        "CD2_3": 0.0,
        "CD3_1": 0.25,
        "CD3_2": -0.5,
        "CD3_3": 10.0,
    }
    wcs = WCS(header)

    np.testing.assert_array_equal(
        wcs.axis_correlation_matrix,
        np.array(
            [
                [True, True, False],
                [True, True, False],
                [True, True, True],
            ]
        ),
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
    x_values, y_values = wcs.world_to_pixel_values(ra, dec)
    pixels = wcs.world_to_pixel(torch.stack([ra, dec], dim=-1))
    y_index, x_index = wcs.world_to_array_index_values(ra, dec)
    ra_arr, dec_arr = wcs.array_index_to_world_values(y, x)

    torch.testing.assert_close(pixels[:, 0], x, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(pixels[:, 1], y, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(x_values, x, atol=2e-7, rtol=1e-9)
    torch.testing.assert_close(y_values, y, atol=2e-7, rtol=1e-9)
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


def test_wcs_slice_negative_step_flips_cd_and_roundtrips() -> None:
    header = {
        "NAXIS": 2,
        "NAXIS1": 20,
        "NAXIS2": 12,
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
    sliced = wcs.slice((slice(None, None, -1), slice(None, None, -1)))
    sliced_pixels = torch.tensor([[0.0, 0.0], [3.0, 5.0]], dtype=torch.float64)
    original_pixels = torch.stack(
        [
            (header["NAXIS1"] - 1) - sliced_pixels[:, 0],
            (header["NAXIS2"] - 1) - sliced_pixels[:, 1],
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        sliced.pixel_to_world(sliced_pixels),
        wcs.pixel_to_world(original_pixels),
        atol=1e-12,
        rtol=0.0,
    )
    assert sliced.pixel_shape == (20, 12)


def test_wcs_slice_integer_index_drops_numpy_order_spectral_axis():
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
    sliced = cube_wcs.slice((-1, slice(10, 50), slice(20, 70)))
    pixels_2d = torch.tensor([[0.0, 0.0], [15.0, 20.0]], dtype=torch.float64)
    pixels_3d = torch.stack(
        [
            20.0 + pixels_2d[:, 0],
            10.0 + pixels_2d[:, 1],
            torch.full((pixels_2d.shape[0],), 11.0, dtype=torch.float64),
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        sliced.pixel_to_world(pixels_2d),
        cube_wcs.pixel_to_world(pixels_3d)[:, :2],
        atol=2e-10,
        rtol=0.0,
    )
    assert sliced.pixel_n_dim == 2
    assert sliced.pixel_shape == (50, 40)


def test_wcs_slice_integer_index_drops_uncoupled_fits_order_axis():
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
    sliced = wcs.slice((slice(2, 8), 4, slice(5, 20, 3)), numpy_order=False)
    pixels_2d = torch.tensor([[0.0, 0.0], [3.0, 2.0]], dtype=torch.float64)
    pixels_3d = torch.stack(
        [
            2.0 + pixels_2d[:, 0],
            torch.full((pixels_2d.shape[0],), 4.0, dtype=torch.float64),
            5.0 + 3.0 * pixels_2d[:, 1],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        sliced.pixel_to_world(pixels_2d),
        wcs.pixel_to_world(pixels_3d)[:, [0, 2]],
    )
    assert sliced.world_axis_names == ("X", "Z")
    assert sliced.pixel_shape == (6, 5)


def test_wcs_slice_integer_index_folds_coupled_linear_axis_into_crval():
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
    reduced = wcs.slice((slice(None), 4, slice(None)), numpy_order=False)
    pixels_2d = torch.tensor([[0.0, 0.0], [3.0, 2.0]], dtype=torch.float64)
    pixels_3d = torch.stack(
        [
            pixels_2d[:, 0],
            torch.full((pixels_2d.shape[0],), 4.0, dtype=torch.float64),
            pixels_2d[:, 1],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        reduced.pixel_to_world(pixels_2d),
        wcs.pixel_to_world(pixels_3d)[:, [0, 2]],
    )
    assert reduced.world_axis_names == ("X", "Z")


def test_wcs_slice_preserves_spatial_cube_extra_axis_cd_scale():
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
    sliced = cube_wcs.slice((slice(2, 8, 2), slice(10, 50), slice(20, 70)))
    pixels_3d_sliced = torch.tensor(
        [[0.0, 0.0, 0.0], [15.0, 20.0, 2.0]], dtype=torch.float64
    )
    pixels_3d_original = torch.stack(
        [
            20.0 + pixels_3d_sliced[:, 0],
            10.0 + pixels_3d_sliced[:, 1],
            2.0 + 2.0 * pixels_3d_sliced[:, 2],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        sliced.pixel_to_world(pixels_3d_sliced),
        cube_wcs.pixel_to_world(pixels_3d_original),
        atol=2e-10,
        rtol=0.0,
    )
    assert sliced.pixel_shape == (50, 40, 3)


def test_wcs_sub_reorders_uncoupled_linear_axes():
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
        "CD1_1": 2.0,
        "CD1_2": 0.25,
        "CD1_3": 0.0,
        "CD2_1": -0.5,
        "CD2_2": 3.0,
        "CD2_3": 0.0,
        "CD3_1": 0.0,
        "CD3_2": 0.0,
        "CD3_3": 4.0,
    }
    wcs = WCS(header)
    reordered = wcs.sub([1, 0, 2])
    pixels = torch.tensor([[4.0, 2.0, 5.0], [6.0, 3.0, 8.0]], dtype=torch.float64)

    torch.testing.assert_close(
        reordered.pixel_to_world(pixels[:, [1, 0, 2]]),
        wcs.pixel_to_world(pixels)[:, [1, 0, 2]],
    )
    assert reordered.world_axis_names == ("Y", "X", "Z")
    assert reordered.pixel_shape == (20, 10, 30)


def test_wcs_sub_reorders_spatial_extra_axes_after_celestial_pair():
    header = {
        "NAXIS": 4,
        "NAXIS1": 80,
        "NAXIS2": 60,
        "NAXIS3": 12,
        "NAXIS4": 7,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE",
        "CTYPE4": "TIME",
        "CRPIX1": 40.0,
        "CRPIX2": 30.0,
        "CRPIX3": 1.0,
        "CRPIX4": 2.0,
        "CRVAL1": 12.0,
        "CRVAL2": -5.0,
        "CRVAL3": 500.0,
        "CRVAL4": 60000.0,
        "CDELT1": -0.002,
        "CDELT2": 0.002,
        "CDELT3": 10.0,
        "CDELT4": 0.5,
    }
    wcs = WCS(header)
    reordered = wcs.sub([0, 1, 3, 2])
    pixels = torch.tensor(
        [[39.0, 29.0, 3.0, 4.0], [42.0, 31.0, 5.0, 1.0]], dtype=torch.float64
    )

    torch.testing.assert_close(
        reordered.pixel_to_world(pixels[:, [0, 1, 3, 2]]),
        wcs.pixel_to_world(pixels)[:, [0, 1, 3, 2]],
        atol=2e-10,
        rtol=0.0,
    )
    assert reordered.world_axis_names == ("RA---TAN", "DEC--TAN", "TIME", "WAVE")
    assert reordered.pixel_shape == (80, 60, 7, 12)


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


def test_wcs_dropaxis_bakes_coupled_axis_at_crpix():
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
    reduced = wcs.dropaxis(1)
    pixels_2d = torch.tensor([[0.0, 2.0], [4.0, 5.0]], dtype=torch.float64)
    # dropaxis bakes axis 1 at CRPIX (0-based pixel crpix-1 = 0).
    pixels_3d = torch.stack(
        [
            pixels_2d[:, 0],
            torch.zeros(pixels_2d.shape[0], dtype=torch.float64),
            pixels_2d[:, 1],
        ],
        dim=-1,
    )

    torch.testing.assert_close(
        reduced.pixel_to_world(pixels_2d),
        wcs.pixel_to_world(pixels_3d)[:, [0, 2]],
    )
    assert reduced.world_axis_names == ("X", "Z")


def test_wcs_dropaxis_bakes_celestial_axis_at_crpix():
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
    cube = WCS(header)
    reduced = cube.dropaxis(1)
    pixels_2d = torch.tensor([[35.0, 2.0], [42.0, 5.0]], dtype=torch.float64)
    pixels_3d = torch.stack(
        [
            pixels_2d[:, 0],
            torch.full((2,), 29.0, dtype=torch.float64),  # CRPIX2 - 1
            pixels_2d[:, 1],
        ],
        dim=-1,
    )
    torch.testing.assert_close(
        reduced.pixel_to_world(pixels_2d),
        cube.pixel_to_world(pixels_3d)[:, [0, 2]],
        atol=2e-10,
        rtol=0.0,
    )


def test_wcs_mixed_linear_known_mask_varies_by_row():
    header = {
        "NAXIS": 2,
        "CTYPE1": "X",
        "CTYPE2": "Y",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 2.0,
        "CD1_2": 0.5,
        "CD2_1": -0.25,
        "CD2_2": 3.0,
    }
    wcs = WCS(header)
    pixels_true = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)
    mixed_pixel = torch.tensor(
        [[1.0, float("nan")], [float("nan"), 4.0]], dtype=torch.float64
    )
    mixed_world = torch.tensor(
        [[float("nan"), world_true[0, 1]], [world_true[1, 0], float("nan")]],
        dtype=torch.float64,
    )
    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )
    torch.testing.assert_close(solved_pixel, pixels_true, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(solved_world, world_true, atol=1e-10, rtol=0.0)


def test_wcs_sip_with_pixel_affine_correction_roundtrip():
    header = {
        "NAXIS": 2,
        "NAXIS1": 1024,
        "NAXIS2": 1024,
        "CTYPE1": "RA---TAN-SIP",
        "CTYPE2": "DEC--TAN-SIP",
        "CRPIX1": 512.0,
        "CRPIX2": 512.0,
        "CRVAL1": 180.0,
        "CRVAL2": 5.0,
        "CD1_1": -2.8e-4,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 2.8e-4,
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 3.0e-6,
        "A_0_2": 1.0e-6,
        "B_2_0": -2.5e-6,
        "B_0_2": 2.5e-6,
        "AP_ORDER": 2,
        "BP_ORDER": 2,
        "AP_2_0": -2.0e-6,
        "BP_0_2": -2.0e-6,
    }
    wcs = WCS(header)
    matrix = torch.tensor([[1.02, 0.01], [-0.015, 0.98]], dtype=torch.float64)
    offset = torch.tensor([1.5, -2.0], dtype=torch.float64)
    source = torch.tensor(
        [[480.0, 500.0], [512.0, 512.0], [530.0, 490.0]], dtype=torch.float64
    )
    corrected_pixels = torch.matmul(source, matrix.T) + offset
    corrected = wcs.with_pixel_affine_correction(matrix, offset)
    assert corrected._pixel_affine_matrix is not None
    torch.testing.assert_close(
        corrected.pixel_to_world(corrected_pixels),
        wcs.pixel_to_world(source),
        atol=2e-6,
        rtol=0.0,
    )
    world = wcs.pixel_to_world(source)
    # SIP inverse is approximate; corrected world->pixel inherits that residual.
    torch.testing.assert_close(
        corrected.world_to_pixel(world),
        corrected_pixels,
        atol=5e-3,
        rtol=0.0,
    )
    assert corrected.to_header()["TSKYAF0"] == "pixel_affine"


def test_wcs_pixel_affine_correction_coupled_extra_axis():
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
        "CD1_2": 0.0,
        "CD1_3": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
        "CD2_3": 0.0,
        "CD3_1": 0.2,
        "CD3_2": -0.1,
        "CD3_3": 1.0,
    }
    wcs = WCS(header)
    matrix = torch.tensor([[1.02, 0.01], [-0.015, 0.98]], dtype=torch.float64)
    offset = torch.tensor([1.5, -2.0], dtype=torch.float64)
    source = torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0]], dtype=torch.float64)
    xy = source[:, :2]
    corrected_xy = torch.matmul(xy, matrix.T) + offset
    corrected_pixels = torch.cat([corrected_xy, source[:, 2:]], dim=-1)
    corrected = wcs.with_pixel_affine_correction(matrix, offset)
    torch.testing.assert_close(
        corrected.pixel_to_world(corrected_pixels),
        wcs.pixel_to_world(source),
        atol=1e-10,
        rtol=0.0,
    )


@pytest.mark.parametrize("unit,scale", [("rad", np.pi / 180.0), ("arcmin", 60.0)])
def test_celestial_non_degree_units_match_astropy_and_serialize(
    unit: str, scale: float
) -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": unit,
        "CUNIT2": unit,
        "CRPIX1": 32.0,
        "CRPIX2": 24.0,
        "CRVAL1": 150.0 * scale,
        "CRVAL2": -20.0 * scale,
        "CD1_1": -0.02 * scale,
        "CD1_2": 0.003 * scale,
        "CD2_1": 0.002 * scale,
        "CD2_2": 0.018 * scale,
    }
    pixels = np.array([[31.0, 23.0], [35.5, 19.25], [12.0, 40.0]])
    actual = WCS(header)
    expected = AstropyWCS(header)

    np.testing.assert_allclose(
        actual.pixel_to_world(torch.from_numpy(pixels)).numpy(),
        expected.all_pix2world(pixels, 0),
        atol=2e-10,
        rtol=0.0,
    )
    serialized = actual.to_header()
    assert serialized["CUNIT1"] == unit
    assert serialized["CUNIT2"] == unit
    assert serialized["CRVAL1"] == pytest.approx(header["CRVAL1"])
    assert serialized["CRVAL2"] == pytest.approx(header["CRVAL2"])
    for i in (1, 2):
        for j in (1, 2):
            assert serialized[f"CD{i}_{j}"] == pytest.approx(header[f"CD{i}_{j}"])


def test_general_nd_inverse_accepts_noncontiguous_stacked_input() -> None:
    wcs = WCS(
        {
            "NAXIS": 3,
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
    )
    pixels = torch.arange(24, dtype=torch.float64).reshape(2, 4, 3).transpose(0, 1)
    assert not pixels.is_contiguous()
    world = wcs.pixel_to_world(pixels)
    world_noncontiguous = world.transpose(0, 1)
    assert not world_noncontiguous.is_contiguous()
    torch.testing.assert_close(
        wcs.world_to_pixel(world_noncontiguous), pixels.transpose(0, 1)
    )


def test_separate_coordinate_apis_broadcast_compatible_shapes() -> None:
    wcs = WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CD1_1": 1.0,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1.0,
        }
    )
    x = torch.tensor([[0.0], [1.0]], dtype=torch.float64)
    y = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float64)
    ra, dec = wcs.pixel_to_world(x, y)
    assert ra.shape == dec.shape == (2, 3)
    px, py = wcs.world_to_pixel(ra[:, :1], dec[:1, :])
    assert px.shape == py.shape == (2, 3)


def test_separable_zero_cd_terms_do_not_spread_nan_between_axes() -> None:
    wcs = WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "RA---CAR",
            "CTYPE2": "DEC--CAR",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CD1_1": 1.0,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 1.0,
        }
    )
    ra, dec = wcs.pixel_to_world(
        torch.tensor([torch.nan], dtype=torch.float64),
        torch.tensor([3.0], dtype=torch.float64),
    )
    assert torch.isnan(ra).all()
    torch.testing.assert_close(dec, torch.tensor([3.0], dtype=torch.float64))
    x, y = wcs.world_to_pixel(
        torch.tensor([torch.nan], dtype=torch.float64),
        torch.tensor([4.0], dtype=torch.float64),
    )
    assert torch.isnan(x).all()
    torch.testing.assert_close(y, torch.tensor([4.0], dtype=torch.float64))


def _distorted_tan_header(*, tpv: bool = False) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "NAXIS1": 64,
        "NAXIS2": 64,
        "CTYPE1": "RA---TPV" if tpv else "RA---TAN-SIP",
        "CTYPE2": "DEC--TPV" if tpv else "DEC--TAN-SIP",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRVAL1": 50.0,
        "CRVAL2": -20.0,
        "CD1_1": -1.0e-3,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0e-3,
    }
    if tpv:
        header.update({"PV1_1": 1.0, "PV1_4": 1.0e-4, "PV2_1": 1.0})
    else:
        header.update(
            {
                "A_ORDER": 2,
                "B_ORDER": 2,
                "A_2_0": 2.0e-5,
                "B_0_2": -2.0e-5,
            }
        )
    return header


@pytest.mark.parametrize("tpv", [False, True])
def test_identity_slice_preserves_sip_or_tpv_state(tpv: bool) -> None:
    wcs = WCS(_distorted_tan_header(tpv=tpv))
    identity = wcs.slice((slice(None), slice(None)))
    pixels = torch.tensor([[20.0, 25.0], [38.0, 41.0]], dtype=torch.float64)
    torch.testing.assert_close(
        identity.pixel_to_world(pixels), wcs.pixel_to_world(pixels)
    )
    assert (identity.tpv is not None) == tpv
    assert (identity.sip is not None) == (not tpv)


def test_identity_slice_and_affine_preserve_lut_and_existing_pixel_affine() -> None:
    table = DistortionLookupTable(
        torch.tensor([[0.0, 0.01], [0.02, 0.03]], dtype=torch.float64),
        crpix=(1.0, 1.0),
        crval=(1.0, 1.0),
        cdelt=(64.0, 64.0),
    )
    wcs = WCS(_distorted_tan_header()).with_pixel_affine_correction(
        torch.eye(2), torch.tensor([0.25, -0.5])
    )
    wcs.cpdis1 = table
    wcs.cpdis2 = table
    wcs.det2im1 = table
    wcs.det2im2 = table
    pixels = torch.tensor([[20.0, 25.0], [38.0, 41.0]], dtype=torch.float64)
    for identity in (
        wcs.slice((slice(None), slice(None))),
        wcs.with_pixel_affine_correction(torch.eye(2), torch.zeros(2)),
    ):
        torch.testing.assert_close(
            identity.pixel_to_world(pixels), wcs.pixel_to_world(pixels)
        )
        assert identity.cpdis1 is not None and identity.cpdis2 is not None
        assert identity.det2im1 is not None and identity.det2im2 is not None
        assert identity._pixel_affine_matrix is not None


def test_lut_rebasing_and_pixel_affine_composition_are_supported() -> None:
    table = DistortionLookupTable(
        torch.tensor([[0.0, 0.01], [0.02, 0.03]], dtype=torch.float64),
        crpix=(1.0, 1.0),
        crval=(1.0, 1.0),
        cdelt=(64.0, 64.0),
    )
    wcs = WCS(_distorted_tan_header(), cpdis1=table)
    pixels = torch.tensor([[4.0, 5.0], [12.0, 18.0]], dtype=torch.float64)
    sliced = wcs.slice((slice(1, None), slice(None)))
    torch.testing.assert_close(
        sliced.pixel_to_world(pixels),
        wcs.pixel_to_world(pixels + torch.tensor([0.0, 1.0])),
    )

    matrix = torch.tensor([[1.0, 0.1], [0.0, 1.0]], dtype=torch.float64)
    corrected = wcs.with_pixel_affine_correction(matrix, torch.zeros(2))
    torch.testing.assert_close(
        corrected.pixel_to_world(pixels @ matrix.T), wcs.pixel_to_world(pixels)
    )
