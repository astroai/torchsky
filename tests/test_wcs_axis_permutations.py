import numpy as np
import pytest
import torch
from astropy.wcs import DistortionLookupTable as AstropyDistortionLookupTable
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import CoupledTabularLookupWCS, DistortionLookupTable, WCS


def _permuted_header(*, use_pc: bool = False) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {
        "NAXIS": 4,
        "NAXIS1": 7,
        "NAXIS2": 48,
        "NAXIS3": 9,
        "NAXIS4": 64,
        "CTYPE1": "TIME",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "FREQ",
        "CTYPE4": "RA---TAN",
        "CUNIT1": "s",
        "CUNIT2": "deg",
        "CUNIT3": "Hz",
        "CUNIT4": "deg",
        "CRPIX1": 2.0,
        "CRPIX2": 24.0,
        "CRPIX3": 3.0,
        "CRPIX4": 32.0,
        "CRVAL1": 100.0,
        "CRVAL2": -12.0,
        "CRVAL3": 1.4e9,
        "CRVAL4": 130.0,
    }
    matrix = np.array(
        [
            [2.0, 0.0, 0.05, 0.0],
            [0.0, 8.0e-4, 0.0, 4.0e-5],
            [2.0e4, 0.0, 1.0e6, 0.0],
            [0.0, -3.0e-5, 0.0, -1.1e-3],
        ],
        dtype=np.float64,
    )
    if use_pc:
        cdelt = np.array([2.0, 8.0e-4, 1.0e6, -1.1e-3])
        for axis, value in enumerate(cdelt, 1):
            header[f"CDELT{axis}"] = float(value)
        pc = matrix / cdelt[:, None]
        for row in range(4):
            for column in range(4):
                header[f"PC{row + 1}_{column + 1}"] = float(pc[row, column])
    else:
        for row in range(4):
            for column in range(4):
                header[f"CD{row + 1}_{column + 1}"] = float(matrix[row, column])
    return header


@pytest.mark.parametrize("use_pc", [False, True])
def test_permuted_reversed_4d_pc_cd_matches_wcslib(use_pc: bool) -> None:
    header = _permuted_header(use_pc=use_pc)
    pixels = np.array(
        [[1.0, 23.0, 2.0, 31.0], [3.5, 27.0, 5.0, 36.0]], dtype=np.float64
    )
    actual = WCS(header)
    expected = AstropyWCS(header)

    world = actual.pixel_to_world(torch.from_numpy(pixels))
    np.testing.assert_allclose(
        world.numpy(), expected.all_pix2world(pixels, 0), atol=2e-8
    )
    torch.testing.assert_close(
        actual.world_to_pixel(world), torch.from_numpy(pixels), atol=2e-8, rtol=0
    )
    np.testing.assert_array_equal(
        actual.axis_correlation_matrix, expected.axis_correlation_matrix
    )
    assert actual.world_axis_physical_types == tuple(expected.world_axis_physical_types)


def test_reversed_2d_stacked_separate_broadcast_and_noncontiguous_order() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "DEC--TAN",
        "CTYPE2": "RA---TAN",
        "CRPIX1": 20.0,
        "CRPIX2": 30.0,
        "CRVAL1": 5.0,
        "CRVAL2": 10.0,
        "CD1_1": 0.01,
        "CD1_2": 3.0e-4,
        "CD2_1": -2.0e-4,
        "CD2_2": -0.01,
    }
    wcs = WCS(header)
    storage = torch.tensor(
        [[19.0, -1.0, 29.0], [21.0, -1.0, 31.0]], dtype=torch.float64
    )
    pixels = storage[:, ::2]
    assert not pixels.is_contiguous()
    stacked = wcs.pixel_to_world(pixels)
    first, second = wcs.pixel_to_world(pixels[:, 0], pixels[:, 1])
    broadcast = wcs.pixel_to_world(pixels[:, 0], pixels[:1, 1])

    torch.testing.assert_close(stacked, torch.stack((first, second), dim=-1))
    assert broadcast[0].shape == (2,)
    assert broadcast[1].shape == (2,)
    torch.testing.assert_close(wcs.world_to_pixel(stacked), pixels, atol=2e-9, rtol=0)


def test_permuted_celestial_with_coupled_spectral_time_tabular_axes() -> None:
    wave_index = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    time_index = torch.tensor([0.0, 1.0], dtype=torch.float64)
    wave_grid = wave_index[:, None].expand(3, 2)
    time_grid = time_index[None, :].expand(3, 2)
    lookup = CoupledTabularLookupWCS(
        (wave_index, time_index),
        torch.stack(
            (
                500.0 + 10.0 * wave_grid + 2.0 * time_grid,
                60000.0 + 0.25 * wave_grid + time_grid,
            ),
            dim=-1,
        ),
    )
    wcs = WCS(
        {
            "NAXIS": 4,
            "CTYPE1": "WAVE-TAB",
            "CTYPE2": "DEC--TAN",
            "CTYPE3": "TIME-TAB",
            "CTYPE4": "RA---TAN",
            "CRPIX1": 1.0,
            "CRPIX2": 24.0,
            "CRPIX3": 1.0,
            "CRPIX4": 32.0,
            "CRVAL1": 0.0,
            "CRVAL2": -12.0,
            "CRVAL3": 0.0,
            "CRVAL4": 130.0,
            "CDELT1": 1.0,
            "CDELT2": 8.0e-4,
            "CDELT3": 1.0,
            "CDELT4": -1.1e-3,
        },
        coupled_tabular=lookup,
        coupled_tabular_axes=(1, 3),
        coupled_tabular_pixel_axes=(1, 3),
    )
    pixels = torch.tensor(
        [[0.5, 23.0, 0.25, 31.0], [1.5, 26.0, 0.75, 35.0]],
        dtype=torch.float64,
    )
    world = wcs.pixel_to_world(pixels)

    torch.testing.assert_close(wcs.world_to_pixel(world), pixels, atol=3e-8, rtol=0)
    assert wcs.spectral_axis_indices == (0,)
    assert wcs.temporal_axis_indices == (2,)


def test_permuted_slice_affine_chain_serialization_and_celestial_extract() -> None:
    parent = WCS(_permuted_header())
    view = (slice(1, 8, 2), slice(2, 9), slice(4, 44, 2), slice(1, 7))
    child = parent.slice(view)
    pixels = torch.tensor(
        [[0.5, 2.0, 3.0, 1.0], [2.0, 8.0, 1.0, 2.0]], dtype=torch.float64
    )
    source = torch.stack(
        (
            1.0 + pixels[:, 0],
            4.0 + 2.0 * pixels[:, 1],
            2.0 + pixels[:, 2],
            1.0 + 2.0 * pixels[:, 3],
        ),
        dim=-1,
    )
    torch.testing.assert_close(
        child.pixel_to_world(pixels), parent.pixel_to_world(source), atol=3e-10, rtol=0
    )

    matrix1 = torch.tensor([[1.03, 0.04], [-0.02, 0.97]], dtype=torch.float64)
    offset1 = torch.tensor([0.8, -1.1], dtype=torch.float64)
    matrix2 = torch.tensor([[0.99, -0.03], [0.05, 1.01]], dtype=torch.float64)
    offset2 = torch.tensor([-0.2, 0.4], dtype=torch.float64)
    corrected = parent.with_affine(matrix1, offset1).with_affine(matrix2, offset2)
    corrected_pixels = source.clone()
    pair = source[:, [1, 3]] @ matrix1.T + offset1
    corrected_pixels[:, [1, 3]] = pair @ matrix2.T + offset2
    torch.testing.assert_close(
        corrected.pixel_to_world(corrected_pixels),
        parent.pixel_to_world(source),
        atol=3e-10,
        rtol=0,
    )
    restored = WCS(corrected.to_header(include_naxis=True))
    torch.testing.assert_close(
        restored.pixel_to_world(corrected_pixels),
        corrected.pixel_to_world(corrected_pixels),
        atol=3e-10,
        rtol=0,
    )
    celestial_header = _permuted_header()
    celestial_header.update({"CD1_3": 0.0, "CD3_1": 0.0})
    celestial = WCS(celestial_header).celestial
    assert celestial.world_axis_names == ("DEC--TAN", "RA---TAN")
    assert celestial.pixel_shape == (48, 64)


@pytest.mark.parametrize("use_pc", [False, True])
def test_tpv_pc_cd_matches_wcslib_and_serializes(use_pc: bool) -> None:
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "NAXIS1": 48,
        "NAXIS2": 64,
        "CTYPE1": "RA---TPV",
        "CTYPE2": "DEC--TPV",
        "CRPIX1": 24.0,
        "CRPIX2": 32.0,
        "CRVAL1": 130.0,
        "CRVAL2": -12.0,
        "PV1_1": 1.0,
        "PV1_5": -1.0e-4,
        "PV2_1": 1.0,
        "PV2_4": 2.0e-4,
    }
    if use_pc:
        header.update({"CDELT1": -1.1e-3, "CDELT2": 8.0e-4})
    else:
        header.update(
            {"CD1_1": -1.1e-3, "CD1_2": -3.0e-5, "CD2_1": 4.0e-5, "CD2_2": 8.0e-4}
        )
    pixels = np.array([[23.0, 31.0], [26.0, 35.0]])
    actual = WCS(header)
    expected = AstropyWCS(header).all_pix2world(pixels, 0)

    np.testing.assert_allclose(
        actual.pixel_to_world(torch.from_numpy(pixels)).numpy(), expected, atol=3e-8
    )
    restored = WCS(actual.to_header(include_naxis=True))
    np.testing.assert_allclose(
        restored.pixel_to_world(torch.from_numpy(pixels)).numpy(), expected, atol=3e-8
    )


def test_reversed_sip_and_lookup_distortion_match_astropy() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "DEC--TAN-SIP",
        "CTYPE2": "RA---TAN-SIP",
        "CRPIX1": 20.0,
        "CRPIX2": 30.0,
        "CRVAL1": 5.0,
        "CRVAL2": 10.0,
        "CD1_1": 0.01,
        "CD1_2": 3.0e-4,
        "CD2_1": -2.0e-4,
        "CD2_2": -0.01,
        "A_ORDER": 2,
        "B_ORDER": 2,
        "A_2_0": 1.0e-5,
        "B_0_2": -1.5e-5,
    }
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 4), torch.linspace(-1.0, 1.0, 5), indexing="ij"
    )
    first = DistortionLookupTable(0.01 * x, cdelt=(12.0, 16.0))
    second = DistortionLookupTable(-0.008 * y, cdelt=(12.0, 16.0))
    actual = WCS(header, cpdis1=first, cpdis2=second)
    expected = AstropyWCS(header)
    expected.cpdis1 = AstropyDistortionLookupTable(
        first.data.numpy(), first.crpix, first.crval, first.cdelt
    )
    expected.cpdis2 = AstropyDistortionLookupTable(
        second.data.numpy(), second.crpix, second.crval, second.cdelt
    )
    pixels = np.array([[19.0, 29.0], [22.5, 34.0]], dtype=np.float64)
    np.testing.assert_allclose(
        actual.pixel_to_world(torch.from_numpy(pixels)).numpy(),
        expected.all_pix2world(pixels, 0),
        atol=4e-8,
    )


def test_permuted_gradcheck_and_compile_preserve_fits_order() -> None:
    wcs = WCS(
        {
            "NAXIS": 2,
            "CTYPE1": "DEC--TAN",
            "CTYPE2": "RA---TAN",
            "CRPIX1": 20.0,
            "CRPIX2": 30.0,
            "CRVAL1": 5.0,
            "CRVAL2": 10.0,
            "CDELT1": 0.01,
            "CDELT2": -0.01,
        }
    )
    pixels = torch.tensor([[20.25, 31.5]], dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        lambda value: wcs.pixel_to_world_differentiable(value),
        (pixels,),
        fast_mode=True,
    )
    expected = wcs.pixel_to_world(pixels.detach())
    wcs.compile(backend="eager")
    torch.testing.assert_close(wcs.pixel_to_world(pixels.detach()), expected)
    torch.testing.assert_close(
        wcs.world_to_pixel(expected), pixels.detach(), atol=2e-9, rtol=0
    )


def test_multidimensional_sip_fails_clearly() -> None:
    header = _permuted_header()
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["CTYPE4"] = "RA---TAN-SIP"
    header.update({"A_ORDER": 2, "B_ORDER": 2, "A_2_0": 1.0e-5})
    with pytest.raises(ValueError, match="require a two-dimensional WCS"):
        WCS(header)


def test_permuted_tpv_fails_clearly() -> None:
    header = _permuted_header()
    header["CTYPE2"] = "DEC--TPV"
    header["CTYPE4"] = "RA---TPV"
    header.update({"PV2_1": 1.0, "PV4_1": 1.0})
    with pytest.raises(ValueError, match="longitude/latitude to be FITS axes 1/2"):
        WCS(header)


def test_reversed_tpv_fails_clearly() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "DEC--TPV",
        "CTYPE2": "RA---TPV",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": -12.0,
        "CRVAL2": 130.0,
        "CDELT1": 8.0e-4,
        "CDELT2": -1.1e-3,
        "PV1_1": 1.0,
        "PV2_1": 1.0,
    }
    with pytest.raises(ValueError, match="longitude/latitude to be FITS axes 1/2"):
        WCS(header)
