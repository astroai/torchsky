import math

import pytest
import torch

from torchsky.coords import TabularAxis as CoordTabularAxis
from torchsky.wcs import TabularAxis, TabularLookupWCS, WCS, tabular_axes_from_fits


def test_tabular_axis_forward_and_inverse_piecewise_linear() -> None:
    axis = TabularAxis(
        pixel=torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64),
        world=torch.tensor([10.0, 20.0, 80.0], dtype=torch.float64),
        unit="nm",
    )

    pixel = torch.tensor([1.5, 3.0], dtype=torch.float64)
    world = axis.pixel_to_world(pixel)
    pixel_back = axis.world_to_pixel(world)

    torch.testing.assert_close(world, torch.tensor([15.0, 50.0], dtype=torch.float64))
    torch.testing.assert_close(pixel_back, pixel)
    assert axis.name is None
    assert axis.unit == "nm"


def test_tabular_axis_decreasing_world_inverse() -> None:
    axis = TabularAxis([0.0, 1.0, 2.0], [30.0, 20.0, 10.0])

    world = torch.tensor([25.0, 15.0], dtype=torch.float64)
    pixel = axis.world_to_pixel(world)

    torch.testing.assert_close(pixel, torch.tensor([0.5, 1.5], dtype=torch.float64))
    assert axis.world_direction == -1


def test_tabular_axis_bounds_nan_and_clamp() -> None:
    axis = TabularAxis([0.0, 1.0, 2.0], [0.0, 10.0, 30.0])

    nan_values = axis.pixel_to_world(torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float64))
    clamp_values = axis.pixel_to_world(
        torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float64),
        bounds="clamp",
    )

    assert math.isnan(float(nan_values[0]))
    assert math.isnan(float(nan_values[2]))
    torch.testing.assert_close(nan_values[1], torch.tensor(5.0, dtype=torch.float64))
    torch.testing.assert_close(
        clamp_values,
        torch.tensor([0.0, 5.0, 30.0], dtype=torch.float64),
    )


def test_tabular_axis_autograd_uses_local_linear_slope() -> None:
    axis = TabularAxis(
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
        torch.tensor([0.0, 2.0, 3.0], dtype=torch.float64),
    )
    pixel = torch.tensor([0.25, 1.50], dtype=torch.float64, requires_grad=True)

    world = axis.pixel_to_world(pixel)
    world.sum().backward()

    torch.testing.assert_close(world, torch.tensor([0.5, 2.5], dtype=torch.float64))
    torch.testing.assert_close(
        pixel.grad,
        torch.tensor([2.0, 1.0], dtype=torch.float64),
    )


def test_tabular_lookup_wcs_separable_axes_roundtrip() -> None:
    lookup = TabularLookupWCS(
        [
            TabularAxis([0.0, 1.0, 3.0], [0.0, 2.0, 10.0], name="lambda"),
            TabularAxis([0.0, 2.0], [100.0, 60.0], name="time"),
        ]
    )

    pixels = torch.tensor([[0.5, 1.0], [2.0, 0.25]], dtype=torch.float64)
    world = lookup.pixel_to_world(pixels)
    pixels_back = lookup.world_to_pixel(world)

    torch.testing.assert_close(
        world,
        torch.tensor([[1.0, 80.0], [6.0, 95.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels)
    assert lookup.pixel_n_dim == 2
    assert lookup.world_n_dim == 2


def test_tabular_axis_rejects_invalid_tables_and_nonmonotonic_inverse() -> None:
    with pytest.raises(ValueError, match="pixel and world tables"):
        TabularAxis([0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="pixel must be strictly monotonic"):
        TabularAxis([0.0, 0.0, 1.0], [0.0, 1.0, 2.0])

    axis = TabularAxis([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="world must be strictly monotonic"):
        axis.world_to_pixel([1.5])


def test_tabular_axis_is_exported_from_coords_namespace() -> None:
    assert CoordTabularAxis is TabularAxis


def test_wcs_uses_independent_tabular_axis_forward_inverse_and_jacobian() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "CUNIT1": "nm",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 1.0,
    }
    axis = TabularAxis([0.0, 1.0, 3.0], [400.0, 410.0, 450.0], unit="nm")
    wcs = WCS(header, tabular_axes={1: axis})
    pixels = torch.tensor([[0.5], [2.0]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels, origin=0)
    pixels_back = wcs.world_to_pixel(world, origin=0)
    jac = wcs.pixel_to_world_jacobian(pixels)

    torch.testing.assert_close(
        world,
        torch.tensor([[405.0], [430.0]], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels)
    torch.testing.assert_close(
        jac[..., 0, 0],
        torch.tensor([10.0, 20.0], dtype=torch.float64),
    )


def test_wcs_tabular_axis_on_spatial_cube_extra_axis() -> None:
    header = {
        "NAXIS": 3,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE-TAB",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CDELT3": 1.0,
    }
    axis = TabularAxis([0.0, 2.0, 4.0], [500.0, 520.0, 550.0], unit="nm")
    wcs = WCS(header, tabular_axes={3: axis})
    pixels = torch.tensor([[31.0, 31.0, 1.0], [32.0, 31.5, 3.0]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels, origin=0)
    pixels_back = wcs.world_to_pixel(world, origin=0)

    torch.testing.assert_close(
        world[:, 2],
        torch.tensor([510.0, 535.0], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels, atol=1e-7, rtol=1e-9)


def test_wcs_tabular_axes_require_tab_ctype() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 1.0,
    }
    axis = TabularAxis([0.0, 1.0], [400.0, 410.0])
    with pytest.raises(ValueError, match="CTYPE1 ending in '-TAB'"):
        WCS(header, tabular_axes={1: axis})


def test_tabular_axes_from_fits_metadata_with_indexing_vector() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "CUNIT1": "nm",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 1.0,
        "PS1_0": "WCS-TAB",
        "PS1_1": "WAVE",
        "PS1_2": "PIX",
    }
    tables = {
        "WCS-TAB": {
            "PIX": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
            "WAVE": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
        }
    }

    axes = tabular_axes_from_fits(header, tables)
    wcs = WCS(header, tabular_axes=axes)
    world = wcs.pixel_to_world(torch.tensor([[1.0], [3.5]], dtype=torch.float64))

    assert set(axes) == {1}
    assert axes[1].name == "WAVE"
    assert axes[1].unit == "nm"
    torch.testing.assert_close(
        world,
        torch.tensor([[410.0], [465.0]], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_metadata_defaults_to_sample_index() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "FREQ-TAB",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 1.0,
        "PS1_1": "FREQ",
    }

    axes = tabular_axes_from_fits(
        header,
        {"FREQ": torch.tensor([1.0e9, 1.1e9, 1.4e9], dtype=torch.float64)},
    )

    torch.testing.assert_close(
        axes[1].pixel,
        torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64),
    )
    torch.testing.assert_close(
        axes[1].world,
        torch.tensor([1.0e9, 1.1e9, 1.4e9], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_rejects_coupled_table_arrays() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_1": "COORD",
    }
    with pytest.raises(ValueError, match="one independent tabular axis"):
        tabular_axes_from_fits(
            header,
            {"COORD": torch.ones((2, 2), dtype=torch.float64)},
        )
