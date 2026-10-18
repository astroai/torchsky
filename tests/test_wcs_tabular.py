import math

import pytest
import torch
import torchfits

from torchsky.wcs import (
    CoupledTabularLookupWCS,
    TabularAxis,
    TabularLookupWCS,
    WCS,
    coupled_tabular_wcs_from_fits,
    tabular_axes_from_fits,
)


def _wcs_from_fits(path, *, hdu: int = 0, key: str = "") -> WCS:
    header = dict(torchfits.read_header(str(path), hdu))
    with torchfits.open(str(path)) as hdul:
        return WCS(header, key=key, tables=hdul)


class _TabTable:
    def __init__(self, header: dict, data: dict) -> None:
        self.header = header
        self.data = data


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


def test_tabular_axis_decreasing_pixel_grid_forward_inverse() -> None:
    axis = TabularAxis([2.0, 1.0, 0.0], [20.0, 10.0, 0.0])

    pixel = torch.tensor([1.5, 0.5], dtype=torch.float64)
    world = axis.pixel_to_world(pixel)
    pixel_back = axis.world_to_pixel(world)

    torch.testing.assert_close(world, torch.tensor([15.0, 5.0], dtype=torch.float64))
    torch.testing.assert_close(pixel_back, pixel)


def test_tabular_axis_bounds_nan_and_clamp() -> None:
    axis = TabularAxis([0.0, 1.0, 2.0], [0.0, 10.0, 30.0])

    nan_values = axis.pixel_to_world(
        torch.tensor([-1.0, 0.5, 3.0], dtype=torch.float64)
    )
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


def test_coupled_tabular_lookup_wcs_multilinear_forward_and_autograd() -> None:
    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    xx = x[:, None].expand(2, 3)
    yy = y[None, :].expand(2, 3)
    world = torch.stack((xx + 0.1 * yy, 0.2 * xx + yy), dim=-1)
    lookup = CoupledTabularLookupWCS(
        (x, y), world, names=("tilted", "product"), units=("nm", "d")
    )
    pixels = torch.tensor([[1.0, 5.0], [2.0, 15.0]], dtype=torch.float64)

    got = lookup.pixel_to_world(pixels)
    pixels_back = lookup.world_to_pixel(got)

    torch.testing.assert_close(
        got,
        torch.tensor([[1.5, 5.2], [3.5, 15.4]], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels, atol=1e-8, rtol=1e-8)
    assert lookup.pixel_n_dim == 2
    assert lookup.world_n_dim == 2
    assert lookup.grid_shape == (2, 3)
    assert lookup.names == ("tilted", "product")
    assert lookup.units == ("nm", "d")

    batch_pixels = torch.stack(
        (
            torch.linspace(0.0, 2.0, 16),
            torch.linspace(0.0, 20.0, 16),
        ),
        dim=-1,
    ).to(dtype=torch.float64)
    batch_world = lookup.pixel_to_world(batch_pixels)
    torch.testing.assert_close(
        lookup.world_to_pixel(batch_world),
        batch_pixels,
        atol=1e-8,
        rtol=1e-8,
    )

    differentiable_pixel = torch.tensor(
        [[1.0, 5.0]], dtype=torch.float64, requires_grad=True
    )
    lookup.pixel_to_world(differentiable_pixel)[..., 0].sum().backward()
    torch.testing.assert_close(
        differentiable_pixel.grad,
        torch.tensor([[1.0, 0.1]], dtype=torch.float64),
    )


def test_coupled_tabular_lookup_wcs_inverse_fails_closed_for_folded_grid() -> None:
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 2)
    yy = y[None, :].expand(3, 2)
    world = torch.stack((torch.abs(xx - 1.0), yy), dim=-1)
    lookup = CoupledTabularLookupWCS((x, y), world)

    ambiguous_world = torch.tensor([[0.5, 0.5]], dtype=torch.float64)
    pixels = lookup.world_to_pixel(ambiguous_world)

    assert bool(torch.isnan(pixels).all().item())


def test_coupled_tabular_lookup_wcs_inverse_deduplicates_cell_boundaries() -> None:
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 2)
    yy = y[None, :].expand(3, 2)
    world = torch.stack((xx, yy), dim=-1)
    lookup = CoupledTabularLookupWCS((x, y), world)

    world_on_boundary = torch.tensor([[1.0, 0.5]], dtype=torch.float64)
    pixels = lookup.world_to_pixel(world_on_boundary)

    torch.testing.assert_close(
        pixels,
        torch.tensor([[1.0, 0.5]], dtype=torch.float64),
    )


def test_coupled_tabular_lookup_wcs_bounds_nan_and_clamp() -> None:
    lookup = CoupledTabularLookupWCS(
        (
            torch.tensor([0.0, 1.0], dtype=torch.float64),
            torch.tensor([0.0, 1.0], dtype=torch.float64),
        ),
        torch.tensor(
            [[[0.0], [10.0]], [[100.0], [110.0]]],
            dtype=torch.float64,
        ),
    )
    query = torch.tensor([[-1.0, 0.5], [0.5, 2.0]], dtype=torch.float64)

    nan_out = lookup.pixel_to_world(query)
    clamp_out = lookup.pixel_to_world(query, bounds="clamp")

    assert torch.isnan(nan_out).all()
    torch.testing.assert_close(
        clamp_out,
        torch.tensor([[5.0], [60.0]], dtype=torch.float64),
    )
    # Underdetermined (2 pixel axes, 1 world): particular inverse recovers world.
    w = torch.tensor([[55.0]], dtype=torch.float64)
    pix = lookup.world_to_pixel(w)
    assert torch.isfinite(pix).all()
    torch.testing.assert_close(lookup.pixel_to_world(pix), w, atol=1e-6, rtol=0.0)


def test_coupled_tabular_overdetermined_inverse_roundtrip_and_off_manifold() -> None:
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 3)
    yy = y[None, :].expand(3, 3)
    world = torch.stack((xx, yy, xx + yy), dim=-1)
    lookup = CoupledTabularLookupWCS((x, y), world)
    pix = torch.tensor([[0.5, 1.25], [1.5, 0.75]], dtype=torch.float64)
    on_manifold = lookup.pixel_to_world(pix)
    back = lookup.world_to_pixel(on_manifold)
    torch.testing.assert_close(back, pix, atol=1e-8, rtol=0.0)
    off = on_manifold.clone()
    off[:, 2] = off[:, 2] + 5.0
    lost = lookup.world_to_pixel(off)
    assert torch.isnan(lost).all()


def test_wcs_coupled_tabular_overdetermined_fits_inverse() -> None:
    x = torch.arange(3, dtype=torch.float64).reshape(3, 1)
    y = torch.arange(4, dtype=torch.float64).reshape(1, 4)
    wave = 400.0 + 12.0 * x + 0.25 * y
    time = 59000.0 + 0.5 * y + 0.02 * x
    aux = wave + time
    header = {
        "NAXIS": 3,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CTYPE3": "AUX-TAB",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRPIX3": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRVAL3": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "CDELT3": 1.0,
        "PS1_1": "WAVE",
        "PV1_3": 1,
        "PS2_1": "TIME",
        "PV2_3": 2,
        "PS3_1": "AUX",
        "PV3_3": 1,
    }
    tables = {
        "WAVE": wave,
        "TIME": time,
        "AUX": aux,
    }
    wcs = WCS(header, tables=tables)
    assert wcs.coupled_tabular is not None
    assert wcs.coupled_tabular.pixel_n_dim == 2
    assert wcs.coupled_tabular.world_n_dim == 3
    assert wcs.coupled_tabular_axes == (0, 1, 2)
    assert wcs.coupled_tabular_pixel_axes == (0, 1)
    pixels = torch.tensor([[0.5, 1.5, 0.0], [2.0, 3.0, 0.0]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels, origin=0)
    back = wcs.world_to_pixel(world, origin=0)
    torch.testing.assert_close(back[:, :2], pixels[:, :2], atol=1e-7, rtol=0.0)


def test_coupled_tabular_lookup_wcs_normalizes_descending_grid_axes() -> None:
    x_desc = torch.tensor([2.0, 0.0], dtype=torch.float64)
    y = torch.tensor([0.0, 10.0], dtype=torch.float64)
    xx = x_desc[:, None].expand(2, 2)
    yy = y[None, :].expand(2, 2)
    world = (10.0 * xx + yy).unsqueeze(-1)
    lookup = CoupledTabularLookupWCS((x_desc, y), world)

    got = lookup.pixel_to_world(torch.tensor([[1.0, 5.0]], dtype=torch.float64))

    torch.testing.assert_close(
        lookup.pixel_axes[0], torch.tensor([0.0, 2.0], dtype=torch.float64)
    )
    torch.testing.assert_close(got, torch.tensor([[15.0]], dtype=torch.float64))


def test_tabular_axis_rejects_invalid_tables_and_nonmonotonic_inverse() -> None:
    with pytest.raises(ValueError, match="pixel and world tables"):
        TabularAxis([0.0, 1.0], [0.0, 1.0, 2.0])
    with pytest.raises(ValueError, match="pixel must be strictly monotonic"):
        TabularAxis([0.0, 0.0, 1.0], [0.0, 1.0, 2.0])

    axis = TabularAxis([0.0, 1.0, 2.0], [0.0, 2.0, 1.0])
    with pytest.raises(ValueError, match="world must be strictly monotonic"):
        axis.world_to_pixel([1.5])


def test_tabular_axis_caches_monotonic_inverse_direction() -> None:
    axis = TabularAxis([0.0, 1.0, 2.0], [0.0, 2.0, 4.0])

    assert axis.world_direction == 1
    assert axis.world_direction == 1
    assert getattr(axis, "_world_direction") == 1


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


def test_tabular_axes_from_fits_separable_multidimensional_arrays_with_pv3() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CUNIT1": "nm",
        "CUNIT2": "d",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "PS1_1": "WAVE_GRID",
        "PV1_3": 1,
        "PS2_1": "TIME_GRID",
        "PV2_3": 2,
    }
    tables = {
        "WAVE_GRID": torch.tensor(
            [[400.0, 400.0], [420.0, 420.0], [460.0, 460.0]],
            dtype=torch.float64,
        ),
        "TIME_GRID": torch.tensor(
            [[10.0, 20.0], [10.0, 20.0], [10.0, 20.0]],
            dtype=torch.float64,
        ),
    }

    axes = tabular_axes_from_fits(header, tables)
    wcs = WCS(header, tabular_axes=axes)
    pixels = torch.tensor([[0.5, 0.5], [1.5, 0.25]], dtype=torch.float64)
    world = wcs.pixel_to_world(pixels, origin=0)
    pixels_back = wcs.world_to_pixel(world, origin=0)

    assert set(axes) == {1, 2}
    torch.testing.assert_close(
        world,
        torch.tensor([[410.0, 15.0], [440.0, 12.5]], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels, atol=1e-12, rtol=0.0)


def test_coupled_tabular_wcs_from_fits_accepts_nonseparable_arrays() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CUNIT1": "nm",
        "CUNIT2": "d",
        "PS1_1": "WAVE_GRID",
        "PV1_3": 1,
        "PS2_1": "TIME_GRID",
        "PV2_3": 2,
    }
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 2)
    yy = y[None, :].expand(3, 2)
    tables = {
        "WAVE_GRID": 400.0 + 10.0 * xx + 2.0 * yy + xx * yy,
        "TIME_GRID": 50.0 + 5.0 * yy + xx * yy,
    }

    lookup = coupled_tabular_wcs_from_fits(header, tables)
    got = lookup.pixel_to_world(torch.tensor([[1.5, 0.5]], dtype=torch.float64))

    assert lookup.pixel_n_dim == 2
    assert lookup.world_n_dim == 2
    assert lookup.names == ("WAVE_GRID", "TIME_GRID")
    assert lookup.units == ("nm", "d")
    torch.testing.assert_close(
        got,
        torch.tensor([[416.75, 53.25]], dtype=torch.float64),
    )


def test_wcs_auto_loads_coupled_tabular_lookup_forward_and_jacobian() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CUNIT1": "nm",
        "CUNIT2": "d",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "PS1_1": "WAVE_GRID",
        "PV1_3": 1,
        "PS2_1": "TIME_GRID",
        "PV2_3": 2,
    }
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 2)
    yy = y[None, :].expand(3, 2)
    tables = {
        "WAVE_GRID": 400.0 + 10.0 * xx + 2.0 * yy + xx * yy,
        "TIME_GRID": 50.0 + 5.0 * yy + xx * yy,
    }
    wcs = WCS(header, tables=tables)
    pixels = torch.tensor([[1.5, 0.5]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels, origin=0)
    pixels_back = wcs.world_to_pixel(world, origin=0)
    jac = wcs.pixel_to_world_jacobian(pixels, origin=0)
    inv_jac = wcs.world_to_pixel_jacobian(world, origin=0)

    assert wcs.coupled_tabular is not None
    assert wcs.tabular_axes == {}
    torch.testing.assert_close(
        world,
        torch.tensor([[416.75, 53.25]], dtype=torch.float64),
    )
    torch.testing.assert_close(pixels_back, pixels, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        jac,
        torch.tensor([[[10.5, 3.5], [0.5, 6.5]]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        inv_jac,
        torch.linalg.inv(jac),
        atol=1e-7,
        rtol=1e-7,
    )


def test_wcs_mixed_spatial_and_coupled_tabular_extra_axes_forward() -> None:
    header = {
        "NAXIS": 4,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CTYPE3": "WAVE-TAB",
        "CTYPE4": "TIME-TAB",
        "CUNIT3": "nm",
        "CUNIT4": "d",
        "CRPIX1": 32.0,
        "CRPIX2": 32.0,
        "CRPIX3": 1.0,
        "CRPIX4": 1.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CRVAL3": 0.0,
        "CRVAL4": 0.0,
        "CDELT1": -0.01,
        "CDELT2": 0.01,
        "CDELT3": 1.0,
        "CDELT4": 1.0,
        "PS3_1": "WAVE_GRID",
        "PV3_3": 1,
        "PS4_1": "TIME_GRID",
        "PV4_3": 2,
    }
    x = torch.tensor([0.0, 1.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 1.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 2)
    yy = y[None, :].expand(3, 2)
    tables = {
        "WAVE_GRID": 400.0 + 10.0 * xx + 2.0 * yy + xx * yy,
        "TIME_GRID": 50.0 + 5.0 * yy + xx * yy,
    }
    wcs = WCS(header, tables=tables)
    spatial_header = dict(header)
    spatial_header["CTYPE3"] = "LINEAR"
    spatial_header["CTYPE4"] = "LINEAR"
    spatial_header.pop("PS3_1")
    spatial_header.pop("PV3_3")
    spatial_header.pop("PS4_1")
    spatial_header.pop("PV4_3")
    spatial_wcs = WCS(spatial_header)
    pixels = torch.tensor([[31.0, 31.0, 1.5, 0.5]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels, origin=0)
    pixels_back = wcs.world_to_pixel(world, origin=0)
    spatial_world = spatial_wcs.pixel_to_world(pixels, origin=0)
    jac = wcs.pixel_to_world_jacobian(pixels, origin=0)

    assert wcs.coupled_tabular is not None
    assert wcs.coupled_tabular_axes == (2, 3)
    torch.testing.assert_close(world[:, :2], spatial_world[:, :2])
    torch.testing.assert_close(
        world[:, 2:],
        torch.tensor([[416.75, 53.25]], dtype=torch.float64),
    )
    torch.testing.assert_close(
        jac[0, 2:, 2:],
        torch.tensor([[10.5, 3.5], [0.5, 6.5]], dtype=torch.float64),
    )
    inv_jac = wcs.world_to_pixel_jacobian(world, origin=0)
    assert torch.isfinite(inv_jac).all()
    torch.testing.assert_close(
        inv_jac[0, 2:, 2:],
        torch.linalg.inv(jac[0, 2:, 2:]),
        atol=1e-8,
        rtol=1e-8,
    )
    torch.testing.assert_close(pixels_back, pixels, atol=1e-8, rtol=1e-8)


def test_tabular_axes_from_fits_uses_unique_case_insensitive_column_match() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "CUNIT1": "nm",
        "PS1_1": "WAVE",
        "PS1_2": "PIX",
    }
    tables = {
        "wave": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
        "pix": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
    }

    axes = tabular_axes_from_fits(header, tables)

    torch.testing.assert_close(
        axes[1].pixel_to_world(torch.tensor([1.0, 3.5], dtype=torch.float64)),
        torch.tensor([410.0, 465.0], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_rejects_ambiguous_case_column_match() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_1": "Wave",
    }
    tables = {
        "wave": torch.tensor([400.0, 420.0], dtype=torch.float64),
        "WAVE": torch.tensor([500.0, 520.0], dtype=torch.float64),
    }

    with pytest.raises(KeyError, match="ambiguous"):
        tabular_axes_from_fits(header, tables)


def test_wcs_auto_loads_tabular_axes_from_tables_mapping() -> None:
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
        "wcs-tab": {
            "PIX": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
            "WAVE": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
        }
    }

    wcs = WCS(header, tables=tables)
    world = wcs.pixel_to_world(torch.tensor([[1.0], [3.5]], dtype=torch.float64))

    assert set(wcs.tabular_axes) == {0}
    torch.testing.assert_close(
        world,
        torch.tensor([[410.0], [465.0]], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_selects_extver_and_extlevel() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "CUNIT1": "nm",
        "PS1_0": "WCS-TAB",
        "PS1_1": "WAVE",
        "PV1_1": 2,
        "PV1_2": 3,
    }
    tables = [
        _TabTable(
            {"EXTNAME": "WCS-TAB", "EXTVER": 1, "EXTLEVEL": 3},
            {"WAVE": torch.tensor([100.0, 110.0], dtype=torch.float64)},
        ),
        _TabTable(
            {"EXTNAME": "WCS-TAB", "EXTVER": 2, "EXTLEVEL": 3},
            {"WAVE": torch.tensor([400.0, 420.0], dtype=torch.float64)},
        ),
    ]

    axes = tabular_axes_from_fits(header, tables)

    assert set(axes) == {1}
    torch.testing.assert_close(
        axes[1].pixel_to_world(torch.tensor([0.0, 1.0], dtype=torch.float64)),
        torch.tensor([400.0, 420.0], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_versioned_mapping_keys() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_0": "WCS-TAB",
        "PS1_1": "WAVE",
        "PV1_1": 2,
        "PV1_2": 1,
    }
    tables = {
        ("WCS-TAB", 1, 1): {"WAVE": torch.tensor([10.0, 20.0])},
        ("WCS-TAB", 2, 1): {"WAVE": torch.tensor([40.0, 60.0])},
    }

    axes = tabular_axes_from_fits(header, tables)

    torch.testing.assert_close(
        axes[1].pixel_to_world(torch.tensor([0.0, 0.5, 1.0])),
        torch.tensor([40.0, 50.0, 60.0], dtype=torch.float64),
    )


def test_tabular_axes_from_fits_rejects_missing_extver_extlevel() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_0": "WCS-TAB",
        "PS1_1": "WAVE",
        "PV1_1": 2,
        "PV1_2": 4,
    }
    tables = [
        _TabTable(
            {"EXTNAME": "WCS-TAB", "EXTVER": 2, "EXTLEVEL": 3},
            {"WAVE": torch.tensor([400.0, 420.0], dtype=torch.float64)},
        )
    ]

    with pytest.raises(KeyError, match="EXTVER=2 EXTLEVEL=4"):
        tabular_axes_from_fits(header, tables)


def test_wcs_tabular_axis_requires_coordinate_table() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "CRPIX1": 1.0,
        "CRVAL1": 0.0,
        "CDELT1": 1.0,
        "PS1_1": "WAVE",
    }

    with pytest.raises(ValueError, match="FITS -TAB axis"):
        WCS(header)


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


def test_tabular_axes_from_fits_requires_pv3_for_multidimensional_arrays() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_1": "COORD",
    }
    with pytest.raises(ValueError, match="requires PVi_3"):
        tabular_axes_from_fits(
            header,
            {"COORD": torch.ones((2, 2), dtype=torch.float64)},
        )


def test_tabular_axes_from_fits_rejects_nonseparable_multidimensional_arrays() -> None:
    header = {
        "NAXIS": 1,
        "CTYPE1": "WAVE-TAB",
        "PS1_1": "WAVE",
        "PV1_3": 1,
    }

    with pytest.raises(ValueError, match="coupled multidimensional coordinate array"):
        tabular_axes_from_fits(
            header,
            {
                "WAVE": torch.tensor(
                    [[400.0, 401.0], [420.0, 422.0], [460.0, 464.0]],
                    dtype=torch.float64,
                )
            },
        )


def test_wcs_tabular_fits_lazy_auto_loading(tmp_path) -> None:
    fits_path = tmp_path / "tabular_auto.fits"

    # Write a FITS file with image (referencing tabular axes) and table HDU
    from torchfits.hdu import TensorHDU, TableHDU, HDUList, Header

    image_header = Header()
    image_header["SIMPLE"] = True
    image_header["BITPIX"] = -64
    image_header["NAXIS"] = 1
    image_header["NAXIS1"] = 10
    image_header["CTYPE1"] = "WAVE-TAB"
    image_header["CUNIT1"] = "nm"
    image_header["CRPIX1"] = 1.0
    image_header["CRVAL1"] = 0.0
    image_header["CDELT1"] = 1.0
    image_header["PS1_0"] = "WCS-TAB"
    image_header["PS1_1"] = "WAVE"
    image_header["PS1_2"] = "PIX"

    image_hdu = TensorHDU(data=torch.zeros(10), header=image_header)

    table_header = Header()
    table_header["XTENSION"] = "BINTABLE"
    table_header["EXTNAME"] = "WCS-TAB"
    table_hdu = TableHDU(
        tensor_dict={
            "PIX": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
            "WAVE": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
        },
        header=table_header,
    )

    hdul = HDUList([image_hdu, table_hdu])
    hdul.write(str(fits_path), overwrite=True)

    root_wcs = _wcs_from_fits(fits_path)
    assert 0 in root_wcs.tabular_axes
    torch.testing.assert_close(
        root_wcs.pixel_to_world(torch.tensor([[1.0], [3.5]], dtype=torch.float64)),
        torch.tensor([[410.0], [465.0]], dtype=torch.float64),
    )


def test_wcs_coupled_tabular_fits_lazy_auto_loading(tmp_path) -> None:
    fits_path = tmp_path / "coupled_tabular_auto.fits"

    from torchfits.hdu import HDUList, Header, TableHDU, TensorHDU

    image_header = Header()
    image_header["SIMPLE"] = True
    image_header["BITPIX"] = -64
    image_header["NAXIS"] = 2
    image_header["NAXIS1"] = 3
    image_header["NAXIS2"] = 4
    image_header["CTYPE1"] = "WAVE-TAB"
    image_header["CTYPE2"] = "TIME-TAB"
    image_header["CUNIT1"] = "nm"
    image_header["CUNIT2"] = "d"
    image_header["CRPIX1"] = 1.0
    image_header["CRPIX2"] = 1.0
    image_header["CRVAL1"] = 0.0
    image_header["CRVAL2"] = 0.0
    image_header["CDELT1"] = 1.0
    image_header["CDELT2"] = 1.0
    image_header["PS1_0"] = "WCS-TAB"
    image_header["PS1_1"] = "WAVE"
    image_header["PV1_3"] = 1
    image_header["PS2_0"] = "WCS-TAB"
    image_header["PS2_1"] = "TIME"
    image_header["PV2_3"] = 2

    x = torch.arange(3, dtype=torch.float64).reshape(3, 1)
    y = torch.arange(4, dtype=torch.float64).reshape(1, 4)
    wave_grid = 400.0 + 12.0 * x + 0.25 * y
    time_grid = 59000.0 + 0.5 * y + 0.02 * x

    table_header = Header()
    table_header["XTENSION"] = "BINTABLE"
    table_header["EXTNAME"] = "WCS-TAB"
    table_hdu = TableHDU(
        tensor_dict={
            "WAVE": wave_grid,
            "TIME": time_grid.expand_as(wave_grid),
        },
        header=table_header,
    )
    HDUList([TensorHDU(data=torch.zeros(4, 3), header=image_header), table_hdu]).write(
        str(fits_path), overwrite=True
    )

    pixels = torch.tensor([[0.5, 1.5], [2.0, 3.0]], dtype=torch.float64)
    expected_world = torch.stack(
        (
            400.0 + 12.0 * pixels[:, 0] + 0.25 * pixels[:, 1],
            59000.0 + 0.5 * pixels[:, 1] + 0.02 * pixels[:, 0],
        ),
        dim=-1,
    )

    root_wcs = _wcs_from_fits(fits_path)
    assert root_wcs.coupled_tabular is not None
    assert root_wcs.coupled_tabular_axes == (0, 1)
    torch.testing.assert_close(root_wcs.pixel_to_world(pixels), expected_world)
    torch.testing.assert_close(root_wcs.world_to_pixel(expected_world), pixels)


def test_wcs_from_header_loads_alternate_key_tabular_axes(tmp_path) -> None:
    fits_path = tmp_path / "tabular_alt_auto.fits"

    from torchfits.hdu import HDUList, Header, TableHDU, TensorHDU

    image_header = Header()
    image_header["SIMPLE"] = True
    image_header["BITPIX"] = -64
    image_header["NAXIS"] = 1
    image_header["NAXIS1"] = 10
    image_header["CTYPE1"] = "WAVE"
    image_header["CRPIX1"] = 1.0
    image_header["CRVAL1"] = 0.0
    image_header["CDELT1"] = 1.0
    image_header["CTYPE1A"] = "WAVE-TAB"
    image_header["CUNIT1A"] = "nm"
    image_header["CRPIX1A"] = 1.0
    image_header["CRVAL1A"] = 0.0
    image_header["CDELT1A"] = 1.0
    image_header["PS1_0A"] = "WCS-TAB"
    image_header["PS1_1A"] = "WAVE"
    image_header["PS1_2A"] = "PIX"

    table_header = Header()
    table_header["XTENSION"] = "BINTABLE"
    table_header["EXTNAME"] = "WCS-TAB"
    table_hdu = TableHDU(
        tensor_dict={
            "PIX": torch.tensor([0.0, 2.0, 5.0], dtype=torch.float64),
            "WAVE": torch.tensor([400.0, 420.0, 510.0], dtype=torch.float64),
        },
        header=table_header,
    )
    HDUList([TensorHDU(data=torch.zeros(10), header=image_header), table_hdu]).write(
        str(fits_path), overwrite=True
    )

    wcs = _wcs_from_fits(fits_path, key="A")

    assert wcs.wcs_key == "A"
    assert 0 in wcs.tabular_axes
    torch.testing.assert_close(
        wcs.pixel_to_world(torch.tensor([[1.0], [3.5]], dtype=torch.float64)),
        torch.tensor([[410.0], [465.0]], dtype=torch.float64),
    )


def test_wcs_coupled_tabular_2d_roundtrip_vs_lookup() -> None:
    """2D coupled -TAB WCS roundtrips and matches the underlying lookup."""
    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    xx = x[:, None].expand(2, 3)
    yy = y[None, :].expand(2, 3)
    world_grid = torch.stack((xx + 0.1 * yy, 0.2 * xx + yy), dim=-1)
    lookup = CoupledTabularLookupWCS(
        (x, y), world_grid, names=("lon", "lat"), units=("deg", "deg")
    )
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAB",
        "CTYPE2": "DEC--TAB",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
    }
    wcs = WCS(
        header,
        coupled_tabular=lookup,
        coupled_tabular_axes=(1, 2),
        coupled_tabular_pixel_axes=(1, 2),
    )
    pixels = torch.tensor([[1.0, 5.0], [2.0, 15.0]], dtype=torch.float64)

    world = wcs.pixel_to_world(pixels, origin=0)
    expected = lookup.pixel_to_world(pixels)
    pixels_back = wcs.world_to_pixel(world, origin=0)

    torch.testing.assert_close(world, expected)
    torch.testing.assert_close(pixels_back, pixels, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(
        wcs.world_to_pixel_differentiable(world),
        pixels,
        atol=1e-8,
        rtol=1e-8,
    )


def test_wcs_mixed_coupled_tabular_complete_pixel_or_world() -> None:
    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    xx = x[:, None].expand(2, 3)
    yy = y[None, :].expand(2, 3)
    world_grid = torch.stack((xx + 0.1 * yy, 0.2 * xx + yy), dim=-1)
    lookup = CoupledTabularLookupWCS((x, y), world_grid)
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAB",
        "CTYPE2": "DEC--TAB",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
    }
    wcs = WCS(
        header,
        coupled_tabular=lookup,
        coupled_tabular_axes=(1, 2),
        coupled_tabular_pixel_axes=(1, 2),
    )
    pixels_true = torch.tensor([[1.0, 5.0], [2.0, 15.0]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)
    nan = float("nan")
    pix_in = torch.stack(
        [pixels_true[0], torch.tensor([nan, nan], dtype=torch.float64)]
    )
    world_in = torch.stack(
        [torch.tensor([nan, nan], dtype=torch.float64), world_true[1]]
    )
    solved_p, solved_w = wcs.mixed_pixel_world(pixel=pix_in, world=world_in)
    torch.testing.assert_close(solved_p, pixels_true, atol=1e-8, rtol=1e-8)
    torch.testing.assert_close(solved_w, world_true, atol=1e-8, rtol=1e-8)


def test_wcs_mixed_coupled_tabular_partial_pixel_and_world() -> None:
    x = torch.tensor([0.0, 2.0], dtype=torch.float64)
    y = torch.tensor([0.0, 10.0, 20.0], dtype=torch.float64)
    xx = x[:, None].expand(2, 3)
    yy = y[None, :].expand(2, 3)
    world_grid = torch.stack((xx + 0.1 * yy, 0.2 * xx + yy), dim=-1)
    lookup = CoupledTabularLookupWCS((x, y), world_grid)
    header = {
        "NAXIS": 2,
        "CTYPE1": "RA---TAB",
        "CTYPE2": "DEC--TAB",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
    }
    wcs = WCS(
        header,
        coupled_tabular=lookup,
        coupled_tabular_axes=(1, 2),
        coupled_tabular_pixel_axes=(1, 2),
    )
    pixels_true = torch.tensor([[1.0, 5.0]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)
    pix_in = torch.tensor([[1.0, float("nan")]], dtype=torch.float64)
    world_in = torch.tensor([[float("nan"), float(world_true[0, 1])]], dtype=torch.float64)
    solved_p, solved_w = wcs.mixed_pixel_world(pixel=pix_in, world=world_in)
    torch.testing.assert_close(solved_p, pixels_true, atol=1e-6, rtol=1e-6)
    torch.testing.assert_close(solved_w, world_true, atol=1e-6, rtol=1e-6)


def test_wcs_mixed_separable_tabular_fills_per_axis() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
    }
    tabular_axes = {
        1: TabularAxis([0.0, 1.0, 2.0], [400.0, 420.0, 460.0], name="wave"),
        2: TabularAxis([0.0, 1.0, 2.0], [10.0, 20.0, 40.0], name="time"),
    }
    wcs = WCS(header, tabular_axes=tabular_axes)
    pixels_true = torch.tensor([[0.5, 1.5], [1.5, 0.5]], dtype=torch.float64)
    world_true = wcs.pixel_to_world(pixels_true)
    mixed_pixel = torch.stack(
        [pixels_true[:, 0], torch.full((2,), float("nan"), dtype=torch.float64)],
        dim=-1,
    )
    mixed_world = torch.stack(
        [torch.full((2,), float("nan"), dtype=torch.float64), world_true[:, 1]],
        dim=-1,
    )
    solved_pixel, solved_world = wcs.mixed_pixel_world(
        pixel=mixed_pixel, world=mixed_world
    )
    torch.testing.assert_close(solved_pixel, pixels_true, atol=1e-10, rtol=0.0)
    torch.testing.assert_close(solved_world, world_true, atol=1e-10, rtol=0.0)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tabular_axis_cuda_matches_cpu() -> None:
    axis = TabularAxis(
        pixel=torch.tensor([1.0, 2.0, 4.0], dtype=torch.float64),
        world=torch.tensor([10.0, 20.0, 80.0], dtype=torch.float64),
        unit="nm",
    ).to(device="cuda", dtype=torch.float64)
    pixel = torch.tensor([1.5, 3.0], dtype=torch.float64, device="cuda")
    world = axis.pixel_to_world(pixel)
    pixel_back = axis.world_to_pixel(world)
    assert world.device.type == "cuda"
    torch.testing.assert_close(
        world.cpu(), torch.tensor([15.0, 50.0], dtype=torch.float64)
    )
    torch.testing.assert_close(pixel_back, pixel)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_coupled_tabular_wcs_cuda_roundtrip() -> None:
    header = {
        "NAXIS": 2,
        "CTYPE1": "WAVE-TAB",
        "CTYPE2": "TIME-TAB",
        "CUNIT1": "nm",
        "CUNIT2": "d",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CDELT1": 1.0,
        "CDELT2": 1.0,
        "PS1_1": "WAVE",
        "PV1_3": 1,
        "PS2_1": "TIME",
        "PV2_3": 2,
    }
    axis = torch.linspace(0.0, 3.0, 4, dtype=torch.float64)
    x = axis[:, None].expand(4, 4)
    y = axis[None, :].expand(4, 4)
    tables = {
        "WAVE": 400.0 + 2.0 * x + 0.125 * y,
        "TIME": 59000.0 + 0.03 * x + 0.5 * y,
    }
    cpu = WCS(header, tables=tables)
    gpu = WCS(header, tables=tables).to("cuda")
    pixels = torch.tensor([[0.5, 1.5], [2.0, 2.5]], dtype=torch.float64)
    world_c = cpu.pixel_to_world(pixels)
    world_g = gpu.pixel_to_world(pixels.cuda())
    assert world_g.device.type == "cuda"
    torch.testing.assert_close(world_g.cpu(), world_c, atol=1e-10, rtol=0.0)
    pix_g = gpu.world_to_pixel(world_g)
    torch.testing.assert_close(pix_g.cpu(), pixels, atol=1e-8, rtol=0.0)

