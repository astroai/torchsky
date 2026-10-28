import math

import numpy as np
import pytest
import torch
from astropy.wcs import DistortionLookupTable as AstropyDistortionLookupTable
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import CoupledTabularLookupWCS, DistortionLookupTable, TabularAxis, WCS


def _tan_header(*, distortion: str = "") -> dict[str, float | int | str]:
    ctype = "TPV" if distortion == "tpv" else "TAN"
    suffix = "-SIP" if distortion == "sip" else ""
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "NAXIS1": 72,
        "NAXIS2": 56,
        "CTYPE1": f"RA---{ctype}{suffix}",
        "CTYPE2": f"DEC--{ctype}{suffix}",
        "CRPIX1": 35.25,
        "CRPIX2": 27.75,
        "CRVAL1": 132.0,
        "CRVAL2": -18.0,
        "CD1_1": -1.1e-3,
        "CD1_2": 7.0e-5,
        "CD2_1": 4.0e-5,
        "CD2_2": 9.0e-4,
    }
    if distortion == "sip":
        header.update(
            {
                "A_ORDER": 3,
                "B_ORDER": 3,
                "A_2_0": 1.7e-5,
                "A_1_1": -8.0e-6,
                "A_0_3": 2.0e-8,
                "B_0_2": -1.3e-5,
                "B_2_1": 3.0e-8,
                "AP_ORDER": 2,
                "BP_ORDER": 2,
                "AP_2_0": -1.7e-5,
                "BP_0_2": 1.3e-5,
            }
        )
    elif distortion == "tpv":
        header.update(
            {
                "PV1_1": 1.0,
                "PV1_4": 2.0e-4,
                "PV1_5": -1.0e-4,
                "PV2_1": 1.0,
                "PV2_4": -1.5e-4,
                "PV2_6": 8.0e-5,
            }
        )
    return header


def _slice_pixel_map(
    view: tuple[slice, slice], shape: tuple[int, int], pixels: torch.Tensor
) -> torch.Tensor:
    y_start, _, y_step = view[0].indices(shape[1])
    x_start, _, x_step = view[1].indices(shape[0])
    return torch.stack(
        (
            x_start + x_step * pixels[:, 0],
            y_start + y_step * pixels[:, 1],
        ),
        dim=-1,
    )


@pytest.mark.parametrize(
    "view",
    [
        (slice(4, 48), slice(6, 64)),
        (slice(3, 51, 2), slice(5, 68, 3)),
        (slice(None, None, -1), slice(None, None, -1)),
    ],
)
@pytest.mark.parametrize("distortion", ["sip", "tpv"])
def test_polynomial_distortion_slice_matches_parent_astropy_and_serialized_header(
    distortion: str, view: tuple[slice, slice]
) -> None:
    header = _tan_header(distortion=distortion)
    parent = WCS(header)
    child = parent.slice(view)
    pixels = torch.tensor(
        [[0.25, 0.5], [4.0, 3.0], [9.5, 7.25]], dtype=torch.float64
    )
    source = _slice_pixel_map(view, (72, 56), pixels)
    expected = parent.pixel_to_world(source)
    astropy_parent = AstropyWCS(header)
    astropy_expected = astropy_parent.all_pix2world(source.numpy(), 0)

    torch.testing.assert_close(child.pixel_to_world(pixels), expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(expected.numpy(), astropy_expected, atol=3e-8, rtol=0)
    serialized = child.to_header(relax=True, include_naxis=True)
    np.testing.assert_allclose(
        AstropyWCS(serialized).all_pix2world(pixels.numpy(), 0),
        astropy_expected,
        atol=3e-8,
        rtol=0,
    )
    restored = WCS(serialized)
    torch.testing.assert_close(
        restored.pixel_to_world(pixels), child.pixel_to_world(pixels), atol=3e-10, rtol=0
    )


def _lookup_tables(
    *, dtype: torch.dtype = torch.float64
) -> tuple[DistortionLookupTable, DistortionLookupTable]:
    y, x = torch.meshgrid(
        torch.linspace(-1.0, 1.0, 5, dtype=dtype),
        torch.linspace(-1.0, 1.0, 6, dtype=dtype),
        indexing="ij",
    )
    return (
        DistortionLookupTable(
            0.018 * x + 0.007 * y,
            crpix=(1.0, 1.0),
            crval=(1.0, 1.0),
            cdelt=(14.2, 13.75),
        ),
        DistortionLookupTable(
            -0.006 * x + 0.015 * y,
            crpix=(1.0, 1.0),
            crval=(1.0, 1.0),
            cdelt=(14.2, 13.75),
        ),
    )


def _lookup_wcs(
    kind: str,
) -> tuple[WCS, AstropyWCS, tuple[DistortionLookupTable, DistortionLookupTable]]:
    header = _tan_header()
    first, second = _lookup_tables()
    kwargs = {f"{kind}1": first, f"{kind}2": second}
    torch_wcs = WCS(header, **kwargs)
    astropy_wcs = AstropyWCS(header)
    for index, table in enumerate((first, second), start=1):
        setattr(
            astropy_wcs,
            f"{kind}{index}",
            AstropyDistortionLookupTable(
                table.data.to(torch.float32).numpy(),
                table.crpix,
                table.crval,
                table.cdelt,
            ),
        )
    return torch_wcs, astropy_wcs, (first, second)


@pytest.mark.parametrize("kind", ["cpdis", "det2im"])
@pytest.mark.parametrize(
    "view",
    [
        (slice(4, 52, 2), slice(7, 68, 3)),
        (slice(None, None, -1), slice(None, None, -1)),
    ],
)
def test_lookup_distortion_slice_rebases_grid_without_resampling(
    kind: str, view: tuple[slice, slice]
) -> None:
    parent, astropy_parent, tables = _lookup_wcs(kind)
    child = parent.slice(view)
    pixels = torch.tensor(
        [[0.5, 0.25], [3.0, 4.0], [8.25, 6.5]], dtype=torch.float64
    )
    source = _slice_pixel_map(view, (72, 56), pixels)
    expected = parent.pixel_to_world(source)
    astropy_expected = astropy_parent.all_pix2world(source.numpy(), 0)

    torch.testing.assert_close(child.pixel_to_world(pixels), expected, atol=3e-10, rtol=0)
    np.testing.assert_allclose(expected.numpy(), astropy_expected, atol=3e-8, rtol=0)
    rebased = getattr(child, f"{kind}1")
    assert rebased is not None
    x_start, _, x_step = view[1].indices(72)
    y_start, _, y_step = view[0].indices(56)
    assert rebased.crval == pytest.approx(
        (
            1.0 + (tables[0].crval[0] - 1.0 - x_start) / x_step,
            1.0 + (tables[0].crval[1] - 1.0 - y_start) / y_step,
        )
    )
    restored_kwargs = {
        f"{kind}1": getattr(child, f"{kind}1"),
        f"{kind}2": getattr(child, f"{kind}2"),
    }
    restored = WCS(child.to_header(include_naxis=True), **restored_kwargs)
    torch.testing.assert_close(
        restored.pixel_to_world(pixels), child.pixel_to_world(pixels), atol=3e-10, rtol=0
    )


def _affine_cases() -> list[tuple[torch.Tensor, torch.Tensor]]:
    angle = math.radians(17.0)
    rotation = torch.tensor(
        [[math.cos(angle), -math.sin(angle)], [math.sin(angle), math.cos(angle)]],
        dtype=torch.float64,
    )
    return [
        (torch.eye(2, dtype=torch.float64), torch.tensor([2.5, -1.75], dtype=torch.float64)),
        (torch.diag(torch.tensor([1.08, 0.93], dtype=torch.float64)), torch.zeros(2, dtype=torch.float64)),
        (rotation, torch.tensor([-0.4, 1.2], dtype=torch.float64)),
        (
            torch.tensor([[1.03, 0.06], [-0.04, 0.97]], dtype=torch.float64),
            torch.tensor([1.1, -0.8], dtype=torch.float64),
        ),
    ]


def _separable_tabular_wcs() -> WCS:
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": 12,
            "NAXIS2": 10,
            "CTYPE1": "X-TAB",
            "CTYPE2": "Y-TAB",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CD1_1": 0.25,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.5,
        },
        tabular_axes={
            1: TabularAxis([-1.0, 0.0, 1.5, 3.0], [-4.0, 0.0, 9.0, 30.0]),
            2: TabularAxis([-1.0, 0.0, 2.0, 5.0], [8.0, 10.0, 13.0, 25.0]),
        },
    )


def _family_wcs(family: str) -> WCS:
    if family in {"cpdis", "det2im"}:
        parent, _, _ = _lookup_wcs(family)
        return parent
    if family == "tabular":
        return _separable_tabular_wcs()
    if family == "coupled":
        return _coupled_tabular_wcs()
    return WCS(_tan_header(distortion=family))


def _family_source_pixels(family: str) -> torch.Tensor:
    if family in {"tabular", "coupled"}:
        return torch.tensor([[3.0, 3.0], [5.0, 4.0], [8.0, 6.0]], dtype=torch.float64)
    return torch.tensor(
        [[22.0, 18.0], [35.0, 27.0], [48.0, 39.0]], dtype=torch.float64
    )


@pytest.mark.parametrize(
    "family", ["sip", "tpv", "cpdis", "det2im", "tabular", "coupled"]
)
@pytest.mark.parametrize("matrix,offset", _affine_cases())
def test_pixel_affine_correction_matches_source_for_each_distortion_family(
    family: str, matrix: torch.Tensor, offset: torch.Tensor
) -> None:
    parent = _family_wcs(family)
    source = _family_source_pixels(family)
    corrected_pixels = source @ matrix.T + offset
    corrected = parent.with_pixel_affine_correction(matrix, offset)

    torch.testing.assert_close(
        corrected.pixel_to_world(corrected_pixels),
        parent.pixel_to_world(source),
        atol=4e-10,
        rtol=0,
    )
    torch.testing.assert_close(
        corrected.world_to_pixel(parent.pixel_to_world(source)),
        corrected_pixels,
        atol=3e-4 if family in {"sip", "tpv"} else 2e-8,
        rtol=0,
    )


@pytest.mark.parametrize(
    "family", ["sip", "tpv", "cpdis", "det2im", "tabular", "coupled"]
)
def test_chained_affine_and_serialized_live_state_are_exact(family: str) -> None:
    parent = _family_wcs(family)
    matrix1, offset1 = _affine_cases()[2]
    matrix2, offset2 = _affine_cases()[3]
    source = _family_source_pixels(family)[:2]
    corrected1 = source @ matrix1.T + offset1
    corrected2 = corrected1 @ matrix2.T + offset2
    chained = parent.with_affine(matrix1, offset1).with_affine(matrix2, offset2)

    torch.testing.assert_close(
        chained.pixel_to_world(corrected2), parent.pixel_to_world(source), atol=4e-10, rtol=0
    )
    restored_kwargs = {
        name: getattr(chained, name)
        for name in ("cpdis1", "cpdis2", "det2im1", "det2im2")
    }
    restored_kwargs.update(
        {
            "tabular_axes": {
                axis + 1: table for axis, table in chained.tabular_axes.items()
            },
            "coupled_tabular": chained.coupled_tabular,
            "coupled_tabular_axes": tuple(
                axis + 1 for axis in chained.coupled_tabular_axes
            ),
            "coupled_tabular_pixel_axes": tuple(
                axis + 1 for axis in chained.coupled_tabular_pixel_axes
            ),
        }
    )
    restored = WCS(chained.to_header(include_naxis=True), **restored_kwargs)
    torch.testing.assert_close(
        restored.pixel_to_world(corrected2), chained.pixel_to_world(corrected2), atol=4e-10, rtol=0
    )


def _coupled_tabular_wcs() -> WCS:
    x = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    y = torch.tensor([1.0, 4.0, 8.0], dtype=torch.float64)
    xx = x[:, None].expand(3, 3)
    yy = y[None, :].expand(3, 3)
    lookup = CoupledTabularLookupWCS(
        (x, y), torch.stack((2.0 * xx + 0.3 * yy, -0.2 * xx + yy), dim=-1)
    )
    return WCS(
        {
            "NAXIS": 2,
            "NAXIS1": 12,
            "NAXIS2": 10,
            "CTYPE1": "X-TAB",
            "CTYPE2": "Y-TAB",
            "CRPIX1": 1.0,
            "CRPIX2": 1.0,
            "CRVAL1": 0.0,
            "CRVAL2": 0.0,
            "CD1_1": 0.25,
            "CD1_2": 0.0,
            "CD2_1": 0.0,
            "CD2_2": 0.5,
        },
        coupled_tabular=lookup,
        coupled_tabular_axes=(1, 2),
        coupled_tabular_pixel_axes=(1, 2),
    )


def test_coupled_tabular_slice_and_affine_preserve_lookup_state() -> None:
    parent = _coupled_tabular_wcs()
    view = (slice(1, 9, 2), slice(2, 11, 2))
    pixels = torch.tensor([[0.25, 0.5], [2.0, 1.5]], dtype=torch.float64)
    source = _slice_pixel_map(view, (12, 10), pixels)
    sliced = parent.slice(view)
    torch.testing.assert_close(sliced.pixel_to_world(pixels), parent.pixel_to_world(source))
    assert sliced.coupled_tabular is not None

    matrix, offset = _affine_cases()[3]
    corrected_pixels = source @ matrix.T + offset
    corrected = parent.with_affine(matrix, offset)
    torch.testing.assert_close(
        corrected.pixel_to_world(corrected_pixels), parent.pixel_to_world(source)
    )
    assert corrected.coupled_tabular is not None


def test_separable_tabular_slice_and_roundtrip_preserve_lookup_state() -> None:
    parent = _separable_tabular_wcs()
    view = (slice(1, 9, 2), slice(2, 11, 2))
    pixels = torch.tensor([[0.25, 0.5], [2.0, 1.5]], dtype=torch.float64)
    source = _slice_pixel_map(view, (12, 10), pixels)
    sliced = parent.slice(view)

    torch.testing.assert_close(sliced.pixel_to_world(pixels), parent.pixel_to_world(source))
    assert set(sliced.tabular_axes) == {0, 1}
    restored = WCS(
        sliced.to_header(include_naxis=True),
        tabular_axes={axis + 1: table for axis, table in sliced.tabular_axes.items()},
    )
    torch.testing.assert_close(restored.pixel_to_world(pixels), sliced.pixel_to_world(pixels))


def test_nonidentity_slice_composes_existing_pixel_affine_state() -> None:
    parent = WCS(_tan_header(distortion="sip"))
    matrix, offset = _affine_cases()[3]
    corrected = parent.with_affine(matrix, offset)
    view = (slice(3, 51, 2), slice(5, 68, 3))
    pixels = torch.tensor([[0.5, 0.25], [4.0, 3.5], [8.0, 6.0]], dtype=torch.float64)
    source = _slice_pixel_map(view, (72, 56), pixels)
    sliced = corrected.slice(view)

    torch.testing.assert_close(
        sliced.pixel_to_world(pixels), corrected.pixel_to_world(source), atol=4e-10, rtol=0
    )
    assert sliced._pixel_affine_matrix is not None
    restored = WCS(sliced.to_header(include_naxis=True))
    torch.testing.assert_close(
        restored.pixel_to_world(pixels), sliced.pixel_to_world(pixels), atol=4e-10, rtol=0
    )


def test_identity_slice_and_affine_are_bit_exact_for_all_state() -> None:
    first, second = _lookup_tables()
    parent = WCS(
        _tan_header(distortion="sip"),
        cpdis1=first,
        cpdis2=second,
        det2im1=first,
        det2im2=second,
    ).with_affine(*_affine_cases()[3])
    pixels = torch.tensor([[18.0, 14.0], [39.0, 31.0]], dtype=torch.float64)
    expected = parent.pixel_to_world(pixels)

    for identity in (
        parent.slice((slice(None), slice(None))),
        parent.with_affine(torch.eye(2, dtype=torch.float64), torch.zeros(2, dtype=torch.float64)),
    ):
        assert torch.equal(identity.pixel_to_world(pixels), expected)
        assert torch.equal(identity._pixel_affine_matrix, parent._pixel_affine_matrix)
        assert torch.equal(identity._pixel_affine_offset, parent._pixel_affine_offset)
        assert torch.equal(identity.cpdis1.data, parent.cpdis1.data)
        assert torch.equal(identity.det2im2.data, parent.det2im2.data)


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param(
            "mps",
            marks=pytest.mark.skipif(
                not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
                reason="MPS unavailable",
            ),
        ),
    ],
)
def test_sliced_lookup_distortion_preserves_gradient_device_and_dtype(device: str) -> None:
    dtype = torch.float32 if device == "mps" else torch.float64
    first, second = _lookup_tables(dtype=dtype)
    parent = WCS(_tan_header(), cpdis1=first, cpdis2=second).to(device)
    sliced = parent.slice((slice(2, 50, 2), slice(3, 66, 3)))
    pixels = torch.tensor(
        [[1.25, 2.5], [4.0, 5.0]], device=device, dtype=dtype, requires_grad=True
    )
    world = sliced.pixel_to_world(pixels)
    gradient = torch.autograd.grad(world.sum(), pixels)[0]

    assert world.device.type == device
    assert world.dtype == dtype
    assert gradient.device.type == device
    assert gradient.dtype == dtype
    assert torch.isfinite(gradient).all()
