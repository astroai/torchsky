from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import WCS


R2D = 180.0 / math.pi


def _header(
    projection: str,
    *,
    crval1: float = 0.0,
    crval2: float = 0.0,
) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "CTYPE1": f"RA---{projection}",
        "CTYPE2": f"DEC--{projection}",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": 1.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
    }
    if projection == "CEA":
        header["PV2_1"] = 1.0
    return header


def _finite_rows(values: np.ndarray | torch.Tensor) -> np.ndarray:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
    )
    return np.isfinite(array).all(axis=-1)


@pytest.mark.parametrize(
    "projection,pixels,expected",
    [
        (
            "ARC",
            np.array(
                [
                    [181.0, 1.0],
                    [1.0 + math.nextafter(180.0, math.inf), 1.0],
                    [181.0 + 1e-13, 1.0],
                ]
            ),
            np.array([True, True, False]),
        ),
        (
            "CEA",
            np.array(
                [
                    [1.0, 1.0 + R2D],
                    [1.0, 1.0 + R2D * (1.0 + 1e-13)],
                    [1.0, 1.0 + R2D * (1.0 + 1e-11)],
                ]
            ),
            np.array([True, True, False]),
        ),
        (
            "MOL",
            np.array(
                [
                    [1.0, 1.0 + math.sqrt(2.0) * R2D],
                    [2.0, 1.0 + math.sqrt(2.0) * R2D],
                    [1.0 + 2.0 * math.sqrt(2.0) * R2D, 1.0],
                    [1.0 + 2.0 * math.sqrt(2.0) * R2D * (1.0 + 1e-13), 1.0],
                ]
            ),
            np.array([True, False, True, False]),
        ),
    ],
)
def test_plane_domain_masks_match_wcslib(
    projection: str,
    pixels: np.ndarray,
    expected: np.ndarray,
) -> None:
    header = _header(projection, crval1=17.0, crval2=23.0)
    oracle = AstropyWCS(header).all_pix2world(pixels, 1)
    assert np.array_equal(_finite_rows(oracle), expected)

    wcs = WCS(header)
    actual = wcs.pixel_to_world(torch.from_numpy(pixels), origin=1)
    assert np.array_equal(_finite_rows(actual), expected)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy()[expected],
        oracle[expected],
        atol=2e-9,
        rtol=0.0,
    )


def test_tan_horizon_and_back_hemisphere_masks() -> None:
    angles = np.array([89.999, 90.0, 90.001, 91.0, 180.0])
    world = np.column_stack((angles, np.zeros_like(angles)))
    header = _header("TAN")
    oracle = AstropyWCS(header).all_world2pix(world, 1)
    assert np.array_equal(
        _finite_rows(oracle), np.array([True, True, False, False, False])
    )

    wcs = WCS(header)
    actual = wcs.world_to_pixel(torch.from_numpy(world), origin=1)
    expected = np.array([True, False, False, False, False])
    assert np.array_equal(_finite_rows(actual), expected)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy()[0], oracle[0], atol=2e-4, rtol=1e-12
    )

    # The two-argument public path must give the same masks as the stacked one.
    ra = torch.from_numpy(angles)
    dec = torch.zeros_like(ra)
    x, y = wcs.world_to_pixel(ra, dec, origin=1)
    assert np.array_equal(_finite_rows(torch.stack((x, y), dim=-1)), expected)


@pytest.mark.parametrize("projection", ["TAN", "ARC"])
def test_zenithal_handedness_matches_wcslib(projection: str) -> None:
    header = _header(projection, crval1=17.0, crval2=23.0)
    pixels = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [-2.0, 3.0]])
    oracle = AstropyWCS(header).all_pix2world(pixels, 1)
    wcs = WCS(header)
    actual = wcs.pixel_to_world(torch.from_numpy(pixels), origin=1)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), oracle, atol=2e-11)


def test_mollweide_polar_forward_matches_wcslib() -> None:
    header = _header("MOL")
    world = np.array([[0.0, 90.0], [45.0, 89.999], [120.0, -90.0], [179.0, 0.0]])
    oracle = AstropyWCS(header).all_world2pix(world, 1)
    wcs = WCS(header)
    actual = wcs.world_to_pixel(torch.from_numpy(world), origin=1)
    assert _finite_rows(actual).all()
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), oracle, atol=5e-7, rtol=0.0
    )


_DOMAIN_PV: dict[str, dict[str, float]] = {
    "CEA": {"PV2_1": 0.7},
    "CYP": {"PV2_1": 1.0, "PV2_2": 0.70710678},
    "COD": {"PV2_1": 45.0, "PV2_2": 25.0},
    "COE": {"PV2_1": 45.0, "PV2_2": 25.0},
    "COO": {"PV2_1": 45.0, "PV2_2": 25.0},
    "COP": {"PV2_1": 45.0, "PV2_2": 25.0},
    "BON": {"PV2_1": 45.0},
    "AZP": {"PV2_1": 2.0, "PV2_2": 30.0},
    "SZP": {"PV2_1": 2.0, "PV2_2": 180.0, "PV2_3": 60.0},
    "ZPN": {"PV2_0": 0.01, "PV2_1": 0.95, "PV2_3": -0.05},
    "SIN": {"PV2_1": 0.2, "PV2_2": -0.1},
}


def _domain_header(
    projection: str, crval: tuple[float, float], scale: float, pv: bool = True
) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "CTYPE1": f"RA---{projection}",
        "CTYPE2": f"DEC--{projection}",
        "CRPIX1": 101.3,
        "CRPIX2": 100.8,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
        "CD1_1": -scale,
        "CD1_2": 0.02 * scale,
        "CD2_1": 0.03 * scale,
        "CD2_2": scale,
    }
    if pv:
        header.update(_DOMAIN_PV.get(projection, {}))
    return header


@pytest.mark.parametrize(
    "projection",
    [
        "CAR",
        "CEA",
        "MER",
        "CYP",
        "SFL",
        "PAR",
        "MOL",
        "AIT",
        "COD",
        "COE",
        "COO",
        "COP",
        "BON",
        "PCO",
        "AZP",
        "SZP",
        "XPH",
    ],
)
@pytest.mark.parametrize("crval", [(0.0, 0.0), (150.0, 30.0)])
def test_plane_points_outside_native_domain_match_wcslib(
    projection: str, crval: tuple[float, float]
) -> None:
    # A 2 deg/pixel grid spanning +-300 deg of plane: most of it lies outside
    # the projection domain, where WCSLIB (strict bounds) returns NaN.
    header = _domain_header(projection, crval, 2.0)
    grid = np.linspace(-50.0, 250.0, 151)
    x, y = np.meshgrid(grid, grid)
    pixels = np.column_stack((x.ravel(), y.ravel()))
    oracle = AstropyWCS(header).wcs_pix2world(pixels, 0)
    actual = WCS(header).pixel_to_world(torch.from_numpy(pixels)).numpy()
    oracle_ok = _finite_rows(oracle)
    actual_ok = _finite_rows(actual)
    assert oracle_ok.any() and not oracle_ok.all()
    mismatch = np.flatnonzero(oracle_ok != actual_ok)
    assert mismatch.size == 0, (
        f"{mismatch.size} mask mismatches, e.g. pixel {pixels[mismatch[0]]}: "
        f"wcslib {oracle[mismatch[0]]}, torchsky {actual[mismatch[0]]}"
    )
    lon_diff = np.abs(
        (actual[oracle_ok, 0] - oracle[oracle_ok, 0] + 180.0) % 360.0 - 180.0
    )
    lon_diff *= np.cos(np.radians(oracle[oracle_ok, 1]))
    assert lon_diff.max() < 1e-8
    assert np.abs(actual[oracle_ok, 1] - oracle[oracle_ok, 1]).max() < 1e-8


@pytest.mark.parametrize(
    "projection,pv",
    [
        ("SIN", False),
        ("SIN", True),
        ("AZP", False),
        ("AZP", True),
        ("SZP", True),
        ("ZPN", True),
        ("COP", True),
    ],
)
@pytest.mark.parametrize("crval", [(150.0, 30.0), (45.0, 90.0), (210.0, -89.9)])
def test_unprojectable_world_points_match_wcslib(
    projection: str, pv: bool, crval: tuple[float, float]
) -> None:
    # Far-side (SIN), overlap/divergence (AZP, SZP), beyond-turning-point (ZPN)
    # and wrong-side-of-apex (COP) world points have no pixel in WCSLIB.
    header = _domain_header(projection, crval, 0.05, pv=pv)
    lon, lat = np.meshgrid(np.linspace(0.0, 359.0, 180), np.linspace(-89.0, 89.0, 90))
    world = np.column_stack((lon.ravel(), lat.ravel()))
    oracle = AstropyWCS(header).wcs_world2pix(world, 0)
    wcs = WCS(header)
    actual = wcs.world_to_pixel(torch.from_numpy(world)).numpy()
    # Points on a divergence circle map to ~1e16 pixels in both or are
    # rejected by one of them depending on the last bit; ignore them.
    finite_oracle = np.nan_to_num(oracle, nan=0.0)
    regular = np.hypot(finite_oracle[:, 0], finite_oracle[:, 1]) < 1e7
    oracle_ok = _finite_rows(oracle) & regular
    actual_ok = _finite_rows(actual) & regular
    assert oracle_ok.any() and not oracle_ok.all()
    mismatch = np.flatnonzero((oracle_ok != actual_ok) & regular)
    assert mismatch.size == 0, (
        f"{mismatch.size} mask mismatches, e.g. world {world[mismatch[0]]}: "
        f"wcslib {oracle[mismatch[0]]}, torchsky {actual[mismatch[0]]}"
    )
    np.testing.assert_allclose(
        actual[oracle_ok], oracle[oracle_ok], atol=1e-6, rtol=1e-9
    )
    # The two-argument public path gives the same answer.
    x, y = wcs.world_to_pixel(
        torch.from_numpy(world[:, 0].copy()), torch.from_numpy(world[:, 1].copy())
    )
    two_arg = torch.stack((x, y), dim=-1).numpy()
    np.testing.assert_array_equal(_finite_rows(two_arg), _finite_rows(actual))


@pytest.mark.parametrize("value", [0.0, math.nan, math.inf])
def test_cea_rejects_invalid_lambda_like_wcslib(value: float) -> None:
    header = _header("CEA")
    header["PV2_1"] = value
    with pytest.raises(ValueError):
        AstropyWCS(header)
    with pytest.raises(ValueError, match="finite and non-zero"):
        WCS(header)
