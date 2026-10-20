from __future__ import annotations

import math

import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky._native import cpp
from torchsky.wcs import WCS
from torchsky.wcs.zenithal import deproject_zenithal, project_zenithal


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


def _force_path(wcs: WCS, path: str) -> None:
    wcs._cpp_center_pixel_to_world = None
    wcs._cpp_center_world_to_pixel = None
    wcs._disable_cpp_pixel_to_world = True
    wcs._disable_cpp_world_to_pixel = True
    wcs._disable_fused_pixel_to_world = path != "fused"
    wcs._disable_fused_world_to_pixel = path != "fused"
    if path == "fused":
        assert wcs._can_use_fused_p2w_static
        assert wcs._can_use_fused_w2p_static


def _finite_rows(values: np.ndarray | torch.Tensor) -> np.ndarray:
    array = (
        values.detach().cpu().numpy() if isinstance(values, torch.Tensor) else values
    )
    return np.isfinite(array).all(axis=-1)


@pytest.mark.parametrize("path", ["fused", "nonfused"])
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
    path: str,
    projection: str,
    pixels: np.ndarray,
    expected: np.ndarray,
) -> None:
    header = _header(projection, crval1=17.0, crval2=23.0)
    oracle = AstropyWCS(header).all_pix2world(pixels, 1)
    assert np.array_equal(_finite_rows(oracle), expected)

    wcs = WCS(header)
    _force_path(wcs, path)
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
    wcs._disable_fused_world_to_pixel = True
    wcs._disable_cpp_world_to_pixel = True
    wcs._cpp_tan_intermediate_from_radec = None
    actual = wcs.world_to_pixel(torch.from_numpy(world), origin=1)
    expected = np.array([True, False, False, False, False])
    assert np.array_equal(_finite_rows(actual), expected)
    np.testing.assert_allclose(
        actual.detach().cpu().numpy()[0], oracle[0], atol=2e-4, rtol=1e-12
    )

    ra = torch.from_numpy(angles)
    dec = torch.zeros_like(ra)
    x = torch.empty_like(ra)
    y = torch.empty_like(dec)
    empty_idx = torch.empty((0, 3), dtype=torch.int64)
    empty_coeff = torch.empty(0, dtype=torch.float64)
    cpp.wcs_world_to_pixel_fused_cpu(
        ra,
        dec,
        x,
        y,
        1.0,
        1.0,
        1.0,
        0.0,
        0.0,
        1.0,
        "TAN",
        1.0,
        4.0,
        3.0,
        0.0,
        1.0,
        0.0,
        0.0,
        1.0,
        1.0,
        0.0,
        0.0,
        0.0,
        0.0,
        math.pi,
        "pole",
        0.0,
        0.0,
        False,
        empty_idx,
        empty_coeff,
        empty_idx,
        empty_coeff,
    )
    assert np.array_equal(_finite_rows(torch.stack((x, y), dim=-1)), expected)


@pytest.mark.parametrize("projection", ["TAN", "ARC"])
@pytest.mark.parametrize("path", ["fused", "nonfused"])
def test_zenithal_handedness_matches_wcslib(projection: str, path: str) -> None:
    header = _header(projection, crval1=17.0, crval2=23.0)
    pixels = np.array([[1.0, 1.0], [2.0, 1.0], [1.0, 2.0], [-2.0, 3.0]])
    oracle = AstropyWCS(header).all_pix2world(pixels, 1)
    wcs = WCS(header)
    _force_path(wcs, path)
    actual = wcs.pixel_to_world(torch.from_numpy(pixels), origin=1)
    np.testing.assert_allclose(actual.detach().cpu().numpy(), oracle, atol=2e-11)


def test_native_zenithal_helpers_use_package_handedness() -> None:
    xi = torch.tensor([1.0, 0.0, -1.0], dtype=torch.float64)
    eta = torch.tensor([0.0, 1.0, 0.0], dtype=torch.float64)
    phi_cpp, theta_cpp = cpp.wcs_zenithal_project(xi, eta, "TAN")
    phi_py, theta_py = project_zenithal(xi, eta, "TAN")
    torch.testing.assert_close(phi_cpp, phi_py)
    torch.testing.assert_close(theta_cpp, theta_py)
    xi_cpp, eta_cpp = cpp.wcs_zenithal_deproject(phi_cpp, theta_cpp, "TAN")
    xi_py, eta_py = deproject_zenithal(phi_py, theta_py, "TAN")
    torch.testing.assert_close(xi_cpp, xi_py, atol=1e-12, rtol=0.0)
    torch.testing.assert_close(eta_cpp, eta_py, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize("path", ["fused", "nonfused"])
def test_mollweide_polar_forward_matches_wcslib(path: str) -> None:
    header = _header("MOL")
    world = np.array([[0.0, 90.0], [45.0, 89.999], [120.0, -90.0], [179.0, 0.0]])
    oracle = AstropyWCS(header).all_world2pix(world, 1)
    wcs = WCS(header)
    _force_path(wcs, path)
    actual = wcs.world_to_pixel(torch.from_numpy(world), origin=1)
    assert _finite_rows(actual).all()
    np.testing.assert_allclose(
        actual.detach().cpu().numpy(), oracle, atol=5e-7, rtol=0.0
    )


@pytest.mark.parametrize("value", [0.0, math.nan, math.inf])
def test_cea_rejects_invalid_lambda_like_wcslib(value: float) -> None:
    header = _header("CEA")
    header["PV2_1"] = value
    with pytest.raises(ValueError):
        AstropyWCS(header)
    with pytest.raises(ValueError, match="finite and non-zero"):
        WCS(header)


def test_cpp_cea_helpers_reject_invalid_lambda() -> None:
    values = torch.zeros(1, dtype=torch.float64)
    for lam in (0.0, math.nan, math.inf):
        with pytest.raises(RuntimeError, match="finite and non-zero"):
            cpp.wcs_cylindrical_project(values, values, "CEA", lam)
        with pytest.raises(RuntimeError, match="finite and non-zero"):
            cpp.wcs_cylindrical_deproject(values, values, "CEA", lam)
