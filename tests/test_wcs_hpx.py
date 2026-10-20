import math

import numpy as np
import pytest
import torch
from astropy.wcs import Prjprm
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs import WCS


def _projection(h: float, k: float) -> Prjprm:
    projection = Prjprm()
    projection.code = "HPX"
    projection.set_pvi(1, h)
    projection.set_pvi(2, k)
    projection.set()
    return projection


def _header(h: float, k: float) -> dict[str, float | str | int]:
    return {
        "NAXIS": 2,
        "CTYPE1": "RA---HPX",
        "CTYPE2": "DEC--HPX",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CD1_1": 1.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0,
        "PV2_1": h,
        "PV2_2": k,
    }


def _force_path(wcs: WCS, path: str) -> None:
    wcs._disable_fused_pixel_to_world = path != "fused"
    wcs._disable_fused_world_to_pixel = path != "fused"
    wcs._disable_cpp_pixel_to_world = path == "python"
    wcs._disable_cpp_world_to_pixel = path == "python"
    if path != "center":
        wcs._cpp_center_pixel_to_world = None
        wcs._cpp_center_world_to_pixel = None
    if path == "standalone":
        assert wcs._cpp_hpx_project is None
        assert wcs._cpp_hpx_deproject is None
    if path == "fused":
        assert wcs._can_use_fused_p2w_static
        assert wcs._can_use_fused_w2p_static


def _wrapped_delta(actual: np.ndarray, expected: np.ndarray) -> np.ndarray:
    return np.remainder(actual - expected + 180.0, 360.0) - 180.0


@pytest.mark.parametrize("h,k", [(4.0, 3.0), (6.0, 4.0), (3.0, 2.0)])
def test_hpx_cpp_standalone_matches_wcslib_across_geometry(h: float, k: float) -> None:
    import torchsky._cpp as cpp

    transition = math.degrees(math.asin((k - 1.0) / k))
    phi = np.array(
        [-180.0, -120.0, -90.0, -1e-12, 0.0, 1e-12, 45.0, 90.0, 120.0, 179.0]
    )
    theta = np.array(
        [-90.0, -70.0, -transition, -1e-12, 0.0, 1e-12, transition, 70.0, 89.0, 90.0]
    )
    expected_x, expected_y = _projection(h, k).prjs2x(phi, theta)
    actual_x, actual_y = cpp.wcs_hpx_deproject(
        torch.from_numpy(phi), torch.from_numpy(theta), h, k
    )
    np.testing.assert_allclose(actual_x.numpy(), expected_x, atol=2e-12, rtol=0.0)
    np.testing.assert_allclose(actual_y.numpy(), expected_y, atol=2e-12, rtol=0.0)

    expected_phi, expected_theta = _projection(h, k).prjx2s(expected_x, expected_y)
    actual_phi, actual_theta = cpp.wcs_hpx_project(
        torch.from_numpy(expected_x), torch.from_numpy(expected_y), h, k
    )
    non_pole = np.abs(theta) < 90.0
    np.testing.assert_allclose(
        _wrapped_delta(actual_phi.numpy()[non_pole], expected_phi[non_pole]),
        0.0,
        atol=2e-11,
        rtol=0.0,
    )
    np.testing.assert_allclose(actual_theta.numpy(), expected_theta, atol=2e-11, rtol=0.0)


@pytest.mark.parametrize("h,k", [(4.0, 3.0), (6.0, 4.0), (3.0, 2.0)])
def test_hpx_cpp_plane_domain_matches_wcslib(h: float, k: float) -> None:
    import torchsky._cpp as cpp

    half_width = 180.0 / h
    pole_eta = 90.0 * (k + 1.0) / h
    eta = pole_eta - 0.25 * half_width
    sigma = 0.25
    center = -180.0 + half_width
    xi = np.array(
        [center, center + half_width * sigma, center + half_width * sigma + 1e-6, center, center]
    )
    yi = np.array([eta, eta, eta, pole_eta, pole_eta + 1e-6])
    expected_phi, expected_theta = _projection(h, k).prjx2s(xi, yi)
    actual_phi, actual_theta = cpp.wcs_hpx_project(
        torch.from_numpy(xi), torch.from_numpy(yi), h, k
    )
    assert np.array_equal(np.isfinite(actual_phi.numpy()), np.isfinite(expected_phi))
    assert np.array_equal(np.isfinite(actual_theta.numpy()), np.isfinite(expected_theta))


@pytest.mark.parametrize("path", ["standalone", "center", "fused", "python"])
@pytest.mark.parametrize("h,k", [(4.0, 3.0), (6.0, 4.0), (3.0, 2.0)])
def test_hpx_forced_paths_match_astropy_both_directions(
    path: str, h: float, k: float
) -> None:
    header = _header(h, k)
    oracle = AstropyWCS(header)
    wcs = WCS(header).to("cpu")
    _force_path(wcs, path)
    transition = math.degrees(math.asin((k - 1.0) / k))
    world = np.array(
        [
            [0.0, 0.0],
            [60.0, transition],
            [120.0, -transition],
            [1e-10, 70.0],
            [360.0 - 1e-10, -70.0],
            [45.0, 89.0],
        ],
        dtype=np.float64,
    )
    expected_pixel = oracle.all_world2pix(world, 1)
    actual_x, actual_y = wcs.world_to_pixel(
        torch.from_numpy(world[:, 0]), torch.from_numpy(world[:, 1]), origin=1
    )
    actual_pixel = np.column_stack((actual_x.numpy(), actual_y.numpy()))
    np.testing.assert_allclose(actual_pixel, expected_pixel, atol=3e-10, rtol=0.0)

    expected_world = oracle.all_pix2world(expected_pixel, 1)
    actual_ra, actual_dec = wcs.pixel_to_world(
        torch.from_numpy(expected_pixel[:, 0]),
        torch.from_numpy(expected_pixel[:, 1]),
        origin=1,
    )
    np.testing.assert_allclose(
        _wrapped_delta(actual_ra.numpy(), expected_world[:, 0]), 0.0, atol=3e-10, rtol=0.0
    )
    np.testing.assert_allclose(actual_dec.numpy(), expected_world[:, 1], atol=3e-10, rtol=0.0)


@pytest.mark.parametrize(
    "h,k",
    [
        (0.0, 3.0),
        (-1.0, 3.0),
        (4.5, 3.0),
        (math.nan, 3.0),
        (4.0, 0.0),
        (4.0, -1.0),
        (4.0, 3.5),
        (4.0, math.inf),
    ],
)
def test_hpx_rejects_invalid_parameters(h: float, k: float) -> None:
    import torchsky._cpp as cpp

    with pytest.raises(ValueError, match="HPX [HK] must be a finite positive integer"):
        WCS(_header(h, k))
    values = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(RuntimeError, match="HPX [HK] must be a finite positive integer"):
        cpp.wcs_hpx_project(values, values, h, k)


def test_hpx_default_parameters_are_unchanged() -> None:
    explicit = WCS(_header(4.0, 3.0))
    implicit_header = _header(4.0, 3.0)
    del implicit_header["PV2_1"]
    del implicit_header["PV2_2"]
    implicit = WCS(implicit_header)
    world = torch.tensor([[0.0, 0.0], [45.0, 45.0], [120.0, -80.0]], dtype=torch.float64)
    torch.testing.assert_close(
        implicit.world_to_pixel(world, origin=1),
        explicit.world_to_pixel(world, origin=1),
        atol=0.0,
        rtol=0.0,
    )
