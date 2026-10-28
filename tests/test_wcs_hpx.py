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
    with pytest.raises(ValueError, match="HPX [HK] must be a finite positive integer"):
        WCS(_header(h, k))


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
