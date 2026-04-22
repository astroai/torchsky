import numpy as np
import pytest
import torch

from torchsky.wcs.core import WCS, _wrap_lon180, _wrap_lon360


def _tan_header() -> dict[str, float | str]:
    return {
        "NAXIS": 2,
        "NAXIS1": 256,
        "NAXIS2": 256,
        "CTYPE1": "RA---TAN",  # type: ignore
        "CTYPE2": "DEC--TAN",  # type: ignore
        "CRPIX1": 128.0,
        "CRPIX2": 128.0,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CD1_1": -0.01,  # type: ignore
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.01,
    }


def _center_equator_header(projection: str) -> dict[str, float | str]:
    header: dict[str, float | str] = {  # type: ignore
        "NAXIS": 2,
        "NAXIS1": 4096,
        "NAXIS2": 4096,
        "CTYPE1": f"RA---{projection}",  # type: ignore
        "CTYPE2": f"DEC--{projection}",  # type: ignore
        "CRPIX1": 2048.0,
        "CRPIX2": 2048.0,
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CUNIT1": "deg",
        "CUNIT2": "deg",
    }
    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 1.0  # type: ignore
    else:
        header["CD1_1"] = -0.5  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 0.5  # type: ignore
        if projection == "CEA":
            header["PV2_1"] = 1.0  # type: ignore
    return header  # type: ignore


def test_tan_sip_projection_suffix_parsing() -> None:
    header = _tan_header()  # type: ignore
    header["CTYPE1"] = "RA---TAN-SIP"  # type: ignore
    header["CTYPE2"] = "DEC--TAN-SIP"  # type: ignore
    header["A_ORDER"] = 2  # type: ignore
    header["B_ORDER"] = 2  # type: ignore
    header["A_2_0"] = 1.0e-4  # type: ignore
    header["B_0_2"] = -2.0e-4  # type: ignore

    wcs = WCS(header)  # type: ignore

    assert wcs._proj_code == "TAN"
    assert wcs.sip is not None


def test_header_unit_and_pole_keywords_preserved() -> None:
    header = _tan_header()  # type: ignore
    header["CUNIT1"] = "deg"  # type: ignore
    header["CUNIT2"] = "deg"  # type: ignore
    header["LONPOLE"] = 123.0  # type: ignore
    header["LATPOLE"] = 77.0  # type: ignore

    wcs = WCS(header)  # type: ignore

    assert wcs.wcs_params["CUNIT1"] == "deg"
    assert wcs.wcs_params["CUNIT2"] == "deg"
    assert wcs.phi_p == 123.0
    assert wcs.theta_p == 77.0


def test_cd_matrix_takes_priority_over_cdelt() -> None:
    header = _tan_header()  # type: ignore
    header["CDELT1"] = -0.2  # type: ignore
    header["CDELT2"] = 0.2  # type: ignore

    wcs = WCS(header)  # type: ignore

    assert torch.allclose(  # type: ignore
        wcs.cd,
        torch.tensor([[-0.01, 0.0], [0.0, 0.01]], dtype=torch.float64),  # type: ignore
    )


def test_world_to_pixel_default_origin_zero() -> None:
    wcs = WCS(_tan_header())  # type: ignore

    world = torch.tensor([[180.0, 0.0]], dtype=torch.float64)  # type: ignore
    pixel = wcs.world_to_pixel(world)

    expected = torch.tensor([[127.0, 127.0]], dtype=torch.float64)  # type: ignore
    assert torch.allclose(pixel, expected, atol=1e-9)  # type: ignore


def test_dtype_is_promoted_to_float64_for_transform_inputs() -> None:
    wcs = WCS(_tan_header())  # type: ignore

    pixels = torch.tensor([[127.0, 127.0], [140.0, 150.0]], dtype=torch.float32)  # type: ignore
    world = wcs.pixel_to_world(pixels)

    assert world.dtype == torch.float64  # type: ignore

    pixels_back = wcs.world_to_pixel(world.to(torch.float32))  # type: ignore
    assert pixels_back.dtype == torch.float64  # type: ignore


def test_to_moves_all_cached_tensors_and_tpv_coefficients() -> None:
    header = _tan_header()  # type: ignore
    header["CTYPE1"] = "RA---TPV"  # type: ignore
    header["CTYPE2"] = "DEC--TPV"  # type: ignore
    header["PV1_1"] = 1.0  # type: ignore
    header["PV2_2"] = 1.0  # type: ignore
    header["PV1_7"] = 1.0e-7  # type: ignore
    header["PV2_7"] = 1.0e-7  # type: ignore

    wcs = WCS(header)  # type: ignore

    wcs.to("cpu")  # type: ignore
    assert wcs.crpix.device.type == "cpu"
    assert wcs.cd.device.type == "cpu"
    assert wcs._sin_dec_p.device.type == "cpu"
    assert wcs.tpv is not None
    assert wcs.tpv.c1.device.type == "cpu"
    assert wcs.tpv.c2.device.type == "cpu"

    if torch.cuda.is_available():
        wcs.to("cuda")  # type: ignore
        assert wcs.crpix.device.type == "cuda"
        assert wcs.cd.device.type == "cuda"
        assert wcs._sin_dec_p.device.type == "cuda"
        assert wcs.tpv.c1.device.type == "cuda"
        assert wcs.tpv.c2.device.type == "cuda"

        # Smoke check transform path on CUDA tensors.
        x = torch.tensor([120.0, 140.0], device="cuda", dtype=torch.float64)  # type: ignore
        y = torch.tensor([120.0, 140.0], device="cuda", dtype=torch.float64)  # type: ignore
        ra, dec = wcs.pixel_to_world(x, y)
        assert torch.isfinite(ra).all()
        assert torch.isfinite(dec).all()


def test_origin_zero_and_one_consistent_shift() -> None:
    wcs = WCS(_tan_header())  # type: ignore

    x = np.array([120.0, 140.0], dtype=np.float64)  # type: ignore
    y = np.array([120.0, 140.0], dtype=np.float64)  # type: ignore

    w0 = wcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)
    w1 = wcs.pixel_to_world(
        torch.from_numpy(x + 1.0), torch.from_numpy(y + 1.0), origin=1
    )

    np.testing.assert_allclose(w0[0].cpu().numpy(), w1[0].cpu().numpy(), atol=1e-12)  # type: ignore
    np.testing.assert_allclose(w0[1].cpu().numpy(), w1[1].cpu().numpy(), atol=1e-12)  # type: ignore


def test_cpp_helpers_match_python_fallbacks_for_inverse_rotation_and_tan_intermediate() -> (
    None
):
    wcs = WCS(_tan_header()).to("cpu")  # type: ignore

    if (
        wcs._cpp_inverse_spherical_rotation_pole is None
        or wcs._cpp_tan_intermediate_from_radec is None
    ):
        return

    ra = torch.tensor([179.9, 180.0, 180.3, 181.2], dtype=torch.float64)  # type: ignore
    dec = torch.tensor([-0.4, 0.0, 0.2, 0.8], dtype=torch.float64)  # type: ignore

    phi_cpp, theta_cpp = wcs._inverse_spherical_rotation(ra, dec, "pole")
    xi_cpp, eta_cpp = wcs._tan_intermediate_from_radec(ra, dec)

    old_inv = wcs._cpp_inverse_spherical_rotation_pole
    old_tan = wcs._cpp_tan_intermediate_from_radec
    try:
        wcs._cpp_inverse_spherical_rotation_pole = None
        wcs._cpp_tan_intermediate_from_radec = None
        phi_py, theta_py = wcs._inverse_spherical_rotation(ra, dec, "pole")
        xi_py, eta_py = wcs._tan_intermediate_from_radec(ra, dec)
    finally:
        wcs._cpp_inverse_spherical_rotation_pole = old_inv
        wcs._cpp_tan_intermediate_from_radec = old_tan

    assert torch.allclose(phi_cpp, phi_py, atol=1e-12, rtol=0.0)  # type: ignore
    assert torch.allclose(theta_cpp, theta_py, atol=1e-12, rtol=0.0)  # type: ignore
    assert torch.allclose(xi_cpp, xi_py, atol=1e-12, rtol=0.0)  # type: ignore
    assert torch.allclose(eta_cpp, eta_py, atol=1e-12, rtol=0.0)  # type: ignore


@pytest.mark.parametrize("projection", ["CAR", "SFL", "CEA", "MER", "AIT", "HPX"])
def test_center_cpp_small_batch_matches_python_fallback(projection: str) -> None:
    wcs = WCS(_center_equator_header(projection)).to("cpu")  # type: ignore

    x = torch.linspace(1984.0, 2111.0, 128, dtype=torch.float64)  # type: ignore
    y = torch.linspace(1988.0, 2107.0, 128, dtype=torch.float64)  # type: ignore

    ra_cpp, dec_cpp = wcs.pixel_to_world(x, y, origin=0)
    x_cpp, y_cpp = wcs.world_to_pixel(ra_cpp, dec_cpp, origin=0)

    old_center_p2w = wcs._cpp_center_pixel_to_world
    old_center_w2p = wcs._cpp_center_world_to_pixel
    old_disable_p2w = wcs._disable_cpp_pixel_to_world
    old_disable_w2p = wcs._disable_cpp_world_to_pixel
    old_disable_fused_p2w = wcs._disable_fused_pixel_to_world
    old_disable_fused_w2p = wcs._disable_fused_world_to_pixel
    try:
        wcs._cpp_center_pixel_to_world = None
        wcs._cpp_center_world_to_pixel = None
        wcs._disable_cpp_pixel_to_world = True
        wcs._disable_cpp_world_to_pixel = True
        wcs._disable_fused_pixel_to_world = True
        wcs._disable_fused_world_to_pixel = True

        ra_py, dec_py = wcs.pixel_to_world(x, y, origin=0)
        x_py, y_py = wcs.world_to_pixel(ra_cpp, dec_cpp, origin=0)
    finally:
        wcs._cpp_center_pixel_to_world = old_center_p2w
        wcs._cpp_center_world_to_pixel = old_center_w2p
        wcs._disable_cpp_pixel_to_world = old_disable_p2w
        wcs._disable_cpp_world_to_pixel = old_disable_w2p
        wcs._disable_fused_pixel_to_world = old_disable_fused_p2w
        wcs._disable_fused_world_to_pixel = old_disable_fused_w2p

    assert torch.allclose(ra_cpp, ra_py, atol=1e-9, rtol=0.0, equal_nan=True)  # type: ignore
    assert torch.allclose(dec_cpp, dec_py, atol=1e-9, rtol=0.0, equal_nan=True)  # type: ignore
    assert torch.allclose(x_cpp, x_py, atol=1e-9, rtol=0.0, equal_nan=True)  # type: ignore
    assert torch.allclose(y_cpp, y_py, atol=1e-9, rtol=0.0, equal_nan=True)  # type: ignore


def test_center_cpp_small_batch_threshold_skips_large_inputs() -> None:
    wcs = WCS(_center_equator_header("CAR")).to("cpu")  # type: ignore

    x = torch.linspace(1024.0, 3071.0, 2048, dtype=torch.float64)  # type: ignore
    y = torch.linspace(1024.0, 3071.0, 2048, dtype=torch.float64)  # type: ignore
    ra = torch.linspace(300.0, 60.0, 2048, dtype=torch.float64)  # type: ignore
    dec = torch.linspace(-45.0, 45.0, 2048, dtype=torch.float64)  # type: ignore

    def _unexpected(*args, **kwargs):
        raise AssertionError("center helper should not run above the small-N limit")  # type: ignore

    old_center_p2w = wcs._cpp_center_pixel_to_world
    old_center_w2p = wcs._cpp_center_world_to_pixel
    try:
        wcs._cpp_center_pixel_to_world = _unexpected
        wcs._cpp_center_world_to_pixel = _unexpected
        out_ra, out_dec = wcs.pixel_to_world(x, y, origin=0)
        out_x, out_y = wcs.world_to_pixel(ra, dec, origin=0)
    finally:
        wcs._cpp_center_pixel_to_world = old_center_p2w
        wcs._cpp_center_world_to_pixel = old_center_w2p

    assert torch.isfinite(out_ra).all()
    assert torch.isfinite(out_dec).all()
    assert torch.isfinite(out_x).all()
    assert torch.isfinite(out_y).all()


def test_wrap_lon_grad_vs_nograd_close_on_boundaries() -> None:
    """Fast path (no grad) vs remainder path (grad) should agree within float noise."""
    t = torch.tensor(
        [-1.0, 0.0, 359.5, 360.0, 720.25, -180.5, 179.5],  # type: ignore
        dtype=torch.float64,  # type: ignore
    )
    t360_grad = t.clone().requires_grad_(True)
    t180_grad = t.clone().requires_grad_(True)
    out360_g = _wrap_lon360(t360_grad)
    out180_g = _wrap_lon180(t180_grad)
    out360_ng = _wrap_lon360(t.detach())
    out180_ng = _wrap_lon180(t.detach())
    assert torch.allclose(out360_g.detach(), out360_ng, atol=1e-12, rtol=0.0)  # type: ignore
    assert torch.allclose(out180_g.detach(), out180_ng, atol=1e-12, rtol=0.0)  # type: ignore
