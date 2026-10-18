import numpy as np
import pytest
import torch

from torchsky.wcs.core import WCS as TorchWCS


PROJECTIONS = (
    "TAN",
    "SIN",
    "ARC",
    "AIT",
    "MOL",
    "HPX",
    "CEA",
    "MER",
    "SFL",
    "CAR",
    "ZEA",
    "STG",
    "CYP",
)


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
) -> dict:  # type: ignore
    header = dict()  # type: ignore
    header["NAXIS"] = 2  # type: ignore
    header["NAXIS1"] = naxis1  # type: ignore
    header["NAXIS2"] = naxis2  # type: ignore
    header["CTYPE1"] = f"RA---{projection}"  # type: ignore
    header["CTYPE2"] = f"DEC--{projection}"  # type: ignore
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5  # type: ignore
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5  # type: ignore
    header["CRVAL1"] = crval1  # type: ignore
    header["CRVAL2"] = crval2  # type: ignore
    header["CUNIT1"] = "deg"  # type: ignore
    header["CUNIT2"] = "deg"  # type: ignore

    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 1.0  # type: ignore
    elif projection in {"CEA", "MER", "CYP", "CAR", "SFL"}:
        header["CD1_1"] = -0.5  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 0.5  # type: ignore
        if projection == "CEA":
            header["PV2_1"] = 1.0  # type: ignore
        if projection == "CYP":
            header["PV2_1"] = 2.0  # type: ignore
            header["PV2_2"] = 1.0  # type: ignore
    else:
        header["CD1_1"] = -0.01  # type: ignore
        header["CD1_2"] = 0.0  # type: ignore
        header["CD2_1"] = 0.0  # type: ignore
        header["CD2_2"] = 0.01  # type: ignore

    return header  # type: ignore


def _sample_pixels(
    header: dict,
    n_side: int = 7,  # type: ignore
) -> tuple[np.ndarray, np.ndarray]:  # type: ignore
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)  # type: ignore
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)  # type: ignore
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0  # type: ignore






@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_matches_separate_args(
    projection: str, origin: int
) -> None:  # type: ignore
    """(N, 2) tensor path agrees with ``pixel_to_world(x, y)`` (dispatch sanity)."""
    header = _make_header(projection)  # type: ignore
    twcs = TorchWCS(dict(header))  # type: ignore
    x, y = _sample_pixels(header)  # type: ignore
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    ra_a, dec_a = twcs.pixel_to_world(xt, yt, origin=origin)
    world = twcs.pixel_to_world(torch.stack([xt, yt], dim=-1), origin=origin)  # type: ignore
    torch.testing.assert_close(world[..., 0], ra_a, atol=1e-9, rtol=1e-12)  # type: ignore
    torch.testing.assert_close(world[..., 1], dec_a, atol=1e-9, rtol=1e-12)  # type: ignore












def test_off_equator_cea_fused_matches_python_fallback() -> None:
    header = _make_header(  # type: ignore
        "CEA",
        crval1=12.0,
        crval2=-3.0,  # type: ignore
        naxis1=1024,
        naxis2=640,
    )
    twcs = TorchWCS(dict(header))  # type: ignore

    x = torch.tensor([512.5, 512.5, 600.0, 700.0], dtype=torch.float64)
    y = torch.tensor([320.5, 400.0, 320.5, 500.0], dtype=torch.float64)

    ra_default, dec_default = twcs.pixel_to_world(x, y, origin=1)

    twcs._disable_fused_pixel_to_world = True
    twcs._disable_cpp_pixel_to_world = True
    ra_python, dec_python = twcs.pixel_to_world(x, y, origin=1)

    np.testing.assert_allclose(
        ra_default.cpu().numpy(), ra_python.cpu().numpy(), atol=5e-9
    )
    np.testing.assert_allclose(
        dec_default.cpu().numpy(), dec_python.cpu().numpy(), atol=5e-9
    )




def test_float32_two_arg_matches_stacked() -> None:
    header = _make_header("TAN")
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    xt32 = torch.from_numpy(x).to(dtype=torch.float32)
    yt32 = torch.from_numpy(y).to(dtype=torch.float32)
    ra2, dec2 = twcs.pixel_to_world(xt32, yt32, origin=0)
    world = twcs.pixel_to_world(torch.stack([xt32, yt32], dim=-1), origin=0)
    torch.testing.assert_close(world[..., 0], ra2, atol=1e-9, rtol=1e-12)
    torch.testing.assert_close(world[..., 1], dec2, atol=1e-9, rtol=1e-12)
    assert ra2.dtype == torch.float64
    assert world.dtype == torch.float64


@pytest.mark.skipif(
    not (hasattr(torch.backends, "mps") and torch.backends.mps.is_available()),
    reason="MPS not available",
)
def test_mps_two_arg_world_to_pixel_stays_float32() -> None:
    header = _make_header("TAN")
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    ra = torch.from_numpy(x).to(device="mps", dtype=torch.float32)
    dec = torch.from_numpy(y).to(device="mps", dtype=torch.float32)
    px, py = twcs.world_to_pixel(ra, dec, origin=0)
    stacked = twcs.world_to_pixel(torch.stack([ra, dec], dim=-1), origin=0)
    assert px.device.type == "mps"
    assert px.dtype == torch.float32
    assert stacked.dtype == torch.float32
    torch.testing.assert_close(stacked[..., 0], px, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(stacked[..., 1], py, atol=1e-5, rtol=1e-5)
