import numpy as np
import pytest
import torch
from astropy.io import fits
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs.core import WCS as TorchWCS


PROJECTIONS = ("TAN", "SIN", "ARC", "AIT", "MOL", "HPX", "CEA", "MER")


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
) -> fits.Header:
    header = fits.Header()
    header["NAXIS"] = 2
    header["NAXIS1"] = naxis1
    header["NAXIS2"] = naxis2
    header["CTYPE1"] = f"RA---{projection}"
    header["CTYPE2"] = f"DEC--{projection}"
    header["CRPIX1"] = (naxis1 / 2.0) + 0.5
    header["CRPIX2"] = (naxis2 / 2.0) + 0.5
    header["CRVAL1"] = crval1
    header["CRVAL2"] = crval2
    header["CUNIT1"] = "deg"
    header["CUNIT2"] = "deg"

    if projection in {"AIT", "MOL", "HPX"}:
        header["CD1_1"] = -1.0
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 1.0
    elif projection in {"CEA", "MER"}:
        header["CD1_1"] = -0.5
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.5
        if projection == "CEA":
            header["PV2_1"] = 1.0
    else:
        header["CD1_1"] = -0.01
        header["CD1_2"] = 0.0
        header["CD2_1"] = 0.0
        header["CD2_2"] = 0.01

    return header


def _sample_pixels(
    header: fits.Header, n_side: int = 7
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)

    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x),
        torch.from_numpy(y),
        origin=origin,
    )

    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_tensor_matches_astropy(
    projection: str, origin: int
) -> None:
    """Stacked (N, 2) pixel tensor path must match Astropy (same as two-arg semantics)."""
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    pixels = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=-1)
    world = twcs.pixel_to_world(pixels, origin=origin)
    ra_t_np = world[..., 0].cpu().numpy().reshape(-1)
    dec_t_np = world[..., 1].cpu().numpy().reshape(-1)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_matches_separate_args(
    projection: str, origin: int
) -> None:
    """(N, 2) tensor path agrees with ``pixel_to_world(x, y)`` (dispatch sanity)."""
    header = _make_header(projection)
    twcs = TorchWCS(dict(header))
    x, y = _sample_pixels(header)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    ra_a, dec_a = twcs.pixel_to_world(xt, yt, origin=origin)
    world = twcs.pixel_to_world(torch.stack([xt, yt], dim=-1), origin=origin)
    torch.testing.assert_close(world[..., 0], ra_a, atol=1e-9, rtol=1e-12)
    torch.testing.assert_close(world[..., 1], dec_a, atol=1e-9, rtol=1e-12)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_matches_astropy(projection: str, origin: int) -> None:
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"

    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    x_t, y_t = twcs.world_to_pixel(
        torch.from_numpy(ra_valid),
        torch.from_numpy(dec_valid),
        origin=origin,
    )

    np.testing.assert_allclose(x_t.cpu().numpy(), x_ast, atol=1e-6)
    np.testing.assert_allclose(y_t.cpu().numpy(), y_ast, atol=1e-6)

    np.testing.assert_allclose(x_t.cpu().numpy(), x[valid], atol=1e-6)
    np.testing.assert_allclose(y_t.cpu().numpy(), y[valid], atol=1e-6)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_stack_tensor_matches_astropy(
    projection: str, origin: int
) -> None:
    """Stacked (N, 2) world tensor inverse must match Astropy ``all_world2pix``."""
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"

    ra_valid = ra_ast[valid]
    dec_valid = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_valid, dec_valid, origin)
    world = torch.stack(
        [torch.from_numpy(ra_valid), torch.from_numpy(dec_valid)], dim=-1
    )
    pix_t = twcs.world_to_pixel(world, origin=origin)
    x_t = pix_t[..., 0].cpu().numpy().reshape(-1)
    y_t = pix_t[..., 1].cpu().numpy().reshape(-1)

    np.testing.assert_allclose(x_t, x_ast, atol=1e-6)
    np.testing.assert_allclose(y_t, y_ast, atol=1e-6)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_forward_stack_compiled_matches_astropy(
    projection: str, origin: int
) -> None:
    """After ``compile()``, stacked forward path still matches Astropy (legacy gate)."""
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    try:
        twcs.compile(mode="reduce-overhead")
    except Exception:
        pytest.skip("torch.compile not available")

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    pixels = torch.stack([torch.from_numpy(x), torch.from_numpy(y)], dim=-1)
    world = twcs.pixel_to_world(pixels, origin=origin)
    ra_t_np = world[..., 0].cpu().numpy().reshape(-1)
    dec_t_np = world[..., 1].cpu().numpy().reshape(-1)

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)


@pytest.mark.parametrize("projection", PROJECTIONS)
@pytest.mark.parametrize("origin", [0, 1])
def test_projection_inverse_stack_compiled_matches_astropy(
    projection: str, origin: int
) -> None:
    """After ``compile()``, stacked inverse still matches Astropy (legacy gate)."""
    header = _make_header(projection)
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))
    try:
        twcs.compile(mode="reduce-overhead")
    except Exception:
        pytest.skip("torch.compile not available")

    x, y = _sample_pixels(header)
    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    assert np.any(valid), f"No finite inverse points for projection={projection}"
    ra_v = ra_ast[valid]
    dec_v = dec_ast[valid]

    x_ast, y_ast = awcs.all_world2pix(ra_v, dec_v, origin)
    world = torch.stack(
        [torch.from_numpy(ra_v), torch.from_numpy(dec_v)], dim=-1
    )
    pix = twcs.world_to_pixel(world, origin=origin)
    x_t = pix[..., 0].cpu().numpy().reshape(-1)
    y_t = pix[..., 1].cpu().numpy().reshape(-1)

    np.testing.assert_allclose(x_t, x_ast, atol=1e-6)
    np.testing.assert_allclose(y_t, y_ast, atol=1e-6)


@pytest.mark.parametrize("projection", ["CEA", "MER"])
@pytest.mark.parametrize("origin", [0, 1])
def test_off_equator_cylindrical_forward_matches_astropy(
    projection: str, origin: int
) -> None:
    header = _make_header(
        projection,
        crval1=12.0,
        crval2=-3.0,
        naxis1=1024,
        naxis2=640,
    )
    awcs = AstropyWCS(header)
    twcs = TorchWCS(dict(header))

    x = np.array([512.5, 512.5, 600.0, 700.0], dtype=np.float64)
    y = np.array([320.5, 400.0, 320.5, 500.0], dtype=np.float64)

    ra_ast, dec_ast = awcs.all_pix2world(x, y, origin)
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x),
        torch.from_numpy(y),
        origin=origin,
    )

    ra_t_np = ra_t.cpu().numpy()
    dec_t_np = dec_t.cpu().numpy()

    valid = np.isfinite(ra_ast) & np.isfinite(dec_ast)
    valid &= np.isfinite(ra_t_np) & np.isfinite(dec_t_np)
    assert np.any(valid), f"No finite off-equator parity points for projection={projection}"

    dra = _ra_wrap_delta(ra_t_np[valid], ra_ast[valid])
    ddec = dec_t_np[valid] - dec_ast[valid]

    np.testing.assert_allclose(dra, 0.0, atol=5e-9)
    np.testing.assert_allclose(ddec, 0.0, atol=5e-9)


def test_off_equator_cea_fused_matches_python_fallback() -> None:
    header = _make_header(
        "CEA",
        crval1=12.0,
        crval2=-3.0,
        naxis1=1024,
        naxis2=640,
    )
    twcs = TorchWCS(dict(header))

    x = torch.tensor([512.5, 512.5, 600.0, 700.0], dtype=torch.float64)
    y = torch.tensor([320.5, 400.0, 320.5, 500.0], dtype=torch.float64)

    ra_default, dec_default = twcs.pixel_to_world(x, y, origin=1)

    twcs._disable_fused_pixel_to_world = True
    twcs._disable_cpp_pixel_to_world = True
    ra_python, dec_python = twcs.pixel_to_world(x, y, origin=1)

    np.testing.assert_allclose(ra_default.cpu().numpy(), ra_python.cpu().numpy(), atol=5e-9)
    np.testing.assert_allclose(dec_default.cpu().numpy(), dec_python.cpu().numpy(), atol=5e-9)
