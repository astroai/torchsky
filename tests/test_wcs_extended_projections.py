"""Parity tests for extended WCS projections (AZP, SZP, AIR, PAR, BON, COP, COE, COD, COO).

Validates torch-native WCS transforms against Astropy/WCSLIB fixture behavior
for forward (pixel->world) and inverse (world->pixel) directions.
"""

import os
import numpy as np
import pytest
import torch
from astropy.wcs import Prjprm
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs.core import WCS as TorchWCS
from torchsky.wcs import quadcube
from torchsky.wcs.zenithal import deproject_zenithal, project_zenithal


def _make_header(
    projection: str,
    *,
    crval1: float = 180.0,
    crval2: float = 0.0,
    naxis1: int = 256,
    naxis2: int = 256,
    pv2_1: float | None = None,
    pv2_2: float | None = None,
    cd_scale: float = 0.01,
) -> dict:
    header = {}
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
    header["CD1_1"] = -cd_scale
    header["CD1_2"] = 0.0
    header["CD2_1"] = 0.0
    header["CD2_2"] = cd_scale

    if pv2_1 is not None:
        header["PV2_1"] = pv2_1
    if pv2_2 is not None:
        header["PV2_2"] = pv2_2

    return header


def _sample_pixels(
    header: dict,
    n_side: int = 7,
) -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(96.0, header["NAXIS1"] - 96.0, n_side, dtype=np.float64)
    y = np.linspace(96.0, header["NAXIS2"] - 96.0, n_side, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def _ra_wrap_delta(ra_a: np.ndarray, ra_b: np.ndarray) -> np.ndarray:
    return ((ra_a - ra_b + 180.0) % 360.0) - 180.0


# --- Zenithal projections ---


def test_air_antipode_and_near_antipode_match_wcslib() -> None:
    projection = Prjprm()
    projection.code = "AIR"
    projection.set_pvi(1, 90.0)
    phi = np.array([0.0, 30.0, -45.0], dtype=np.float64)
    theta = np.array([-89.999999, -89.9, -80.0], dtype=np.float64)
    x_ref, y_ref = projection.prjs2x(phi, theta)

    xi, eta = deproject_zenithal(torch.from_numpy(phi), torch.from_numpy(theta), "AIR")
    np.testing.assert_allclose(xi.numpy(), -x_ref, rtol=2.0e-10, atol=1.0e-8)
    np.testing.assert_allclose(eta.numpy(), y_ref, rtol=3.0e-8, atol=1.0e-8)

    phi_back, theta_back = project_zenithal(xi, eta, "AIR")
    np.testing.assert_allclose(_ra_wrap_delta(phi_back.numpy(), phi), 0.0, atol=2.0e-10)
    np.testing.assert_allclose(theta_back.numpy(), theta, atol=2.0e-9)

    xi_pole, eta_pole = deproject_zenithal(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor(-90.0, dtype=torch.float64),
        "AIR",
    )
    x_pole, y_pole = projection.prjs2x(
        np.array([0.0], dtype=np.float64), np.array([-90.0], dtype=np.float64)
    )
    assert np.isnan(x_pole[0]) and np.isnan(y_pole[0])
    assert torch.isnan(xi_pole) and torch.isnan(eta_pole)


# --- Pseudocylindrical (PAR) ---


# --- Bonne (BON) ---


# --- Conic projections (COP, COE, COD, COO) ---

# All conic projections follow the WCSLIB convention: PV2_1 = sigma
# (reference / cone-apex latitude), PV2_2 = delta (half-difference of
# standard parallels).  The two standard parallels are sigma - delta
# and sigma + delta.


# --- PCO (American polyconic) ---


# --- XPH (HEALPix polar / butterfly) ---


# --- NCP (obsolete SIN slant) ---


# --- Quad-cube (TSC, CSC, QSC) ---


@pytest.mark.skipif(
    os.environ.get("TORCHSKY_DISABLE_CPP") == "1", reason="Requires fused C++ backend"
)
def test_csc_python_matches_fused_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """ATen and fused CSC must independently match WCSLIB float32 semantics."""
    header = _make_header("CSC", cd_scale=0.3)
    awcs = AstropyWCS(header)
    x, y = _sample_pixels(header, n_side=15)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    monkeypatch.setenv("TORCHSKY_WCS_FUSED", "1")
    fused = TorchWCS(dict(header))
    monkeypatch.setenv("TORCHSKY_WCS_FUSED", "0")
    python = TorchWCS(dict(header))
    assert fused._can_use_fused_p2w_static
    assert not python._can_use_fused_p2w_static
    ra_f, dec_f = fused.pixel_to_world(xt, yt, origin=0)
    ra_p, dec_p = python.pixel_to_world(xt, yt, origin=0)
    dra = _ra_wrap_delta(ra_p.numpy(), ra_f.numpy())
    ddec = dec_p.numpy() - dec_f.numpy()
    np.testing.assert_allclose(dra, 0.0, atol=1e-6)
    np.testing.assert_allclose(ddec, 0.0, atol=1e-6)
    ra_a, dec_a = awcs.all_pix2world(x, y, 0)
    dra_a = _ra_wrap_delta(ra_p.numpy(), ra_a)
    ddec_a = dec_p.numpy() - dec_a
    np.testing.assert_allclose(dra_a, 0.0, atol=1e-6)
    np.testing.assert_allclose(ddec_a, 0.0, atol=1e-6)
    np.testing.assert_allclose(_ra_wrap_delta(ra_f.numpy(), ra_a), 0.0, atol=1e-6)
    np.testing.assert_allclose(dec_f.numpy() - dec_a, 0.0, atol=1e-6)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
@pytest.mark.parametrize(
    "projection,cd_scale,crval2",
    [
        ("PCO", 0.5, 0.0),
        ("XPH", 0.5, 0.0),
        ("TSC", 0.3, 0.0),
        ("CSC", 0.3, 0.0),
        ("QSC", 0.3, 0.0),
        ("NCP", 0.01, 45.0),
    ],
)
def test_paper2_cuda_matches_cpu(
    projection: str, cd_scale: float, crval2: float
) -> None:
    header = _make_header(projection, crval2=crval2, cd_scale=cd_scale)
    cpu = TorchWCS(dict(header))
    gpu = TorchWCS(dict(header)).to("cuda")
    assert gpu._pv2_tensor.device.type == "cuda"
    x, y = _sample_pixels(header)
    xt = torch.from_numpy(x)
    yt = torch.from_numpy(y)
    ra_c, dec_c = cpu.pixel_to_world(xt, yt, origin=0)
    ra_g, dec_g = gpu.pixel_to_world(xt.cuda(), yt.cuda(), origin=0)
    assert ra_g.device.type == "cuda"
    valid = (
        torch.isfinite(ra_c)
        & torch.isfinite(dec_c)
        & torch.isfinite(ra_g.cpu())
        & torch.isfinite(dec_g.cpu())
    )
    assert bool(valid.any()), f"No finite CUDA points for {projection}"
    dra = _ra_wrap_delta(ra_g.cpu().numpy()[valid.numpy()], ra_c.numpy()[valid.numpy()])
    ddec = dec_g.cpu().numpy()[valid.numpy()] - dec_c.numpy()[valid.numpy()]
    # CSC evaluates the WCSLIB Chan-O'Neill polynomial in float32; at the
    # sextant seams that amplifies the CUDA/CPU operation-order difference to
    # ~1e-6 even though both paths agree away from the seams. TSC/QSC and all
    # non-quad-cube projections are expected to match at 1e-10.
    parity_atol = 1.0e-6 if projection == "CSC" else 1.0e-10
    np.testing.assert_allclose(dra, 0.0, atol=parity_atol)
    np.testing.assert_allclose(ddec, 0.0, atol=parity_atol)
    ra_v = ra_c[valid]
    dec_v = dec_c[valid]
    x_c, y_c = cpu.world_to_pixel(ra_v, dec_v, origin=0)
    x_g, y_g = gpu.world_to_pixel(ra_v.cuda(), dec_v.cuda(), origin=0)
    np.testing.assert_allclose(x_g.cpu().numpy(), x_c.numpy(), atol=parity_atol)
    np.testing.assert_allclose(y_g.cpu().numpy(), y_c.numpy(), atol=parity_atol)


def test_csc_forward_poly_is_independent_of_tensor_length():
    # Fused vs unfused addcmul differed between the vectorised body and scalar
    # tail of the CPU kernel (len 7 vs len 64 prefix disagreed in 1 ulp).
    g = torch.Generator().manual_seed(1)
    c = torch.rand(200_000, generator=g, dtype=torch.float64) * 1.6 - 0.8
    p = torch.rand(200_000, generator=g, dtype=torch.float64) * 1.6 - 0.8
    full = quadcube._csc_forward_poly(c, p)
    for n in (7, 13, 1001):
        assert torch.equal(quadcube._csc_forward_poly(c[:n], p[:n]), full[:n])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
def test_csc_s2x_x2s_cuda_matches_cpu():
    g = torch.Generator().manual_seed(2)
    phi = torch.rand(50_000, generator=g, dtype=torch.float64) * 360 - 180
    th = torch.rand(50_000, generator=g, dtype=torch.float64) * 180 - 90
    for fn in (quadcube._csc_s2x, quadcube._csc_x2s):
        cpu = fn(phi, th)
        gpu = fn(phi.cuda(), th.cuda())
        for a, b in zip(cpu, gpu, strict=True):
            # Off-face samples are NaN on both devices; transcendental kernels may
            # still differ by an ulp, far below the 3.6e-3 arcsec release gate.
            torch.testing.assert_close(b.cpu(), a, rtol=0.0, atol=1e-10, equal_nan=True)
