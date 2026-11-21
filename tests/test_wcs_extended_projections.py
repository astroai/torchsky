"""Parity tests for extended WCS projections (AZP, SZP, AIR, PAR, BON, COP, COE, COD, COO).

Validates torch-native WCS transforms against Astropy/WCSLIB fixture behavior
for forward (pixel->world) and inverse (world->pixel) directions.
"""

import numpy as np
import pytest
import torch
from astropy.wcs import Prjprm

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


# torchsky never fuses a*b+c (identical results on every platform). WCSLIB's
# float32 CSC steps round differently per build: AArch64 builds (macOS arm64
# astropy) fuse, and the conda-forge linux-64 build differed by up to 1.3e-6 deg
# on the inverse (astropy 8.0.x, CI run 44876047856). One float32 ulp is
# 45 * 2**-22 deg on the plane (offsets up to 8 face units) and ~5e-6 deg on the
# sphere; the float64 plane-coordinate bug these tests guard against was 2e-5 deg.
ATOL_FWD = 2.5e-5
ATOL_INV = 6e-6


def test_csc_matches_wcslib_float32_semantics_on_every_face():
    """CSC reproduces WCSLIB's float32 polynomials and float32 face-offset sums.

    Before, the plane coordinate was formed in float64 (``45 * xf`` + offset) and
    differed from WCSLIB by up to 2e-5 deg on the non-base faces.
    """
    astropy_wcs = pytest.importorskip("astropy.wcs")
    prj = astropy_wcs.Prjprm()
    prj.code = "CSC"
    prj.set()
    rng = np.random.default_rng(1)
    phi = rng.uniform(-180.0, 180.0, 5000)
    theta = rng.uniform(-90.0, 90.0, 5000)
    xw, yw = prj.prjs2x(phi, theta)
    xi, eta = quadcube._csc_s2x(torch.from_numpy(phi), torch.from_numpy(theta))
    np.testing.assert_allclose(xi.numpy(), xw, rtol=0, atol=ATOL_FWD)
    np.testing.assert_allclose(eta.numpy(), yw, rtol=0, atol=ATOL_FWD)
    x = rng.uniform(-45.0, 315.0, 5000)
    y = rng.uniform(-135.0, 135.0, 5000)
    pw, tw = prj.prjx2s(x, y)
    p, t = quadcube._csc_x2s(torch.from_numpy(x), torch.from_numpy(y))
    ok = np.isfinite(pw)
    assert np.array_equal(np.isfinite(p.numpy()), ok)
    np.testing.assert_allclose(
        _ra_wrap_delta(p.numpy()[ok], pw[ok]), 0.0, rtol=0, atol=ATOL_INV
    )
    np.testing.assert_allclose(t.numpy()[ok], tw[ok], rtol=0, atol=ATOL_INV)


@pytest.mark.parametrize(
    "projection,pv2",
    [
        ("SZP", (0.0, 2.0, 180.0, 60.0)),
        ("SZP", (0.0, 0.5, 30.0, 70.0)),
        ("AIR", (0.0, 45.0)),
        ("AIR", (0.0, 90.0)),
    ],
)
def test_zenithal_float32_keeps_float32_accuracy(
    projection: str, pv2: tuple[float, ...]
) -> None:
    # MPS runs these kernels in float32. Recovering theta from sin(theta) by
    # asin (SZP), or log(cos(xi)) (AIR), loses half the digits near the pole:
    # arcminute errors, and AIR's 1e-10 convergence test rejected every point.
    radius = torch.logspace(-4, 1.5, 4000, dtype=torch.float64)
    angle = torch.linspace(0.0, 2.0 * np.pi, 4000, dtype=torch.float64)
    xi = radius * torch.cos(angle)
    eta = radius * torch.sin(angle)
    phi64, theta64 = project_zenithal(xi, eta, projection, None, pv2)
    phi32, theta32 = project_zenithal(xi.float(), eta.float(), projection, None, pv2)
    ok = torch.isfinite(theta64)
    assert ok.all()
    assert torch.equal(torch.isfinite(theta32), ok)
    assert float((theta32.double() - theta64).abs().max()) < 1e-4
    dphi = (phi32.double() - phi64 + 180.0) % 360.0 - 180.0
    colat = torch.deg2rad(90.0 - theta64)
    assert float((dphi * torch.sin(colat)).abs().max()) < 1e-4

    xi64, eta64 = deproject_zenithal(phi64, theta64, projection, None, pv2)
    xi32, eta32 = deproject_zenithal(
        phi64.float(), theta64.float(), projection, None, pv2
    )
    assert float(torch.hypot(xi32.double() - xi64, eta32.double() - eta64).max()) < 1e-4


def _sphere_samples_near_centres_and_poles(
    phi_shift: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    generator = torch.Generator().manual_seed(3)
    offsets = torch.logspace(-5, 0.5, 400, dtype=torch.float64)
    angle = torch.rand(400, generator=generator, dtype=torch.float64) * 2.0 * np.pi
    phis = []
    thetas = []
    # Quad-cube face centres (and HEALPix/XPH facet centres) on the equator,
    # and both poles, approached from all directions.
    for centre_phi, centre_theta in (
        (0.0, 0.0),
        (90.0, 0.0),
        (180.0, 0.0),
        (-90.0, 0.0),
    ):
        phis.append(centre_phi + phi_shift + offsets * torch.cos(angle))
        thetas.append(centre_theta + offsets * torch.sin(angle))
    for pole in (90.0, -90.0):
        phis.append(angle * (180.0 / np.pi) - 180.0)
        thetas.append(pole - torch.sign(torch.tensor(pole)) * offsets)
    phis.append(
        torch.rand(4000, generator=generator, dtype=torch.float64) * 360.0 - 180.0
    )
    thetas.append(
        torch.rad2deg(
            torch.asin(
                torch.rand(4000, generator=generator, dtype=torch.float64) * 2.0 - 1.0
            )
        )
    )
    return torch.cat(phis), torch.cat(thetas)


# CSC is left out: WCSLIB evaluates its polynomials in float32 by design.
@pytest.mark.parametrize("projection", ["TSC", "QSC", "HPX", "XPH"])
def test_allsky_float32_keeps_float32_accuracy(projection: str) -> None:
    # 1 - cos and asin near 1 lose half the float32 digits next to face
    # centres and poles (up to 2e-3 deg); the stable forms stay at a few ulp.
    from torchsky.wcs.allsky import deproject_allsky, project_allsky

    if projection in ("TSC", "CSC", "QSC"):
        s2x = lambda p, t: quadcube.deproject_quadcube(p, t, projection)  # noqa: E731
        x2s = lambda x, y: quadcube.project_quadcube(x, y, projection)  # noqa: E731
    else:
        s2x = lambda p, t: deproject_allsky(p, t, projection)  # noqa: E731
        x2s = lambda x, y: project_allsky(x, y, projection)  # noqa: E731

    def separation(p1, t1, p2, t2):
        # Angular distance (deg); seam-agnostic, unlike plane coordinates.
        p1, t1, p2, t2 = (torch.deg2rad(v.double()) for v in (p1, t1, p2, t2))
        cos_d = torch.sin(t1) * torch.sin(t2) + torch.cos(t1) * torch.cos(
            t2
        ) * torch.cos(p1 - p2)
        cross = torch.hypot(
            torch.cos(t2) * torch.sin(p1 - p2),
            torch.cos(t1) * torch.sin(t2)
            - torch.sin(t1) * torch.cos(t2) * torch.cos(p1 - p2),
        )
        return torch.rad2deg(torch.atan2(cross, cos_d))

    # XPH facets are centred between the quadrant boundaries phi = k * 90.
    phi, theta = _sphere_samples_near_centres_and_poles(
        45.0 if projection == "XPH" else 0.0
    )
    x64, y64 = s2x(phi, theta)
    x32, y32 = s2x(phi.float(), theta.float())
    # sphere -> plane in float32, judged by mapping back in float64
    assert float(separation(*x2s(x32.double(), y32.double()), phi, theta).max()) < 1e-4
    # plane -> sphere in float32
    assert float(separation(*x2s(x64.float(), y64.float()), phi, theta).max()) < 1e-4
