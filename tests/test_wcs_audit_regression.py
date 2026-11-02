"""Regression tests for the WCS/geometry audit (parity against astropy.wcs / WCSLIB)."""

import warnings

import numpy as np
import pytest
import torch

astropy_wcs = pytest.importorskip("astropy.wcs")

from torchsky.wcs import WCS  # noqa: E402

DIFF_PIX = 60.0


def _hdr(proj, extra=None, crval=(150.0, 20.0), scale=0.2, pv=None):
    h = {
        "NAXIS": 2,
        "CRPIX1": 50.0,
        "CRPIX2": 40.0,
        "CDELT1": -scale,
        "CDELT2": scale,
        "CTYPE1": "RA---" + proj,
        "CTYPE2": "DEC--" + proj,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
    }
    h.update(pv or {})
    h.update(extra or {})
    return h


def _angsep(r1, d1, r2, d2):
    dr = ((r1 - r2 + 180.0) % 360.0 - 180.0) * np.cos(np.deg2rad(d2))
    return np.hypot(dr, d1 - d2)


def _assert_parity(h, tol_deg=1e-9, seed=0, n=200):
    rng = np.random.default_rng(seed)
    x = rng.uniform(0, 100, n)
    y = rng.uniform(0, 80, n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = astropy_wcs.WCS(h)
        ra, dec = a.all_pix2world(x, y, 0)
    t = WCS(h)
    ok = np.isfinite(ra)
    r1, d1 = (v.numpy() for v in t.pixel_to_world(torch.tensor(x), torch.tensor(y)))
    pix = torch.tensor(np.c_[x, y])
    w1 = t.pixel_to_world(pix).numpy()
    wd = t.pixel_to_world_differentiable(pix).detach().numpy()
    for r, d in ((r1, d1), (w1[:, 0], w1[:, 1]), (wd[:, 0], wd[:, 1])):
        assert np.nanmax(_angsep(r, d, ra, dec)[ok]) < tol_deg
    # inverse
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        px, py = a.all_world2pix(ra[ok], dec[ok], 0)
    q = t.world_to_pixel(torch.tensor(ra[ok]), torch.tensor(dec[ok]))
    assert np.max(np.hypot(q[0].numpy() - px, q[1].numpy() - py)) < 1e-6
    qd = (
        t.world_to_pixel_differentiable(torch.tensor(np.c_[ra[ok], dec[ok]]))
        .detach()
        .numpy()
    )
    assert np.max(np.hypot(qd[:, 0] - px, qd[:, 1] - py)) < 1e-6


@pytest.mark.parametrize("crota", [30.0, -75.0, 180.0])
def test_crota2_without_pc_cd(crota):
    _assert_parity(_hdr("TAN", {"CROTA2": crota}, scale=0.3))


def test_crota2_ignored_when_pc_present():
    h = _hdr(
        "TAN", {"CROTA2": 30.0, "PC1_1": 1.0, "PC1_2": 0.0, "PC2_1": 0.0, "PC2_2": 1.0}
    )
    _assert_parity(h)


@pytest.mark.parametrize(
    "proj", ["TAN", "SIN", "ZEA", "ARC", "STG", "AZP", "ZPN", "XPH"]
)
@pytest.mark.parametrize("lonpole", [170.0, 30.0, 0.0, -100.0])
@pytest.mark.parametrize("crval", [(150.0, 20.0), (150.0, -40.0)])
def test_zenithal_lonpole(proj, lonpole, crval):
    pv = {"PV2_1": 1.0, "PV2_2": 0.01} if proj == "ZPN" else None
    _assert_parity(_hdr(proj, {"LONPOLE": lonpole}, crval, pv=pv))


@pytest.mark.parametrize(
    "proj",
    [
        "TAN",
        "SIN",
        "ZEA",
        "ARC",
        "STG",
        "CAR",
        "CEA",
        "AIT",
        "MOL",
        "COE",
        "BON",
        "PCO",
        "TSC",
        "QSC",
    ],
)
def test_default_lonpole_at_pole(proj):
    pv = {"COE": {"PV2_1": 30.0, "PV2_2": 0.0}, "BON": {"PV2_1": 30.0}}.get(proj)
    _assert_parity(_hdr(proj, crval=(0.0, 90.0), pv=pv))
    _assert_parity(_hdr(proj, crval=(10.0, -90.0), pv=pv))


@pytest.mark.parametrize(
    "proj",
    [
        "CAR",
        "CEA",
        "MER",
        "SFL",
        "AIT",
        "MOL",
        "HPX",
        "PAR",
        "BON",
        "COE",
        "COD",
        "PCO",
        "TSC",
        "QSC",
    ],
)
@pytest.mark.parametrize(
    "extra,crval",
    [
        ({"LATPOLE": -30.0}, (150.0, 20.0)),
        ({"LONPOLE": 30.0, "LATPOLE": -20.0}, (150.0, 40.0)),
        ({"LONPOLE": 180.0}, (150.0, 0.0)),
        ({"PV1_3": 30.0, "PV1_4": -20.0}, (150.0, 40.0)),
    ],
)
def test_nonzenithal_latpole_lonpole(proj, extra, crval):
    pv = {
        "COE": {"PV2_1": 30.0, "PV2_2": 0.0},
        "COD": {"PV2_1": 30.0, "PV2_2": 0.0},
        "BON": {"PV2_1": 30.0},
    }.get(proj)
    _assert_parity(_hdr(proj, extra, crval, pv=pv))


def test_zenithal_pv1_3_is_lonpole():
    _assert_parity(_hdr("SIN", {"PV1_3": 170.0}))


def test_native_fiducial_pv1_1_not_silently_ignored():
    with pytest.raises(NotImplementedError):
        WCS(_hdr("CAR", {"PV1_1": 10.0, "PV1_2": 20.0}))


@pytest.mark.parametrize(
    "proj,pv",
    [
        ("CYP", {"PV2_1": 0.0, "PV2_2": 1.0}),
        ("CYP", {"PV2_1": 0.0, "PV2_2": 0.5}),
        ("CYP", {}),
        ("CEA", {}),
        ("CEA", {"PV2_1": 0.5}),
        ("SZP", {"PV2_1": 1.0, "PV2_2": 30.0, "PV2_3": 0.0}),
        ("SZP", {"PV2_1": 1.0, "PV2_2": 30.0, "PV2_3": 60.0}),
        ("SZP", {"PV2_1": 0.0, "PV2_2": 0.0, "PV2_3": 45.0}),
        ("AZP", {"PV2_1": 0.0, "PV2_2": 0.0}),
        ("AZP", {"PV2_1": 2.0, "PV2_2": 20.0}),
        ("AIR", {"PV2_1": 0.0}),
        ("AIR", {}),
        ("COE", {"PV2_1": 30.0, "PV2_2": 0.0}),
        ("BON", {"PV2_1": 30.0}),
    ],
)
def test_explicit_zero_pv_and_defaults(proj, pv):
    _assert_parity(_hdr(proj, pv=pv, scale=0.2))


@pytest.mark.parametrize("proj", ["TSC", "CSC", "QSC"])
def test_quadcube_negative_x_wraps(proj):
    h = {
        "NAXIS": 2,
        "CTYPE1": f"RA---{proj}",
        "CTYPE2": f"DEC--{proj}",
        "CRVAL1": 0.0,
        "CRVAL2": 0.0,
        "CRPIX1": 100.5,
        "CRPIX2": 80.25,
        "CDELT1": -0.5,
        "CDELT2": 0.5,
    }
    rng = np.random.default_rng(0)
    x, y = rng.uniform(0, 200, 2000), rng.uniform(0, 200, 2000)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ra, dec = astropy_wcs.WCS(h).all_pix2world(x, y, 0)
    t = WCS(h)
    r1, d1 = (v.numpy() for v in t.pixel_to_world(torch.tensor(x), torch.tensor(y)))
    wd = t.pixel_to_world_differentiable(torch.tensor(np.c_[x, y])).detach().numpy()
    ok = np.isfinite(ra)
    assert ok.sum() > 1900
    for r, d in ((r1, d1), (wd[:, 0], wd[:, 1])):
        assert np.array_equal(np.isfinite(r), ok)
        # CSC's polynomial approximation differs from WCSLIB at the 1e-5 deg level.
        assert np.max(_angsep(r[ok], d[ok], ra[ok], dec[ok])) < (
            1e-4 if proj == "CSC" else 1e-6
        )


def _grad_hdr(proj):
    h = {
        "NAXIS": 2,
        "CTYPE1": f"RA---{proj}",
        "CTYPE2": f"DEC--{proj}",
        "CRVAL1": 150.0,
        "CRVAL2": 30.0,
        "CRPIX1": 50.0,
        "CRPIX2": 50.0,
        "CD1_1": -0.1,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 0.1,
    }
    if proj in ("COE", "BON", "COP", "COD", "COO"):
        h["PV2_1"] = 30.0
    if proj == "ZPN":
        h.update({"PV2_1": 1.0, "PV2_3": -0.1})
    if proj == "AIR":
        h["PV2_1"] = 45.0
    if proj == "TPV":
        h.update({"PV1_1": 1.0, "PV1_4": 0.01, "PV2_1": 1.0, "PV2_5": -0.02})
    return h


def _jac(f, x):
    x = x.clone().requires_grad_(True)
    y = f(x)
    return torch.stack(
        [torch.autograd.grad(y[0, i], x, retain_graph=True)[0][0] for i in range(2)]
    )


def _fd(f, x, eps, wrap=False):
    j = torch.zeros(2, 2, dtype=torch.float64)
    for k in range(2):
        d = torch.zeros_like(x)
        d[0, k] = eps
        diff = (f(x + d) - f(x - d))[0]
        if wrap:
            diff[0] = torch.remainder(diff[0] + 180, 360) - 180
        j[:, k] = diff / (2 * eps)
    return j


@pytest.mark.parametrize(
    "proj",
    [
        "SIN",
        "ARC",
        "STG",
        "ZEA",
        "AZP",
        "AIR",
        "PCO",
        "SZP",
        "MOL",
        "TSC",
        "TPV",
        "TAN",
        "ZPN",
        "CAR",
        "AIT",
    ],
)
@pytest.mark.parametrize("off", [0.0, 1.3])
def test_jacobians_finite_and_match_fd_at_reference(proj, off):
    w = WCS(_grad_hdr(proj))
    pix = torch.tensor([[49.0 + off, 49.0 + 0.7 * off]], dtype=torch.float64)
    p2w = w.pixel_to_world_differentiable
    jp = _jac(p2w, pix)
    assert torch.isfinite(jp).all()
    assert torch.allclose(jp, _fd(p2w, pix, 1e-5, wrap=True), atol=2e-6, rtol=1e-5)
    world = p2w(pix).detach()
    w2p = w.world_to_pixel_differentiable
    jw = _jac(w2p, world)
    assert torch.isfinite(jw).all()
    assert torch.allclose(jw, _fd(w2p, world, 1e-7), atol=2e-3, rtol=1e-5)
    assert torch.allclose(jw @ jp, torch.eye(2, dtype=torch.float64), atol=1e-5)


def test_value_at_reference_pixel_unchanged_by_grad_safe_path():
    w = WCS(_grad_hdr("SIN"))
    pix = torch.tensor(
        [[49.0, 49.0], [60.0, 41.0]], dtype=torch.float64, requires_grad=True
    )
    a = w.pixel_to_world_differentiable(pix)
    b = w.pixel_to_world_differentiable(pix.detach())
    assert torch.equal(a.detach(), b)
    assert abs(a[0, 0].item() - 150.0) < 1e-12 and abs(a[0, 1].item() - 30.0) < 1e-12


def test_zpn_differentiable_backward_and_gradient():
    h = _grad_hdr("ZPN")
    h.update({"PV2_1": 1.0, "PV2_2": 0.01, "PV2_3": -0.1, "PV2_4": 0.02})
    w = WCS(h)
    pix = torch.tensor([[60.3, 41.7]], dtype=torch.float64)
    jp = _jac(w.pixel_to_world_differentiable, pix)
    jf = _fd(w.pixel_to_world_differentiable, pix, 1e-5, wrap=True)
    assert torch.allclose(jp, jf, atol=1e-7)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = astropy_wcs.WCS(h)
    ra, dec = a.all_pix2world(60.3, 41.7, 0)
    out = w.pixel_to_world_differentiable(pix)[0]
    assert (
        abs(out[0].item() - float(ra)) < 1e-9 and abs(out[1].item() - float(dec)) < 1e-9
    )
