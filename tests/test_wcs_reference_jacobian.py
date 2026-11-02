"""Exact autograd Jacobians at and near CRPIX for every projection family."""

import math

import pytest
import torch

from torchsky.wcs import WCS

CASES = [
    ("TAN", {}),
    ("SIN", {}),
    ("SIN", {"PV2_1": 0.1, "PV2_2": -0.05}),
    ("ARC", {}),
    ("STG", {}),
    ("ZEA", {}),
    ("ZPN", {"PV2_0": 0.0, "PV2_1": 1.0, "PV2_2": 0.1, "PV2_3": -0.05}),
    ("AZP", {"PV2_1": 2.0, "PV2_2": 10.0}),
    ("SZP", {"PV2_1": 2.0, "PV2_2": 10.0, "PV2_3": 30.0}),
    ("AIR", {"PV2_1": 45.0}),
    ("CAR", {}),
    ("CEA", {"PV2_1": 1.0}),
    ("MER", {}),
    ("CYP", {"PV2_1": 2.0, "PV2_2": 1.0}),
    ("AIT", {}),
    ("MOL", {}),
    ("SFL", {}),
    ("PAR", {}),
    ("HPX", {}),
    ("COE", {"PV2_1": 30.0, "PV2_2": 15.0}),
    ("COD", {"PV2_1": 30.0, "PV2_2": 15.0}),
    ("COP", {"PV2_1": 45.0, "PV2_2": 10.0}),
    ("COO", {"PV2_1": 30.0, "PV2_2": 15.0}),
    ("BON", {"PV2_1": 30.0}),
    ("PCO", {}),
    ("TSC", {}),
    ("CSC", {}),
]
KINKED = [("XPH", {}), ("QSC", {})]
CRVALS = [(150.0, 20.0), (30.0, -60.0), (200.0, 0.0)]


def _hdr(proj, pv, crval, rot=0.0):
    wide = proj in {"AIT", "MOL", "HPX", "XPH", "PAR", "TSC", "CSC", "QSC", "SFL"}
    s = 0.5 if wide else 0.05
    c, sn = math.cos(math.radians(rot)), math.sin(math.radians(rot))
    h = {
        "NAXIS": 2,
        "CRPIX1": 50.0,
        "CRPIX2": 40.0,
        "CD1_1": -s * c,
        "CD1_2": s * sn,
        "CD2_1": s * sn,
        "CD2_2": s * c,
        "CTYPE1": "RA---" + proj,
        "CTYPE2": "DEC--" + proj,
        "CRVAL1": crval[0],
        "CRVAL2": crval[1],
    }
    h.update(pv)
    return h


def _richardson(f, x, h=2e-3):
    def central(step):
        cols = []
        for j in range(2):
            e = torch.zeros(2, dtype=torch.float64)
            e[j] = step
            cols.append((f(x + e) - f(x - e)) / (2 * step))
        return torch.stack(cols, dim=1)

    return (4.0 * central(h / 2) - central(h)) / 3.0


def _lonlat(wcs, ref_lon):
    def f(p):
        w = wcs.pixel_to_world(p[None])[0]
        return torch.stack([(w[0] - ref_lon + 180.0) % 360.0 - 180.0, w[1]])

    return f


@pytest.mark.parametrize("crval", CRVALS)
@pytest.mark.parametrize(
    "proj,pv", CASES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_reference_jacobian_matches_finite_differences(proj, pv, crval):
    wcs = WCS(_hdr(proj, pv, crval, rot=17.0))
    x0 = torch.tensor([49.0, 39.0], dtype=torch.float64)
    world0 = wcs.pixel_to_world(x0[None])[0]
    f = _lonlat(wcs, float(world0[0]))
    ref = _richardson(f, x0)
    jac = torch.autograd.functional.jacobian(
        lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x0
    )
    assert torch.isfinite(jac).all()
    scale = float(ref.abs().max())
    # ZPN (Newton root) and CSC (polynomial fit) carry ~1e-9 forward noise.
    rtol = 1e-6 if proj in {"ZPN", "CSC"} else 1e-8
    assert float((jac - ref).abs().max()) < rtol * scale

    # inverse direction: exact inverse Jacobian at the reference world point
    jinv = torch.autograd.functional.jacobian(
        lambda w: wcs.world_to_pixel_differentiable(w[None])[0],
        world0.detach().clone(),
    )
    assert torch.isfinite(jinv).all()
    assert float((jinv @ jac - torch.eye(2, dtype=torch.float64)).abs().max()) < 1e-8


@pytest.mark.parametrize("crval", CRVALS)
@pytest.mark.parametrize(
    "proj,pv", CASES, ids=lambda v: v if isinstance(v, str) else ""
)
def test_jacobian_continuous_across_reference(proj, pv, crval):
    wcs = WCS(_hdr(proj, pv, crval, rot=17.0))

    def jac_at(offset):
        x = torch.tensor([49.0 + offset, 39.0 + 0.4 * offset], dtype=torch.float64)
        return torch.autograd.functional.jacobian(
            lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x
        )

    j0, j_near = jac_at(0.0), jac_at(1e-5)
    assert float((j0 - j_near).abs().max()) < 1e-6 * float(j0.abs().max())


@pytest.mark.parametrize("proj", ["TAN", "SIN", "ARC", "STG", "ZEA"])
@pytest.mark.parametrize("crval", CRVALS)
def test_zenithal_reference_jacobian_is_analytic(proj, crval):
    """R'(theta_0) = -(180/pi) for these families: J = diag(-s/cos(d0), s)."""
    s = 0.05
    h = _hdr(proj, {}, crval)
    h.pop("CD1_1"), h.pop("CD1_2"), h.pop("CD2_1"), h.pop("CD2_2")
    h.update({"CDELT1": -s, "CDELT2": s})
    wcs = WCS(h)
    x0 = torch.tensor([49.0, 39.0], dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(
        lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x0
    )
    expected = torch.tensor(
        [[-s / math.cos(math.radians(crval[1])), 0.0], [0.0, s]], dtype=torch.float64
    )
    assert float((jac - expected).abs().max()) < 1e-12


def test_zpn_reference_jacobian_uses_linear_coefficient():
    p1 = 1.3
    h = _hdr("ZPN", {"PV2_0": 0.0, "PV2_1": p1, "PV2_2": 0.2}, (150.0, 20.0))
    h.pop("CD1_2"), h.pop("CD2_1")
    wcs = WCS(h)
    x0 = torch.tensor([49.0, 39.0], dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(
        lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x0
    )
    s = 0.05
    expected = torch.tensor(
        [[-s / (p1 * math.cos(math.radians(20.0))), 0.0], [0.0, s / p1]],
        dtype=torch.float64,
    )
    assert float((jac - expected).abs().max()) < 1e-12


@pytest.mark.parametrize(
    "proj,pv", KINKED, ids=lambda v: v if isinstance(v, str) else ""
)
def test_kinked_projections_return_consistent_subgradient(proj, pv):
    """XPH/QSC are not differentiable at CRPIX: a branch derivative is returned and
    the world-to-pixel Jacobian is its exact inverse."""
    wcs = WCS(_hdr(proj, pv, (150.0, 20.0)))
    x0 = torch.tensor([49.0, 39.0], dtype=torch.float64)
    jac = torch.autograd.functional.jacobian(
        lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x0
    )
    world0 = wcs.pixel_to_world(x0[None])[0]
    jinv = torch.autograd.functional.jacobian(
        lambda w: wcs.world_to_pixel_differentiable(w[None])[0],
        world0.detach().clone(),
    )
    assert torch.isfinite(jac).all() and torch.isfinite(jinv).all()
    assert float((jinv @ jac - torch.eye(2, dtype=torch.float64)).abs().max()) < 1e-8
    # deterministic: repeated evaluation returns the same subgradient
    jac2 = torch.autograd.functional.jacobian(
        lambda p: wcs.pixel_to_world_differentiable(p[None])[0], x0
    )
    assert torch.equal(jac, jac2)


def test_batch_mixes_reference_and_regular_rows():
    wcs = WCS(_hdr("SIN", {}, (150.0, 20.0)))
    pix = torch.tensor(
        [[49.0, 39.0], [60.0, 30.0], [49.0, 39.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    out = wcs.pixel_to_world_differentiable(pix)
    out[:, 1].sum().backward()
    assert pix.grad is not None and torch.isfinite(pix.grad).all()
    assert torch.allclose(pix.grad[0], pix.grad[2])
