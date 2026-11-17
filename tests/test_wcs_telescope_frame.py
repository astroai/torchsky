"""Tests for ``torchsky.wcs.telescope_frame`` (pixel -> ideal -> telescope -> sky).

The NIRCam science-to-ideal coefficients below are literals generated once from
``pysiaf`` 0.29.0 with JWST PRD version PRDOPSSOC-075 (apertures
``NRCB1_FULL`` and ``NRCB5_FULL``, the SIAF name of ``NRCBLONG``). Only the one
test marked ``importorskip("pysiaf")`` imports it.
"""

import copy
import math

import pytest
import torch

from torchsky.map.field.observation_ops import (
    crop_field_observation_to_shared_sky_patch,
    observation_pixel_to_world,
)
from torchsky.map.field.types import FieldObservation
from torchsky.wcs.detector_astrometry import AffineFreeResidualGrid2D
from torchsky.wcs.telescope_frame import (
    PolynomialDetectorDistortion,
    TelescopeAttitude,
    TelescopeFrameDetectorToSky,
)

F64 = torch.float64

# --- pysiaf 0.29.0, PRD PRDOPSSOC-075; coefficients in SIAF order a00, a10, a11, a20, ...
_NRCB1 = {
    "aperture": "NRCB1_FULL",
    "coeffs_x": [
        0.0,
        0.030698752454,
        0.0,
        3.7396187387e-09,
        -1.6280585258e-07,
        3.5716671727e-08,
        1.1953695854e-11,
        -8.614814524600001e-13,
        1.1743509374e-11,
        1.5232300686e-12,
        -2.5411045195e-16,
        -7.7982680849e-16,
        5.0539403436e-16,
        8.6797336326e-16,
        -1.9617850102e-16,
        -2.1814934498e-18,
        4.3010148452e-19,
        1.2141756578e-18,
        7.602571546200001e-19,
        -8.7016124175e-19,
        -1.5502128934e-18,
    ],
    "coeffs_y": [
        0.0,
        4.0713505991e-05,
        0.030793495171,
        8.6111644609e-08,
        -3.6689711148e-08,
        -7.7637950438e-08,
        -7.842339419900001e-13,
        9.2394544612e-12,
        1.0261627019e-12,
        8.9336982547e-12,
        3.4432599198e-17,
        9.390713794e-16,
        1.9568708919e-16,
        1.4481649957e-15,
        -6.586018112999999e-16,
        5.0631323004e-19,
        6.455856155400001e-19,
        -2.433850504e-19,
        1.0087840081e-18,
        -5.6450343212e-19,
        -1.2092725729e-19,
    ],
    "XSciRef": 1024.5,
    "YSciRef": 1024.5,
    "V2Ref": -122.427986,
    "V3Ref": -457.706754,
    "V3IdlYAngle": 0.34481904,
    "VIdlParity": -1,
}
_NRCBLONG = {
    "aperture": "NRCB5_FULL",
    "coeffs_x": [
        0.0,
        0.062853717411,
        8.2580439779e-07,
        -1.1860258827e-07,
        -7.3793399127e-07,
        1.0180580395e-07,
        1.0320808723e-10,
        5.6063588829e-12,
        1.1014696852e-10,
        1.7711583332e-12,
        6.3228628861e-16,
        -3.2648987349e-15,
        -2.9523824945e-16,
        -2.9082752463e-15,
        5.7570691789e-16,
        5.8544731462e-19,
        -1.0715109108e-19,
        6.707657109e-18,
        -1.5750059213e-18,
        1.7312994599e-18,
        -1.2311983495e-18,
    ],
    "coeffs_y": [
        0.0,
        8.244875759e-05,
        0.06314947292,
        3.1082701921e-07,
        -2.322713115e-07,
        -4.398738398e-07,
        -1.8020310051e-12,
        1.0646042423e-10,
        6.2466319383e-12,
        1.0943313489e-10,
        7.225244450099999e-16,
        -2.5597735549e-16,
        8.351866159899999e-16,
        -3.0894306554e-16,
        -1.5352188449e-15,
        1.2891501427e-18,
        4.991200472600001e-19,
        -1.5638134052e-18,
        4.3692520382e-18,
        -7.8359490385e-19,
        1.7592873561e-18,
    ],
    "XSciRef": 1024.5,
    "YSciRef": 1024.5,
    "V2Ref": -89.402822,
    "V3Ref": -491.360126,
    "V3IdlYAngle": 0.02319705,
    "VIdlParity": -1,
}

_FIXTURES = {"NRCB1": _NRCB1, "NRCBLONG": _NRCBLONG}
_SHAPE = (2048, 2048)
# Attitude of a real exposure (DEC_REF approximate), as in the task.
_ATT = {
    "v2_ref_arcsec": -122.427986,
    "v3_ref_arcsec": -457.706754,
    "ra_ref_deg": 110.8745219187291,
    "dec_ref_deg": -73.4540,
    "roll_ref_deg": 144.93904872291884,
}


def _poly(name: str = "NRCB1", **kwargs) -> PolynomialDetectorDistortion:
    f = _FIXTURES[name]
    return PolynomialDetectorDistortion.from_siaf(
        f["coeffs_x"],
        f["coeffs_y"],
        x_sci_ref=f["XSciRef"],
        y_sci_ref=f["YSciRef"],
        detector_shape=_SHAPE,
        **kwargs,
    )


def _attitude(name: str = "NRCB1", **kwargs) -> TelescopeAttitude:
    f = _FIXTURES[name]
    return TelescopeAttitude(
        f["V2Ref"] if name != "NRCB1" else _ATT["v2_ref_arcsec"],
        f["V3Ref"] if name != "NRCB1" else _ATT["v3_ref_arcsec"],
        _ATT["ra_ref_deg"],
        _ATT["dec_ref_deg"],
        _ATT["roll_ref_deg"],
        v3_idl_yangle_deg=f["V3IdlYAngle"],
        v_idl_parity=f["VIdlParity"],
        **kwargs,
    )


def _transform(name: str = "NRCB1", **kwargs) -> TelescopeFrameDetectorToSky:
    return TelescopeFrameDetectorToSky(_poly(name), _attitude(name), **kwargs)


def _lattice(n: int = 5, lo: float = 0.0, hi: float = 2047.0) -> torch.Tensor:
    g = torch.linspace(lo, hi, n, dtype=F64)
    yy, xx = torch.meshgrid(g, g, indexing="ij")
    return torch.stack((xx, yy), dim=-1)


def _sep_arcsec(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Great-circle separation of (RA, Dec) degree pairs, arcsec (float64)."""
    d2r = math.pi / 180.0

    def unit(p):
        return torch.stack(
            (
                torch.cos(p[..., 1] * d2r) * torch.cos(p[..., 0] * d2r),
                torch.cos(p[..., 1] * d2r) * torch.sin(p[..., 0] * d2r),
                torch.sin(p[..., 1] * d2r),
            ),
            dim=-1,
        )

    u, v = unit(a), unit(b)
    cross = torch.linalg.vector_norm(torch.linalg.cross(u, v, dim=-1), dim=-1)
    return torch.atan2(cross, (u * v).sum(-1)) / d2r * 3600.0


def _synthetic_poly(**kwargs) -> PolynomialDetectorDistortion:
    """Degree-3 polynomial on a 64 x 64 detector, moderate coefficients."""
    a = torch.zeros(4, 4, dtype=F64)
    b = torch.zeros(4, 4, dtype=F64)
    a[1, 0], a[1, 1], a[2, 0], a[2, 1], a[3, 2] = 0.03, 1e-3, 2e-6, -1e-6, 3e-8
    b[1, 0], b[1, 1], b[2, 1], b[2, 2], a[3, 0] = 2e-3, 0.031, 1e-6, -2e-6, 1e-8
    return PolynomialDetectorDistortion(
        a, b, ref_pixel=(31.5, 31.5), detector_shape=(64, 64), **kwargs
    )


def _synthetic_attitude(**kwargs) -> TelescopeAttitude:
    return TelescopeAttitude(
        -120.0,
        -450.0,
        10.0,
        20.0,
        30.0,
        v3_idl_yangle_deg=0.3,
        v_idl_parity=-1,
        **kwargs,
    )


# ---------------------------------------------------------------- polynomial


def test_polynomial_matches_hand_summation_and_siaf_ordering() -> None:
    cx = [1.0, 2.0, 0.1, 0.01, 0.02, 0.03]  # a00, a10, a11, a20, a21, a22
    cy = [-1.0, 0.05, 3.0, 0.04, 0.02, 0.01]
    a = torch.zeros(3, 3, dtype=F64)
    b = torch.zeros(3, 3, dtype=F64)
    for k, (i, j) in enumerate([(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2)]):
        a[i, j], b[i, j] = cx[k], cy[k]
    poly = PolynomialDetectorDistortion(
        a, b, ref_pixel=(10.0, 20.0), detector_shape=(40, 30)
    )
    u, v = 3.0 - 10.0, 25.0 - 20.0
    # a[i, j] multiplies u**(i - j) * v**j (SIAF Sci2IdlX{i}{j}).
    ex = 1 + 2 * u + 0.1 * v + 0.01 * u**2 + 0.02 * u * v + 0.03 * v**2
    ey = -1 + 0.05 * u + 3 * v + 0.04 * u**2 + 0.02 * u * v + 0.01 * v**2
    out = poly(torch.tensor([3.0, 25.0], dtype=F64))
    torch.testing.assert_close(
        out, torch.tensor([ex, ey], dtype=F64), atol=1e-12, rtol=0
    )
    flat = PolynomialDetectorDistortion(
        torch.tensor(cx, dtype=F64),
        torch.tensor(cy, dtype=F64),
        ref_pixel=(10.0, 20.0),
        detector_shape=(40, 30),
    )
    torch.testing.assert_close(flat.coefficients, poly.coefficients)
    tri_a, tri_b = poly.triangular_coefficients
    torch.testing.assert_close(tri_a, a)
    torch.testing.assert_close(tri_b, b)


def test_from_siaf_uses_one_based_reference_pixel() -> None:
    poly = _poly()
    torch.testing.assert_close(
        poly.ref_pixel, torch.tensor([1023.5, 1023.5], dtype=F64)
    )
    # Zero-based pixel 1023.5 is SIAF science pixel 1024.5 = (XSciRef, YSciRef):
    # the polynomial constant term (zero) is returned there.
    out = poly(torch.tensor([1023.5, 1023.5], dtype=F64))
    torch.testing.assert_close(out, torch.zeros(2, dtype=F64), atol=1e-14, rtol=0)


@pytest.mark.parametrize("name", ["NRCB1", "NRCBLONG"])
def test_polynomial_jacobian_matches_finite_differences(name: str) -> None:
    poly = _poly(name)
    pts = _lattice(4, 3.0, 2040.0).reshape(-1, 2)
    jac = poly.jacobian(pts)
    h = 1e-3
    for col in range(2):
        step = torch.zeros(2, dtype=F64)
        step[col] = h
        fd = (poly(pts + step) - poly(pts - step)) / (2 * h)
        torch.testing.assert_close(jac[..., col], fd, atol=1e-9, rtol=1e-8)
    assert bool(
        (jac[..., 0, 0] * jac[..., 1, 1] - jac[..., 0, 1] * jac[..., 1, 0] > 0).all()
    )


@pytest.mark.parametrize("name", ["NRCB1", "NRCBLONG"])
def test_polynomial_inverse_round_trip(name: str) -> None:
    poly = _poly(name)
    pts = _lattice(7, -20.0, 2067.0)
    back = poly.inverse(poly(pts))
    assert float((back - pts).abs().max()) < 1e-8


def test_polynomial_inverse_fails_closed_and_nan_mode() -> None:
    poly = _poly()
    far = torch.tensor([[2.0e5, -3.0e5], [0.0, 0.0]], dtype=F64)
    with pytest.raises(ValueError, match="did not converge"):
        poly.inverse(far)
    out = poly.inverse(far, fail_closed=False)
    assert bool(torch.isnan(out[0]).all()) and bool(torch.isfinite(out[1]).all())


def test_polynomial_rejects_folded_reflected_and_malformed_coefficients() -> None:
    f = _NRCB1
    with pytest.raises(ValueError, match="folded, reflected or singular"):
        # Reflect the x axis: determinant is negative.
        PolynomialDetectorDistortion.from_siaf(
            [-c for c in f["coeffs_x"]],
            f["coeffs_y"],
            x_sci_ref=1024.5,
            y_sci_ref=1024.5,
            detector_shape=_SHAPE,
        )
    with pytest.raises(ValueError, match="folded, reflected or singular"):
        # A huge quadratic term folds the map inside the detector.
        bad = list(f["coeffs_x"])
        bad[3] = 1.0
        PolynomialDetectorDistortion.from_siaf(
            bad,
            f["coeffs_y"],
            x_sci_ref=1024.5,
            y_sci_ref=1024.5,
            detector_shape=_SHAPE,
        )
    ok = torch.zeros(3, 3, dtype=F64)
    ok[1, 0] = 0.03
    with pytest.raises(ValueError, match="above the diagonal"):
        bad_tri = ok.clone()
        bad_tri[0, 2] = 1.0
        PolynomialDetectorDistortion(
            bad_tri, ok, ref_pixel=(0.0, 0.0), detector_shape=(8, 8)
        )
    with pytest.raises(ValueError, match="triangular number"):
        PolynomialDetectorDistortion(
            torch.zeros(5, dtype=F64),
            torch.zeros(5, dtype=F64),
            ref_pixel=(0.0, 0.0),
            detector_shape=(8, 8),
        )
    with pytest.raises(ValueError, match="same degree"):
        PolynomialDetectorDistortion(
            torch.zeros(3, dtype=F64),
            torch.zeros(6, dtype=F64),
            ref_pixel=(0.0, 0.0),
            detector_shape=(8, 8),
        )
    with pytest.raises(ValueError, match="finite"):
        nan = ok.clone()
        nan[1, 1] = float("nan")
        PolynomialDetectorDistortion(
            nan, ok, ref_pixel=(0.0, 0.0), detector_shape=(8, 8)
        )
    with pytest.raises(ValueError, match="ref_pixel"):
        PolynomialDetectorDistortion(
            ok, ok, ref_pixel=(0.0, 0.0, 1.0), detector_shape=(8, 8)
        )
    with pytest.raises(ValueError, match="detector_shape"):
        PolynomialDetectorDistortion(
            ok, ok, ref_pixel=(0.0, 0.0), detector_shape=(1, 8)
        )


def test_polynomial_rejects_bad_point_shapes_and_dtypes() -> None:
    poly = _poly()
    with pytest.raises(ValueError, match="last dimension must be 2"):
        poly(torch.zeros(4, 3, dtype=F64))
    with pytest.raises(ValueError, match="floating-point"):
        poly(torch.zeros(4, 2, dtype=torch.int64))
    with pytest.raises(ValueError, match="finite"):
        poly(torch.tensor([[float("inf"), 0.0]], dtype=F64))


def test_assert_invertible_detects_a_learned_fold() -> None:
    poly = _synthetic_poly(learnable=True)
    poly.assert_invertible()
    with torch.no_grad():
        poly.coefficients[0, 1] = -0.03  # flips the sign of d X / d x
    with pytest.raises(ValueError, match="folded, reflected or singular"):
        poly.assert_invertible()


# ------------------------------------------------------------------ attitude


def test_attitude_reference_point_and_round_trip() -> None:
    att = _attitude()
    ref = att(torch.zeros(2, dtype=F64))
    torch.testing.assert_close(
        ref,
        torch.tensor([_ATT["ra_ref_deg"], _ATT["dec_ref_deg"]], dtype=F64),
        atol=1e-11,
        rtol=0,
    )
    ideal = torch.empty(50, 2, dtype=F64).uniform_(
        -70.0, 70.0, generator=torch.Generator().manual_seed(1)
    )
    world = att(ideal)
    assert bool(((world[:, 0] >= 0) & (world[:, 0] < 360)).all())
    torch.testing.assert_close(att.inverse(world), ideal, atol=1e-9, rtol=0)
    tel = att.ideal_to_telescope(ideal)
    torch.testing.assert_close(att.telescope_to_ideal(tel), ideal, atol=1e-11, rtol=0)


@pytest.mark.parametrize("parity", [-1, 1])
def test_attitude_is_exactly_spherical_for_large_offsets(parity: int) -> None:
    # Pole-adjacent attitude and a 3 degree offset: a small-angle shortcut fails
    # here by arcseconds; the exact rotation round-trips to 1e-9 arcsec.
    att = TelescopeAttitude(
        10.0, -20.0, 200.0, 88.0, 75.0, v3_idl_yangle_deg=1.2, v_idl_parity=parity
    )
    ideal = torch.tensor(
        [[3600.0 * 3, 0.0], [0.0, -3600.0 * 3], [-4000.0, 5000.0]], dtype=F64
    )
    world = att(ideal)
    torch.testing.assert_close(att.inverse(world), ideal, atol=1e-9, rtol=0)
    # Separation between two ideal points equals their planar separation (the
    # telescope step is a rotation of angles) only to first order; on the sphere
    # the rotation preserves angular distances exactly.
    tel = att.ideal_to_telescope(ideal)
    arc = math.pi / 180.0 / 3600.0

    def unit(lon, lat):
        return torch.stack(
            (
                torch.cos(lat) * torch.cos(lon),
                torch.cos(lat) * torch.sin(lon),
                torch.sin(lat),
            ),
            -1,
        )

    t_vec = unit(tel[:, 0] * arc, tel[:, 1] * arc)
    w_vec = unit(world[:, 0] * math.pi / 180, world[:, 1] * math.pi / 180)
    torch.testing.assert_close(t_vec @ t_vec.T, w_vec @ w_vec.T, atol=1e-14, rtol=0)


def test_attitude_corrections_move_pointing_and_roll_exactly() -> None:
    att = _synthetic_attitude(learnable=True)

    def at(p):
        return att(p).detach()

    zero = at(torch.zeros(2, dtype=F64))
    with torch.no_grad():
        att.correction.copy_(torch.tensor([5.0, 0.0, 0.0], dtype=F64))
    east = at(torch.zeros(2, dtype=F64))
    # Gnomonic (tangent-plane) offset of 5 arcsec: separation = atan(5 arcsec).
    expected = math.atan(5.0 * math.pi / 180 / 3600) * 180 / math.pi * 3600
    assert abs(float(_sep_arcsec(zero, east).detach()) - expected) < 1e-9
    assert float(east[0]) > float(zero[0])  # moved to increasing RA (east)
    with torch.no_grad():
        att.correction.copy_(torch.tensor([0.0, 5.0, 0.0], dtype=F64))
    north = at(torch.zeros(2, dtype=F64))
    assert abs(float(_sep_arcsec(zero, north).detach()) - expected) < 1e-9
    assert float(north[1]) > float(zero[1])
    # A roll offset leaves the reference point fixed and turns the field by
    # d_roll arcsec: a point 100 arcsec away moves by 100 * d_roll (rad).
    with torch.no_grad():
        att.correction.copy_(torch.tensor([0.0, 0.0, 10.0], dtype=F64))
    rolled_ref = at(torch.zeros(2, dtype=F64))
    torch.testing.assert_close(rolled_ref, zero, atol=1e-11, rtol=0)
    probe = torch.tensor([100.0, 0.0], dtype=F64)
    with torch.no_grad():
        att.correction.zero_()
    base = at(probe)
    with torch.no_grad():
        att.correction[2] = 10.0
    moved = float(_sep_arcsec(base, at(probe)).detach())
    assert moved == pytest.approx(100.0 * 10.0 * math.pi / 180 / 3600, abs=1e-6)


def test_attitude_corrections_default_to_frozen_zero() -> None:
    att = _synthetic_attitude()
    assert not att.correction.requires_grad
    assert float(att.correction.abs().sum()) == 0.0
    assert _synthetic_attitude(learnable=True).correction.requires_grad


def test_attitude_rejects_bad_constants_and_inputs() -> None:
    kw = {"v3_idl_yangle_deg": 0.0, "v_idl_parity": -1}
    with pytest.raises(ValueError, match="dec_ref_deg"):
        TelescopeAttitude(0.0, 0.0, 10.0, 91.0, 0.0, **kw)
    with pytest.raises(ValueError, match="v_idl_parity"):
        TelescopeAttitude(
            0.0, 0.0, 10.0, 0.0, 0.0, v3_idl_yangle_deg=0.0, v_idl_parity=2
        )
    with pytest.raises(ValueError, match="finite"):
        TelescopeAttitude(float("nan"), 0.0, 10.0, 0.0, 0.0, **kw)
    att = _synthetic_attitude()
    with pytest.raises(ValueError, match="last dimension must be 2"):
        att(torch.zeros(3, dtype=F64))
    with pytest.raises(ValueError, match="latitude"):
        att.inverse(torch.tensor([[10.0, 95.0]], dtype=F64))


# --------------------------------------------------------------- composition


@pytest.mark.parametrize("name", ["NRCB1", "NRCBLONG"])
def test_pixel_to_sky_to_pixel_round_trip(name: str) -> None:
    t = _transform(name)
    pts = _lattice(9)
    x, y = t.pixel_to_world(pts[..., 0], pts[..., 1])
    bx, by = t.world_to_pixel(x, y)
    assert float((bx - pts[..., 0]).abs().max()) < 1e-8
    assert float((by - pts[..., 1]).abs().max()) < 1e-8
    # The reference pixel is the attitude reference point.
    f = _FIXTURES[name]
    ra, dec = t.pixel_to_world(
        torch.tensor(f["XSciRef"] - 1.0, dtype=F64),
        torch.tensor(f["YSciRef"] - 1.0, dtype=F64),
    )
    torch.testing.assert_close(
        torch.stack((ra, dec)),
        torch.tensor([_ATT["ra_ref_deg"], _ATT["dec_ref_deg"]], dtype=F64),
        atol=1e-11,
        rtol=0,
    )


def test_sky_jacobian_matches_finite_differences_and_plate_scale() -> None:
    t = _transform()
    p = torch.tensor([[300.0, 1500.0], [1900.0, 100.0]], dtype=F64)
    h = 1e-3
    for point in p:
        auto = torch.autograd.functional.jacobian(t, point)  # (2 sky, 2 pixel)
        for col in range(2):
            step = torch.zeros(2, dtype=F64)
            step[col] = h
            fd = (t(point + step) - t(point - step)) / (2 * h)
            torch.testing.assert_close(auto[:, col], fd, atol=1e-10, rtol=1e-5)
    # Plate scale: 0.0307 arcsec/pixel on the sky.
    sep = _sep_arcsec(t(p), t(p + torch.tensor([1.0, 0.0], dtype=F64)))
    assert float(sep.min()) == pytest.approx(0.0307, abs=5e-4)


def test_gradcheck_coefficients_and_attitude_corrections() -> None:
    poly = _synthetic_poly(learnable=True)
    att = _synthetic_attitude(learnable=True)
    t = TelescopeFrameDetectorToSky(poly, att)
    pts = torch.tensor([[3.5, 10.25], [40.0, 60.5], [31.5, 31.5]], dtype=F64)

    def fn(coeff: torch.Tensor, corr: torch.Tensor, xy: torch.Tensor) -> torch.Tensor:
        params = {"distortion.coefficients": coeff, "attitude.correction": corr}
        return torch.func.functional_call(t, params, (xy,))

    coeff = poly.coefficients.detach().clone().requires_grad_(True)
    corr = torch.tensor([0.3, -0.2, 0.5], dtype=F64, requires_grad=True)
    xy = pts.clone().requires_grad_(True)
    assert torch.autograd.gradcheck(
        fn, (coeff, corr, xy), eps=1e-6, atol=1e-5, rtol=1e-4
    )


def test_gradcheck_real_nircam_coefficients_via_relative_scales() -> None:
    # Perturbing a 1e-18 coefficient by 1e-6 absolute is meaningless; scale instead.
    poly = _poly(learnable=True)
    t = TelescopeFrameDetectorToSky(poly, _attitude(learnable=True))
    base = poly.coefficients.detach().clone()
    pts = torch.tensor([[100.0, 200.0], [1800.0, 1900.0]], dtype=F64)

    def fn(scale: torch.Tensor) -> torch.Tensor:
        params = {"distortion.coefficients": base * (1.0 + scale)}
        return torch.func.functional_call(t, params, (pts,))

    scale = torch.zeros_like(base, requires_grad=True)
    assert torch.autograd.gradcheck(fn, (scale,), eps=1e-6, atol=1e-6, rtol=1e-4)


def test_gradcheck_world_to_pixel_is_differentiable() -> None:
    poly = _synthetic_poly(learnable=True)
    att = _synthetic_attitude(learnable=True)
    t = TelescopeFrameDetectorToSky(poly, att)
    ra0, dec0 = t.pixel_to_world(
        torch.tensor([20.0, 45.0], dtype=F64), torch.tensor([12.0, 50.0], dtype=F64)
    )

    class Inverse(torch.nn.Module):
        def __init__(self, inner: TelescopeFrameDetectorToSky) -> None:
            super().__init__()
            self.inner = inner

        def forward(self, ra, dec):
            return torch.stack(self.inner.world_to_pixel(ra, dec), dim=-1)

    wrapper = Inverse(t)
    base = poly.coefficients.detach().clone()

    def fn(scale, corr, ra, dec):
        # Relative coefficient scales: absolute perturbations of 1e-8 coefficients
        # would fold the polynomial. A roll correction moves pixels by ~1e-4 px per
        # arcsec here, so a larger step keeps the finite differences above rounding.
        params = {
            "inner.distortion.coefficients": base * (1.0 + scale),
            "inner.attitude.correction": corr,
        }
        return torch.func.functional_call(wrapper, params, (ra, dec))

    args = (
        torch.zeros_like(base, requires_grad=True),
        torch.tensor([0.1, 0.2, -0.3], dtype=F64, requires_grad=True),
        ra0.detach().clone().requires_grad_(True),
        dec0.detach().clone().requires_grad_(True),
    )
    assert torch.autograd.gradcheck(fn, args, eps=1e-5, atol=1e-4, rtol=1e-4)


def test_attitude_corrections_are_learnable_end_to_end() -> None:
    # Recover a known pointing and roll correction from sky positions over the detector.
    t = TelescopeFrameDetectorToSky(_poly(), _attitude(learnable=True))
    truth = torch.tensor([2.0, -1.5, 50.0], dtype=F64)
    pts = _lattice(5).reshape(-1, 2)
    expected = TelescopeFrameDetectorToSky(_poly(), _attitude())
    with torch.no_grad():
        expected.attitude.correction.copy_(truth)
        want = expected(pts)
    opt = torch.optim.LBFGS(
        [t.attitude.correction],
        lr=1.0,
        max_iter=100,
        tolerance_grad=1e-14,
        tolerance_change=1e-16,
        line_search_fn="strong_wolfe",
    )

    def closure():
        opt.zero_grad()
        loss = (((t(pts) - want) * 3.6e6) ** 2).sum()  # milliarcsec squared
        loss.backward()
        return loss

    opt.step(closure)
    torch.testing.assert_close(t.attitude.correction.detach(), truth, atol=1e-3, rtol=0)
    assert not t.distortion.coefficients.requires_grad  # frozen by default


def test_residual_composes_projects_out_affine_modes_and_inverts() -> None:
    height = width = 5
    gen = torch.Generator().manual_seed(3)
    control = 0.02 * torch.randn(height, width, 2, dtype=F64, generator=gen)
    spacing = (2047.0 / (width - 1), 2047.0 / (height - 1))
    res = AffineFreeResidualGrid2D(control, spacing=spacing, mode="bicubic")
    plain = _transform()
    t = TelescopeFrameDetectorToSky(_poly(), _attitude(), residual=res)
    pts = _lattice(6)
    ideal_shift = t.pixel_to_ideal(pts) - plain.pixel_to_ideal(pts)
    torch.testing.assert_close(ideal_shift, res(pts), atol=1e-12, rtol=0)
    assert float(ideal_shift.abs().max().detach()) > 1e-3
    # Constant and affine control modes are projected out: adding them changes nothing.
    yy, xx = torch.meshgrid(
        torch.linspace(-1, 1, 5, dtype=F64),
        torch.linspace(-1, 1, 5, dtype=F64),
        indexing="ij",
    )
    before = t(pts)
    with torch.no_grad():
        res.control += torch.stack((0.5 + 0.2 * xx - 0.1 * yy, -0.3 + 0.4 * yy), dim=-1)
    torch.testing.assert_close(t(pts), before, atol=1e-9, rtol=0)
    x, y = t.pixel_to_world(pts[..., 0], pts[..., 1])
    bx, by = (v.detach() for v in t.world_to_pixel(x.detach(), y.detach()))
    assert float((bx - pts[..., 0]).abs().max()) < 1e-7
    assert float((by - pts[..., 1]).abs().max()) < 1e-7
    # Gradients reach the residual control grid.
    t(pts).sum().backward()
    assert res.control.grad is not None and float(res.control.grad.abs().sum()) > 0
    # Outside the residual grid the forward model raises, as the neighbour class does.
    with pytest.raises(ValueError, match="outside the regular-grid domain"):
        t(torch.tensor([[2100.0, 10.0]], dtype=F64))


def test_world_to_pixel_failure_modes() -> None:
    t = _transform()
    with pytest.raises(ValueError, match="did not converge"):
        t.world_to_pixel(
            torch.tensor([110.87 + 5.0], dtype=F64), torch.tensor([-73.45], dtype=F64)
        )
    x, y = t.world_to_pixel(
        torch.tensor([110.87 + 5.0, 110.8745], dtype=F64),
        torch.tensor([-73.45, -73.454], dtype=F64),
        fail_closed=False,
    )
    assert bool(torch.isnan(x[0]) and torch.isnan(y[0]))
    assert bool(torch.isfinite(x[1]) and torch.isfinite(y[1]))
    with pytest.raises(ValueError, match="floating-point"):
        t.world_to_pixel(torch.tensor([1, 2]), torch.tensor([1.0, 2.0]))
    with pytest.raises(ValueError, match="max_iter"):
        t.world_to_pixel(
            torch.tensor([110.87], dtype=F64),
            torch.tensor([-73.45], dtype=F64),
            max_iter=0,
        )
    with pytest.raises(ValueError, match="floating-point"):
        t.pixel_to_world(torch.tensor([1, 2]), torch.tensor([1, 2]))


def test_constructor_types_are_checked() -> None:
    with pytest.raises(TypeError, match="PolynomialDetectorDistortion"):
        TelescopeFrameDetectorToSky(object(), _attitude())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="TelescopeAttitude"):
        TelescopeFrameDetectorToSky(_poly(), object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="AffineFreeResidualGrid2D"):
        TelescopeFrameDetectorToSky(_poly(), _attitude(), residual=object())  # type: ignore[arg-type]


# ------------------------------------------------------------------- headers

# Keywords of a real NIRCam count-rate SCI header (DEC_REF as in the exposure).
_HEADER = {
    "NAXIS1": 2048,
    "NAXIS2": 2048,
    "CRPIX1": 1024.5,
    "CRPIX2": 1024.5,
    "CRVAL1": 110.8745219187291,
    "CRVAL2": -73.45405756551722,
    "V2_REF": -122.427986,
    "V3_REF": -457.706754,
    "VPARITY": -1,
    "V3I_YANG": 0.34481904,
    "RA_REF": 110.8745219187291,
    "DEC_REF": -73.45405756551722,
    "ROLL_REF": 144.93904872291884,
}


def _from_header(header: dict, **kwargs) -> TelescopeFrameDetectorToSky:
    f = _NRCB1
    return TelescopeFrameDetectorToSky.from_header(
        header,
        f["coeffs_x"],
        f["coeffs_y"],
        x_sci_ref=f["XSciRef"],
        y_sci_ref=f["YSciRef"],
        **kwargs,
    )


def test_from_header_builds_the_same_chain() -> None:
    t = _from_header(_HEADER)
    assert t.distortion.detector_shape == (2048, 2048)
    ra, dec = t.pixel_to_world(
        torch.tensor(1023.5, dtype=F64), torch.tensor(1023.5, dtype=F64)
    )
    assert float(ra) == pytest.approx(_HEADER["RA_REF"], abs=1e-11)
    assert float(dec) == pytest.approx(_HEADER["DEC_REF"], abs=1e-11)
    lower = _from_header({k.lower(): v for k, v in _HEADER.items()})
    torch.testing.assert_close(lower(_lattice(3)), t(_lattice(3)))
    learn = _from_header(_HEADER, learnable=True)
    assert (
        learn.distortion.coefficients.requires_grad
        and learn.attitude.correction.requires_grad
    )


def test_from_header_names_every_missing_keyword() -> None:
    header = {
        k: v for k, v in _HEADER.items() if k not in ("V2_REF", "ROLL_REF", "VPARITY")
    }
    with pytest.raises(ValueError) as info:
        _from_header(header)
    message = str(info.value)
    for key in ("V2_REF", "ROLL_REF", "VPARITY"):
        assert key in message
    with pytest.raises(ValueError, match="DEC_REF"):
        _from_header({k: v for k, v in _HEADER.items() if k != "DEC_REF"})


def test_from_header_rejects_bad_values() -> None:
    with pytest.raises(ValueError, match="RA_REF"):
        _from_header({**_HEADER, "RA_REF": "not a number"})
    with pytest.raises(ValueError, match="V3I_YANG"):
        _from_header({**_HEADER, "V3I_YANG": float("nan")})
    with pytest.raises(ValueError, match="VPARITY"):
        _from_header({**_HEADER, "VPARITY": 0})
    no_axes = {k: v for k, v in _HEADER.items() if not k.startswith("NAXIS")}
    with pytest.raises(ValueError, match="NAXIS1/NAXIS2"):
        _from_header(no_axes)
    assert _from_header(
        no_axes, detector_shape=(2048, 2048)
    ).distortion.detector_shape == (2048, 2048)


# --------------------------------------------- Protocol, crop, dtype and device


def _observation(t: TelescopeFrameDetectorToSky, offset: int = 900) -> FieldObservation:
    return FieldObservation(
        image=torch.zeros(128, 128, dtype=F64),
        header={"NAXIS1": 128, "NAXIS2": 128},
        detector_to_sky=t,
        detector_x0_pix=float(offset),
        detector_y0_pix=float(offset),
    )


def test_field_observation_accepts_the_transform_and_crops_through_world_to_pixel() -> (
    None
):
    ref = _observation(_from_header(_HEADER))
    dithered = _observation(
        _from_header({**_HEADER, "RA_REF": _HEADER["RA_REF"] + 0.0005})
    )
    x = torch.tensor([0.0, 127.0], dtype=F64)
    ra, dec = observation_pixel_to_world(ref, x, x)
    full = ref.detector_to_sky.pixel_to_world(x + 900.0, x + 900.0)
    torch.testing.assert_close(torch.stack((ra, dec)), torch.stack(full))
    cropped = crop_field_observation_to_shared_sky_patch(
        dithered,
        reference_observation=ref,
        x0=40,
        y0=40,
        patch_size=32,
        psf_halo=(0, 0),
    )
    assert cropped is not None
    # Expected box: the reference patch corners and centre mapped into the other exposure.
    rx = torch.tensor([40, 71, 40, 71, 55.5], dtype=F64)
    ry = torch.tensor([40, 40, 71, 71, 55.5], dtype=F64)
    corner_ra, corner_dec = observation_pixel_to_world(ref, rx, ry)
    px, py = dithered.detector_to_sky.world_to_pixel(corner_ra, corner_dec)
    px, py = px - 900.0, py - 900.0
    assert 0 <= px.min() and px.max() < 128  # the test is meaningful: inside the image
    x0 = int(torch.floor(px.min()))
    x1 = int(torch.ceil(px.max())) + 1
    y0 = int(torch.floor(py.min()))
    y1 = int(torch.ceil(py.max())) + 1
    assert tuple(cropped.image.shape) == (y1 - y0, x1 - x0)
    assert (
        cropped.detector_x0_pix == 900.0 + x0 and cropped.detector_y0_pix == 900.0 + y0
    )


def test_dtype_is_preserved_and_parameters_stay_float64() -> None:
    t = _from_header(_HEADER)
    p64 = _lattice(3)
    ra32, dec32 = t.pixel_to_world(p64[..., 0].float(), p64[..., 1].float())
    ra64, dec64 = t.pixel_to_world(p64[..., 0], p64[..., 1])
    assert ra32.dtype == torch.float32 and dec32.dtype == torch.float32
    # float32 output is limited by its own precision (~8e-6 deg at RA 110), not by the geometry.
    assert float((ra32.double() - ra64).abs().max()) < 2e-5
    copy_t = copy.deepcopy(t).to(dtype=torch.float32)
    assert all(p.dtype == F64 for p in copy_t.parameters())
    assert all(b.dtype in (F64, torch.long) for b in copy_t.buffers())
    obs = _observation(t).to(dtype=torch.float32)
    assert obs.detector_to_sky.attitude.constants.dtype == F64
    torch.testing.assert_close(
        torch.stack(obs.detector_to_sky.pixel_to_world(p64[..., 0], p64[..., 1])),
        torch.stack((ra64, dec64)),
        atol=1e-12,
        rtol=0,
    )


def test_batch_shapes_and_broadcasting() -> None:
    t = _from_header(_HEADER)
    x = torch.linspace(10.0, 2000.0, 6, dtype=F64).reshape(2, 3)
    ra, dec = t.pixel_to_world(x, torch.tensor(500.0, dtype=F64))
    assert ra.shape == dec.shape == (2, 3)
    bx, by = t.world_to_pixel(ra, dec)
    torch.testing.assert_close(bx, x, atol=1e-8, rtol=0)
    torch.testing.assert_close(by, torch.full_like(x, 500.0), atol=1e-8, rtol=0)


# ----------------------------------------------------- optional pysiaf oracle


def test_against_pysiaf_oracle_optional() -> None:
    """OPTIONAL: compares with pysiaf when it is installed (skipped otherwise)."""
    pysiaf = pytest.importorskip("pysiaf")
    import numpy as np
    from pysiaf.utils import rotations

    siaf = pysiaf.Siaf("NIRCam")
    g = np.linspace(0.0, 2047.0, 9)
    gx, gy = np.meshgrid(g, g)
    for name, key in (("NRCB1", "NRCB1_FULL"), ("NRCBLONG", "NRCB5_FULL")):
        ap = siaf[key]
        t = _transform(name)
        # (1) science to ideal, to 1e-9 arcsec (SIAF pixels are one-based).
        ix, iy = ap.sci_to_idl(gx + 1.0, gy + 1.0)
        ours = t.distortion(torch.tensor(np.stack((gx, gy), -1)))
        assert np.abs(ours[..., 0].numpy() - ix).max() < 1e-9
        assert np.abs(ours[..., 1].numpy() - iy).max() < 1e-9
        # (2) full chain: attitude + pointing + idl_to_tel, to 1e-7 arcsec.
        v2, v3 = ap.idl_to_tel(ix, iy)
        att = rotations.attitude(
            ap.V2Ref,
            ap.V3Ref,
            _ATT["ra_ref_deg"],
            _ATT["dec_ref_deg"],
            _ATT["roll_ref_deg"],
        )
        ra, dec = rotations.pointing(att, v2.ravel(), v3.ravel())
        mine = t(torch.tensor(np.stack((gx, gy), -1))).reshape(-1, 2).numpy()
        theirs = torch.tensor(np.stack((ra, dec), -1))
        assert float(_sep_arcsec(torch.tensor(mine), theirs).max()) < 1e-7
