import numpy as np
import pytest
import torch
from astropy.wcs import WCS as AstropyWCS

from torchsky.wcs.core import WCS
from torchsky.wcs.tpv import TPV


def _tpv_header() -> dict:
    header = {}
    header["NAXIS"] = 2
    header["NAXIS1"] = 1024
    header["NAXIS2"] = 1024
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["CRPIX1"] = 512.0
    header["CRPIX2"] = 512.0
    header["CRVAL1"] = 200.0
    header["CRVAL2"] = -20.0
    header["CD1_1"] = -2.8e-4
    header["CD1_2"] = 1.2e-7
    header["CD2_1"] = -1.1e-7
    header["CD2_2"] = 2.8e-4

    # Keep a mix of linear and non-linear terms; this remains stable for forward parity.
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = 0.0
    header["PV1_4"] = 2.0e-4
    header["PV1_5"] = -3.0e-4
    header["PV1_6"] = 1.5e-4
    header["PV1_7"] = 2.0e-6
    header["PV1_39"] = 2.0e-11

    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.0
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4
    header["PV2_5"] = 2.5e-4
    header["PV2_6"] = -2.0e-4
    header["PV2_7"] = -1.5e-6
    header["PV2_39"] = -1.0e-11

    return header


def _tpv_affine_linear_header() -> dict:
    header = _tpv_header()
    # Pure affine TPV (non-identity, with offsets and cross terms)
    for k in list(header.keys()):
        if k.startswith("PV1_") or k.startswith("PV2_"):
            del header[k]
    header["PV1_0"] = 0.05
    header["PV1_1"] = 1.12
    header["PV1_2"] = -0.08
    # Axis-2 TPV uses swapped arguments: PV2_2 contributes u, PV2_1 contributes v
    header["PV2_0"] = -0.03
    header["PV2_1"] = 0.04
    header["PV2_2"] = 0.91
    return header


def _well_conditioned_tpv_header() -> dict:
    header = _tpv_header()
    header["PV2_1"] = 1.0
    header["PV2_2"] = 2.0e-4
    return header


def _sample_pixels() -> tuple[np.ndarray, np.ndarray]:
    x = np.linspace(430.0, 590.0, 9, dtype=np.float64)
    y = np.linspace(420.0, 600.0, 9, dtype=np.float64)
    xx, yy = np.meshgrid(x, y, indexing="xy")
    return xx.ravel(), yy.ravel()


def test_tpv_analytic_jacobian_matches_finite_difference() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None

    # Probe a range of tangent-plane coordinates (degrees), including near-origin
    # values where TPV radial terms are most numerically delicate.
    u = torch.tensor(
        [0.0, 1e-6, -2e-6, 1e-3, -2e-3, 0.1, -0.2],
        dtype=torch.float64,
    )
    v = torch.tensor(
        [0.0, -3e-6, 4e-6, -2e-3, 3e-3, -0.15, 0.25],
        dtype=torch.float64,
    )

    xi, eta, j11, j12, j21, j22 = twcs.tpv._distort_and_jacobian(u, v)
    eps = 1e-7
    xi_dx, eta_dx = twcs.tpv.distort(u + eps, v)
    xi_dy, eta_dy = twcs.tpv.distort(u, v + eps)

    np.testing.assert_allclose(
        j11.cpu().numpy(), ((xi_dx - xi) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j12.cpu().numpy(), ((xi_dy - xi) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j21.cpu().numpy(), ((eta_dx - eta) / eps).cpu().numpy(), atol=5e-8
    )
    np.testing.assert_allclose(
        j22.cpu().numpy(), ((eta_dy - eta) / eps).cpu().numpy(), atol=5e-8
    )


def test_tpv_high_order_terms_change_solution() -> None:
    base_header = _tpv_header()
    reduced_header = _tpv_header()

    # Remove high-order terms from comparison model.
    for key in ["PV1_7", "PV2_7", "PV1_39", "PV2_39"]:
        reduced_header[key] = 0.0

    full = WCS(dict(base_header))
    reduced = WCS(dict(reduced_header))

    x = torch.tensor([540.0, 560.0], dtype=torch.float64)
    y = torch.tensor([520.0, 545.0], dtype=torch.float64)

    ra_full, dec_full = full.pixel_to_world(x, y, origin=0)
    ra_reduced, dec_reduced = reduced.pixel_to_world(x, y, origin=0)

    assert torch.max(torch.abs(ra_full - ra_reduced)) > 0.0
    assert torch.max(torch.abs(dec_full - dec_reduced)) > 0.0


def test_tpv_inverse_trace_records_iterations() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None

    x, y = _sample_pixels()
    ra_t, dec_t = twcs.pixel_to_world(
        torch.from_numpy(x), torch.from_numpy(y), origin=0
    )

    twcs.tpv.set_invert_trace(True)
    x_rt, y_rt = twcs.world_to_pixel(ra_t, dec_t, origin=0)
    trace = twcs.tpv.get_last_invert_trace()
    twcs.tpv.set_invert_trace(False)
    x_ref, y_ref = twcs.world_to_pixel(ra_t, dec_t, origin=0)

    assert trace is not None
    assert int(trace.get("n_points", 0)) == x.size
    assert 0 <= int(trace.get("converged", -1)) <= x.size
    assert float(trace.get("mean_point_iterations", 0.0)) >= 0.0
    assert (trace.get("active_counts") or trace.get("active_counts_sum")) is not None
    np.testing.assert_allclose(x_rt.cpu().numpy(), x_ref.cpu().numpy(), atol=1.0)
    np.testing.assert_allclose(y_rt.cpu().numpy(), y_ref.cpu().numpy(), atol=1.0)


def test_tpv_affine_seed_exact_for_affine_only_tpv() -> None:
    twcs = WCS(dict(_tpv_affine_linear_header()))
    assert twcs.tpv is not None
    assert twcs.tpv._affine_only
    twcs.tpv.set_cpp_invert_max_points(0)

    x, y = _sample_pixels()
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    ra_t, dec_t = twcs.pixel_to_world(x_t, y_t, origin=0)

    twcs.tpv.set_invert_trace(True)
    x_rt, y_rt = twcs.world_to_pixel(ra_t, dec_t, origin=0)
    trace = twcs.tpv.get_last_invert_trace()
    twcs.tpv.set_invert_trace(False)

    np.testing.assert_allclose(x_rt.cpu().numpy(), x, atol=2e-5)
    np.testing.assert_allclose(y_rt.cpu().numpy(), y, atol=2e-5)
    assert trace is not None
    # Pure affine TPV should solve in a single Newton evaluation from the affine seed.
    assert int(trace.get("iterations", trace.get("iterations_max", 0))) <= 1
    assert int(trace.get("final_active", 1)) == 0


def test_tpv_nonlinear_header_is_not_affine_only() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None
    assert not twcs.tpv._affine_only


def test_tpv_skips_cpp_direct_tan_projection_shortcuts() -> None:
    header = dict(_tpv_header())
    ref = WCS(dict(header)).to("cpu")
    twcs = WCS(header).to("cpu")
    assert twcs.tpv is not None
    assert not twcs._has_cpp_direct_p2w
    assert not twcs._has_cpp_direct_w2p

    def _unexpected(*args, **kwargs):
        raise AssertionError("TPV should not use direct TAN C++ projection shortcuts")

    x, y = _sample_pixels()
    x_t = torch.from_numpy(x)
    y_t = torch.from_numpy(y)
    ra_ref, dec_ref = ref.pixel_to_world(x_t, y_t, origin=0)
    x_ref, y_ref = ref.world_to_pixel(ra_ref, dec_ref, origin=0)

    old_p2w = twcs._cpp_zenithal_project
    old_w2p = twcs._cpp_zenithal_deproject
    try:
        twcs._cpp_zenithal_project = _unexpected
        twcs._cpp_zenithal_deproject = _unexpected
        ra_t, dec_t = twcs.pixel_to_world(x_t, y_t, origin=0)
        x_rt, y_rt = twcs.world_to_pixel(ra_t, dec_t, origin=0)
    finally:
        twcs._cpp_zenithal_project = old_p2w
        twcs._cpp_zenithal_deproject = old_w2p

    np.testing.assert_allclose(ra_t.cpu().numpy(), ra_ref.cpu().numpy(), atol=1e-12)
    np.testing.assert_allclose(dec_t.cpu().numpy(), dec_ref.cpu().numpy(), atol=1e-12)
    np.testing.assert_allclose(x_rt.cpu().numpy(), x_ref.cpu().numpy(), atol=1e-12)
    np.testing.assert_allclose(y_rt.cpu().numpy(), y_ref.cpu().numpy(), atol=1e-12)




def test_tpv_identity_pv2_1_mixed_fov_roundtrip() -> None:
    header = _tpv_header()
    header["PV2_1"] = 1.0
    header["PV2_2"] = 2.0e-4
    twcs = WCS(dict(header))
    rng = np.random.default_rng(1)
    x = rng.uniform(64.0, 960.0, size=200)
    y = rng.uniform(64.0, 960.0, size=200)
    ra, dec = twcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)
    xb, yb = twcs.world_to_pixel(ra, dec, origin=0)
    err = np.hypot(xb.numpy() - x, yb.numpy() - y)
    assert float(np.nanmax(err)) < 1.0e-5


def test_fused_wcs_enabled_by_default_for_tpv() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs._enable_fused_wcs
    assert twcs._can_use_fused_w2p_static


def test_fused_wcs_can_be_disabled_with_env(monkeypatch) -> None:
    monkeypatch.setenv("TORCHSKY_WCS_FUSED", "0")
    twcs = WCS(dict(_tpv_header()))
    assert not twcs._enable_fused_wcs
    assert not twcs._can_use_fused_w2p_static


def test_tpv_cpp_invert_handles_2d_inputs_when_fused_disabled() -> None:
    header = dict(_tpv_header())
    twcs_cpp = WCS(dict(header))
    twcs_py = WCS(dict(header))
    assert twcs_cpp.tpv is not None
    assert twcs_py.tpv is not None

    twcs_cpp._disable_fused_world_to_pixel = True
    twcs_py._disable_fused_world_to_pixel = True
    twcs_py.tpv._can_use_cpp_invert = lambda *args, **kwargs: False

    x = torch.linspace(430.0, 590.0, 7, dtype=torch.float64)
    y = torch.linspace(420.0, 600.0, 5, dtype=torch.float64)
    yy, xx = torch.meshgrid(y, x, indexing="ij")

    ra_t, dec_t = twcs_cpp.pixel_to_world(xx, yy, origin=0)
    x_cpp, y_cpp = twcs_cpp.world_to_pixel(ra_t, dec_t, origin=0)
    x_py, y_py = twcs_py.world_to_pixel(ra_t, dec_t, origin=0)

    assert x_cpp.shape == xx.shape
    assert y_cpp.shape == yy.shape
    np.testing.assert_allclose(x_cpp.cpu().numpy(), x_py.cpu().numpy(), atol=5e-4)
    np.testing.assert_allclose(y_cpp.cpu().numpy(), y_py.cpu().numpy(), atol=5e-4)


def test_tpv_invert_with_gradients_uses_differentiable_path() -> None:
    twcs = WCS(dict(_tpv_header()))
    assert twcs.tpv is not None
    xi = torch.tensor([0.01, -0.02], dtype=torch.float64, requires_grad=True)
    eta = torch.tensor([-0.015, 0.025], dtype=torch.float64, requires_grad=True)

    u, v = twcs.tpv.invert(xi, eta)
    (u + 0.5 * v).sum().backward()

    assert xi.grad is not None and torch.isfinite(xi.grad).all()
    assert eta.grad is not None and torch.isfinite(eta.grad).all()


def _force_inverse_path(wcs: WCS, path: str) -> None:
    wcs._disable_fused_world_to_pixel = path != "fused"
    if path == "python":
        assert wcs.tpv is not None
        wcs.tpv._can_use_cpp_invert = lambda *args, **kwargs: False


def test_empty_tpv_is_identity() -> None:
    tpv = TPV({})
    u = torch.tensor([0.1, -0.2, torch.nan], dtype=torch.float64)
    v = torch.tensor([-0.3, 0.4, 0.5], dtype=torch.float64)
    expected_u = torch.tensor([0.1, -0.2, torch.nan], dtype=torch.float64)
    expected_v = torch.tensor([-0.3, 0.4, torch.nan], dtype=torch.float64)

    du, dv = tpv.distort(u, v)
    iu, iv = tpv.invert(u, v)
    torch.testing.assert_close(du, expected_u, equal_nan=True)
    torch.testing.assert_close(dv, expected_v, equal_nan=True)
    torch.testing.assert_close(iu, expected_u, equal_nan=True)
    torch.testing.assert_close(iv, expected_v, equal_nan=True)

@pytest.mark.parametrize("path", ["fused", "standalone", "python"])
def test_rank_deficient_affine_returns_nan_in_forced_inverse_paths(path: str) -> None:
    header = _tpv_header()
    for key in list(header):
        if key.startswith(("PV1_", "PV2_")):
            del header[key]
    header.update({"PV1_1": 1.0, "PV2_2": 1.0})
    wcs = WCS(header).to("cpu")
    _force_inverse_path(wcs, path)

    x, y = wcs.world_to_pixel(
        torch.tensor([200.0], dtype=torch.float64),
        torch.tensor([-20.0], dtype=torch.float64),
        origin=0,
    )
    assert torch.isnan(x).all()
    assert torch.isnan(y).all()


def test_folded_polynomial_at_singular_fold_returns_nan() -> None:
    tpv = TPV({"PV1_4": 1.0, "PV2_1": 1.0})
    xi = torch.tensor([0.0, 0.25], dtype=torch.float64)
    eta = torch.tensor([0.2, 0.2], dtype=torch.float64)

    u, v = tpv.invert(xi, eta)

    assert torch.isnan(u[0]) and torch.isnan(v[0])
    assert torch.isfinite(u[1]) and torch.isfinite(v[1])
    xi_rt, eta_rt = tpv.distort(u[1:], v[1:])
    assert torch.hypot(xi_rt - xi[1:], eta_rt - eta[1:]).item() <= 1e-11


@pytest.mark.parametrize("bad", [torch.nan, torch.inf, -torch.inf])
def test_tpv_rejects_nonfinite_coefficients(bad: float) -> None:
    with pytest.raises(ValueError, match="must be finite"):
        TPV({"PV1_1": bad})


@pytest.mark.parametrize("use_cpp", [False, True])
def test_tpv_nonfinite_inverse_inputs_fail_per_point(use_cpp: bool) -> None:
    tpv = TPV(_well_conditioned_tpv_header())
    if not use_cpp:
        tpv._can_use_cpp_invert = lambda *args, **kwargs: False
    xi = torch.tensor([0.01, torch.nan, 0.02], dtype=torch.float64)
    eta = torch.tensor([-0.02, 0.03, torch.inf], dtype=torch.float64)

    u, v = tpv.invert(xi, eta)

    assert torch.isfinite(u[0]) and torch.isfinite(v[0])
    assert torch.isnan(u[1:]).all()
    assert torch.isnan(v[1:]).all()


@pytest.mark.parametrize("max_iter", [0, -1])
def test_tpv_invert_rejects_invalid_max_iter(max_iter: int) -> None:
    tpv = TPV({})
    values = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="positive integer"):
        tpv.invert(values, values, max_iter=max_iter)


@pytest.mark.parametrize("tol", [0.0, -1.0, float("nan"), float("inf")])
def test_tpv_invert_rejects_invalid_tolerance(tol: float) -> None:
    tpv = TPV({})
    values = torch.zeros(1, dtype=torch.float64)
    with pytest.raises(ValueError, match="finite and positive"):
        tpv.invert(values, values, tol=tol)


@pytest.mark.parametrize("use_cpp", [False, True])
def test_every_finite_tpv_inverse_satisfies_requested_residual(use_cpp: bool) -> None:
    tpv = TPV(_well_conditioned_tpv_header())
    if not use_cpp:
        tpv._can_use_cpp_invert = lambda *args, **kwargs: False
    u = torch.linspace(-0.2, 0.2, 17, dtype=torch.float64)
    v = torch.linspace(0.15, -0.15, 17, dtype=torch.float64)
    xi, eta = tpv.distort(u, v)
    u_inv, v_inv = tpv.invert(xi, eta, max_iter=20, tol=1e-12)
    finite = torch.isfinite(u_inv) & torch.isfinite(v_inv)
    assert finite.all()
    xi_rt, eta_rt = tpv.distort(u_inv, v_inv)
    assert torch.all(torch.hypot(xi_rt - xi, eta_rt - eta) <= 1e-12)


@pytest.mark.parametrize("path", ["fused", "standalone", "python"])
def test_valid_tpv_fixture_matches_astropy_in_forced_paths(path: str) -> None:
    header = _well_conditioned_tpv_header()
    oracle = AstropyWCS(header)
    wcs = WCS(header).to("cpu")
    _force_inverse_path(wcs, path)
    x, y = _sample_pixels()
    pixels = np.column_stack((x, y))
    expected_world = oracle.all_pix2world(pixels, 0)
    ra, dec = wcs.pixel_to_world(torch.from_numpy(x), torch.from_numpy(y), origin=0)
    actual_world = np.column_stack((ra.numpy(), dec.numpy()))
    delta_ra = np.remainder(actual_world[:, 0] - expected_world[:, 0] + 180.0, 360.0) - 180.0
    np.testing.assert_allclose(delta_ra, 0.0, atol=2e-9, rtol=0.0)
    np.testing.assert_allclose(actual_world[:, 1], expected_world[:, 1], atol=2e-9, rtol=0.0)

    px, py = wcs.world_to_pixel(ra, dec, origin=0)
    np.testing.assert_allclose(px.numpy(), x, atol=2e-5, rtol=0.0)
    np.testing.assert_allclose(py.numpy(), y, atol=2e-5, rtol=0.0)


def test_tpv_inverse_gradcheck_away_from_singularities() -> None:
    tpv = TPV(_well_conditioned_tpv_header())
    target = torch.tensor(
        [[0.02, -0.015], [-0.03, 0.025]], dtype=torch.float64, requires_grad=True
    )

    def inverse(values: torch.Tensor) -> torch.Tensor:
        u, v = tpv.invert(values[:, 0], values[:, 1], max_iter=20, tol=1e-12)
        return torch.stack((u, v), dim=-1)

    assert torch.autograd.gradcheck(inverse, (target,), atol=1e-6, rtol=1e-5)
