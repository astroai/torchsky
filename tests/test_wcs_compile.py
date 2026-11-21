"""``torch.compile(fullgraph=True)`` traces WCS pixel<->world as one graph.

Validation and early exits that need a host sync (``.item()``, ``bool(...)``,
data-dependent row gathers) break Dynamo graphs. The traced forms (fixed-count
masked Newton sweeps for SIP and TPV, host-side projection constants) must
trace without a break and agree with the eager paths.
"""

from __future__ import annotations

import warnings

import pytest
import torch
import torch._dynamo

from torchsky.wcs import WCS

requires_mps = pytest.mark.skipif(
    not torch.backends.mps.is_available(), reason="MPS not available"
)


def _header(code: str, **extra: float) -> dict[str, float | int | str]:
    header: dict[str, float | int | str] = {
        "NAXIS": 2,
        "NAXIS1": 2048,
        "NAXIS2": 2048,
        "CTYPE1": f"RA---{code}",
        "CTYPE2": f"DEC--{code}",
        "CRPIX1": 1024.5,
        "CRPIX2": 1024.5,
        "CRVAL1": 150.0,
        "CRVAL2": 30.0,
        "CD1_1": -2.0e-4,
        "CD1_2": 1.0e-5,
        "CD2_1": 1.5e-5,
        "CD2_2": 2.0e-4,
    }
    header.update(extra)
    return header


_SIP_QUADRATIC = {
    "A_ORDER": 2,
    "B_ORDER": 2,
    "A_2_0": 1e-6,
    "A_1_1": -2e-6,
    "A_0_2": 3e-6,
    "B_2_0": 2e-6,
    "B_1_1": 1e-6,
    "B_0_2": -1e-6,
}
_SIP_CUBIC_WITH_INVERSE = {
    "A_ORDER": 3,
    "B_ORDER": 3,
    "A_2_0": 2.1e-6,
    "A_0_2": 7e-7,
    "A_3_0": 1e-9,
    "B_1_1": 1.8e-6,
    "B_0_3": -1e-9,
    "AP_ORDER": 2,
    "BP_ORDER": 2,
    "AP_2_0": -2.1e-6,
    "AP_0_2": -7e-7,
    "BP_1_1": -1.8e-6,
}
_TPV = {
    "PV1_1": 1.0,
    "PV1_4": 2.0e-4,
    "PV1_5": -3.0e-4,
    "PV1_6": 1.5e-4,
    "PV1_7": 2.0e-2,
    "PV1_11": 1.0e-3,
    "PV2_1": 1.0,
    "PV2_4": 1.0e-4,
    "PV2_5": 2.0e-4,
    "PV2_6": -1.0e-4,
    "PV2_7": 1.0e-2,
    "PV2_11": -1.0e-3,
}

HEADERS = {
    "TAN": _header("TAN"),
    "SIN": _header("SIN"),
    "SIN-slant": _header("SIN", PV2_1=0.1, PV2_2=-0.05),
    "ZEA": _header("ZEA"),
    "ARC": _header("ARC"),
    "STG": _header("STG"),
    "AZP": _header("AZP", PV2_1=2.0, PV2_2=30.0),
    "SZP": _header("SZP", PV2_1=2.0, PV2_2=180.0, PV2_3=60.0),
    "AIR": _header("AIR", PV2_1=45.0),
    "ZPN": _header("ZPN", PV2_0=0.0, PV2_1=1.0, PV2_3=0.05),
    "CAR": _header("CAR"),
    "CAR-equator": _header("CAR", CRVAL2=0.0),
    "CEA": _header("CEA", PV2_1=0.7),
    "MER": _header("MER"),
    "CYP": _header("CYP", PV2_1=1.0, PV2_2=0.7),
    "SFL": _header("SFL"),
    "PAR": _header("PAR"),
    "MOL": _header("MOL"),
    "AIT": _header("AIT"),
    "COD": _header("COD", PV2_1=45.0, PV2_2=10.0),
    "COE": _header("COE", PV2_1=45.0, PV2_2=10.0),
    "COO": _header("COO", PV2_1=45.0, PV2_2=10.0),
    "COP": _header("COP", PV2_1=45.0, PV2_2=10.0),
    "BON": _header("BON", PV2_1=45.0),
    "PCO": _header("PCO"),
    "TSC": _header("TSC"),
    "CSC": _header("CSC"),
    "QSC": _header("QSC"),
    "HPX": _header("HPX"),
    "XPH": _header("XPH"),
    "TAN-SIP-quadratic": _header("TAN-SIP", **_SIP_QUADRATIC),
    "TAN-SIP-cubic-APBP": _header("TAN-SIP", **_SIP_CUBIC_WITH_INVERSE),
    "TPV": _header("TPV", **_TPV),
}


@pytest.fixture(autouse=True)
def _fresh_dynamo():
    torch._dynamo.reset()
    yield
    torch._dynamo.reset()


def _pixels(device: str, dtype: torch.dtype, n: int = 2000):
    generator = torch.Generator().manual_seed(0)
    # Extend past the image so invalid (NaN) rows are traced as well.
    x = torch.rand(n, generator=generator, dtype=torch.float64) * 2600.0 - 300.0
    y = torch.rand(n, generator=generator, dtype=torch.float64) * 2600.0 - 300.0
    return x.to(device=device, dtype=dtype), y.to(device=device, dtype=dtype)


def _assert_same(actual, expected, *, atol: float) -> None:
    for a, e in zip(actual, expected):
        assert torch.equal(torch.isnan(a), torch.isnan(e))
        torch.testing.assert_close(
            torch.nan_to_num(a, nan=0.0),
            torch.nan_to_num(e, nan=0.0),
            atol=atol,
            rtol=0,
        )


def _check_fullgraph(wcs: WCS, x, y, *, backend: str, atol: float) -> None:
    with torch.no_grad():
        world = wcs.pixel_to_world(x, y)
        pixels = wcs.world_to_pixel(*world)
        stacked = wcs.pixel_to_world(torch.stack((x, y), dim=-1))

        p2w = torch.compile(
            lambda a, b: wcs.pixel_to_world(a, b), fullgraph=True, backend=backend
        )
        w2p = torch.compile(
            lambda a, b: wcs.world_to_pixel(a, b), fullgraph=True, backend=backend
        )
        p2w_stacked = torch.compile(
            lambda p: wcs.pixel_to_world(p), fullgraph=True, backend=backend
        )
        _assert_same(p2w(x, y), world, atol=atol)
        _assert_same(w2p(*world), pixels, atol=atol)
        _assert_same((p2w_stacked(torch.stack((x, y), dim=-1)),), (stacked,), atol=atol)


@pytest.mark.parametrize("name", list(HEADERS))
def test_fullgraph_trace_matches_eager_cpu(name: str) -> None:
    wcs = WCS(HEADERS[name])
    x, y = _pixels("cpu", torch.float64)
    # Dynamo tracing is what can break; the eager backend keeps the test fast.
    _check_fullgraph(wcs, x, y, backend="eager", atol=1e-9)


@pytest.mark.parametrize("name", ["TAN", "SIN", "TAN-SIP-cubic-APBP", "TPV"])
def test_fullgraph_inductor_matches_eager_cpu(name: str) -> None:
    wcs = WCS(HEADERS[name])
    x, y = _pixels("cpu", torch.float64, n=4096)
    _check_fullgraph(wcs, x, y, backend="inductor", atol=1e-9)


def test_traced_tpv_inverse_matches_eager_newton() -> None:
    wcs = WCS(HEADERS["TPV"])
    tpv = wcs.tpv
    assert tpv is not None
    generator = torch.Generator().manual_seed(1)
    xi = (torch.rand(5000, generator=generator, dtype=torch.float64) - 0.5) * 0.6
    eta = (torch.rand(5000, generator=generator, dtype=torch.float64) - 0.5) * 0.6
    xi[:3] = torch.tensor([float("nan"), float("inf"), 40.0], dtype=torch.float64)
    previous = tpv._USE_NATIVE_INVERT
    tpv._USE_NATIVE_INVERT = False
    try:
        eager = tpv.invert(xi, eta)
    finally:
        tpv._USE_NATIVE_INVERT = previous
    traced = tpv._invert_traced(xi, eta, 20, 1e-11)
    _assert_same(traced, eager, atol=1e-14)


def test_traced_sip_inverse_matches_eager() -> None:
    for name in ("TAN-SIP-quadratic", "TAN-SIP-cubic-APBP"):
        sip = WCS(HEADERS[name]).sip
        assert sip is not None
        generator = torch.Generator().manual_seed(2)
        u = (torch.rand(5000, generator=generator, dtype=torch.float64) - 0.5) * 2400.0
        v = (torch.rand(5000, generator=generator, dtype=torch.float64) - 0.5) * 2400.0
        eager = sip.invert_distortion(u, v)
        traced = sip._invert_distortion_traced(u, v, 6, 1e-6)
        _assert_same(traced, eager, atol=1e-12)


@requires_mps
@pytest.mark.parametrize("name", ["TAN", "SIN", "TAN-SIP-quadratic", "TPV"])
def test_fullgraph_trace_matches_eager_mps(name: str) -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(HEADERS[name]).to("mps")
    x, y = _pixels("mps", torch.float32)
    # Traced MPS graphs use sqrt(a*a + b*b) for hypot (see _math._hypot) and a
    # different TPV evaluation order: float32 rounding, up to ~0.01 px.
    _check_fullgraph(wcs, x, y, backend="eager", atol=0.05)


@requires_mps
def test_mps_precision_estimate_is_cached() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(HEADERS["TAN"]).to("mps")
    calls = 0
    original = wcs._mps_float32_error_pixels

    def counting() -> float:
        nonlocal calls
        calls += 1
        return original()

    wcs._mps_float32_error_pixels = counting  # type: ignore[method-assign]
    x, y = _pixels("mps", torch.float32, n=16)
    for _ in range(3):
        wcs.pixel_to_world(x, y)
    assert calls == 0  # estimated once by ``to("mps")``


@requires_mps
@pytest.mark.parametrize("name", ["TAN", "MOL"])
def test_fullgraph_inductor_matches_eager_mps(name: str) -> None:
    # Inductor's Metal backend has no hypot/copysign codegen; the WCS kernels
    # substitute them while tracing for MPS. Metal's float32 transcendental
    # functions differ from the eager MPS kernels by a few ulps of the
    # intermediate angles (~0.03 px here). The SIP and TPV inverses trace as
    # one graph on MPS too, but Inductor fuses each into a kernel with more
    # than Metal's 31 buffer arguments (PyTorch 2.14), so they are covered by
    # the eager-backend test above.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        wcs = WCS(HEADERS[name]).to("mps")
    x, y = _pixels("mps", torch.float32)
    _check_fullgraph(wcs, x, y, backend="inductor", atol=0.1)
