"""``torch.compile(fullgraph=True)`` contracts for the HEALPix point operations.

The pure-torch paths (used on CUDA/MPS and without the native extension) must
trace as a single graph and return exactly the eager pixels. The native CPU
kernels have fake (meta) implementations, so they compile as opaque ops.
Most checks use the cheap ``aot_eager`` backend (``fullgraph=True`` still
fails on any graph break); one test per family runs Inductor.
"""

import math

import pytest
import torch

import torchsky._ops as _ops
from torchsky.sphere import healpix as hp


@pytest.fixture(autouse=True)
def _fresh_dynamo() -> None:
    torch._dynamo.reset()


@pytest.fixture(params=["torch", "native"])
def path(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> str:
    if request.param == "native":
        if not _ops.has_fixed_dispatcher():
            pytest.skip("native extension unavailable")
    else:
        monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    return request.param


def _points(n: int, device: str = "cpu") -> tuple[torch.Tensor, torch.Tensor]:
    gen = torch.Generator().manual_seed(1234)
    lon = torch.rand(n, generator=gen, dtype=torch.float64) * 360.0
    lat = torch.rad2deg(
        torch.asin(2.0 * torch.rand(n, generator=gen, dtype=torch.float64) - 1.0)
    )
    # Poles, the z = +-2/3 transition and the longitude wrap.
    extra_lat = torch.tensor(
        [
            90.0,
            -90.0,
            89.9999999,
            -89.9999999,
            0.0,
            41.8103148957786,
            -41.8103148957786,
        ],
        dtype=torch.float64,
    )
    extra_lon = torch.tensor(
        [0.0, 359.9999999, 45.0, 90.0, 180.0, 270.0, 1.0e-9], dtype=torch.float64
    )
    lon = torch.cat((lon, extra_lon))
    lat = torch.cat((lat, extra_lat))
    if device == "mps":
        return lon.float().to(device), lat.float().to(device)
    return lon, lat


def _point_ops(nside: int, nest: bool):  # noqa: ANN202
    def fn(lon: torch.Tensor, lat: torch.Tensor, pix: torch.Tensor):  # noqa: ANN202
        theta = torch.deg2rad(90.0 - lat)
        phi = torch.deg2rad(lon)
        return (
            hp.ang2pix(nside, lon, lat, nest=nest, lonlat=True),
            hp.ang2pix(nside, theta, phi, nest=nest),
            *hp.pix2ang(nside, pix, nest=nest, lonlat=True),
            hp.ring2nest(nside, pix),
            hp.nest2ring(nside, pix),
            *hp.get_interp_weights(nside, lon, lat, nest=nest, lonlat=True),
        )

    return fn


def _sample_pixels(nside: int, device: str = "cpu") -> torch.Tensor:
    npix = 12 * nside * nside
    gen = torch.Generator().manual_seed(nside % 9973)
    pix = torch.cat(
        (
            torch.arange(min(npix, 300)),
            torch.randint(0, npix, (2000,), generator=gen),
            npix - 1 - torch.arange(min(npix, 300)),
        )
    )
    return pix.to(device)


def _assert_same(got: tuple, ref: tuple, atol: float = 1.0e-12) -> None:
    for g, r in zip(got, ref, strict=True):
        if r.dtype == torch.int64:
            assert torch.equal(g.cpu(), r.cpu())
        else:
            torch.testing.assert_close(g.cpu(), r.cpu(), rtol=0.0, atol=atol)


@pytest.mark.parametrize("nside", [1, 2, 64, 1 << 29])
@pytest.mark.parametrize("nest", [False, True])
def test_point_ops_trace_as_one_graph_and_match_eager(
    path: str, nside: int, nest: bool
) -> None:
    lon, lat = _points(2000)
    pix = _sample_pixels(nside)
    fn = _point_ops(nside, nest)
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    _assert_same(compiled(lon, lat, pix), fn(lon, lat, pix))


@pytest.mark.parametrize("nest", [False, True])
def test_inductor_ang2pix_and_interp_val_match_eager(
    monkeypatch: pytest.MonkeyPatch, nest: bool
) -> None:
    monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    nside = 1 << 20
    lon, lat = _points(4000)
    m = torch.linspace(-1.0, 1.0, 12 * 64 * 64, dtype=torch.float64)

    def fn(m: torch.Tensor, lon: torch.Tensor, lat: torch.Tensor):  # noqa: ANN202
        return (
            hp.ang2pix(nside, lon, lat, nest=nest, lonlat=True),
            hp.get_interp_val(m, lon, lat, nest=nest, lonlat=True),
        )

    compiled = torch.compile(fn, fullgraph=True, dynamic=False)
    _assert_same(compiled(m, lon, lat), fn(m, lon, lat))


def test_interp_value_gradients_compile_without_nans(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    nside = 16
    m = torch.linspace(0.0, 1.0, 12 * nside * nside, dtype=torch.float64)
    lat = torch.tensor([90.0, 89.0, 0.0, -89.5, -90.0], dtype=torch.float64)
    lon = torch.tensor([0.0, 10.0, 20.0, 30.0, 40.0], dtype=torch.float64)

    def loss(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        return hp.get_interp_val(m, lon, lat, lonlat=True).sum()

    lon_e = lon.clone().requires_grad_()
    lat_e = lat.clone().requires_grad_()
    loss(lon_e, lat_e).backward()
    lon_c = lon.clone().requires_grad_()
    lat_c = lat.clone().requires_grad_()
    compiled = torch.compile(loss, fullgraph=True, dynamic=False, backend="aot_eager")
    compiled(lon_c, lat_c).backward()
    assert lon_e.grad is not None and lat_e.grad is not None
    assert lon_c.grad is not None and lat_c.grad is not None
    assert torch.isfinite(lon_e.grad).all() and torch.isfinite(lat_e.grad).all()
    torch.testing.assert_close(lon_c.grad, lon_e.grad, rtol=1.0e-10, atol=1.0e-12)
    torch.testing.assert_close(lat_c.grad, lat_e.grad, rtol=1.0e-10, atol=1.0e-12)


def test_eager_validation_still_raises_when_not_compiling(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_ops, "has_fixed_dispatcher", lambda: False)
    with pytest.raises(ValueError, match="latitude"):
        hp.ang2pix(4, torch.tensor([0.0]), torch.tensor([91.0]), lonlat=True)
    with pytest.raises(ValueError, match="finite"):
        hp.ang2pix(4, torch.tensor([math.nan]), torch.tensor([0.0]), lonlat=True)
    with pytest.raises(ValueError, match="out of range"):
        hp.pix2ang(4, torch.tensor([12 * 16]), nest=True)
    with pytest.raises(ValueError, match="out of range"):
        hp.pix2ang(4, torch.tensor([-1]))


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
@pytest.mark.parametrize("nside", [256, 1 << 29])
@pytest.mark.parametrize("nest", [False, True])
def test_mps_point_ops_trace_as_one_graph(nside: int, nest: bool) -> None:
    lon, lat = _points(2000, device="mps")
    pix = _sample_pixels(nside, device="mps")
    fn = _point_ops(nside, nest)
    compiled = torch.compile(fn, fullgraph=True, dynamic=False, backend="aot_eager")
    eager = fn(lon, lat, pix)
    # Adjacent order-29 rings can share a float32 colatitude: weights stay finite.
    assert all(torch.isfinite(t).all() for t in eager if t.is_floating_point())
    _assert_same(compiled(lon, lat, pix), eager, atol=1.0e-4)
