import math

import numpy as np
import torch
import pytest

from torchsky.sphere.geom import (
    PixelizedRegion,
    SphericalMultiPolygon,
    SphericalPolygon,
    query_polygon_general,
    spherical_polygon_area,
    spherical_polygon_contains,
    spherical_polygons_intersect,
    spherical_polygon_signed_area,
)
from torchsky.sphere import healpix as hp


def _has_cpp_geom() -> bool:
    try:
        from torchsky._native import cpp as _cpp_mod  # type: ignore
    except Exception:
        return False
    return hasattr(_cpp_mod, "_sphere_planar_point_in_polygon_cpu") and hasattr(
        _cpp_mod, "_sphere_arc_intersections_matrix_cpu"
    )


def _concave_polygon() -> tuple[torch.Tensor, torch.Tensor]:
    # Concave L-shaped polygon near the equator.
    lon = torch.tensor([-20.0, 20.0, 20.0, 0.0, 0.0, -20.0], dtype=torch.float64)
    lat = torch.tensor([-20.0, -20.0, 0.0, 0.0, 20.0, 20.0], dtype=torch.float64)
    return lon, lat


def test_spherical_polygon_signed_area_orientation() -> None:
    lon = torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64)
    a_pos = spherical_polygon_signed_area(lon, lat)
    a_neg = spherical_polygon_signed_area(
        torch.flip(lon, dims=[0]), torch.flip(lat, dims=[0])
    )
    np.testing.assert_allclose(float(a_pos), math.pi / 2.0, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(float(a_neg), -math.pi / 2.0, atol=1e-12, rtol=0.0)


def test_spherical_polygon_contains_concave_shape() -> None:
    lon_poly, lat_poly = _concave_polygon()
    inside_a = spherical_polygon_contains(-10.0, 10.0, lon_poly, lat_poly)
    inside_b = spherical_polygon_contains(10.0, -10.0, lon_poly, lat_poly)
    outside_notch = spherical_polygon_contains(10.0, 10.0, lon_poly, lat_poly)
    outside_far = spherical_polygon_contains(80.0, 0.0, lon_poly, lat_poly)
    assert bool(inside_a)
    assert bool(inside_b)
    assert not bool(outside_notch)
    assert not bool(outside_far)


def test_query_polygon_general_matches_convex_baseline() -> None:
    nside = 64
    lon = torch.tensor([0.0, 20.0, 20.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 20.0, 20.0], dtype=torch.float64)

    got = query_polygon_general(nside, lon, lat, nest=False)
    verts = hp.ang2vec(lon, lat, lonlat=True)
    expected = hp.query_polygon(nside, verts, nest=False)
    torch.testing.assert_close(got, expected)


def _geom_accel_device() -> torch.device | None:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return None


def test_spherical_polygon_contains_cuda_or_mps_matches_cpu() -> None:
    device = _geom_accel_device()
    if device is None:
        pytest.skip("no CUDA/MPS device")
    dtype = torch.float32 if device.type == "mps" else torch.float64
    lon_poly, lat_poly = _concave_polygon()
    lon = torch.tensor([-10.0, 10.0, 10.0, 80.0], dtype=dtype)
    lat = torch.tensor([10.0, -10.0, 10.0, 0.0], dtype=dtype)
    cpu = spherical_polygon_contains(lon, lat, lon_poly.to(dtype), lat_poly.to(dtype))
    got = spherical_polygon_contains(
        lon.to(device),
        lat.to(device),
        lon_poly.to(device=device, dtype=dtype),
        lat_poly.to(device=device, dtype=dtype),
    )
    assert got.device.type == device.type
    torch.testing.assert_close(got.cpu(), cpu)


def test_query_polygon_general_cuda_or_mps_matches_cpu() -> None:
    device = _geom_accel_device()
    if device is None:
        pytest.skip("no CUDA/MPS device")
    nside = 32
    dtype = torch.float32 if device.type == "mps" else torch.float64
    lon = torch.tensor([0.0, 20.0, 20.0, 0.0], dtype=dtype)
    lat = torch.tensor([0.0, 0.0, 20.0, 20.0], dtype=dtype)
    cpu = query_polygon_general(nside, lon, lat, nest=False)
    got = query_polygon_general(nside, lon.to(device), lat.to(device), nest=False)
    torch.testing.assert_close(got.cpu().sort().values, cpu.sort().values)


def test_spherical_polygons_intersect_cuda_or_mps_matches_cpu() -> None:
    device = _geom_accel_device()
    if device is None:
        pytest.skip("no CUDA/MPS device")
    dtype = torch.float32 if device.type == "mps" else torch.float64
    lon1 = torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=dtype)
    lat1 = torch.tensor([30.0, 30.0, 0.0, 0.0], dtype=dtype)
    lon2 = torch.tensor([-5.0, 15.0, 15.0, -5.0], dtype=dtype)
    lat2 = torch.tensor([20.0, 20.0, -10.0, -10.0], dtype=dtype)
    lon3 = torch.tensor([-20.0, 20.0, 20.0, -20.0], dtype=dtype)
    lat3 = torch.tensor([-20.0, -20.0, -10.0, -10.0], dtype=dtype)
    cpu_hit = spherical_polygons_intersect(lon1, lat1, lon2, lat2)
    cpu_miss = spherical_polygons_intersect(lon1, lat1, lon3, lat3)
    assert (
        spherical_polygons_intersect(
            lon1.to(device), lat1.to(device), lon2.to(device), lat2.to(device)
        )
        == cpu_hit
    )
    assert (
        spherical_polygons_intersect(
            lon1.to(device), lat1.to(device), lon3.to(device), lat3.to(device)
        )
        == cpu_miss
    )


def test_spherical_polygon_pixel_boolean_ops() -> None:
    nside = 32
    lon1 = torch.tensor([0.0, 25.0, 25.0, 0.0], dtype=torch.float64)
    lat1 = torch.tensor([0.0, 0.0, 25.0, 25.0], dtype=torch.float64)
    lon2 = torch.tensor([15.0, 40.0, 40.0, 15.0], dtype=torch.float64)
    lat2 = torch.tensor([10.0, 10.0, 35.0, 35.0], dtype=torch.float64)

    r1 = SphericalPolygon(lon1, lat1).pixelize(nside=nside, nest=False)
    r2 = SphericalPolygon(lon2, lat2).pixelize(nside=nside, nest=False)

    u = r1.union(r2)
    i = r1.intersection(r2)
    d = r1.difference(r2)

    assert isinstance(u, PixelizedRegion)
    assert isinstance(i, PixelizedRegion)
    assert isinstance(d, PixelizedRegion)
    assert i.pixels.numel() <= min(r1.pixels.numel(), r2.pixels.numel())
    assert d.pixels.numel() <= r1.pixels.numel()
    assert u.pixels.numel() >= max(r1.pixels.numel(), r2.pixels.numel())
    assert i.area() <= r1.area() + 1e-12


def test_pixelized_region_set_ops_preserve_sorted_unique_contract() -> None:
    r1 = PixelizedRegion(
        nside=32,
        nest=False,
        pixels=torch.tensor([7, 3, 5, 3, 1], dtype=torch.int64),
    )
    r2 = PixelizedRegion(
        nside=32,
        nest=False,
        pixels=torch.tensor([6, 5, 4, 3, 3], dtype=torch.int64),
    )

    torch.testing.assert_close(r1.pixels, torch.tensor([1, 3, 5, 7], dtype=torch.int64))
    torch.testing.assert_close(r2.pixels, torch.tensor([3, 4, 5, 6], dtype=torch.int64))
    torch.testing.assert_close(
        r1.intersection(r2).pixels,
        torch.tensor([3, 5], dtype=torch.int64),
    )
    torch.testing.assert_close(
        r1.difference(r2).pixels,
        torch.tensor([1, 7], dtype=torch.int64),
    )
    torch.testing.assert_close(
        r1.union(r2).pixels,
        torch.tensor([1, 3, 4, 5, 6, 7], dtype=torch.int64),
    )


def test_spherical_polygon_class_contracts() -> None:
    lon, lat = _concave_polygon()
    poly = SphericalPolygon(lon, lat)
    area = poly.area(degrees=False)
    signed = poly.signed_area(degrees=False)
    assert float(area) > 0.0
    assert float(abs(signed)) > 0.0

    # Reversing winding flips signed area and preserves absolute area.
    poly_rev = SphericalPolygon(torch.flip(lon, dims=[0]), torch.flip(lat, dims=[0]))
    signed_rev = poly_rev.signed_area(degrees=False)
    area_rev = poly_rev.area(degrees=False)
    np.testing.assert_allclose(float(signed), -float(signed_rev), atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(float(area), float(area_rev), atol=1e-12, rtol=0.0)

    assert bool(poly.contains(-10.0, 10.0))
    assert not bool(poly.contains(10.0, 10.0))


def test_spherical_polygon_area_units() -> None:
    lon = torch.tensor([0.0, 90.0, 0.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 0.0, 90.0], dtype=torch.float64)
    sr = spherical_polygon_area(lon, lat, degrees=False)
    deg2 = spherical_polygon_area(lon, lat, degrees=True)
    np.testing.assert_allclose(
        float(deg2), float(sr) * (180.0 / math.pi) ** 2, atol=1e-10, rtol=0.0
    )


def test_spherical_polygons_intersect_basic_cases() -> None:
    lon1 = torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64)
    lat1 = torch.tensor([30.0, 30.0, 0.0, 0.0], dtype=torch.float64)

    lon2 = torch.tensor([-5.0, 15.0, 15.0, -5.0], dtype=torch.float64)
    lat2 = torch.tensor([20.0, 20.0, -10.0, -10.0], dtype=torch.float64)

    lon3 = torch.tensor([-20.0, 20.0, 20.0, -20.0], dtype=torch.float64)
    lat3 = torch.tensor([-20.0, -20.0, -10.0, -10.0], dtype=torch.float64)

    assert spherical_polygons_intersect(lon1, lat1, lon2, lat2)
    assert not spherical_polygons_intersect(lon1, lat1, lon3, lat3)


def test_spherical_polygon_inside_reference_flip() -> None:
    # Square around the north pole. Inside reference at equator should flip semantics.
    lon = torch.tensor([0.0, 90.0, 180.0, 270.0], dtype=torch.float64)
    lat = torch.tensor([80.0, 80.0, 80.0, 80.0], dtype=torch.float64)

    north = spherical_polygon_contains(0.0, 89.0, lon, lat)
    south_default = spherical_polygon_contains(0.0, 0.0, lon, lat)
    south_flipped = spherical_polygon_contains(
        0.0,
        0.0,
        lon,
        lat,
        inside_lon_deg=0.0,
        inside_lat_deg=0.0,
    )
    assert bool(north)
    assert not bool(south_default)
    assert bool(south_flipped)


def test_spherical_polygon_intersects_method_and_multipolygon() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([30.0, 30.0, 0.0, 0.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([-5.0, 15.0, 15.0, -5.0], dtype=torch.float64),
        torch.tensor([20.0, 20.0, -10.0, -10.0], dtype=torch.float64),
    )
    p3 = SphericalPolygon(
        torch.tensor([120.0, 140.0, 140.0, 120.0], dtype=torch.float64),
        torch.tensor([-20.0, -20.0, -40.0, -40.0], dtype=torch.float64),
    )
    assert p1.intersects(p2)
    assert not p1.intersects(p3)

    mp = SphericalMultiPolygon((p1, p3))
    assert mp.intersects(p2)
    assert bool(mp.contains(0.0, 10.0))


@pytest.mark.skipif(
    not _has_cpp_geom(), reason="C++ sphere geometry kernels unavailable"
)
def test_spherical_polygon_contains_cpp_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    monkeypatch.setattr(geom, "_PLANAR_PIP_CPP_MIN_OPS", 0)
    lon_poly, lat_poly = _concave_polygon()
    q_lon = torch.linspace(-30.0, 30.0, 257, dtype=torch.float64)
    q_lat = torch.linspace(-30.0, 30.0, 257, dtype=torch.float64)
    grid_lon, grid_lat = torch.meshgrid(q_lon, q_lat, indexing="xy")

    got_cpp = geom.spherical_polygon_contains(
        grid_lon,
        grid_lat,
        lon_poly,
        lat_poly,
        inclusive=False,
    )

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_py = geom.spherical_polygon_contains(
        grid_lon,
        grid_lat,
        lon_poly,
        lat_poly,
        inclusive=False,
    )
    torch.testing.assert_close(got_cpp, got_py)


@pytest.mark.skipif(
    not _has_cpp_geom(), reason="C++ sphere geometry kernels unavailable"
)
def test_spherical_polygons_intersect_cpp_matches_python_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    lon1 = torch.tensor([100.0, 114.0, 130.0, 114.0, 100.0, 86.0], dtype=torch.float64)
    lat1 = torch.tensor([-25.0, -15.0, -20.0, -35.0, -40.0, -32.0], dtype=torch.float64)
    lon2 = torch.tensor([112.0, 126.0, 138.0, 128.0, 112.0, 100.0], dtype=torch.float64)
    lat2 = torch.tensor([-18.0, -10.0, -18.0, -30.0, -34.0, -26.0], dtype=torch.float64)
    lon3 = torch.tensor([210.0, 222.0, 234.0, 226.0, 212.0, 202.0], dtype=torch.float64)
    lat3 = torch.tensor([18.0, 26.0, 18.0, 6.0, 2.0, 10.0], dtype=torch.float64)

    got_cpp = (
        geom.spherical_polygons_intersect(lon1, lat1, lon2, lat2),
        geom.spherical_polygons_intersect(lon1, lat1, lon3, lat3),
    )

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_py = (
        geom.spherical_polygons_intersect(lon1, lat1, lon2, lat2),
        geom.spherical_polygons_intersect(lon1, lat1, lon3, lat3),
    )
    assert got_cpp == got_py


@pytest.mark.skipif(
    not _has_cpp_geom(), reason="C++ sphere geometry kernels unavailable"
)
def test_cpp_geometry_predicate_accepts_autograd_inputs() -> None:
    import torchsky.sphere.geom as geom

    a0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    a1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    b0 = torch.tensor([[0.5, 0.5, -1.0]], dtype=torch.float64)
    b0 = (b0 / torch.linalg.norm(b0, dim=-1, keepdim=True)).requires_grad_(True)
    b1 = torch.tensor([[0.5, 0.5, 1.0]], dtype=torch.float64)
    b1 = (b1 / torch.linalg.norm(b1, dim=-1, keepdim=True)).requires_grad_(True)

    out = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1)
    assert out.dtype == torch.bool
    assert not out.requires_grad


def _unit_xyz_rad(lon_rad: torch.Tensor, lat_rad: torch.Tensor) -> torch.Tensor:
    cos_lat = torch.cos(lat_rad)
    return torch.stack(
        (
            cos_lat * torch.cos(lon_rad),
            cos_lat * torch.sin(lon_rad),
            torch.sin(lat_rad),
        ),
        dim=-1,
    )


def test_short_great_circle_arcs_cross_at_two_e_minus_eight_radians(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    delta = 1e-8
    zeros = torch.zeros(1, dtype=torch.float64)
    a0 = _unit_xyz_rad(torch.tensor([-delta], dtype=torch.float64), zeros)
    a1 = _unit_xyz_rad(torch.tensor([delta], dtype=torch.float64), zeros)
    b0 = _unit_xyz_rad(zeros, torch.tensor([-delta], dtype=torch.float64))
    b1 = _unit_xyz_rad(zeros, torch.tensor([delta], dtype=torch.float64))

    got = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1, tol_rad=0.0)
    assert got.tolist() == [[True]]

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_python = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1, tol_rad=0.0)
    assert got_python.tolist() == [[True]]


def test_coincident_great_circle_arcs_overlap_touch_and_disjoint(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    lat = torch.zeros(3, dtype=torch.float64)
    a0 = _unit_xyz_rad(torch.deg2rad(torch.tensor([0.0])), lat[:1])
    a1 = _unit_xyz_rad(torch.deg2rad(torch.tensor([20.0])), lat[:1])
    b0 = _unit_xyz_rad(torch.deg2rad(torch.tensor([10.0, 20.0, 21.0])), lat)
    b1 = _unit_xyz_rad(torch.deg2rad(torch.tensor([30.0, 30.0, 30.0])), lat)
    expected = [[True, True, False]]

    got = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1, tol_rad=0.0)
    assert got.tolist() == expected

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_python = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1, tol_rad=0.0)
    assert got_python.tolist() == expected


def test_zero_length_and_antipodal_arc_behavior(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    zero = torch.zeros(3, dtype=torch.float64)
    a0 = _unit_xyz_rad(
        torch.deg2rad(torch.tensor([10.0, 30.0, 0.0], dtype=torch.float64)), zero
    )
    a1 = _unit_xyz_rad(
        torch.deg2rad(torch.tensor([10.0, 30.0, 180.0], dtype=torch.float64)), zero
    )
    b0 = _unit_xyz_rad(
        torch.deg2rad(torch.tensor([0.0], dtype=torch.float64)), zero[:1]
    )
    b1 = _unit_xyz_rad(
        torch.deg2rad(torch.tensor([20.0], dtype=torch.float64)), zero[:1]
    )
    expected = [[True], [False], [False]]

    got = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1)
    assert got.tolist() == expected

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_python = geom._great_circle_arcs_intersect_matrix(a0, a1, b0, b1)
    assert got_python.tolist() == expected


def test_planar_point_in_polygon_handles_shallow_and_horizontal_edges(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    polygon = torch.tensor(
        [[0.0, 0.0], [2.0, 1e-16], [2.0, 1.0], [0.0, 1.0]],
        dtype=torch.float64,
    )
    points = torch.tensor([[0.5, 5e-17], [1.5, 5e-17], [0.5, 0.5]], dtype=torch.float64)
    expected = [True, False, True]

    monkeypatch.setattr(geom, "_PLANAR_PIP_CPP_MIN_OPS", 0)
    got = geom._planar_point_in_polygon(points, polygon, eps=1e-15)
    assert got.tolist() == expected

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    got_python = geom._planar_point_in_polygon(points, polygon, eps=1e-15)
    assert got_python.tolist() == expected


@pytest.mark.parametrize("bad", [-1.0, math.inf, math.nan])
def test_geometry_predicates_reject_invalid_tolerances(bad: float) -> None:
    import torchsky.sphere.geom as geom

    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    polygon = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    arc0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    arc1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

    with pytest.raises(ValueError, match="finite and nonnegative"):
        geom._planar_point_in_polygon(points, polygon, eps=bad)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        geom._great_circle_arcs_intersect_matrix(arc0, arc1, arc0, arc1, tol_rad=bad)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        spherical_polygon_contains(
            0.1,
            0.1,
            polygon[:, 0],
            polygon[:, 1],
            atol_deg=bad,
        )


def test_python_geometry_predicates_return_non_differentiable_bools(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import torchsky.sphere.geom as geom

    monkeypatch.setattr(geom._ops, "has_fixed_dispatcher", lambda: False)
    points = torch.tensor([[0.25, 0.25]], dtype=torch.float64, requires_grad=True)
    polygon = torch.tensor(
        [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
        dtype=torch.float64,
        requires_grad=True,
    )
    pip = geom._planar_point_in_polygon(points, polygon)

    a0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64, requires_grad=True)
    a1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64, requires_grad=True)
    arcs = geom._great_circle_arcs_intersect_matrix(a0, a1, a0, a1)

    assert pip.dtype == torch.bool and not pip.requires_grad
    assert arcs.dtype == torch.bool and not arcs.requires_grad


@pytest.mark.skipif(
    not _has_cpp_geom(), reason="C++ sphere geometry kernels unavailable"
)
def test_cpp_geometry_predicates_reject_invalid_tolerances() -> None:
    from torchsky._native import cpp as cpp

    points = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    polygon = torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64)
    arc0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    arc1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)

    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        cpp._sphere_planar_point_in_polygon_cpu(points, polygon, -1.0)
    with pytest.raises(RuntimeError, match="finite and nonnegative"):
        cpp._sphere_arc_intersections_matrix_cpu(arc0, arc1, arc0, arc1, math.inf)
