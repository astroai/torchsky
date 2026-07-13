import math
from pathlib import Path
import sys

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

try:
    from spherical_geometry.polygon import SphericalPolygon as SGP  # type: ignore
except Exception:  # pragma: no cover - optional comparator
    SGP = None

try:
    from spherical_geometry.tests.helpers import ROOT_DIR as SG_ROOT_DIR  # type: ignore
except Exception:  # pragma: no cover - optional comparator fixtures
    SG_ROOT_DIR = None


def _has_cpp_geom() -> bool:
    try:
        from torchsky import cpp as _cpp_mod  # type: ignore
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


def _strip_closed(
    lon: np.ndarray,
    lat: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _xy_to_lonlat(xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    x = xyz[:, 0]
    y = xyz[:, 1]
    z = xyz[:, 2]
    lon = np.remainder(np.degrees(np.arctan2(y, x)), 360.0)
    lat = np.degrees(np.arcsin(np.clip(z / np.linalg.norm(xyz, axis=1), -1.0, 1.0)))
    return lon.astype(np.float64), lat.astype(np.float64)


def _load_difficult_pairs(
    max_pairs: int,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    if SG_ROOT_DIR is None:
        return []
    path = Path(SG_ROOT_DIR) / "difficult_intersections.txt"
    if not path.exists():
        return []

    raw = path.read_bytes().splitlines()
    pairs: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for i in range(0, len(raw), 4):
        if len(pairs) >= max_pairs or i + 3 >= len(raw):
            break
        a_points = np.frombuffer(
            bytes.fromhex(raw[i].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        a_inside = np.frombuffer(
            bytes.fromhex(raw[i + 1].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        b_points = np.frombuffer(
            bytes.fromhex(raw[i + 2].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        b_inside = np.frombuffer(
            bytes.fromhex(raw[i + 3].decode("ascii")), dtype="<f8"
        ).reshape((-1, 3))
        pairs.append((a_points, a_inside, b_points, b_inside))
    return pairs


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

    monkeypatch.setattr(geom, "_cpp", None)
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

    monkeypatch.setattr(geom, "_cpp", None)
    got_py = (
        geom.spherical_polygons_intersect(lon1, lat1, lon2, lat2),
        geom.spherical_polygons_intersect(lon1, lat1, lon3, lat3),
    )
    assert got_cpp == got_py


@pytest.mark.skipif(SGP is None, reason="spherical-geometry not available")
def test_spherical_polygon_contains_matches_spherical_geometry() -> None:
    lon_poly = np.array(
        [106.0, 114.0, 130.0, 114.0, 106.0, 98.0, 82.0, 98.0, 106.0], dtype=np.float64
    )
    lat_poly = np.array(
        [-20.0, -15.0, -20.0, -25.0, -32.0, -25.0, -20.0, -15.0, -20.0],
        dtype=np.float64,
    )
    rng = np.random.default_rng(123)
    q_lon = rng.uniform(70.0, 140.0, size=2000)
    q_lat = rng.uniform(-40.0, -8.0, size=2000)

    if hasattr(SGP, "from_lonlat"):
        sg_poly = SGP.from_lonlat(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_lonlat(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    elif hasattr(SGP, "from_radec"):
        sg_poly = SGP.from_radec(lon_poly, lat_poly, degrees=True)
        sg_contains = np.asarray(
            [
                sg_poly.contains_radec(float(lo), float(la), degrees=True)
                for lo, la in zip(q_lon, q_lat, strict=False)
            ],
            dtype=bool,
        )
    else:
        pytest.skip("unsupported spherical-geometry API")

    tf_contains = (
        spherical_polygon_contains(
            torch.from_numpy(q_lon),
            torch.from_numpy(q_lat),
            torch.from_numpy(lon_poly),
            torch.from_numpy(lat_poly),
            inclusive=False,
        )
        .cpu()
        .numpy()
    )
    np.testing.assert_array_equal(tf_contains, sg_contains)


@pytest.mark.skipif(
    SGP is None or SG_ROOT_DIR is None,
    reason="spherical-geometry fixtures not available",
)
def test_spherical_polygon_intersects_matches_upstream_difficult_pairs() -> None:
    difficult_pairs = _load_difficult_pairs(max_pairs=6)
    if not difficult_pairs:
        pytest.skip("upstream difficult_intersections fixtures unavailable")

    for a_points, a_inside, b_points, b_inside in difficult_pairs:
        sg_a = SGP(a_points, a_inside)
        sg_b = SGP(b_points, b_inside)

        lon_a, lat_a = _xy_to_lonlat(a_points)
        lon_b, lat_b = _xy_to_lonlat(b_points)
        inside_a_lon, inside_a_lat = _xy_to_lonlat(a_inside[:1])
        inside_b_lon, inside_b_lat = _xy_to_lonlat(b_inside[:1])
        lon_a, lat_a = _strip_closed(lon_a, lat_a)
        lon_b, lat_b = _strip_closed(lon_b, lat_b)

        tf_a = SphericalPolygon(
            torch.from_numpy(lon_a),
            torch.from_numpy(lat_a),
            inside_lon_deg=float(inside_a_lon[0]),
            inside_lat_deg=float(inside_a_lat[0]),
        )
        tf_b = SphericalPolygon(
            torch.from_numpy(lon_b),
            torch.from_numpy(lat_b),
            inside_lon_deg=float(inside_b_lon[0]),
            inside_lat_deg=float(inside_b_lat[0]),
        )

        assert tf_a.intersects(tf_b) == bool(sg_a.intersects_poly(sg_b))


@pytest.mark.skipif(
    SGP is None or SG_ROOT_DIR is None,
    reason="spherical-geometry fixtures not available",
)
@pytest.mark.performance
def test_upstream_polygon_replay_gate_small() -> None:
    root = Path(__file__).resolve().parents[1]
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))

    from benchmarks.replays.replay_upstream_spherical_geometry_polygons import (
        ReplaySettings,
        run_replay,
    )

    payload = run_replay(
        ReplaySettings(
            n_points=4_000,
            runs=1,
            seed=123,
            intersection_nside=256,
            max_difficult_pairs=1,
            max_contains_mismatches_nonboundary=0,
            max_pair_intersection_mismatches=0,
            max_difficult_nonempty_mismatches=0,
            max_area_rel_error=1e-10,
            difficult_area_nsides=(128, 256, 512, 1024, 2048, 4096, 8192, 16384),
            max_difficult_area_rel_error_final=0.12,
            max_difficult_area_convergence_rel=0.10,
            allow_nonrelease_distributions=True,
        )
    )
    assert payload["failures"] == []
