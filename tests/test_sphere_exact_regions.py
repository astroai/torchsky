import math

import numpy as np
import pytest
import torch

from torchsky.sphere.core import lonlat_to_unit_xyz, unit_xyz_to_lonlat
from torchsky.sphere.geom import (
    NativeApproximateSphericalRegion,
    SphericalBooleanRegion,
    SphericalCap,
    SphericalMultiPolygon,
    SphericalPolygon,
    spherical_polygon_area,
    to_exact_region,
)


def test_native_exact_region_basic_area_contains() -> None:
    poly = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64),
    )
    reg = to_exact_region(poly, backend="native", nsides=(64, 128, 256))
    assert isinstance(reg, NativeApproximateSphericalRegion)
    assert reg.area() > 0.0
    assert bool(reg.contains(0.0, 0.0))
    assert not bool(reg.contains(60.0, 60.0))


def test_native_exact_region_boolean_area_ordering() -> None:
    p1 = SphericalPolygon(
        torch.tensor([-8.0, 8.0, 8.0, -8.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    p2 = SphericalPolygon(
        torch.tensor([0.0, 14.0, 14.0, 0.0], dtype=torch.float64),
        torch.tensor([-6.0, -6.0, 6.0, 6.0], dtype=torch.float64),
    )
    r1 = to_exact_region(p1, backend="native", nsides=(64, 128, 256))
    u = r1.union(p2)
    i = r1.intersection(p2)
    d = r1.difference(p2)

    assert not u.is_empty
    with pytest.raises(RuntimeError, match="native intersection"):
        u.intersects(p1)
    # Polygon booleans have an exact area; the HEALPix estimate converges to it.
    np.testing.assert_allclose(
        u.area(), r1.area() + float(p2.area()) - i.area(), rtol=1e-12
    )
    np.testing.assert_allclose(d.area() + i.area(), r1.area(), rtol=1e-12)
    for region in (u, i, d):
        np.testing.assert_allclose(
            region.area_estimate().area_sr, region.area(), rtol=0.0, atol=5e-3
        )


def test_native_tiny_cap_uses_analytic_area_and_emptiness() -> None:
    cap = SphericalCap(12.0, -7.0, 1e-7)
    region = to_exact_region(cap, backend="native", nsides=(1, 2, 4))

    assert bool(region.contains(cap.lon_deg, cap.lat_deg))
    assert region.area() > 0.0
    assert not region.is_empty
    assert region.area_estimate().area_sr == 0.0


def test_native_cap_intersection_is_analytic_below_pixel_scale() -> None:
    left = to_exact_region(
        SphericalCap(0.0, 0.0, 1e-4), backend="native", nsides=(1, 2)
    )
    overlap = SphericalCap(0.00015, 0.0, 1e-4)
    disjoint = SphericalCap(0.00025, 0.0, 1e-4)

    assert left.intersects(overlap)
    assert not left.intersects(disjoint)


def test_native_approximate_region_clean_breaks() -> None:
    region = to_exact_region(SphericalCap(0.0, 0.0, 1.0), backend="native")
    assert isinstance(region, NativeApproximateSphericalRegion)
    with pytest.raises(TypeError):
        region.contains(0.0, 0.0, inclusive=False)
    with pytest.raises(TypeError):
        region.intersects(SphericalCap(0.0, 0.0, 1.0), cap_steps=32)
    with pytest.raises(TypeError):
        region.union(SphericalCap(0.0, 0.0, 1.0), cap_steps=32)


def test_exact_region_backend_auto_is_removed() -> None:
    with pytest.raises(ValueError, match="spherical-geometry.*native"):
        to_exact_region(SphericalCap(0.0, 0.0, 1.0), backend="auto")  # type: ignore[arg-type]


def _poly(lon: list[float], lat: list[float], **kwargs) -> SphericalPolygon:
    return SphericalPolygon(
        torch.tensor(lon, dtype=torch.float64),
        torch.tensor(lat, dtype=torch.float64),
        **kwargs,
    )


_OCTANT = _poly([0.0, 90.0, 0.0], [0.0, 0.0, 90.0])
_OCTANT_45 = _poly([45.0, 135.0, 0.0], [0.0, 0.0, 90.0])


def test_exact_polygon_booleans_have_exact_area() -> None:
    # Two octants rotated by 45 degrees share a lune of exactly pi / 4.
    intersection = _OCTANT.exact_intersection(_OCTANT_45)
    assert isinstance(intersection, SphericalBooleanRegion)
    quarter = math.pi / 4.0
    np.testing.assert_allclose(intersection.area(), quarter, rtol=1e-14)
    np.testing.assert_allclose(
        _OCTANT.exact_union(_OCTANT_45).area(), 3.0 * quarter, rtol=1e-14
    )
    np.testing.assert_allclose(
        _OCTANT.exact_difference(_OCTANT_45).area(), quarter, rtol=1e-14
    )
    np.testing.assert_allclose(
        intersection.area(degrees=True), 129600.0 / math.pi / 16.0, rtol=1e-14
    )
    np.testing.assert_allclose(intersection.exact_area(), quarter, rtol=1e-14)
    # Winding order of an operand does not matter.
    reversed_octant = _poly([0.0, 0.0, 90.0], [0.0, 90.0, 0.0])
    np.testing.assert_allclose(
        reversed_octant.intersection(_OCTANT_45).area(), quarter, rtol=1e-14
    )


def test_exact_area_of_non_convex_clip_matches_constructed_polygon() -> None:
    l_shape = _poly([0, 10, 10, 5, 5, 0], [0, 0, 5, 5, 10, 10])
    square = _poly([3, 8, 8, 3], [3, 3, 8, 8])
    clipped = l_shape.exact_intersection(square)

    def crossing(p, q, r, s):
        a, b, c, d = (
            lonlat_to_unit_xyz(*torch.tensor(z, dtype=torch.float64))
            for z in (p, q, r, s)
        )
        x = torch.linalg.cross(torch.linalg.cross(a, b), torch.linalg.cross(c, d))
        x = x / x.norm()
        return x if float((x * (a + b)).sum()) > 0 else -x

    # The clipped boundary built by hand: its two crossing vertices are exact
    # great-circle intersections, so Girard's area is the oracle.
    vertices = torch.stack(
        [
            lonlat_to_unit_xyz(*torch.tensor((3.0, 3.0), dtype=torch.float64)),
            lonlat_to_unit_xyz(*torch.tensor((8.0, 3.0), dtype=torch.float64)),
            crossing((10, 5), (5, 5), (8, 3), (8, 8)),
            lonlat_to_unit_xyz(*torch.tensor((5.0, 5.0), dtype=torch.float64)),
            crossing((5, 5), (5, 10), (3, 8), (8, 8)),
            lonlat_to_unit_xyz(*torch.tensor((3.0, 8.0), dtype=torch.float64)),
        ]
    )
    lon, lat = unit_xyz_to_lonlat(vertices)
    expected = float(spherical_polygon_area(lon, lat))
    np.testing.assert_allclose(clipped.area(), expected, rtol=1e-12)
    union = l_shape.exact_union(square).area()
    np.testing.assert_allclose(
        union, float(l_shape.area()) + float(square.area()) - expected, rtol=1e-12
    )


def test_exact_area_shared_edges_nesting_and_disjoint() -> None:
    adjacent = _poly([90.0, 180.0, 0.0], [0.0, 0.0, 90.0])
    np.testing.assert_allclose(
        _OCTANT.exact_union(adjacent).area(), math.pi, rtol=1e-14
    )
    assert _OCTANT.exact_intersection(adjacent).area() == pytest.approx(0.0, abs=1e-14)
    np.testing.assert_allclose(
        _OCTANT.exact_intersection(_OCTANT).area(), math.pi / 2.0, rtol=1e-14
    )
    assert _OCTANT.exact_difference(_OCTANT).area() == pytest.approx(0.0, abs=1e-14)
    inner = _poly([10.0, 20.0, 15.0], [10.0, 10.0, 20.0])
    np.testing.assert_allclose(
        _OCTANT.exact_intersection(inner).area(), float(inner.area()), rtol=1e-12
    )
    # An inside point on the far side selects the polygon's complement.
    outside_inner = _poly(
        [10.0, 20.0, 15.0],
        [10.0, 10.0, 20.0],
        inside_lon_deg=200.0,
        inside_lat_deg=-40.0,
    )
    np.testing.assert_allclose(
        _OCTANT.exact_intersection(outside_inner).area(),
        math.pi / 2.0 - float(inner.area()),
        rtol=1e-12,
    )
    opposite = _poly([180.0, 270.0, 0.0], [0.0, 0.0, -90.0])
    assert _OCTANT.exact_intersection(opposite).area() == 0.0
    np.testing.assert_allclose(
        _OCTANT.exact_union(opposite).area(), math.pi, rtol=1e-14
    )
    # Nested booleans and multipolygons with overlapping members.
    nested = _OCTANT.intersection(_OCTANT_45).union(adjacent)
    np.testing.assert_allclose(nested.area(), 0.75 * math.pi, rtol=1e-14)
    overlapping = SphericalMultiPolygon((_OCTANT, _OCTANT_45))
    # The members overlap; the multipolygon area counts the overlap once.
    np.testing.assert_allclose(float(overlapping.area()), 0.75 * math.pi, rtol=1e-14)
    np.testing.assert_allclose(
        to_exact_region(overlapping).area(), 0.75 * math.pi, rtol=1e-14
    )


def test_exact_area_fails_closed_for_small_circle_operands() -> None:
    cap = SphericalCap(10.0, 10.0, 5.0)
    np.testing.assert_allclose(
        cap.area(), 2.0 * math.pi * (1.0 - math.cos(math.radians(5.0))), rtol=1e-14
    )
    with pytest.raises(ValueError, match="great-circle"):
        _OCTANT.exact_intersection(cap).area()
    with pytest.raises(RuntimeError, match="area_estimate"):
        to_exact_region(_OCTANT.intersection(cap)).area()
