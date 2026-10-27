import numpy as np
import pytest
import torch

from torchsky.sphere.geom import (
    NativeApproximateSphericalRegion,
    SphericalCap,
    SphericalPolygon,
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

    with pytest.raises(RuntimeError, match="area_estimate"):
        u.area()
    with pytest.raises(RuntimeError, match="analytic native area"):
        _ = u.is_empty
    with pytest.raises(RuntimeError, match="native intersection"):
        u.intersects(p1)
    u_est = u.area_estimate().area_sr
    i_est = i.area_estimate().area_sr
    d_est = d.area_estimate().area_sr
    assert u_est >= r1.area()
    assert i_est <= r1.area()
    np.testing.assert_allclose(d_est + i_est, r1.area(), rtol=0.0, atol=5e-3)


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
