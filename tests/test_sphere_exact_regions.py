import numpy as np
import torch

from torchsky.sphere.geom import (
    NativeExactSphericalRegion,
    SphericalPolygon,
    to_exact_region,
)












def test_native_exact_region_basic_area_contains() -> None:
    poly = SphericalPolygon(
        torch.tensor([-10.0, 10.0, 10.0, -10.0], dtype=torch.float64),
        torch.tensor([-5.0, -5.0, 5.0, 5.0], dtype=torch.float64),
    )
    reg = to_exact_region(poly, backend="native", nsides=(64, 128, 256))
    assert isinstance(reg, NativeExactSphericalRegion)
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

    assert u.area() >= r1.area()
    assert i.area() <= r1.area()
    np.testing.assert_allclose(d.area() + i.area(), r1.area(), rtol=0.0, atol=5e-3)
