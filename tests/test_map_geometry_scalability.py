from __future__ import annotations

from types import SimpleNamespace

import pytest
import torch

from torchsky.map.operators import geometry_drizzle as drizzle_geometry
from torchsky.map.operators.geometry_conservative import (
    conservative_polygon_project,
)
from torchsky.map.operators.geometry_drizzle import (
    exact_point_drizzle_project,
    exact_square_drizzle_project,
)


class _IdentityWCS:
    def __init__(self) -> None:
        self.batch_sizes: list[int] = []

    def pixel_to_world(self, x: torch.Tensor, y: torch.Tensor):
        self.batch_sizes.append(int(x.numel()))
        return x, y

    def world_to_pixel(self, x: torch.Tensor, y: torch.Tensor):
        self.batch_sizes.append(int(x.numel()))
        return x, y


class _MagnifyingWCS:
    def __init__(self, size: int) -> None:
        self.size = size

    def world_to_pixel(self, x: torch.Tensor, y: torch.Tensor):
        scale = float(self.size)
        return (x + 0.5) * scale - 0.5, (y + 0.5) * scale - 0.5


class _BowTieWCS:
    def world_to_pixel(self, x: torch.Tensor, y: torch.Tensor):
        return x + 1.0, x * y + 1.0


def _unused_pixel_area(*args, **kwargs):
    raise AssertionError("surface-brightness projection must not request pixel areas")


def _target(shape: tuple[int, int], wcs) -> SimpleNamespace:
    return SimpleNamespace(shape=shape, wcs=wcs)


def _project(
    kernel: str,
    image: torch.Tensor,
    weight: torch.Tensor,
    source_wcs,
    target,
    *,
    chunk_pixels: int,
):
    kwargs = {
        "value_semantics": "surface_brightness",
        "pixel_area_sr_map_fn": _unused_pixel_area,
        "chunk_pixels": chunk_pixels,
    }
    if kernel == "point":
        return exact_point_drizzle_project(
            image, weight, source_wcs, target, **kwargs
        )
    function = (
        exact_square_drizzle_project
        if kernel == "square"
        else conservative_polygon_project
    )
    return function(
        image,
        weight,
        source_wcs,
        target,
        drizzle_pixfrac=1.0,
        **kwargs,
    )


@pytest.mark.parametrize("kernel", ["point", "square", "conservative"])
def test_geometry_projection_constructs_only_tile_local_coordinates(
    kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    height, width = 29, 31
    chunk_pixels = 37
    image = torch.ones((height, width), dtype=torch.float32)
    weight = torch.ones_like(image)
    wcs = _IdentityWCS()
    nonzero_sizes: list[int] = []
    real_nonzero = torch.nonzero

    def bounded_nonzero(values, *args, **kwargs):
        nonzero_sizes.append(int(values.numel()))
        return real_nonzero(values, *args, **kwargs)

    monkeypatch.setattr(torch, "nonzero", bounded_nonzero)
    _project(
        kernel,
        image,
        weight,
        wcs,
        _target((height, width), wcs),
        chunk_pixels=chunk_pixels,
    )

    assert nonzero_sizes
    assert max(nonzero_sizes) <= chunk_pixels
    coordinate_multiplier = 1 if kernel == "point" else 4
    assert max(wcs.batch_sizes) <= coordinate_multiplier * chunk_pixels


@pytest.mark.parametrize("kernel", ["square", "conservative"])
def test_extreme_magnification_streams_bounded_candidate_pairs(
    kernel: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    size = 64
    pair_limit = 127
    source_wcs = _IdentityWCS()
    target = _target((size, size), _MagnifyingWCS(size))
    image = torch.tensor([[2.0]])
    weight = torch.tensor([[1.0]])
    observed_batch_sizes: list[int] = []
    real_areas = drizzle_geometry._batch_quads_pixel_areas

    def bounded_areas(quad_x, quad_y, centers):
        observed_batch_sizes.append(int(centers.shape[0]))
        return real_areas(quad_x, quad_y, centers)

    # The candidate-pair batching belongs to the pure-torch overlap path; the native CPU
    # kernel streams per quad without materialising pairs.
    monkeypatch.setattr(drizzle_geometry, "_NATIVE_OVERLAP_ENABLED", False)
    monkeypatch.setattr(
        drizzle_geometry, "_MAX_OVERLAP_CANDIDATE_PAIRS", pair_limit
    )
    monkeypatch.setattr(
        drizzle_geometry, "_batch_quads_pixel_areas", bounded_areas
    )
    numer, projected_weight = _project(
        kernel,
        image,
        weight,
        source_wcs,
        target,
        chunk_pixels=1,
    )

    assert len(observed_batch_sizes) > 1
    assert max(observed_batch_sizes) <= pair_limit
    torch.testing.assert_close(numer.sum(), torch.tensor(2.0), rtol=0.0, atol=1e-6)
    torch.testing.assert_close(
        projected_weight.sum(), torch.tensor(1.0), rtol=0.0, atol=1e-6
    )


@pytest.mark.parametrize("kernel", ["square", "conservative"])
def test_projection_rejects_self_intersecting_footprints(kernel: str) -> None:
    source_wcs = _IdentityWCS()
    with pytest.raises(ValueError, match="self-intersecting"):
        _project(
            kernel,
            torch.ones((1, 1)),
            torch.ones((1, 1)),
            source_wcs,
            _target((3, 3), _BowTieWCS()),
            chunk_pixels=1,
        )


@pytest.mark.parametrize("kernel", ["point", "square", "conservative"])
def test_tile_chunking_preserves_projection_parity(kernel: str) -> None:
    image = torch.arange(35, dtype=torch.float32).reshape(5, 7)
    weight = torch.linspace(0.5, 1.5, 35, dtype=torch.float32).reshape(5, 7)
    source_wcs = _IdentityWCS()
    target = _target((5, 7), _IdentityWCS())

    tiled = _project(
        kernel,
        image,
        weight,
        source_wcs,
        target,
        chunk_pixels=3,
    )
    whole = _project(
        kernel,
        image,
        weight,
        source_wcs,
        target,
        chunk_pixels=image.numel(),
    )

    torch.testing.assert_close(tiled[0], whole[0], rtol=0.0, atol=0.0)
    torch.testing.assert_close(tiled[1], whole[1], rtol=0.0, atol=0.0)


_DEVICES = ["cpu"]
if torch.cuda.is_available():
    _DEVICES.append("cuda")


@pytest.mark.parametrize("device", _DEVICES)
@pytest.mark.parametrize("kernel", ["square", "conservative"])
def test_streamed_polygon_projection_preserves_gradients_and_device(
    kernel: str, device: str
) -> None:
    execution_device = torch.device(device)
    image = torch.arange(
        1, 17, dtype=torch.float32, device=execution_device
    ).reshape(4, 4)
    image.requires_grad_()
    weight = torch.ones_like(image)
    source_wcs = _IdentityWCS()
    numer, projected_weight = _project(
        kernel,
        image,
        weight,
        source_wcs,
        _target((4, 4), _IdentityWCS()),
        chunk_pixels=5,
    )

    assert numer.device.type == execution_device.type
    assert projected_weight.device.type == execution_device.type
    numer.sum().backward()
    assert image.grad is not None
    torch.testing.assert_close(image.grad, torch.ones_like(image))
