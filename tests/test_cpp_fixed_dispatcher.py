from __future__ import annotations

import pytest
import torch
from collections.abc import Callable
from torch._subclasses.fake_tensor import FakeTensorMode, is_fake

from torchsky import _ops
from torchsky._native import cpp as _cpp
from torchsky.sphere import geom, healpix
from torchsky.sphere.sparse import SparseHealpixMap

pytestmark = pytest.mark.skipif(_cpp is None, reason="native extension unavailable")


def test_fixed_dispatcher_schemas_are_clean() -> None:
    names = (
        "sphere_planar_point_in_polygon",
        "sphere_arc_intersections_matrix",
        "healpix_ring2nest",
        "healpix_nest2ring",
        "healpix_ang2pix_ring",
        "healpix_ang2pix_nested",
        "healpix_pix2ang_ring",
        "healpix_pix2ang_nested",
        "healpix_neighbors_ring",
        "healpix_neighbors_nested",
        "healpix_get_interp_weights_ring",
        "healpix_get_interp_weights_nested",
        "healpix_get_interp_val_ring",
        "healpix_get_interp_val_nested",
        "sparse_get_values_blocks",
    )
    assert _ops.has_fixed_dispatcher()
    for name in names:
        schema = str(getattr(torch.ops.torchsky, name).default._schema)
        assert schema.startswith(f"torchsky::{name}(")
        assert "_cpu" not in schema
        assert torch._C._dispatch_has_kernel_for_dispatch_key(
            f"torchsky::{name}", "CPU"
        )
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"torchsky::{name}", "CUDA"
        )
        assert not torch._C._dispatch_has_kernel_for_dispatch_key(
            f"torchsky::{name}", "MPS"
        )


def test_geometry_dispatcher_matches_legacy_adapters() -> None:
    polygon = torch.tensor(
        [[0.0, 0.0], [2.0, 0.0], [2.0, 2.0], [0.0, 2.0]],
        dtype=torch.float64,
    )
    points = torch.tensor([[1.0, 1.0], [3.0, 1.0]], dtype=torch.float64)
    actual = torch.ops.torchsky.sphere_planar_point_in_polygon(points, polygon, 1e-15)
    expected = _cpp._sphere_planar_point_in_polygon_cpu(points, polygon, 1e-15)
    torch.testing.assert_close(actual, expected)

    a0 = torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float64)
    a1 = torch.tensor([[0.0, 1.0, 0.0]], dtype=torch.float64)
    b0 = torch.tensor([[2.0**-0.5, 2.0**-0.5, -1.0]], dtype=torch.float64)
    b0 = b0 / torch.linalg.norm(b0, dim=-1, keepdim=True)
    b1 = b0.clone()
    b1[:, 2] *= -1
    actual_arcs = torch.ops.torchsky.sphere_arc_intersections_matrix(
        a0, a1, b0, b1, 1e-10
    )
    expected_arcs = _cpp._sphere_arc_intersections_matrix_cpu(a0, a1, b0, b1, 1e-10)
    torch.testing.assert_close(actual_arcs, expected_arcs)


@pytest.mark.parametrize("nested", [False, True])
def test_healpix_dispatcher_matches_legacy_adapters(nested: bool) -> None:
    nside = 4
    pixels = torch.tensor([[0, 1], [23, 47]], dtype=torch.int64)
    lon = torch.tensor([[0.0, 45.0], [123.0, 359.0]], dtype=torch.float64)
    lat = torch.tensor([[0.0, 30.0], [-45.0, 80.0]], dtype=torch.float64)
    suffix = "nested" if nested else "ring"

    ang_op = getattr(torch.ops.torchsky, f"healpix_ang2pix_{suffix}")
    pix_op = getattr(torch.ops.torchsky, f"healpix_pix2ang_{suffix}")
    neighbor_op = getattr(torch.ops.torchsky, f"healpix_neighbors_{suffix}")
    weights_op = getattr(torch.ops.torchsky, f"healpix_get_interp_weights_{suffix}")
    values_op = getattr(torch.ops.torchsky, f"healpix_get_interp_val_{suffix}")
    legacy_ang = getattr(_cpp, f"healpix_ang2pix_{suffix}_torch_cpu")
    legacy_pix = getattr(_cpp, f"healpix_pix2ang_{suffix}_torch_cpu")
    legacy_neighbors = getattr(_cpp, f"healpix_neighbors_{suffix}_torch_cpu")
    legacy_weights = getattr(_cpp, f"healpix_get_interp_weights_{suffix}_torch_cpu")
    legacy_values = getattr(_cpp, f"healpix_get_interp_val_{suffix}_torch_cpu")

    torch.testing.assert_close(ang_op(nside, lon, lat), legacy_ang(nside, lon, lat))
    actual_lon, actual_lat = pix_op(nside, pixels)
    expected_lon, expected_lat = legacy_pix(nside, pixels)
    torch.testing.assert_close(actual_lon, expected_lon)
    torch.testing.assert_close(actual_lat, expected_lat)
    torch.testing.assert_close(
        neighbor_op(nside, pixels), legacy_neighbors(nside, pixels)
    )
    actual_pixels, actual_weights = weights_op(nside, lon, lat)
    expected_pixels, expected_weights = legacy_weights(nside, lon, lat)
    torch.testing.assert_close(actual_pixels, expected_pixels)
    torch.testing.assert_close(actual_weights, expected_weights)

    maps = torch.arange(12 * nside * nside, dtype=torch.float64).reshape(1, -1)
    torch.testing.assert_close(
        values_op(nside, maps, lon, lat), legacy_values(nside, maps, lon, lat)
    )

    conversion = "nest2ring" if nested else "ring2nest"
    legacy_conversion = getattr(_cpp, f"healpix_{conversion}_torch_cpu")
    torch.testing.assert_close(
        getattr(torch.ops.torchsky, f"healpix_{conversion}")(nside, pixels),
        legacy_conversion(nside, pixels),
    )


def test_sparse_dispatcher_matches_legacy_adapter() -> None:
    query = torch.tensor([-1, 0, 3, 4, 99], dtype=torch.int64)
    index = torch.tensor([1, 0], dtype=torch.int64)
    blocks = torch.tensor([[[-1.0, -1.0, -1.0, -1.0], [10.0, 11.0, 12.0, 13.0]]])
    fill = torch.tensor(-1.0)
    actual = torch.ops.torchsky.sparse_get_values_blocks(query, index, blocks, fill)
    expected = _cpp._sparse_get_values_blocks_cpu(query, index, blocks, fill)
    torch.testing.assert_close(actual, expected)


def test_dispatcher_outputs_do_not_alias_inputs() -> None:
    pixels = torch.arange(12, dtype=torch.int64)
    converted = torch.ops.torchsky.healpix_ring2nest(1, pixels)
    assert converted.untyped_storage().data_ptr() != pixels.untyped_storage().data_ptr()

    lon = torch.tensor([0.0, 90.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 45.0], dtype=torch.float64)
    _, weights = torch.ops.torchsky.healpix_get_interp_weights_ring(1, lon, lat)
    assert weights.untyped_storage().data_ptr() != lon.untyped_storage().data_ptr()
    assert weights.untyped_storage().data_ptr() != lat.untyped_storage().data_ptr()


@pytest.mark.parametrize(
    ("call", "match"),
    [
        (
            lambda: torch.ops.torchsky.healpix_ang2pix_ring(
                1, torch.zeros(2), torch.zeros(2)
            ),
            "float64",
        ),
        (
            lambda: torch.ops.torchsky.healpix_ring2nest(
                1, torch.zeros(2, dtype=torch.int32)
            ),
            "int64",
        ),
        (
            lambda: torch.ops.torchsky.sphere_planar_point_in_polygon(
                torch.zeros((2, 3), dtype=torch.float64),
                torch.zeros((3, 2), dtype=torch.float64),
            ),
            "shape",
        ),
        (
            lambda: torch.ops.torchsky.healpix_get_interp_val_ring(
                1,
                torch.zeros(11, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64),
                torch.zeros(1, dtype=torch.float64),
            ),
            "12.nside.nside",
        ),
        (
            lambda: torch.ops.torchsky.sparse_get_values_blocks(
                torch.zeros(1, dtype=torch.int64),
                torch.zeros(1, dtype=torch.int64),
                torch.zeros((1, 1), dtype=torch.float32),
                torch.zeros(1, dtype=torch.float64),
            ),
            "dtype",
        ),
    ],
)
def test_dispatcher_rejects_malformed_contracts(
    call: Callable[[], object], match: str
) -> None:
    with pytest.raises((RuntimeError, ValueError), match=match):
        call()


def test_fixed_dispatcher_fake_tensor_shapes() -> None:
    mode = FakeTensorMode()
    with mode:
        lon = torch.empty((2, 3), dtype=torch.float64)
        lat = torch.empty((2, 3), dtype=torch.float64)
        pixels, weights = torch.ops.torchsky.healpix_get_interp_weights_nested(
            4, lon, lat
        )
        neighbors = torch.ops.torchsky.healpix_neighbors_ring(
            4, torch.empty((2, 3), dtype=torch.int64)
        )
        maps = torch.empty((5, 192), dtype=torch.float32)
        values = torch.ops.torchsky.healpix_get_interp_val_ring(4, maps, lon, lat)
        sparse = torch.ops.torchsky.sparse_get_values_blocks(
            torch.empty((2, 3), dtype=torch.int64),
            torch.empty(12, dtype=torch.int64),
            torch.empty((5, 4, 16), dtype=torch.float32),
            torch.empty((), dtype=torch.float32),
        )
        contained = torch.ops.torchsky.sphere_planar_point_in_polygon(
            torch.empty((7, 2), dtype=torch.float64),
            torch.empty((3, 2), dtype=torch.float64),
        )
        intersections = torch.ops.torchsky.sphere_arc_intersections_matrix(
            torch.empty((2, 3), dtype=torch.float64),
            torch.empty((2, 3), dtype=torch.float64),
            torch.empty((4, 3), dtype=torch.float64),
            torch.empty((4, 3), dtype=torch.float64),
        )
        converted = torch.ops.torchsky.healpix_ring2nest(
            4, torch.empty((2, 3), dtype=torch.int64)
        )
        out_lon, out_lat = torch.ops.torchsky.healpix_pix2ang_nested(
            4, torch.empty((2, 3), dtype=torch.int64)
        )
    assert is_fake(pixels) and pixels.shape == (4, 2, 3)
    assert is_fake(weights) and weights.shape == (4, 2, 3)
    assert is_fake(neighbors) and neighbors.shape == (2, 3, 8)
    assert is_fake(values) and values.shape == (5, 2, 3)
    assert is_fake(sparse) and sparse.shape == (5, 6)
    assert is_fake(contained) and contained.shape == (7,)
    assert is_fake(intersections) and intersections.shape == (2, 4)
    assert is_fake(converted) and converted.shape == (2, 3)
    assert is_fake(out_lon) and out_lon.shape == (2, 3)
    assert is_fake(out_lat) and out_lat.shape == (2, 3)


def test_discrete_and_native_interpolation_outputs_are_nondifferentiable() -> None:
    lon = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    lat = torch.tensor([0.0, 5.0], dtype=torch.float64, requires_grad=True)
    pixels = torch.ops.torchsky.healpix_ang2pix_ring(2, lon, lat)
    _, weights = torch.ops.torchsky.healpix_get_interp_weights_ring(2, lon, lat)
    maps = torch.arange(48, dtype=torch.float64, requires_grad=True)
    values = torch.ops.torchsky.healpix_get_interp_val_ring(2, maps, lon, lat)
    assert not pixels.requires_grad
    assert not weights.requires_grad
    assert not values.requires_grad


def test_public_interpolation_keeps_pure_torch_gradients() -> None:
    lon = torch.tensor([10.0, 20.0], dtype=torch.float64, requires_grad=True)
    lat = torch.tensor([0.0, 5.0], dtype=torch.float64, requires_grad=True)
    _, weights = healpix.get_interp_weights(2, lon, lat, lonlat=True)
    weights.sum().backward()
    assert lon.grad is not None
    assert lat.grad is not None

    maps = torch.arange(48, dtype=torch.float64, requires_grad=True)
    values = healpix.get_interp_val(maps, lon.detach(), lat.detach(), lonlat=True)
    values.sum().backward()
    assert maps.grad is not None


def test_public_fixed_shape_paths_use_dispatcher(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: set[str] = set()
    original = _ops._fixed

    def recording(name: str, *args: object) -> object:
        calls.add(name)
        return original(name, *args)

    monkeypatch.setattr(_ops, "_fixed", recording)
    healpix.ring2nest(1, torch.tensor([0], dtype=torch.int64))
    healpix.neighbors(1, torch.tensor([0], dtype=torch.int64))
    geom._planar_point_in_polygon(
        torch.zeros((500_000, 2), dtype=torch.float64),
        torch.tensor(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 0.0]], dtype=torch.float64
        ),
    )
    sparse = SparseHealpixMap.from_pixels(
        torch.tensor([0], dtype=torch.int64),
        torch.tensor([1.0]),
        2,
        nside_coverage=1,
    )
    sparse.get_values(torch.tensor([0], dtype=torch.int64))
    assert {
        "healpix_ring2nest",
        "healpix_neighbors_ring",
        "sphere_planar_point_in_polygon",
        "sparse_get_values_blocks",
    } <= calls


def test_fixed_dispatcher_opcheck() -> None:
    lon = torch.tensor([0.0, 30.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 45.0], dtype=torch.float64)
    torch.library.opcheck(
        torch.ops.torchsky.healpix_get_interp_weights_ring.default,
        (2, lon, lat),
    )
    torch.library.opcheck(
        torch.ops.torchsky.sphere_planar_point_in_polygon.default,
        (
            torch.tensor([[0.2, 0.2]], dtype=torch.float64),
            torch.tensor([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        ),
    )
    torch.library.opcheck(
        torch.ops.torchsky.sparse_get_values_blocks.default,
        (
            torch.tensor([0, 1], dtype=torch.int64),
            torch.tensor([0], dtype=torch.int64),
            torch.zeros((1, 1), dtype=torch.float32),
            torch.tensor(0.0, dtype=torch.float32),
        ),
    )


def test_fixed_dispatcher_compile() -> None:
    def operation(lon: torch.Tensor, lat: torch.Tensor) -> torch.Tensor:
        pixels, weights = torch.ops.torchsky.healpix_get_interp_weights_ring(
            2, lon, lat
        )
        return weights + pixels.to(weights.dtype) * 0.0

    compiled = torch.compile(operation, fullgraph=True, backend="aot_eager")
    lon = torch.tensor([0.0, 30.0], dtype=torch.float64)
    lat = torch.tensor([0.0, 45.0], dtype=torch.float64)
    torch.testing.assert_close(compiled(lon, lat), operation(lon, lat))
