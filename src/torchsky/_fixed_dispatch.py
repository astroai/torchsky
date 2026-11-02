from __future__ import annotations

from collections.abc import Callable
import math
from typing import Any

import torch
from torch import Tensor


def _check_nside(nside: int) -> None:
    torch._check(nside > 0, lambda: "nside must be positive")
    torch._check((nside & (nside - 1)) == 0, lambda: "nside must be a power of two")


def _check_pair(first: Tensor, second: Tensor, dtype: torch.dtype, name: str) -> None:
    torch._check(first.shape == second.shape, lambda: f"{name} shapes must match")
    torch._check(first.dtype == dtype, lambda: f"{name} tensors have invalid dtype")
    torch._check(second.dtype == dtype, lambda: f"{name} tensors have invalid dtype")
    torch._check(
        first.device == second.device, lambda: f"{name} tensors must share a device"
    )


def _fake_planar(points_xy: Tensor, polygon_xy: Tensor, eps: float = 1e-15) -> Tensor:
    del eps
    torch._check(points_xy.ndim == 2, lambda: "points_xy must be rank 2")
    torch._check(points_xy.shape[1] == 2, lambda: "points_xy must have shape [N, 2]")
    torch._check(polygon_xy.ndim == 2, lambda: "polygon_xy must be rank 2")
    torch._check(polygon_xy.shape[1] == 2, lambda: "polygon_xy must have shape [N, 2]")
    torch._check(
        polygon_xy.shape[0] >= 3,
        lambda: "polygon_xy must contain at least three vertices",
    )
    torch._check(points_xy.dtype == torch.float64, lambda: "points_xy must be float64")
    torch._check(
        polygon_xy.dtype == torch.float64, lambda: "polygon_xy must be float64"
    )
    torch._check(
        points_xy.device == polygon_xy.device,
        lambda: "geometry tensors must share a device",
    )
    return torch.empty(points_xy.shape[:1], dtype=torch.bool, device=points_xy.device)


def _fake_arcs(
    a0: Tensor,
    a1: Tensor,
    b0: Tensor,
    b1: Tensor,
    tol_rad: float = 1e-10,
) -> Tensor:
    del tol_rad
    for tensor, name in ((a0, "a0"), (a1, "a1"), (b0, "b0"), (b1, "b1")):
        torch._check(tensor.ndim == 2, lambda name=name: f"{name} must be rank 2")
        torch._check(
            tensor.shape[1] == 3, lambda name=name: f"{name} must have shape [N, 3]"
        )
        torch._check(
            tensor.dtype == torch.float64, lambda name=name: f"{name} must be float64"
        )
        torch._check(
            tensor.device == a0.device, lambda: "geometry tensors must share a device"
        )
    torch._check(a0.shape == a1.shape, lambda: "a0 and a1 shapes must match")
    torch._check(b0.shape == b1.shape, lambda: "b0 and b1 shapes must match")
    return torch.empty((a0.shape[0], b0.shape[0]), dtype=torch.bool, device=a0.device)


def _fake_pixels(nside: int, pixels: Tensor) -> Tensor:
    _check_nside(nside)
    torch._check(pixels.dtype == torch.int64, lambda: "pixels must be int64")
    return torch.empty_like(pixels, memory_format=torch.contiguous_format)


def _fake_angles(nside: int, pixels: Tensor) -> tuple[Tensor, Tensor]:
    _check_nside(nside)
    torch._check(pixels.dtype == torch.int64, lambda: "pixels must be int64")
    options = {"dtype": torch.float64, "device": pixels.device}
    return torch.empty(pixels.shape, **options), torch.empty(pixels.shape, **options)


def _fake_ang2pix(nside: int, lon: Tensor, lat: Tensor) -> Tensor:
    _check_nside(nside)
    _check_pair(lon, lat, torch.float64, "longitude/latitude")
    return torch.empty(lon.shape, dtype=torch.int64, device=lon.device)


def _fake_neighbors(nside: int, pixels: Tensor) -> Tensor:
    _check_nside(nside)
    torch._check(pixels.dtype == torch.int64, lambda: "pixels must be int64")
    return torch.empty((*pixels.shape, 8), dtype=torch.int64, device=pixels.device)


def _fake_weights(nside: int, lon: Tensor, lat: Tensor) -> tuple[Tensor, Tensor]:
    _check_nside(nside)
    _check_pair(lon, lat, torch.float64, "longitude/latitude")
    shape = (4, *lon.shape)
    return (
        torch.empty(shape, dtype=torch.int64, device=lon.device),
        torch.empty(shape, dtype=torch.float64, device=lon.device),
    )


def _fake_interp(nside: int, maps: Tensor, lon: Tensor, lat: Tensor) -> Tensor:
    _check_nside(nside)
    _check_pair(lon, lat, torch.float64, "longitude/latitude")
    torch._check(maps.ndim in (1, 2), lambda: "maps must be rank 1 or 2")
    torch._check(maps.is_floating_point(), lambda: "maps must be floating point")
    torch._check(
        maps.device == lon.device, lambda: "maps and coordinates must share a device"
    )
    expected = 12 * nside * nside
    torch._check(
        maps.shape[-1] == expected, lambda: "map length must equal 12*nside*nside"
    )
    shape = lon.shape if maps.ndim == 1 else (maps.shape[0], *lon.shape)
    return torch.empty(shape, dtype=maps.dtype, device=maps.device)


def _fake_query_disc(
    nside: int, vec: Tensor, cos_lim: float, nest: bool = False
) -> Tensor:
    del nest
    _check_nside(nside)
    torch._check(nside <= 1 << 29, lambda: "nside must not exceed 2**29")
    torch._check(vec.dtype == torch.float64, lambda: "vec must be float64")
    torch._check(vec.numel() == 3, lambda: "vec must contain exactly three values")
    torch._check(
        math.isfinite(cos_lim) and -1.0 <= cos_lim <= 1.0,
        lambda: "cos_lim must be finite and in [-1, 1]",
    )
    count = torch.library.get_ctx().new_dynamic_size(min=0, max=12 * nside * nside)
    return torch.empty((count,), dtype=torch.int64, device=vec.device)


def _fake_sparse(
    query: Tensor, block_index: Tensor, block_values: Tensor, fill: Tensor
) -> Tensor:
    torch._check(query.dtype == torch.int64, lambda: "query must be int64")
    torch._check(block_index.dtype == torch.int64, lambda: "block_index must be int64")
    torch._check(block_values.ndim >= 2, lambda: "block_values must be at least rank 2")
    torch._check(fill.numel() == 1, lambda: "fill must contain one element")
    torch._check(
        fill.dtype == block_values.dtype, lambda: "fill dtype must match block_values"
    )
    for tensor in (block_index, block_values, fill):
        torch._check(
            tensor.device == query.device,
            lambda: "sparse lookup tensors must share a device",
        )
    return torch.empty(
        (*block_values.shape[:-2], query.numel()),
        dtype=block_values.dtype,
        device=block_values.device,
    )


def _setup_nondifferentiable(ctx: Any, inputs: tuple[Any, ...], output: Any) -> None:
    del inputs
    tensors = output if isinstance(output, tuple) else (output,)
    ctx.mark_non_differentiable(*tensors)


def _backward_none(ctx: Any, *grad_outputs: Tensor | None) -> tuple[None, ...]:
    del grad_outputs
    return (None,) * ctx.num_inputs


def _register_fake(name: str, implementation: Callable[..., Any]) -> None:
    torch.library.register_fake(f"torchsky::{name}")(implementation)


def _register() -> None:
    fake_implementations: dict[str, Callable[..., Any]] = {
        "sphere_planar_point_in_polygon": _fake_planar,
        "sphere_arc_intersections_matrix": _fake_arcs,
        "healpix_ring2nest": _fake_pixels,
        "healpix_nest2ring": _fake_pixels,
        "healpix_ang2pix_ring": _fake_ang2pix,
        "healpix_ang2pix_nested": _fake_ang2pix,
        "healpix_pix2ang_ring": _fake_angles,
        "healpix_pix2ang_nested": _fake_angles,
        "healpix_neighbors_ring": _fake_neighbors,
        "healpix_neighbors_nested": _fake_neighbors,
        "healpix_get_interp_weights_ring": _fake_weights,
        "healpix_get_interp_weights_nested": _fake_weights,
        "healpix_get_interp_val_ring": _fake_interp,
        "healpix_get_interp_val_nested": _fake_interp,
        "healpix_query_disc": _fake_query_disc,
        "sparse_get_values_blocks": _fake_sparse,
    }
    for name, implementation in fake_implementations.items():
        _register_fake(name, implementation)
        schema = getattr(torch.ops.torchsky, name).default._schema

        def setup_context(
            ctx: Any,
            inputs: tuple[Any, ...],
            output: Any,
            num_inputs: int = len(schema.arguments),
        ) -> None:
            ctx.num_inputs = num_inputs
            _setup_nondifferentiable(ctx, inputs, output)

        torch.library.register_autograd(
            f"torchsky::{name}",
            _backward_none,
            setup_context=setup_context,
        )


_register()
