"""Conservative overlap geometry helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from ...device import require_float64
from .geometry import ConservativePolygonRemapOp
from .geometry_drizzle import (
    _accumulate_quadrilateral_overlaps,
    _iter_valid_source_pixel_chunks,
)


def _exact_conservative_polygon_project(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    drizzle_pixfrac: float = 1.0,
    pixel_area_sr_map_fn,
    chunk_pixels: int,
    return_variance_weight: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
    require_float64(image.device, what="exact conservative polygon projection")
    image = image.to(dtype=torch.float32)
    source_weight = source_weight.to(dtype=torch.float32)
    if value_semantics == "flux":
        source_area_sr = pixel_area_sr_map_fn(
            source_wcs,
            tuple(image.shape),
            device=image.device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * torch.square(source_area_sr)
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight

    valid = (
        torch.isfinite(weighted_weight)
        & torch.isfinite(weighted_image)
        & (weighted_weight > 0.0)
    )
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=image.device)
        if return_variance_weight:
            return zeros, zeros, zeros
        return zeros, zeros

    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    flat_variance_weight = (
        torch.zeros_like(flat_numer) if return_variance_weight else None
    )
    width = int(target.shape[1])
    height = int(target.shape[0])
    half = 0.5 * max(float(drizzle_pixfrac), 1e-6)

    flat_weighted_image = weighted_image.reshape(-1)
    flat_weighted_weight = weighted_weight.reshape(-1)
    for flat_index, source_x, source_y in _iter_valid_source_pixel_chunks(
        valid, chunk_pixels
    ):
        x_chunk = source_x.to(dtype=torch.float64)
        y_chunk = source_y.to(dtype=torch.float64)
        numer_chunk = flat_weighted_image[flat_index]
        weight_chunk = flat_weighted_weight[flat_index]
        sx = torch.stack(
            (x_chunk - half, x_chunk + half, x_chunk + half, x_chunk - half),
            dim=1,
        )
        sy = torch.stack(
            (y_chunk - half, y_chunk - half, y_chunk + half, y_chunk + half),
            dim=1,
        )
        ra_deg, dec_deg = source_wcs.pixel_to_world(sx.reshape(-1), sy.reshape(-1))
        x_tgt, y_tgt = target.wcs.world_to_pixel(ra_deg, dec_deg)
        x_tgt = x_tgt.reshape(-1, 4).to(dtype=torch.float64)
        y_tgt = y_tgt.reshape(-1, 4).to(dtype=torch.float64)
        finite = torch.isfinite(x_tgt).all(dim=1) & torch.isfinite(y_tgt).all(dim=1)
        if not bool(finite.any()):
            continue

        x_tgt = x_tgt[finite]
        y_tgt = y_tgt[finite]
        numer_chunk = numer_chunk[finite]
        weight_chunk = weight_chunk[finite]
        _accumulate_quadrilateral_overlaps(
            x_tgt,
            y_tgt,
            numer_chunk,
            weight_chunk,
            width=width,
            height=height,
            flat_numer=flat_numer,
            flat_weight=flat_weight,
            flat_variance_weight=flat_variance_weight,
        )

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    if return_variance_weight:
        assert flat_variance_weight is not None
        variance_weight = flat_variance_weight.reshape(target.shape)
        variance_weight = torch.where(
            weight > 0.0,
            variance_weight,
            torch.zeros_like(variance_weight),
        )
        return numer, weight, variance_weight
    return numer, weight


def conservative_polygon_project(
    image,
    source_weight,
    source_wcs,
    target,
    *,
    value_semantics: str,
    drizzle_pixfrac: float = 1.0,
    pixel_area_sr_map_fn,
    chunk_pixels: int,
    return_variance_weight: bool = False,
):
    """Project full source-pixel footprints onto the target grid.

    This is the native conservative-overlap substrate: full source-pixel
    footprints are projected exactly onto the target grid without drizzle's
    pixmap-domain scanner gating.
    """
    return _exact_conservative_polygon_project(
        image,
        source_weight,
        source_wcs,
        target,
        value_semantics=value_semantics,
        drizzle_pixfrac=drizzle_pixfrac,
        pixel_area_sr_map_fn=pixel_area_sr_map_fn,
        chunk_pixels=chunk_pixels,
        return_variance_weight=return_variance_weight,
    )


__all__ = ["ConservativePolygonRemapOp", "conservative_polygon_project"]
