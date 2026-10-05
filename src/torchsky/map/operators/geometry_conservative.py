"""Conservative overlap geometry helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import ConservativePolygonRemapOp
from .geometry_drizzle import _batch_quads_pixel_areas


def _exact_conservative_polygon_project(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    drizzle_pixfrac: float = 1.0,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
    return_variance_weight: bool = False,
) -> tuple[Tensor, Tensor] | tuple[Tensor, Tensor, Tensor]:
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

    valid = torch.isfinite(weighted_weight) & (weighted_weight > 0.0)
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=image.device)
        if return_variance_weight:
            return zeros, zeros, zeros
        return zeros, zeros

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
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

    for start in range(0, int(xs.numel()), int(chunk_pixels)):
        end = min(start + int(chunk_pixels), int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]

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

        for quad_start in range(0, int(x_tgt.shape[0]), 32_768):
            quad_end = min(quad_start + 32_768, int(x_tgt.shape[0]))
            quad_x = x_tgt[quad_start:quad_end]
            quad_y = y_tgt[quad_start:quad_end]
            quad_numer = numer_chunk[quad_start:quad_end]
            quad_weight = weight_chunk[quad_start:quad_end]
            areas = 0.5 * torch.abs(
                torch.sum(
                    quad_x * torch.roll(quad_y, shifts=-1, dims=1)
                    - torch.roll(quad_x, shifts=-1, dims=1) * quad_y,
                    dim=1,
                )
            )
            min_x = torch.ceil(quad_x.min(dim=1).values - 0.5).to(torch.int64)
            max_x = torch.floor(quad_x.max(dim=1).values + 0.5).to(torch.int64)
            min_y = torch.ceil(quad_y.min(dim=1).values - 0.5).to(torch.int64)
            max_y = torch.floor(quad_y.max(dim=1).values + 0.5).to(torch.int64)
            min_x.clamp_(min=0)
            max_x.clamp_(max=width - 1)
            min_y.clamp_(min=0)
            max_y.clamp_(max=height - 1)
            nx = (max_x - min_x + 1).clamp(min=0)
            ny = (max_y - min_y + 1).clamp(min=0)
            counts = torch.where(areas > 0.0, nx * ny, torch.zeros_like(nx))
            pair_count = int(counts.sum().item())
            if pair_count == 0:
                continue
            owners = torch.repeat_interleave(
                torch.arange(quad_x.shape[0], device=image.device), counts
            )
            starts = torch.cumsum(counts, dim=0) - counts
            local = torch.arange(
                pair_count, device=image.device
            ) - torch.repeat_interleave(starts, counts)
            owner_nx = nx[owners]
            center_x = min_x[owners] + torch.remainder(local, owner_nx)
            center_y = min_y[owners] + torch.div(
                local, owner_nx, rounding_mode="floor"
            )
            centers = torch.stack((center_x, center_y), dim=1).to(torch.float64)
            overlaps = _batch_quads_pixel_areas(
                quad_x[owners], quad_y[owners], centers
            )
            overlap_valid = overlaps > 0.0
            if not bool(overlap_valid.any()):
                continue
            owners = owners[overlap_valid]
            center_x = center_x[overlap_valid]
            center_y = center_y[overlap_valid]
            output_index = center_y * width + center_x
            coefficients = (overlaps[overlap_valid] / areas[owners]).to(
                dtype=torch.float32
            )
            flat_numer.scatter_add_(
                0, output_index, quad_numer[owners] * coefficients
            )
            flat_weight.scatter_add_(
                0, output_index, quad_weight[owners] * coefficients
            )
            if flat_variance_weight is not None:
                flat_variance_weight.scatter_add_(
                    0,
                    output_index,
                    quad_weight[owners] * torch.square(coefficients),
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
    representative_pixel_area_fn,
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
        representative_pixel_area_fn=representative_pixel_area_fn,
        chunk_pixels=chunk_pixels,
        return_variance_weight=return_variance_weight,
    )


__all__ = ["ConservativePolygonRemapOp", "conservative_polygon_project"]
