"""Conservative overlap geometry helpers."""

from __future__ import annotations

import torch
from torch import Tensor

from .geometry import ConservativePolygonRemapOp
from .geometry_drizzle import _batch_quad_pixel_areas


def _exact_conservative_polygon_project(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
) -> tuple[Tensor, Tensor]:
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
        return zeros, zeros

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    width = int(target.shape[1])
    height = int(target.shape[0])
    half = 0.5

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

        for quad_x, quad_y, c_numer, c_weight in zip(
            x_tgt.unbind(0),
            y_tgt.unbind(0),
            numer_chunk.unbind(0),
            weight_chunk.unbind(0),
            strict=False,
        ):
            area = 0.5 * torch.abs(
                torch.sum(
                    quad_x * torch.roll(quad_y, shifts=-1)
                    - torch.roll(quad_x, shifts=-1) * quad_y
                )
            )
            if not (area > 0.0):
                continue
            min_x = max(int(torch.ceil(quad_x.min() - 0.5).item()), 0)
            max_x = min(int(torch.floor(quad_x.max() + 0.5).item()), width - 1)
            min_y = max(int(torch.ceil(quad_y.min() - 0.5).item()), 0)
            max_y = min(int(torch.floor(quad_y.max() + 0.5).item()), height - 1)
            if min_x > max_x or min_y > max_y:
                continue
            cy = torch.arange(min_y, max_y + 1, dtype=torch.float64, device=image.device)
            cx = torch.arange(min_x, max_x + 1, dtype=torch.float64, device=image.device)
            gy, gx = torch.meshgrid(cy, cx, indexing="ij")
            centers = torch.stack((gx.reshape(-1), gy.reshape(-1)), dim=-1)
            overlaps = _batch_quad_pixel_areas(quad_x, quad_y, centers)
            valid = overlaps > 0.0
            if not valid.any():
                continue
            vc = centers[valid]
            vo = overlaps[valid]
            i_idx = (vc[:, 1].to(dtype=torch.int64) * width + vc[:, 0].to(dtype=torch.int64)).to(dtype=torch.int64)
            c_coeff = (vo / area).to(dtype=torch.float32)
            flat_numer.scatter_add_(0, i_idx, c_numer * c_coeff)
            flat_weight.scatter_add_(0, i_idx, c_weight * c_coeff)

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    return numer, weight


def conservative_polygon_project(
    image,
    source_weight,
    source_wcs,
    target,
    *,
    value_semantics: str,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
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
        pixel_area_sr_map_fn=pixel_area_sr_map_fn,
        representative_pixel_area_fn=representative_pixel_area_fn,
        chunk_pixels=chunk_pixels,
    )


__all__ = ["ConservativePolygonRemapOp", "conservative_polygon_project"]
