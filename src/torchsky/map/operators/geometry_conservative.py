"""Conservative overlap geometry helpers."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .geometry import ConservativePolygonRemapOp
from .geometry_drizzle import _clip_polygon_to_pixel, _polygon_area


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
            tuple(int(v) for v in image.shape),
            device=image.device,
        )
        source_representative_area_sr = float(
            representative_pixel_area_fn(
                source_wcs,
                tuple(int(v) for v in image.shape),
                device=image.device,
            )
        )
        target_representative_area_sr = (
            float(target.representative_pixel_area_sr)
            if target.representative_pixel_area_sr is not None
            else float(
                representative_pixel_area_fn(
                    target.wcs,
                    tuple(int(v) for v in target.shape),
                    device=image.device,
                )
            )
        )
        flux_scale = (
            target_representative_area_sr / max(source_representative_area_sr, 1.0e-16)
        ) ** 2
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * torch.square(source_area_sr)
        weighted_image = weighted_image * float(flux_scale)
        weighted_weight = weighted_weight * float(flux_scale)
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

        for idx in range(int(x_tgt.shape[0])):
            quad = [
                (float(x_tgt[idx, corner].item()), float(y_tgt[idx, corner].item()))
                for corner in range(4)
            ]
            area = _polygon_area(quad)
            if not math.isfinite(area) or area <= 0.0:
                continue
            min_x = max(int(math.ceil(min(point[0] for point in quad) - 0.5)), 0)
            max_x = min(int(math.floor(max(point[0] for point in quad) + 0.5)), width - 1)
            min_y = max(int(math.ceil(min(point[1] for point in quad) - 0.5)), 0)
            max_y = min(int(math.floor(max(point[1] for point in quad) + 0.5)), height - 1)
            if min_x > max_x or min_y > max_y:
                continue
            idxs: list[int] = []
            coeffs: list[float] = []
            for iy in range(min_y, max_y + 1):
                for ix in range(min_x, max_x + 1):
                    clipped = _clip_polygon_to_pixel(quad, x_center=ix, y_center=iy)
                    overlap = _polygon_area(clipped)
                    if overlap <= 0.0:
                        continue
                    idxs.append(iy * width + ix)
                    coeffs.append(float(overlap / area))
            if not idxs:
                continue
            idx_tensor = torch.tensor(idxs, dtype=torch.int64, device=image.device)
            coeff_tensor = torch.tensor(coeffs, dtype=torch.float32, device=image.device)
            flat_numer.scatter_add_(0, idx_tensor, numer_chunk[idx] * coeff_tensor)
            flat_weight.scatter_add_(0, idx_tensor, weight_chunk[idx] * coeff_tensor)

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
