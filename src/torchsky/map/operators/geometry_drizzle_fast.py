"""Vectorized square-drizzle geometry helpers.

The reference exact-square drizzle implementation is intentionally simple and
correct, but its inner loop iterates over individual source pixels.  On CUDA that
creates many tiny kernels and synchronizes repeatedly through ``.item()`` calls.

This module keeps the same square-drizzle math, but batches the target-pixel
candidate generation and polygon clipping per chunk so CUDA work is issued as a
few large tensor operations.
"""

from __future__ import annotations

import os
from typing import Any

import torch
from torch import Tensor

from .geometry_drizzle import (
    _DRIZZLE_BOUNDARY_EPS,
    _build_center_pixmap,
    _half_plane_clip_batch,
    _scanner_row_bounds,
    _source_row_vertical_bounds,
    exact_square_drizzle_project,
)


def _move_wcs_to_tensor_device(wcs: Any, device: torch.device) -> Any:
    """Best-effort WCS device alignment for torch-native WCS objects."""
    to_method = getattr(wcs, "to", None)
    if to_method is None:
        return wcs
    try:
        return to_method(device)
    except Exception:
        return wcs


def _quad_areas_for_candidate_pixels(
    quad_x: Tensor,
    quad_y: Tensor,
    centers: Tensor,
) -> Tensor:
    """Area of each per-row quadrilateral clipped to its candidate pixel.

    Parameters
    ----------
    quad_x, quad_y:
        ``(N, 4)`` tensors containing one projected source-pixel quadrilateral
        per candidate row.
    centers:
        ``(N, 2)`` tensor containing target-pixel centers ``(x, y)``.
    """
    if centers.numel() == 0:
        return torch.empty(0, dtype=quad_x.dtype, device=quad_x.device)

    x = quad_x.clone()
    y = quad_y.clone()
    n = torch.full((x.shape[0],), 4, dtype=torch.int64, device=x.device)

    x, y, n = _half_plane_clip_batch(
        x,
        y,
        n,
        axis=0,
        boundaries=centers[:, 0] - 0.5,
        keep_greater=True,
    )
    x, y, n = _half_plane_clip_batch(
        x,
        y,
        n,
        axis=0,
        boundaries=centers[:, 0] + 0.5,
        keep_greater=False,
    )
    x, y, n = _half_plane_clip_batch(
        x,
        y,
        n,
        axis=1,
        boundaries=centers[:, 1] - 0.5,
        keep_greater=True,
    )
    x, y, n = _half_plane_clip_batch(
        x,
        y,
        n,
        axis=1,
        boundaries=centers[:, 1] + 0.5,
        keep_greater=False,
    )

    v_slots = x.shape[1]
    x_next = torch.cat([x[:, 1:], x[:, 0:1]], dim=1)
    y_next = torch.cat([y[:, 1:], y[:, 0:1]], dim=1)
    open_sum = (x[:, : v_slots - 1] * y_next[:, : v_slots - 1]).sum(dim=1) - (
        x_next[:, : v_slots - 1] * y[:, : v_slots - 1]
    ).sum(dim=1)

    last_idx = (n - 1).clamp(min=0).unsqueeze(1)
    x_last = torch.gather(x, 1, last_idx).squeeze(1)
    y_last = torch.gather(y, 1, last_idx).squeeze(1)
    close_sum = x_last * y[:, 0] - x[:, 0] * y_last

    valid = n >= 3
    area = 0.5 * torch.abs(open_sum + close_sum * valid.to(open_sum.dtype))
    return torch.where(valid, area, torch.zeros_like(area))


def _max_candidate_pixels_per_source() -> int:
    raw = os.environ.get("TORCHSKY_DRIZZLE_MAX_CANDIDATES_PER_SOURCE", "128")
    try:
        return max(4, int(raw))
    except ValueError:
        return 128


def _max_candidate_pairs_per_batch() -> int:
    raw = os.environ.get("TORCHSKY_DRIZZLE_MAX_CANDIDATE_PAIRS", "2000000")
    try:
        return max(4096, int(raw))
    except ValueError:
        return 2_000_000


def exact_square_drizzle_project_vectorized(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    drizzle_pixfrac: float,
    pixel_area_sr_map_fn,
    representative_pixel_area_fn,
    chunk_pixels: int,
) -> tuple[Tensor, Tensor]:
    """Vectorized exact-square drizzle projection.

    The public signature matches ``exact_square_drizzle_project``.  The fallback
    is intentionally conservative: if a projection chunk produces a very large
    per-source bounding box, we delegate to the existing scalar implementation
    rather than risk an unexpectedly large temporary tensor.
    """
    image = image.to(dtype=torch.float32)
    source_weight = source_weight.to(dtype=torch.float32, device=image.device)
    device = image.device

    source_wcs = _move_wcs_to_tensor_device(source_wcs, device)
    target_wcs = _move_wcs_to_tensor_device(target.wcs, device)

    h_src, _ = image.shape
    if value_semantics == "flux":
        source_area_sr = pixel_area_sr_map_fn(
            source_wcs,
            tuple(image.shape),
            device=device,
        )
        weighted_image = image * source_weight * source_area_sr
        weighted_weight = source_weight * torch.square(source_area_sr)
    else:
        weighted_image = image * source_weight
        weighted_weight = source_weight

    valid = torch.isfinite(weighted_weight) & (weighted_weight > 0.0)
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=device)
        return zeros, zeros

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=device
    )
    flat_weight = torch.zeros_like(flat_numer)

    width = int(target.shape[1])
    height = int(target.shape[0])
    half = 0.5 * max(float(drizzle_pixfrac), 1.0e-6)

    pixmap = _build_center_pixmap(
        source_wcs,
        target_wcs,
        tuple(image.shape),
        device,
    )
    row_min, row_max = _scanner_row_bounds(
        pixmap,
        source_shape=tuple(image.shape),
        target_shape=tuple(target.shape),
    )

    max_candidates_per_source = _max_candidate_pixels_per_source()
    max_pairs_per_batch = _max_candidate_pairs_per_batch()
    step = max(1, int(chunk_pixels))

    for start in range(0, int(xs.numel()), step):
        end = min(start + step, int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]

        y_index = y_chunk.to(dtype=torch.int64)
        x_index = x_chunk.to(dtype=torch.int64)
        row_min_chunk = row_min[y_index]
        row_max_chunk = row_max[y_index]
        scan_keep = (
            (row_min_chunk >= 0)
            & (x_index >= row_min_chunk)
            & (x_index <= row_max_chunk)
        )
        if not bool(scan_keep.any()):
            continue

        x_chunk = x_chunk[scan_keep]
        y_chunk = y_chunk[scan_keep]
        numer_chunk = numer_chunk[scan_keep]
        weight_chunk = weight_chunk[scan_keep]

        ra_center_deg, dec_center_deg = source_wcs.pixel_to_world(x_chunk, y_chunk)
        x_center_tgt, y_center_tgt = target_wcs.world_to_pixel(
            ra_center_deg,
            dec_center_deg,
        )
        center_finite = torch.isfinite(x_center_tgt) & torch.isfinite(y_center_tgt)
        y_lower, y_upper = _source_row_vertical_bounds(
            y_chunk,
            source_height=h_src,
            target_height=height,
        )
        center_inside = (
            center_finite
            & (x_center_tgt >= -0.5 - _DRIZZLE_BOUNDARY_EPS)
            & (x_center_tgt <= float(width - 1) + 0.5 + _DRIZZLE_BOUNDARY_EPS)
            & (y_center_tgt >= y_lower)
            & (y_center_tgt <= y_upper)
        )
        if not bool(center_inside.any()):
            continue

        x_chunk = x_chunk[center_inside]
        y_chunk = y_chunk[center_inside]
        numer_chunk = numer_chunk[center_inside]
        weight_chunk = weight_chunk[center_inside]

        sx = torch.stack(
            (x_chunk - half, x_chunk + half, x_chunk + half, x_chunk - half), dim=1
        )
        sy = torch.stack(
            (y_chunk - half, y_chunk - half, y_chunk + half, y_chunk + half), dim=1
        )
        ra_deg, dec_deg = source_wcs.pixel_to_world(sx.reshape(-1), sy.reshape(-1))
        x_tgt, y_tgt = target_wcs.world_to_pixel(ra_deg, dec_deg)
        x_tgt = x_tgt.reshape(-1, 4).to(dtype=torch.float64)
        y_tgt = y_tgt.reshape(-1, 4).to(dtype=torch.float64)

        finite = torch.isfinite(x_tgt).all(dim=1) & torch.isfinite(y_tgt).all(dim=1)
        if not bool(finite.any()):
            continue
        x_tgt = x_tgt[finite]
        y_tgt = y_tgt[finite]
        numer_chunk = numer_chunk[finite]
        weight_chunk = weight_chunk[finite]

        quad_area = 0.5 * torch.abs(
            torch.sum(
                x_tgt * torch.roll(y_tgt, shifts=-1, dims=1)
                - torch.roll(x_tgt, shifts=-1, dims=1) * y_tgt,
                dim=1,
            )
        )
        area_keep = quad_area > 0.0
        if not bool(area_keep.any()):
            continue
        x_tgt = x_tgt[area_keep]
        y_tgt = y_tgt[area_keep]
        quad_area = quad_area[area_keep]
        numer_chunk = numer_chunk[area_keep]
        weight_chunk = weight_chunk[area_keep]

        min_x = torch.ceil(x_tgt.min(dim=1).values - 0.5).to(torch.int64).clamp(0, width - 1)
        max_x = torch.floor(x_tgt.max(dim=1).values + 0.5).to(torch.int64).clamp(0, width - 1)
        min_y = torch.ceil(y_tgt.min(dim=1).values - 0.5).to(torch.int64).clamp(0, height - 1)
        max_y = torch.floor(y_tgt.max(dim=1).values + 0.5).to(torch.int64).clamp(0, height - 1)

        bbox_keep = (min_x <= max_x) & (min_y <= max_y)
        if not bool(bbox_keep.any()):
            continue
        x_tgt = x_tgt[bbox_keep]
        y_tgt = y_tgt[bbox_keep]
        quad_area = quad_area[bbox_keep]
        numer_chunk = numer_chunk[bbox_keep]
        weight_chunk = weight_chunk[bbox_keep]
        min_x = min_x[bbox_keep]
        max_x = max_x[bbox_keep]
        min_y = min_y[bbox_keep]
        max_y = max_y[bbox_keep]

        bbox_w = max_x - min_x + 1
        bbox_h = max_y - min_y + 1
        max_w = int(bbox_w.max().item())
        max_h = int(bbox_h.max().item())
        candidates_per_source = max_w * max_h
        if candidates_per_source > max_candidates_per_source:
            return exact_square_drizzle_project(
                image,
                source_weight,
                source_wcs,
                target,
                value_semantics=value_semantics,
                drizzle_pixfrac=drizzle_pixfrac,
                pixel_area_sr_map_fn=pixel_area_sr_map_fn,
                representative_pixel_area_fn=representative_pixel_area_fn,
                chunk_pixels=chunk_pixels,
            )

        sources_per_subbatch = max(1, max_pairs_per_batch // max(candidates_per_source, 1))
        n_src = int(x_tgt.shape[0])
        for sub_start in range(0, n_src, sources_per_subbatch):
            sub_end = min(sub_start + sources_per_subbatch, n_src)
            sl = slice(sub_start, sub_end)
            sub_n = sub_end - sub_start

            dx = torch.arange(max_w, dtype=torch.int64, device=device).view(1, 1, max_w)
            dy = torch.arange(max_h, dtype=torch.int64, device=device).view(1, max_h, 1)
            cx = min_x[sl].view(sub_n, 1, 1) + dx
            cy = min_y[sl].view(sub_n, 1, 1) + dy
            cx_grid = cx.expand(sub_n, max_h, max_w)
            cy_grid = cy.expand(sub_n, max_h, max_w)
            candidate_mask = (
                (dx < bbox_w[sl].view(sub_n, 1, 1))
                & (dy < bbox_h[sl].view(sub_n, 1, 1))
            )

            centers = torch.stack(
                (cx_grid.to(dtype=torch.float64), cy_grid.to(dtype=torch.float64)), dim=-1
            ).reshape(-1, 2)
            candidate_mask_flat = candidate_mask.reshape(-1)
            if not bool(candidate_mask_flat.any()):
                continue

            quad_x_rep = (
                x_tgt[sl]
                .view(sub_n, 1, 1, 4)
                .expand(sub_n, max_h, max_w, 4)
                .reshape(-1, 4)
            )
            quad_y_rep = (
                y_tgt[sl]
                .view(sub_n, 1, 1, 4)
                .expand(sub_n, max_h, max_w, 4)
                .reshape(-1, 4)
            )
            overlap = _quad_areas_for_candidate_pixels(quad_x_rep, quad_y_rep, centers)
            overlap = torch.where(candidate_mask_flat, overlap, torch.zeros_like(overlap))
            valid_overlap = overlap > 0.0
            if not bool(valid_overlap.any()):
                continue

            source_index = (
                torch.arange(sub_n, dtype=torch.int64, device=device)
                .view(sub_n, 1, 1)
                .expand(sub_n, max_h, max_w)
                .reshape(-1)
            )
            cx_flat = cx_grid.reshape(-1)
            cy_flat = cy_grid.reshape(-1)
            out_index = (cy_flat[valid_overlap] * width + cx_flat[valid_overlap]).to(
                dtype=torch.int64
            )
            src_index = source_index[valid_overlap]
            coeff = (overlap[valid_overlap] / quad_area[sl][src_index]).to(
                dtype=torch.float32
            )
            flat_numer.scatter_add_(
                0,
                out_index,
                numer_chunk[sl][src_index] * coeff,
            )
            flat_weight.scatter_add_(
                0,
                out_index,
                weight_chunk[sl][src_index] * coeff,
            )

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    return numer, weight
