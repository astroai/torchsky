"""Drizzle and splat geometry helpers."""

from __future__ import annotations

import torch
from torch import Tensor

_DRIZZLE_BOUNDARY_EPS = 1.0e-4
_MAX_OVERLAP_CANDIDATE_PAIRS = 65_536


def _iter_valid_source_pixel_chunks(valid: Tensor, chunk_pixels: int):
    chunk_size = int(chunk_pixels)
    if chunk_size <= 0:
        raise ValueError("chunk_pixels must be positive")
    width = int(valid.shape[1])
    flat_valid = valid.reshape(-1)
    for start in range(0, int(flat_valid.numel()), chunk_size):
        stop = min(start + chunk_size, int(flat_valid.numel()))
        local = torch.nonzero(flat_valid[start:stop], as_tuple=False).flatten()
        if local.numel() == 0:
            continue
        flat_index = local + start
        source_y = torch.div(flat_index, width, rounding_mode="floor")
        source_x = torch.remainder(flat_index, width)
        yield flat_index, source_x, source_y


def _source_row_vertical_bounds(
    source_y: Tensor,
    *,
    source_height: int,
    target_height: int,
) -> tuple[Tensor, Tensor]:
    """Point-drizzle center bounds preserving the established outer-edge contract."""
    lower = torch.full_like(source_y, -_DRIZZLE_BOUNDARY_EPS)
    upper = torch.full_like(
        source_y,
        float(target_height - 1) + 0.5 + _DRIZZLE_BOUNDARY_EPS,
    )
    upper = torch.where(
        source_y >= float(source_height - 1),
        float(target_height - 1) + _DRIZZLE_BOUNDARY_EPS,
        upper,
    )
    return lower, upper


def _half_plane_clip_batch(
    x: Tensor,
    y: Tensor,
    n: Tensor,
    axis: int,
    boundaries: Tensor,
    keep_greater: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    M, V = x.shape
    device = x.device

    nan_scalar = torch.tensor(float("nan"), device=device)
    Vn_slots = 2 * V
    slots = torch.full((M, Vn_slots, 2), float("nan"), dtype=x.dtype, device=device)
    idx_m = torch.arange(M, device=device)

    def inside(vals: Tensor, bnd: Tensor) -> Tensor:
        return vals >= bnd if keep_greater else vals <= bnd

    bnd = boundaries

    last_idx = (n - 1).clamp(min=0)
    e_valid = n > 0
    px = x[idx_m, last_idx]
    py = y[idx_m, last_idx]
    cx_ = x[:, 0]
    cy_ = y[:, 0]

    pv = px if axis == 0 else py
    cv = cx_ if axis == 0 else cy_
    pi = inside(pv, bnd)
    ci = inside(cv, bnd)

    dv = cv - pv
    dv_s = torch.where(torch.abs(dv) < 1e-12, torch.ones_like(dv), dv)
    t = torch.where(torch.abs(dv) < 1e-12, torch.zeros_like(dv), (bnd - pv) / dv_s)
    xi_ = px + t * (cx_ - px)
    yi_ = py + t * (cy_ - py)

    add_both = ci & ~pi
    add_curr = ci & pi
    add_int = ~ci & pi

    s0_mask = (add_both | add_int) & e_valid
    s1_mask = (add_both | add_curr) & e_valid
    slots[:, 0, 0] = torch.where(s0_mask, xi_, nan_scalar)
    slots[:, 0, 1] = torch.where(s0_mask, yi_, nan_scalar)
    slots[:, 1, 0] = torch.where(s1_mask, cx_, nan_scalar)
    slots[:, 1, 1] = torch.where(s1_mask, cy_, nan_scalar)

    for e in range(1, V):
        px = x[:, e - 1]
        py = y[:, e - 1]
        cx_ = x[:, e]
        cy_ = y[:, e]

        pv = px if axis == 0 else py
        cv = cx_ if axis == 0 else cy_
        pi = inside(pv, bnd)
        ci = inside(cv, bnd)
        e_valid = e < n

        dv = cv - pv
        dv_s = torch.where(torch.abs(dv) < 1e-12, torch.ones_like(dv), dv)
        t = torch.where(torch.abs(dv) < 1e-12, torch.zeros_like(dv), (bnd - pv) / dv_s)
        xi_ = px + t * (cx_ - px)
        yi_ = py + t * (cy_ - py)

        add_both = ci & ~pi
        add_curr = ci & pi
        add_int = ~ci & pi

        s0_mask = (add_both | add_int) & e_valid
        s1_mask = (add_both | add_curr) & e_valid
        s0 = 2 * e
        s1 = 2 * e + 1
        slots[:, s0, 0] = torch.where(s0_mask, xi_, nan_scalar)
        slots[:, s0, 1] = torch.where(s0_mask, yi_, nan_scalar)
        slots[:, s1, 0] = torch.where(s1_mask, cx_, nan_scalar)
        slots[:, s1, 1] = torch.where(s1_mask, cy_, nan_scalar)

    is_nan = torch.isnan(slots[..., 0])
    count = Vn_slots - is_nan.sum(dim=-1)
    n_out = count.clamp(max=V + 1)
    order = torch.argsort(is_nan.to(torch.int64), dim=-1, stable=True)
    x_all = torch.gather(slots[..., 0], 1, order)
    y_all = torch.gather(slots[..., 1], 1, order)
    x_out = x_all[:, : V + 1]
    y_out = y_all[:, : V + 1]
    x_out = torch.nan_to_num(x_out, 0.0)
    y_out = torch.nan_to_num(y_out, 0.0)

    return x_out, y_out, n_out


def _batch_quads_pixel_areas(
    quad_x: Tensor,
    quad_y: Tensor,
    centers: Tensor,
) -> Tensor:
    """Clip one quadrilateral per target-pixel centre in one tensor batch."""
    M = centers.shape[0]
    if M == 0:
        return torch.empty(0, dtype=quad_x.dtype, device=quad_x.device)

    x = quad_x.clone()
    y = quad_y.clone()
    n = torch.full((M,), 4, dtype=torch.int64, device=quad_x.device)

    b = centers[:, 0] - 0.5
    x, y, n = _half_plane_clip_batch(x, y, n, 0, b, keep_greater=True)

    b = centers[:, 0] + 0.5
    x, y, n = _half_plane_clip_batch(x, y, n, 0, b, keep_greater=False)

    b = centers[:, 1] - 0.5
    x, y, n = _half_plane_clip_batch(x, y, n, 1, b, keep_greater=True)

    b = centers[:, 1] + 0.5
    x, y, n = _half_plane_clip_batch(x, y, n, 1, b, keep_greater=False)

    Vn = x.shape[1]
    x_r = torch.cat([x[:, 1:], x[:, 0:1]], dim=1)
    y_r = torch.cat([y[:, 1:], y[:, 0:1]], dim=1)
    s1 = (x[:, : Vn - 1] * y_r[:, : Vn - 1] - x_r[:, : Vn - 1] * y[:, : Vn - 1]).sum(
        dim=1
    )
    idx = (n - 1).clamp(min=0).unsqueeze(1)
    xn1 = torch.gather(x, 1, idx).squeeze(1)
    yn1 = torch.gather(y, 1, idx).squeeze(1)
    s2 = xn1 * y[:, 0] - x[:, 0] * yn1
    valid = n >= 3
    s = s1 + s2 * valid.to(s1.dtype)
    areas = 0.5 * s.abs()
    areas = areas * valid.to(areas.dtype)

    return areas


def _quadrilateral_areas_and_validity(
    quad_x: Tensor, quad_y: Tensor
) -> tuple[Tensor, Tensor]:
    next_x = torch.roll(quad_x, shifts=-1, dims=1)
    next_y = torch.roll(quad_y, shifts=-1, dims=1)
    following_x = torch.roll(quad_x, shifts=-2, dims=1)
    following_y = torch.roll(quad_y, shifts=-2, dims=1)
    turns = (next_x - quad_x) * (following_y - next_y) - (
        next_y - quad_y
    ) * (following_x - next_x)
    valid = (turns > 0.0).all(dim=1) | (turns < 0.0).all(dim=1)
    areas = 0.5 * torch.abs(
        (quad_x[:, 2] - quad_x[:, 0]) * (quad_y[:, 3] - quad_y[:, 1])
        - (quad_y[:, 2] - quad_y[:, 0]) * (quad_x[:, 3] - quad_x[:, 1])
    )
    return areas, valid


def _accumulate_quadrilateral_overlaps(
    quad_x: Tensor,
    quad_y: Tensor,
    quad_numer: Tensor,
    quad_weight: Tensor,
    *,
    width: int,
    height: int,
    flat_numer: Tensor,
    flat_weight: Tensor,
    flat_variance_weight: Tensor | None,
) -> None:
    if quad_x.numel() == 0:
        return
    areas, valid_footprints = _quadrilateral_areas_and_validity(quad_x, quad_y)
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
    counts = torch.where(
        valid_footprints & (areas > 0.0), nx * ny, torch.zeros_like(nx)
    )
    ends = torch.cumsum(counts, dim=0)
    summary = torch.stack((ends[-1], (~valid_footprints).sum())).tolist()
    pair_count = int(summary[0])
    if int(summary[1]) > 0:
        raise ValueError(
            "projected source footprint is self-intersecting, non-convex, or degenerate"
        )
    if pair_count == 0:
        return
    starts = ends - counts
    pair_chunk = _MAX_OVERLAP_CANDIDATE_PAIRS
    for pair_start in range(0, pair_count, pair_chunk):
        pair_stop = min(pair_start + pair_chunk, pair_count)
        if pair_start == 0 and pair_stop == pair_count:
            owners = torch.repeat_interleave(
                torch.arange(
                    quad_x.shape[0], dtype=torch.int64, device=quad_x.device
                ),
                counts,
            )
        else:
            first_owner = int(
                torch.searchsorted(
                    ends,
                    torch.tensor(
                        pair_start, dtype=torch.int64, device=quad_x.device
                    ),
                    right=True,
                ).item()
            )
            last_owner = int(
                torch.searchsorted(
                    ends,
                    torch.tensor(
                        pair_stop - 1, dtype=torch.int64, device=quad_x.device
                    ),
                    right=True,
                ).item()
            )
            owner_ids = torch.arange(
                first_owner,
                last_owner + 1,
                dtype=torch.int64,
                device=quad_x.device,
            )
            selected_starts = torch.maximum(
                starts[owner_ids],
                torch.as_tensor(
                    pair_start, dtype=torch.int64, device=quad_x.device
                ),
            )
            selected_ends = torch.minimum(
                ends[owner_ids],
                torch.as_tensor(pair_stop, dtype=torch.int64, device=quad_x.device),
            )
            owners = torch.repeat_interleave(
                owner_ids, selected_ends - selected_starts
            )
        pair_ids = torch.arange(
            pair_start,
            pair_stop,
            dtype=torch.int64,
            device=quad_x.device,
        )
        local = pair_ids - starts[owners]
        owner_nx = nx[owners]
        center_x = min_x[owners] + torch.remainder(local, owner_nx)
        center_y = min_y[owners] + torch.div(
            local, owner_nx, rounding_mode="floor"
        )
        centers = torch.stack((center_x, center_y), dim=1).to(quad_x.dtype)
        overlaps = _batch_quads_pixel_areas(
            quad_x[owners], quad_y[owners], centers
        )
        overlap_valid = overlaps > 0.0
        if not bool(overlap_valid.any()):
            continue
        owners = owners[overlap_valid]
        output_index = (
            center_y[overlap_valid] * width + center_x[overlap_valid]
        )
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


def exact_square_drizzle_project(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    drizzle_pixfrac: float,
    pixel_area_sr_map_fn,
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
    half = 0.5 * max(float(drizzle_pixfrac), 1.0e-6)
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
            (x_chunk - half, x_chunk + half, x_chunk + half, x_chunk - half), dim=1
        )
        sy = torch.stack(
            (y_chunk - half, y_chunk - half, y_chunk + half, y_chunk + half), dim=1
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


def exact_point_drizzle_project(
    image: Tensor,
    source_weight: Tensor,
    source_wcs,
    target,
    *,
    value_semantics: str,
    pixel_area_sr_map_fn,
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

    valid = (
        torch.isfinite(weighted_weight)
        & torch.isfinite(weighted_image)
        & (weighted_weight > 0.0)
    )
    if not bool(valid.any()):
        zeros = torch.zeros(target.shape, dtype=torch.float32, device=image.device)
        return zeros, zeros

    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    width = int(target.shape[1])
    height = int(target.shape[0])
    flat_weighted_image = weighted_image.reshape(-1)
    flat_weighted_weight = weighted_weight.reshape(-1)
    for flat_index, source_x, source_y in _iter_valid_source_pixel_chunks(
        valid, chunk_pixels
    ):
        x_chunk = source_x.to(dtype=torch.float64)
        y_chunk = source_y.to(dtype=torch.float64)
        numer_chunk = flat_weighted_image[flat_index]
        weight_chunk = flat_weighted_weight[flat_index]
        ra_deg, dec_deg = source_wcs.pixel_to_world(x_chunk, y_chunk)
        x_tgt, y_tgt = target.wcs.world_to_pixel(ra_deg, dec_deg)
        x_tgt = x_tgt.to(dtype=torch.float32)
        y_tgt = y_tgt.to(dtype=torch.float32)
        y_lower, y_upper = _source_row_vertical_bounds(
            y_chunk,
            source_height=int(image.shape[0]),
            target_height=height,
        )
        finite = torch.isfinite(x_tgt) & torch.isfinite(y_tgt)
        inside = (
            finite
            & (x_tgt >= -0.5 - _DRIZZLE_BOUNDARY_EPS)
            & (x_tgt <= float(width - 1) + 0.5 + _DRIZZLE_BOUNDARY_EPS)
            & (y_tgt >= y_lower.to(dtype=torch.float32))
            & (y_tgt <= y_upper.to(dtype=torch.float32))
        )
        if not bool(inside.any()):
            continue
        x_tgt = x_tgt[inside]
        y_tgt = y_tgt[inside]
        numer_chunk = numer_chunk[inside]
        weight_chunk = weight_chunk[inside]
        ix = torch.floor(x_tgt + 0.5).to(dtype=torch.int64)
        iy = torch.floor(y_tgt + 0.5).to(dtype=torch.int64)
        ix = torch.clamp(ix, min=0, max=width - 1)
        iy = torch.clamp(iy, min=0, max=height - 1)
        idx = iy * width + ix
        flat_numer.scatter_add_(0, idx, numer_chunk)
        flat_weight.scatter_add_(0, idx, weight_chunk)

    numer = flat_numer.reshape(target.shape)
    weight = flat_weight.reshape(target.shape)
    numer = torch.where(weight > 0.0, numer, torch.zeros_like(numer))
    weight = torch.where(weight > 0.0, weight, torch.zeros_like(weight))
    return numer, weight
