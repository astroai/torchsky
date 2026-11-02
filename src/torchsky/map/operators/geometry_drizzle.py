"""Drizzle and splat geometry helpers."""

from __future__ import annotations

import os

import torch
from torch import Tensor

from ...device import require_float64

_DRIZZLE_BOUNDARY_EPS = 1.0e-4
# Candidate (quad, box) pairs evaluated per tensor batch.  The CPU limit keeps the working
# set of the ~100 elementwise passes inside the cache; accelerators amortise launches with
# larger batches.
_MAX_OVERLAP_CANDIDATE_PAIRS = 65_536
_MAX_OVERLAP_CANDIDATE_PAIRS_ACCELERATOR = 1_048_576
# ``TORCHSKY_DRIZZLE_NATIVE=0`` forces the pure-torch overlap path (identical semantics).
_NATIVE_OVERLAP_ENABLED = os.environ.get("TORCHSKY_DRIZZLE_NATIVE", "1") != "0"


def _native_overlap_op():
    """Return the compiled CPU overlap-accumulation operator, or ``None``."""
    if not _NATIVE_OVERLAP_ENABLED:
        return None
    from ..._native import cpp

    if cpp is None:
        return None
    return getattr(torch.ops.torchsky, "drizzle_square_accumulate", None)


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
    """Point-drizzle y-centre bounds, the same ``[-0.5, n - 0.5]`` edges as x.

    ``source_height`` is kept for call compatibility; no source row is special.
    """
    lower = torch.full_like(source_y, -0.5 - _DRIZZLE_BOUNDARY_EPS)
    upper = torch.full_like(
        source_y,
        float(target_height - 1) + 0.5 + _DRIZZLE_BOUNDARY_EPS,
    )
    return lower, upper


def _relu_split(x: Tensor) -> Tensor:
    """``max(x, 0)`` whose gradient at exactly ``x == 0`` is 1/2 (``Tensor.clamp`` gives 0).

    Pixel-aligned grids put polygon vertices exactly on box edges, where a zero
    sub-gradient would drop the boundary contribution from autograd.
    """
    return torch.maximum(x, torch.zeros_like(x))


def _unit_clamp_split(x: Tensor) -> Tensor:
    """``clamp(x, 0, 1)`` with tie-splitting gradients at both bounds."""
    return torch.minimum(_relu_split(x), torch.ones_like(x))


def _positive_part_mean(lo: Tensor, hi: Tensor) -> Tensor:
    """Mean of ``max(u, 0)`` along a segment whose ends carry ``u = lo`` and ``u = hi``.

    Exact for a linear ``u``.  A sign change splits the segment at the zero crossing,
    where the positive branch is a triangle of mean ``max(lo, hi)**2 / (2 |hi - lo|)``;
    all other cases are the trapezoid ``(max(lo, 0) + max(hi, 0)) / 2``.  Both branches
    are sums of non-negative terms, so there is no cancellation for near-vertical edges.
    """
    p = _relu_split(lo)
    q = _relu_split(hi)
    crossing = (torch.minimum(lo, hi) < 0.0) & (torch.maximum(lo, hi) > 0.0)
    top = torch.maximum(p, q)
    span = (hi - lo).abs().clamp(min=1.0e-300)
    return torch.where(crossing, top * top / (2.0 * span), 0.5 * (p + q))


def _unit_box_overlap_areas_value(x: Tensor, y: Tensor) -> Tensor:
    """Exact area of convex quadrilaterals ``x[:, k], y[:, k]`` inside ``[0, 1]**2``.

    ``x`` and ``y`` have shape ``(4, P)`` (vertex-major, contiguous rows) and are given
    relative to the lower-left corner of the box.  The overlap is the Green's-theorem
    boundary integral ``oint clamp(x, 0, 1) dy`` over the polygon edges clipped to
    ``0 <= y <= 1`` (the specialised axis-aligned form of what STScI ``boxer``/``sgarea``
    do per edge), so no intermediate polygon is built.  A separating-axis test (the four
    quad edge normals and the two box axes) returns exactly zero for non-overlapping
    candidates instead of rounding noise.  The tests are non-strict: a box that only
    touches the quadrilateral goes through the boundary integral (which is exactly zero
    there) so the tie-split sub-gradient of the touching boundary term reaches autograd.
    """
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)
    dx = x_next - x
    dy = y_next - y
    # Orientation from the shoelace sum: +1 counter-clockwise, -1 clockwise, 0 degenerate.
    sign = torch.sign((x * y_next - x_next * y).sum(dim=0))
    # Separating axis on the quad edges: the box corner deepest inside each edge's
    # half-plane must be strictly inside, otherwise the box and the quad do not overlap.
    ex = dx * sign
    ey = dy * sign
    deepest = ex.clamp(min=0.0) + (-ey).clamp(min=0.0) - ex * y + ey * x
    overlapping = (deepest >= 0.0).all(dim=0) & (sign != 0.0)
    # ... and on the two box axes.
    overlapping &= (
        (x.amin(dim=0) <= 1.0)
        & (x.amax(dim=0) >= 0.0)
        & (y.amin(dim=0) <= 1.0)
        & (y.amax(dim=0) >= 0.0)
    )

    ya = _unit_clamp_split(y)
    yb = _unit_clamp_split(y_next)
    clipped_dy = yb - ya
    # A horizontal edge (``dy == 0``) lies wholly inside or outside the clip band; its
    # parameter span is the full edge, ``t`` in ``[0, 1]``, so that the derivative of
    # ``clipped_dy`` (nonzero when the edge rotates) multiplies the mean over the whole edge.
    horizontal = dy == 0.0
    safe_dy = torch.where(horizontal, torch.ones_like(dy), dy)
    ta = torch.where(horizontal, torch.zeros_like(dy), (ya - y) / safe_dy)
    tb = torch.where(horizontal, torch.ones_like(dy), (yb - y) / safe_dy)
    xa = x + dx * ta
    xb = x + dx * tb
    mean_clamped = _positive_part_mean(xa, xb) - _positive_part_mean(xa - 1.0, xb - 1.0)
    signed = (clipped_dy * mean_clamped).sum(dim=0)
    return torch.where(
        overlapping, _relu_split(signed * sign), torch.zeros_like(signed)
    )


def _clip_parameter_range(p: Tensor, d: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    """Parameter interval of ``p + t d`` inside ``0 <= . <= 1`` and a boundary weight.

    An edge parallel to the slab and lying exactly on one of its two bounding lines gets
    weight 1/2: the area derivative there is one-sided and the symmetric (mean)
    sub-gradient assigns half to the box on each side of the line.
    """
    moving = d != 0.0
    safe = torch.where(moving, d, torch.ones_like(d))
    t0 = -p / safe
    t1 = (1.0 - p) / safe
    on_line = (p == 0.0) | (p == 1.0)
    within = ((p > 0.0) & (p < 1.0)) | on_line
    inf = torch.full_like(p, float("inf"))
    lo = torch.where(moving, torch.minimum(t0, t1), torch.where(within, -inf, inf))
    hi = torch.where(moving, torch.maximum(t0, t1), torch.where(within, inf, -inf))
    weight = torch.where(
        ~moving & on_line, 0.5 * torch.ones_like(p), torch.ones_like(p)
    )
    return lo, hi, weight


def _unit_box_overlap_vertex_gradient(x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    """Analytic ``(dA/dx, dA/dy)`` of the quad / unit-box overlap area ``A``.

    The box is fixed, so ``dA = oint_{edges of the quad inside the box} (v . n) ds`` with
    ``n`` the outward normal and ``v`` the (linearly interpolated) vertex velocity.  Per
    edge ``i -> i+1`` with inside parameter range ``[ta, tb]`` the vertex ``i`` receives
    ``n * int (1 - t) dt`` and vertex ``i+1`` receives ``n * int t dt``, ``n = (dy, -dx)``
    for a counter-clockwise quad.  Unlike autograd through the clamps this has no
    compounding tie-splitting: boxes that only touch the quad and pixel-aligned
    geometry give the exact mean of the one-sided derivatives.
    """
    x_next = torch.roll(x, shifts=-1, dims=0)
    y_next = torch.roll(y, shifts=-1, dims=0)
    dx = x_next - x
    dy = y_next - y
    sign = torch.sign((x * y_next - x_next * y).sum(dim=0))
    lo_x, hi_x, w_x = _clip_parameter_range(x, dx)
    lo_y, hi_y, w_y = _clip_parameter_range(y, dy)
    zero = torch.zeros_like(x)
    ta = torch.maximum(torch.maximum(lo_x, lo_y), zero)
    tb = torch.minimum(torch.minimum(hi_x, hi_y), torch.ones_like(x))
    empty = tb <= ta
    ta = torch.where(empty, zero, ta)
    tb = torch.where(empty, zero, tb)
    second = 0.5 * (tb * tb - ta * ta)
    first = (tb - ta) - second
    coef = sign * w_x * w_y
    nx = coef * dy
    ny = -coef * dx
    grad_x = nx * first + torch.roll(nx * second, shifts=1, dims=0)
    grad_y = ny * first + torch.roll(ny * second, shifts=1, dims=0)
    return grad_x, grad_y


class _UnitBoxOverlap(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: Tensor, y: Tensor) -> Tensor:  # type: ignore[override]
        ctx.save_for_backward(x, y)
        return _unit_box_overlap_areas_value(x, y)

    @staticmethod
    def backward(ctx, grad_output: Tensor):  # type: ignore[override]
        x, y = ctx.saved_tensors
        grad_x, grad_y = _unit_box_overlap_vertex_gradient(x, y)
        return grad_x * grad_output, grad_y * grad_output


def _unit_box_overlap_areas(x: Tensor, y: Tensor) -> Tensor:
    """Exact quad / unit-box overlap areas; differentiable with analytic vertex gradients."""
    if torch.is_grad_enabled() and (x.requires_grad or y.requires_grad):
        return _UnitBoxOverlap.apply(x, y)
    return _unit_box_overlap_areas_value(x, y)


def _batch_quads_pixel_areas(
    quad_x: Tensor,
    quad_y: Tensor,
    centers: Tensor,
) -> Tensor:
    """Overlap area of each quadrilateral with the unit pixel box at its own centre."""
    M = centers.shape[0]
    if M == 0:
        return torch.empty(0, dtype=quad_x.dtype, device=quad_x.device)
    left = (centers[:, 0] - 0.5).unsqueeze(0)
    bottom = (centers[:, 1] - 0.5).unsqueeze(0)
    return _unit_box_overlap_areas(
        quad_x.t().contiguous() - left, quad_y.t().contiguous() - bottom
    )


def _quadrilateral_areas_and_validity(
    quad_x: Tensor, quad_y: Tensor
) -> tuple[Tensor, Tensor]:
    """Quadrilateral areas and strict convexity (all four turns of one sign)."""
    next_x = torch.cat((quad_x[:, 1:], quad_x[:, :1]), dim=1)
    next_y = torch.cat((quad_y[:, 1:], quad_y[:, :1]), dim=1)
    edge_x = next_x - quad_x
    edge_y = next_y - quad_y
    following_x = torch.cat((edge_x[:, 1:], edge_x[:, :1]), dim=1)
    following_y = torch.cat((edge_y[:, 1:], edge_y[:, :1]), dim=1)
    turns = edge_x * following_y - edge_y * following_x
    valid = (turns > 0.0).all(dim=1) | (turns < 0.0).all(dim=1)
    areas = 0.5 * torch.abs(
        (quad_x[:, 2] - quad_x[:, 0]) * (quad_y[:, 3] - quad_y[:, 1])
        - (quad_y[:, 2] - quad_y[:, 0]) * (quad_x[:, 3] - quad_x[:, 1])
    )
    return areas, valid


def _scatter_add_ordered(
    targets: list[Tensor],
    index: Tensor,
    values: list[Tensor],
    *,
    ordered: bool | None = None,
) -> None:
    """``targets[k].scatter_add_(0, index, values[k])`` with a run-independent result.

    CUDA ``scatter_add_`` accumulates colliding entries with float atomics, whose order
    varies from run to run, so the rounded sums are not reproducible.  On CUDA the
    entries are instead stable-sorted by target index and each run of equal indices is
    summed sequentially (``segment_reduce``, no atomics) and added to the target once
    per index, which fixes the summation order.  Other devices use ``scatter_add_``
    (serial on CPU, hence already deterministic).  ``ordered`` forces either path.
    """
    if ordered is None:
        ordered = index.device.type == "cuda"
    if not ordered:
        for target, value in zip(targets, values):
            target.scatter_add_(0, index, value)
        return
    order = torch.sort(index, stable=True).indices
    sorted_index = index[order]
    unique_index, counts = torch.unique_consecutive(sorted_index, return_counts=True)
    for target, value in zip(targets, values):
        sums = torch.segment_reduce(value[order], "sum", lengths=counts)
        target[unique_index] = target[unique_index] + sums


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
    quad_variance: Tensor | None = None,
) -> None:
    """Accumulate exact quad / pixel overlaps into the flat numerator and weight maps.

    ``quad_variance`` (optional) is the per-quad variance numerator when it is not the
    weight itself; the variance map then accumulates ``quad_variance * coefficient**2``.
    """
    if quad_x.numel() == 0:
        return
    if quad_variance is not None and flat_variance_weight is not None:
        _accumulate_quadrilateral_overlaps(
            quad_x,
            quad_y,
            quad_numer,
            quad_weight,
            width=width,
            height=height,
            flat_numer=flat_numer,
            flat_weight=flat_weight,
            flat_variance_weight=None,
        )
        _accumulate_quadrilateral_overlaps(
            quad_x,
            quad_y,
            quad_variance,
            quad_variance,
            width=width,
            height=height,
            flat_numer=torch.zeros_like(flat_numer),
            flat_weight=torch.zeros_like(flat_weight),
            flat_variance_weight=flat_variance_weight,
        )
        return
    native = _native_overlap_op()
    if (
        native is not None
        # The native kernel has no autograd formula: differentiable inputs use the torch path.
        and not (
            torch.is_grad_enabled()
            and any(
                t.requires_grad
                for t in (
                    quad_x,
                    quad_y,
                    quad_numer,
                    quad_weight,
                    flat_numer,
                    flat_weight,
                )
            )
        )
        and quad_x.device.type == "cpu"
        and quad_x.dtype == torch.float64
        and quad_y.dtype == torch.float64
        and flat_numer.dtype == torch.float32
        and quad_numer.dtype == torch.float32
        and quad_weight.dtype == torch.float32
    ):
        variance = (
            flat_variance_weight if flat_variance_weight is not None else flat_numer[:0]
        )
        invalid = native(
            quad_x.contiguous(),
            quad_y.contiguous(),
            quad_numer.contiguous(),
            quad_weight.contiguous(),
            int(width),
            int(height),
            flat_numer,
            flat_weight,
            variance,
            flat_variance_weight is not None,
        )
        if invalid:
            raise ValueError(
                "projected source footprint is self-intersecting, non-convex, or degenerate"
            )
        return
    finite = torch.isfinite(quad_x).all(dim=1) & torch.isfinite(quad_y).all(dim=1)
    if not bool(finite.all()):
        quad_x = quad_x[finite]
        quad_y = quad_y[finite]
        quad_numer = quad_numer[finite]
        quad_weight = quad_weight[finite]
        if quad_x.numel() == 0:
            return
    areas, valid_footprints = _quadrilateral_areas_and_validity(quad_x, quad_y)
    # Pixel i covers [i - 0.5, i + 0.5].  Boxes that only touch the bounding box have
    # zero overlap, so the value-only candidate range is the open overlap with
    # [min, max].  Under autograd the touching boxes are candidates too: the overlap
    # is a kinked function of the vertices exactly at pixel-aligned geometry, and the
    # touching boxes carry the other half of the boundary sub-gradient.
    differentiable = torch.is_grad_enabled() and any(
        t.requires_grad for t in (quad_x, quad_y, quad_numer, quad_weight)
    )
    lo_x = quad_x.min(dim=1).values - 0.5
    hi_x = quad_x.max(dim=1).values + 0.5
    lo_y = quad_y.min(dim=1).values - 0.5
    hi_y = quad_y.max(dim=1).values + 0.5
    if differentiable:
        min_x = torch.ceil(lo_x).to(torch.int64)
        max_x = torch.floor(hi_x).to(torch.int64)
        min_y = torch.ceil(lo_y).to(torch.int64)
        max_y = torch.floor(hi_y).to(torch.int64)
    else:
        min_x = torch.floor(lo_x).to(torch.int64) + 1
        max_x = torch.ceil(hi_x).to(torch.int64) - 1
        min_y = torch.floor(lo_y).to(torch.int64) + 1
        max_y = torch.ceil(hi_y).to(torch.int64) - 1
    min_x.clamp_(min=0)
    max_x.clamp_(max=width - 1)
    min_y.clamp_(min=0)
    max_y.clamp_(max=height - 1)
    nx = (max_x - min_x + 1).clamp(min=0)
    ny = (max_y - min_y + 1).clamp(min=0)
    counts = torch.where(
        valid_footprints & (areas > 0.0), nx * ny, torch.zeros_like(nx)
    )
    # Overlaps below the representation error of the input vertices (``eps * |coordinate|``
    # times the chord, bounded by the footprint extent) are grazing contacts, not overlap.
    coordinate_scale = torch.maximum(
        torch.maximum(quad_x.abs().amax(dim=1), quad_y.abs().amax(dim=1)),
        torch.ones_like(areas),
    )
    extent = torch.maximum(
        quad_x.amax(dim=1) - quad_x.amin(dim=1), quad_y.amax(dim=1) - quad_y.amin(dim=1)
    )
    resolution = 16.0 * torch.finfo(quad_x.dtype).eps * coordinate_scale * extent
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
    pair_chunk = (
        _MAX_OVERLAP_CANDIDATE_PAIRS
        if quad_x.device.type == "cpu"
        else _MAX_OVERLAP_CANDIDATE_PAIRS_ACCELERATOR
    )
    for pair_start in range(0, pair_count, pair_chunk):
        pair_stop = min(pair_start + pair_chunk, pair_count)
        if pair_start == 0 and pair_stop == pair_count:
            owners = torch.repeat_interleave(
                torch.arange(quad_x.shape[0], dtype=torch.int64, device=quad_x.device),
                counts,
            )
        else:
            first_owner = int(
                torch.searchsorted(
                    ends,
                    torch.tensor(pair_start, dtype=torch.int64, device=quad_x.device),
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
                torch.as_tensor(pair_start, dtype=torch.int64, device=quad_x.device),
            )
            selected_ends = torch.minimum(
                ends[owner_ids],
                torch.as_tensor(pair_stop, dtype=torch.int64, device=quad_x.device),
            )
            owners = torch.repeat_interleave(owner_ids, selected_ends - selected_starts)
        pair_ids = torch.arange(
            pair_start,
            pair_stop,
            dtype=torch.int64,
            device=quad_x.device,
        )
        local = pair_ids - starts[owners]
        owner_nx = nx[owners]
        center_x = min_x[owners] + torch.remainder(local, owner_nx)
        center_y = min_y[owners] + torch.div(local, owner_nx, rounding_mode="floor")
        centers = torch.stack((center_x, center_y), dim=1).to(quad_x.dtype)
        overlaps = _batch_quads_pixel_areas(quad_x[owners], quad_y[owners], centers)
        overlap_valid = overlaps > resolution[owners]
        if differentiable:
            # Grazing contacts add exactly zero to the maps but keep their sub-gradient.
            overlaps = torch.where(
                overlap_valid, overlaps, overlaps - overlaps.detach()
            )
            overlap_valid = overlaps.new_ones(overlaps.shape, dtype=torch.bool)
        if not bool(overlap_valid.any()):
            continue
        owners = owners[overlap_valid]
        output_index = center_y[overlap_valid] * width + center_x[overlap_valid]
        coefficients = (overlaps[overlap_valid] / areas[owners]).to(dtype=torch.float32)
        targets = [flat_numer, flat_weight]
        values = [quad_numer[owners] * coefficients, quad_weight[owners] * coefficients]
        if flat_variance_weight is not None:
            targets.append(flat_variance_weight)
            values.append(quad_weight[owners] * torch.square(coefficients))
        _scatter_add_ordered(targets, output_index, values)


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
    """Square drizzle: exact polygon overlap of each ``pixfrac``-scaled source pixel.

    Every source pixel's drop (a square of side ``pixfrac`` source pixels centred on the
    pixel) is projected through the WCS pair and overlapped exactly with the target pixels
    (rotation, shear and scale ratios included).  The result is differentiable with respect
    to the projected vertices through analytic overlap-area derivatives, which are exact
    (mean of the one-sided derivatives) at pixel-aligned geometry.
    """
    require_float64(image.device, what="exact square drizzle")
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
        # Footprints with a non-finite corner are skipped inside the accumulation.
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
    """Point drizzle: each source pixel centre drops its whole value on one target pixel.

    Edge semantics (deliberately differs from STScI ``drizzle`` 3.0): a drop is kept when
    its target-pixel centre lies in ``[-0.5, n - 0.5]`` along each axis (the whole target
    grid including the outer half-pixel rim, x and y treated identically, with a
    1e-4 pixel tolerance), and lands on pixel ``floor(centre + 0.5)``.  STScI drizzle 3.0
    drops centres in the outer half-pixel rim, i.e. outside ``[0, n - 1]``.  torchsky keeps
    them so that no flux whose pixel footprint overlaps the target grid is discarded.
    """
    require_float64(image.device, what="exact point drizzle")
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
