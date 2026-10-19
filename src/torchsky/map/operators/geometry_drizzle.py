"""Drizzle and splat geometry helpers."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

_DRIZZLE_BOUNDARY_EPS = 1.0e-4
_DRIZZLE_VERTEX_ATOL = 1.0e-12
_DRIZZLE_MAX_INV_ERR = 0.03


@dataclass(frozen=True)
class _Edge:
    v1: tuple[float, float]
    v2: tuple[float, float]
    m: float
    c: float


@dataclass
class _Scanner:
    left_edges: list[_Edge]
    right_edges: list[_Edge]
    left_index: int
    right_index: int
    min_y: float
    max_y: float
    xmin: int
    xmax: int
    ymax_limit: int


def _signed_area(points: list[tuple[float, float]]) -> float:
    if len(points) < 3:
        return 0.0
    total = 0.0
    for idx, (x0, y0) in enumerate(points):
        x1, y1 = points[(idx + 1) % len(points)]
        total += x0 * y1 - x1 * y0
    return 0.5 * total


def _orient_ccw(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if _signed_area(points) < 0.0:
        return list(reversed(points))
    return points


def _simplify_polygon(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    if len(points) < 3:
        return points
    simplified: list[tuple[float, float]] = []
    count = len(points)
    for idx in range(count):
        x_prev, y_prev = points[idx - 1]
        x_curr, y_curr = points[idx]
        x_next, y_next = points[(idx + 1) % count]
        dx0 = x_curr - x_prev
        dy0 = y_curr - y_prev
        dx1 = x_next - x_curr
        dy1 = y_next - y_curr
        cross = dx0 * dy1 - dy0 * dx1
        norm = math.hypot(x_next - x_prev, y_next - y_prev)
        if (
            abs(cross) > 1.0e3 * float.fromhex("0x0.0000000000001p-1022")
            and norm > _DRIZZLE_VERTEX_ATOL
        ):
            simplified.append((x_curr, y_curr))
    return simplified if simplified else points


def _interpolate_point_from_pixmap(
    pixmap: Tensor,
    xin: float,
    yin: float,
) -> tuple[float, float] | None:
    height = int(pixmap.shape[0])
    width = int(pixmap.shape[1])
    i0 = int(xin)
    j0 = int(yin)
    nx2 = width - 2
    ny2 = height - 2
    i0 = min(max(i0, 0), nx2)
    j0 = min(max(j0, 0), ny2)
    x = xin - float(i0)
    y = yin - float(j0)
    x1 = 1.0 - x
    y1 = 1.0 - y
    f00 = pixmap[j0, i0]
    f10 = pixmap[j0, i0 + 1]
    f01 = pixmap[j0 + 1, i0]
    f11 = pixmap[j0 + 1, i0 + 1]
    xout = (
        float(f00[0]) * x1 * y1
        + float(f10[0]) * x * y1
        + float(f01[0]) * x1 * y
        + float(f11[0]) * x * y
    )
    yout = (
        float(f00[1]) * x1 * y1
        + float(f10[1]) * x * y1
        + float(f01[1]) * x1 * y
        + float(f11[1]) * x * y
    )
    if not math.isfinite(xout) or not math.isfinite(yout):
        return None
    return (xout, yout)


def _build_center_pixmap(
    source_wcs, target_wcs, shape: tuple[int, int], device
) -> Tensor:
    ys, xs = torch.meshgrid(
        torch.arange(shape[0], dtype=torch.float64, device=device),
        torch.arange(shape[1], dtype=torch.float64, device=device),
        indexing="ij",
    )
    ra_deg, dec_deg = source_wcs.pixel_to_world(xs.reshape(-1), ys.reshape(-1))
    x_tgt, y_tgt = target_wcs.world_to_pixel(ra_deg, dec_deg)
    return (
        torch.stack((x_tgt, y_tgt), dim=-1)
        .reshape(shape[0], shape[1], 2)
        .to(dtype=torch.float64)
    )


def _clip_polygon_to_rect(
    points: list[tuple[float, float]],
    *,
    xmin: float,
    xmax: float,
    ymin: float,
    ymax: float,
) -> list[tuple[float, float]]:
    poly = _clip_polygon_half_plane(points, axis=0, boundary=xmin, keep_greater=True)
    poly = _clip_polygon_half_plane(poly, axis=0, boundary=xmax, keep_greater=False)
    poly = _clip_polygon_half_plane(poly, axis=1, boundary=ymin, keep_greater=True)
    poly = _clip_polygon_half_plane(poly, axis=1, boundary=ymax, keep_greater=False)
    return poly


def _batch_interpolate_pixmap(
    pixmap: Tensor, xs: Tensor, ys: Tensor
) -> Tensor:
    H, W = pixmap.shape[:2]
    i0 = xs.to(torch.int64).clamp(0, W - 2)
    j0 = ys.to(torch.int64).clamp(0, H - 2)
    x = xs - i0.to(xs.dtype)
    y = ys - j0.to(ys.dtype)
    x1 = 1.0 - x
    y1 = 1.0 - y
    f00 = pixmap[j0, i0]
    f10 = pixmap[j0, i0 + 1]
    f01 = pixmap[j0 + 1, i0]
    f11 = pixmap[j0 + 1, i0 + 1]
    return (
        f00 * (x1 * y1).unsqueeze(-1)
        + f10 * (x * y1).unsqueeze(-1)
        + f01 * (x1 * y).unsqueeze(-1)
        + f11 * (x * y).unsqueeze(-1)
    )


def _invert_pixmap(
    pixmap: Tensor,
    *,
    xmin: int,
    xmax: int,
    ymin: int,
    ymax: int,
    xout: float,
    yout: float,
) -> tuple[float, float] | None:
    gr = 0.6180339887498949
    xmin_f = float(xmin) - 0.5
    xmax_f = float(xmax) + 0.5
    ymin_f = float(ymin) - 0.5
    ymax_f = float(ymax) + 0.5
    dx = xmax_f - xmin_f
    dy = ymax_f - ymin_f
    device = pixmap.device
    dtype = pixmap.dtype
    xs = torch.empty(4, device=device, dtype=dtype)
    ys = torch.empty(4, device=device, dtype=dtype)
    xout_t = torch.tensor(xout, device=device, dtype=dtype)
    yout_t = torch.tensor(yout, device=device, dtype=dtype)
    niter = 0
    while (dx > _DRIZZLE_MAX_INV_ERR or dy > _DRIZZLE_MAX_INV_ERR) and niter < 50:
        niter += 1
        x1_val = xmax_f - gr * dx
        x2_val = xmin_f + gr * dx
        y1_val = ymax_f - gr * dy
        y2_val = ymin_f + gr * dy
        xs[0] = x1_val
        xs[1] = x1_val
        xs[2] = x2_val
        xs[3] = x2_val
        ys[0] = y1_val
        ys[1] = y2_val
        ys[2] = y1_val
        ys[3] = y2_val
        mapped = _batch_interpolate_pixmap(pixmap, xs, ys)
        if not torch.isfinite(mapped).all():
            return None
        errs = (mapped[:, 0] - xout_t).pow_(2) + (mapped[:, 1] - yout_t).pow_(2)
        d11 = errs[0].item()
        d12 = errs[1].item()
        d21 = errs[2].item()
        d22 = errs[3].item()
        if d11 < d12 and d11 < d21 and d11 < d22:
            xmax_f = x2_val
            ymax_f = y2_val
        elif d12 < d11 and d12 < d21 and d12 < d22:
            xmax_f = x2_val
            ymin_f = y1_val
        elif d21 < d11 and d21 < d12 and d21 < d22:
            xmin_f = x1_val
            ymax_f = y2_val
        else:
            xmin_f = x1_val
            ymin_f = y1_val
        dx = xmax_f - xmin_f
        dy = ymax_f - ymin_f
    if niter == 50:
        return None
    return (0.5 * (xmin_f + xmax_f), 0.5 * (ymin_f + ymax_f))


def _init_edge(
    v1: tuple[float, float], v2: tuple[float, float], position: int
) -> _Edge:
    m = (v2[0] - v1[0]) / (v2[1] - v1[1])
    b = (v1[0] * v2[1] - v1[1] * v2[0]) / (v2[1] - v1[1])
    c = b - math.copysign(0.5 + 0.5 * abs(m), float(position))
    return _Edge(v1=v1, v2=v2, m=m, c=c)


def _init_scanner(
    polygon: list[tuple[float, float]],
    *,
    xmin: int,
    xmax: int,
    ymax: int,
) -> _Scanner | None:
    if len(polygon) < 3:
        return None
    points = _orient_ccw(_simplify_polygon(polygon))
    npts = len(points)
    min_left = min(range(npts), key=lambda idx: points[idx][1])
    min_y = points[min_left][1]
    i1 = (min_left - 1) % npts
    i2 = (min_left + 1) % npts
    min_right = i1 if points[i1][1] < points[i2][1] else i2
    if points[min_right][1] <= min_y * (
        1.0 + math.copysign(_DRIZZLE_VERTEX_ATOL, min_y)
    ):
        if points[min_left][0] > points[min_right][0]:
            min_left, min_right = min_right, min_left
    else:
        min_right = min_left

    max_right = max(range(npts), key=lambda idx: points[idx][1])
    max_y = points[max_right][1]
    i1 = (max_right - 1) % npts
    i2 = (max_right + 1) % npts
    max_left = i1 if points[i1][1] > points[i2][1] else i2
    if points[max_left][1] >= max_y * (
        1.0 - math.copysign(_DRIZZLE_VERTEX_ATOL, max_y)
    ):
        if points[max_left][0] > points[max_right][0]:
            max_left, max_right = max_right, max_left
    else:
        max_left = max_right

    left_edges: list[_Edge] = []
    right_edges: list[_Edge] = []
    min_left_adj = min_left + npts if max_left > min_left else min_left
    for k in range(min_left_adj - max_left):
        a = (min_left_adj - k) % npts
        b = (a - 1) % npts
        left_edges.append(_init_edge(points[a], points[b], -1))
    max_right_adj = max_right + npts if max_right < min_right else max_right
    for k in range(max_right_adj - min_right):
        a = (min_right + k) % npts
        b = (a + 1) % npts
        right_edges.append(_init_edge(points[a], points[b], 1))
    if not left_edges or not right_edges:
        return None
    return _Scanner(
        left_edges=left_edges,
        right_edges=right_edges,
        left_index=0,
        right_index=0,
        min_y=min_y,
        max_y=max_y,
        xmin=xmin,
        xmax=xmax,
        ymax_limit=ymax,
    )


def _get_scanline_limits(scanner: _Scanner, y: int) -> tuple[int, int] | None:
    pyb = float(y) - 0.5
    pyt = float(y) + 0.5
    if y < 0 or y > scanner.ymax_limit:
        return None
    if pyt <= scanner.min_y or pyb >= scanner.max_y + 1.0:
        return None
    if scanner.left_index >= len(scanner.left_edges) or scanner.right_index >= len(
        scanner.right_edges
    ):
        return None

    while pyb > scanner.left_edges[scanner.left_index].v2[1]:
        scanner.left_index += 1
        if scanner.left_index >= len(scanner.left_edges):
            return None
    while pyb > scanner.right_edges[scanner.right_index].v2[1]:
        scanner.right_index += 1
        if scanner.right_index >= len(scanner.right_edges):
            return None

    left = scanner.left_edges[scanner.left_index]
    right = scanner.right_edges[scanner.right_index]
    xlb = left.m * float(y) + left.c - _DRIZZLE_MAX_INV_ERR
    xrb = right.m * float(y) + right.c + _DRIZZLE_MAX_INV_ERR

    edge_ymax = left.v2[1] + 0.5 + _DRIZZLE_MAX_INV_ERR
    while pyt > edge_ymax:
        scanner.left_index += 1
        if scanner.left_index >= len(scanner.left_edges):
            return None
        left = scanner.left_edges[scanner.left_index]
        edge_ymax = left.v2[1] + 0.5 + _DRIZZLE_MAX_INV_ERR

    edge_ymax = right.v2[1] + 0.5 + _DRIZZLE_MAX_INV_ERR
    while pyt > edge_ymax:
        scanner.right_index += 1
        if scanner.right_index >= len(scanner.right_edges):
            return None
        right = scanner.right_edges[scanner.right_index]
        edge_ymax = right.v2[1] + 0.5 + _DRIZZLE_MAX_INV_ERR

    xlt = left.m * float(y) + left.c - _DRIZZLE_MAX_INV_ERR
    xrt = right.m * float(y) + right.c + _DRIZZLE_MAX_INV_ERR
    xlb = max(xlb, float(scanner.xmin))
    xlt = max(xlt, float(scanner.xmin))
    xrb = min(xrb, float(scanner.xmax))
    xrt = min(xrt, float(scanner.xmax))

    if xlt >= xrt:
        x1 = int(round(xlb))
        x2 = int(round(xrb))
        if xlb >= xrb:
            return None
    elif xlb >= xrb:
        x1 = int(round(xlt))
        x2 = int(round(xrt))
    else:
        x1 = int(round(max(xlb, xlt)))
        x2 = int(round(min(xrb, xrt)))
    return (x1, x2)


def _scanner_row_bounds(
    pixmap: Tensor,
    *,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    target_origin_yx: tuple[int, int] = (0, 0),
) -> tuple[Tensor, Tensor]:
    src_h, src_w = source_shape
    tgt_h, tgt_w = target_shape
    origin_y, origin_x = target_origin_yx
    input_poly = [
        (-0.5, -0.5),
        (float(src_w) - 0.5, -0.5),
        (float(src_w) - 0.5, float(src_h) - 0.5),
        (-0.5, float(src_h) - 0.5),
    ]
    mapped_poly = [_interpolate_point_from_pixmap(pixmap, x, y) for x, y in input_poly]
    if any(point is None for point in mapped_poly):
        # ponytail: a distorted detector may have invalid outer corners while
        # its interior overlaps the target. Scan all rows in that rare case;
        # upgrade to a finite-region contour if such detectors become common.
        return (
            torch.zeros((src_h,), dtype=torch.int64, device=pixmap.device),
            torch.full(
                (src_h,), src_w - 1, dtype=torch.int64, device=pixmap.device
            ),
        )
    clipped = _clip_polygon_to_rect(
        [point for point in mapped_poly if point is not None],
        xmin=-0.5 - float(origin_x),
        xmax=float(tgt_w) - 0.5 - float(origin_x),
        ymin=-0.5 - float(origin_y),
        ymax=float(tgt_h) - 0.5 - float(origin_y),
    )
    if len(clipped) < 3:
        # ponytail: four detector corners cannot prove that a curved WCS
        # footprint misses the target. Scan the detector in this uncertain
        # case; upgrade to a sampled boundary polygon if this ceiling matters.
        return (
            torch.zeros((src_h,), dtype=torch.int64, device=pixmap.device),
            torch.full((src_h,), src_w - 1, dtype=torch.int64, device=pixmap.device),
        )
    input_overlap: list[tuple[float, float]] = []
    for xout, yout in clipped:
        vin = _invert_pixmap(
            pixmap,
            xmin=0,
            xmax=src_w - 1,
            ymin=0,
            ymax=src_h - 1,
            xout=xout,
            yout=yout,
        )
        if vin is None:
            return (
                torch.zeros((src_h,), dtype=torch.int64, device=pixmap.device),
                torch.full(
                    (src_h,), src_w - 1, dtype=torch.int64, device=pixmap.device
                ),
            )
        input_overlap.append(vin)
    scanner = _init_scanner(
        input_overlap, xmin=0, xmax=src_w - 1, ymax=src_h - 1
    )
    if scanner is None:
        return (
            torch.zeros((src_h,), dtype=torch.int64, device=pixmap.device),
            torch.full(
                (src_h,), src_w - 1, dtype=torch.int64, device=pixmap.device
            ),
        )
    row_min = torch.full((src_h,), -1, dtype=torch.int64, device=pixmap.device)
    row_max = torch.full((src_h,), -1, dtype=torch.int64, device=pixmap.device)
    ymin = max(0, int(scanner.min_y + 0.5 + 2.0 * _DRIZZLE_MAX_INV_ERR))
    ymax = min(src_h - 1, int(scanner.max_y + 2.0 * _DRIZZLE_MAX_INV_ERR))
    for y in range(ymin, ymax + 1):
        limits = _get_scanline_limits(scanner, y)
        if limits is None:
            continue
        row_min[y] = limits[0]
        row_max[y] = limits[1]
    return row_min, row_max


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



def _clip_polygon_half_plane(
    points: list[tuple[float, float]],
    *,
    axis: int,
    boundary: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []

    def inside(point: tuple[float, float]) -> bool:
        value = point[axis]
        return value >= boundary if keep_greater else value <= boundary

    def intersect(
        p0: tuple[float, float], p1: tuple[float, float]
    ) -> tuple[float, float]:
        x0, y0 = p0
        x1, y1 = p1
        v0 = x0 if axis == 0 else y0
        v1 = x1 if axis == 0 else y1
        dv = v1 - v0
        if abs(dv) < 1.0e-12:
            return p1
        t = (boundary - v0) / dv
        return (x0 + t * (x1 - x0), y0 + t * (y1 - y0))

    output: list[tuple[float, float]] = []
    prev = points[-1]
    prev_inside = inside(prev)
    for curr in points:
        curr_inside = inside(curr)
        if curr_inside:
            if not prev_inside:
                output.append(intersect(prev, curr))
            output.append(curr)
        elif prev_inside:
            output.append(intersect(prev, curr))
        prev = curr
        prev_inside = curr_inside
    return output



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
    h_src, w_src = image.shape
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

    ys, xs = torch.nonzero(valid, as_tuple=True)
    flat_values_numer = weighted_image[ys, xs].to(dtype=torch.float32)
    flat_values_weight = weighted_weight[ys, xs].to(dtype=torch.float32)
    flat_numer = torch.zeros(
        target.shape[0] * target.shape[1], dtype=torch.float32, device=image.device
    )
    flat_weight = torch.zeros_like(flat_numer)
    flat_variance_weight = torch.zeros_like(flat_numer)
    width = int(target.shape[1])
    height = int(target.shape[0])
    half = 0.5 * max(float(drizzle_pixfrac), 1.0e-6)
    has_distortion = any(
        getattr(wcs, "sip", None) is not None or getattr(wcs, "tpv", None) is not None
        for wcs in (source_wcs, target.wcs)
    )
    if has_distortion:
        # ponytail: the corner-derived scanner assumes near-linear footprints.
        # Exact Drizzle must not drop curved SIP/TPV interiors; use the full
        # detector and upgrade to a sampled-boundary scanner if profiling asks.
        row_min = torch.zeros((h_src,), dtype=torch.int64, device=image.device)
        row_max = torch.full(
            (h_src,), w_src - 1, dtype=torch.int64, device=image.device
        )
        pixmap = None
    else:
        pixmap = _build_center_pixmap(
            source_wcs,
            target.wcs,
            tuple(image.shape),
            image.device,
        )
        row_min, row_max = _scanner_row_bounds(
            pixmap,
            source_shape=tuple(image.shape),
            target_shape=tuple(target.shape),
        )
    if pixmap is not None:
        base_min = row_min
        base_max = row_max
        # A finite square drop can cross either a tile or the outer canvas when
        # its center falls outside it, so dilate the center-derived scan rows.
        row_min = torch.full_like(base_min, w_src)
        row_max = torch.full_like(base_max, -1)
        candidate_margin = 1 + int(math.ceil(0.5 * float(drizzle_pixfrac)))
        for shift in range(-candidate_margin, candidate_margin + 1):
            src_start = max(-shift, 0)
            src_stop = min(h_src - shift, h_src)
            dst_start = src_start + shift
            dst_stop = src_stop + shift
            valid_rows = base_min[src_start:src_stop] >= 0
            candidate_min = torch.where(
                valid_rows,
                base_min[src_start:src_stop] - candidate_margin,
                torch.full_like(base_min[src_start:src_stop], w_src),
            )
            candidate_max = torch.where(
                valid_rows,
                base_max[src_start:src_stop] + candidate_margin,
                torch.full_like(base_max[src_start:src_stop], -1),
            )
            row_min[dst_start:dst_stop] = torch.minimum(
                row_min[dst_start:dst_stop], candidate_min
            )
            row_max[dst_start:dst_stop] = torch.maximum(
                row_max[dst_start:dst_stop], candidate_max
            )
        row_min = torch.where(row_max >= 0, row_min.clamp(min=0), row_min)
        row_max = row_max.clamp(max=w_src - 1)
        valid_rows = row_max >= row_min
        row_min = torch.where(valid_rows, row_min, torch.full_like(row_min, -1))
        row_max = torch.where(valid_rows, row_max, torch.full_like(row_max, -1))

    for start in range(0, int(xs.numel()), int(chunk_pixels)):
        end = min(start + int(chunk_pixels), int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]
        row_min_chunk = row_min[y_chunk.to(dtype=torch.int64)]
        row_max_chunk = row_max[y_chunk.to(dtype=torch.int64)]
        scan_keep = (
            (row_min_chunk >= 0)
            & (x_chunk.to(dtype=torch.int64) >= row_min_chunk)
            & (x_chunk.to(dtype=torch.int64) <= row_max_chunk)
        )
        if not bool(scan_keep.any()):
            continue
        x_chunk = x_chunk[scan_keep]
        y_chunk = y_chunk[scan_keep]
        numer_chunk = numer_chunk[scan_keep]
        weight_chunk = weight_chunk[scan_keep]

        ra_center_deg, dec_center_deg = source_wcs.pixel_to_world(x_chunk, y_chunk)
        x_center_tgt, y_center_tgt = target.wcs.world_to_pixel(
            ra_center_deg,
            dec_center_deg,
        )
        center_finite = torch.isfinite(x_center_tgt) & torch.isfinite(y_center_tgt)
        if not bool(center_finite.any()):
            continue
        # A drop may overlap the canvas even when its centre lies outside it.
        # The transformed corners and exact clip below are the support oracle.
        x_chunk = x_chunk[center_finite]
        y_chunk = y_chunk[center_finite]
        numer_chunk = numer_chunk[center_finite]
        weight_chunk = weight_chunk[center_finite]

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
        # ponytail: 32k-quad batches bound clipping workspace; switch to
        # candidate-count binning if highly magnified WCS footprints become common.
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
            local = torch.arange(pair_count, device=image.device) - torch.repeat_interleave(
                starts, counts
            )
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
    pixmap = _build_center_pixmap(
        source_wcs,
        target.wcs,
        tuple(image.shape),
        image.device,
    )
    row_min, row_max = _scanner_row_bounds(
        pixmap,
        source_shape=tuple(image.shape),
        target_shape=tuple(target.shape),
    )

    for start in range(0, int(xs.numel()), int(chunk_pixels)):
        end = min(start + int(chunk_pixels), int(xs.numel()))
        x_chunk = xs[start:end].to(dtype=torch.float64)
        y_chunk = ys[start:end].to(dtype=torch.float64)
        numer_chunk = flat_values_numer[start:end]
        weight_chunk = flat_values_weight[start:end]
        row_min_chunk = row_min[y_chunk.to(dtype=torch.int64)]
        row_max_chunk = row_max[y_chunk.to(dtype=torch.int64)]
        scan_keep = (
            (row_min_chunk >= 0)
            & (x_chunk.to(dtype=torch.int64) >= row_min_chunk)
            & (x_chunk.to(dtype=torch.int64) <= row_max_chunk)
        )
        if not bool(scan_keep.any()):
            continue
        x_chunk = x_chunk[scan_keep]
        y_chunk = y_chunk[scan_keep]
        numer_chunk = numer_chunk[scan_keep]
        weight_chunk = weight_chunk[scan_keep]
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
