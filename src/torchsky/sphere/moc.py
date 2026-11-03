"""Multi-Order Coverage (MOC) support for HEALPix."""

import json
import math
from typing import Any, Literal

import torch
from torch import Tensor

from .. import _ops
from .._dtype import work_dtype as _work_dtype
from . import healpix as _healpix
from .core import great_circle_distance, lonlat_to_unit_xyz
from .geom import PixelizedRegion, spherical_polygon_contains
from .randoms import PreparedRandomSampler, _prepare_moc_random_sampler, sample_randoms
from .sparse import SparseHealpixMap
from ..device import _to


def _order_from_nside(nside: int, *, name: str = "nside") -> int:
    nside_i = int(nside)
    if nside_i <= 0 or (nside_i & (nside_i - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two")
    return nside_i.bit_length() - 1


_MAX_HEALPIX_ORDER = 29
_MAX_HEALPIX_NSIDE = 1 << _MAX_HEALPIX_ORDER


def _as_int64_exact(value: int | Tensor, *, name: str) -> Tensor:
    tensor = torch.as_tensor(value)
    if tensor.numel() > 0 and (
        tensor.dtype == torch.bool
        or tensor.dtype.is_floating_point
        or tensor.dtype.is_complex
    ):
        raise ValueError(f"{name} must contain integers")
    return tensor.to(torch.int64)


def _nside_from_max_order(max_order: int) -> int:
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    if order > _MAX_HEALPIX_ORDER:
        raise ValueError(f"max_order must be <= {_MAX_HEALPIX_ORDER}; got {order}")
    return 1 << order


def _order_from_uniq(uniq: Tensor) -> Tensor:
    """Return the HEALPix order of each UNIQ value with integer arithmetic.

    UNIQ encodes the order as ``4 * nside**2 + nested_pixel`` with
    ``nside = 2**order``.  A float ``log2`` loses the low bits of large
    int64 values and misclassifies the last cells of high orders, so the
    order is derived from an exact integer bit length.
    """
    u = _as_int64_exact(uniq, name="uniq")
    if bool((u < 4).any()):
        raise ValueError("UNIQ values must be >= 4 (nside >= 1)")
    # floor(log2(u)) through a fixed unrolled binary search; exact for int64.
    exponent = torch.zeros_like(u)
    value = u
    for shift in (32, 16, 8, 4, 2, 1):
        shifted = torch.bitwise_right_shift(value, shift)
        nonzero = shifted > 0
        exponent = torch.where(nonzero, exponent + shift, exponent)
        value = torch.where(nonzero, shifted, value)
    # uniq in [4**(order+1), 4**(order+2) - 1]
    order = torch.bitwise_right_shift(exponent - 2, 1)
    if bool((order > _MAX_HEALPIX_ORDER).any()):
        raise ValueError(f"UNIQ values must encode nside <= {_MAX_HEALPIX_NSIDE}")
    nside = torch.bitwise_left_shift(torch.ones_like(order), order)
    pix = u - 4 * (nside * nside)
    if bool(((pix < 0) | (pix >= 12 * nside * nside)).any()):
        raise ValueError("UNIQ values must use the canonical NUNIQ encoding")
    return order


def nest2uniq(nside: int | Tensor, pix: int | Tensor) -> Tensor:
    """Convert NESTED pixel index to UNIQ pixel index."""
    n = _as_int64_exact(nside, name="nside")
    p = _as_int64_exact(pix, name="pix")
    device = p.device if isinstance(pix, Tensor) else n.device
    n = n.to(device=device)
    p = p.to(device=device)
    n, p = torch.broadcast_tensors(n, p)
    if bool((n <= 0).any()) or bool((n > _MAX_HEALPIX_NSIDE).any()):
        raise ValueError(f"nside must be in [1, {_MAX_HEALPIX_NSIDE}]")
    if bool((n & (n - 1) != 0).any()):
        raise ValueError("nside must be a positive power of two")
    if bool(((p < 0) | (p >= 12 * n * n)).any()):
        raise ValueError("pix must satisfy 0 <= pix < 12 * nside**2")
    return 4 * (n * n) + p


def uniq2nest(uniq: int | Tensor) -> tuple[Tensor, Tensor]:
    """Convert UNIQ pixel index to (nside, NESTED pixel index)."""
    u = _as_int64_exact(uniq, name="uniq")
    order = _order_from_uniq(u)
    nside = torch.bitwise_left_shift(torch.ones_like(order), order)
    pix = u - 4 * (nside * nside)
    return nside, pix


def _order_of_uniq_scalar(value: int) -> int:
    """Order of one UNIQ value by exact integer bit length (validating)."""
    if value >= 4:
        order = (value.bit_length() - 3) >> 1
        if order <= _MAX_HEALPIX_ORDER and value - (4 << (2 * order)) < (
            12 << (2 * order)
        ):
            return order
    # Invalid value: the tensor path raises the descriptive error.
    return int(_order_from_uniq(torch.tensor(value, dtype=torch.int64)).item())


def _max_order_from_uniq(uniq: Tensor) -> int:
    if uniq.numel() == 0:
        return 0
    return _order_of_uniq_scalar(int(uniq.max().item()))


def _merge_sorted_ranges(starts: Tensor, ends: Tensor) -> tuple[Tensor, Tensor]:
    """Coalesce overlapping or touching half-open ranges sorted by start."""
    n = int(starts.numel())
    if n == 0:
        return starts, ends
    run_end = torch.cummax(ends, dim=0).values
    new_group = torch.ones(n, dtype=torch.bool, device=starts.device)
    new_group[1:] = starts[1:] > run_end[:-1]
    first = torch.nonzero(new_group).reshape(-1)
    last = torch.cat((first[1:] - 1, first.new_full((1,), n - 1)))
    return starts[first], run_end[last]


def _merge_ranges(starts: Tensor, ends: Tensor) -> tuple[Tensor, Tensor]:
    """Sort and coalesce arbitrary half-open ranges."""
    if starts.numel() == 0:
        return starts, ends
    order = torch.argsort(starts)
    return _merge_sorted_ranges(starts[order], ends[order])


def _uniq_to_range_tensors_torch(uniq: Tensor, max_order: int) -> tuple[Tensor, Tensor]:
    """Sorted disjoint ``[start, end)`` NESTED ranges at ``max_order`` for UNIQ cells."""
    u = _as_int64_exact(uniq, name="uniq").reshape(-1)
    if u.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=u.device)
        return empty, empty
    orders = _order_from_uniq(u)
    pix = u - torch.bitwise_left_shift(torch.ones_like(u), 2 * orders + 2)
    delta = 2 * (int(max_order) - orders)
    up = delta.clamp_min(0)
    down = (-delta).clamp_min(0)
    pix = torch.bitwise_right_shift(pix, down)
    return _merge_ranges(
        torch.bitwise_left_shift(pix, up), torch.bitwise_left_shift(pix + 1, up)
    )


def _range_tensors_to_uniq_torch(
    starts: Tensor, ends: Tensor, max_order: int
) -> Tensor:
    """Maximal aligned UNIQ cells covering disjoint ranges, ordered by position.

    Each range ``[s, e)`` splits into the cells that trim it to 4**(k+1)
    alignment at every level ``k``; the work is ``O(max_order)`` vector passes
    plus the output size, with no per-cell Python.
    """
    device = starts.device
    if starts.numel() == 0:
        return torch.empty((0,), dtype=torch.int64, device=device)
    order = int(max_order)
    s = starts.clone()
    e = ends.clone()
    cell_start: list[Tensor] = []
    cell_uniq: list[Tensor] = []

    def emit(first: Tensor, count: Tensor, k: int, cap: int) -> None:
        # At most ``cap`` equal cells per range side: a (n, cap) grid + mask
        # avoids data-dependent repeat/scan passes.
        offs = torch.arange(cap, device=device)
        valid = offs[None, :] < count[:, None]
        pos = (first[:, None] + (offs[None, :] << (2 * k)))[valid]
        if pos.numel() == 0:
            return
        cell_start.append(pos)
        cell_uniq.append((pos >> (2 * k)) + (4 << (2 * (order - k))))

    for k in range(order + 1):
        if k == order:
            emit(s, ((e - s).clamp_min(0)) >> (2 * k), k, 12)
            break
        block = 1 << (2 * (k + 1))
        s_up = (s + (block - 1)) & ~(block - 1)
        e_dn = e & ~(block - 1)
        single = s_up > e_dn  # no whole 4**(k+1) block: finish at this level
        head_end = torch.where(single, e, s_up)
        emit(s, ((head_end - s).clamp_min(0)) >> (2 * k), k, 3)
        tail_start = torch.where(single, e, e_dn)
        emit(tail_start, ((e - tail_start).clamp_min(0)) >> (2 * k), k, 3)
        s = torch.where(single, e, s_up)
        e = torch.where(single, e, e_dn)
        # Ranges that finished (typically most of them, at low levels) drop out,
        # so later levels touch geometrically fewer elements.
        live = s < e
        if not bool(live.any()):
            break
        s = s[live]
        e = e[live]
    if not cell_uniq:
        return torch.empty((0,), dtype=torch.int64, device=device)
    uniq = torch.cat(cell_uniq)
    return uniq[torch.argsort(torch.cat(cell_start))]


def _combine_range_tensors_torch(
    a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor], op: str
) -> tuple[Tensor, Tensor]:
    """Boolean algebra of two disjoint range sets by sweeping coverage counts.

    ``a`` contributes weight 1 and ``b`` weight 2, so the running sum is
    1 (a only), 2 (b only) or 3 (both) between consecutive sorted endpoints.
    """
    pos = torch.cat((a[0], a[1], b[0], b[1]))
    if pos.numel() == 0:
        return pos, pos
    weight = torch.cat(
        (
            torch.ones_like(a[0]),
            -torch.ones_like(a[1]),
            2 * torch.ones_like(b[0]),
            -2 * torch.ones_like(b[1]),
        )
    )
    order = torch.argsort(pos)
    pos = pos[order]
    cover = torch.cumsum(weight[order], dim=0)[:-1]
    if op == "union":
        keep = cover > 0
    elif op == "intersection":
        keep = cover == 3
    elif op == "difference":
        keep = cover == 1
    else:
        raise ValueError(f"unknown range operation {op!r}")
    keep = keep & (pos[:-1] < pos[1:])
    return _merge_sorted_ranges(pos[:-1][keep], pos[1:][keep])


_NATIVE_COMBINE_OPS = {"union": 0, "intersection": 1, "difference": 2}


def _native_moc_op(name: str) -> Any:
    """Native CPU range kernel (``torchsky::moc_*``) or None when unavailable."""
    return (
        getattr(torch.ops.torchsky, name, None) if _ops.has_fixed_dispatcher() else None
    )


def _native_ok(*tensors: Tensor) -> bool:
    return all(
        t.device.type == "cpu" and t.dtype == torch.int64 and t.ndim == 1
        for t in tensors
    )


def _uniq_to_range_tensors(uniq: Tensor, max_order: int) -> tuple[Tensor, Tensor]:
    """Sorted disjoint ``[start, end)`` NESTED ranges at ``max_order`` for UNIQ cells.

    CPU int64 inputs run the native kernel; invalid UNIQ values and every other
    device/dtype use the torch implementation (which raises the descriptive errors).
    """
    op = _native_moc_op("moc_uniq_to_ranges")
    if op is not None and _native_ok(uniq):
        try:
            return op(uniq.contiguous(), int(max_order))
        except RuntimeError:
            pass
    return _uniq_to_range_tensors_torch(uniq, max_order)


def _range_tensors_to_uniq(starts: Tensor, ends: Tensor, max_order: int) -> Tensor:
    """Maximal aligned UNIQ cells covering disjoint ranges, ordered by position."""
    op = _native_moc_op("moc_ranges_to_uniq")
    if op is not None and _native_ok(starts, ends):
        try:
            return op(starts.contiguous(), ends.contiguous(), int(max_order))
        except RuntimeError:
            pass
    return _range_tensors_to_uniq_torch(starts, ends, max_order)


def _combine_range_tensors(
    a: tuple[Tensor, Tensor], b: tuple[Tensor, Tensor], op: str
) -> tuple[Tensor, Tensor]:
    """Union / intersection / difference of two sorted disjoint range sets."""
    native = _native_moc_op("moc_combine_ranges")
    if native is not None and op in _NATIVE_COMBINE_OPS and _native_ok(*a, *b):
        return native(
            a[0].contiguous(),
            a[1].contiguous(),
            b[0].contiguous(),
            b[1].contiguous(),
            _NATIVE_COMBINE_OPS[op],
        )
    return _combine_range_tensors_torch(a, b, op)


def _complement_range_tensors(
    starts: Tensor, ends: Tensor, max_order: int
) -> tuple[Tensor, Tensor]:
    """Complement of sorted disjoint ranges within ``[0, 12 * 4**max_order)``."""
    total = 12 << (2 * int(max_order))
    native = _native_moc_op("moc_complement_ranges")
    if native is not None and _native_ok(starts, ends):
        return native(starts.contiguous(), ends.contiguous(), total)
    full = (
        torch.zeros((1,), dtype=torch.int64, device=starts.device),
        torch.full((1,), total, dtype=torch.int64, device=starts.device),
    )
    return _combine_range_tensors_torch(full, (starts, ends), "difference")


def _uniq_to_ranges(uniq: Tensor, max_order: int) -> list[tuple[int, int]]:
    starts, ends = _uniq_to_range_tensors(uniq, max_order)
    return list(zip(starts.tolist(), ends.tolist(), strict=True))


def _ranges_to_tensors(
    ranges: list[tuple[int, int]], device: torch.device | None = None
) -> tuple[Tensor, Tensor]:
    if not ranges:
        empty = torch.empty((0,), dtype=torch.int64, device=device)
        return empty, empty
    arr = torch.tensor(ranges, dtype=torch.int64, device=device)
    return arr[:, 0].contiguous(), arr[:, 1].contiguous()


def _union_ranges(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge two sorted half-open range lists."""
    s, e = _combine_range_tensors(
        _ranges_to_tensors(left), _ranges_to_tensors(right), "union"
    )
    return list(zip(s.tolist(), e.tolist(), strict=True))


def _ranges_to_uniq(ranges: list[tuple[int, int]], max_order: int) -> Tensor:
    starts, ends = _ranges_to_tensors(ranges)
    return _range_tensors_to_uniq(starts, ends, max_order)


def _normalize_moc(uniq: Tensor) -> Tensor:
    """Sort and merge redundant pixels in a MOC."""
    if uniq.numel() == 0:
        return uniq

    u = _as_int64_exact(uniq, name="uniq").reshape(-1)
    max_order = _max_order_from_uniq(u)
    starts, ends = _uniq_to_range_tensors(u, max_order)
    return _range_tensors_to_uniq(starts, ends, max_order)


def _normalize_bmoc(uniq: Tensor, flag: Tensor) -> tuple[Tensor, Tensor]:
    """Remove duplicate and parent/child aliases using boundary-dominant flags.

    When overlapping representations disagree, ``flag == 0`` wins. This
    conservative, order-independent rule prevents a partial cell from being
    promoted to strictly interior and is used by construction and BMOC algebra.
    """
    if uniq.numel() == 0:
        return uniq, flag

    orders = _order_from_uniq(uniq)
    max_order = int(orders.max().item())
    pixels = uniq - torch.bitwise_left_shift(torch.ones_like(uniq), 2 * orders + 2)
    shifts = 2 * (max_order - orders)
    starts = pixels << shifts
    ends = (pixels + 1) << shifts
    # Sort by (start ascending, end descending): parents precede descendants.
    by_end = torch.argsort(-ends, stable=True)
    by_start = torch.argsort(starts[by_end], stable=True)
    idx = by_end[by_start]
    s = starts[idx]
    e = ends[idx]
    prev_end = torch.cummax(e, dim=0).values
    top = torch.ones(s.numel(), dtype=torch.bool, device=s.device)
    top[1:] = s[1:] >= prev_end[:-1]
    group = torch.cumsum(top.to(torch.int64), dim=0) - 1
    ngroups = int(group[-1].item()) + 1
    group_flag = torch.full((ngroups,), 255, dtype=torch.int64, device=s.device)
    group_flag = group_flag.scatter_reduce(
        0, group, flag.to(s.device)[idx].to(torch.int64), reduce="amin"
    )
    normalized_uniq = uniq[idx[top]]
    normalized_flag = _to(group_flag, device=flag.device, dtype=torch.uint8)
    order = torch.argsort(normalized_uniq)
    return normalized_uniq[order], normalized_flag[order]


def _hierarchical_disc_bmoc(
    lon_deg: float,
    lat_deg: float,
    radius_deg: float,
    max_order: int,
    *,
    inclusive: bool,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    """Coarse-to-fine disc BMOC returning (uniq, flag) with flag=1 inside, 0 boundary."""
    order = 0
    pix = torch.arange(12, dtype=torch.int64, device=device)
    kept_uniq: list[Tensor] = []
    kept_flag: list[Tensor] = []
    lon0 = torch.as_tensor(lon_deg, device=device)
    lat0 = torch.as_tensor(lat_deg, device=device)
    radius = float(radius_deg)
    v_center = lonlat_to_unit_xyz(lon0, lat0)
    r_leaf = _healpix.max_pixel_radius(1 << max_order, degrees=True)
    while True:
        nside = 1 << order
        lon_p, lat_p = _healpix.pix2ang(nside, pix, nest=True, lonlat=True)
        v_p = lonlat_to_unit_xyz(lon_p, lat_p)
        cross = torch.linalg.norm(
            torch.cross(v_center.expand_as(v_p), v_p, dim=-1), dim=-1
        )
        dot = torch.clamp((v_center * v_p).sum(dim=-1), -1.0, 1.0)
        dist = torch.rad2deg(torch.atan2(cross, dot))
        r_pix = _healpix.max_pixel_radius(nside, degrees=True)
        fully_in = dist + r_pix <= radius
        fully_out = dist - r_pix > (radius + r_leaf if inclusive else radius)
        leaf_hit = dist <= (radius + r_leaf if inclusive else radius)
        if order == max_order:
            inside_leaf = pix[fully_in]
            if inside_leaf.numel() > 0:
                kept_uniq.append(nest2uniq(nside, inside_leaf))
                kept_flag.append(
                    torch.ones(inside_leaf.numel(), dtype=torch.uint8, device=device)
                )
            bndry_leaf = pix[leaf_hit & (~fully_in)]
            if bndry_leaf.numel() > 0:
                kept_uniq.append(nest2uniq(nside, bndry_leaf))
                kept_flag.append(
                    torch.zeros(bndry_leaf.numel(), dtype=torch.uint8, device=device)
                )
            break
        inside = pix[fully_in]
        if inside.numel() > 0:
            kept_uniq.append(nest2uniq(nside, inside))
            kept_flag.append(
                torch.ones(inside.numel(), dtype=torch.uint8, device=device)
            )
        boundary = pix[~(fully_in | fully_out)]
        if boundary.numel() == 0:
            break
        pix = _healpix.nest_children(boundary, levels=1).reshape(-1)
        order += 1
    if not kept_uniq:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.uint8, device=device),
        )
    return torch.cat(kept_uniq), torch.cat(kept_flag)


def _hierarchical_disc_uniq(
    lon_deg: float,
    lat_deg: float,
    radius_deg: float,
    max_order: int,
    *,
    inclusive: bool,
    device: torch.device,
) -> Tensor:
    """Coarse-to-fine disc MOC: keep interior cells, subdivide only the boundary."""
    uniq, _ = _hierarchical_disc_bmoc(
        lon_deg,
        lat_deg,
        radius_deg,
        max_order,
        inclusive=inclusive,
        device=device,
    )
    return uniq


def _hierarchical_polygon_bmoc(
    lon_deg: Tensor,
    lat_deg: Tensor,
    max_order: int,
) -> tuple[Tensor, Tensor]:
    """Coarse-to-fine polygon BMOC returning (uniq, flag) with flag=1 inside, 0 boundary."""
    poly_lon = torch.as_tensor(lon_deg)
    device = poly_lon.device
    poly_lat = torch.as_tensor(lat_deg, device=device)
    if poly_lon.numel() < 3:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.uint8, device=device),
        )
    verts = lonlat_to_unit_xyz(poly_lon, poly_lat)
    center_xyz = verts.mean(dim=0)
    center_xyz = center_xyz / torch.linalg.norm(center_xyz).clamp_min(1e-15)
    cen_lon, cen_lat = _healpix.xyz_to_lonlat(
        center_xyz[0], center_xyz[1], center_xyz[2]
    )
    poly_radius = float(
        great_circle_distance(cen_lon, cen_lat, poly_lon, poly_lat, degrees=True)
        .max()
        .item()
    )
    r_leaf = _healpix.max_pixel_radius(1 << max_order, degrees=True)
    order = 0
    pix = torch.arange(12, dtype=torch.int64, device=device)
    kept_uniq: list[Tensor] = []
    kept_flag: list[Tensor] = []
    while True:
        nside = 1 << order
        lon_p, lat_p = _healpix.pix2ang(nside, pix, nest=True, lonlat=True)
        dist = great_circle_distance(cen_lon, cen_lat, lon_p, lat_p, degrees=True)
        r_pix = _healpix.max_pixel_radius(nside, degrees=True)
        maybe = dist - r_pix <= poly_radius + r_leaf
        pix = pix[maybe]
        if pix.numel() == 0:
            break
        lon_p, lat_p = _healpix.pix2ang(nside, pix, nest=True, lonlat=True)
        center_in = spherical_polygon_contains(lon_p, lat_p, poly_lon, poly_lat)
        corners = _healpix.boundaries(nside, pix, step=1, nest=True)
        lon_c, lat_c = _healpix.xyz_to_lonlat(
            corners[:, 0, :].reshape(-1),
            corners[:, 1, :].reshape(-1),
            corners[:, 2, :].reshape(-1),
        )
        corner_in = spherical_polygon_contains(
            lon_c, lat_c, poly_lon, poly_lat
        ).reshape(pix.numel(), 4)
        fully_in = center_in & corner_in.all(dim=1)
        if order == max_order:
            inside_leaf = pix[fully_in]
            if inside_leaf.numel() > 0:
                kept_uniq.append(nest2uniq(nside, inside_leaf))
                kept_flag.append(
                    torch.ones(inside_leaf.numel(), dtype=torch.uint8, device=device)
                )
            any_in = center_in | corner_in.any(dim=1)
            bndry_leaf = pix[any_in & (~fully_in)]
            # Also include leaf cells containing polygon vertices (handles tiny footprints)
            vert_pix = _healpix.ang2pix(
                nside, poly_lon, poly_lat, nest=True, lonlat=True
            ).to(torch.int64)
            if inside_leaf.numel() > 0:
                vert_pix = vert_pix[~torch.isin(vert_pix, inside_leaf)]
            bndry_all = torch.unique(torch.cat([bndry_leaf, vert_pix]))
            if bndry_all.numel() > 0:
                kept_uniq.append(nest2uniq(nside, bndry_all))
                kept_flag.append(
                    torch.zeros(bndry_all.numel(), dtype=torch.uint8, device=device)
                )
            break
        # ponytail: 5-sample fully-inside can miss a concave dent; exact cell-in-polygon is the upgrade.
        inside = pix[fully_in]
        if inside.numel() > 0:
            kept_uniq.append(nest2uniq(nside, inside))
            kept_flag.append(
                torch.ones(inside.numel(), dtype=torch.uint8, device=device)
            )
        boundary = pix[~fully_in]
        if boundary.numel() == 0:
            break
        pix = _healpix.nest_children(boundary, levels=1).reshape(-1)
        order += 1
    if not kept_uniq:
        return (
            torch.empty(0, dtype=torch.int64, device=device),
            torch.empty(0, dtype=torch.uint8, device=device),
        )
    return torch.cat(kept_uniq), torch.cat(kept_flag)


def _hierarchical_polygon_uniq(
    lon_deg: Tensor,
    lat_deg: Tensor,
    max_order: int,
    *,
    inclusive: bool = False,
) -> Tensor:
    """Coarse-to-fine polygon MOC. Interior cells stay coarse; boundary subdivides."""
    uniq, flag = _hierarchical_polygon_bmoc(lon_deg, lat_deg, max_order)
    if inclusive or uniq.numel() == 0:
        return uniq
    # Standard MOC.from_polygon leaf rule: include interior cells (flag==1)
    # plus leaf cells whose center is inside the polygon.
    poly_lon = torch.as_tensor(lon_deg)
    poly_lat = torch.as_tensor(lat_deg, device=poly_lon.device)
    interior = uniq[flag > 0]
    bndry = uniq[flag == 0]
    if bndry.numel() == 0:
        return interior
    nside = 1 << max_order
    _, pix_b = uniq2nest(bndry)
    lon_p, lat_p = _healpix.pix2ang(nside, pix_b, nest=True, lonlat=True)
    c_in = spherical_polygon_contains(lon_p, lat_p, poly_lon, poly_lat)
    return torch.cat([interior, bndry[c_in]])


def _moc_table_hdu(path: str, *, spatial_only: bool = False) -> str | int:
    """Locate the IVOA MOC binary table: ``EXTNAME='MOC'`` or the first extension.

    The IVOA MOC standard puts the table in the first extension and identifies
    it by ``ORDERING='NUNIQ'`` (``TTYPE1='UNIQ'``); it does not require an
    ``EXTNAME``. With ``spatial_only``, time and space-time MOCs (``MOCDIM``
    other than ``SPACE``) and non-ICRS ``COORDSYS`` are rejected.
    """
    import torchfits

    try:
        header = torchfits.read_header(path, "MOC")
        hdu: str | int = "MOC"
    except ValueError:
        try:
            header = torchfits.read_header(path, 1)
        except (ValueError, OSError, IndexError) as exc:
            raise ValueError(f"No MOC binary table extension in {path!r}") from exc
        hdu = 1
        is_table = str(header.get("XTENSION", "")).strip() == "BINTABLE"
        is_nuniq = str(header.get("ORDERING", "")).strip().upper() == "NUNIQ" or (
            str(header.get("TTYPE1", "")).strip().upper() == "UNIQ"
        )
        if not (is_table and is_nuniq):
            raise ValueError(
                f"No MOC table in {path!r}: expected EXTNAME 'MOC' or a first "
                "binary-table extension with ORDERING='NUNIQ' or a UNIQ column"
            )
    dim = str(header.get("MOCDIM", "SPACE") or "SPACE").strip().upper()
    if spatial_only and dim != "SPACE":
        raise ValueError(f"{path!r} is a {dim} MOC; only spatial MOCs are supported")
    coordsys = str(header.get("COORDSYS", "C") or "C").strip().upper()
    if spatial_only and coordsys != "C":
        raise ValueError(
            f"{path!r} has COORDSYS={coordsys!r}; only ICRS ('C') MOCs are supported"
        )
    return hdu


def _uniq_column_from_table_dict(table: dict) -> Tensor:
    """Pick UNIQ column or the first 1D integer column (legacy MOC files)."""
    for key in ("UNIQ", "uniq"):
        if key in table:
            return torch.as_tensor(table[key], dtype=torch.int64).reshape(-1)
    for _name in sorted(table.keys()):
        val = table[_name]
        t = torch.as_tensor(val)
        if t.dtype in (torch.int32, torch.int64):
            return t.reshape(-1).to(torch.int64)
    raise ValueError(
        "MOC FITS table has no UNIQ column and no integer column suitable as UNIQ"
    )


def _parse_ascii_pixel_tokens(pix_str: str) -> list[int]:
    pixels: list[int] = []
    for token in pix_str.replace(",", " ").split():
        if "-" not in token:
            pixels.append(int(token))
            continue
        start_s, end_s = token.split("-", 1)
        start = int(start_s)
        end = int(end_s)
        if end < start:
            raise ValueError(f"invalid descending MOC pixel range {token!r}")
        pixels.extend(range(start, end + 1))
    return pixels


def _format_ascii_pixels(pixels: Tensor, *, compact: bool) -> str:
    values = [int(p) for p in pixels.tolist()]
    if not compact or len(values) <= 1:
        return " ".join(map(str, values))
    parts: list[str] = []
    start = values[0]
    prev = values[0]
    for value in values[1:]:
        if value == prev + 1:
            prev = value
            continue
        parts.append(f"{start}-{prev}" if start != prev else str(start))
        start = prev = value
    parts.append(f"{start}-{prev}" if start != prev else str(start))
    return " ".join(parts)


class MOC:
    """Multi-order sky coverage encoded as normalized IVOA NUNIQ cells.

    Args:
        uniq: Integer tensor of NUNIQ cell identifiers.

    Examples:
        >>> from torchsky.sphere import MOC
        >>> MOC.empty().is_empty
        True
    """

    def __init__(self, uniq: Tensor):
        self.uniq = _normalize_moc(_as_int64_exact(uniq, name="uniq"))

    @classmethod
    def empty(cls) -> "MOC":
        return cls(torch.empty(0, dtype=torch.int64))

    @property
    def is_empty(self) -> bool:
        return int(self.uniq.numel()) == 0

    def __bool__(self) -> bool:
        return int(self.uniq.numel()) > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, MOC):
            return False
        return bool(torch.equal(self.uniq, other.uniq.to(device=self.uniq.device)))

    @classmethod
    def full(cls) -> "MOC":
        return cls(nest2uniq(1, torch.arange(12, dtype=torch.int64)))

    def _cache(self) -> dict:
        """Per-instance memo (max order, range tensors), reset if ``uniq`` changes."""
        u = self.uniq
        key = (id(u), u._version, int(u.numel()))
        memo = self.__dict__.get("_memo")
        if memo is None or memo[0] != key:
            memo = (key, {})
            self.__dict__["_memo"] = memo
        return memo[1]

    @property
    def max_order(self) -> int:
        if self.uniq.numel() == 0:
            return 0
        cache = self._cache()
        if "max_order" not in cache:
            cache["max_order"] = _order_of_uniq_scalar(int(self.uniq.max().item()))
        return cache["max_order"]

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        """Check if points are contained in the MOC."""
        lon = _healpix._as_float64(lon_deg)
        lat = _healpix._as_float64(lat_deg)
        lat = lat.to(device=lon.device)
        lon, lat = torch.broadcast_tensors(lon, lat)

        if self.uniq.numel() == 0:
            return torch.zeros_like(lon, dtype=torch.bool)

        max_o = self.max_order
        nside_max = 1 << max_o
        p_max = _healpix.ang2pix(nside_max, lon, lat, nest=True, lonlat=True)
        return self.contains_pixels(nside_max, p_max, nest=True)

    def contains_rows(self, catalog: Any) -> Tensor:
        """Boolean mask for catalog rows. Prefer ``Catalog.inside(moc)``."""
        return self.contains(catalog.lon, catalog.lat)

    def contains_pixels(
        self,
        nside: int,
        pixels: Tensor | int,
        *,
        nest: bool = True,
    ) -> Tensor:
        """Check whether HEALPix pixels are covered by this MOC.

        ``pixels`` are interpreted at ``nside``. MOC cells are internally NUNIQ /
        NESTED, so RING inputs are converted exactly when ``nest=False``.
        """
        pix = torch.as_tensor(pixels, dtype=torch.int64)
        shape = pix.shape
        if self.uniq.numel() == 0:
            return torch.zeros(shape, dtype=torch.bool, device=pix.device)

        order = _order_from_nside(nside)
        if order < self.max_order:
            raise ValueError("nside must be at least the MOC maximum order resolution")
        pix_nest = pix.reshape(-1)
        if not nest and pix_nest.numel() > 0:
            pix_nest = _healpix.ring2nest(int(nside), pix_nest)

        starts, ends = self._range_tensors(order)
        starts = starts.to(pix.device)
        ends = ends.to(pix.device)

        idx = torch.searchsorted(starts, pix_nest, right=True) - 1
        in_range = idx >= 0
        idx_safe = idx.clamp(min=0)
        mask = in_range & (pix_nest < ends[idx_safe])
        return mask.reshape(shape)

    def overlap_fraction(self, uniq: Tensor | int) -> Tensor:
        """Fraction of each canonical UNIQ cell covered by this MOC."""
        query = _as_int64_exact(uniq, name="uniq")
        shape = query.shape
        query = query.reshape(-1)
        fraction_dtype = _work_dtype(query.device)
        if query.numel() == 0:
            return torch.zeros(shape, dtype=fraction_dtype, device=query.device)
        query_orders = _order_from_uniq(query)
        if self.uniq.numel() == 0:
            return torch.zeros(shape, dtype=fraction_dtype, device=query.device)

        max_order = max(self.max_order, int(query_orders.max().item()))
        starts, ends = self._range_tensors(max_order)
        starts = starts.to(query.device)
        ends = ends.to(query.device)
        lengths = ends - starts
        cumulative = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=query.device),
                torch.cumsum(lengths, dim=0),
            )
        )
        _, pixels = uniq2nest(query)
        shifts = 2 * (max_order - query_orders)
        query_starts = pixels << shifts
        query_ends = (pixels + 1) << shifts

        def covered_before(position: Tensor) -> Tensor:
            count = torch.searchsorted(ends, position, right=True)
            safe = count.clamp(max=ends.numel() - 1)
            partial = torch.where(
                count < ends.numel(),
                torch.minimum((position - starts[safe]).clamp_min(0), lengths[safe]),
                torch.zeros_like(position),
            )
            return cumulative[count] + partial

        overlap = covered_before(query_ends) - covered_before(query_starts)
        cell_size = query_ends - query_starts
        return (overlap.to(fraction_dtype) / cell_size.to(fraction_dtype)).reshape(
            shape
        )

    def intersects_uniq(self, uniq: Tensor | int) -> Tensor:
        """Return which canonical UNIQ cells intersect this MOC."""
        return self.overlap_fraction(uniq) > 0.0

    def contains_uniq(self, uniq: Tensor | int) -> Tensor:
        """Return which canonical UNIQ cells are fully covered by this MOC."""
        return self.overlap_fraction(uniq) == 1.0

    def filter_catalog(self, lon_deg: Tensor, lat_deg: Tensor) -> tuple[Tensor, Tensor]:
        """Filter a catalogue by point-in-MOC containment. Returns (lon, lat) of contained points."""
        mask = self.contains(lon_deg, lat_deg)
        return lon_deg[mask], lat_deg[mask]

    @property
    def area(self) -> float:
        """Total area covered by the MOC in steradians."""
        if self.uniq.numel() == 0:
            return 0.0
        # Exact and device-independent: count cells at the deepest order in
        # integer arithmetic (no float64 tensors, which MPS lacks), then scale.
        order = self.max_order
        starts, ends = self._range_tensors(order)
        n_cells = int((ends - starts).sum().item())
        return n_cells * (4.0 * math.pi) / (12.0 * float(4**order))

    @property
    def area_sq_deg(self) -> float:
        """Total area covered by the MOC in square degrees."""
        return self.area * ((180.0 / math.pi) * (180.0 / math.pi))

    def contains_moc(self, other: "MOC") -> bool:
        """Return True if this MOC contains all of other MOC (subset check)."""
        # self contains other if intersection(self, other) == other
        inter = self.intersection(other)
        return torch.equal(inter.uniq, other.uniq)

    def overlaps(self, other: "MOC") -> bool:
        """Return whether two footprint MOCs have non-empty intersection."""
        return bool(self.intersection(other))

    def _binary_op(self, other: "MOC", op: str) -> "MOC":
        max_o = max(self.max_order, other.max_order)
        device = self.uniq.device
        left = self._range_tensors(max_o)
        right = tuple(t.to(device) for t in other._range_tensors(max_o))
        return MOC._from_range_tensors(
            *_combine_range_tensors(left, right, op),  # type: ignore[arg-type]
            max_o,
        )

    def union(self, other: "MOC") -> "MOC":
        if self.uniq.numel() == 0:
            return MOC(other.uniq)
        if other.uniq.numel() == 0:
            return MOC(self.uniq)
        return self._binary_op(other, "union")

    def intersection(self, other: "MOC") -> "MOC":
        """Cells covered by both MOCs (range sweep at the finer ``max_order``)."""
        return self._binary_op(other, "intersection")

    def difference(self, other: "MOC") -> "MOC":
        """Subtract other MOC from this MOC."""
        return self._binary_op(other, "difference")

    def __or__(self, other: "MOC") -> "MOC":
        return self.union(other)

    def __and__(self, other: "MOC") -> "MOC":
        return self.intersection(other)

    def __sub__(self, other: "MOC") -> "MOC":
        return self.difference(other)

    def degrade(self, max_order: int) -> "MOC":
        """Drop cells finer than ``max_order`` onto their nested parents."""
        order = int(max_order)
        if order < 0:
            raise ValueError("max_order must be non-negative")
        if self.uniq.numel() == 0:
            return type(self).empty()
        if order >= self.max_order:
            return type(self)(self.uniq)
        native = _native_moc_op("moc_degrade_uniq")
        if native is not None and _native_ok(self.uniq):
            return type(self)(native(self.uniq.contiguous(), order))
        nsides, pix = uniq2nest(self.uniq)
        cell_orders = _order_from_uniq(self.uniq)
        delta = (cell_orders - order).clamp(min=0)
        pix_d = pix >> (2 * delta)
        nside_d = torch.where(delta > 0, torch.full_like(nsides, 1 << order), nsides)
        return type(self)(nest2uniq(nside_d, pix_d))

    def _range_tensors(self, max_order: int) -> tuple[Tensor, Tensor]:
        cache = self._cache()
        key = ("ranges", int(max_order))
        if key not in cache:
            cache[key] = _uniq_to_range_tensors(self.uniq, max_order)
        return cache[key]

    def _to_ranges(self, max_order: int) -> list[tuple[int, int]]:
        return _uniq_to_ranges(self.uniq, max_order)

    @classmethod
    def _from_range_tensors(cls, starts: Tensor, ends: Tensor, max_order: int) -> "MOC":
        """Convert merged ranges to UNIQ cells without a second normalize pass."""
        moc = cls.__new__(cls)
        moc.uniq = _range_tensors_to_uniq(starts, ends, max_order)
        # The ranges are already the canonical merged form: chain operations
        # without converting the UNIQ cells back.
        moc._cache()[("ranges", int(max_order))] = (starts, ends)
        return moc

    @classmethod
    def _from_ranges(cls, ranges: list[tuple[int, int]], max_order: int) -> "MOC":
        """Convert merged ranges to UNIQ cells without a second normalize pass."""
        return cls._from_range_tensors(*_ranges_to_tensors(ranges), max_order)

    def _flat_pixels(self, nside: int) -> Tensor:
        max_o = _order_from_nside(nside)
        if max_o < self.max_order:
            raise ValueError("nside must be at least the MOC maximum order resolution")
        starts, ends = self._range_tensors(max_o)
        counts = ends - starts
        total = int(counts.sum().item())
        if total == 0:
            return torch.empty((0,), dtype=torch.int64, device=self.uniq.device)
        offs = torch.cumsum(counts, dim=0) - counts
        return torch.arange(total, device=starts.device) + torch.repeat_interleave(
            starts - offs, counts
        )

    def to_sparse_map(
        self,
        nside: int,
        sentinel: float = 0.0,
        *,
        nside_coverage: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> SparseHealpixMap:
        """Realize MOC coverage as a sparse HEALPix map at fixed resolution."""
        nside_i = int(nside)
        if nside_coverage is None:
            nside_coverage = max(1, nside_i // 8)
        return SparseHealpixMap.from_moc(
            self,
            nside=nside_i,
            nside_coverage=nside_coverage,
            value=1.0,
            sentinel=sentinel,
            dtype=dtype,
        )

    def flatten(self, nside: int) -> Tensor:
        """Project MOC to a dense HEALPix mask at fixed resolution."""
        pix = self._flat_pixels(nside)
        mask = torch.zeros(12 * nside**2, dtype=torch.bool, device=pix.device)
        if pix.numel() > 0:
            mask[pix] = True
        return mask

    def to_pixelized_region(self, nside: int, *, nest: bool = True) -> PixelizedRegion:
        """Realize this MOC as a ``PixelizedRegion`` at ``nside``.

        MOC UNIQ cells are NESTED by definition. Set ``nest=False`` to receive
        a RING-ordered ``PixelizedRegion`` after exact ordering conversion.
        """
        pix = self._flat_pixels(nside)
        if not nest and pix.numel() > 0:
            pix = _healpix.nest2ring(int(nside), pix)
        return PixelizedRegion(nside=int(nside), nest=nest, pixels=pix)

    @classmethod
    def from_sparse_map(cls, smap: SparseHealpixMap) -> "MOC":
        """Create MOC from the coverage of a sparse map."""
        # Get all sparse pixels
        pix = smap.get_covered_pixels()
        max_order = int(math.log2(smap.nside_sparse))

        # Convert to ranges
        if pix.numel() == 0:
            return cls(torch.empty((0,), dtype=torch.int64))
        if not smap.nest:
            pix = _healpix.ring2nest(smap.nside_sparse, pix)

        pix = torch.sort(pix)[0]
        # Contiguous runs: break wherever consecutive sorted pixels leave a gap.
        breaks = torch.nonzero(torch.diff(pix) > 1).reshape(-1)
        starts = pix[torch.cat((breaks.new_zeros(1), breaks + 1))]
        ends = pix[torch.cat((breaks, breaks.new_full((1,), pix.numel() - 1)))] + 1
        return cls._from_range_tensors(starts, ends, max_order)

    @classmethod
    def from_lonlat(
        cls,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        max_order: int,
    ) -> "MOC":
        """Coverage of catalog points, compacted to parent UNIQ cells."""
        order = int(max_order)
        nside = _nside_from_max_order(order)
        pix = _healpix.ang2pix(nside, lon_deg, lat_deg, nest=True, lonlat=True).to(
            torch.int64
        )
        return cls(nest2uniq(nside, pix))

    @classmethod
    def from_circle(
        cls,
        lon_deg: float,
        lat_deg: float,
        radius_deg: float,
        max_order: int = 10,
        inclusive: bool = True,
    ) -> "MOC":
        """Create a MOC from a circular cap without listing every leaf pixel."""
        order = int(max_order)
        _nside_from_max_order(order)
        if float(radius_deg) <= 0.0:
            return cls.empty()
        lon_t = torch.as_tensor(lon_deg)
        uniq = _hierarchical_disc_uniq(
            float(lon_deg),
            float(lat_deg),
            float(radius_deg),
            order,
            inclusive=inclusive,
            device=lon_t.device,
        )
        return cls(uniq)

    @classmethod
    def from_polygon(
        cls,
        lon_deg: Tensor | list[float],
        lat_deg: Tensor | list[float],
        max_order: int = 10,
    ) -> "MOC":
        """Create a MOC from a simple spherical polygon.

        Convex and non-convex simple polygons are supported. Interior cells are
        kept coarse; only the boundary is subdivided to ``max_order``.
        """
        order = int(max_order)
        _nside_from_max_order(order)
        uniq = _hierarchical_polygon_uniq(
            torch.as_tensor(lon_deg),
            torch.as_tensor(lat_deg),
            order,
        )
        return cls(uniq)

    @classmethod
    def from_region(cls, region: Any, max_order: int | None = None) -> "MOC":
        """Create a MOC from a torchsky spherical region or PixelizedRegion.

        Pixelized regions preserve their current resolution by default. Other
        regions are pixelized at ``2**max_order`` in NESTED ordering before
        conversion to UNIQ, defaulting to order 10 when ``max_order`` is omitted.
        Pass a matching ``max_order`` with a ``PixelizedRegion`` to assert the
        expected resolution.
        """
        if isinstance(region, PixelizedRegion):
            nside = int(region.nside)
            if max_order is not None and nside != _nside_from_max_order(max_order):
                raise ValueError(
                    "PixelizedRegion nside does not match requested max_order"
                )
            pix = region.pixels.to(dtype=torch.int64)
            if not region.nest and pix.numel() > 0:
                pix = _healpix.ring2nest(nside, pix)
            return cls(nest2uniq(nside, pix))

        order = 10 if max_order is None else int(max_order)
        nside = _nside_from_max_order(order)
        if hasattr(region, "pixelize"):
            pixelized = region.pixelize(nside=nside, nest=True)
            pix = torch.as_tensor(pixelized.pixels, dtype=torch.int64)
            return cls(nest2uniq(nside, pix))
        if hasattr(region, "query_pixels"):
            pix = torch.as_tensor(
                region.query_pixels(nside, nest=True), dtype=torch.int64
            )
            return cls(nest2uniq(nside, pix))
        raise TypeError(
            "region must be a PixelizedRegion or expose pixelize/query_pixels"
        )

    @classmethod
    def from_threshold(
        cls,
        smap: SparseHealpixMap,
        threshold: float,
        op: Literal[">", ">=", "<", "<="] = ">",
    ) -> "MOC":
        """Create a MOC from pixels in a sparse map that satisfy a threshold."""
        if op == ">":
            mask = smap.values > threshold
        elif op == ">=":
            mask = smap.values >= threshold
        elif op == "<":
            mask = smap.values < threshold
        elif op == "<=":
            mask = smap.values <= threshold
        else:
            raise ValueError(f"Invalid operator {op}")

        if mask.ndim > 1:
            mask = torch.any(mask.reshape(-1, mask.shape[-1]), dim=0)
        if mask.shape[-1] != smap.pixels.numel():
            raise ValueError("threshold mask must align with sparse-map pixels")
        if not bool(mask.any()):
            return cls(torch.empty((0,), dtype=torch.int64))

        pix = smap.pixels[mask]
        if not smap.nest:
            pix = _healpix.ring2nest(smap.nside_sparse, pix)
        uniqs = nest2uniq(smap.nside_sparse, pix)
        return cls(uniqs)

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write the MOC to a FITS file following the IVOA MOC standard.

        Writes the table through :mod:`torchfits`.
        """
        from torchfits import write

        header = {
            "PIXTYPE": "HEALPIX",
            "ORDERING": "NUNIQ",
            "COORDSYS": "C",
            "MOCORDER": self.max_order,
            "MOCTOOL": "torchsky",
            "EXTNAME": "MOC",
        }
        u = self.uniq.detach().cpu().reshape(-1).to(torch.int64)
        write(filename, {"UNIQ": u}, header=header, overwrite=overwrite)

    @classmethod
    def read_fits(cls, filename: str) -> "MOC":
        """Read a MOC from an IVOA FITS file through :mod:`torchfits`."""
        from torchfits.table import read_torch as read_table

        hdu = _moc_table_hdu(filename, spatial_only=True)
        table = read_table(filename, hdu=hdu)
        uniqs = _uniq_column_from_table_dict(table)
        return cls(uniqs)

    @classmethod
    def from_mocpy(cls, moc: Any) -> "MOC":
        """Import a host-side ``mocpy.MOC`` through IVOA JSON."""
        payload = moc.to_json()
        if isinstance(payload, str):
            return cls.from_json(payload)
        return cls.from_json(json.dumps(payload))

    def to_mocpy(self) -> Any:
        """Export to ``mocpy.MOC`` when mocpy is installed."""
        try:
            from mocpy import MOC as MocpyMOC
        except ImportError as exc:
            raise ImportError("mocpy is required for to_mocpy()") from exc

        return MocpyMOC.from_json(json.loads(self.to_json()))

    def boundary(self) -> list[Tensor]:
        """
        Return the boundaries of the MOC as a list of vertex loops.
        Each loop is a Tensor of shape [N, 2] (lon, lat) in degrees.
        """
        if self.uniq.numel() == 0:
            return []

        nside = 1 << self.max_order
        pix = self._flat_pixels(nside)
        if pix.numel() == 0:
            return []

        corners_xyz = _healpix.boundaries(nside, pix, step=1, nest=True).permute(
            0, 2, 1
        )
        quant_scale = 1.0e12

        def _key(v: Tensor) -> tuple[int, int, int]:
            vals = torch.round(v.cpu().to(torch.float64) * quant_scale).to(torch.int64)
            return int(vals[0].item()), int(vals[1].item()), int(vals[2].item())

        vertex_xyz: dict[tuple[int, int, int], Tensor] = {}
        undirected_edges: dict[
            tuple[tuple[int, int, int], tuple[int, int, int]],
            tuple[tuple[int, int, int], tuple[int, int, int]],
        ] = {}
        for poly in corners_xyz:
            keys = [_key(poly[i]) for i in range(poly.shape[0])]
            for key, xyz in zip(keys, poly, strict=False):
                vertex_xyz[key] = xyz.cpu().to(torch.float64)
            for i in range(poly.shape[0]):
                a = keys[i]
                b = keys[(i + 1) % poly.shape[0]]
                edge_key = (a, b) if a <= b else (b, a)
                if edge_key in undirected_edges:
                    del undirected_edges[edge_key]
                else:
                    undirected_edges[edge_key] = (a, b)

        if not undirected_edges:
            return []

        adjacency: dict[tuple[int, int, int], list[tuple[int, int, int]]] = {}
        for a, b in undirected_edges.values():
            adjacency.setdefault(a, []).append(b)
            adjacency.setdefault(b, []).append(a)

        loops: list[Tensor] = []
        unused = {tuple(sorted((a, b))) for a, b in undirected_edges.values()}
        while unused:
            start_edge = next(iter(unused))
            start = start_edge[0]
            current = start_edge[1]
            previous = start
            path = [start, current]
            unused.remove(start_edge)

            while current != start:
                neighbors = adjacency[current]
                next_vertex = None
                for candidate in neighbors:
                    if (
                        candidate != previous
                        and tuple(sorted((current, candidate))) in unused
                    ):
                        next_vertex = candidate
                        break
                if next_vertex is None:
                    break
                unused.remove(tuple(sorted((current, next_vertex))))
                previous, current = current, next_vertex
                path.append(current)

            xyz = torch.stack([vertex_xyz[key] for key in path], dim=0)
            lon, lat = _healpix.xyz_to_lonlat(xyz[:, 0], xyz[:, 1], xyz[:, 2])
            loops.append(torch.stack((lon, lat), dim=-1).to(torch.float64))
        return loops

    def to_ascii(self, *, compact: bool = False) -> str:
        """Return IVOA ASCII representation of the MOC."""
        # ASCII format is: ORDER/PIX1 PIX2 ORDER2/PIX3 ...
        if self.uniq.numel() == 0:
            return ""
        nsides, pix_nested = uniq2nest(self.uniq)
        orders = (
            torch.log2(nsides.to(torch.float32)).round().to(torch.int64)
        )  # exact: nside is a power of two

        unique_orders = torch.unique(orders, sorted=True)
        parts = []
        for o in unique_orders:
            mask = orders == o
            o_pix = pix_nested[mask].sort()[0]
            pix_str = _format_ascii_pixels(o_pix, compact=compact)
            parts.append(f"{int(o.item())}/{pix_str}")
        return " ".join(parts)

    @classmethod
    def from_ascii(cls, ascii_str: str) -> "MOC":
        """Create a MOC from an IVOA ASCII string."""
        if not ascii_str.strip():
            return cls(torch.empty((0,), dtype=torch.int64))

        uniqs = []
        # Standard: ORDER/PIX PIX ...
        # We split by ORDER/ prefixes
        import re

        parts = re.split(r"(\d+)/", ascii_str.strip())
        # parts will be ['', '0', '0 1 2 ', '1', '4 5'] etc.
        for i in range(1, len(parts), 2):
            order = int(parts[i])
            pix_str = parts[i + 1].strip()
            if not pix_str:
                continue
            nside = 1 << order
            pix_list = _parse_ascii_pixel_tokens(pix_str)
            uniqs.append(nest2uniq(nside, torch.tensor(pix_list, dtype=torch.int64)))

        if not uniqs:
            return cls(torch.empty((0,), dtype=torch.int64))
        return cls(torch.cat(uniqs))

    def to_json(self) -> str:
        """Return IVOA JSON representation of the MOC."""
        if self.uniq.numel() == 0:
            return "{}"

        nsides, pix_nested = uniq2nest(self.uniq)
        orders = (
            torch.log2(nsides.to(torch.float32)).round().to(torch.int64)
        )  # exact: nside is a power of two

        unique_orders = torch.unique(orders, sorted=True)
        data = {}
        for o in unique_orders:
            mask = orders == o
            o_pix = pix_nested[mask].sort()[0]
            data[str(int(o.item()))] = o_pix.tolist()

        return json.dumps(data)

    @classmethod
    def from_json(cls, json_str: str) -> "MOC":
        """Create a MOC from an IVOA JSON string."""
        if not json_str.strip():
            return cls(torch.empty((0,), dtype=torch.int64))

        data = json.loads(json_str)
        if not data:
            return cls(torch.empty((0,), dtype=torch.int64))
        if not isinstance(data, dict):
            raise ValueError("MOC JSON payload must be a dict of order -> pixels")

        uniqs: list[Tensor] = []
        for order_str, pixels in data.items():
            order = int(order_str)
            nside = 1 << order
            pix_t = torch.as_tensor(pixels, dtype=torch.int64)
            if pix_t.numel() == 0:
                continue
            uniqs.append(nest2uniq(nside, pix_t.reshape(-1)))

        if not uniqs:
            return cls(torch.empty((0,), dtype=torch.int64))
        return cls(torch.cat(uniqs))

    @classmethod
    def from_wcs(
        cls,
        wcs: Any,
        shape: tuple[int, int] | None = None,
        *,
        max_order: int = 10,
        mask: Tensor | None = None,
    ) -> "MOC":
        """Build a MOC from a 2D WCS (and optional valid-pixel mask)."""
        bmoc = BMOC.from_wcs(wcs, shape=shape, max_order=max_order, mask=mask)
        return bmoc.to_moc(include_boundary=True)

    @classmethod
    def from_observations(
        cls,
        observations: Any,
        *,
        max_order: int = 10,
    ) -> "MOC":
        """Build the union MOC of a sequence of observations or WCS objects."""
        moc = cls.empty()
        for obs in observations:
            if hasattr(obs, "wcs") and hasattr(obs, "data"):
                H, W = int(obs.data.shape[-2]), int(obs.data.shape[-1])
                valid_mask = None
                if getattr(obs, "mask", None) is not None:
                    valid_mask = ~torch.as_tensor(obs.mask, dtype=torch.bool)
                w_plane = getattr(obs, "weight_map", None)
                if w_plane is None and isinstance(getattr(obs, "weight", None), Tensor):
                    w_t = torch.as_tensor(obs.weight)
                    if w_t.ndim >= 2:
                        w_plane = w_t
                if w_plane is not None:
                    w_ok = torch.as_tensor(w_plane) > 0
                    valid_mask = w_ok if valid_mask is None else (valid_mask & w_ok)
                obs_moc = cls.from_wcs(
                    obs.wcs,
                    shape=(H, W),
                    max_order=max_order,
                    mask=valid_mask,
                )
            elif isinstance(obs, tuple) and len(obs) == 2:
                obs_moc = cls.from_wcs(obs[0], shape=obs[1], max_order=max_order)
            else:
                obs_moc = cls.from_wcs(obs, max_order=max_order)
            moc = moc.union(obs_moc)
        return moc

    def sample_randoms(
        self,
        n: int,
        *,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw ``n`` points uniformly in solid angle inside this MOC."""
        return sample_randoms(n, self, generator=generator, device=device)

    def prepare_random_sampler(
        self, *, device: torch.device | str | None = None
    ) -> PreparedRandomSampler:
        """Snapshot immutable, device-bound state for repeated random draws.

        Mutating ``self.uniq`` afterward does not update the returned sampler;
        call this method again to prepare from the changed MOC.
        """
        return _prepare_moc_random_sampler(self.uniq, device=device)


def _wcs_perimeter_lonlat(
    wcs: Any,
    shape: tuple[int, int] | None = None,
    *,
    edge_steps: int = 16,
) -> tuple[Tensor, Tensor, int, int]:
    if shape is None:
        arr_shape = getattr(wcs, "array_shape", None)
        if arr_shape is not None and len(arr_shape) >= 2:
            H, W = int(arr_shape[-2]), int(arr_shape[-1])
        else:
            pix_shape = getattr(wcs, "pixel_shape", None)
            if pix_shape is not None and len(pix_shape) >= 2:
                W, H = int(pix_shape[0]), int(pix_shape[1])
            else:
                raise ValueError(
                    "shape=(H, W) must be provided when wcs has no pixel_shape/array_shape"
                )
    else:
        H, W = int(shape[0]), int(shape[1])

    device = getattr(wcs, "crval", torch.tensor(0.0)).device
    dtype = getattr(wcs, "crval", torch.tensor(0.0, dtype=torch.float64)).dtype
    steps = max(2, int(edge_steps))
    t = torch.linspace(0.0, 1.0, steps + 1, device=device, dtype=dtype)[:-1]
    x0, x1 = -0.5, float(W) - 0.5
    y0, y1 = -0.5, float(H) - 0.5
    xs = torch.cat(
        [
            x0 + t * (x1 - x0),
            torch.full_like(t, x1),
            x1 - t * (x1 - x0),
            torch.full_like(t, x0),
        ]
    )
    ys = torch.cat(
        [
            torch.full_like(t, y0),
            y0 + t * (y1 - y0),
            torch.full_like(t, y1),
            y1 - t * (y1 - y0),
        ]
    )
    ra, dec = wcs.pixel_to_world(xs, ys)
    return ra.reshape(-1), dec.reshape(-1), H, W


def _bmoc_ranges(bmoc: "BMOC", max_order: int) -> tuple[Tensor, Tensor, Tensor]:
    """Start-sorted ``(starts, ends, flags)`` tensors of a BMOC at ``max_order``."""
    if bmoc.uniq.numel() == 0:
        empty = torch.empty((0,), dtype=torch.int64, device=bmoc.uniq.device)
        return empty, empty, empty
    _, pixels = uniq2nest(bmoc.uniq)
    orders = _order_from_uniq(bmoc.uniq)
    shifts = 2 * (max_order - orders)
    starts = pixels << shifts
    ends = (pixels + 1) << shifts
    indices = torch.argsort(starts)
    return starts[indices], ends[indices], bmoc.flag.to(torch.int64)[indices]


def _combine_bmoc(left: "BMOC", right: "BMOC", operation: str) -> "BMOC":
    max_order = max(left.max_order, right.max_order)
    device = left.uniq.device if left.uniq.numel() > 0 else right.uniq.device
    l_s, l_e, l_f = (t.to(device) for t in _bmoc_ranges(left, max_order))
    r_s, r_e, r_f = (t.to(device) for t in _bmoc_ranges(right, max_order))
    endpoints = torch.unique(torch.cat((l_s, l_e, r_s, r_e)))
    if endpoints.numel() < 2:
        return BMOC.empty()
    seg_start = endpoints[:-1]
    seg_end = endpoints[1:]

    def state_at(starts: Tensor, ends: Tensor, flags: Tensor) -> Tensor:
        if starts.numel() == 0:
            return torch.full_like(seg_start, -1)
        index = torch.searchsorted(starts, seg_start, right=True) - 1
        safe = index.clamp_min(0)
        valid = (index >= 0) & (seg_start < ends[safe])
        return torch.where(valid, flags[safe], torch.full_like(flags[safe], -1))

    left_state = state_at(l_s, l_e, l_f)
    right_state = state_at(r_s, r_e, r_f)
    minus_one = torch.full_like(left_state, -1)
    if operation == "union":
        state = torch.maximum(left_state, right_state)
    elif operation == "intersection":
        lo = torch.minimum(left_state, right_state)
        state = torch.where(lo >= 0, lo, minus_one)
    else:
        state = torch.where(
            (left_state >= 0) & (right_state < 0), left_state, minus_one
        )

    uniq_parts: list[Tensor] = []
    flag_parts: list[Tensor] = []
    for value in (0, 1):
        sel = state == value
        if not bool(sel.any()):
            continue
        starts, ends = _merge_sorted_ranges(seg_start[sel], seg_end[sel])
        cells = _range_tensors_to_uniq(starts, ends, max_order)
        uniq_parts.append(cells)
        flag_parts.append(
            torch.full((cells.numel(),), value, dtype=torch.uint8, device=device)
        )
    if not uniq_parts:
        return BMOC.empty()
    return BMOC(torch.cat(uniq_parts), torch.cat(flag_parts))


class BMOC:
    """Boolean Multi-Order Coverage map storing UNIQ cells and interior/boundary flags.

    ``flag == 1`` indicates a cell completely inside the region; ``flag == 0``
    indicates a partial/boundary cell at ``max_order``.

    Overlapping duplicate or parent/child cells are represented once. If their
    flags disagree, boundary (``0``) deterministically dominates interior
    (``1``), so normalization never overstates strictly interior coverage.
    """

    def __init__(self, uniq: Tensor, flag: Tensor | None = None):
        u = _as_int64_exact(uniq, name="uniq").reshape(-1)
        if flag is None:
            f = torch.ones_like(u, dtype=torch.uint8)
        else:
            raw_flag = torch.as_tensor(flag, device=u.device).reshape(-1)
            if raw_flag.numel() > 0 and bool(((raw_flag != 0) & (raw_flag != 1)).any()):
                raise ValueError("flag values must be 0 or 1")
            f = raw_flag.to(torch.uint8)
            if f.numel() != u.numel():
                raise ValueError("flag must have the same length as uniq")
        self.uniq, self.flag = _normalize_bmoc(u, f)

    @classmethod
    def empty(cls) -> "BMOC":
        return cls(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.uint8),
        )

    @property
    def is_empty(self) -> bool:
        return int(self.uniq.numel()) == 0

    def __bool__(self) -> bool:
        return int(self.uniq.numel()) > 0

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BMOC):
            return False
        return bool(
            torch.equal(self.uniq, other.uniq.to(device=self.uniq.device))
            and torch.equal(self.flag, other.flag.to(device=self.flag.device))
        )

    @classmethod
    def full(cls) -> "BMOC":
        u = nest2uniq(1, torch.arange(12, dtype=torch.int64))
        return cls(u, torch.ones(12, dtype=torch.uint8))

    @property
    def max_order(self) -> int:
        return _max_order_from_uniq(self.uniq)

    @property
    def area(self) -> float:
        """Total area (interior + boundary) in steradians."""
        if self.uniq.numel() == 0:
            return 0.0
        nsides, _ = uniq2nest(self.uniq)
        nsides_f64 = nsides.cpu().to(torch.float64)
        return float(
            ((4.0 * math.pi) / (12.0 * (nsides_f64 * nsides_f64))).sum().item()
        )

    @property
    def area_sq_deg(self) -> float:
        return self.area * ((180.0 / math.pi) * (180.0 / math.pi))

    @property
    def interior_area(self) -> float:
        """Area of strictly interior cells (``flag == 1``) in steradians."""
        ins = self.uniq[self.flag > 0]
        if ins.numel() == 0:
            return 0.0
        nsides, _ = uniq2nest(ins)
        nsides_f64 = nsides.cpu().to(torch.float64)
        return float(
            ((4.0 * math.pi) / (12.0 * (nsides_f64 * nsides_f64))).sum().item()
        )

    @property
    def interior_area_sq_deg(self) -> float:
        return self.interior_area * ((180.0 / math.pi) * (180.0 / math.pi))

    def to_moc(self, *, include_boundary: bool = True) -> MOC:
        """Convert to a standard ``MOC`` (optionally keeping only interior cells)."""
        if include_boundary:
            return MOC(self.uniq)
        return MOC(self.uniq[self.flag > 0])

    def interior_moc(self) -> MOC:
        """Return the ``MOC`` of strictly interior cells (``flag == 1``)."""
        return MOC(self.uniq[self.flag > 0])

    def boundary_moc(self) -> MOC:
        """Return the ``MOC`` of boundary cells (``flag == 0``)."""
        return MOC(self.uniq[self.flag == 0])

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        include_boundary: bool = True,
    ) -> Tensor:
        """Test whether points fall inside the BMOC."""
        return self.to_moc(include_boundary=include_boundary).contains(lon_deg, lat_deg)

    def union(self, other: "BMOC") -> "BMOC":
        """Union coverage; interior dominates boundary where operands overlap."""
        return _combine_bmoc(self, other, "union")

    def intersection(self, other: "BMOC") -> "BMOC":
        """Intersect coverage; boundary dominates interior where both overlap."""
        return _combine_bmoc(self, other, "intersection")

    def difference(self, other: "BMOC") -> "BMOC":
        """Subtract all covered cells in ``other``, preserving remaining flags."""
        return _combine_bmoc(self, other, "difference")

    def __or__(self, other: "BMOC") -> "BMOC":
        return self.union(other)

    def __and__(self, other: "BMOC") -> "BMOC":
        return self.intersection(other)

    def __sub__(self, other: "BMOC") -> "BMOC":
        return self.difference(other)

    @classmethod
    def from_cone(
        cls,
        lon_deg: float,
        lat_deg: float,
        radius_deg: float,
        max_order: int = 10,
        *,
        inclusive: bool = True,
    ) -> "BMOC":
        """Build a BMOC for a spherical cone/disc."""
        order = int(max_order)
        _nside_from_max_order(order)
        if float(radius_deg) <= 0.0:
            return cls.empty()
        lon_t = torch.as_tensor(lon_deg)
        uniq, flag = _hierarchical_disc_bmoc(
            float(lon_deg),
            float(lat_deg),
            float(radius_deg),
            order,
            inclusive=inclusive,
            device=lon_t.device,
        )
        return cls(uniq, flag)

    from_circle = from_cone

    @classmethod
    def from_polygon(
        cls,
        lon_deg: Tensor | list[float],
        lat_deg: Tensor | list[float],
        max_order: int = 10,
    ) -> "BMOC":
        """Build a BMOC for a spherical polygon."""
        order = int(max_order)
        _nside_from_max_order(order)
        uniq, flag = _hierarchical_polygon_bmoc(
            torch.as_tensor(lon_deg),
            torch.as_tensor(lat_deg),
            order,
        )
        return cls(uniq, flag)

    @classmethod
    def from_wcs(
        cls,
        wcs: Any,
        shape: tuple[int, int] | None = None,
        *,
        max_order: int = 10,
        mask: Tensor | None = None,
    ) -> "BMOC":
        """Build a BMOC from a 2D WCS footprint (and optional valid-pixel mask)."""
        order = int(max_order)
        nside = _nside_from_max_order(order)
        if mask is not None:
            mask_t = torch.as_tensor(mask, dtype=torch.bool)
            if mask_t.ndim != 2:
                raise ValueError("mask must be a 2D tensor of shape (H, W)")
            if not bool(mask_t.any()):
                return cls.empty()
            device = getattr(wcs, "crval", mask_t).device
            dtype = getattr(wcs, "crval", torch.tensor(0.0, dtype=torch.float64)).dtype
            if not dtype.is_floating_point:
                dtype = torch.float32 if device.type == "mps" else torch.float64
            yy, xx = torch.nonzero(mask_t.to(device=device), as_tuple=True)
            ra, dec = wcs.pixel_to_world(xx.to(dtype), yy.to(dtype))
            valid = torch.isfinite(ra) & torch.isfinite(dec)
            if not bool(valid.any()):
                return cls.empty()
            pix = torch.unique(
                _healpix.ang2pix(
                    nside, ra[valid], dec[valid], nest=True, lonlat=True
                ).to(torch.int64)
            )
            # Determine interior vs boundary on the mask grid via 8-neighbors in pix
            nb = _healpix.neighbors(nside, pix, nest=True)
            is_interior = (nb >= 0).all(dim=-1) & torch.isin(nb, pix).all(dim=-1)
            uniq = nest2uniq(nside, pix)
            flag = is_interior.to(torch.uint8)
            return cls(uniq, flag)

        ra_p, dec_p, _, _ = _wcs_perimeter_lonlat(wcs, shape=shape)
        valid = torch.isfinite(ra_p) & torch.isfinite(dec_p)
        if int(valid.sum().item()) < 3:
            return cls.empty()
        return cls.from_polygon(ra_p[valid], dec_p[valid], max_order=order)

    def sample_randoms(
        self,
        n: int,
        *,
        include_boundary: bool = True,
        generator: torch.Generator | None = None,
        device: torch.device | str | None = None,
    ) -> tuple[Tensor, Tensor]:
        """Draw ``n`` points uniformly in solid angle inside this BMOC."""
        return sample_randoms(
            n,
            self.to_moc(include_boundary=include_boundary),
            generator=generator,
            device=device,
        )

    def prepare_random_sampler(
        self,
        *,
        include_boundary: bool = True,
        device: torch.device | str | None = None,
    ) -> PreparedRandomSampler:
        """Snapshot immutable, device-bound state for repeated random draws.

        Boundary filtering and cell state are captured during preparation.
        Later mutation of ``uniq`` or ``flag`` has no effect; prepare again to
        observe changes.
        """
        uniq = self.uniq if include_boundary else self.uniq[self.flag > 0]
        return _prepare_moc_random_sampler(uniq, device=device)


__all__ = ["BMOC", "MOC", "nest2uniq", "sample_randoms", "uniq2nest"]
