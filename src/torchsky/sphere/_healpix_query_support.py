"""Ring bookkeeping and nested-range helpers behind HEALPix region queries."""

import math
from typing import Callable, Sequence

import torch

from .. import _ops
from .._dtype import work_dtype as _float_dtype_for_device
from torch import Tensor

from ._healpix_scheme import (
    _as_float64,
    _validate_nside,
)


def _normalize_xyz(
    x: Tensor, y: Tensor, z: Tensor, eps: float = 1.0e-15
) -> tuple[Tensor, Tensor, Tensor]:
    scale = torch.maximum(
        torch.maximum(torch.abs(x), torch.abs(y)), torch.abs(z)
    ).clamp_min(eps)
    xs = x / scale
    ys = y / scale
    zs = z / scale
    n = torch.sqrt(xs * xs + ys * ys + zs * zs)
    return xs / n, ys / n, zs / n


def _pixel_range_chunks(
    npix: int, chunk_size: int = 1_000_000
) -> list[tuple[int, int]]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")
    out: list[tuple[int, int]] = []
    start = 0
    while start < npix:
        end = min(npix, start + chunk_size)
        out.append((start, end))
        start = end
    return out


def _ring_above(nside: int, z: Tensor) -> Tensor:
    """Vectorized HEALPix ring_above(z) helper."""
    zf = _as_float64(z)
    az = torch.abs(zf)
    eq = az <= (2.0 / 3.0)

    ir_eq = torch.floor(nside * (2.0 - 1.5 * zf)).to(torch.int64)
    ir_p = torch.floor(nside * torch.sqrt(torch.clamp(3.0 * (1.0 - az), min=0.0))).to(
        torch.int64
    )
    ir = torch.where(eq, ir_eq, torch.where(zf > 0.0, ir_p, 4 * nside - ir_p - 1))
    return ir


def _get_ring_info2(nside: int, ring: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    """Vectorized get_ring_info2 helper for interpolation."""
    r = torch.as_tensor(ring, dtype=torch.int64)
    fd = _float_dtype_for_device(r.device)
    npix = _core_mod.nside2npix(nside)
    ncap = 2 * nside * (nside - 1)
    fact2 = 4.0 / float(npix)
    fact1 = (2.0 * float(nside)) * fact2

    northr = torch.where(r > 2 * nside, 4 * nside - r, r)
    northf = northr.to(fd)
    polar = northr < nside

    tmp = northf * northf * fact2
    costheta_p = 1.0 - tmp
    sintheta_p = torch.sqrt(torch.clamp(tmp * (2.0 - tmp), min=0.0))
    theta_p = torch.atan2(sintheta_p, costheta_p)
    ringpix_p = 4 * northr
    shifted_p = torch.ones_like(polar, dtype=torch.bool)
    startpix_p = 2 * northr * (northr - 1)

    theta_e = torch.acos(torch.clamp((2.0 * float(nside) - northf) * fact1, -1.0, 1.0))
    ringpix_e = torch.full_like(r, 4 * nside)
    shifted_e = ((northr - nside) & 1) == 0
    startpix_e = ncap + (northr - nside) * ringpix_e

    theta = torch.where(polar, theta_p, theta_e)
    ringpix = torch.where(polar, ringpix_p, ringpix_e)
    shifted = torch.where(polar, shifted_p, shifted_e)
    startpix = torch.where(polar, startpix_p, startpix_e)

    south = northr != r
    theta = torch.where(south, math.pi - theta, theta)
    startpix = torch.where(south, npix - startpix - ringpix, startpix)

    return startpix, ringpix, theta, shifted


def _ring_pixels_colatitude_band(
    nside: int,
    theta_lo: float,
    theta_hi: float,
    *,
    nest: bool,
) -> Tensor:
    """RING indices for pixels whose ring colatitude lies in ``[theta_lo, theta_hi]``.

    Colatitude is radians from the north pole (0 at pole, ``pi`` at south pole), matching
    ``theta = deg2rad(90 - lat)`` from ``pix2ang(..., lonlat=True)``.

    Cost is **O(nside)** ring metadata plus **O(output)** index materialization.
    """
    _validate_nside(nside)
    lo = min(theta_lo, theta_hi)
    hi = max(theta_lo, theta_hi)
    if hi < 0.0 or lo > math.pi:
        return torch.empty((0,), dtype=torch.int64)
    lo = max(0.0, lo)
    hi = min(math.pi, hi)

    if 4 * nside <= 1:
        return torch.empty((0,), dtype=torch.int64)

    rings = torch.arange(1, 4 * nside, dtype=torch.int64)
    startpix, ringpix, theta, _shifted = _get_ring_info2(nside, rings)
    mask = (theta >= lo) & (theta <= hi)
    idx = torch.nonzero(mask, as_tuple=False).flatten()
    if idx.numel() == 0:
        return torch.empty((0,), dtype=torch.int64)

    parts: list[Tensor] = []
    for j in idx:
        sp = int(startpix[j].item())
        rp = int(ringpix[j].item())
        parts.append(torch.arange(sp, sp + rp, dtype=torch.int64))
    ring_out = torch.cat(parts, dim=0)
    if nest:
        return _core_mod.ring2nest(nside, ring_out)
    return ring_out


def _query_disc_nest_ranges(
    nside: int,
    vec: Tensor | Sequence[float],
    radius: float,
    *,
    inclusive: bool = False,
    fact: int = 4,
) -> Tensor | None:
    """NESTED half-open ``(N, 2)`` pixel ranges of :func:`query_disc`.

    Hierarchical quadtree descent: cells inside the disc become one range
    without visiting their pixels, so the cost is ``O(nside + ranges)`` rather
    than ``O(pixels)``. Without the native kernel (MPS or no extension) the same
    descent runs in torch (:func:`_disc_nest_ranges_torch`).
    """
    _validate_nside(nside)
    radius = float(radius)
    if not math.isfinite(radius) or radius < 0.0 or radius > math.pi:
        raise ValueError("radius must be finite and in [0, pi]")
    fact_i = int(fact)
    if fact_i < 1:
        raise ValueError("fact must be a positive integer")
    if fact_i & (fact_i - 1) != 0:
        raise ValueError("for nest ordering, fact must be a power of two")
    v_any = torch.as_tensor(vec).reshape(-1)
    if v_any.numel() != 3:
        raise ValueError("vec must be length-3 Cartesian vector")
    out_device = v_any.device
    if not bool(torch.isfinite(v_any).all()):
        raise ValueError("vec must contain exactly 3 finite values")
    if not bool(torch.any(v_any != 0)):
        raise ValueError("vec norm must be positive")
    if inclusive and fact_i != 4:
        nside_hi = int(nside) * fact_i
        _validate_nside(nside_hi)
        hi = _query_disc_nest_ranges(nside_hi, v_any, radius, inclusive=True, fact=4)
        if hi is None or hi.numel() == 0:
            return hi
        block = fact_i * fact_i
        starts = torch.div(hi[:, 0], block, rounding_mode="floor")
        ends = torch.div(hi[:, 1] + (block - 1), block, rounding_mode="floor")
        run_end = torch.cummax(ends, dim=0).values
        new_group = torch.ones_like(starts, dtype=torch.bool)
        new_group[1:] = starts[1:] > run_end[:-1]
        first = torch.nonzero(new_group).reshape(-1)
        last = torch.cat((first[1:] - 1, first.new_full((1,), starts.numel() - 1)))
        return torch.stack((starts[first], run_end[last]), dim=1)
    v64 = v_any.detach().to(device="cpu").to(dtype=torch.float64).reshape(-1)
    radius_eff = min(
        math.pi,
        radius
        + (_core_mod.max_pixel_radius(nside, degrees=False) if inclusive else 0.0),
    )
    cos_lim = float(math.cos(radius_eff))
    if out_device.type == "mps" or not _ops.has_fixed_dispatcher():
        out = _disc_nest_ranges_torch(int(nside), v64, cos_lim)
    else:
        out = torch.ops.torchsky.sphere_query_disc_nest_ranges(
            int(nside), v64.contiguous(), cos_lim
        )
    return out if out_device.type == "cpu" else out.to(device=out_device)


def _disc_nest_ranges_torch(nside: int, vec: Tensor, cos_lim: float) -> Tensor:
    """Torch twin of the native ``sphere_query_disc_nest_ranges`` kernel.

    NESTED half-open ranges of pixels whose centre ``p`` satisfies
    ``p . vec >= cos_lim`` (``vec`` a float64 CPU 3-vector). Quadtree descent
    from the 12 base cells: cells wholly inside the cap emit one range, cells
    wholly outside are dropped, and only boundary cells are refined, so the cost
    is ``O(boundary cells x depth + ranges)`` rather than ``O(npix)``.
    """
    if cos_lim <= -1.0:
        return torch.tensor([[0, 12 * nside * nside]], dtype=torch.int64)
    v = vec.detach().to(device="cpu").to(dtype=torch.float64).reshape(3)
    vx, vy, vz = _normalize_xyz(v[0], v[1], v[2])
    centre = torch.stack((vx, vy, vz))
    radius = math.atan2(math.sqrt(max(0.0, 1.0 - cos_lim * cos_lim)), cos_lim)
    margin = 1.0e-9

    def _classify(centres: Tensor, cell_radius: float) -> tuple[Tensor, Tensor]:
        cross = torch.linalg.cross(centres, centre.expand_as(centres), dim=1)
        dist = torch.atan2(cross.norm(dim=1), centres @ centre)
        inside = (dist + cell_radius) <= (radius - margin)
        outside = (dist - cell_radius) >= (radius + margin)
        return inside, outside

    def _leaf(pix: Tensor) -> Tensor:
        x, y, z = _core_mod.pix2vec(nside, pix, nest=True)
        return (x * vx + y * vy + z * vz) >= cos_lim

    return _nest_ranges_by_descent(nside, _classify, _leaf)


def _nest_ranges_to_ring_pixels(nside: int, ranges: Tensor) -> Tensor:
    """Ascending RING pixels of NESTED half-open ``ranges`` (``O(output)``)."""
    nest_pix = _ranges_to_sorted_pixels(ranges)
    if nest_pix.numel() == 0:
        return nest_pix
    return torch.sort(_core_mod.nest2ring(nside, nest_pix)).values


_CELL_RADIUS_EXACT_LEVELS = 5


_CELL_RADIUS_CACHE: dict[int, float] = {}


def _cell_radius_bound(level: int) -> float:
    """Upper bound (rad) on the centre-to-boundary distance of order-``level`` cells.

    Exhaustive over corners and edge midpoints up to ``nside = 32``, then the
    self-similar ``1 / nside`` law with 8% slack (the native disc kernel uses the
    same construction).
    """
    exact = _CELL_RADIUS_EXACT_LEVELS
    if level > exact:
        return _cell_radius_bound(exact) * 2.0 ** (exact - level) * 1.08
    cached = _CELL_RADIUS_CACHE.get(level)
    if cached is None:
        ns = 1 << level
        pix = torch.arange(12 * ns * ns, dtype=torch.int64)
        cx, cy, cz = _core_mod.pix2vec(ns, pix, nest=True)
        centre = torch.stack((cx, cy, cz), dim=-1).unsqueeze(-1)  # (N, 3, 1)
        pts = _core_mod.boundaries(ns, pix, step=2, nest=True)  # (N, 3, 8)
        cross = torch.cross(centre.expand_as(pts), pts, dim=1).norm(dim=1)
        dot = (centre * pts).sum(dim=1)
        cached = float(torch.atan2(cross, dot).max()) * 1.02 + 1.0e-12
        _CELL_RADIUS_CACHE[level] = cached
    return cached


def _nest_ranges_by_descent(
    nside: int,
    classify: Callable[[Tensor, float], tuple[Tensor, Tensor]],
    leaf_keep: Callable[[Tensor], Tensor],
) -> Tensor:
    """Hierarchical NESTED range query: sorted disjoint half-open ``(N, 2)`` ranges.

    ``classify(centres, radius)`` returns ``(inside, outside)`` masks for cells
    whose centre-to-boundary distance is at most ``radius`` (conservative: a cell
    is ``inside``/``outside`` only if every point of it is). Inside cells emit
    their contiguous NESTED range, outside cells vanish, and ambiguous cells split
    into four children down to pixels, where ``leaf_keep(pixels)`` decides exactly.
    Work is ``O(boundary cells)`` per level; interior pixels are never visited.
    """
    levels = int(nside).bit_length() - 1
    cells = torch.arange(12, dtype=torch.int64)
    starts: list[Tensor] = []
    ends: list[Tensor] = []
    for level in range(levels + 1):
        if cells.numel() == 0:
            break
        ns_j = 1 << level
        x, y, z = _core_mod.pix2vec(ns_j, cells, nest=True)
        inside, outside = classify(
            torch.stack((x, y, z), dim=-1), _cell_radius_bound(level)
        )
        shift = 2 * (levels - level)
        if bool(inside.any()):
            c = cells[inside]
            starts.append(c << shift)
            ends.append((c + 1) << shift)
        boundary = cells[~(inside | outside)]
        if level == levels:
            if boundary.numel() > 0:
                keep = boundary[leaf_keep(boundary)]
                starts.append(keep)
                ends.append(keep + 1)
            break
        boundary = boundary.unsqueeze(1) * 4 + torch.arange(4, dtype=torch.int64)
        cells = boundary.reshape(-1)
    if not starts:
        return torch.empty((0, 2), dtype=torch.int64)
    st = torch.cat(starts)
    en = torch.cat(ends)
    if st.numel() == 0:
        return torch.empty((0, 2), dtype=torch.int64)
    order = torch.argsort(st)
    st = st[order]
    en = en[order]
    run_end = torch.cummax(en, dim=0).values
    new_group = torch.ones_like(st, dtype=torch.bool)
    new_group[1:] = st[1:] > run_end[:-1]
    first = torch.nonzero(new_group).reshape(-1)
    last = torch.cat((first[1:] - 1, first.new_full((1,), st.numel() - 1)))
    return torch.stack((st[first], run_end[last]), dim=1)


def _ranges_to_sorted_pixels(ranges: Tensor) -> Tensor:
    return _core_mod.pixel_ranges_to_pixels(ranges, inclusive=False)


def _strip_nest_pixels(nside: int, lo: float, hi: float) -> Tensor:
    """NESTED pixels of the rings with centre colatitude in ``[lo, hi]`` (ascending).

    Hierarchical: cells whose colatitude interval lies inside the band emit one
    range; only cells straddling a band edge are refined, and pixels are decided by
    their ring's colatitude exactly as the ring-ordered band query does.
    """
    if hi < 0.0 or lo > math.pi or 4 * nside <= 1:
        return torch.empty((0,), dtype=torch.int64)
    rings = torch.arange(1, 4 * nside, dtype=torch.int64)
    startpix, _ringpix, theta, _shifted = _get_ring_info2(nside, rings)
    ring_in = (theta >= lo) & (theta <= hi)
    margin = 1.0e-9

    def _classify(centres: Tensor, radius: float) -> tuple[Tensor, Tensor]:
        colat = torch.atan2(torch.hypot(centres[:, 0], centres[:, 1]), centres[:, 2])
        inside = ((colat - radius) >= (lo + margin)) & (
            (colat + radius) <= (hi - margin)
        )
        outside = ((colat + radius) < (lo - margin)) | (
            (colat - radius) > (hi + margin)
        )
        return inside, outside

    def _leaf(pix: Tensor) -> Tensor:
        ring_pix = _core_mod.nest2ring(nside, pix)
        ring_idx = torch.searchsorted(startpix, ring_pix, right=True) - 1
        return ring_in[ring_idx]

    ranges = _nest_ranges_by_descent(nside, _classify, _leaf)
    return _ranges_to_sorted_pixels(ranges)


def _normalize_order(order: str | None) -> str:
    if order is None:
        raise ValueError("order must be provided")
    s = str(order).upper()
    if s.startswith("RING"):
        return "RING"
    if s.startswith("NEST"):
        return "NEST"
    raise ValueError("order must be 'RING' or 'NEST'")


def _resolve_dtype(dtype: object | None, fallback: torch.dtype) -> torch.dtype:
    if dtype is None:
        return fallback
    if isinstance(dtype, torch.dtype):
        return dtype
    name = str(dtype).lower()
    if "float16" in name:
        return torch.float16
    if "float32" in name:
        return torch.float32
    if "float64" in name or "double" in name:
        return torch.float64
    if "int16" in name:
        return torch.int16
    if "int32" in name:
        return torch.int32
    if "int64" in name or "int" in name:
        return torch.int64
    return fallback


# Late-bound so monkeypatching ``healpix`` attributes still reaches this code.
from . import healpix as _core_mod  # noqa: E402
