"""Exact overlap integration of image pixels onto HEALPix cells.

HEALPix is equal-area in its lattice coordinates (Gorski et al. 2005;
Calabretta & Roukema 2007). In the equatorial zone (|z| <= 2/3) the chart

    jp = N (1/2 + t - 3z/4),  jm = N (1/2 + t + 3z/4),  t = 2 phi / pi,

and in each polar quadrant q the chart

    jp = tp s,  jm = (1 - tp) s,  s = N sqrt(3 (1 - |z|)),  tp = t - q,

map the sphere so that every cell is a unit square ``[i, i+1) x [j, j+1)`` and
``d(jp) d(jm) = (3 N^2 / pi) dOmega``: area in lattice units is area in cells.
An image pixel is the quadrilateral of its four projected corners (a chord
polygon in the chart; neighbouring pixels share corners, so the chords tile
the image without gaps). Its overlap with each cell is the exact area of the
intersection of the quadrilateral, the unit square, and the chart's own
domain (the equatorial band or the polar triangle), so pixels that straddle
a chart boundary are split exactly between the charts.

Pixels whose chord polygon is not a valid convex quadrilateral in a chart,
spans more than two cells, or spans more than ``_MAX_TT_SPAN`` in longitude
(only near a pole) are split into four
sub-pixels, recursively. After ``_MAX_DEPTH`` splits the remainder (a few
sub-pixels touching a pole) is assigned whole by its centre; its area is
bounded by ``16 (p / 2**_MAX_DEPTH)**2``.
"""

from __future__ import annotations

import math
from typing import Any

import torch
from torch import Tensor
from ..device import _to

_MAX_DEPTH = 20
# Largest longitude span of one (sub-)pixel, in quadrants (1/64 = 1.4 deg).
# The polar charts are bilinear in (longitude, colatitude), so a chord polygon
# spanning dphi radians misplaces about dphi**2 / 8 of a straddling pixel's
# area; 1/64 quadrant keeps that below 1e-4 of those pixels (and the total
# area within 1e-6, see tests/test_map_healpix_overlap.py). Only pixels within
# ~40 pixel sizes of a pole are split by this rule.
_MAX_TT_SPAN = 1.0 / 64.0
_CHUNK = 8192
# Charts: 0 equatorial, 1..4 north quadrants, 5..8 south quadrants.
_N_CHARTS = 9


def _unwrap(t: Tensor, ref: Tensor) -> Tensor:
    return ref + torch.remainder(t - ref + 2.0, 4.0) - 2.0


def _region(z: Tensor, t: Tensor) -> Tensor:
    q = torch.clamp(torch.floor(t), 0, 3).to(torch.int64)
    return torch.where(
        z > 2.0 / 3.0, 1 + q, torch.where(z < -2.0 / 3.0, 5 + q, torch.zeros_like(q))
    )


def _chart(z: Tensor, t: Tensor, chart: int, n: float) -> tuple[Tensor, Tensor]:
    """Lattice coordinates of points (any region) in the extended chart."""
    if chart == 0:
        t_u = _unwrap(t, t[..., :1])
        return n * (0.5 + t_u - 0.75 * z), n * (0.5 + t_u + 0.75 * z)
    q = float((chart - 1) % 4)
    one_minus = (1.0 - z) if chart <= 4 else (1.0 + z)
    s = n * torch.sqrt(torch.clamp(3.0 * one_minus, min=0.0))
    tp = torch.remainder(t - q + 2.0, 4.0) - 2.0
    return tp * s, (1.0 - tp) * s


def _domain(chart: int, n: float, like: Tensor) -> tuple[Tensor, Tensor]:
    """Half-planes ``n . x <= c`` of the chart's domain, padded to three."""
    if chart == 0:
        normals = [(-1.0, 1.0), (1.0, -1.0), (0.0, 0.0)]
        consts = [n, n, 0.0]
    else:
        normals = [(-1.0, 0.0), (0.0, -1.0), (1.0, 1.0)]
        consts = [0.0, 0.0, n]
    return (
        torch.tensor(normals, dtype=like.dtype, device=like.device),
        torch.tensor(consts, dtype=like.dtype, device=like.device),
    )


def _cell_lonlat(i: Tensor, j: Tensor, chart: int, n: float) -> tuple[Tensor, Tensor]:
    """Sky position of the centre of lattice cell ``(i, j)`` in a chart."""
    u = i.to(torch.float64) + 0.5
    v = j.to(torch.float64) + 0.5
    if chart == 0:
        t = (u + v) / (2.0 * n) - 0.5
        z = (v - u) / (1.5 * n)
    else:
        s = u + v
        t = float((chart - 1) % 4) + u / s
        z = 1.0 - s * s / (3.0 * n * n)
        if chart > 4:
            z = -z
    lon = torch.remainder(t * 90.0, 360.0)
    lat = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    return lon, lat


def _intersection_area(normals: Tensor, consts: Tensor) -> Tensor:
    """Area of the bounded intersection of half-planes ``n_k . x <= c_k``.

    Green's theorem over the boundary: each line contributes ``int x dy`` along
    the part of it that satisfies every other half-plane. Rows of zero normal
    are absent. Exactly coincident same-direction lines count once.
    """
    nn = (normals * normals).sum(-1)  # (M, K)
    present = nn > 0.0
    safe = torch.where(present, nn, torch.ones_like(nn))
    origin = normals * (consts / safe).unsqueeze(-1)  # (M, K, 2)
    direction = torch.stack((-normals[..., 1], normals[..., 0]), dim=-1)
    # a[m, j, k] = n_j . d_k ; slack[m, j, k] = c_j - n_j . o_k
    a = torch.einsum("mjc,mkc->mjk", normals, direction)
    slack = consts.unsqueeze(-1) - torch.einsum("mjc,mkc->mjk", normals, origin)
    k_count = normals.shape[1]
    idx = torch.arange(k_count, device=normals.device)
    not_self = (idx.view(-1, 1) != idx.view(1, -1)).unsqueeze(0)
    same_dir = torch.einsum("mjc,mkc->mjk", normals, normals) > 0.0
    later = (idx.view(-1, 1) < idx.view(1, -1)).unsqueeze(0)  # j < k
    parallel = a == 0.0
    par_ok = torch.where(same_dir & later, slack > 0.0, slack >= 0.0)
    feasible = torch.where(parallel & not_self, par_ok, torch.ones_like(par_ok)).all(1)
    inf = torch.full_like(a, math.inf)
    ratio = slack / torch.where(parallel, torch.ones_like(a), a)
    t_hi = torch.where((a > 0.0) & not_self, ratio, inf).amin(1)
    t_lo = torch.where((a < 0.0) & not_self, ratio, -inf).amax(1)
    ok = present & feasible & (t_hi > t_lo) & torch.isfinite(t_hi - t_lo)
    t_hi = torch.where(ok, t_hi, torch.zeros_like(t_hi))
    t_lo = torch.where(ok, t_lo, torch.zeros_like(t_lo))
    p0 = origin + t_lo.unsqueeze(-1) * direction
    p1 = origin + t_hi.unsqueeze(-1) * direction
    contrib = 0.5 * (p0[..., 0] + p1[..., 0]) * (p1[..., 1] - p0[..., 1])
    return torch.where(ok, contrib, torch.zeros_like(contrib)).sum(-1)


def _quad_halfplanes(u: Tensor, v: Tensor) -> tuple[Tensor, Tensor]:
    """Half-planes of a counter-clockwise quadrilateral (M, 4)."""
    du = torch.roll(u, -1, dims=1) - u
    dv = torch.roll(v, -1, dims=1) - v
    normals = torch.stack((dv, -du), dim=-1)
    consts = normals[..., 0] * u + normals[..., 1] * v
    return normals, consts


def _signed_area(u: Tensor, v: Tensor) -> Tensor:
    return 0.5 * (u * torch.roll(v, -1, 1) - torch.roll(u, -1, 1) * v).sum(1)


def _convex(u: Tensor, v: Tensor) -> Tensor:
    du = torch.roll(u, -1, 1) - u
    dv = torch.roll(v, -1, 1) - v
    cross = du * torch.roll(dv, -1, 1) - dv * torch.roll(du, -1, 1)
    return (cross > 0.0).all(1)


def _spherical_quad_area(lon: Tensor, lat: Tensor) -> Tensor:
    """Solid angle (sr) of quadrilaterals with corners ``(M, 4)`` in degrees."""
    lo, la = torch.deg2rad(lon), torch.deg2rad(lat)
    xyz = torch.stack(
        (torch.cos(la) * torch.cos(lo), torch.cos(la) * torch.sin(lo), torch.sin(la)),
        dim=-1,
    )

    def tri(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
        num = (a * torch.linalg.cross(b, c)).sum(-1).abs()
        den = 1.0 + (a * b).sum(-1) + (b * c).sum(-1) + (c * a).sum(-1)
        return 2.0 * torch.atan2(num, den)

    return tri(xyz[:, 0], xyz[:, 1], xyz[:, 2]) + tri(xyz[:, 0], xyz[:, 2], xyz[:, 3])


def central_pixel_solid_angle(wcs: Any, height: int, width: int) -> float:
    """Solid angle (sr) of the pixel at the image centre."""
    xc, yc = (width - 1) / 2.0, (height - 1) / 2.0
    dx = torch.tensor([-0.5, 0.5, 0.5, -0.5], dtype=torch.float64)
    dy = torch.tensor([-0.5, -0.5, 0.5, 0.5], dtype=torch.float64)
    lon, lat = _world(wcs, xc + dx, yc + dy)
    return float(_spherical_quad_area(lon.view(1, 4), lat.view(1, 4))[0])


def _world(wcs: Any, x: Tensor, y: Tensor) -> tuple[Tensor, Tensor]:
    dev = wcs.device if hasattr(wcs, "device") else x.device
    dtype = torch.float32 if torch.device(dev).type == "mps" else torch.float64
    lon, lat = wcs.pixel_to_world(
        _to(x, device=dev, dtype=dtype), _to(y, device=dev, dtype=dtype)
    )
    return lon.cpu().to(torch.float64), lat.cpu().to(torch.float64)


def integrate_exposure(
    wcs: Any,
    numer: Tensor,
    denom: Tensor,
    *,
    nside: int,
    nest: bool,
    chunk_size: int = _CHUNK,
) -> tuple[Tensor, Tensor, Tensor]:
    """Overlap-integrate valid image pixels onto HEALPix cells.

    ``numer`` and ``denom`` are per-pixel ``w d`` and ``w`` (zero where
    invalid). Returns unique cells and ``sum_i A_ip w_i d_i``,
    ``sum_i A_ip w_i`` with ``A_ip`` the overlap of pixel ``i`` and cell ``p``
    in units of the cell area (drizzle's convention at ``pixfrac = 1``).
    Geometry is float64 on the CPU; values keep their device and autograd.
    """
    from ..sphere import healpix as _hp

    out_device = numer.device
    n = float(nside)
    width = int(numer.shape[-1])
    flat_num = numer.reshape(-1)
    flat_den = denom.reshape(-1)
    valid = (flat_den > 0.0).detach().cpu()
    ids = torch.nonzero(valid, as_tuple=False).view(-1)
    omega_cell = 4.0 * math.pi / (12.0 * nside * nside)

    cells: list[Tensor] = []
    owners: list[Tensor] = []
    areas: list[Tensor] = []

    for chunk in ids.split(max(1, min(int(chunk_size), _CHUNK))):
        xs = (chunk % width).to(torch.float64)
        ys = torch.div(chunk, width, rounding_mode="floor").to(torch.float64)
        half = torch.full_like(xs, 0.5)
        owner = chunk
        for depth in range(_MAX_DEPTH + 1):
            if owner.numel() == 0:
                break
            dx = torch.tensor([-1.0, 1.0, 1.0, -1.0], dtype=torch.float64)
            dy = torch.tensor([-1.0, -1.0, 1.0, 1.0], dtype=torch.float64)
            cx = (xs.view(-1, 1) + half.view(-1, 1) * dx).reshape(-1)
            cy = (ys.view(-1, 1) + half.view(-1, 1) * dy).reshape(-1)
            lon, lat = _world(wcs, cx, cy)
            lon, lat = lon.view(-1, 4), lat.view(-1, 4)
            finite = (torch.isfinite(lon) & torch.isfinite(lat)).all(1)
            lon = torch.where(torch.isfinite(lon), lon, torch.zeros_like(lon))
            lat = torch.where(torch.isfinite(lat), lat, torch.zeros_like(lat))
            z = torch.sin(torch.deg2rad(lat))
            t = torch.remainder(lon, 360.0) / 90.0
            t_u = _unwrap(t, t[:, :1])
            bad = (t_u.amax(1) - t_u.amin(1)) > _MAX_TT_SPAN
            region = _region(z, t)

            pieces: list[tuple[int, Tensor, Tensor, Tensor]] = []
            for chart in range(_N_CHARTS):
                sel = (region == chart).any(1) & finite
                if not bool(sel.any()):
                    continue
                rows = torch.nonzero(sel, as_tuple=False).view(-1)
                u, v = _chart(z[rows], t[rows], chart, n)
                area = _signed_area(u, v)
                flip = area < 0.0
                u = torch.where(flip.view(-1, 1), u.flip(1), u)
                v = torch.where(flip.view(-1, 1), v.flip(1), v)
                i0 = torch.floor(u.amin(1))
                j0 = torch.floor(v.amin(1))
                wide = (torch.floor(u.amax(1)) - i0 > 2.0) | (
                    torch.floor(v.amax(1)) - j0 > 2.0
                )
                bad[rows] |= wide | ~_convex(u, v)
                pieces.append(
                    (chart, rows, torch.stack((u, v), -1), torch.stack((i0, j0), -1))
                )

            # Near a pole, a (sub-)pixel whose four corners share one cell lies
            # inside it: take its solid angle instead of splitting further.
            near_pole = finite & bad & (region != 0).all(1)
            if bool(near_pole.any()):
                r = torch.nonzero(near_pole, as_tuple=False).view(-1)
                corner_cells = _hp.ang2pix(
                    nside,
                    lon[r].reshape(-1),
                    lat[r].reshape(-1),
                    nest=nest,
                    lonlat=True,
                ).view(-1, 4)
                same = (corner_cells == corner_cells[:, :1]).all(1).cpu()
                if bool(same.any()):
                    r_same = r[same]
                    cells.append(corner_cells[same, 0].cpu())
                    owners.append(owner[r_same])
                    areas.append(
                        _spherical_quad_area(lon[r_same], lat[r_same]) / omega_cell
                    )
                    finite[r_same] = False

            good = finite & ~bad
            for chart, rows, uv, origin in pieces:
                keep = good[rows]
                if not bool(keep.any()):
                    continue
                rows, uv, origin = rows[keep], uv[keep], origin[keep]
                # Chord quadrilaterals inside one lattice cell (and one chart)
                # need no clipping: the overlap is the quadrilateral's area.
                inner = (
                    (region[rows] == chart).all(1)
                    & (torch.floor(uv[..., 0]) == origin[:, :1]).all(1)
                    & (torch.floor(uv[..., 1]) == origin[:, 1:]).all(1)
                )
                if bool(inner.any()):
                    o = origin[inner]
                    lon_c, lat_c = _cell_lonlat(o[:, 0], o[:, 1], chart, n)
                    cells.append(
                        _hp.ang2pix(nside, lon_c, lat_c, nest=nest, lonlat=True).cpu()
                    )
                    owners.append(owner[rows[inner]])
                    areas.append(_signed_area(uv[inner, :, 0], uv[inner, :, 1]))
                outer = ~inner
                if not bool(outer.any()):
                    continue
                rows, uv, origin = rows[outer], uv[outer], origin[outer]
                cell_ids, piece_rows, piece_area = _clip_to_cells(
                    uv, origin, chart, n, nside, nest
                )
                cells.append(cell_ids)
                owners.append(owner[rows[piece_rows]])
                areas.append(piece_area)

            redo = finite & bad
            if depth == _MAX_DEPTH:
                if bool(redo.any()):
                    r = torch.nonzero(redo, as_tuple=False).view(-1)
                    clon, clat = _world(wcs, xs[r], ys[r])
                    cells.append(
                        _hp.ang2pix(nside, clon, clat, nest=nest, lonlat=True).cpu()
                    )
                    owners.append(owner[r])
                    areas.append(_spherical_quad_area(lon[r], lat[r]) / omega_cell)
                break
            r = torch.nonzero(redo, as_tuple=False).view(-1)
            q = half[r] / 2.0
            xs = (xs[r].view(-1, 1) + q.view(-1, 1) * dx.view(1, 4)).reshape(-1)
            ys = (ys[r].view(-1, 1) + q.view(-1, 1) * dy.view(1, 4)).reshape(-1)
            half = q.repeat_interleave(4)
            owner = owner[r].repeat_interleave(4)

    if not cells:
        empty = torch.empty(0, dtype=torch.int64, device=out_device)
        zero = flat_num.new_zeros(0)
        return empty, zero, flat_den.new_zeros(0)
    all_cells = torch.cat(cells).to(torch.int64)
    all_owner = torch.cat(owners).to(device=out_device)
    all_area = torch.cat(areas)
    uniq, inverse = torch.unique(all_cells, sorted=True, return_inverse=True)
    uniq = uniq.to(out_device)
    inverse = inverse.to(out_device)
    # Near a pole a cell collects many tiny sub-pixel pieces; accumulate in
    # float64 (float32 sums drift by ~1e-4 there) except on MPS.
    acc = torch.float32 if out_device.type == "mps" else torch.float64
    area_v = _to(all_area, device=out_device, dtype=acc)
    num_out = torch.zeros(uniq.numel(), dtype=acc, device=out_device).index_add(
        0, inverse, area_v * flat_num[all_owner].to(acc)
    )
    den_out = torch.zeros(uniq.numel(), dtype=acc, device=out_device).index_add(
        0, inverse, area_v * flat_den[all_owner].to(acc)
    )
    num_out = num_out.to(flat_num.dtype)
    den_out = den_out.to(flat_den.dtype)
    return uniq, num_out, den_out


def _clip_to_cells(
    uv: Tensor, origin: Tensor, chart: int, n: float, nside: int, nest: bool
) -> tuple[Tensor, Tensor, Tensor]:
    """Overlap of each quadrilateral with the 3x3 lattice cells at ``origin``."""
    from ..sphere import healpix as _hp

    m = uv.shape[0]
    offs = torch.stack(
        torch.meshgrid(
            torch.arange(3, dtype=uv.dtype),
            torch.arange(3, dtype=uv.dtype),
            indexing="ij",
        ),
        -1,
    ).reshape(9, 2)
    # Local coordinates relative to each candidate cell's lower corner.
    local = uv.unsqueeze(1) - (origin.unsqueeze(1) + offs.unsqueeze(0)).unsqueeze(2)
    local = local.reshape(m * 9, 4, 2)
    q_n, q_c = _quad_halfplanes(local[..., 0], local[..., 1])
    cell_n = torch.tensor(
        [[0.0, -1.0], [1.0, 0.0], [0.0, 1.0], [-1.0, 0.0]], dtype=uv.dtype
    ).expand(m * 9, 4, 2)
    cell_c = torch.tensor([0.0, 1.0, 1.0, 0.0], dtype=uv.dtype).expand(m * 9, 4)
    d_n, d_c = _domain(chart, n, uv)
    corner = (origin.unsqueeze(1) + offs.unsqueeze(0)).reshape(m * 9, 2)
    # Domain constants shift into the cell's local frame: c - n . corner.
    dom_c = d_c.unsqueeze(0) - corner @ d_n.T
    dom_n = d_n.unsqueeze(0).expand(m * 9, 3, 2)
    normals = torch.cat((q_n, cell_n, dom_n), dim=1)
    consts = torch.cat((q_c, cell_c, dom_c), dim=1)
    area = _intersection_area(normals, consts)
    hit = area > 0.0
    if not bool(hit.any()):
        empty = torch.empty(0, dtype=torch.int64)
        return empty, empty, torch.empty(0, dtype=uv.dtype)
    corner = corner[hit]
    lon, lat = _cell_lonlat(corner[:, 0], corner[:, 1], chart, n)
    cell_ids = _hp.ang2pix(nside, lon, lat, nest=nest, lonlat=True).cpu()
    rows = torch.arange(m).repeat_interleave(9)[hit]
    return cell_ids, rows, area[hit]
