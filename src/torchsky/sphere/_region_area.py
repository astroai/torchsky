"""Exact area of boolean combinations of spherical polygons.

The boundary of a boolean combination of polygons is made of pieces of the
operands' great-circle edges. Every edge is split at all crossings with the
other operands' edges; a piece belongs to the result's boundary when the result
holds on exactly one side of it, and is oriented with the result on its left.
The area is the closed-boundary sum of signed spherical triangles
(Van Oosterom-Strackee) against a reference point, which equals the area modulo
4 pi; containment of the reference point resolves the remaining 0 / 4 pi case.
No loop assembly is needed, so touching vertices and shared edges are handled by
the same rule.
"""

from __future__ import annotations

import math
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import torch
from torch import Tensor

FOUR_PI = 4.0 * math.pi
# Angular tolerance (radians) for "on an edge" and coincident split points;
# float64 unit vectors from degree inputs carry ~1e-16 rounding.
_TOL_RAD = 1.0e-11


@dataclass(frozen=True)
class OrientedLoop:
    """Polygon boundary oriented with the region on the left, plus a predicate."""

    vertices: Tensor  # [N, 3] float64 unit vectors
    contains: Callable[[Tensor], Tensor]  # [M, 3] -> [M] bool, off-boundary points


def _unit(v: Tensor) -> Tensor:
    return v / torch.linalg.norm(v, dim=-1, keepdim=True)


def triangle_signed_area(r: Tensor, a: Tensor, b: Tensor) -> Tensor:
    """Signed area of the spherical triangle ``(r, a, b)`` (unclamped)."""
    det = (r * torch.linalg.cross(a, b, dim=-1)).sum(dim=-1)
    den = 1.0 + (r * a).sum(-1) + (a * b).sum(-1) + (b * r).sum(-1)
    return 2.0 * torch.atan2(det, den)


def loop_left_area(vertices: Tensor) -> float:
    """Area to the left of a closed simple loop, in ``[0, 4 pi)``."""
    a = vertices
    b = torch.roll(vertices, shifts=-1, dims=0)
    return float(_boundary_area(a, b, None))


def _arc_parameters(a: Tensor, b: Tensor) -> tuple[Tensor, Tensor, Tensor]:
    n = torch.linalg.cross(a, b, dim=-1)
    n_norm = torch.linalg.norm(n, dim=-1)
    if bool((n_norm <= 1.0e-15).any()):
        raise ValueError("polygon has a degenerate (zero-length or antipodal) edge")
    n = n / n_norm.unsqueeze(-1)
    length = torch.atan2(n_norm, (a * b).sum(-1))
    return n, length, torch.linalg.cross(n, a, dim=-1)


def _angle_on_arc(a: Tensor, n: Tensor, p: Tensor) -> Tensor:
    """Angle from ``a`` to ``p`` measured along the circle with normal ``n``."""
    return torch.atan2((n * torch.linalg.cross(a, p, dim=-1)).sum(-1), (a * p).sum(-1))


# Rows of the edge-pair tests per chunk: bounds memory to O(chunk * E).
_PAIR_CHUNK = 512


def _split_angles(
    a: Tensor, n: Tensor, length: Tensor, owner: Tensor, b: Tensor
) -> list[Tensor]:
    """Interior split angles of every edge at crossings with other loops' edges."""
    count = int(a.shape[0])
    rows_all: list[Tensor] = []
    values_all: list[Tensor] = []
    a_f, n_f, len_f = a.unsqueeze(0), n.unsqueeze(0), length.unsqueeze(0)
    for r0 in range(0, count, _PAIR_CHUNK):
        r1 = min(r0 + _PAIR_CHUNK, count)
        rows = r1 - r0
        other = owner[r0:r1].unsqueeze(1) != owner.unsqueeze(0)
        n_e = n[r0:r1].unsqueeze(1)
        a_e = a[r0:r1].unsqueeze(1)
        len_e = length[r0:r1].unsqueeze(1)
        cross_n = torch.linalg.cross(
            n_e.expand(rows, count, 3), n_f.expand(rows, count, 3), dim=-1
        )
        cross_norm = torch.linalg.norm(cross_n, dim=-1)
        distinct = cross_norm > 1.0e-14
        p = cross_n / torch.where(distinct, cross_norm, 1.0).unsqueeze(-1)
        # Crossings of distinct great circles; on a shared great circle the
        # other edge's endpoints split this edge.
        candidates = (
            (p, distinct),
            (-p, distinct),
            (a_f.expand(rows, count, 3), ~distinct),
            (b.unsqueeze(0).expand(rows, count, 3), ~distinct),
        )
        for q, kind in candidates:
            on_e_circle = (q * n_e).sum(-1).abs() <= _TOL_RAD
            on_f_circle = (q * n_f).sum(-1).abs() <= _TOL_RAD
            t_e = _angle_on_arc(a_e, n_e, q)
            t_f = _angle_on_arc(a_f, n_f, q)
            inside_e = (t_e > _TOL_RAD) & (t_e < len_e - _TOL_RAD)
            on_f = (t_f >= -_TOL_RAD) & (t_f <= len_f + _TOL_RAD)
            keep = other & kind & on_e_circle & on_f_circle & inside_e & on_f
            hit_rows, hit_cols = torch.nonzero(keep, as_tuple=True)
            rows_all.append(hit_rows + r0)
            values_all.append(t_e[hit_rows, hit_cols])
    edge = torch.cat(rows_all)
    value = torch.cat(values_all)
    order = torch.argsort(edge, stable=True)
    counts = torch.bincount(edge, minlength=count).tolist()
    return list(torch.split(value[order], counts))


def _point_arc_distance(q: Tensor, a: Tensor, n: Tensor, length: Tensor) -> Tensor:
    """Angular distance of points ``q`` [M,3] from arcs [P] -> [M, P]."""
    qe = q.unsqueeze(1)
    t = _angle_on_arc(a.unsqueeze(0), n.unsqueeze(0), qe)
    plane = torch.asin((qe * n.unsqueeze(0)).sum(-1).abs().clamp(max=1.0))
    b = torch.cos(length).unsqueeze(-1) * a + torch.sin(length).unsqueeze(
        -1
    ) * torch.linalg.cross(n, a, dim=-1)
    to_a = torch.acos((qe * a.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0))
    to_b = torch.acos((qe * b.unsqueeze(0)).sum(-1).clamp(-1.0, 1.0))
    within = (t >= 0.0) & (t <= length.unsqueeze(0))
    return torch.where(within, plane, torch.minimum(to_a, to_b))


_REFERENCE_CANDIDATES = _unit(
    torch.tensor(
        [
            [x, y, z]
            for x in (-1.0, 0.0, 1.0)
            for y in (-1.0, 0.0, 1.0)
            for z in (-1.0, 0.0, 1.0)
            if (x, y, z) != (0.0, 0.0, 0.0)
        ],
        dtype=torch.float64,
    )
    # An irrational tilt keeps candidates off axis-aligned edges.
    @ torch.linalg.matrix_exp(
        torch.tensor(
            [[0.0, -0.3, 0.2], [0.3, 0.0, -0.1], [-0.2, 0.1, 0.0]],
            dtype=torch.float64,
        )
        * math.sqrt(2.0)
    )
)


def _boundary_area(
    a: Tensor, b: Tensor, contains: Callable[[Tensor], Tensor] | None
) -> Tensor:
    """Area left of oriented arcs ``a -> b``; ``contains`` resolves 0 vs 4 pi."""
    if a.shape[0] == 0:
        raise ValueError("boundary is empty")
    n, length, _ = _arc_parameters(a, b)
    candidates = _REFERENCE_CANDIDATES.to(device=a.device)
    # The triangle sum is singular where an arc passes through -r and the
    # 0 / 4 pi decision needs r off the boundary: take the clearest candidate.
    clearance = torch.minimum(
        _point_arc_distance(candidates, a, n, length).min(dim=1).values,
        _point_arc_distance(-candidates, a, n, length).min(dim=1).values,
    )
    ref = candidates[int(torch.argmax(clearance))]
    total = triangle_signed_area(ref.expand_as(a), a, b).sum()
    area = torch.remainder(total, FOUR_PI)
    if contains is None:
        return area
    ref_inside = bool(contains(ref.unsqueeze(0))[0])
    slack = 1.0e-9
    if ref_inside and float(area) < slack:
        return area + FOUR_PI
    if not ref_inside and float(area) > FOUR_PI - slack:
        return area - FOUR_PI
    return area


def boolean_region_area(
    loops: Sequence[OrientedLoop],
    evaluate: Callable[[Sequence[Tensor]], Tensor],
    region_contains: Callable[[Tensor], Tensor],
) -> float:
    """Exact area (steradians) of a boolean combination of polygon regions.

    ``evaluate`` maps per-loop membership tensors to the result's membership;
    ``region_contains`` is the result's own point predicate.
    """
    if not loops:
        raise ValueError("at least one polygon is required")
    a = torch.cat([loop.vertices for loop in loops])
    b = torch.cat([torch.roll(loop.vertices, shifts=-1, dims=0) for loop in loops])
    owner = torch.cat(
        [
            torch.full((loop.vertices.shape[0],), index, dtype=torch.int64)
            for index, loop in enumerate(loops)
        ]
    )
    n, length, u = _arc_parameters(a, b)
    splits = _split_angles(a, n, length, owner, b)

    piece_a: list[Tensor] = []
    piece_b: list[Tensor] = []
    piece_mid: list[Tensor] = []
    piece_owner: list[int] = []
    for edge in range(int(a.shape[0])):
        t = torch.cat(
            [a.new_zeros(1), torch.sort(splits[edge]).values, length[edge : edge + 1]]
        )
        keep = torch.cat([torch.ones(1, dtype=torch.bool), torch.diff(t) > _TOL_RAD])
        t = t[keep]
        if float(length[edge] - t[-1]) > _TOL_RAD:
            t = torch.cat([t, length[edge : edge + 1]])
        else:
            t[-1] = length[edge]
        t0, t1 = t[:-1], t[1:]
        mid = 0.5 * (t0 + t1)

        def at(theta: Tensor, edge: int = edge) -> Tensor:
            return (
                torch.cos(theta).unsqueeze(-1) * a[edge]
                + torch.sin(theta).unsqueeze(-1) * u[edge]
            )

        piece_a.append(at(t0))
        piece_b.append(at(t1))
        piece_mid.append(at(mid))
        piece_owner.extend([int(owner[edge])] * int(t0.shape[0]))
    pa = torch.cat(piece_a)
    pb = torch.cat(piece_b)
    pm = torch.cat(piece_mid)
    powner = torch.tensor(piece_owner, dtype=torch.int64)
    pn = _unit(torch.linalg.cross(pa, pb, dim=-1))
    tangent = torch.linalg.cross(pn, pm, dim=-1)

    left_values: list[Tensor] = []
    right_values: list[Tensor] = []
    for index, loop in enumerate(loops):
        own = powner == index
        la = loop.vertices
        lb = torch.roll(la, shifts=-1, dims=0)
        ln, llen, _ = _arc_parameters(la, lb)
        # Exact culling: a cap of radius < 90 deg around the vertices contains
        # every minor-arc edge, so pieces outside it are off this boundary and
        # on the side away from the small-area region.
        centre = _unit(la.sum(dim=0))
        radius = float(torch.acos((la @ centre).clamp(-1.0, 1.0)).max())
        near = own.clone()
        if radius < 0.5 * math.pi - 1.0e-6:
            near |= torch.acos((pm @ centre).clamp(-1.0, 1.0)) <= radius + 1.0e-9
        else:
            near[:] = True
        small_region = loop_left_area(la) <= 2.0 * math.pi
        on_boundary = torch.zeros_like(own)
        same_direction = torch.zeros_like(own)
        inside = torch.full_like(own, not small_region)
        candidates = near & ~own
        if bool(candidates.any()):
            pm_c = pm[candidates]
            on_edge = _point_arc_distance(pm_c, la, ln, llen).min(dim=1)
            on_c = on_edge.values <= _TOL_RAD
            edge_tangent = torch.linalg.cross(ln[on_edge.indices], pm_c, dim=-1)
            on_boundary[candidates] = on_c
            same_direction[candidates] = (edge_tangent * tangent[candidates]).sum(
                -1
            ) > 0.0
            off_c = ~on_c
            if bool(off_c.any()):
                inside_c = inside[candidates]
                inside_c[off_c] = loop.contains(pm_c[off_c])
                inside[candidates] = inside_c
        off = ~(own | on_boundary)
        left = own | (on_boundary & same_direction) | (off & inside)
        right = (on_boundary & ~same_direction) | (off & inside)
        left_values.append(left)
        right_values.append(right)
    result_left = evaluate(left_values)
    result_right = evaluate(right_values)
    boundary = result_left != result_right
    if not bool(boundary.any()):
        # No boundary: the result is empty or the whole sphere.
        probe = _REFERENCE_CANDIDATES[:1].to(device=a.device)
        return FOUR_PI if bool(region_contains(probe)[0]) else 0.0
    forward = result_left[boundary]
    start = torch.where(forward.unsqueeze(-1), pa[boundary], pb[boundary])
    end = torch.where(forward.unsqueeze(-1), pb[boundary], pa[boundary])
    mids = pm[boundary]
    # Coincident pieces from different operands describe one boundary piece.
    separation = torch.cdist(mids, mids)
    duplicate = torch.triu(separation <= _TOL_RAD, diagonal=1).any(dim=0)
    start, end = start[~duplicate], end[~duplicate]
    return float(_boundary_area(start, end, region_contains))
