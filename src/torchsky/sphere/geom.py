"""Spherical geometry primitives and polygon utilities."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Literal, Protocol, TYPE_CHECKING

import torch
from torch import Tensor

if TYPE_CHECKING:
    import numpy as np

try:
    from torch.func import vmap
except ImportError:
    vmap = None

from .. import _ops

from . import healpix as _healpix
from .core import (
    great_circle_distance,
    lonlat_to_unit_xyz,
    unit_xyz_to_lonlat,
    wrap_longitude,
)

_PLANAR_PIP_CPP_MIN_OPS = 2_000_000
# Caps peak memory for M×K pairwise dot products (winding fallback, convex half-spaces).
_GEOM_PAIRWISE_CHUNK: int = max(
    1024, int(os.environ.get("TORCHSKY_GEOM_PAIRWISE_CHUNK", "8192"))
)


@dataclass(frozen=True)
class LonLat:
    """Spherical coordinates in degrees."""

    lon: float
    lat: float

    def to_unit_xyz(self) -> "UnitVector3d":
        v = lonlat_to_unit_xyz(self.lon, self.lat)
        return UnitVector3d(float(v[0]), float(v[1]), float(v[2]))


@dataclass(frozen=True)
class Vector3d:
    """3D Cartesian vector."""

    x: float
    y: float
    z: float

    def norm(self) -> float:
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)

    def to_unit(self) -> "UnitVector3d":
        n = self.norm()
        if n < 1e-15:
            raise ValueError("cannot normalize zero vector")
        return UnitVector3d(self.x / n, self.y / n, self.z / n)


@dataclass(frozen=True)
class UnitVector3d:
    """Unit-length 3D Cartesian vector."""

    x: float
    y: float
    z: float

    def to_lonlat(self, wrap_center_deg: float = 180.0) -> LonLat:
        v = torch.tensor([self.x, self.y, self.z], dtype=torch.float64)
        lon, lat = unit_xyz_to_lonlat(v, wrap_center_deg=wrap_center_deg)
        return LonLat(float(lon), float(lat))


def _as_geom_tensor(
    value: Tensor | float | list[float], *, like: Tensor | None = None
) -> Tensor:
    tensor = torch.as_tensor(value)
    if like is not None:
        return tensor.to(device=like.device, dtype=like.dtype)
    if tensor.device.type == "mps":
        return tensor.to(dtype=torch.float32)
    return tensor.to(dtype=torch.float64)


def _validate_nonnegative_finite(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(f"{name} must be finite and nonnegative")
    return value


def _validate_ellipse_axes(semi_major_deg: float, semi_minor_deg: float) -> None:
    major = float(semi_major_deg)
    minor = float(semi_minor_deg)
    if not math.isfinite(major) or not math.isfinite(minor):
        raise ValueError("ellipse axes must be finite")
    if minor <= 0.0 or major <= 0.0:
        raise ValueError("semi_major_deg and semi_minor_deg must be positive")
    if minor > major:
        raise ValueError("semi_minor_deg must be <= semi_major_deg")
    if major >= 90.0:
        raise ValueError("focal-form semi_major_deg must be less than 90 degrees")


def _normalize_polygon_vertices(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
) -> tuple[Tensor, Tensor]:
    lon_t = _as_geom_tensor(lon_deg)
    lat_t = _as_geom_tensor(lat_deg, like=lon_t)
    if lon_t.ndim != 1 or lat_t.ndim != 1:
        raise ValueError("polygon vertices must be 1D sequences")
    if lon_t.shape[0] != lat_t.shape[0]:
        raise ValueError("lon_deg and lat_deg must have the same number of vertices")
    if lon_t.shape[0] < 3:
        raise ValueError("polygon must contain at least three vertices")

    if lon_t.shape[0] >= 4:
        # Accept closed polygons and normalize to open [N] form.
        if bool(torch.isclose(lon_t[0], lon_t[-1], atol=1e-12, rtol=0.0)) and bool(
            torch.isclose(lat_t[0], lat_t[-1], atol=1e-12, rtol=0.0)
        ):
            lon_t = lon_t[:-1]
            lat_t = lat_t[:-1]
    if lon_t.shape[0] < 3:
        raise ValueError("polygon must contain at least three distinct vertices")
    return lon_t, lat_t


def _signed_area_triangle(a: Tensor, b: Tensor, c: Tensor) -> Tensor:
    det = (a * torch.cross(b, c, dim=-1)).sum(dim=-1)
    den = 1.0 + (a * b).sum(dim=-1) + (b * c).sum(dim=-1) + (c * a).sum(dim=-1)
    return 2.0 * torch.atan2(det, den.clamp(min=1e-15))


def _points_on_any_polygon_edge(
    points: Tensor, vertices: Tensor, atol_rad: float
) -> Tensor:
    """(M,) mask: each query point lies on some polygon edge (great-circle arc), within atol."""
    v0 = vertices
    v1 = torch.roll(vertices, shifts=-1, dims=0)
    m = int(points.shape[0])
    out = torch.zeros(m, dtype=torch.bool, device=points.device)
    for j in range(int(vertices.shape[0])):
        out |= _arc_contains_point(v0[j], v1[j], points, atol_rad)
    return out


def _points_near_any_vertex(
    points: Tensor, vertices: Tensor, *, eps: float = 1e-14
) -> Tensor:
    """(M,) mask: point projects to a vertex within `eps` in tangent distance (original inclusive rule)."""
    m = int(points.shape[0])
    out = torch.zeros(m, dtype=torch.bool, device=points.device)
    chunk = _GEOM_PAIRWISE_CHUNK
    for m0 in range(0, m, chunk):
        m1 = min(m0 + chunk, m)
        pc = points[m0:m1]
        pv = pc @ vertices.T
        proj = vertices.unsqueeze(0) - pc.unsqueeze(1) * pv.unsqueeze(-1)
        proj_norm = torch.linalg.norm(proj, dim=-1)
        out[m0:m1] = (proj_norm < eps).any(dim=-1)
    return out


def _winding_fallback_chunked(
    points: Tensor, vertices: Tensor, *, chunk: int | None = None
) -> Tensor:
    """Tangent-plane winding magnitude per point; same math as the former dense einsum path."""
    if chunk is None:
        chunk = _GEOM_PAIRWISE_CHUNK
    m = int(points.shape[0])
    device, dtype = points.device, points.dtype
    winding = torch.empty(m, dtype=dtype, device=device)
    for m0 in range(0, m, chunk):
        m1 = min(m0 + chunk, m)
        pc = points[m0:m1]
        pv = pc @ vertices.T
        proj = vertices.unsqueeze(0) - pc.unsqueeze(1) * pv.unsqueeze(-1)
        proj_norm = torch.linalg.norm(proj, dim=-1, keepdim=True)
        proj = proj / proj_norm.clamp_min(1e-14)
        proj_next = torch.roll(proj, shifts=-1, dims=1)
        cross = torch.cross(proj, proj_next, dim=-1)
        num = (pc.unsqueeze(1) * cross).sum(dim=-1)
        den = torch.clamp((proj * proj_next).sum(dim=-1), -1.0, 1.0)
        winding[m0:m1] = torch.atan2(num, den).sum(dim=-1)
    return winding


def _arc_contains_point(a: Tensor, b: Tensor, p: Tensor, tol_rad: float) -> Tensor:
    """
    Check whether point(s) `p` lie on the minor great-circle arc from `a` to `b`.

    a,b: [3]
    p: [..., 3]
    """
    tol_rad = _validate_nonnegative_finite(tol_rad, "tol_rad")
    a, b, p = torch.broadcast_tensors(a, b, p)
    cross_ab = torch.linalg.cross(a, b, dim=-1)
    normal_norm = torch.linalg.norm(cross_ab, dim=-1)
    dot_ab = torch.clamp((a * b).sum(dim=-1), -1.0, 1.0)
    proper = normal_norm > 1e-15
    point_arc = (~proper) & (dot_ab >= 0.0)

    safe_norm = torch.where(proper, normal_norm, torch.ones_like(normal_norm))
    normal = cross_ab / safe_norm.unsqueeze(-1)
    arc_length = torch.atan2(normal_norm, dot_ab)

    cross_ap = torch.linalg.cross(a, p, dim=-1)
    dot_ap = torch.clamp((a * p).sum(dim=-1), -1.0, 1.0)
    signed = torch.atan2((normal * cross_ap).sum(dim=-1), dot_ap)
    signed = torch.remainder(signed, 2.0 * math.pi)
    plane_slack = 8.0 * torch.finfo(p.dtype).eps
    plane_tol = math.sin(min(tol_rad, math.pi / 2.0)) + plane_slack
    on_circle = torch.abs((p * normal).sum(dim=-1)) <= plane_tol
    on_proper = (
        proper
        & on_circle
        & ((signed <= arc_length + tol_rad) | (signed >= 2.0 * math.pi - tol_rad))
    )

    point_distance = torch.atan2(torch.linalg.norm(cross_ap, dim=-1), dot_ap)
    return on_proper | (point_arc & (point_distance <= tol_rad))


def _great_circle_arcs_intersect_matrix(
    a0: Tensor,
    a1: Tensor,
    b0: Tensor,
    b1: Tensor,
    *,
    tol_rad: float = 1e-10,
) -> Tensor:
    """
    Pairwise arc intersection matrix.

    a0,a1: [E1,3], b0,b1: [E2,3]
    returns: [E1,E2] bool
    """
    tol_rad = _validate_nonnegative_finite(tol_rad, "tol_rad")
    if (
        _ops.has_fixed_dispatcher()
        and a0.device.type == "cpu"
        and a1.device.type == "cpu"
        and b0.device.type == "cpu"
        and b1.device.type == "cpu"
    ):
        return _ops.sphere_arc_intersections_matrix(
            a0.detach().contiguous(),
            a1.detach().contiguous(),
            b0.detach().contiguous(),
            b1.detach().contiguous(),
            float(tol_rad),
        )

    n1 = torch.cross(a0, a1, dim=-1)
    n2 = torch.cross(b0, b1, dim=-1)
    n1_norm = torch.linalg.norm(n1, dim=-1)
    n2_norm = torch.linalg.norm(n2, dim=-1)
    dot1 = torch.clamp((a0 * a1).sum(dim=-1), -1.0, 1.0)
    dot2 = torch.clamp((b0 * b1).sum(dim=-1), -1.0, 1.0)
    proper1 = n1_norm > 1e-15
    proper2 = n2_norm > 1e-15
    point1 = (~proper1) & (dot1 >= 0.0)
    point2 = (~proper2) & (dot2 >= 0.0)

    n1 = n1 / torch.where(proper1, n1_norm, torch.ones_like(n1_norm)).unsqueeze(-1)
    n2 = n2 / torch.where(proper2, n2_norm, torch.ones_like(n2_norm)).unsqueeze(-1)

    a0e = a0[:, None, :]
    a1e = a1[:, None, :]
    b0e = b0[None, :, :]
    b1e = b1[None, :, :]
    proper_pair = proper1.unsqueeze(1) & proper2.unsqueeze(0)

    point_distance = torch.atan2(
        torch.linalg.norm(torch.linalg.cross(a0e, b0e, dim=-1), dim=-1),
        torch.clamp((a0e * b0e).sum(dim=-1), -1.0, 1.0),
    )
    hits = point1.unsqueeze(1) & point2.unsqueeze(0) & (point_distance <= tol_rad)
    hits |= (
        point1.unsqueeze(1)
        & proper2.unsqueeze(0)
        & _arc_contains_point(b0e, b1e, a0e, tol_rad)
    )
    hits |= (
        proper1.unsqueeze(1)
        & point2.unsqueeze(0)
        & _arc_contains_point(a0e, a1e, b0e, tol_rad)
    )

    p = torch.cross(n1[:, None, :], n2[None, :, :], dim=-1)
    p_norm = torch.linalg.norm(p, dim=-1)
    distinct_circles = p_norm > 1e-15
    i1 = p / torch.where(distinct_circles, p_norm, torch.ones_like(p_norm)).unsqueeze(
        -1
    )
    i2 = -i1

    on_a_i1 = _arc_contains_point(a0e, a1e, i1, tol_rad)
    on_a_i2 = _arc_contains_point(a0e, a1e, i2, tol_rad)
    on_b_i1 = _arc_contains_point(b0e, b1e, i1, tol_rad)
    on_b_i2 = _arc_contains_point(b0e, b1e, i2, tol_rad)
    hits |= proper_pair & distinct_circles & ((on_a_i1 & on_b_i1) | (on_a_i2 & on_b_i2))

    coincident = proper_pair & ~distinct_circles
    coincident_overlap = (
        _arc_contains_point(a0e, a1e, b0e, tol_rad)
        | _arc_contains_point(a0e, a1e, b1e, tol_rad)
        | _arc_contains_point(b0e, b1e, a0e, tol_rad)
        | _arc_contains_point(b0e, b1e, a1e, tol_rad)
    )
    return hits | (coincident & coincident_overlap)


def _sorted_unique_intersection(left: Tensor, right: Tensor) -> Tensor:
    """Intersect sorted unique integer tensors with searchsorted-based membership."""
    if left.numel() == 0 or right.numel() == 0:
        return left[:0]
    idx = torch.searchsorted(right, left)
    idx_safe = idx.clamp(max=right.numel() - 1)
    mask = (idx < right.numel()) & (right[idx_safe] == left)
    return left[mask]


def _sorted_unique_difference(left: Tensor, right: Tensor) -> Tensor:
    """Compute sorted unique left-minus-right for integer tensors."""
    if left.numel() == 0 or right.numel() == 0:
        return left
    idx = torch.searchsorted(right, left)
    idx_safe = idx.clamp(max=right.numel() - 1)
    mask = (idx < right.numel()) & (right[idx_safe] == left)
    return left[~mask]


def _planar_point_in_polygon(
    points_xy: Tensor,
    polygon_xy: Tensor,
    *,
    eps: float = 1e-15,
) -> Tensor:
    """
    Even-odd rule for a simple planar polygon.

    points_xy: [M, 2]
    polygon_xy: [N, 2]
    returns: [M] bool
    """
    eps = _validate_nonnegative_finite(eps, "eps")
    if (
        _ops.has_fixed_dispatcher()
        and points_xy.device.type == "cpu"
        and polygon_xy.device.type == "cpu"
        and (points_xy.shape[0] * polygon_xy.shape[0]) >= _PLANAR_PIP_CPP_MIN_OPS
    ):
        return _ops.sphere_planar_point_in_polygon(
            points_xy.detach().contiguous(),
            polygon_xy.detach().contiguous(),
            float(eps),
        )

    x = points_xy[:, 0]
    y = points_xy[:, 1]
    inside = torch.zeros(points_xy.shape[0], dtype=torch.bool, device=points_xy.device)
    for i in range(int(polygon_xy.shape[0])):
        j = (i + 1) % int(polygon_xy.shape[0])
        xi, yi = polygon_xy[i]
        xj, yj = polygon_xy[j]
        crossing = (yi > y) != (yj > y)
        if not bool(crossing.any()):
            continue
        active = torch.nonzero(crossing, as_tuple=True)[0]
        x_inter = xi + (xj - xi) * (y[active] - yi) / (yj - yi)
        inside[active] ^= x[active] < x_inter
    return inside


def _destination_lonlat_deg(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    bearing_rad: Tensor | float,
    distance_rad: Tensor | float,
) -> tuple[Tensor, Tensor]:
    """Great-circle destination from start point + initial bearing + angular distance."""
    lon1 = torch.deg2rad(torch.as_tensor(lon_deg))
    lat1 = torch.deg2rad(torch.as_tensor(lat_deg))
    bearing = torch.as_tensor(bearing_rad)
    dist = torch.as_tensor(distance_rad)
    lon1, lat1, bearing, dist = torch.broadcast_tensors(lon1, lat1, bearing, dist)

    sin_lat2 = torch.sin(lat1) * torch.cos(dist) + torch.cos(lat1) * torch.sin(
        dist
    ) * torch.cos(bearing)
    lat2 = torch.asin(torch.clamp(sin_lat2, -1.0, 1.0))

    y = torch.sin(bearing) * torch.sin(dist) * torch.cos(lat1)
    x = torch.cos(dist) - (torch.sin(lat1) * torch.sin(lat2))
    lon2 = lon1 + torch.atan2(y, x)
    lon2 = torch.remainder(lon2, 2.0 * math.pi)
    return torch.rad2deg(lon2), torch.rad2deg(lat2)


def _make_tangent_basis(center: Tensor) -> tuple[Tensor, Tensor]:
    ref = torch.tensor([0.0, 0.0, 1.0], dtype=center.dtype, device=center.device)
    if float(torch.abs((ref * center).sum()).item()) > 0.9:
        ref = torch.tensor([1.0, 0.0, 0.0], dtype=center.dtype, device=center.device)
    e1 = torch.cross(ref, center, dim=-1)
    e1 = e1 / torch.linalg.norm(e1).clamp_min(1e-15)
    e2 = torch.cross(center, e1, dim=-1)
    return e1, e2


def _gnomonic_project(vertices_xyz: Tensor, center: Tensor) -> "np.ndarray":
    e1, e2 = _make_tangent_basis(center)
    den = (vertices_xyz * center.unsqueeze(0)).sum(dim=-1)
    if bool(torch.any(den <= 1e-8)):
        raise ValueError(
            "polygon cannot be projected gnomonically (vertices must lie in one open hemisphere)"
        )
    x = (vertices_xyz * e1.unsqueeze(0)).sum(dim=-1) / den
    y = (vertices_xyz * e2.unsqueeze(0)).sum(dim=-1) / den
    return torch.stack((x, y), dim=-1).cpu().numpy()


def _signed_area_2d(xy: "np.ndarray") -> float:
    import numpy as np

    x = xy[:, 0]
    y = xy[:, 1]
    return 0.5 * float(np.sum(x * np.roll(y, -1) - y * np.roll(x, -1)))


def _is_point_in_triangle_2d_vec(
    p: Tensor, a: Tensor, b: Tensor, c: Tensor, eps: float = 1e-15
) -> Tensor:
    """Vectorized check for multiple points p [M, 2] in triangle (a, b, c)."""
    v0 = c - a
    v1 = b - a
    v2 = p - a

    den = (v0[0] * v1[1]) - (v1[0] * v0[1])
    if abs(den) < eps:
        return torch.zeros(p.shape[0], dtype=torch.bool, device=p.device)

    u = (v2[:, 0] * v1[1] - v1[0] * v2[:, 1]) / den
    v = (v0[0] * v2[:, 1] - v2[:, 0] * v0[1]) / den

    return (u >= -eps) & (v >= -eps) & (u + v <= 1.0 + eps)


def _triangulate_simple_polygon_2d(xy_np: "np.ndarray") -> list[tuple[int, int, int]]:
    xy = torch.from_numpy(xy_np)
    n = xy.shape[0]
    if n < 3:
        raise ValueError("need at least three vertices")
    if n == 3:
        return [(0, 1, 2)]

    orientation = 1.0 if _signed_area_2d(xy_np) >= 0.0 else -1.0
    indices = list(range(n))
    triangles: list[tuple[int, int, int]] = []
    guard = 0
    max_iters = n * n

    while len(indices) > 3 and guard < max_iters:
        guard += 1
        ear_found = False
        m = len(indices)
        for k in range(m):
            i_prev = indices[(k - 1) % m]
            i_curr = indices[k]
            i_next = indices[(k + 1) % m]
            a = xy[i_prev]
            b = xy[i_curr]
            c = xy[i_next]

            cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
            if cross * orientation <= 1e-15:
                continue

            # Vectorized check for all other vertices
            other_indices = [
                idx for idx in indices if idx not in (i_prev, i_curr, i_next)
            ]
            if not other_indices:
                has_inner = False
            else:
                other_points = xy[other_indices]
                inner_mask = _is_point_in_triangle_2d_vec(other_points, a, b, c)
                has_inner = bool(inner_mask.any())

            if has_inner:
                continue

            triangles.append((i_prev, i_curr, i_next))
            del indices[k]
            ear_found = True
            break

        if not ear_found:
            break

    if len(indices) == 3:
        triangles.append((indices[0], indices[1], indices[2]))
    if len(indices) != 3:
        raise ValueError(
            "failed to triangulate polygon; ensure polygon is simple and non-self-intersecting"
        )
    if not triangles:
        raise ValueError(
            "failed to triangulate polygon; ensure polygon is simple and non-self-intersecting"
        )
    return triangles


def spherical_triangle_area(
    lon1_deg: Tensor | float,
    lat1_deg: Tensor | float,
    lon2_deg: Tensor | float,
    lat2_deg: Tensor | float,
    lon3_deg: Tensor | float,
    lat3_deg: Tensor | float,
    *,
    degrees: bool = False,
) -> Tensor:
    """
    Spherical triangle area on the unit sphere.

    Returns steradians by default, or square degrees when `degrees=True`.
    """
    a = lonlat_to_unit_xyz(lon1_deg, lat1_deg)
    b = lonlat_to_unit_xyz(lon2_deg, lat2_deg)
    c = lonlat_to_unit_xyz(lon3_deg, lat3_deg)
    area_sr = torch.abs(_signed_area_triangle(a, b, c))
    if degrees:
        return area_sr * ((180.0 / math.pi) * (180.0 / math.pi))
    return area_sr


def spherical_polygon_signed_area(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    degrees: bool = False,
) -> Tensor:
    """
    Signed area of a simple spherical polygon on the unit sphere.

    Contract:
    - vertices must define a simple, non-self-intersecting polygon
    - polygon is expected to lie in a region smaller than a full sphere
    - sign follows vertex winding orientation
    """
    lon_t, lat_t = _normalize_polygon_vertices(lon_deg, lat_deg)
    v = lonlat_to_unit_xyz(lon_t, lat_t)

    a = v[0].expand(v.shape[0] - 2, 3)
    b = v[1:-1]
    c = v[2:]
    area = _signed_area_triangle(a, b, c).sum()
    if degrees:
        return area * ((180.0 / math.pi) * (180.0 / math.pi))
    return area


def spherical_polygon_area(
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    degrees: bool = False,
    oriented: bool = False,
) -> Tensor:
    """
    Area of a simple spherical polygon.

    By default returns the non-oriented enclosed area in [0, 2*pi] steradians.
    Set `oriented=True` to preserve signed winding orientation.
    """
    signed = spherical_polygon_signed_area(lon_deg, lat_deg, degrees=False)
    if oriented:
        area = signed
    else:
        area = torch.abs(signed)
        full = 4.0 * math.pi
        if float(area.item()) > 2.0 * math.pi:
            area = torch.tensor(full, dtype=area.dtype, device=area.device) - area
    if degrees:
        return area * ((180.0 / math.pi) * (180.0 / math.pi))
    return area


def spherical_polygon_contains(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    poly_lon_deg: Tensor | list[float],
    poly_lat_deg: Tensor | list[float],
    *,
    inclusive: bool = True,
    atol_deg: float = 1e-10,
    inside_lon_deg: Tensor | float | None = None,
    inside_lat_deg: Tensor | float | None = None,
) -> Tensor:
    """
    Point-in-polygon for simple (possibly non-convex) spherical polygons.

    Uses tangent-plane winding number around each query point.
    """
    atol_deg = _validate_nonnegative_finite(atol_deg, "atol_deg")
    lon_t = _as_geom_tensor(lon_deg)
    lat_t = _as_geom_tensor(lat_deg, like=lon_t)
    lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
    shape = lon_t.shape
    lon_poly, lat_poly = _normalize_polygon_vertices(poly_lon_deg, poly_lat_deg)
    lon_poly = lon_poly.to(device=lon_t.device, dtype=lon_t.dtype)
    lat_poly = lat_poly.to(device=lon_t.device, dtype=lon_t.dtype)
    vertices = lonlat_to_unit_xyz(lon_poly, lat_poly).to(dtype=lon_t.dtype)
    points = lonlat_to_unit_xyz(lon_t, lat_t).reshape(-1, 3).to(dtype=lon_t.dtype)

    # Fast/stable path: if polygon is in one hemisphere, gnomonic projection maps
    # great-circle edges to straight segments exactly.
    center = vertices.mean(dim=0)
    center = center / torch.linalg.norm(center).clamp_min(1e-15)
    vden = (vertices * center.unsqueeze(0)).sum(dim=-1)
    use_gnomonic = bool(torch.all(vden > 1e-10))

    if use_gnomonic:
        e1, e2 = _make_tangent_basis(center)
        vx = (vertices * e1.unsqueeze(0)).sum(dim=-1) / vden
        vy = (vertices * e2.unsqueeze(0)).sum(dim=-1) / vden
        poly_xy = torch.stack((vx, vy), dim=-1)

        pden = (points * center.unsqueeze(0)).sum(dim=-1)
        valid = pden > 1e-12
        inside = torch.zeros(points.shape[0], dtype=torch.bool, device=points.device)
        if valid.any():
            p = points[valid]
            px = (p * e1.unsqueeze(0)).sum(dim=-1) / pden[valid]
            py = (p * e2.unsqueeze(0)).sum(dim=-1) / pden[valid]
            inside_valid = _planar_point_in_polygon(
                torch.stack((px, py), dim=-1), poly_xy
            )
            inside[valid] = inside_valid
    else:
        # Fallback path for polygons not entirely in one hemisphere.
        winding = _winding_fallback_chunked(points, vertices)
        inside = torch.abs(winding) > math.pi

    if inclusive:
        atol_rad = math.radians(atol_deg)
        on_edge = _points_on_any_polygon_edge(points, vertices, atol_rad)
        near_v = _points_near_any_vertex(points, vertices, eps=1e-14)
        inside = inside | near_v | on_edge

    if inside_lon_deg is not None and inside_lat_deg is not None:
        ref = spherical_polygon_contains(
            _as_geom_tensor(inside_lon_deg, like=lon_poly),
            _as_geom_tensor(inside_lat_deg, like=lon_poly),
            lon_poly,
            lat_poly,
            inclusive=True,
            atol_deg=atol_deg,
        )
        if not bool(torch.as_tensor(ref).item()):
            inside = ~inside

    return inside.reshape(shape)


def convex_polygon_contains(
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    poly_lon_deg: Tensor | list[float],
    poly_lat_deg: Tensor | list[float],
    *,
    atol: float = 1e-12,
) -> Tensor:
    """Fast half-space point-in-convex-spherical-polygon predicate."""
    atol = _validate_nonnegative_finite(atol, "atol")
    v_poly = lonlat_to_unit_xyz(
        torch.as_tensor(poly_lon_deg), torch.as_tensor(poly_lat_deg)
    )
    if v_poly.ndim != 2 or v_poly.shape[0] < 3:
        raise ValueError("polygon must contain at least three vertices")

    v_next = torch.roll(v_poly, shifts=-1, dims=0)
    normals = torch.cross(v_poly, v_next, dim=-1)
    normal_norm = torch.linalg.norm(normals, dim=-1)
    if bool(torch.any(normal_norm <= 1e-15)):
        raise ValueError("polygon edges must have distinct, non-antipodal endpoints")
    normals = normals / normal_norm.unsqueeze(-1)

    centroid = v_poly.sum(dim=0)
    centroid = centroid / torch.linalg.norm(centroid).clamp_min(1e-15)
    orient = torch.sign((normals * centroid).sum(dim=-1))
    orient = torch.where(orient == 0, torch.ones_like(orient), orient)
    normals = normals * orient.unsqueeze(-1)

    points = lonlat_to_unit_xyz(lon_deg, lat_deg)
    shape = points.shape[:-1]
    pts = points.reshape(-1, 3)
    chunk = _GEOM_PAIRWISE_CHUNK
    bits: list[Tensor] = []
    for m0 in range(0, pts.shape[0], chunk):
        m1 = min(m0 + chunk, pts.shape[0])
        d = pts[m0:m1] @ normals.T
        bits.append(torch.all(d >= -atol, dim=-1))
    return torch.cat(bits, dim=0).reshape(shape)


def query_polygon_general(
    nside: int,
    lon_deg: Tensor | list[float],
    lat_deg: Tensor | list[float],
    *,
    nest: bool = False,
    inclusive: bool = False,
) -> Tensor:
    """
    Query HEALPix pixels inside a simple spherical polygon (convex or non-convex).

    Non-convex polygons are triangulated via gnomonic-plane ear clipping.
    """
    lon_t, lat_t = _normalize_polygon_vertices(lon_deg, lat_deg)
    query_device = lon_t.device
    vertices_xyz = lonlat_to_unit_xyz(lon_t, lat_t).to(dtype=lon_t.dtype)

    if vertices_xyz.shape[0] == 3:
        return _healpix.query_polygon(
            nside, vertices_xyz, nest=nest, inclusive=inclusive
        )

    center = vertices_xyz.mean(dim=0)
    center = center / torch.linalg.norm(center).clamp_min(1e-15)
    xy = _gnomonic_project(vertices_xyz.detach().cpu(), center.detach().cpu())
    triangles = _triangulate_simple_polygon_2d(xy)

    chunks: list[Tensor] = []
    for i, j, k in triangles:
        verts = vertices_xyz[[i, j, k]].to(device=query_device)
        chunks.append(
            _healpix.query_polygon(nside, verts, nest=nest, inclusive=inclusive)
        )
    if not chunks:
        return torch.empty(0, dtype=torch.int64, device=query_device)
    return torch.unique(torch.cat(chunks)).to(torch.int64)


def spherical_polygons_intersect(
    lon1_deg: Tensor | list[float],
    lat1_deg: Tensor | list[float],
    lon2_deg: Tensor | list[float],
    lat2_deg: Tensor | list[float],
    *,
    inside1_lon_deg: Tensor | float | None = None,
    inside1_lat_deg: Tensor | float | None = None,
    inside2_lon_deg: Tensor | float | None = None,
    inside2_lat_deg: Tensor | float | None = None,
    atol_deg: float = 1e-8,
) -> bool:
    """
    Polygon intersection predicate for simple spherical polygons.

    Returns True if interiors intersect or boundaries cross/touch.
    """
    atol_deg = _validate_nonnegative_finite(atol_deg, "atol_deg")
    lon1, lat1 = _normalize_polygon_vertices(lon1_deg, lat1_deg)
    lon2, lat2 = _normalize_polygon_vertices(lon2_deg, lat2_deg)
    v1 = lonlat_to_unit_xyz(lon1, lat1)
    v2 = lonlat_to_unit_xyz(lon2, lat2).to(device=v1.device, dtype=v1.dtype)
    tol_rad = math.radians(atol_deg)

    v1_next = torch.roll(v1, shifts=-1, dims=0)
    v2_next = torch.roll(v2, shifts=-1, dims=0)

    edge_hits = _great_circle_arcs_intersect_matrix(
        v1, v1_next, v2, v2_next, tol_rad=tol_rad
    )
    if bool(edge_hits.any()):
        return True

    # If no edge crossings, containment of one representative vertex is sufficient.
    rep1_lon = inside1_lon_deg if inside1_lon_deg is not None else lon1[0]
    rep1_lat = inside1_lat_deg if inside1_lat_deg is not None else lat1[0]
    rep2_lon = inside2_lon_deg if inside2_lon_deg is not None else lon2[0]
    rep2_lat = inside2_lat_deg if inside2_lat_deg is not None else lat2[0]

    c12 = spherical_polygon_contains(
        rep1_lon,
        rep1_lat,
        lon2,
        lat2,
        inclusive=True,
        atol_deg=atol_deg,
        inside_lon_deg=inside2_lon_deg,
        inside_lat_deg=inside2_lat_deg,
    )
    if bool(torch.as_tensor(c12).item()):
        return True

    c21 = spherical_polygon_contains(
        rep2_lon,
        rep2_lat,
        lon1,
        lat1,
        inclusive=True,
        atol_deg=atol_deg,
        inside_lon_deg=inside1_lon_deg,
        inside_lat_deg=inside1_lat_deg,
    )
    if bool(torch.as_tensor(c21).item()):
        return True

    return False


@dataclass(frozen=True)
class PixelizedRegion:
    """Pixelized spherical region with set-like boolean operations."""

    nside: int
    nest: bool
    pixels: Tensor

    def __post_init__(self) -> None:
        if isinstance(self.nside, bool) or not isinstance(self.nside, int):
            raise TypeError("nside must be an integer")
        npix = _healpix.nside2npix(self.nside)
        if not isinstance(self.nest, bool):
            raise TypeError("nest must be a bool")
        raw = torch.as_tensor(self.pixels)
        if raw.dtype == torch.bool or raw.is_floating_point() or raw.is_complex():
            raise TypeError("pixels must contain integer indices")
        pix = raw.to(dtype=torch.int64).reshape(-1)
        if bool(torch.any((pix < 0) | (pix >= npix))):
            raise ValueError("pixel index out of range for nside")
        pix = torch.unique(pix)
        object.__setattr__(self, "pixels", pix)

    def _check_compatible(self, other: "PixelizedRegion") -> None:
        if self.nside != other.nside or self.nest != other.nest:
            raise ValueError("regions must have same nside and nest ordering")

    def union(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(
            self.nside, self.nest, torch.cat((self.pixels, other.pixels))
        )

    def intersection(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(
            self.nside,
            self.nest,
            _sorted_unique_intersection(self.pixels, other.pixels),
        )

    def difference(self, other: "PixelizedRegion") -> "PixelizedRegion":
        self._check_compatible(other)
        return PixelizedRegion(
            self.nside,
            self.nest,
            _sorted_unique_difference(self.pixels, other.pixels),
        )

    def area(self, *, degrees: bool = False) -> float:
        return float(self.pixels.numel()) * _healpix.nside2pixarea(
            self.nside, degrees=degrees
        )


@dataclass(frozen=True)
class RegionAreaEstimate:
    """Controlled-error area estimate summary over a multi-resolution nside ladder."""

    area_sr: float
    area_deg2: float
    nside: int
    pixels: int
    convergence_rel: float | None
    area_by_nside_sr: dict[str, float]
    pixels_by_nside: dict[str, int]


def _parse_nside_ladder(nsides: tuple[int, ...] | list[int]) -> tuple[int, ...]:
    ladder = tuple(nsides)
    if len(ladder) == 0:
        raise ValueError("nsides ladder must contain at least one nside")
    for i, n in enumerate(ladder):
        if n <= 0 or (n & (n - 1)) != 0:
            raise ValueError(f"invalid nside {n}: must be a positive power of two")
        if i > 0 and n <= ladder[i - 1]:
            raise ValueError("nsides ladder must be strictly increasing")
    return ladder


def _estimate_area_from_query(
    query_fn,
    *,
    nsides: tuple[int, ...] | list[int],
    nest: bool,
) -> RegionAreaEstimate:
    ladder = _parse_nside_ladder(nsides)
    area_by_nside_sr: dict[str, float] = {}
    pixels_by_nside: dict[str, int] = {}
    prev_area = None

    for nside in ladder:
        pix = torch.as_tensor(query_fn(nside, nest=nest), dtype=torch.int64).reshape(-1)
        pix = torch.unique(pix)
        count = int(pix.numel())
        area_sr = float(count) * _healpix.nside2pixarea(nside, degrees=False)
        area_by_nside_sr[str(nside)] = area_sr
        pixels_by_nside[str(nside)] = count
        prev_area = area_sr

    assert prev_area is not None
    convergence_rel = None
    if len(ladder) >= 2:
        a_prev = area_by_nside_sr[str(ladder[-2])]
        convergence_rel = abs(prev_area - a_prev) / max(abs(prev_area), 1e-15)
    return RegionAreaEstimate(
        area_sr=prev_area,
        area_deg2=prev_area * ((180.0 / math.pi) * (180.0 / math.pi)),
        nside=ladder[-1],
        pixels=pixels_by_nside[str(ladder[-1])],
        convergence_rel=convergence_rel,
        area_by_nside_sr=area_by_nside_sr,
        pixels_by_nside=pixels_by_nside,
    )


@dataclass(frozen=True)
class SphericalPolygon:
    """Simple spherical polygon with signed-area and containment contracts."""

    lon_deg: Tensor
    lat_deg: Tensor
    inside_lon_deg: float | None = None
    inside_lat_deg: float | None = None

    def __post_init__(self) -> None:
        lon_t, lat_t = _normalize_polygon_vertices(self.lon_deg, self.lat_deg)
        object.__setattr__(self, "lon_deg", lon_t)
        object.__setattr__(self, "lat_deg", lat_t)
        if (self.inside_lon_deg is None) ^ (self.inside_lat_deg is None):
            raise ValueError(
                "inside_lon_deg and inside_lat_deg must both be set or both be None"
            )

    def signed_area(self, *, degrees: bool = False) -> Tensor:
        return spherical_polygon_signed_area(
            self.lon_deg, self.lat_deg, degrees=degrees
        )

    def area(self, *, degrees: bool = False, oriented: bool = False) -> Tensor:
        return spherical_polygon_area(
            self.lon_deg, self.lat_deg, degrees=degrees, oriented=oriented
        )

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        return spherical_polygon_contains(
            lon_deg,
            lat_deg,
            self.lon_deg,
            self.lat_deg,
            inclusive=inclusive,
            atol_deg=atol_deg,
            inside_lon_deg=self.inside_lon_deg,
            inside_lat_deg=self.inside_lat_deg,
        )

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return query_polygon_general(nside, self.lon_deg, self.lat_deg, nest=nest)

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def intersects(self, other: "SphericalPolygon", *, atol_deg: float = 1e-8) -> bool:
        return spherical_polygons_intersect(
            self.lon_deg,
            self.lat_deg,
            other.lon_deg,
            other.lat_deg,
            inside1_lon_deg=self.inside_lon_deg,
            inside1_lat_deg=self.inside_lat_deg,
            inside2_lon_deg=other.inside_lon_deg,
            inside2_lat_deg=other.inside_lat_deg,
            atol_deg=atol_deg,
        )

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(
        self,
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps, backend=backend)

    def exact_union(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.union(other)
        return exact.union(other, cap_steps=cap_steps)

    def exact_intersection(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.intersection(other)
        return exact.intersection(other, cap_steps=cap_steps)

    def exact_difference(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.difference(other)
        return exact.difference(other, cap_steps=cap_steps)


@dataclass(frozen=True)
class SphericalMultiPolygon:
    """Collection of spherical polygons with boolean-style predicates."""

    polygons: tuple[SphericalPolygon, ...]

    def __post_init__(self) -> None:
        polys = tuple(self.polygons)
        if len(polys) == 0:
            raise ValueError("SphericalMultiPolygon requires at least one polygon")
        object.__setattr__(self, "polygons", polys)

    def area(self, *, degrees: bool = False, oriented: bool = False) -> Tensor:
        vals = [p.area(degrees=degrees, oriented=oriented) for p in self.polygons]
        out = vals[0]
        for v in vals[1:]:
            out = out + v
        return out

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        out = self.polygons[0].contains(
            lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg
        )
        for p in self.polygons[1:]:
            out = out | p.contains(
                lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg
            )
        return out

    def intersects(
        self,
        other: "SphericalPolygon | SphericalMultiPolygon",
        *,
        atol_deg: float = 1e-8,
    ) -> bool:
        others = (
            other.polygons if isinstance(other, SphericalMultiPolygon) else (other,)
        )
        for p in self.polygons:
            for q in others:
                if p.intersects(q, atol_deg=atol_deg):
                    return True
        return False

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        chunks = [p.query_pixels(nside, nest=nest) for p in self.polygons]
        if not chunks:
            return torch.empty(0, dtype=torch.int64)
        return torch.unique(torch.cat(chunks))

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(
        self,
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps, backend=backend)

    def exact_union(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.union(other)
        return exact.union(other, cap_steps=cap_steps)

    def exact_intersection(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.intersection(other)
        return exact.intersection(other, cap_steps=cap_steps)

    def exact_difference(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.difference(other)
        return exact.difference(other, cap_steps=cap_steps)


@dataclass(frozen=True)
class SphericalCap:
    """Spherical cap centered at (lon, lat) with angular radius in degrees."""

    lon_deg: float
    lat_deg: float
    radius_deg: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.lon_deg):
            raise ValueError("lon_deg must be finite")
        if not math.isfinite(self.lat_deg) or not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("lat_deg must be finite and in [-90, 90]")
        if not math.isfinite(self.radius_deg) or not 0.0 < self.radius_deg <= 180.0:
            raise ValueError("radius_deg must be finite and in (0, 180]")

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 0.0,
    ) -> Tensor:
        atol_deg = _validate_nonnegative_finite(atol_deg, "atol_deg")
        dist = great_circle_distance(
            self.lon_deg, self.lat_deg, lon_deg, lat_deg, degrees=True
        )
        if inclusive:
            return dist <= self.radius_deg + atol_deg
        return dist < self.radius_deg - atol_deg

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return _healpix.query_circle(
            nside,
            self.lon_deg,
            self.lat_deg,
            self.radius_deg,
            degrees=True,
            nest=nest,
        )

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(
        self,
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps, backend=backend)

    def exact_union(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.union(other)
        return exact.union(other, cap_steps=cap_steps)

    def exact_intersection(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.intersection(other)
        return exact.intersection(other, cap_steps=cap_steps)

    def exact_difference(
        self,
        other: "RegionOperand",
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        exact = self.to_exact(cap_steps=cap_steps, backend=backend)
        if isinstance(exact, NativeApproximateSphericalRegion):
            return exact.difference(other)
        return exact.difference(other, cap_steps=cap_steps)


@dataclass(frozen=True)
class SphericalBox:
    """Longitude/Latitude angle box."""

    lon_min: float
    lon_max: float
    lat_min: float
    lat_max: float

    def __post_init__(self) -> None:
        if not all(
            math.isfinite(value)
            for value in (self.lon_min, self.lon_max, self.lat_min, self.lat_max)
        ):
            raise ValueError("box bounds must be finite")
        if not -90.0 <= self.lat_min <= 90.0 or not -90.0 <= self.lat_max <= 90.0:
            raise ValueError("latitude bounds must be in [-90, 90]")
        if self.lat_min > self.lat_max:
            raise ValueError("lat_min must be <= lat_max")

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        lon_t = _as_geom_tensor(lon_deg)
        lat_t = _as_geom_tensor(lat_deg, like=lon_t)
        lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)

        if self.lon_min > self.lon_max:
            lon_0_360 = torch.remainder(lon_t, 360.0)
            lo = self.lon_min % 360.0
            hi = self.lon_max % 360.0
            if inclusive:
                lon_mask = (lon_0_360 >= lo - atol_deg) | (lon_0_360 <= hi + atol_deg)
            else:
                lon_mask = (lon_0_360 > lo + atol_deg) | (lon_0_360 < hi - atol_deg)
        elif self.lon_max - self.lon_min >= 360.0:
            lon_mask = torch.ones_like(lon_t, dtype=torch.bool)
        else:
            lon_wrapped = wrap_longitude(
                lon_t, center_deg=(self.lon_min + self.lon_max) / 2
            )
            if inclusive:
                lon_mask = (lon_wrapped >= self.lon_min - atol_deg) & (
                    lon_wrapped <= self.lon_max + atol_deg
                )
            else:
                lon_mask = (lon_wrapped > self.lon_min + atol_deg) & (
                    lon_wrapped < self.lon_max - atol_deg
                )

        if inclusive:
            lat_mask = (lat_t >= self.lat_min - atol_deg) & (
                lat_t <= self.lat_max + atol_deg
            )
        else:
            lat_mask = (lat_t > self.lat_min + atol_deg) & (
                lat_t < self.lat_max - atol_deg
            )
        return lon_mask & lat_mask

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return _healpix.query_box(
            nside,
            self.lon_min,
            self.lat_min,
            self.lon_max,
            self.lat_max,
            nest=nest,
        )

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)


@dataclass(frozen=True)
class SphericalEllipse:
    """Spherical ellipse."""

    lon_deg: float
    lat_deg: float
    semi_major_deg: float
    semi_minor_deg: float
    pa_deg: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.lon_deg):
            raise ValueError("lon_deg must be finite")
        if not math.isfinite(self.lat_deg) or not -90.0 <= self.lat_deg <= 90.0:
            raise ValueError("lat_deg must be finite and in [-90, 90]")
        _validate_ellipse_axes(self.semi_major_deg, self.semi_minor_deg)
        if not math.isfinite(self.pa_deg):
            raise ValueError("pa_deg must be finite")

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        # Focal distance calculation similar to query_ellipse
        major = float(math.radians(self.semi_major_deg))
        minor = float(math.radians(self.semi_minor_deg))
        cos_ratio = math.cos(major) / max(math.cos(minor), 1e-15)
        cos_ratio = min(1.0, max(-1.0, cos_ratio))
        focal_sep = math.acos(cos_ratio)
        pa = math.radians(self.pa_deg)

        lon_f1, lat_f1 = _destination_lonlat_deg(
            self.lon_deg, self.lat_deg, pa, focal_sep
        )
        lon_f2, lat_f2 = _destination_lonlat_deg(
            self.lon_deg, self.lat_deg, pa + math.pi, focal_sep
        )

        d1 = great_circle_distance(lon_f1, lat_f1, lon_deg, lat_deg, degrees=False)
        d2 = great_circle_distance(lon_f2, lat_f2, lon_deg, lat_deg, degrees=False)

        thresh = (2.0 * major) + math.radians(atol_deg)
        if inclusive:
            return (d1 + d2) <= thresh + 1e-12
        return (d1 + d2) < thresh - 1e-12

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return query_ellipse(
            nside,
            self.lon_deg,
            self.lat_deg,
            self.semi_major_deg,
            self.semi_minor_deg,
            pa_deg=self.pa_deg,
            nest=nest,
            inclusive=True,
        )

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def union(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: "RegionOperand") -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)


class RegionOperand(Protocol):
    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor: ...

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion: ...


def _require_spherical_geometry():
    try:
        from spherical_geometry.polygon import SphericalPolygon as _SGPolygon
    except Exception as exc:  # pragma: no cover - optional backend
        raise RuntimeError(
            "spherical-geometry is required for exact region boolean operations"
        ) from exc
    return _SGPolygon


def _sg_from_lonlat(lon_deg: Tensor, lat_deg: Tensor):
    import numpy as np

    sg_poly = _require_spherical_geometry()
    lon_np = lon_deg.detach().cpu().numpy().astype(np.float64)
    lat_np = lat_deg.detach().cpu().numpy().astype(np.float64)
    if hasattr(sg_poly, "from_lonlat"):
        return sg_poly.from_lonlat(lon_np, lat_np, degrees=True)
    return sg_poly.from_radec(lon_np, lat_np, degrees=True)


def _strip_closed_np(
    lon: "np.ndarray", lat: "np.ndarray"
) -> tuple["np.ndarray", "np.ndarray"]:
    import numpy as np

    if lon.size >= 4 and np.isclose(lon[0], lon[-1]) and np.isclose(lat[0], lat[-1]):
        return lon[:-1], lat[:-1]
    return lon, lat


def _numpy_sg_contains_lonlat_bulk(
    sg_poly: object, lon_deg: "np.ndarray", lat_deg: "np.ndarray"
) -> "np.ndarray":
    """Bulk ``(lon, lat)`` containment using ``spherical_geometry`` vector APIs when available.

    Falls back to per-point calls only when array APIs are missing or return an
    unexpected shape (matches the probe logic in ``benchmarks/bench_sphere_polygons``).
    """
    import numpy as np

    lon_deg = np.asarray(lon_deg, dtype=np.float64).reshape(-1)
    lat_deg = np.asarray(lat_deg, dtype=np.float64).reshape(-1)
    if lon_deg.shape != lat_deg.shape:
        raise ValueError("lon_deg and lat_deg must have the same shape")
    if hasattr(sg_poly, "contains_lonlat"):
        try:
            raw = sg_poly.contains_lonlat(lon_deg, lat_deg, degrees=True)
            out = np.asarray(raw, dtype=bool)
            if out.shape == lon_deg.shape:
                return out
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(sg_poly, "contains_radec"):
        try:
            raw = sg_poly.contains_radec(lon_deg, lat_deg, degrees=True)
            out = np.asarray(raw, dtype=bool)
            if out.shape == lon_deg.shape:
                return out
        except (TypeError, ValueError, AttributeError):
            pass
    if hasattr(sg_poly, "contains_lonlat"):
        return np.asarray(
            [
                bool(sg_poly.contains_lonlat(float(lo), float(la), degrees=True))
                for lo, la in zip(lon_deg, lat_deg, strict=False)
            ],
            dtype=bool,
        )
    if hasattr(sg_poly, "contains_radec"):
        return np.asarray(
            [
                bool(sg_poly.contains_radec(float(lo), float(la), degrees=True))
                for lo, la in zip(lon_deg, lat_deg, strict=False)
            ],
            dtype=bool,
        )
    raise TypeError("spherical_geometry polygon lacks contains_lonlat/contains_radec")


def _region_to_sg(region: RegionOperand, *, cap_steps: int = 256):
    if isinstance(region, NativeApproximateSphericalRegion):
        return _region_to_sg(region._region, cap_steps=cap_steps)
    if isinstance(region, ExactSphericalRegion):
        return region._sg_poly
    if isinstance(region, SphericalPolygon):
        return _sg_from_lonlat(region.lon_deg, region.lat_deg)
    if isinstance(region, SphericalMultiPolygon):
        sg_poly = _require_spherical_geometry()
        polys = [_sg_from_lonlat(p.lon_deg, p.lat_deg) for p in region.polygons]
        return sg_poly.multi_union(polys)
    if isinstance(region, SphericalCap):
        sg_poly = _require_spherical_geometry()
        return sg_poly.from_cone(
            region.lon_deg,
            region.lat_deg,
            region.radius_deg,
            degrees=True,
            steps=cap_steps,
        )
    if isinstance(region, SphericalBooleanRegion):
        left = _region_to_sg(region.left, cap_steps=cap_steps)
        right = _region_to_sg(region.right, cap_steps=cap_steps)
        if region.op == "union":
            return left.union(right)
        if region.op == "intersection":
            return left.intersection(right)
        return left.intersection(right.invert_polygon())
    raise TypeError(f"unsupported region type for exact conversion: {type(region)!r}")


def _region_contains(
    region: RegionOperand,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    *,
    inclusive: bool = True,
    atol_deg: float = 1e-10,
) -> Tensor:
    if isinstance(region, SphericalCap):
        return region.contains(lon_deg, lat_deg, inclusive=inclusive)
    if isinstance(region, SphericalBooleanRegion):
        return region.contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)
    return region.contains(lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg)


def _region_pixelize(region: RegionOperand, nside: int, nest: bool) -> PixelizedRegion:
    return region.pixelize(nside=nside, nest=nest)


@dataclass(frozen=True)
class SphericalBooleanRegion:
    """Boolean composition of spherical regions with controlled-error HEALPix evaluation."""

    op: Literal["union", "intersection", "difference"]
    left: RegionOperand
    right: RegionOperand

    def __post_init__(self) -> None:
        if self.op not in {"union", "intersection", "difference"}:
            raise ValueError(
                "op must be one of {'union', 'intersection', 'difference'}"
            )

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        left = _region_contains(
            self.left, lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg
        )
        right = _region_contains(
            self.right, lon_deg, lat_deg, inclusive=inclusive, atol_deg=atol_deg
        )
        if self.op == "union":
            return left | right
        if self.op == "intersection":
            return left & right
        return left & (~right)

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        left = _region_pixelize(self.left, nside, nest)
        right = _region_pixelize(self.right, nside, nest)
        if self.op == "union":
            return left.union(right).pixels
        if self.op == "intersection":
            return left.intersection(right).pixels
        return left.difference(right).pixels

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def area_estimate(
        self,
        *,
        nsides: tuple[int, ...] | list[int] = (128, 256, 512, 1024),
        nest: bool = False,
    ) -> RegionAreaEstimate:
        return _estimate_area_from_query(self.query_pixels, nsides=nsides, nest=nest)

    def union(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="union", left=self, right=other)

    def intersection(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="intersection", left=self, right=other)

    def difference(self, other: RegionOperand) -> "SphericalBooleanRegion":
        return SphericalBooleanRegion(op="difference", left=self, right=other)

    def to_exact(
        self,
        *,
        cap_steps: int = 256,
        backend: Literal["spherical-geometry", "native"] = "native",
    ) -> "ExactSphericalRegion | NativeApproximateSphericalRegion":
        return to_exact_region(self, cap_steps=cap_steps, backend=backend)

    def exact_area(self, *, degrees: bool = False, cap_steps: int = 256) -> float:
        return self.to_exact(cap_steps=cap_steps).area(degrees=degrees)


@dataclass(frozen=True)
class NativeApproximateSphericalRegion:
    """
    Torch-native region wrapper with explicit center-sampled approximations.

    Analytic area and emptiness are used where available. Unsupported exact
    predicates fail closed; ``area_estimate`` is the explicit HEALPix-center
    approximation. ``spherical-geometry`` remains the opt-in exact backend.
    """

    _region: RegionOperand
    _nsides: tuple[int, ...] = (256, 512, 1024, 2048)

    @property
    def is_empty(self) -> bool:
        area = self._analytic_area_sr()
        if area > 0.0:
            return False
        raise RuntimeError("native emptiness is unavailable for zero-area regions")

    def _analytic_area_sr(self) -> float:
        region = self._region
        if isinstance(region, SphericalCap):
            radius = math.radians(region.radius_deg)
            return 4.0 * math.pi * math.sin(0.5 * radius) ** 2
        if isinstance(region, SphericalPolygon):
            return float(region.area(degrees=False))
        if isinstance(region, SphericalBox):
            if region.lon_max - region.lon_min >= 360.0:
                width = 2.0 * math.pi
            else:
                width = math.radians((region.lon_max - region.lon_min) % 360.0)
            return width * (
                math.sin(math.radians(region.lat_max))
                - math.sin(math.radians(region.lat_min))
            )
        raise RuntimeError(
            "analytic native area is unavailable; use area_estimate() or "
            "backend='spherical-geometry'"
        )

    def area(self, *, degrees: bool = False) -> float:
        area_sr = self._analytic_area_sr()
        if area_sr <= 0.0:
            raise RuntimeError("native area is indeterminate for zero-area regions")
        if degrees:
            return area_sr * ((180.0 / math.pi) * (180.0 / math.pi))
        return area_sr

    def area_estimate(self) -> RegionAreaEstimate:
        return _estimate_area_from_query(
            self.query_pixels, nsides=self._nsides, nest=False
        )

    def intersects(self, other: RegionOperand) -> bool:
        other_region = (
            other._region
            if isinstance(other, NativeApproximateSphericalRegion)
            else other
        )
        if isinstance(self._region, SphericalCap) and isinstance(
            other_region, SphericalCap
        ):
            distance = float(
                great_circle_distance(
                    self._region.lon_deg,
                    self._region.lat_deg,
                    other_region.lon_deg,
                    other_region.lat_deg,
                    degrees=True,
                )
            )
            return distance <= self._region.radius_deg + other_region.radius_deg
        if isinstance(self._region, SphericalPolygon) and isinstance(
            other_region, SphericalPolygon
        ):
            return self._region.intersects(other_region)
        if isinstance(self._region, SphericalMultiPolygon) and isinstance(
            other_region, (SphericalPolygon, SphericalMultiPolygon)
        ):
            return self._region.intersects(other_region)
        if isinstance(self._region, SphericalPolygon) and isinstance(
            other_region, SphericalMultiPolygon
        ):
            return other_region.intersects(self._region)
        raise RuntimeError(
            "native intersection is unavailable for these region types; "
            "use backend='spherical-geometry'"
        )

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
    ) -> Tensor:
        return _region_contains(self._region, lon_deg, lat_deg)

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        return _region_pixelize(self._region, nside=nside, nest=nest).pixels

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def union(self, other: RegionOperand) -> "NativeApproximateSphericalRegion":
        return NativeApproximateSphericalRegion(
            SphericalBooleanRegion(op="union", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def intersection(self, other: RegionOperand) -> "NativeApproximateSphericalRegion":
        return NativeApproximateSphericalRegion(
            SphericalBooleanRegion(op="intersection", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def difference(self, other: RegionOperand) -> "NativeApproximateSphericalRegion":
        return NativeApproximateSphericalRegion(
            SphericalBooleanRegion(op="difference", left=self._region, right=other),
            _nsides=self._nsides,
        )

    def to_multipolygon(self) -> SphericalMultiPolygon:
        if isinstance(self._region, SphericalPolygon):
            return SphericalMultiPolygon((self._region,))
        if isinstance(self._region, SphericalMultiPolygon):
            return self._region
        raise ValueError(
            "native approximate region cannot be converted to multipolygon for this operand type"
        )


@dataclass(frozen=True)
class ExactSphericalRegion:
    """Exact spherical region wrapper backed by optional `spherical-geometry`."""

    _sg_poly: object

    @property
    def is_empty(self) -> bool:
        return len(list(self._sg_poly)) == 0

    def area(self, *, degrees: bool = False) -> float:
        area_sr = float(self._sg_poly.area())
        if degrees:
            return area_sr * ((180.0 / math.pi) * (180.0 / math.pi))
        return area_sr

    def intersects(self, other: RegionOperand, *, cap_steps: int = 256) -> bool:
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return bool(self._sg_poly.intersects_poly(other_sg))

    def contains(
        self,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
        *,
        inclusive: bool = True,
        atol_deg: float = 1e-10,
    ) -> Tensor:
        del inclusive, atol_deg
        lon_t = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat_t = torch.as_tensor(lat_deg, dtype=torch.float64)
        lon_t, lat_t = torch.broadcast_tensors(lon_t, lat_t)
        shape = lon_t.shape
        lon_f = lon_t.reshape(-1).detach().cpu().numpy()
        lat_f = lat_t.reshape(-1).detach().cpu().numpy()
        out = _numpy_sg_contains_lonlat_bulk(self._sg_poly, lon_f, lat_f)
        return torch.from_numpy(out.reshape(shape))

    def query_pixels(self, nside: int, *, nest: bool = False) -> Tensor:
        # Candidate pixels from per-component polygon queries, then exact center filtering.
        import numpy as np

        chunks: list[Tensor] = []
        for lon, lat in self._sg_poly.to_lonlat():
            lon_np = np.asarray(lon, dtype=np.float64)
            lat_np = np.asarray(lat, dtype=np.float64)
            lon_np, lat_np = _strip_closed_np(lon_np, lat_np)
            if lon_np.size < 3:
                continue
            pix = query_polygon_general(
                nside,
                torch.from_numpy(lon_np),
                torch.from_numpy(lat_np),
                nest=nest,
            )
            if pix.numel() > 0:
                chunks.append(pix)
        if not chunks:
            return torch.empty(0, dtype=torch.int64)
        candidate = torch.unique(torch.cat(chunks))
        lon_c, lat_c = _healpix.pix2ang(nside, candidate, nest=nest, lonlat=True)
        keep = self.contains(lon_c, lat_c)
        return candidate[keep]

    def pixelize(self, nside: int, *, nest: bool = False) -> PixelizedRegion:
        return PixelizedRegion(
            nside=nside, nest=nest, pixels=self.query_pixels(nside, nest=nest)
        )

    def union(
        self, other: RegionOperand, *, cap_steps: int = 256
    ) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(self._sg_poly.union(other_sg))

    def intersection(
        self, other: RegionOperand, *, cap_steps: int = 256
    ) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(self._sg_poly.intersection(other_sg))

    def difference(
        self, other: RegionOperand, *, cap_steps: int = 256
    ) -> "ExactSphericalRegion":
        other_sg = _region_to_sg(other, cap_steps=cap_steps)
        return ExactSphericalRegion(
            self._sg_poly.intersection(other_sg.invert_polygon())
        )

    def to_multipolygon(self) -> SphericalMultiPolygon:
        import numpy as np

        polys: list[SphericalPolygon] = []
        for lon, lat in self._sg_poly.to_lonlat():
            lon_np = np.asarray(lon, dtype=np.float64)
            lat_np = np.asarray(lat, dtype=np.float64)
            lon_np, lat_np = _strip_closed_np(lon_np, lat_np)
            if lon_np.size >= 3:
                polys.append(
                    SphericalPolygon(torch.from_numpy(lon_np), torch.from_numpy(lat_np))
                )
        if not polys:
            raise ValueError(
                "exact region is empty; cannot convert to SphericalMultiPolygon"
            )
        return SphericalMultiPolygon(tuple(polys))


def to_exact_region(
    region: RegionOperand,
    *,
    cap_steps: int = 256,
    backend: Literal["spherical-geometry", "native"] = "native",
    nsides: tuple[int, ...] | list[int] = (256, 512, 1024, 2048),
) -> ExactSphericalRegion | NativeApproximateSphericalRegion:
    """
    Convert a torchsky spherical region into a selected region backend.

    Default ``native`` is torchsky-owned and exposes analytic results only when
    available, with explicitly named center-sampled estimates otherwise.
    ``spherical-geometry`` is the opt-in exact backend.
    """
    if backend not in {"spherical-geometry", "native"}:
        raise ValueError("backend must be one of {'spherical-geometry', 'native'}")
    if backend == "native":
        return NativeApproximateSphericalRegion(region, tuple(nsides))
    return ExactSphericalRegion(_region_to_sg(region, cap_steps=cap_steps))


def query_ellipse(
    nside: int,
    lon_deg: float | Tensor,
    lat_deg: float | Tensor,
    semi_major_deg: float,
    semi_minor_deg: float,
    *,
    pa_deg: float = 0.0,
    nest: bool = False,
    inclusive: bool = False,
) -> Tensor:
    """
    Query pixels inside a sky ellipse.

    The ellipse is defined in geodesic polar coordinates around the center.
    `pa_deg` is position angle East of North.
    """
    _validate_ellipse_axes(semi_major_deg, semi_minor_deg)

    # Spherical ellipse in focal form:
    # d(P, F1) + d(P, F2) <= 2a, where a is the semi-major axis.
    major = float(math.radians(semi_major_deg))
    minor = float(math.radians(semi_minor_deg))
    cos_ratio = math.cos(major) / max(math.cos(minor), 1e-15)
    cos_ratio = min(1.0, max(-1.0, cos_ratio))
    focal_sep = math.acos(cos_ratio)
    pa = math.radians(pa_deg)

    lon_f1, lat_f1 = _destination_lonlat_deg(lon_deg, lat_deg, pa, focal_sep)
    lon_f2, lat_f2 = _destination_lonlat_deg(lon_deg, lat_deg, pa + math.pi, focal_sep)

    pixrad = _healpix.max_pixel_radius(nside, degrees=False) if inclusive else 0.0
    candidates = _healpix.query_circle(
        nside,
        lon_deg,
        lat_deg,
        semi_major_deg + math.degrees(pixrad),
        degrees=True,
        nest=nest,
        inclusive=False,
    )
    if candidates.numel() == 0:
        return candidates

    lon_p, lat_p = _healpix.pix2ang(nside, candidates, nest=nest, lonlat=True)
    d1 = great_circle_distance(lon_f1, lat_f1, lon_p, lat_p, degrees=False)
    d2 = great_circle_distance(lon_f2, lat_f2, lon_p, lat_p, degrees=False)

    # For inclusive overlap, any point in the pixel can lie up to pixrad from center.
    thresh = (2.0 * major) + (2.0 * pixrad) + 1e-12
    keep = (d1 + d2) <= thresh
    return candidates[keep]


__all__ = [
    "ExactSphericalRegion",
    "LonLat",
    "NativeApproximateSphericalRegion",
    "PixelizedRegion",
    "RegionAreaEstimate",
    "SphericalBooleanRegion",
    "SphericalBox",
    "SphericalCap",
    "SphericalEllipse",
    "SphericalMultiPolygon",
    "SphericalPolygon",
    "UnitVector3d",
    "Vector3d",
    "convex_polygon_contains",
    "query_ellipse",
    "query_polygon_general",
    "spherical_polygon_area",
    "spherical_polygon_contains",
    "spherical_polygons_intersect",
    "spherical_polygon_signed_area",
    "spherical_triangle_area",
    "to_exact_region",
]
