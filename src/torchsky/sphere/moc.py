"""Multi-Order Coverage (MOC) support for HEALPix."""

import json
import math
from typing import Any, Literal

import torch
from torch import Tensor

from . import healpix as _healpix
from .core import great_circle_distance, lonlat_to_unit_xyz
from .geom import PixelizedRegion, spherical_polygon_contains
from .sparse import HealSparseMap


def _order_from_nside(nside: int, *, name: str = "nside") -> int:
    nside_i = int(nside)
    if nside_i <= 0 or (nside_i & (nside_i - 1)) != 0:
        raise ValueError(f"{name} must be a positive power of two")
    return nside_i.bit_length() - 1


_MAX_HEALPIX_ORDER = 29
_MAX_HEALPIX_NSIDE = 1 << _MAX_HEALPIX_ORDER


def _nside_from_max_order(max_order: int) -> int:
    order = int(max_order)
    if order < 0:
        raise ValueError("max_order must be non-negative")
    if order > _MAX_HEALPIX_ORDER:
        raise ValueError(
            f"max_order must be <= {_MAX_HEALPIX_ORDER}; got {order}"
        )
    return 1 << order


def _order_from_uniq(uniq: Tensor) -> Tensor:
    """Return the HEALPix order of each UNIQ value with integer arithmetic.

    UNIQ encodes the order as ``4 * nside**2 + nested_pixel`` with
    ``nside = 2**order``.  A float ``log2`` loses the low bits of large
    int64 values and misclassifies the last cells of high orders, so the
    order is derived from an exact integer bit length.
    """
    u = uniq.to(torch.int64)
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
        raise ValueError(
            f"UNIQ values must encode nside <= {_MAX_HEALPIX_NSIDE}"
        )
    return order


def nest2uniq(nside: int | Tensor, pix: int | Tensor) -> Tensor:
    """Convert NESTED pixel index to UNIQ pixel index."""
    n = torch.as_tensor(nside, dtype=torch.int64)
    p = torch.as_tensor(pix, dtype=torch.int64)
    if bool((n <= 0).any()) or bool((n > _MAX_HEALPIX_NSIDE).any()):
        raise ValueError(
            f"nside must be in [1, {_MAX_HEALPIX_NSIDE}]"
        )
    if bool((n & (n - 1) != 0).any()):
        raise ValueError("nside must be a positive power of two")
    return 4 * (n * n) + p


def uniq2nest(uniq: int | Tensor) -> tuple[Tensor, Tensor]:
    """Convert UNIQ pixel index to (nside, NESTED pixel index)."""
    u = torch.as_tensor(uniq, dtype=torch.int64)
    order = _order_from_uniq(u)
    nside = torch.bitwise_left_shift(torch.ones_like(order), order)
    pix = u - 4 * (nside * nside)
    return nside, pix


def _max_order_from_uniq(uniq: Tensor) -> int:
    if uniq.numel() == 0:
        return 0
    return int(_order_from_uniq(uniq.max()).item())


def _uniq_to_ranges(uniq: Tensor, max_order: int) -> list[tuple[int, int]]:
    if uniq.numel() == 0:
        return []

    _, pix_nested = uniq2nest(uniq)
    orders = _order_from_uniq(uniq)
    up_shifts = (2 * (max_order - orders)).clamp_min(0)
    down_shifts = (2 * (orders - max_order)).clamp_min(0)
    pix_at_order = pix_nested >> down_shifts
    starts = pix_at_order << up_shifts
    ends = (pix_at_order + 1) << up_shifts

    sort_idx = torch.argsort(starts)
    starts = starts[sort_idx]
    ends = ends[sort_idx]

    merged: list[tuple[int, int]] = []
    cur_start = int(starts[0].item())
    cur_end = int(ends[0].item())
    for start_t, end_t in zip(starts[1:], ends[1:], strict=False):
        start = int(start_t.item())
        end = int(end_t.item())
        if start <= cur_end:
            cur_end = max(cur_end, end)
            continue
        merged.append((cur_start, cur_end))
        cur_start, cur_end = start, end
    merged.append((cur_start, cur_end))
    return merged


def _union_ranges(
    left: list[tuple[int, int]], right: list[tuple[int, int]]
) -> list[tuple[int, int]]:
    """Merge two sorted half-open range lists."""
    i = j = 0
    out: list[tuple[int, int]] = []
    while i < len(left) or j < len(right):
        if j >= len(right) or (i < len(left) and left[i][0] <= right[j][0]):
            start, end = left[i]
            i += 1
        else:
            start, end = right[j]
            j += 1
        if not out or start > out[-1][1]:
            out.append((start, end))
            continue
        out[-1] = (out[-1][0], max(out[-1][1], end))
    return out


def _ranges_to_uniq(ranges: list[tuple[int, int]], max_order: int) -> Tensor:
    uniqs: list[Tensor] = []
    for start, end in ranges:
        curr = int(start)
        while curr < end:
            k = 0
            while (
                curr + (4 ** (k + 1)) <= end
                and curr % (4 ** (k + 1)) == 0
                and k < max_order
            ):
                k += 1
            order = max_order - k
            uniqs.append(
                nest2uniq(1 << order, curr >> (2 * k)).reshape(1).to(torch.int64)
            )
            curr += 4**k
    if not uniqs:
        return torch.empty((0,), dtype=torch.int64)
    return torch.cat(uniqs)


def _normalize_moc(uniq: Tensor) -> Tensor:
    """Sort and merge redundant pixels in a MOC."""
    if uniq.numel() == 0:
        return uniq

    u = torch.unique(torch.as_tensor(uniq, dtype=torch.int64))
    max_order = _max_order_from_uniq(u)
    ranges = _uniq_to_ranges(u, max_order)
    return _ranges_to_uniq(ranges, max_order)



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
    r_leaf = _healpix.max_pixel_radius(1 << max_order, degrees=True)
    while True:
        nside = 1 << order
        lon_p, lat_p = _healpix.pix2ang(nside, pix, nest=True, lonlat=True)
        dist = great_circle_distance(lon0, lat0, lon_p, lat_p, degrees=True)
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
        great_circle_distance(
            cen_lon, cen_lat, poly_lon, poly_lat, degrees=True
        ).max().item()
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



def _resolve_moc_table_hdu_index(path: str) -> int:
    """Return 0-based HDU index of the IVOA MOC binary table extension."""
    import torchfits.cpp as cpp

    if hasattr(cpp, "resolve_hdu_name_cached"):
        try:
            return int(cpp.resolve_hdu_name_cached(path, "MOC"))
        except Exception:
            pass
    i = 0
    while i < 512:
        try:
            h_list = cpp.read_header_dict(path, i)
        except Exception:
            break
        if not h_list:
            break
        ext = h_list.get("EXTNAME")
        if ext == "MOC" or (isinstance(ext, str) and ext.strip() == "MOC"):
            return i
        i += 1
    raise ValueError(f"No FITS extension with EXTNAME 'MOC' in {path!r}")


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


class _EmptyFlag(int):
    def __new__(cls, is_empty: bool, factory: Any) -> "_EmptyFlag":
        obj = super().__new__(cls, int(bool(is_empty)))
        obj._is_empty = bool(is_empty)
        obj._factory = factory
        return obj

    def __bool__(self) -> bool:
        return self._is_empty

    def __call__(self) -> Any:
        return self._factory()


class _EmptyDescriptor:
    def __get__(self, instance: Any, owner: type) -> Any:
        if instance is None:
            return owner._make_empty
        return _EmptyFlag(int(instance.uniq.numel()) == 0, owner._make_empty)


class MOC:
    """Multi-Order Coverage map."""

    def __init__(self, uniq: Tensor):
        self.uniq = _normalize_moc(torch.as_tensor(uniq, dtype=torch.int64))

    @classmethod
    def _make_empty(cls) -> "MOC":
        return cls(torch.empty(0, dtype=torch.int64))

    empty: Any = _EmptyDescriptor()

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
    def full(cls, max_order: int = 0) -> "MOC":
        del max_order
        return cls(nest2uniq(1, torch.arange(12, dtype=torch.int64)))

    @property
    def max_order(self) -> int:
        if self.uniq.numel() == 0:
            return 0
        return int(_order_from_uniq(self.uniq.max()).item())

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        """Check if points are contained in the MOC."""
        lon = _healpix._as_float64(lon_deg)
        lat = _healpix._as_float64(lat_deg)
        lat = lat.to(device=lon.device)

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

        ranges = self._to_ranges(order)
        if not ranges:
            return torch.zeros(shape, dtype=torch.bool, device=pix.device)
        starts = torch.tensor(
            [r[0] for r in ranges], device=pix.device, dtype=torch.int64
        )
        ends = torch.tensor(
            [r[1] for r in ranges], device=pix.device, dtype=torch.int64
        )

        idx = torch.searchsorted(starts, pix_nest, right=True) - 1
        in_range = idx >= 0
        idx_safe = idx.clamp(min=0)
        mask = in_range & (pix_nest < ends[idx_safe])
        return mask.reshape(shape)

    def filter_catalog(self, lon_deg: Tensor, lat_deg: Tensor) -> tuple[Tensor, Tensor]:
        """Filter a catalogue by point-in-MOC containment. Returns (lon, lat) of contained points."""
        mask = self.contains(lon_deg, lat_deg)
        return lon_deg[mask], lat_deg[mask]

    @property
    def area(self) -> float:
        """Total area covered by the MOC in steradians."""
        if self.uniq.numel() == 0:
            return 0.0
        nsides, _ = uniq2nest(self.uniq)
        # Area of a pixel at nside is 4pi / (12 * nside^2)
        # Performance: avoid **2 on tensors
        nsides_f64 = nsides.to(torch.float64)
        pix_areas = (4.0 * math.pi) / (12.0 * (nsides_f64 * nsides_f64))
        return float(pix_areas.sum().item())

    @property
    def area_sq_deg(self) -> float:
        """Total area covered by the MOC in square degrees."""
        return self.area * ((180.0 / math.pi) * (180.0 / math.pi))

    def contains_moc(self, other: "MOC") -> bool:
        """Return True if this MOC contains all of other MOC (subset check)."""
        # self contains other if intersection(self, other) == other
        inter = self.intersection(other)
        return torch.equal(inter.uniq, other.uniq)

    def union(self, other: "MOC") -> "MOC":
        if self.uniq.numel() == 0:
            return MOC(other.uniq)
        if other.uniq.numel() == 0:
            return MOC(self.uniq)
        max_o = max(self.max_order, other.max_order)
        return MOC._from_ranges(
            _union_ranges(self._to_ranges(max_o), other._to_ranges(max_o)),
            max_o,
        )

    def intersection(self, other: "MOC") -> "MOC":
        # Hierarchical intersection:
        # A pixel is in intersection if:
        # 1. It is in both.
        # 2. It is in A and a descendant is in B (then the descendant is in intersection).
        # 3. It is in B and a descendant is in A (then the descendant is in intersection).

        # Implementation via max-order ranges:
        # Intersection of two sets of ranges.
        max_o = max(self.max_order, other.max_order)
        r1 = self._to_ranges(max_o)
        r2 = other._to_ranges(max_o)

        # Range intersection
        # This is a classic algorithm.
        # For simplicity, we can do it via a loop or vectorized if ranges are few.
        i, j = 0, 0
        new_ranges = []
        while i < len(r1) and j < len(r2):
            s1, e1 = r1[i]
            s2, e2 = r2[j]

            s_int = max(s1, s2)
            e_int = min(e1, e2)

            if s_int < e_int:
                new_ranges.append((s_int, e_int))

            if e1 < e2:
                i += 1
            else:
                j += 1
        return MOC._from_ranges(new_ranges, max_o)

    def difference(self, other: "MOC") -> "MOC":
        """Subtract other MOC from this MOC."""
        max_o = max(self.max_order, other.max_order)
        r1 = self._to_ranges(max_o)
        r2 = other._to_ranges(max_o)

        # Range difference: R1 \ R2
        new_ranges = []
        i, j = 0, 0
        curr_s, curr_e = r1[0] if r1 else (0, 0)

        while i < len(r1):
            if j >= len(r2):
                new_ranges.append((curr_s, curr_e))
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]
                continue

            s2, e2 = r2[j]

            if e2 <= curr_s:
                j += 1
                continue
            if s2 >= curr_e:
                new_ranges.append((curr_s, curr_e))
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]
                continue

            # Overlap!
            if s2 > curr_s:
                new_ranges.append((curr_s, s2))

            if e2 < curr_e:
                curr_s = e2
                j += 1
            else:
                i += 1
                if i < len(r1):
                    curr_s, curr_e = r1[i]

        return MOC._from_ranges(new_ranges, max_o)

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
        nsides, pix = uniq2nest(self.uniq)
        cell_orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)
        delta = (cell_orders - order).clamp(min=0)
        pix_d = pix >> (2 * delta)
        nside_d = torch.where(delta > 0, torch.full_like(nsides, 1 << order), nsides)
        return type(self)(nest2uniq(nside_d, pix_d))

    def _to_ranges(self, max_order: int) -> list[tuple[int, int]]:
        return _uniq_to_ranges(self.uniq, max_order)

    @classmethod
    def _from_ranges(cls, ranges: list[tuple[int, int]], max_order: int) -> "MOC":
        """Convert merged ranges to UNIQ cells without a second normalize pass."""
        moc = cls.__new__(cls)
        moc.uniq = _ranges_to_uniq(ranges, max_order)
        return moc

    def _flat_pixels(self, nside: int) -> Tensor:
        max_o = _order_from_nside(nside)
        if max_o < self.max_order:
            raise ValueError("nside must be at least the MOC maximum order resolution")
        ranges = self._to_ranges(max_o)
        if not ranges:
            return torch.empty((0,), dtype=torch.int64, device=self.uniq.device)
        chunks = [
            torch.arange(start, end, dtype=torch.int64, device=self.uniq.device)
            for start, end in ranges
        ]
        return torch.cat(chunks)

    def to_sparse_map(
        self,
        nside: int,
        sentinel: float = 0.0,
        *,
        nside_coverage: int | None = None,
        dtype: torch.dtype = torch.float32,
    ) -> HealSparseMap:
        """Realize MOC coverage as a sparse HEALPix map at fixed resolution."""
        nside_i = int(nside)
        if nside_coverage is None:
            nside_coverage = max(1, nside_i // 8)
        return HealSparseMap.from_moc(
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
    def from_sparse_map(cls, smap: HealSparseMap) -> "MOC":
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
        # Find contiguous chunks
        diffs = torch.diff(pix)
        breaks = torch.where(diffs > 1)[0]

        ranges = []
        start_idx = 0
        for b in breaks:
            ranges.append((int(pix[start_idx].item()), int(pix[b].item()) + 1))
            start_idx = int(b.item()) + 1
        ranges.append((int(pix[start_idx].item()), int(pix[-1].item()) + 1))

        return cls._from_ranges(ranges, max_order)

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
        pix = _healpix.ang2pix(
            nside, lon_deg, lat_deg, nest=True, lonlat=True
        ).to(torch.int64)
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
    def from_region(cls, region: object, max_order: int | None = None) -> "MOC":
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
        smap: HealSparseMap,
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

        hdu = _resolve_moc_table_hdu_index(filename)
        table = read_table(filename, hdu=hdu)
        uniqs = _uniq_column_from_table_dict(table)
        return cls(uniqs)

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
            vals = torch.round(v.to(torch.float64).cpu() * quant_scale).to(torch.int64)
            return int(vals[0].item()), int(vals[1].item()), int(vals[2].item())

        vertex_xyz: dict[tuple[int, int, int], Tensor] = {}
        undirected_edges: dict[
            tuple[tuple[int, int, int], tuple[int, int, int]],
            tuple[tuple[int, int, int], tuple[int, int, int]],
        ] = {}
        for poly in corners_xyz:
            keys = [_key(poly[i]) for i in range(poly.shape[0])]
            for key, xyz in zip(keys, poly, strict=False):
                vertex_xyz[key] = xyz.to(torch.float64).cpu()
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
        orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)

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
        orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)

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


class BMOC:
    """Boolean Multi-Order Coverage map storing UNIQ cells and interior/boundary flags.

    ``flag == 1`` indicates a cell completely inside the region; ``flag == 0``
    indicates a partial/boundary cell at ``max_order``.
    """

    def __init__(self, uniq: Tensor, flag: Tensor | None = None):
        u = torch.as_tensor(uniq, dtype=torch.int64).reshape(-1)
        if flag is None:
            f = torch.ones_like(u, dtype=torch.uint8)
        else:
            f = torch.as_tensor(flag, dtype=torch.uint8, device=u.device).reshape(-1)
            if f.numel() != u.numel():
                raise ValueError("flag must have the same length as uniq")
        if u.numel() > 0:
            order = torch.argsort(u, stable=True)
            u = u[order]
            f = f[order]
            keep = torch.cat(
                [
                    u[1:] != u[:-1],
                    torch.tensor([True], dtype=torch.bool, device=u.device),
                ]
            )
            u = u[keep]
            f = f[keep]
        self.uniq = u
        self.flag = f

    @classmethod
    def _make_empty(cls) -> "BMOC":
        return cls(
            torch.empty(0, dtype=torch.int64),
            torch.empty(0, dtype=torch.uint8),
        )

    empty: Any = _EmptyDescriptor()

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
        nsides_f64 = nsides.to(torch.float64)
        return float(((4.0 * math.pi) / (12.0 * (nsides_f64 * nsides_f64))).sum().item())

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
        nsides_f64 = nsides.to(torch.float64)
        return float(((4.0 * math.pi) / (12.0 * (nsides_f64 * nsides_f64))).sum().item())

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


def _sample_in_nested_cells(
    nsides: Tensor | int,
    pix_nested: Tensor,
    u_x: Tensor,
    u_y: Tensor,
) -> tuple[Tensor, Tensor]:
    """Map uniform (u_x, u_y) in [0, 1)^2 inside NESTED HEALPix cells to (lon_deg, lat_deg).

    Because the HEALPix curvilinear projection has constant Jacobian
    |d(z, phi)/d(x, y)| = pi / 3 everywhere on all 12 base faces (both polar
    caps and equatorial belt), uniform sampling in face coordinates (x, y) is
    mathematically exact uniform solid-angle sampling on S^2 with zero
    rejection.
    """
    pix_t = torch.as_tensor(pix_nested, dtype=torch.int64)
    device = pix_t.device
    f_dtype = _healpix._float_dtype_for_device(device)
    ns_t = torch.as_tensor(nsides, dtype=torch.int64, device=device)
    npface = ns_t * ns_t
    face_num = torch.div(pix_t, npface, rounding_mode="floor")
    ipf = pix_t - face_num * npface
    ix = _healpix.compact_bits(ipf).to(f_dtype)
    iy = _healpix.compact_bits(ipf >> 1).to(f_dtype)
    ns_f = ns_t.to(f_dtype)
    # Clamp u_x, u_y slightly below 1.0 to stay strictly inside the cell
    eps = 1e-7 if f_dtype == torch.float32 else 1e-15
    ux = u_x.to(device=device, dtype=f_dtype).clamp(min=0.0, max=1.0 - eps)
    uy = u_y.to(device=device, dtype=f_dtype).clamp(min=0.0, max=1.0 - eps)
    x = (ix + ux) / ns_f
    y = (iy + uy) / ns_f

    jrll, jpll = _healpix._face_consts(device)
    jrll_f = jrll[face_num].to(f_dtype)
    jpll_f = jpll[face_num].to(f_dtype)

    jr = jrll_f - x - y
    nr = torch.empty_like(jr)
    z = torch.empty_like(jr)

    north = jr < 1.0
    south = jr > 3.0
    equat = ~(north | south)

    nr[north] = jr[north]
    z[north] = 1.0 - torch.square(nr[north]) / 3.0
    nr[south] = 4.0 - jr[south]
    z[south] = -1.0 + torch.square(nr[south]) / 3.0
    nr[equat] = 1.0
    z[equat] = (2.0 - jr[equat]) * (2.0 / 3.0)

    nr_safe = torch.where(nr == 0.0, torch.ones_like(nr), nr)
    phi = (jpll_f * nr + x - y) * (math.pi * 0.25) / nr_safe
    phi = torch.where(nr == 0.0, torch.zeros_like(phi), phi)

    lon_deg = torch.remainder(torch.rad2deg(phi), 360.0)
    lat_deg = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    return lon_deg, lat_deg


def sample_randoms(
    n: int,
    footprint: MOC | BMOC | HealSparseMap | None = None,
    *,
    weights: HealSparseMap | Tensor | str | None = None,
    generator: torch.Generator | None = None,
    device: torch.device | str | None = None,
) -> tuple[Tensor, Tensor]:
    """Generate ``n`` random sky coordinates ``(ra_deg, dec_deg)`` uniformly in solid angle.

    Supports rejection-free sampling inside a :class:`MOC`, :class:`BMOC`, or
    :class:`HealSparseMap`, with optional per-pixel depth/completeness weights.
    """
    n_i = int(n)
    if n_i < 0:
        raise ValueError("n must be non-negative")
    if footprint is None and isinstance(weights, HealSparseMap):
        footprint = weights
        weights = "values"

    if footprint is None:
        dev = torch.device("cpu") if device is None else torch.device(device)
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        u_lon = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        u_z = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        lon_deg = u_lon * 360.0
        z = 2.0 * u_z - 1.0
        lat_deg = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
        return lon_deg, lat_deg

    if isinstance(footprint, BMOC):
        footprint = footprint.to_moc(include_boundary=True)

    if isinstance(footprint, MOC):
        dev = (
            footprint.uniq.device
            if device is None
            else torch.device(device)
        )
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        if footprint.uniq.numel() == 0:
            raise ValueError("cannot sample randoms from an empty MOC")
        uniq = footprint.uniq.to(device=dev)
        nsides, pix_nested = uniq2nest(uniq)
        ns_f = nsides.to(f_dtype)
        cell_weights = 1.0 / (ns_f * ns_f)
        cdf = torch.cumsum(cell_weights, dim=0)
        u_cell = (
            torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator) * cdf[-1]
        )
        idx = torch.searchsorted(cdf, u_cell, right=False).clamp(
            max=uniq.numel() - 1
        )
        ux = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        uy = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        return _sample_in_nested_cells(nsides[idx], pix_nested[idx], ux, uy)

    if isinstance(footprint, HealSparseMap):
        dev = (
            footprint.pixels.device
            if device is None
            else torch.device(device)
        )
        f_dtype = _healpix._float_dtype_for_device(dev)
        if n_i == 0:
            empty = torch.empty(0, dtype=f_dtype, device=dev)
            return empty, empty
        pix = footprint.pixels.to(device=dev)
        if pix.numel() == 0:
            raise ValueError("cannot sample randoms from an empty HealSparseMap")
        if not footprint.nest:
            pix_nest = _healpix.ring2nest(footprint.nside, pix)
        else:
            pix_nest = pix

        if weights is None:
            idx = torch.randint(
                0, pix_nest.numel(), (n_i,), device=dev, generator=generator
            )
        else:
            if isinstance(weights, str):
                if weights == "values":
                    w_raw = footprint.values
                elif footprint.column_names and weights in footprint.column_names:
                    w_raw = footprint[weights].values
                else:
                    raise ValueError(f"unknown weights specification: {weights!r}")
            elif isinstance(weights, HealSparseMap):
                w_raw = weights.get_values(footprint.pixels)
            else:
                w_raw = torch.as_tensor(weights, device=dev)
            w = w_raw.to(device=dev, dtype=f_dtype).reshape(-1)
            if w.numel() != pix_nest.numel():
                raise ValueError(
                    "weights must have one entry per valid pixel in footprint"
                )
            w = torch.where(torch.isfinite(w) & (w > 0), w, torch.zeros_like(w))
            total_w = float(w.sum().item())
            if total_w <= 0.0:
                raise ValueError("weights must have positive sum")
            cdf = torch.cumsum(w, dim=0)
            u_cell = (
                torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
                * cdf[-1]
            )
            idx = torch.searchsorted(cdf, u_cell, right=False).clamp(
                max=pix_nest.numel() - 1
            )
        ux = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        uy = torch.rand(n_i, dtype=f_dtype, device=dev, generator=generator)
        return _sample_in_nested_cells(footprint.nside, pix_nest[idx], ux, uy)

    raise TypeError(
        "footprint must be a MOC, BMOC, HealSparseMap, or None"
    )


__all__ = ["BMOC", "MOC", "nest2uniq", "sample_randoms", "uniq2nest"]
