"""Multi-Order Coverage (MOC) support for HEALPix."""

import json
import math
from typing import Literal

import torch
from torch import Tensor

from . import healpix as _healpix
from .sparse import HealSparseMap


def nest2uniq(nside: int | Tensor, pix: int | Tensor) -> Tensor:
    """Convert NESTED pixel index to UNIQ pixel index."""
    n = torch.as_tensor(nside, dtype=torch.int64)
    p = torch.as_tensor(pix, dtype=torch.int64)
    return 4 * (n * n) + p


def uniq2nest(uniq: int | Tensor) -> tuple[Tensor, Tensor]:
    """Convert UNIQ pixel index to (nside, NESTED pixel index)."""
    u = torch.as_tensor(uniq, dtype=torch.int64)
    # 4 * 4^order + pix = uniq
    # 4^order = uniq // 4 (approx)
    # order = log4(uniq // 4)
    order = torch.div(
        torch.log2(torch.div(u, 4, rounding_mode="floor").to(torch.float64)),
        2.0,
        rounding_mode="floor",
    ).to(torch.int64)
    nside = 1 << order
    pix = u - 4 * (nside * nside)
    return nside, pix


def _max_order_from_uniq(uniq: Tensor) -> int:
    if uniq.numel() == 0:
        return 0
    nside, _ = uniq2nest(uniq.max())
    return int(torch.log2(nside.to(torch.float64)).item())


def _uniq_to_ranges(uniq: Tensor, max_order: int) -> list[tuple[int, int]]:
    if uniq.numel() == 0:
        return []

    nsides, pix_nested = uniq2nest(uniq)
    orders = torch.log2(nsides.to(torch.float64)).to(torch.int64)
    shifts = 2 * (max_order - orders)
    starts = pix_nested << shifts
    ends = (pix_nested + 1) << shifts

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


def _resolve_moc_table_hdu_index(path: str) -> int:
    """Return 0-based HDU index of the IVOA MOC binary table extension."""
    import torchsky.cpp as cpp

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


class MOC:
    """Multi-Order Coverage map."""

    def __init__(self, uniq: Tensor):
        self.uniq = _normalize_moc(torch.as_tensor(uniq, dtype=torch.int64))

    @property
    def max_order(self) -> int:
        if self.uniq.numel() == 0:
            return 0
        nside, _ = uniq2nest(self.uniq.max())
        return int(torch.log2(nside.to(torch.float64)).item())

    def contains(self, lon_deg: Tensor | float, lat_deg: Tensor | float) -> Tensor:
        """Check if points are contained in the MOC."""
        lon = torch.as_tensor(lon_deg, dtype=torch.float64)
        lat = torch.as_tensor(lat_deg, dtype=torch.float64)

        if self.uniq.numel() == 0:
            return torch.zeros_like(lon, dtype=torch.bool)

        # A point is in MOC if its pixel at order O is in MOC for ANY O.
        # Efficient way: for each point, get its pixel at MAX_ORDER.
        # Check if that pixel OR any of its ancestors are in the MOC.

        max_o = self.max_order
        nside_max = 1 << max_o
        p_max = _healpix.ang2pix(nside_max, lon, lat, nest=True, lonlat=True)

        # Convert MOC UNIQ to max_order ranges [start, end)
        nsides, pix_nested = uniq2nest(self.uniq)
        ranges = []
        for ns, p in zip(nsides, pix_nested):
            o = int(torch.log2(ns.to(torch.float64)).item())
            shift = 2 * (max_o - o)
            start = p << shift
            end = (p + 1) << shift
            ranges.append((start.item(), end.item()))

        # Sort and merge ranges (should already be merged if MOC is normalized)
        ranges.sort()

        # Use searchsorted to check if p_max is within any range
        starts = torch.tensor(
            [r[0] for r in ranges], device=p_max.device, dtype=torch.int64
        )
        ends = torch.tensor(
            [r[1] for r in ranges], device=p_max.device, dtype=torch.int64
        )

        idx = torch.searchsorted(starts, p_max, right=True) - 1
        mask = (idx >= 0) & (p_max < ends[idx])
        return mask

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
        return MOC(torch.cat([self.uniq, other.uniq]))

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

    def _to_ranges(self, max_order: int) -> list[tuple[int, int]]:
        return _uniq_to_ranges(self.uniq, max_order)

    @classmethod
    def _from_ranges(cls, ranges: list[tuple[int, int]], max_order: int) -> "MOC":
        """Convert ranges back to UNIQ pixels using greedy filling."""
        return cls(_ranges_to_uniq(ranges, max_order))

    def _flat_pixels(self, nside: int) -> Tensor:
        max_o = int(math.log2(nside))
        ranges = self._to_ranges(max_o)
        if not ranges:
            return torch.empty((0,), dtype=torch.int64)
        chunks = [
            torch.arange(start, end, dtype=torch.int64, device=self.uniq.device)
            for start, end in ranges
        ]
        return torch.cat(chunks)

    def to_sparse_map(self, nside: int, sentinel: float = 0.0) -> HealSparseMap:
        """Realize MOC as a HealSparseMap at fixed resolution."""
        pix = self._flat_pixels(nside)
        if pix.numel() == 0:
            return HealSparseMap.make_empty(nside // 8, nside, torch.float32, sentinel)
        return HealSparseMap.from_pixels(pix, nside, nside // 8, sentinel=sentinel)

    def flatten(self, nside: int) -> Tensor:
        """Project MOC to a dense HEALPix mask at fixed resolution."""
        pix = self._flat_pixels(nside)
        mask = torch.zeros(12 * nside**2, dtype=torch.bool, device=pix.device)
        if pix.numel() > 0:
            mask[pix] = True
        return mask

    @classmethod
    def from_sparse_map(cls, smap: HealSparseMap) -> "MOC":
        """Create MOC from the coverage of a sparse map."""
        # Get all sparse pixels
        pix = smap.get_covered_pixels()
        max_order = int(math.log2(smap.nside_sparse))

        # Convert to ranges
        if pix.numel() == 0:
            return cls(torch.empty((0,), dtype=torch.int64))

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
    def from_circle(
        cls, lon_deg: float, lat_deg: float, radius_deg: float, max_order: int = 10
    ) -> "MOC":
        """Create a MOC from a circular cap."""
        nside = 1 << max_order
        pix = _healpix.query_circle(
            nside, lon_deg, lat_deg, radius_deg, nest=True, degrees=True
        )
        uniqs = nest2uniq(nside, pix)
        return cls(uniqs)

    @classmethod
    def from_polygon(
        cls,
        lon_deg: Tensor | list[float],
        lat_deg: Tensor | list[float],
        max_order: int = 10,
    ) -> "MOC":
        """Create a MOC from a spherical polygon."""
        nside = 1 << max_order
        # query_polygon works on lonlat if requested
        vertices = torch.stack(
            [torch.as_tensor(lon_deg), torch.as_tensor(lat_deg)], dim=-1
        )
        pix = _healpix.query_polygon(nside, vertices, nest=True, lonlat=True)
        uniqs = nest2uniq(nside, pix)
        return cls(uniqs)

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

        # Get covered pixels at nside_sparse
        valid_indices = torch.where(mask)[0]
        if valid_indices.numel() == 0:
            return cls(torch.empty((0,), dtype=torch.int64))

        # We need to map local indices back to global nside_sparse pixel indices
        # This is more efficient if done via coverage_map
        all_global_pix = []
        cov_valid = torch.where(smap.coverage_map >= 0)[0]
        pixels_per_cov = smap.pixels_per_cov

        # We can optimize this by only checking valid_indices within each block
        for cp in cov_valid:
            offset = int(smap.coverage_map[cp].item())
            # Check if any pixels in this block are in valid_indices
            # Actually, simpler: mask the values locally
            block_mask = mask[offset : offset + pixels_per_cov]
            if block_mask.any():
                local_offsets = torch.where(block_mask)[0]
                global_start = int(cp.item()) * pixels_per_cov
                all_global_pix.append(global_start + local_offsets)

        if not all_global_pix:
            return cls(torch.empty((0,), dtype=torch.int64))

        pix = torch.cat(all_global_pix)
        uniqs = nest2uniq(smap.nside_sparse, pix)
        return cls(uniqs)

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write the MOC to a FITS file following the IVOA MOC standard.

        Uses :func:`torchsky.write` (CFITSIO-backed) — no Python ``fitsio`` dependency.
        """
        import torchsky

        header = {
            "PIXTYPE": "HEALPIX",
            "ORDERING": "NUNIQ",
            "COORDSYS": "C",
            "MOCORDER": self.max_order,
            "MOCTOOL": "torchsky",
            "EXTNAME": "MOC",
        }
        u = self.uniq.detach().cpu().reshape(-1).to(torch.int64)
        torchsky.write(filename, {"UNIQ": u}, header=header, overwrite=overwrite)

    @classmethod
    def read_fits(cls, filename: str) -> "MOC":
        """Read a MOC from an IVOA FITS file via :func:`torchsky.read_table`."""
        import torchsky

        hdu = _resolve_moc_table_hdu_index(filename)
        table = torchsky.read_table(filename, hdu=hdu)
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

    def to_ascii(self) -> str:
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
            pix_str = " ".join(map(str, o_pix.tolist()))
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
            pix_list = [int(p) for p in pix_str.replace(",", " ").split()]
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


__all__ = ["MOC", "nest2uniq", "uniq2nest"]
