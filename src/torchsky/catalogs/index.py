"""Reusable nested-HEALPix catalog index."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..sphere import great_circle_distance
from ..sphere import healpix as _healpix

_MAX_NSIDE = 65536
_TARGET_POINTS_PER_PIXEL = 64
_BATCHED_RING_HOPS_MAX = 4


def _work_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def nside_for_cap(radius_rad: float, *, max_nside: int = _MAX_NSIDE) -> int:
    """Coarsest power-of-two NSIDE whose max pixel radius is ``<= radius_rad``.

    If the cap is larger than an nside=1 pixel, return 1. If even ``max_nside``
    pixels are larger than the cap, return ``max_nside`` (covering stays sound
    via the ``r + 2ρ`` pad; tiles are just coarser).
    """
    if not math.isfinite(radius_rad) or radius_rad <= 0.0:
        raise ValueError("radius_rad must be finite and positive")
    _healpix._validate_nside(max_nside)
    nside = 1
    while nside < max_nside:
        if _healpix.max_pixrad(nside, degrees=False) <= radius_rad:
            return nside
        nside *= 2
    return max_nside


def default_nside(n_points: int, *, max_nside: int = _MAX_NSIDE) -> int:
    """NSIDE for an index built without a match radius.

    Targets mean occupancy of about 64 rows per pixel so the bucket array
    stays comparable to catalog size. Crossmatch uses ``nside_for_cap``.
    """
    _healpix._validate_nside(max_nside)
    if n_points <= 0:
        return 64
    # 12 nside^2 pixels; occupancy = n / (12 nside^2) ≈ 64.
    target_pixels = max(1.0, float(n_points) / float(_TARGET_POINTS_PER_PIXEL))
    nside = 1
    while nside * 2 <= max_nside and 12.0 * float(nside * 2) ** 2 <= target_pixels:
        nside *= 2
    return nside


def cap_covering_pixels(
    nside: int,
    lon_deg: Tensor | float,
    lat_deg: Tensor | float,
    radius_rad: float,
    *,
    nest: bool = True,
) -> Tensor:
    """Pixels whose centers lie in the cap of radius ``radius_rad``.

    Small caps expand nested neighbours (``O(hops^2)``). Wide caps use
    ``query_disc``. Callers that need a sound match covering pass ``r + 2ρ``.
    """
    if not math.isfinite(radius_rad) or radius_rad <= 0.0:
        raise ValueError("radius_rad must be finite and positive")
    _healpix._validate_nside(nside)
    lon_t = torch.as_tensor(lon_deg)
    device = lon_t.device
    dtype = _work_dtype(device)
    lon_t = lon_t.to(dtype=dtype).reshape(-1)
    lat_t = torch.as_tensor(lat_deg, device=device, dtype=dtype).reshape(-1)
    if lon_t.numel() != 1 or lat_t.numel() != 1:
        raise ValueError("cap_covering_pixels takes a single center")
    if radius_rad >= math.pi:
        npix = int(_healpix.nside2npix(nside))
        pix = torch.arange(npix, dtype=torch.int64, device=device)
        if nest:
            return pix
        return _healpix.nest2ring(nside, pix)

    pixrad = _healpix.max_pixrad(nside, degrees=False)
    hops = max(1, int(math.ceil(radius_rad / max(pixrad, 1e-12)))) + 1
    if hops > 16:
        x, y, z = _healpix.lonlat_to_xyz(lon_t, lat_t)
        vec = torch.stack((x.reshape(-1)[0], y.reshape(-1)[0], z.reshape(-1)[0]))
        return _healpix.query_disc(
            nside,
            vec,
            float(radius_rad),
            inclusive=False,
            nest=bool(nest),
        )

    center_pix = (
        _healpix.ang2pix(nside, lon_t, lat_t, nest=True, lonlat=True)
        .to(torch.int64)
        .reshape(-1)
    )
    visited = center_pix
    frontier = center_pix
    for _ in range(hops):
        neigh = _healpix.get_all_neighbours(
            nside,
            frontier,
            nest=True,
            lonlat=False,
        )
        neigh = torch.as_tensor(neigh, dtype=torch.int64, device=device).reshape(-1)
        neigh = neigh[neigh >= 0]
        if neigh.numel() == 0:
            break
        merged = torch.unique(torch.cat((visited, neigh)))
        if merged.numel() == visited.numel():
            break
        visited = merged
        frontier = neigh

    cand_lon, cand_lat = _healpix.pix2ang(nside, visited, nest=True, lonlat=True)
    sep = great_circle_distance(
        lon_t.expand_as(cand_lon),
        lat_t.expand_as(cand_lat),
        cand_lon,
        cand_lat,
        degrees=False,
    )
    nest_pix = visited[sep <= radius_rad]
    if nest:
        return nest_pix
    return _healpix.nest2ring(nside, nest_pix)


def candidate_coarse_pixels_for_left_pixel(
    coarse_nside: int,
    coarse_pixel: int,
    *,
    radius_rad: float,
    device: torch.device,
) -> Tensor:
    """NESTED pixels whose centers lie within ``radius_rad`` of ``coarse_pixel``."""
    center_pix = torch.tensor([coarse_pixel], dtype=torch.int64, device=device)
    lon_c, lat_c = _healpix.pix2ang(coarse_nside, center_pix, nest=True, lonlat=True)
    return cap_covering_pixels(
        coarse_nside,
        lon_c,
        lat_c,
        radius_rad,
        nest=True,
    )


def nested_neighbor_rings(nside: int, pix: Tensor, hops: int) -> Tensor:
    """Self plus successive nested neighbour layers. Shape ``[K, N]``, ``-1`` empty."""
    pix_t = torch.as_tensor(pix, dtype=torch.int64).reshape(-1)
    n_pix = int(pix_t.numel())
    if n_pix == 0 or hops < 1:
        return pix_t.reshape(1, n_pix)
    layers = [pix_t.reshape(1, n_pix)]
    layer = pix_t.reshape(1, n_pix)
    for _ in range(int(hops)):
        width = int(layer.shape[0])
        flat = layer.reshape(width * n_pix)
        invalid = flat < 0
        safe = flat.clamp(min=0)
        neigh = _healpix.get_all_neighbours(nside, safe, nest=True, lonlat=False)
        neigh = torch.as_tensor(neigh, dtype=torch.int64, device=pix_t.device)
        if neigh.ndim == 1:
            neigh = neigh.reshape(8, -1)
        neigh = neigh.masked_fill(invalid.unsqueeze(0), -1)
        layer = neigh.reshape(8, width, n_pix).reshape(8 * width, n_pix)
        layers.append(layer)
    return torch.cat(layers, dim=0)


@dataclass
class HealpixIndex:
    """Bucketed nested-HEALPix index over catalog rows.

    Build once, then ``query_disc`` / ``lookup``. Crossmatch uses the same
    buckets; a later fused gather can replace ``lookup`` without API churn.
    """

    nside: int
    nest: bool
    lon: Tensor
    lat: Tensor
    order: Tensor
    unique_pixels: Tensor
    bucket_starts: Tensor
    bucket_stops: Tensor

    @classmethod
    def build(
        cls,
        lon: Tensor,
        lat: Tensor,
        *,
        nside: int | None = None,
        nest: bool = True,
    ) -> HealpixIndex:
        lon_t = torch.as_tensor(lon)
        device = lon_t.device
        dtype = _work_dtype(device)
        lon_t = lon_t.to(dtype=dtype).reshape(-1)
        lat_t = torch.as_tensor(lat, device=device, dtype=dtype).reshape(-1)
        if lon_t.numel() != lat_t.numel():
            raise ValueError("lon and lat must have the same number of rows")
        nside_i = default_nside(int(lon_t.numel())) if nside is None else int(nside)
        _healpix._validate_nside(nside_i)
        empty = torch.empty(0, dtype=torch.int64, device=device)
        if lon_t.numel() == 0:
            return cls(
                nside=nside_i,
                nest=bool(nest),
                lon=lon_t,
                lat=lat_t,
                order=empty,
                unique_pixels=empty,
                bucket_starts=empty,
                bucket_stops=empty,
            )
        pixels = _healpix.ang2pix(
            nside_i,
            lon_t,
            lat_t,
            nest=bool(nest),
            lonlat=True,
        ).to(torch.int64)
        order = torch.argsort(pixels, stable=True)
        sorted_pixels = pixels[order]
        unique_pixels, counts = torch.unique_consecutive(
            sorted_pixels, return_counts=True
        )
        bucket_starts = torch.cat(
            (
                torch.zeros(1, dtype=torch.int64, device=device),
                torch.cumsum(counts[:-1].to(torch.int64), dim=0),
            )
        )
        bucket_stops = bucket_starts + counts.to(torch.int64)
        return cls(
            nside=nside_i,
            nest=bool(nest),
            lon=lon_t,
            lat=lat_t,
            order=order,
            unique_pixels=unique_pixels.to(torch.int64),
            bucket_starts=bucket_starts,
            bucket_stops=bucket_stops,
        )

    def to(self, device: torch.device | str) -> HealpixIndex:
        device_t = torch.device(device)
        dtype = _work_dtype(device_t)
        return HealpixIndex(
            nside=self.nside,
            nest=self.nest,
            lon=self.lon.to(device=device_t, dtype=dtype),
            lat=self.lat.to(device=device_t, dtype=dtype),
            order=self.order.to(device=device_t),
            unique_pixels=self.unique_pixels.to(device=device_t),
            bucket_starts=self.bucket_starts.to(device=device_t),
            bucket_stops=self.bucket_stops.to(device=device_t),
        )

    def lookup(self, pixels: Tensor) -> Tensor:
        """Row indices whose HEALPix pixel is in ``pixels`` (vectorized gather)."""
        pix = torch.as_tensor(
            pixels,
            dtype=torch.int64,
            device=self.unique_pixels.device,
        ).reshape(-1)
        if pix.numel() == 0 or self.unique_pixels.numel() == 0:
            return self.order[:0]

        pix = torch.unique(pix, sorted=True)
        idx = torch.searchsorted(self.unique_pixels, pix)
        n_uniq = int(self.unique_pixels.numel())
        idx_safe = idx.clamp(max=max(n_uniq - 1, 0))
        mask = (idx < n_uniq) & (self.unique_pixels[idx_safe] == pix)
        if not bool(mask.any()):
            return self.order[:0]

        starts = self.bucket_starts[idx_safe[mask]]
        stops = self.bucket_stops[idx_safe[mask]]
        lengths = (stops - starts).to(torch.int64)
        total = int(lengths.sum().item())
        if total == 0:
            return self.order[:0]

        device = self.order.device
        bucket_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=device, dtype=torch.int64),
            lengths,
        )
        offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=device),
                    lengths[:-1],
                )
            ),
            dim=0,
        )
        intra = torch.arange(total, device=device, dtype=torch.int64)
        intra = intra - torch.repeat_interleave(offsets, lengths)
        return self.order[starts[bucket_ids] + intra]

    def expand(self, pixels: Tensor) -> tuple[Tensor, Tensor]:
        """Map per-query pixels ``[N]`` to ``(query_index, row_index)`` pairs."""
        pix = torch.as_tensor(
            pixels,
            dtype=torch.int64,
            device=self.unique_pixels.device,
        ).reshape(-1)
        n_query = int(pix.numel())
        empty_q = torch.empty(0, dtype=torch.int64, device=pix.device)
        if n_query == 0 or self.unique_pixels.numel() == 0:
            return empty_q, empty_q
        idx = torch.searchsorted(self.unique_pixels, pix)
        n_uniq = int(self.unique_pixels.numel())
        idx_safe = idx.clamp(max=max(n_uniq - 1, 0))
        match = (idx < n_uniq) & (self.unique_pixels[idx_safe] == pix) & (pix >= 0)
        if not bool(match.any()):
            return empty_q, empty_q
        query = torch.arange(n_query, device=pix.device, dtype=torch.int64)[match]
        starts = self.bucket_starts[idx_safe[match]]
        stops = self.bucket_stops[idx_safe[match]]
        lengths = (stops - starts).to(torch.int64)
        total = int(lengths.sum().item())
        if total == 0:
            return empty_q, empty_q
        query_exp = torch.repeat_interleave(query, lengths)
        bucket_ids = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=pix.device, dtype=torch.int64),
            lengths,
        )
        offsets = torch.cumsum(
            torch.cat(
                (
                    torch.zeros(1, dtype=torch.int64, device=pix.device),
                    lengths[:-1],
                )
            ),
            dim=0,
        )
        intra = torch.arange(total, device=pix.device, dtype=torch.int64)
        intra = intra - torch.repeat_interleave(offsets, lengths)
        return query_exp, self.order[starts[bucket_ids] + intra]

    def query_disc(
        self, lon: Tensor | float, lat: Tensor | float, radius_deg: float
    ) -> Tensor:
        """Row indices whose positions lie within ``radius_deg`` of the center."""
        if self.lon.numel() == 0:
            return self.order[:0]
        device = self.lon.device
        lon_t = torch.as_tensor(lon, device=device, dtype=self.lon.dtype).reshape(-1)
        lat_t = torch.as_tensor(lat, device=device, dtype=self.lat.dtype).reshape(-1)
        if lon_t.numel() != 1 or lat_t.numel() != 1:
            raise ValueError("query_disc takes a single center")
        radius = float(radius_deg)
        if radius <= 0.0:
            raise ValueError("radius_deg must be positive")

        center_nest = _healpix.ang2pix(
            self.nside, lon_t, lat_t, nest=True, lonlat=True
        ).to(torch.int64)
        candidate_radius_rad = math.radians(radius) + (
            2.0 * _healpix.max_pixrad(self.nside, degrees=False)
        )
        cand_nest = candidate_coarse_pixels_for_left_pixel(
            self.nside,
            int(center_nest.reshape(-1)[0].item()),
            radius_rad=candidate_radius_rad,
            device=device,
        )
        cand = cand_nest if self.nest else _healpix.nest2ring(self.nside, cand_nest)
        idx = self.lookup(cand)
        if idx.numel() == 0:
            return idx
        sep = great_circle_distance(
            lon_t, lat_t, self.lon[idx], self.lat[idx], degrees=True
        )
        return idx[sep <= radius]


__all__ = [
    "HealpixIndex",
    "cap_covering_pixels",
    "candidate_coarse_pixels_for_left_pixel",
    "default_nside",
    "nested_neighbor_rings",
    "nside_for_cap",
]
