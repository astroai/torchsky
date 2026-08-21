"""Reusable nested-HEALPix catalog index."""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor

from ..sphere import great_circle_distance
from ..sphere import healpix as _healpix


def default_nside(n_points: int) -> int:
    """Coarse NSIDE from catalog size. Same bins as the crossmatch prune."""
    if n_points <= 25_000:
        return 64
    if n_points <= 90_000:
        return 128
    if n_points <= 180_000:
        return 512
    if n_points <= 1_000_000:
        return 1024
    return 2048


def candidate_coarse_pixels_for_left_pixel(
    coarse_nside: int,
    coarse_pixel: int,
    *,
    radius_rad: float,
    device: torch.device,
) -> Tensor:
    """NESTED pixels within ``radius_rad`` of ``coarse_pixel``, plus a pixrad pad."""
    center_pix = torch.tensor([coarse_pixel], dtype=torch.int64, device=device)
    pixrad = _healpix.max_pixrad(coarse_nside, degrees=False)
    hops = max(1, int(math.ceil(radius_rad / max(pixrad, 1e-12)))) + 1

    visited = center_pix
    frontier = center_pix
    for _ in range(hops):
        neigh = _healpix.get_all_neighbours(
            coarse_nside,
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

    lon_c, lat_c = _healpix.pix2ang(coarse_nside, center_pix, nest=True, lonlat=True)
    cand_lon, cand_lat = _healpix.pix2ang(coarse_nside, visited, nest=True, lonlat=True)
    sep = great_circle_distance(
        lon_c.expand_as(cand_lon),
        lat_c.expand_as(cand_lat),
        cand_lon,
        cand_lat,
        degrees=False,
    )
    return visited[sep <= radius_rad]


def _work_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


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

    def query_disc(self, lon: Tensor | float, lat: Tensor | float, radius_deg: float) -> Tensor:
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
    "candidate_coarse_pixels_for_left_pixel",
    "default_nside",
]
