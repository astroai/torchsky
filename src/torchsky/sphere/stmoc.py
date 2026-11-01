"""Space-Time Multi-Order Coverage (IVOA STMOC) on torch tensors.

Time cells follow MOC 2.0: order 61 is 1 µs since JD=0. Spatial cells are
torchsky ``MOC`` UNIQ coverage. This is not a mocpy wrapper.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import torch
from torch import Tensor

from torchfits import read_header as get_header, write
from torchfits.table import read_torch as read_table
from . import healpix as _healpix
from .moc import MOC, _moc_table_hdu, nest2uniq

TIME_MAX_ORDER = 61
_US_PER_DAY = 86_400_000_000
_JD_MJD_OFFSET_INT = 2_400_000  # JD = MJD + 2400000.5
_TIME_FLAG = torch.tensor(-9223372036854775808, dtype=torch.int64)  # 1<<63 signed
_CONTAINS_CHUNK_SIZE = 262_144
_SERIAL_CONTAINS_PRODUCT = 64


def mjd_to_time_us(mjd: Tensor | float) -> Tensor:
    """Convert MJD to integer microseconds since JD=0 (TMOC order-61 index)."""
    t = (
        mjd
        if isinstance(mjd, Tensor)
        else torch.as_tensor(mjd, dtype=torch.float64)
    )
    device = t.device
    # ponytail: float32 MJD cannot encode µs near MJD 5e4; host float64 on MPS.
    if device.type == "mps":
        mjd64 = t.detach().cpu().to(dtype=torch.float64)
    else:
        mjd64 = t.to(dtype=torch.float64) if t.dtype != torch.float64 else t
    day = torch.floor(mjd64)
    frac = mjd64 - day
    frac_us = torch.round((frac + 0.5) * float(_US_PER_DAY)).to(torch.int64)
    carry = torch.div(frac_us, _US_PER_DAY, rounding_mode="floor")
    frac_us = frac_us - carry * _US_PER_DAY
    days = day.to(torch.int64) + _JD_MJD_OFFSET_INT + carry
    out = days * _US_PER_DAY + frac_us
    return out.to(device=device) if out.device != device else out


def time_us_to_mjd(time_us: Tensor) -> Tensor:
    """Inverse of :func:`mjd_to_time_us` (sub-microsecond MJD)."""
    us = torch.as_tensor(time_us, dtype=torch.int64)
    days = torch.div(us, _US_PER_DAY, rounding_mode="floor")
    rem = us - days * _US_PER_DAY
    jd = days.to(torch.float64) + rem.to(torch.float64) / float(_US_PER_DAY)
    return jd - 2_400_000.5


def _cell_size_us(time_depth: int) -> int:
    depth = int(time_depth)
    if depth < 0 or depth > TIME_MAX_ORDER:
        raise ValueError(f"time_depth must be in 0..{TIME_MAX_ORDER}")
    return 1 << (TIME_MAX_ORDER - depth)


def _align_range_us(start_us: int, end_us: int, time_depth: int) -> tuple[int, int]:
    if end_us <= start_us:
        raise ValueError("time range must be positive")
    dt = _cell_size_us(time_depth)
    c0 = start_us // dt
    c1 = (end_us - 1) // dt
    return c0 * dt, (c1 + 1) * dt


@dataclass(frozen=True)
class STMOC:
    """Space-time coverage: half-open time ranges (µs since JD=0) × spatial MOC."""

    time_start_us: Tensor
    time_end_us: Tensor
    space: tuple[MOC, ...]
    time_depth: int
    spatial_depth: int
    _time_order: Tensor = field(init=False, repr=False, compare=False)
    _time_start_sorted: Tensor = field(init=False, repr=False, compare=False)
    _time_end_sorted: Tensor = field(init=False, repr=False, compare=False)
    _time_non_overlapping: bool = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        start = torch.as_tensor(self.time_start_us, dtype=torch.int64).reshape(-1)
        end = torch.as_tensor(self.time_end_us, dtype=torch.int64).reshape(-1)
        if start.shape != end.shape:
            raise ValueError("time_start_us and time_end_us must match")
        if start.numel() != len(self.space):
            raise ValueError("space segment count must match time ranges")
        if int(self.time_depth) < 0 or int(self.time_depth) > TIME_MAX_ORDER:
            raise ValueError(f"time_depth must be in 0..{TIME_MAX_ORDER}")
        if int(self.spatial_depth) < 0:
            raise ValueError("spatial_depth must be non-negative")
        if bool((end <= start).any().item()) if start.numel() else False:
            raise ValueError("each time range must satisfy start < end")
        order = torch.argsort(start, stable=True)
        start_sorted = start.index_select(0, order)
        end_sorted = end.index_select(0, order)
        non_overlapping = not (
            bool((start_sorted[1:] < end_sorted[:-1]).any().item())
            if start_sorted.numel() > 1
            else False
        )
        object.__setattr__(self, "time_start_us", start)
        object.__setattr__(self, "time_end_us", end)
        object.__setattr__(self, "time_depth", int(self.time_depth))
        object.__setattr__(self, "spatial_depth", int(self.spatial_depth))
        object.__setattr__(self, "_time_order", order)
        object.__setattr__(self, "_time_start_sorted", start_sorted)
        object.__setattr__(self, "_time_end_sorted", end_sorted)
        object.__setattr__(self, "_time_non_overlapping", non_overlapping)

    @classmethod
    def empty(cls, *, time_depth: int = 0, spatial_depth: int = 0) -> "STMOC":
        return cls(
            torch.empty((0,), dtype=torch.int64),
            torch.empty((0,), dtype=torch.int64),
            (),
            time_depth,
            spatial_depth,
        )

    @property
    def n_segments(self) -> int:
        return int(self.time_start_us.numel())

    def contains(
        self,
        time_mjd: Tensor | float,
        lon_deg: Tensor | float,
        lat_deg: Tensor | float,
    ) -> Tensor:
        """True where (time, lon, lat) lies in the space-time coverage."""
        t = torch.as_tensor(time_mjd)
        lon = torch.as_tensor(lon_deg)
        lat = torch.as_tensor(lat_deg)
        lon = lon.to(device=t.device)
        lat = lat.to(device=t.device)
        try:
            t, lon, lat = torch.broadcast_tensors(t, lon, lat)
        except RuntimeError as exc:
            raise ValueError("time_mjd, lon_deg, and lat_deg must broadcast") from exc
        out = torch.zeros(t.shape, dtype=torch.bool, device=t.device)
        if self.n_segments == 0:
            return out
        t_us = mjd_to_time_us(t.reshape(-1))
        lon_f = lon.reshape(-1)
        lat_f = lat.reshape(-1)
        if (
            t.device.type == "cpu"
            and t_us.numel() * self.n_segments <= _SERIAL_CONTAINS_PRODUCT
        ):
            flat = _contains_serial(self, t_us, lon_f, lat_f)
        elif self._time_non_overlapping:
            flat = _contains_sorted(
                self._time_start_sorted,
                self._time_end_sorted,
                self._time_order,
                self.space,
                t_us,
                lon_f,
                lat_f,
            )
        else:
            starts, ends, space = _canonical_temporal_segments(self)
            flat = _contains_sorted(
                starts,
                ends,
                torch.arange(starts.numel(), dtype=torch.int64),
                space,
                t_us,
                lon_f,
                lat_f,
            )
        return flat.reshape(t.shape)

    def filter_catalog(
        self,
        time_mjd: Tensor,
        lon_deg: Tensor,
        lat_deg: Tensor,
    ) -> tuple[Tensor, Tensor, Tensor]:
        mask = self.contains(time_mjd, lon_deg, lat_deg)
        return time_mjd[mask], lon_deg[mask], lat_deg[mask]

    def query_by_time(self, t_start_mjd: float, t_end_mjd: float) -> MOC:
        """Spatial MOC of segments overlapping ``[t_start_mjd, t_end_mjd)``."""
        t0 = int(mjd_to_time_us(float(t_start_mjd)).item())
        t1 = int(mjd_to_time_us(float(t_end_mjd)).item())
        if t1 <= t0:
            raise ValueError("query time range must be positive")
        if self.n_segments == 0:
            return MOC(torch.empty((0,), dtype=torch.int64))
        starts = self._time_start_sorted
        ends = self._time_end_sorted
        device = starts.device
        query = torch.tensor([t0, t1], dtype=torch.int64, device=device)
        if self._time_non_overlapping:
            lo = torch.searchsorted(ends, query[:1], right=True)
            hi = torch.searchsorted(starts, query[1:], right=False)
            lo_i, hi_i = torch.cat([lo, hi]).detach().cpu().tolist()
            selected = self._time_order[lo_i:hi_i]
        else:
            stop = int(
                torch.searchsorted(starts, query[1:], right=False)
                .detach()
                .cpu()[0]
            )
            selected = self._time_order[:stop][ends[:stop] > query[0]]
        indices = selected.detach().cpu().tolist()
        acc: MOC | None = None
        seen: set[int] = set()
        for index in indices:
            moc = self.space[index]
            key = id(moc)
            if key in seen:
                continue
            seen.add(key)
            acc = moc if acc is None else acc.union(moc)
        return acc if acc is not None else MOC(torch.empty((0,), dtype=torch.int64))

    def to_ascii(self) -> str:
        """IVOA-like STMOC ASCII: ``t<depth>/<c0>-<c1>`` plus spatial MOC ASCII."""
        if self.n_segments == 0:
            return ""
        dt = _cell_size_us(self.time_depth)
        chunks: list[str] = []
        for i in range(self.n_segments):
            s = int(self.time_start_us[i].item())
            e = int(self.time_end_us[i].item())
            c0 = s // dt
            c1 = (e - 1) // dt
            spatial = self.space[i].to_ascii()
            time_tok = f"t{self.time_depth}/{c0}-{c1}"
            chunks.append(f"{time_tok} {spatial}".strip() if spatial else time_tok)
        return " ".join(chunks)

    @classmethod
    def from_ascii(
        cls,
        text: str,
        *,
        spatial_depth: int | None = None,
    ) -> "STMOC":
        """Parse :meth:`to_ascii` output. Spatial depth defaults to the MOC max order."""
        stripped = text.strip()
        if not stripped:
            return cls.empty()
        parts = re.split(r"(t\d+/)", stripped)
        segs: list[tuple[int, int, MOC]] = []
        time_depth = 0
        max_spatial = 0 if spatial_depth is None else int(spatial_depth)
        i = 1
        while i < len(parts):
            prefix = parts[i]
            rest = parts[i + 1] if i + 1 < len(parts) else ""
            i += 2
            depth = int(prefix[1:-1])
            time_depth = max(time_depth, depth)
            token, _, leftover = rest.strip().partition(" ")
            if "-" not in token:
                raise ValueError(f"STMOC ASCII time range must be c0-c1, got {token!r}")
            a_s, b_s = token.split("-", 1)
            c0, c1 = int(a_s), int(b_s)
            dt = _cell_size_us(depth)
            moc = MOC.from_ascii(leftover)
            max_spatial = max(max_spatial, moc.max_order)
            segs.append((c0 * dt, (c1 + 1) * dt, moc))
        return _from_segments(segs, time_depth=time_depth, spatial_depth=max_spatial)

    def union(self, other: "STMOC") -> "STMOC":
        depth_t = max(self.time_depth, other.time_depth)
        depth_s = max(self.spatial_depth, other.spatial_depth)
        return _from_segments(
            _sweep_boolean(self, other, op="union"),
            time_depth=depth_t,
            spatial_depth=depth_s,
        )

    def intersection(self, other: "STMOC") -> "STMOC":
        depth_t = max(self.time_depth, other.time_depth)
        depth_s = max(self.spatial_depth, other.spatial_depth)
        return _from_segments(
            _sweep_boolean(self, other, op="intersection"),
            time_depth=depth_t,
            spatial_depth=depth_s,
        )

    def difference(self, other: "STMOC") -> "STMOC":
        depth_t = max(self.time_depth, other.time_depth)
        depth_s = max(self.spatial_depth, other.spatial_depth)
        return _from_segments(
            _sweep_boolean(self, other, op="difference"),
            time_depth=depth_t,
            spatial_depth=depth_s,
        )

    @classmethod
    def from_spatial_coverages(
        cls,
        t_start_mjd: Tensor | float,
        t_end_mjd: Tensor | float,
        spatial: MOC | tuple[MOC, ...],
        *,
        time_depth: int,
    ) -> "STMOC":
        """One spatial MOC per time interval (broadcasts a single MOC)."""
        t0 = torch.as_tensor(t_start_mjd)
        t1 = torch.as_tensor(t_end_mjd)
        if t0.device.type == "mps" or t1.device.type == "mps":
            t0 = t0.detach().cpu().to(dtype=torch.float64)
            t1 = t1.detach().cpu().to(dtype=torch.float64)
        else:
            t0 = t0.to(dtype=torch.float64)
            t1 = t1.to(dtype=torch.float64)
        t0 = t0.reshape(-1)
        t1 = t1.reshape(-1)
        if t0.numel() != t1.numel():
            raise ValueError("t_start_mjd and t_end_mjd must have the same length")
        mocs: tuple[MOC, ...]
        if isinstance(spatial, MOC):
            mocs = tuple(spatial for _ in range(int(t0.numel())))
        else:
            mocs = tuple(spatial)
            if len(mocs) != int(t0.numel()):
                raise ValueError(
                    "spatial coverages must match the number of time ranges"
                )
        segs: list[tuple[int, int, MOC]] = []
        spatial_depth = 0
        for i in range(int(t0.numel())):
            s = int(mjd_to_time_us(t0[i]).item())
            e = int(mjd_to_time_us(t1[i]).item())
            s, e = _align_range_us(s, e, time_depth)
            segs.append((s, e, mocs[i]))
            spatial_depth = max(spatial_depth, mocs[i].max_order)
        return _from_segments(segs, time_depth=time_depth, spatial_depth=spatial_depth)

    @classmethod
    def from_times_positions(
        cls,
        time_mjd: Tensor,
        lon_deg: Tensor,
        lat_deg: Tensor,
        *,
        time_depth: int,
        spatial_depth: int,
    ) -> "STMOC":
        """Coverage of catalog samples quantized to ``time_depth`` × ``spatial_depth``."""
        t = torch.as_tensor(time_mjd)
        lon = torch.as_tensor(lon_deg)
        lat = torch.as_tensor(lat_deg)
        if t.device.type == "mps":
            t = t.detach().cpu().to(dtype=torch.float64)
            lon = lon.detach().cpu().to(dtype=torch.float64)
            lat = lat.detach().cpu().to(dtype=torch.float64)
        else:
            t = t.to(dtype=torch.float64)
            lon = lon.to(dtype=torch.float64)
            lat = lat.to(dtype=torch.float64)
        t = t.reshape(-1)
        lon = lon.reshape(-1)
        lat = lat.reshape(-1)
        if t.numel() != lon.numel() or t.numel() != lat.numel():
            raise ValueError("time_mjd, lon_deg, and lat_deg must have the same length")
        if t.numel() == 0:
            return cls.empty(time_depth=time_depth, spatial_depth=spatial_depth)
        dt = _cell_size_us(time_depth)
        t_us = mjd_to_time_us(t)
        cell = torch.div(t_us, dt, rounding_mode="floor")
        nside = 1 << int(spatial_depth)
        pix = _healpix.ang2pix(nside, lon, lat, nest=True, lonlat=True).to(torch.int64)
        order = torch.argsort(cell, stable=True)
        cell_s = cell.index_select(0, order)
        pix_s = pix.index_select(0, order)
        breaks = torch.ones(cell_s.numel(), dtype=torch.bool)
        if cell_s.numel() > 1:
            breaks[1:] = cell_s[1:] != cell_s[:-1]
        starts = torch.nonzero(breaks, as_tuple=False).reshape(-1)
        ends = torch.cat(
            [
                starts[1:],
                torch.tensor([cell_s.numel()], dtype=torch.int64, device=starts.device),
            ]
        )
        segs: list[tuple[int, int, MOC]] = []
        for a, b in zip(starts.tolist(), ends.tolist(), strict=True):
            c = int(cell_s[a].item())
            moc = MOC(nest2uniq(nside, pix_s[a:b]))
            segs.append((c * dt, (c + 1) * dt, moc))
        return _from_segments(segs, time_depth=time_depth, spatial_depth=spatial_depth)

    def write_fits(self, filename: str, overwrite: bool = False) -> None:
        """Write IVOA RANGE packaging: interleaved [min,max) int64 pairs.

        Time indices have bit 63 set. Spatial ranges are NESTED pixels at
        ``spatial_depth``.
        """
        pairs: list[int] = []
        flag = int(_TIME_FLAG.item())
        for i in range(self.n_segments):
            s = int(self.time_start_us[i].item())
            e = int(self.time_end_us[i].item())
            pairs.extend([s | flag, e | flag])
            ranges = self.space[i]._to_ranges(self.spatial_depth)
            for a, b in ranges:
                pairs.extend([a, b])
        header = {
            "PIXTYPE": "MOC",
            "ORDERING": "RANGE",
            "MOCDIM": 2,
            "TORDER": self.time_depth,
            "MOCORDER": self.spatial_depth,
            "TIMESYS": "JD",
            "MOCTOOL": "torchsky",
            "EXTNAME": "MOC",
        }
        data = (
            torch.tensor(pairs, dtype=torch.int64)
            if pairs
            else torch.empty((0,), dtype=torch.int64)
        )
        write(filename, {"RANGE": data}, header=header, overwrite=overwrite)

    @classmethod
    def read_fits(cls, filename: str) -> "STMOC":
        hdu = _moc_table_hdu(filename)
        hdr = dict(get_header(filename, hdu=hdu))
        table = read_table(filename, hdu=hdu)
        if "RANGE" not in table and "range" not in table:
            vals = torch.empty((0,), dtype=torch.int64)
        else:
            col = table["RANGE"] if "RANGE" in table else table["range"]
            vals = torch.as_tensor(col, dtype=torch.int64).reshape(-1)
        time_depth = int(hdr.get("TORDER", 0) or 0)
        spatial_depth = int(hdr.get("MOCORDER", 0) or 0)
        segs = _segments_from_range_stream(vals, spatial_depth)
        return _from_segments(segs, time_depth=time_depth, spatial_depth=spatial_depth)


def _contains_serial(
    stmoc: STMOC, t_us: Tensor, lon: Tensor, lat: Tensor
) -> Tensor:
    out = torch.zeros(t_us.numel(), dtype=torch.bool, device=t_us.device)
    starts = stmoc.time_start_us.to(device=t_us.device)
    ends = stmoc.time_end_us.to(device=t_us.device)
    for index, moc in enumerate(stmoc.space):
        hit = (t_us >= starts[index]) & (t_us < ends[index])
        point_indices = torch.nonzero(hit, as_tuple=False).reshape(-1)
        if point_indices.numel() == 0:
            continue
        inside = moc.contains(
            lon.index_select(0, point_indices), lat.index_select(0, point_indices)
        )
        out[point_indices] = out.index_select(0, point_indices) | inside
    return out


def _contains_sorted(
    starts: Tensor,
    ends: Tensor,
    segment_order: Tensor,
    space: tuple[MOC, ...],
    t_us: Tensor,
    lon: Tensor,
    lat: Tensor,
) -> Tensor:
    if starts.numel() == 0:
        return torch.zeros(t_us.numel(), dtype=torch.bool, device=t_us.device)
    starts_d = starts.to(device=t_us.device)
    ends_d = ends.to(device=t_us.device)
    order_d = segment_order.to(device=t_us.device)
    out = torch.zeros(t_us.numel(), dtype=torch.bool, device=t_us.device)
    for offset in range(0, t_us.numel(), _CONTAINS_CHUNK_SIZE):
        stop = min(offset + _CONTAINS_CHUNK_SIZE, t_us.numel())
        times = t_us[offset:stop]
        positions = torch.searchsorted(starts_d, times, right=True) - 1
        safe_positions = positions.clamp(min=0)
        valid = (positions >= 0) & (times < ends_d[safe_positions])
        candidates = order_d[safe_positions]
        active_segments = torch.unique(candidates[valid]).detach().cpu().tolist()
        groups: dict[int, tuple[MOC, list[int]]] = {}
        for segment in active_segments:
            moc = space[segment]
            group = groups.get(id(moc))
            if group is None:
                groups[id(moc)] = (moc, [segment])
            else:
                group[1].append(segment)
        chunk_out = out[offset:stop]
        for moc, segments in groups.values():
            if len(groups) == 1:
                group_hit = valid
            elif len(segments) == 1:
                group_hit = valid & (candidates == segments[0])
            else:
                segment_tensor = torch.tensor(
                    segments, dtype=torch.int64, device=t_us.device
                )
                group_hit = valid & torch.isin(candidates, segment_tensor)
            point_indices = torch.nonzero(group_hit, as_tuple=False).reshape(-1)
            inside = moc.contains(
                lon[offset:stop].index_select(0, point_indices),
                lat[offset:stop].index_select(0, point_indices),
            )
            chunk_out[point_indices] = inside
    return out


def _merge_optional_moc(left: MOC | None, right: MOC | None) -> MOC | None:
    if left is None:
        return right
    if right is None:
        return left
    if left is right:
        return left
    return left.union(right)


def _time_bounds(stmoc: STMOC) -> Tensor:
    if stmoc.n_segments == 0:
        return torch.empty(0, dtype=torch.int64)
    return torch.unique(
        torch.cat(
            [
                stmoc.time_start_us.detach().cpu(),
                stmoc.time_end_us.detach().cpu(),
            ]
        ),
        sorted=True,
    )


def _mocs_over_bounds(stmoc: STMOC, bounds: Tensor) -> list[MOC | None]:
    n_intervals = max(int(bounds.numel()) - 1, 0)
    if n_intervals == 0 or stmoc.n_segments == 0:
        return [None] * n_intervals
    starts = stmoc._time_start_sorted.detach().cpu()
    ends = stmoc._time_end_sorted.detach().cpu()
    order = stmoc._time_order.detach().cpu()
    if stmoc._time_non_overlapping:
        positions = torch.searchsorted(starts, bounds[:-1], right=True) - 1
        safe_positions = positions.clamp(min=0)
        valid = (positions >= 0) & (ends[safe_positions] >= bounds[1:])
        segment_indices = torch.where(valid, order[safe_positions], -1).tolist()
        return [None if index < 0 else stmoc.space[index] for index in segment_indices]

    start_positions = torch.searchsorted(bounds, starts).tolist()
    end_positions = torch.searchsorted(bounds, ends).tolist()
    segment_indices = order.tolist()
    tree_size = 1 << max(n_intervals - 1, 0).bit_length()
    tree: list[MOC | None] = [None] * (2 * tree_size)
    for start, end, segment in zip(
        start_positions, end_positions, segment_indices, strict=True
    ):
        left = start + tree_size
        right = end + tree_size
        moc = stmoc.space[segment]
        while left < right:
            if left & 1:
                tree[left] = _merge_optional_moc(tree[left], moc)
                left += 1
            if right & 1:
                right -= 1
                tree[right] = _merge_optional_moc(tree[right], moc)
            left >>= 1
            right >>= 1

    result: list[MOC | None] = [None] * n_intervals
    stack: list[tuple[int, MOC | None]] = [(1, None)]
    while stack:
        node, inherited = stack.pop()
        combined = _merge_optional_moc(inherited, tree[node])
        if node >= tree_size:
            interval = node - tree_size
            if interval < n_intervals:
                result[interval] = combined
            continue
        stack.append((2 * node + 1, combined))
        stack.append((2 * node, combined))
    return result


def _canonical_temporal_segments(
    stmoc: STMOC,
) -> tuple[Tensor, Tensor, tuple[MOC, ...]]:
    bounds = _time_bounds(stmoc)
    interval_mocs = _mocs_over_bounds(stmoc, bounds)
    bounds_list = bounds.tolist()
    starts: list[int] = []
    ends: list[int] = []
    space: list[MOC] = []
    for start, end, moc in zip(
        bounds_list[:-1], bounds_list[1:], interval_mocs, strict=True
    ):
        if moc is None or moc.uniq.numel() == 0:
            continue
        starts.append(start)
        ends.append(end)
        space.append(moc)
    return (
        torch.tensor(starts, dtype=torch.int64),
        torch.tensor(ends, dtype=torch.int64),
        tuple(space),
    )


def _moc_key(moc: MOC) -> tuple[int, ...]:
    return tuple(int(v) for v in moc.uniq.detach().cpu().tolist())


def _from_segments(
    segs: list[tuple[int, int, MOC]],
    *,
    time_depth: int,
    spatial_depth: int,
) -> STMOC:
    if not segs:
        return STMOC.empty(time_depth=time_depth, spatial_depth=spatial_depth)
    ordered = sorted(segs, key=lambda item: (item[0], item[1]))
    merged: list[tuple[int, int, MOC]] = []
    for start, end, moc in ordered:
        if moc.uniq.numel() == 0:
            continue
        if (
            merged
            and merged[-1][1] == start
            and (
                merged[-1][2] is moc
                or _moc_key(merged[-1][2]) == _moc_key(moc)
            )
        ):
            prev_s, _, prev_m = merged[-1]
            merged[-1] = (prev_s, end, prev_m)
        else:
            merged.append((start, end, moc))
    if not merged:
        return STMOC.empty(time_depth=time_depth, spatial_depth=spatial_depth)
    overlapping = any(merged[i][0] < merged[i - 1][1] for i in range(1, len(merged)))
    if overlapping:
        raw = STMOC(
            torch.tensor([s for s, _, _ in merged], dtype=torch.int64),
            torch.tensor([e for _, e, _ in merged], dtype=torch.int64),
            tuple(m for _, _, m in merged),
            time_depth,
            spatial_depth,
        )
        return _from_segments(
            _sweep_boolean(
                raw,
                STMOC.empty(time_depth=time_depth, spatial_depth=spatial_depth),
                op="union",
            ),
            time_depth=time_depth,
            spatial_depth=spatial_depth,
        )
    starts = torch.tensor([s for s, _, _ in merged], dtype=torch.int64)
    ends = torch.tensor([e for _, e, _ in merged], dtype=torch.int64)
    return STMOC(
        starts, ends, tuple(m for _, _, m in merged), time_depth, spatial_depth
    )


def _sweep_boolean(a: STMOC, b: STMOC, *, op: str) -> list[tuple[int, int, MOC]]:
    parts = [
        tensor.detach().cpu()
        for tensor in (
            a.time_start_us,
            a.time_end_us,
            b.time_start_us,
            b.time_end_us,
        )
        if tensor.numel() > 0
    ]
    if not parts:
        return []
    bounds = torch.unique(torch.cat(parts), sorted=True)
    if bounds.numel() < 2:
        return []
    a_mocs = _mocs_over_bounds(a, bounds)
    b_mocs = _mocs_over_bounds(b, bounds)
    bounds_list = bounds.tolist()
    out: list[tuple[int, int, MOC]] = []
    for lo, hi, acc_a, acc_b in zip(
        bounds_list[:-1], bounds_list[1:], a_mocs, b_mocs, strict=True
    ):
        if op == "union":
            moc = _merge_optional_moc(acc_a, acc_b)
        elif op == "intersection":
            moc = (
                None
                if acc_a is None or acc_b is None
                else acc_a.intersection(acc_b)
            )
        elif op == "difference":
            moc = (
                None
                if acc_a is None
                else acc_a if acc_b is None else acc_a.difference(acc_b)
            )
        else:
            raise ValueError(f"unknown STMOC boolean op: {op!r}")
        if moc is None or moc.uniq.numel() == 0:
            continue
        out.append((lo, hi, moc))
    return out


def _segments_from_range_stream(
    vals: Tensor, spatial_depth: int
) -> list[tuple[int, int, MOC]]:
    if vals.numel() % 2 != 0:
        raise ValueError("RANGE stream must contain [min,max) pairs")
    flag = int(_TIME_FLAG.item())
    segs: list[tuple[int, int, MOC]] = []
    t_start: int | None = None
    t_end: int | None = None
    space_ranges: list[tuple[int, int]] = []

    def _flush() -> None:
        nonlocal t_start, t_end, space_ranges
        if t_start is None or t_end is None:
            return
        moc = MOC._from_ranges(space_ranges, spatial_depth)
        segs.append((t_start, t_end, moc))
        t_start = t_end = None
        space_ranges = []

    flat = vals.detach().cpu().tolist()
    i = 0
    while i < len(flat):
        lo, hi = int(flat[i]), int(flat[i + 1])
        i += 2
        is_time = (lo & flag) == flag and (hi & flag) == flag
        if is_time:
            _flush()
            t_start = lo ^ flag
            t_end = hi ^ flag
        else:
            if t_start is None:
                raise ValueError(
                    "STMOC RANGE stream has spatial ranges before a time range"
                )
            space_ranges.append((lo, hi))
    _flush()
    return segs
