"""Target tile caching and observation footprint helpers for image-plane coaddition."""

from __future__ import annotations

import math
import operator
from collections import OrderedDict
from collections.abc import Mapping
from typing import Any

import torch

from ..wcs import WCS

# Header probe for area scaling: recoverable failures when reading metadata only.
from .observation import (
    ImageObservation,
)
from .operators.config import (
    MapValueSemantics,
)
from .windowing import (
    _LARGE_TAN_WINDOWING_EDGE_SAMPLES,  # noqa: F401
    _LARGE_TAN_WINDOWING_MARGIN_PIXELS,  # noqa: F401
    _LARGE_TAN_WINDOWING_MIN_TARGET_PIXELS,  # noqa: F401
    _PreparedImagePlaneTarget,
)


from .calibration import (  # noqa: F401 - re-exported helpers
    _accumulate_variance_weight,
    _apply_observation_calibration,
    _context_provenance,
    _effective_observation_photometric_scale,
    _linear_inverse_variance,
    _load_observation_weight_map,
    _local_noise_correlation_factor,
    _observation_operator_provenance,
    _set_context_bit,
    _should_apply_surface_brightness_area_scaling,
)

from .image_plane_common import (
    _convert_world_between_wcs,
)
from .io_utils import _observation_header_for_area_scaling


_DEFAULT_TARGET_CACHE_CAPACITY = 0


def _validated_target_cache_capacity(capacity: int) -> int:
    try:
        resolved = operator.index(capacity)
    except TypeError as exc:
        raise TypeError("target_cache_capacity must be an integer") from exc
    if resolved < 0:
        raise ValueError("target_cache_capacity must be non-negative")
    return resolved


def _target_grid_cache_signature(
    header: Mapping[str, Any],
) -> tuple[tuple[str, str], ...]:
    return tuple(sorted((str(key), repr(value)) for key, value in header.items()))


class _PreparedTargetTileCache:
    def __init__(
        self,
        *,
        capacity: int,
        target_header: Mapping[str, Any],
        canvas_shape: tuple[int, int],
        device: torch.device,
        value_semantics: MapValueSemantics,
        representative_pixel_area_sr: float | None,
    ) -> None:
        self.capacity = _validated_target_cache_capacity(capacity)
        self._grid_signature = _target_grid_cache_signature(target_header)
        self._canvas_shape = (int(canvas_shape[0]), int(canvas_shape[1]))
        self._device = str(device)
        self._coordinate_dtype = str(
            torch.float32 if device.type == "mps" else torch.float64
        )
        self._value_semantics: MapValueSemantics = value_semantics
        self._representative_pixel_area_sr = representative_pixel_area_sr
        self._entries: OrderedDict[tuple[Any, ...], _PreparedImagePlaneTarget] = (
            OrderedDict()
        )
        self.hits = 0
        self.misses = 0
        self.evictions = 0
        self.admission_skips = 0

    def get_or_prepare(
        self,
        sub_header: dict,
        tile_shape: tuple[int, int],
        *,
        bounds: tuple[int, int, int, int],
    ) -> _PreparedImagePlaneTarget:
        include_area = self._value_semantics == "flux"
        quadrature_offsets = (
            (
                (-0.5 / math.sqrt(3.0), -0.5 / math.sqrt(3.0)),
                (-0.5 / math.sqrt(3.0), 0.5 / math.sqrt(3.0)),
                (0.5 / math.sqrt(3.0), -0.5 / math.sqrt(3.0)),
                (0.5 / math.sqrt(3.0), 0.5 / math.sqrt(3.0)),
            )
            if include_area
            else ()
        )
        key = (
            self._grid_signature,
            tuple(int(value) for value in bounds),
            self._canvas_shape,
            (int(tile_shape[0]), int(tile_shape[1])),
            self._device,
            self._coordinate_dtype,
            self._value_semantics,
            include_area,
            self._representative_pixel_area_sr,
            quadrature_offsets,
        )
        cached = self._entries.get(key)
        if cached is not None:
            self.hits += 1
            self._entries.move_to_end(key)
            return cached
        self.misses += 1
        prepared = _core_mod._prepare_canvas_tile(
            sub_header,
            tile_shape,
            device=torch.device(self._device),
            canvas_shape=self._canvas_shape,
            origin_yx=(bounds[0], bounds[2]),
            value_semantics=self._value_semantics,
            representative_pixel_area_sr=self._representative_pixel_area_sr,
        )
        if self.capacity == 0:
            return prepared
        if len(self._entries) >= self.capacity:
            self.admission_skips += 1
            return prepared
        self._entries[key] = prepared
        return prepared

    def stats(self) -> dict[str, int]:
        return {
            "capacity": self.capacity,
            "size": len(self._entries),
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "admission_skips": self.admission_skips,
        }


# Margin (target pixels) beyond the sampled observation outline inside which a
# tile is still processed.  Identical to the margin of the bounding-box test.
_FOOTPRINT_MARGIN_PIXELS = 8.0


class _ObservationFootprint:
    """Outline of an observation in target pixel coordinates.

    ``bbox`` is the bounding box of the outline sampled by ``_obs_target_pixel_bbox``
    (margin included).  ``hull`` is the convex hull of the same sample points.  A
    tile is skipped when it lies outside the bounding box, or outside the hull
    grown by the same margin: in both cases the sampler assigns zero weight to
    every pixel of the tile, so skipping it leaves the accumulators unchanged.
    The hull test removes the empty corners of the bounding box of a rotated
    exposure.
    """

    __slots__ = ("bbox", "hull", "margin")

    def __init__(
        self,
        bbox: tuple[float, float, float, float],
        hull: list[tuple[float, float]],
        margin: float,
    ) -> None:
        self.bbox = bbox
        self.hull = hull
        self.margin = margin

    def intersects(self, y0: int, y1: int, x0: int, x1: int) -> bool:
        ymin, ymax, xmin, xmax = self.bbox
        if y1 <= ymin or y0 >= ymax or x1 <= xmin or x0 >= xmax:
            return False
        hull = self.hull
        if len(hull) < 3:
            return True
        # Tile pixel centres span [x0, x1 - 1] x [y0, y1 - 1]; grow by margin + 1.
        grow = self.margin + 1.0
        rx0, rx1 = x0 - grow, x1 - 1 + grow
        ry0, ry1 = y0 - grow, y1 - 1 + grow
        # Separating-axis test between the rectangle and the convex hull.  The
        # rectangle axes are covered by the hull's own extent (bbox above is the
        # sampled extent, the hull has the same extent), so test the hull edges.
        corners = ((rx0, ry0), (rx1, ry0), (rx1, ry1), (rx0, ry1))
        n = len(hull)
        for i in range(n):
            ax, ay = hull[i]
            bx, by = hull[(i + 1) % n]
            nx, ny = by - ay, ax - bx  # outward normal for a counter-clockwise hull
            edge = nx * ax + ny * ay
            if all(nx * cx + ny * cy > edge for cx, cy in corners):
                return False
        return True


def _convex_hull(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Counter-clockwise convex hull (Andrew's monotone chain)."""
    pts = sorted(set(points))
    if len(pts) <= 2:
        return pts

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for pt in pts:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], pt) <= 0.0:
            lower.pop()
        lower.append(pt)
    upper: list[tuple[float, float]] = []
    for pt in reversed(pts):
        while len(upper) >= 2 and cross(upper[-2], upper[-1], pt) <= 0.0:
            upper.pop()
        upper.append(pt)
    return lower[:-1] + upper[:-1]


def _obs_target_footprint(
    source_wcs: WCS,
    source_shape: tuple[int, int],
    full_target_wcs: WCS,
    *,
    margin: float = _FOOTPRINT_MARGIN_PIXELS,
) -> _ObservationFootprint | None:
    """Footprint of a source image on the full target grid, or None if unbounded."""
    hs, ws = int(source_shape[0]), int(source_shape[1])
    if hs <= 0 or ws <= 0:
        return _ObservationFootprint((0.0, -1.0, 0.0, -1.0), [], margin)
    try:
        t = torch.linspace(
            0.0, 1.0, 9, dtype=source_wcs.crval.dtype, device=source_wcs.device
        )
        x0, x1 = -0.5, float(ws) - 0.5
        y0, y1 = -0.5, float(hs) - 0.5
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
        ra, dec = source_wcs.pixel_to_world(xs, ys)
        ra, dec = _convert_world_between_wcs(ra, dec, source_wcs, full_target_wcs)
        tx, ty = full_target_wcs.world_to_pixel(ra, dec)
        if not bool((torch.isfinite(tx) & torch.isfinite(ty)).all()):
            return None
        txl = [float(v) for v in tx.tolist()]
        tyl = [float(v) for v in ty.tolist()]
    except Exception:  # noqa: BLE001 - same fallback as _obs_target_pixel_bbox
        return None
    bbox = (min(tyl) - margin, max(tyl) + margin, min(txl) - margin, max(txl) + margin)
    return _ObservationFootprint(bbox, _convex_hull(list(zip(txl, tyl))), margin)


def _observation_footprint_from_header(
    observation: ImageObservation,
    cached_header: dict | None,
    full_target_wcs: WCS,
) -> _ObservationFootprint | None:
    """Footprint from the observation header alone (no pixel read)."""
    try:
        header = _observation_header_for_area_scaling(
            observation, cached_header=cached_header
        )
        shape = (int(header.get("NAXIS2", 0)), int(header.get("NAXIS1", 0)))
        if shape[0] <= 0 or shape[1] <= 0:
            return None
        return _obs_target_footprint(WCS(header), shape, full_target_wcs)
    except Exception:  # noqa: BLE001 - an unreadable header disables the skip test
        return None


# Late-bound so monkeypatching ``image_plane_coadd`` attributes still reaches this code.
from . import image_plane_coadd as _core_mod  # noqa: E402
