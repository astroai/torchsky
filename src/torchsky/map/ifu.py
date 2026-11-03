"""Torch-native physical IFU cube response planning and execution."""

from __future__ import annotations

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
import math
from typing import Any, Literal, cast

import torch
from torch import Tensor

from ..device import _as_tensor, resolve_device, _to
from ..domain import DomainSpec
from ..wcs.spectral import canonical_spectral_type, spectral_unit_scale
from .observation import (
    CubeResult,
    CubeTarget,
    IFUObservation,
)
from .operators.response import IndexedResponsePlan, apply_response_products
from .quality import DQPolicy


IFUKernel = Literal["drizzle", "emsm", "msm"]
ScaleModel = float | Tensor | Callable[[Tensor], Tensor | float]


def _spectral_factor(source_unit: str, source_type: str, target: CubeTarget) -> float:
    if canonical_spectral_type(source_type) != canonical_spectral_type(
        target.spectral_type
    ):
        raise ValueError(
            "IFU observation and target spectral coordinate types must agree"
        )
    source = spectral_unit_scale(target.spectral_type, source_unit)
    destination = spectral_unit_scale(target.spectral_type, target.spectral_unit)
    return source / destination


def _model_value(model: ScaleModel | None, wavelength: Tensor) -> Tensor | None:
    if model is None:
        return None
    value = model(wavelength) if callable(model) else model
    result = _as_tensor(value, device=wavelength.device, dtype=wavelength.dtype)
    if result.numel() != 1:
        result = torch.broadcast_to(result, wavelength.shape)
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError(
            "wavelength-dependent scale models must be finite and positive"
        )
    return result


# Upper bound on the elements of the (sample x candidate) intermediates one
# planning block materialises. Blocks are cut on sample boundaries, so a single
# sample with a larger candidate set still forms its own block.
_BLOCK_ELEMENTS = 1 << 21
# Samples whose footprints are mapped and bounded together.
_SAMPLE_BLOCK = 1 << 15
# Rounding resolution, in units of eps, of an aperture/pixel overlap fraction
# from ``_disk_polygon_area``: four edge terms of magnitude <= pi r^2 / 2, each
# a few ulp off. A pixel inside the aperture's bounding box that misses or only
# touches the disk has true overlap 0 but evaluates to a few eps of either sign,
# which differs between CPU and CUDA and, through a weighted mean, would decide
# whether the pixel is covered at all. Fractions at or below the bound are 0.
_DISK_FRACTION_ULPS = 64


def _geometry_dtype(device: torch.device) -> torch.dtype:
    return torch.float32 if device.type == "mps" else torch.float64


def _sample_blocks(cost: Tensor, budget: int) -> list[tuple[int, int]]:
    """Contiguous sample ranges whose summed ``cost`` stays within ``budget``."""
    count = int(cost.numel())
    if count == 0:
        return []
    cumulative = torch.cumsum(_to(cost, device="cpu", dtype=torch.int64), dim=0)
    blocks: list[tuple[int, int]] = []
    start = 0
    while start < count:
        base = int(cumulative[start - 1]) if start else 0
        stop = int(
            torch.searchsorted(cumulative, torch.tensor(base + int(budget)), right=True)
        )
        stop = min(count, max(stop, start + 1))
        blocks.append((start, stop))
        start = stop
    return blocks


def _enumerate_ranges(counts: Tensor) -> tuple[Tensor, Tensor]:
    """``(owner, local)`` for ``local in range(counts[owner])``, owner-major."""
    owner = torch.repeat_interleave(
        torch.arange(counts.numel(), device=counts.device), counts
    )
    starts = torch.cumsum(counts, dim=0) - counts
    local = torch.arange(owner.numel(), device=counts.device) - starts[owner]
    return owner, local


def _shoelace_area(x: Tensor, y: Tensor, count: Tensor) -> Tensor:
    """Unsigned areas of padded polygons ``(P, V)`` with ``count`` vertices each.

    The shoelace sum is accumulated vertex by vertex in polygon order, so the
    result does not depend on how many polygons are evaluated together.
    """
    rows = torch.arange(x.shape[0], device=x.device)
    total = torch.zeros(x.shape[0], device=x.device, dtype=x.dtype)
    for index in range(x.shape[1]):
        following = torch.where(index + 1 < count, index + 1, 0)
        term = x[:, index] * y[rows, following] - x[rows, following] * y[:, index]
        total = torch.where(index < count, total + term, total)
    return torch.where(count >= 3, 0.5 * total.abs(), torch.zeros_like(total))


def _clip_half_plane(
    x: Tensor,
    y: Tensor,
    count: Tensor,
    *,
    axis: int,
    bound: Tensor,
    keep_greater: bool,
) -> tuple[Tensor, Tensor, Tensor]:
    """Sutherland-Hodgman clip of padded polygons against ``coord >= / <= bound``."""
    polygons, width = x.shape
    if polygons == 0 or width == 0:
        return x, y, torch.zeros_like(count)
    rows = torch.arange(polygons, device=x.device)
    coordinate = x if axis == 0 else y
    last = (count - 1).clamp_min(0)
    out_x = x.new_zeros((polygons, 2 * width))
    out_y = y.new_zeros((polygons, 2 * width))
    emit = torch.zeros((polygons, 2 * width), device=x.device, dtype=torch.bool)
    for index in range(width):
        if index == 0:
            previous_x, previous_y = x[rows, last], y[rows, last]
            previous_c = coordinate[rows, last]
        else:
            previous_x, previous_y = x[:, index - 1], y[:, index - 1]
            previous_c = coordinate[:, index - 1]
        current_x, current_y = x[:, index], y[:, index]
        current_c = coordinate[:, index]
        if keep_greater:
            current_inside = current_c >= bound
            previous_inside = previous_c >= bound
        else:
            current_inside = current_c <= bound
            previous_inside = previous_c <= bound
        valid = index < count
        denominator = current_c - previous_c
        zero = denominator == 0.0
        fraction = torch.where(
            zero,
            torch.zeros_like(denominator),
            (bound - previous_c) / torch.where(zero, 1.0, denominator),
        )
        out_x[:, 2 * index] = previous_x + fraction * (current_x - previous_x)
        out_y[:, 2 * index] = previous_y + fraction * (current_y - previous_y)
        emit[:, 2 * index] = valid & (current_inside != previous_inside)
        out_x[:, 2 * index + 1] = current_x
        out_y[:, 2 * index + 1] = current_y
        emit[:, 2 * index + 1] = valid & current_inside
    new_count = emit.sum(dim=1)
    new_width = max(int(new_count.max()), 1)
    position = torch.cumsum(emit, dim=1) - 1
    row, column = torch.nonzero(emit, as_tuple=True)
    clipped_x = x.new_zeros((polygons, new_width))
    clipped_y = y.new_zeros((polygons, new_width))
    clipped_x[row, position[row, column]] = out_x[row, column]
    clipped_y[row, position[row, column]] = out_y[row, column]
    return clipped_x, clipped_y, new_count


def _pixel_overlap_areas(
    x: Tensor, y: Tensor, count: Tensor, row: Tensor, column: Tensor
) -> Tensor:
    """Exact overlap areas of padded polygons with the unit pixels they are paired with."""
    for axis, bound, keep_greater in (
        (0, column - 0.5, True),
        (0, column + 0.5, False),
        (1, row - 0.5, True),
        (1, row + 0.5, False),
    ):
        x, y, count = _clip_half_plane(
            x, y, count, axis=axis, bound=bound, keep_greater=keep_greater
        )
    return _shoelace_area(x, y, count)


def _disk_polygon_area(x: Tensor, y: Tensor, radius: Tensor) -> Tensor:
    """Signed area of ``polygon ∩ disk(0, radius)`` for closed polygons ``(P, V)``.

    Sum over edges ``AB`` of the exact area of the disk inside triangle
    ``(0, A, B)``: circular sectors outside the disk, a triangle on the chord.
    """
    radius = radius[:, None]
    ax, ay = x, y
    bx, by = torch.roll(x, -1, dims=1), torch.roll(y, -1, dims=1)
    dx, dy = bx - ax, by - ay
    a = dx * dx + dy * dy
    b = ax * dx + ay * dy
    c = ax * ax + ay * ay - radius * radius
    discriminant = b * b - a * c
    root = torch.sqrt(discriminant.clamp_min(0.0))
    safe_a = torch.where(a > 0, a, torch.ones_like(a))
    t0 = ((-b - root) / safe_a).clamp(0.0, 1.0)
    t1 = ((-b + root) / safe_a).clamp(0.0, 1.0)
    chord = (a > 0) & (discriminant > 0) & (t1 > t0)
    px, py = ax + t0 * dx, ay + t0 * dy
    qx, qy = ax + t1 * dx, ay + t1 * dy

    def sector(ux: Tensor, uy: Tensor, vx: Tensor, vy: Tensor) -> Tensor:
        return 0.5 * radius * radius * torch.atan2(ux * vy - uy * vx, ux * vx + uy * vy)

    inside = sector(ax, ay, px, py) + 0.5 * (px * qy - py * qx) + sector(qx, qy, bx, by)
    contribution = torch.where(chord, inside, sector(ax, ay, bx, by))
    contribution = torch.where(a > 0, contribution, torch.zeros_like(contribution))
    return contribution.sum(dim=1)


def _subdivide_boundaries(
    corners: Tensor,
    *,
    map_points: Callable[[Tensor], Tensor],
    tolerance: float,
    max_depth: int,
) -> tuple[Tensor, Tensor]:
    """Batched :func:`bounded_subdivide_boundary` for ``(B, V, 2)`` corner sets.

    Returns padded mapped boundaries ``(B, Vmax, 2)`` and their vertex counts;
    every boundary starts and ends on its first mapped corner, as in the scalar
    helper.
    """
    polygons, vertices, _ = corners.shape
    if vertices < 3:
        raise ValueError("bounded subdivision requires at least three corners")
    if not math.isfinite(float(tolerance)) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_depth) < 0:
        raise ValueError("max_depth must be non-negative")
    flat = corners.reshape(-1, 2)
    mapped_flat = map_points(flat)
    if mapped_flat.shape != flat.shape or not bool(torch.isfinite(mapped_flat).all()):
        raise RuntimeError("Bounded subdivision produced non-finite corner coordinates")
    mapped_corners = mapped_flat.reshape(polygons, vertices, 2)
    start = flat
    stop = torch.roll(corners, -1, dims=1).reshape(-1, 2)
    mapped_start = mapped_flat
    mapped_stop = torch.roll(mapped_corners, -1, dims=1).reshape(-1, 2)
    owner = torch.arange(polygons, device=corners.device).repeat_interleave(vertices)
    done = torch.zeros(owner.shape, device=corners.device, dtype=torch.bool)
    fractions = corners.new_tensor((0.25, 0.5, 0.75))
    depth = 0
    while True:
        active = torch.nonzero(~done, as_tuple=True)[0]
        if active.numel() == 0:
            break
        segment_start, segment_stop = start[active], stop[active]
        points = (
            segment_start[:, None, :]
            + fractions[None, :, None] * (segment_stop - segment_start)[:, None, :]
        )
        mapped = map_points(points.reshape(-1, 2))
        if mapped.shape != (points.shape[0] * 3, 2) or not bool(
            torch.isfinite(mapped).all()
        ):
            raise RuntimeError(
                "Bounded subdivision produced non-finite mapped coordinates"
            )
        mapped = mapped.reshape(-1, 3, 2)
        linear = (
            mapped_start[active][:, None, :]
            + fractions[None, :, None]
            * (mapped_stop[active] - mapped_start[active])[:, None, :]
        )
        error = torch.linalg.vector_norm(mapped - linear, dim=-1).amax(dim=1)
        accept = error.to(_geometry_dtype(error.device)) <= float(tolerance)
        if bool((~accept).any()) and depth >= int(max_depth):
            raise RuntimeError(
                "Could not resolve mapped boundary within the requested tolerance"
            )
        split = torch.zeros_like(done)
        split[active[~accept]] = True
        midpoint = torch.zeros_like(start)
        mapped_midpoint = torch.zeros_like(mapped_start)
        midpoint[active] = points[:, 1]
        mapped_midpoint[active] = mapped[:, 1]
        newly_done = torch.zeros_like(done)
        newly_done[active[accept]] = True
        counts = torch.where(split, 2, 1)
        parent, rank = _enumerate_ranges(counts)
        first = split[parent] & (rank == 0)
        second = split[parent] & (rank == 1)
        start = torch.where(second[:, None], midpoint[parent], start[parent])
        stop = torch.where(first[:, None], midpoint[parent], stop[parent])
        mapped_start = torch.where(
            second[:, None], mapped_midpoint[parent], mapped_start[parent]
        )
        mapped_stop = torch.where(
            first[:, None], mapped_midpoint[parent], mapped_stop[parent]
        )
        owner = owner[parent]
        done = done[parent] | newly_done[parent]
        depth += 1
    per_polygon = torch.bincount(owner, minlength=polygons)
    rank = (
        torch.arange(owner.numel(), device=owner.device)
        - (torch.cumsum(per_polygon, dim=0) - per_polygon)[owner]
    )
    width = int(per_polygon.max()) + 1
    boundary = mapped_corners.new_zeros((polygons, width, 2))
    boundary[:, 0] = mapped_corners[:, 0]
    boundary[owner, rank + 1] = mapped_stop
    return boundary, per_polygon + 1


def _angular_distance_arcsec(
    ra_deg: Tensor, dec_deg: Tensor, reference_ra_deg: Tensor, reference_dec_deg: Tensor
) -> Tensor:
    radians = math.pi / 180.0
    dec = dec_deg * radians
    reference_dec = reference_dec_deg * radians
    delta_dec = dec - reference_dec
    delta_ra = (
        torch.remainder((ra_deg - reference_ra_deg) * radians + math.pi, 2 * math.pi)
        - math.pi
    )
    haversine = (
        torch.sin(delta_dec / 2).square()
        + torch.cos(dec) * torch.cos(reference_dec) * torch.sin(delta_ra / 2).square()
    )
    # atan2 form: finite gradients at coincident points (asin(sqrt(.)) is not).
    haversine = haversine.clamp(0.0, 1.0)
    angle = 2.0 * torch.atan2(
        torch.sqrt(haversine), torch.sqrt((1.0 - haversine).clamp_min(0.0))
    )
    return angle * (180.0 / math.pi) * 3600.0


def _target_centers(
    target: CubeTarget, *, device: torch.device, dtype: torch.dtype
) -> tuple[Tensor, Tensor]:
    height, width = target.shape
    y, x = torch.meshgrid(
        torch.arange(height, device=device, dtype=dtype),
        torch.arange(width, device=device, dtype=dtype),
        indexing="ij",
    )
    wcs = target.spatial_wcs.to(device)
    # The two-argument form on a 2D WCS returns the (lon, lat) pair.
    return cast("tuple[Tensor, Tensor]", wcs.pixel_to_world(x, y))


def _target_pixel_scale_arcsec(
    target: CubeTarget, *, device: torch.device, dtype: torch.dtype
) -> float:
    height, width = target.shape
    x = torch.tensor(
        [0.5 * (width - 1), 0.5 * (width - 1) + 1.0], device=device, dtype=dtype
    )
    y = torch.tensor(
        [0.5 * (height - 1), 0.5 * (height - 1)], device=device, dtype=dtype
    )
    ra, dec = target.spatial_wcs.to(device).pixel_to_world(x, y)
    value = _angular_distance_arcsec(ra[1], dec[1], ra[0], dec[0])
    scale = float(value.detach().cpu())
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError(
            "CubeTarget spatial WCS must have a finite non-zero pixel scale"
        )
    return scale


@dataclass(frozen=True, slots=True)
class IFUCubeCovariance:
    """Matrix-free ``R diag(variance) R.T`` for one frozen IFU response."""

    plan: IndexedResponsePlan
    source_variance: Tensor = field(repr=False, compare=False)

    def __post_init__(self) -> None:
        self.plan.source.validate(self.source_variance, name="IFU source variance")
        if not bool(torch.isfinite(self.source_variance).all()) or bool(
            (self.source_variance < 0).any()
        ):
            raise ValueError("IFU source variance must be finite and non-negative")

    def apply_covariance(self, value: Tensor) -> Tensor:
        return self.plan.apply(self.source_variance * self.plan.adjoint(value))

    def diagonal(self) -> Tensor:
        return self.plan.apply_squared(self.source_variance)

    def query_variance(self, query: Tensor) -> Tensor:
        return torch.sum(query * self.apply_covariance(query))


@dataclass(frozen=True, slots=True)
class IFUCubeResponse:
    plan: IndexedResponsePlan
    science: Tensor = field(repr=False, compare=False)
    variance_components: Mapping[str, Tensor] = field(repr=False, compare=False)
    dq: Tensor = field(repr=False, compare=False)
    weight: Tensor = field(repr=False, compare=False)
    target: CubeTarget
    provenance: Mapping[str, Any]


def _observation_arrays(
    observations: Sequence[IFUObservation],
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, dict[str, Tensor], Tensor, Tensor, Tensor, list[int]]:
    component_names = sorted(
        {
            name
            for observation in observations
            for name in observation.variance_components
        }
    )
    science_parts: list[Tensor] = []
    component_parts: dict[str, list[Tensor]] = {name: [] for name in component_names}
    dq_parts: list[Tensor] = []
    usable_parts: list[Tensor] = []
    statistical_weight_parts: list[Tensor] = []
    offsets = [0]
    for observation in observations:
        science = _to(observation.sci, device=device, dtype=dtype).reshape(-1)
        science_parts.append(science)
        total_variance = torch.zeros_like(science)
        variance_valid = torch.ones_like(science, dtype=torch.bool)
        for name in component_names:
            if name in observation.variance_components:
                component = _to(
                    observation.variance_components[name], device=device, dtype=dtype
                ).reshape(-1)
                variance_valid &= torch.isfinite(component) & (component >= 0)
                safe = torch.where(
                    variance_valid, component, torch.zeros_like(component)
                )
            else:
                safe = torch.zeros_like(science)
            component_parts[name].append(safe)
            total_variance = total_variance + safe
        policy = observation.dq_policy or DQPolicy()
        if observation.dq is None:
            dq = torch.zeros(science.shape, device=device, dtype=torch.int64)
            dq_usable = torch.ones_like(science, dtype=torch.bool)
        else:
            raw_dq = observation.dq.to(device=device).reshape(-1)
            dq = policy.propagated(raw_dq)
            dq_usable = policy.usability_mask(raw_dq)
        dq_parts.append(dq)
        usable = torch.isfinite(science) & variance_valid & dq_usable
        usable_parts.append(usable)
        if observation.weight is None:
            if observation.variance_components:
                weight = torch.where(
                    variance_valid & (total_variance > 0),
                    total_variance.reciprocal(),
                    torch.zeros_like(total_variance),
                )
            else:
                weight = torch.ones_like(science)
        else:
            weight = torch.as_tensor(observation.weight).to(device=device).to(dtype)
            weight = torch.broadcast_to(weight, observation.sci.shape).reshape(-1)
        usable &= torch.isfinite(weight) & (weight > 0)
        statistical_weight_parts.append(
            torch.where(usable, weight, torch.zeros_like(weight))
        )
        offsets.append(offsets[-1] + science.numel())
    return (
        torch.cat(science_parts),
        {name: torch.cat(parts) for name, parts in component_parts.items()},
        torch.cat(dq_parts),
        torch.cat(usable_parts),
        torch.cat(statistical_weight_parts),
        offsets,
    )


def _sample_slice_ids(
    observation: IFUObservation, transform: Any, device: torch.device
) -> Tensor | None:
    slice_ids = observation.slice_id
    if slice_ids is None:
        slice_ids = getattr(transform, "slice_id_map", None)
    if slice_ids is None:
        return None
    return _to(torch.as_tensor(slice_ids), device=device, dtype=torch.int64)


def _require_grid_neighbours(slice_ids: Tensor, rows: Tensor, columns: Tensor) -> None:
    """Fail closed when a sample's detector-pixel footprint is not defined.

    The default drizzle footprint maps the detector pixel square through the
    detector -> world transform, which is only meaningful when the sample has a
    same-slice neighbour along both detector axes (otherwise, as for row-stacked
    fibre spectra, adjacent columns are unrelated sky positions).
    """
    height, width = slice_ids.shape
    own = slice_ids[rows, columns]
    for axis, (size, coordinate) in enumerate(((width, columns), (height, rows))):
        has_neighbour = torch.zeros_like(own, dtype=torch.bool)
        for step in (-1, 1):
            neighbour = coordinate + step
            inside = (neighbour >= 0) & (neighbour < size)
            clamped = neighbour.clamp(0, size - 1)
            other = (
                slice_ids[rows, clamped] if axis == 0 else slice_ids[clamped, columns]
            )
            has_neighbour |= inside & (other == own)
        if not bool(has_neighbour.all()):
            first = int(torch.nonzero(~has_neighbour)[0, 0])
            raise ValueError(
                "drizzle footprint undefined: detector sample (row "
                f"{int(rows[first])}, column {int(columns[first])}) has no "
                f"same-slice neighbour along detector {'xy'[axis]}, so its detector "
                "pixel has no sky footprint (e.g. row-stacked fibre spectra). Give "
                "IFUObservation.aperture_diameter_arcsec or detector_pixel_footprint"
            )


def _aperture_geometry(
    target_wcs: Any,
    ra: Tensor,
    dec: Tensor,
    *,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Target pixel centre and pixels-per-arcsec Jacobian at each sky position.

    Columns of the ``(N, 2, 2)`` Jacobian are the local east and north
    directions, from autograd through ``target_wcs.world_to_pixel``. Over an
    arcsecond-scale aperture the projection's second-order terms are below
    ``1e-10`` of a pixel, so the aperture maps to the ellipse ``J @ disk``.
    """
    with torch.enable_grad():
        ra_leaf = ra.detach().to(dtype).requires_grad_(True)
        dec_leaf = dec.detach().to(dtype).requires_grad_(True)
        x, y = target_wcs.world_to_pixel(ra_leaf, dec_leaf)
        rows = []
        for value in (x, y):
            grads = torch.autograd.grad(
                value.sum(), (ra_leaf, dec_leaf), retain_graph=True, allow_unused=True
            )
            if grads[0] is None or grads[1] is None:
                raise TypeError(
                    "aperture footprints need a target WCS differentiable in (ra, dec)"
                )
            rows.append(torch.stack(grads, dim=-1))
    cos_dec = torch.cos(torch.deg2rad(dec_leaf.detach()))
    scale = torch.stack(
        (1.0 / (3600.0 * cos_dec), torch.full_like(cos_dec, 1 / 3600.0)), dim=-1
    )
    jacobian = torch.stack(rows, dim=-2).to(dtype) * scale[:, None, :]
    return x.detach().to(dtype), y.detach().to(dtype), jacobian


def _spectral_extent(
    vertex_wavelength: Tensor,
    wavelength_center: Tensor,
    *,
    target_edges: Tensor,
    resolving_power: ScaleModel | None,
    spectral_pixfrac: float,
    geometry_dtype: torch.dtype,
) -> tuple[Tensor, Tensor, Tensor]:
    """Per-sample spectral interval ``[lower, upper]`` and a validity mask."""
    depth = target_edges.numel() - 1
    finite = torch.isfinite(vertex_wavelength)
    valid = finite.any(dim=1)
    infinity = torch.full_like(vertex_wavelength, math.inf)
    lower = torch.where(finite, vertex_wavelength, infinity).amin(dim=1)
    upper = torch.where(finite, vertex_wavelength, -infinity).amax(dim=1)
    lower = lower.to(geometry_dtype)
    upper = upper.to(geometry_dtype)
    spf = float(spectral_pixfrac)
    midpoint = 0.5 * (lower + upper)
    half = 0.5 * (upper - lower) * spf
    out_lower, out_upper = midpoint - half, midpoint + half
    degenerate = valid & ~(upper > lower)
    if bool(degenerate.any()):
        index = torch.nonzero(degenerate, as_tuple=True)[0]
        center = wavelength_center[index]
        power = _model_value(resolving_power, center)
        if power is None:
            nearest = torch.searchsorted(target_edges, center).clamp(1, depth)
            width = target_edges[nearest] - target_edges[nearest - 1]
        else:
            width = center.abs() / power
        width = torch.broadcast_to(width, center.shape)
        half_width = 0.5 * spf * width.to(geometry_dtype)
        center64 = center.to(geometry_dtype)
        out_lower[index] = center64 - half_width
        out_upper[index] = center64 + half_width
    return out_lower, out_upper, valid


def _spectral_range(
    lower: Tensor, upper: Tensor, target_edges: Tensor
) -> tuple[Tensor, Tensor]:
    depth = target_edges.numel() - 1
    dtype = target_edges.dtype
    z_start = (
        torch.searchsorted(target_edges, lower.to(dtype), right=True) - 1
    ).clamp_min(0)
    z_stop = (
        torch.searchsorted(target_edges, upper.to(dtype), right=False) + 1
    ).clamp_max(depth)
    return z_start, z_stop


def _pixel_range(low: Tensor, high: Tensor, size: int) -> tuple[Tensor, Tensor]:
    first = torch.ceil(low - 0.5).clamp_min(0)
    last = torch.floor(high + 0.5).clamp_max(size - 1)
    first = torch.where(torch.isfinite(first), first, torch.zeros_like(first))
    last = torch.where(torch.isfinite(last), last, torch.full_like(last, -1.0))
    return first.to(torch.int64), last.to(torch.int64)


def plan_ifu_cube_response(
    observations: Sequence[IFUObservation],
    target: CubeTarget,
    *,
    kernel: IFUKernel = "drizzle",
    pixfrac: float = 1.0,
    spectral_pixfrac: float = 1.0,
    spatial_roi: ScaleModel | None = None,
    spectral_roi: ScaleModel | None = None,
    resolving_power: ScaleModel | None = None,
    emsm_scale: float = 1.0,
    chunk_size: int = 65_536,
    max_response_entries: int | None = None,
    footprint_tolerance: float = 0.02,
    footprint_max_depth: int = 10,
    device: torch.device | str | None = None,
) -> IFUCubeResponse:
    """Freeze one sparse physical response used by all IFU cube products."""
    if not observations:
        raise ValueError("IFU cube building requires at least one observation")
    if kernel not in {"drizzle", "emsm", "msm"}:
        raise ValueError("physical IFU kernel must be drizzle, emsm, or msm")
    if not math.isfinite(float(pixfrac)) or not 0.0 < float(pixfrac) <= 1.0:
        raise ValueError("pixfrac must be finite in (0, 1]")
    if (
        not math.isfinite(float(spectral_pixfrac))
        or not 0.0 < float(spectral_pixfrac) <= 1.0
    ):
        raise ValueError("spectral_pixfrac must be finite in (0, 1]")
    if int(chunk_size) <= 0:
        raise ValueError("chunk_size must be positive")
    semantics = target.value_semantics
    if any(observation.value_semantics != semantics for observation in observations):
        raise ValueError("IFU observation and target value semantics must agree")
    resolved_device = resolve_device(device, observations)
    dtype = observations[0].sci.dtype
    output_dtype = (
        torch.float32
        if resolved_device.type == "mps" and dtype == torch.float64
        else dtype
    )
    if any(
        observation.sci.dtype != observations[0].sci.dtype
        for observation in observations
    ):
        raise ValueError("IFU observations must share a science dtype")
    # The response is planned in float64 where the device has it. MPS has no
    # float64 and float32 sky coordinates resolve only ~0.03 arcsec at RA ~150
    # deg, coarser than sub-spaxel footprints, so MPS plans on the CPU and the
    # frozen plan (and every product it applies to) moves to MPS.
    planning_device = (
        torch.device("cpu") if resolved_device.type == "mps" else resolved_device
    )
    science, components, dq, usable, statistical_weight, offsets = _observation_arrays(
        observations, device=planning_device, dtype=dtype
    )
    target_edges = target.spectral_bin_edges.to(device=planning_device).to(dtype)
    target_centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    depth, height, width = target_centers.numel(), *target.shape
    target_wcs = target.spatial_wcs.to(planning_device)
    target_ra, target_dec = _target_centers(target, device=planning_device, dtype=dtype)
    pixel_scale = _target_pixel_scale_arcsec(
        target, device=planning_device, dtype=dtype
    )

    source_parts: list[Tensor] = []
    target_parts: list[Tensor] = []
    geometry_parts: list[Tensor] = []
    contributor_parts: list[Tensor] = []
    entry_count = 0
    geometry_dtype = _geometry_dtype(planning_device)
    plane = height * width

    def append(
        source_index: Tensor,
        target_index: Tensor,
        coefficient: Tensor,
        contributor: int,
    ) -> None:
        nonlocal entry_count
        keep = (coefficient > 0.0) & torch.isfinite(coefficient)
        source_index = source_index[keep]
        if source_index.numel() == 0:
            return
        entry_count += int(source_index.numel())
        if max_response_entries is not None and entry_count > int(max_response_entries):
            raise MemoryError("IFU response exceeded max_response_entries")
        source_parts.append(source_index)
        target_parts.append(target_index[keep])
        geometry_parts.append(coefficient[keep].to(dtype))
        contributor_parts.append(torch.full_like(source_index, contributor))

    for observation_index, observation in enumerate(observations):
        transform = observation.detector_to_world
        if callable(getattr(transform, "to", None)):
            transform = transform.to(planning_device)
        rows, columns = observation.shape
        factor = _spectral_factor(
            observation.spectral_unit, observation.spectral_type, target
        )
        slice_ids = _sample_slice_ids(observation, transform, planning_device)
        local = torch.arange(rows * columns, device=planning_device)
        selected = usable[offsets[observation_index] + local]
        if slice_ids is not None:
            selected &= slice_ids.reshape(-1) >= 0
        local = local[selected]
        sample_row = local // columns
        sample_column = local % columns
        ra_center, dec_center, wavelength_center = transform.pixel_to_world(
            sample_column.to(dtype), sample_row.to(dtype)
        )
        wavelength_center = wavelength_center * factor
        finite = (
            torch.isfinite(ra_center)
            & torch.isfinite(dec_center)
            & torch.isfinite(wavelength_center)
        )
        local, sample_row, sample_column = (
            local[finite],
            sample_row[finite],
            sample_column[finite],
        )
        ra_center, dec_center, wavelength_center = (
            ra_center[finite],
            dec_center[finite],
            wavelength_center[finite],
        )
        source_index = offsets[observation_index] + local
        if kernel == "drizzle":
            supplied_footprint = observation.detector_pixel_footprint
            aperture = observation.aperture_diameter_arcsec
            world_footprint = (
                supplied_footprint is not None and supplied_footprint.shape[-1] == 3
            )
            if (
                supplied_footprint is None
                and aperture is None
                and slice_ids is not None
            ):
                _require_grid_neighbours(
                    slice_ids.reshape(rows, columns), sample_row, sample_column
                )
            if supplied_footprint is not None:
                supplied_footprint = supplied_footprint.to(device=planning_device)
            if aperture is not None:
                aperture = (
                    torch.as_tensor(aperture)
                    .to(device=planning_device)
                    .to(dtype=geometry_dtype)
                )
                aperture = torch.broadcast_to(aperture, (rows, columns)).reshape(-1)
            inset = 8.0 * torch.finfo(dtype).eps

            def map_spatial(points: Tensor) -> Tensor:
                ra, dec, _ = transform.pixel_to_world(points[:, 0], points[:, 1])
                x, y = target_wcs.world_to_pixel(ra, dec)
                return torch.stack((x, y), dim=-1).to(dtype=dtype)

            for block_start in range(0, int(local.numel()), _SAMPLE_BLOCK):
                block = slice(block_start, block_start + _SAMPLE_BLOCK)
                row_b, column_b = sample_row[block], sample_column[block]
                center_b = torch.stack((column_b, row_b), dim=-1).to(dtype)
                if world_footprint:
                    assert supplied_footprint is not None
                    world_vertices = supplied_footprint[row_b, column_b].to(dtype=dtype)
                    x_vertex, y_vertex = target_wcs.world_to_pixel(
                        world_vertices[..., 0], world_vertices[..., 1]
                    )
                    mapped = torch.stack((x_vertex, y_vertex), dim=-1).to(dtype=dtype)
                    vertex_count = torch.full(
                        (mapped.shape[0],),
                        mapped.shape[1],
                        device=planning_device,
                        dtype=torch.int64,
                    )
                    vertex_wavelength = world_vertices[..., 2] * factor
                else:
                    if supplied_footprint is None:
                        # (column -/+ 0.5) +/- inset, rounded like the scalar
                        # footprint the loop planner built from Python floats.
                        half = torch.tensor(
                            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
                            device=planning_device,
                            dtype=geometry_dtype,
                        )
                        detector_vertices = (
                            (center_b.to(geometry_dtype)[:, None, :] + half)
                            - torch.sign(half) * inset
                        ).to(dtype)
                    else:
                        detector_vertices = supplied_footprint[row_b, column_b].to(
                            dtype=dtype
                        )
                    detector_vertices = center_b[:, None, :] + (1.0 - inset) * (
                        detector_vertices - center_b[:, None, :]
                    )
                    _, _, vertex_wavelength = transform.pixel_to_world(
                        detector_vertices[..., 0], detector_vertices[..., 1]
                    )
                    vertex_wavelength = vertex_wavelength * factor
                    if aperture is None:
                        mapped, vertex_count = _subdivide_boundaries(
                            detector_vertices,
                            map_points=map_spatial,
                            tolerance=float(footprint_tolerance),
                            max_depth=int(footprint_max_depth),
                        )
                lower, upper, spectral_valid = _spectral_extent(
                    vertex_wavelength,
                    wavelength_center[block],
                    target_edges=target_edges,
                    resolving_power=resolving_power,
                    spectral_pixfrac=spectral_pixfrac,
                    geometry_dtype=geometry_dtype,
                )
                if aperture is None:
                    if pixfrac < 1.0:
                        for size in torch.unique(vertex_count).tolist():
                            members = torch.nonzero(
                                vertex_count == size, as_tuple=True
                            )[0]
                            polygon = mapped[members, :size]
                            spatial_center = polygon.mean(dim=1, keepdim=True)
                            mapped[members, :size] = spatial_center + float(pixfrac) * (
                                polygon - spatial_center
                            )
                    polygon = mapped.to(geometry_dtype)
                    poly_x, poly_y = polygon[..., 0], polygon[..., 1]
                    spatial_area = _shoelace_area(poly_x, poly_y, vertex_count)
                    vertex_valid = (
                        torch.arange(poly_x.shape[1], device=planning_device)[None]
                        < vertex_count[:, None]
                    )
                    infinity = torch.full_like(poly_x, math.inf)
                    low_x = torch.where(vertex_valid, poly_x, infinity).amin(dim=1)
                    high_x = torch.where(vertex_valid, poly_x, -infinity).amax(dim=1)
                    low_y = torch.where(vertex_valid, poly_y, infinity).amin(dim=1)
                    high_y = torch.where(vertex_valid, poly_y, -infinity).amax(dim=1)
                    keep = (spatial_area > 0.0) & spectral_valid
                else:
                    radius = 0.5 * float(pixfrac) * aperture[local[block]]
                    center_x, center_y, jacobian = _aperture_geometry(
                        target_wcs,
                        ra_center[block],
                        dec_center[block],
                        dtype=geometry_dtype,
                    )
                    determinant = (
                        jacobian[:, 0, 0] * jacobian[:, 1, 1]
                        - jacobian[:, 0, 1] * jacobian[:, 1, 0]
                    )
                    half_x = radius * torch.linalg.vector_norm(jacobian[:, 0], dim=-1)
                    half_y = radius * torch.linalg.vector_norm(jacobian[:, 1], dim=-1)
                    low_x, high_x = center_x - half_x, center_x + half_x
                    low_y, high_y = center_y - half_y, center_y + half_y
                    keep = (
                        spectral_valid
                        & torch.isfinite(determinant)
                        & (determinant != 0.0)
                        & torch.isfinite(center_x)
                        & torch.isfinite(center_y)
                    )
                x_first, x_last = _pixel_range(low_x, high_x, width)
                y_first, y_last = _pixel_range(low_y, high_y, height)
                z_start, z_stop = _spectral_range(lower, upper, target_edges)
                x_count = (x_last - x_first + 1).clamp_min(0)
                y_count = (y_last - y_first + 1).clamp_min(0)
                spatial_count = torch.where(keep, x_count * y_count, 0)
                spectral_count = torch.where(keep, (z_stop - z_start).clamp_min(0), 0)
                spatial_count = torch.where(spectral_count > 0, spatial_count, 0)
                spectral_count = torch.where(spatial_count > 0, spectral_count, 0)
                vertex_width = 4 if aperture is not None else int(mapped.shape[1])
                cost = spatial_count * (spectral_count + 4 * vertex_width)
                for sub_start, sub_stop in _sample_blocks(cost, _BLOCK_ELEMENTS):
                    sub = torch.arange(sub_start, sub_stop, device=planning_device)
                    sub = sub[spatial_count[sub] > 0]
                    if sub.numel() == 0:
                        continue
                    s_count, z_count = spatial_count[sub], spectral_count[sub]
                    # Spatial pairs, row-major over each sample's pixel box.
                    pair_owner, pair_local = _enumerate_ranges(s_count)
                    owner = sub[pair_owner]
                    pair_x = x_first[owner] + pair_local % x_count[owner]
                    pair_y = y_first[owner] + pair_local // x_count[owner]
                    if aperture is None:
                        overlap = _pixel_overlap_areas(
                            poly_x[owner],
                            poly_y[owner],
                            vertex_count[owner],
                            pair_y.to(geometry_dtype),
                            pair_x.to(geometry_dtype),
                        )
                        spatial_fraction = overlap / spatial_area[owner]
                    else:
                        corner_dx = pair_x.to(geometry_dtype) - center_x[owner]
                        corner_dy = pair_y.to(geometry_dtype) - center_y[owner]
                        square_x = corner_dx[:, None] + corner_dx.new_tensor(
                            (-0.5, 0.5, 0.5, -0.5)
                        )
                        square_y = corner_dy[:, None] + corner_dy.new_tensor(
                            (-0.5, -0.5, 0.5, 0.5)
                        )
                        inverse = torch.linalg.inv(jacobian[owner])
                        local_x = (
                            inverse[:, 0, 0, None] * square_x
                            + inverse[:, 0, 1, None] * square_y
                        )
                        local_y = (
                            inverse[:, 1, 0, None] * square_x
                            + inverse[:, 1, 1, None] * square_y
                        )
                        pair_radius = radius[owner]
                        spatial_fraction = _disk_polygon_area(
                            local_x, local_y, pair_radius
                        ).abs() / (math.pi * pair_radius * pair_radius)
                        resolution = (
                            _DISK_FRACTION_ULPS
                            * torch.finfo(spatial_fraction.dtype).eps
                        )
                        spatial_fraction = torch.where(
                            spatial_fraction > resolution,
                            spatial_fraction,
                            torch.zeros_like(spatial_fraction),
                        )
                    # Spectral pairs.
                    band_owner, band_local = _enumerate_ranges(z_count)
                    band_sample = sub[band_owner]
                    band = z_start[band_sample] + band_local
                    lower_b, upper_b = lower[band_sample], upper[band_sample]
                    edges = target_edges.to(geometry_dtype)
                    spectral_overlap = (
                        torch.minimum(upper_b, edges[band + 1])
                        - torch.maximum(lower_b, edges[band])
                    ).clamp_min(0.0)
                    spectral_width = upper_b - lower_b
                    # Entries z-major then row-major (y, x) within each sample.
                    entry_owner, entry_local = _enumerate_ranges(s_count * z_count)
                    s_entry = s_count[entry_owner]
                    spatial_pair = (torch.cumsum(s_count, 0) - s_count)[
                        entry_owner
                    ] + entry_local % s_entry
                    band_pair = (torch.cumsum(z_count, 0) - z_count)[
                        entry_owner
                    ] + entry_local // s_entry
                    coefficient = (
                        spatial_fraction[spatial_pair]
                        * spectral_overlap[band_pair]
                        / spectral_width[band_pair]
                    )
                    append(
                        source_index[block][sub[entry_owner]],
                        band[band_pair] * plane
                        + pair_y[spatial_pair] * width
                        + pair_x[spatial_pair],
                        coefficient,
                        observation_index,
                    )
        else:
            emsm_exponent = max(float(emsm_scale), 1.0e-12)
            block_size = max(1, _BLOCK_ELEMENTS // max(plane, depth, 1))
            for block_start in range(0, int(local.numel()), block_size):
                block = slice(block_start, block_start + block_size)
                center = wavelength_center[block]
                count = center.numel()
                spatial_limit = _model_value(spatial_roi, center)
                if spatial_limit is None:
                    spatial_limit = center.new_tensor(2.5 * pixel_scale)
                spatial_limit = torch.broadcast_to(spatial_limit, (count,))
                power = _model_value(resolving_power, center)
                spectral_limit = _model_value(spectral_roi, center)
                if spectral_limit is None:
                    if power is None:
                        nearest = torch.searchsorted(target_edges, center).clamp(
                            1, depth
                        )
                        spectral_limit = 2.5 * (
                            target_edges[nearest] - target_edges[nearest - 1]
                        )
                    else:
                        spectral_limit = 2.5 * center.abs() / power / 2.355
                spectral_limit = torch.broadcast_to(spectral_limit, (count,))
                separation = _angular_distance_arcsec(
                    target_ra[None],
                    target_dec[None],
                    ra_center[block][:, None, None],
                    dec_center[block][:, None, None],
                )
                spatial_mask = separation <= spatial_limit[:, None, None]
                spectral_distance = torch.abs(target_centers[None, :] - center[:, None])
                spectral_mask = spectral_distance <= spectral_limit[:, None]
                spatial_sigma = spatial_limit / 2.355
                spectral_sigma = (
                    torch.broadcast_to(center.abs() / power / 2.355, (count,))
                    if power is not None
                    else spectral_limit / 2.355
                )
                s_count_all = spatial_mask.sum(dim=(1, 2))
                z_count_all = spectral_mask.sum(dim=1)
                s_count_all = torch.where(z_count_all > 0, s_count_all, 0)
                z_count_all = torch.where(s_count_all > 0, z_count_all, 0)
                pixel_sample, pixel_y, pixel_x = torch.nonzero(
                    spatial_mask, as_tuple=True
                )
                band_sample_all, band_all = torch.nonzero(spectral_mask, as_tuple=True)
                pixel_offset = torch.cumsum(spatial_mask.sum(dim=(1, 2)), 0)
                pixel_offset = pixel_offset - spatial_mask.sum(dim=(1, 2))
                band_offset = torch.cumsum(spectral_mask.sum(dim=1), 0)
                band_offset = band_offset - spectral_mask.sum(dim=1)
                del pixel_sample, band_sample_all
                for sub_start, sub_stop in _sample_blocks(
                    s_count_all * z_count_all, _BLOCK_ELEMENTS
                ):
                    sub = torch.arange(sub_start, sub_stop, device=planning_device)
                    sub = sub[s_count_all[sub] > 0]
                    if sub.numel() == 0:
                        continue
                    s_count, z_count = s_count_all[sub], z_count_all[sub]
                    entry_owner, entry_local = _enumerate_ranges(s_count * z_count)
                    sample = sub[entry_owner]
                    s_entry = s_count[entry_owner]
                    pixel = pixel_offset[sample] + entry_local % s_entry
                    band_index = band_offset[sample] + entry_local // s_entry
                    y_index, x_index = pixel_y[pixel], pixel_x[pixel]
                    z_index = band_all[band_index]
                    distance2 = (
                        separation[sample, y_index, x_index] / spatial_sigma[sample]
                    ).square() + (
                        spectral_distance[sample, z_index] / spectral_sigma[sample]
                    ).square()
                    if kernel == "emsm":
                        kernel_weight = torch.exp(-0.5 * distance2 / emsm_exponent**2)
                    else:
                        kernel_weight = torch.pow(1.0 + distance2, -emsm_exponent)
                    total = torch.zeros(
                        sub.numel(), device=planning_device, dtype=kernel_weight.dtype
                    )
                    total.index_add_(0, entry_owner, kernel_weight)
                    kernel_weight = (
                        kernel_weight
                        / total.clamp_min(torch.finfo(dtype).tiny)[entry_owner]
                    )
                    append(
                        source_index[block][sample],
                        z_index * plane + y_index * width + x_index,
                        kernel_weight,
                        observation_index,
                    )

    def joined(parts: list[Tensor], dtype_: torch.dtype) -> Tensor:
        if not parts:
            return torch.zeros(0, device=planning_device, dtype=dtype_)
        return torch.cat(parts)

    source_index_tensor = joined(source_parts, torch.int64)
    target_index_tensor = joined(target_parts, torch.int64)
    geometry = joined(geometry_parts, dtype)
    contributors = joined(contributor_parts, torch.int64)
    if semantics == "surface_brightness" and geometry.numel() > 0:
        raw = geometry * statistical_weight[source_index_tensor]
        denominator = torch.zeros(
            depth * height * width, device=planning_device, dtype=dtype
        )
        denominator.scatter_add_(0, target_index_tensor, raw)
        coefficients = raw / denominator[target_index_tensor].clamp_min(
            torch.finfo(dtype).tiny
        )
        output_weight = denominator.reshape(depth, height, width)
    else:
        coefficients = geometry
        output_weight = torch.zeros(
            depth * height * width, device=planning_device, dtype=dtype
        )
        if geometry.numel() > 0:
            output_weight.scatter_add_(0, target_index_tensor, geometry)
        output_weight = output_weight.reshape(depth, height, width)
    if planning_device != resolved_device or output_dtype != dtype:
        source_index_tensor = source_index_tensor.to(resolved_device)
        target_index_tensor = target_index_tensor.to(resolved_device)
        contributors = contributors.to(resolved_device)
        dq = dq.to(resolved_device)
        coefficients = _to(coefficients, resolved_device, output_dtype)
        output_weight = _to(output_weight, resolved_device, output_dtype)
        science = _to(science, resolved_device, output_dtype)
        components = {
            name: _to(value, resolved_device, output_dtype)
            for name, value in components.items()
        }
    plan = IndexedResponsePlan(
        DomainSpec((science.numel(),), output_dtype),
        DomainSpec((depth, height, width), output_dtype),
        source_index_tensor,
        target_index_tensor,
        coefficients,
        contributor_ids=contributors,
        default_chunk_size=int(chunk_size),
    )
    provenance = {
        "kind": "physical-ifu-cube",
        "kernel": kernel,
        "value_semantics": semantics,
        "spectral_unit": target.spectral_unit,
        "spectral_type": target.spectral_type,
        "response_fingerprint": plan.fingerprint.digest,
        "response_entries": int(coefficients.numel()),
        "covariance_model": "independent-detector-components-linear-response",
        "observation_provenance": tuple(dict(item.provenance) for item in observations),
    }
    return IFUCubeResponse(
        plan=plan,
        science=science,
        variance_components=components,
        dq=dq,
        weight=output_weight,
        target=target,
        provenance=provenance,
    )


def build_ifu_cube(
    observations: Sequence[IFUObservation],
    target: CubeTarget,
    **plan_options: Any,
) -> CubeResult:
    """Execute one frozen physical IFU plan for science and all side products."""
    response = plan_ifu_cube_response(observations, target, **plan_options)
    products = apply_response_products(
        response.plan,
        response.science,
        variance_components=response.variance_components,
        dq=response.dq,
        dq_policy=DQPolicy(reject_bits=0, unknown_bit_policy="preserve"),
    )
    total_variance = None
    covariance = None
    inverse_variance = None
    if response.variance_components:
        total_source_variance = torch.stack(
            tuple(response.variance_components.values())
        ).sum(dim=0)
        covariance = IFUCubeCovariance(response.plan, total_source_variance)
        total_variance = covariance.diagonal()
        inverse_variance = torch.where(
            total_variance > 0,
            total_variance.reciprocal(),
            torch.zeros_like(total_variance),
        )
    band_metadata = target.band_metadata
    observation_count = products.nexp.any(dim=(-2, -1)).to(torch.int64)
    band_metadata.observation_count.copy_(observation_count.cpu())
    band_metadata.spectral_weight_sum.copy_(
        response.weight.sum(dim=(-2, -1)).cpu().to(torch.float64)
    )
    return CubeResult(
        spectral_axis=target.spectral_axis,
        band_metadata=band_metadata,
        data=products.science,
        weight=response.weight,
        header=target.header,
        spatial_wcs=target.spatial_wcs,
        inverse_variance=inverse_variance,
        correlation_factor=None,
        overlap_count=products.nexp.to(torch.int32),
        variance_components=products.variance_components,
        dq=products.dq,
        usability_mask=response.weight > 0,
        context=products.context,
        covariance=covariance,
        response_plan=response.plan,
        provenance=response.provenance,
        value_semantics=target.value_semantics,
    )


__all__ = [
    "IFUCubeCovariance",
    "IFUCubeResponse",
    "build_ifu_cube",
    "plan_ifu_cube_response",
]
