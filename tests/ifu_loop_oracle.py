"""Per-sample Python-loop IFU response planner kept as a parity oracle.

This is the ``plan_ifu_cube_response`` implementation that preceded the
vectorised planner (torchsky 0.5). Tests compare the vectorised planner with it
on small inputs; it is not part of the package.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
import math
from typing import Literal, cast

import torch
from torch import Tensor

from torchsky.device import resolve_device
from torchsky.domain import DomainSpec
from torchsky.wcs.spectral import canonical_spectral_type, spectral_unit_scale
from torchsky.map.observation import (
    CubeTarget,
    IFUObservation,
)
from torchsky.map.operators.response import IndexedResponsePlan
from torchsky.map.operators.subdivision import bounded_subdivide_boundary
from torchsky.map.quality import DQPolicy
from torchsky.map.ifu import IFUCubeResponse


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
    result = torch.as_tensor(value, device=wavelength.device, dtype=wavelength.dtype)
    if result.numel() != 1:
        result = torch.broadcast_to(result, wavelength.shape)
    if not bool(torch.isfinite(result).all()) or bool((result <= 0).any()):
        raise ValueError(
            "wavelength-dependent scale models must be finite and positive"
        )
    return result


def _polygon_area(vertices: list[tuple[float, float]]) -> float:
    if len(vertices) < 3:
        return 0.0
    return 0.5 * abs(
        sum(
            x0 * y1 - x1 * y0
            for (x0, y0), (x1, y1) in zip(
                vertices, vertices[1:] + vertices[:1], strict=True
            )
        )
    )


def _clip_axis(
    vertices: list[tuple[float, float]],
    *,
    axis: int,
    bound: float,
    keep_greater: bool,
) -> list[tuple[float, float]]:
    if not vertices:
        return []
    output: list[tuple[float, float]] = []
    previous = vertices[-1]
    previous_inside = (
        previous[axis] >= bound if keep_greater else previous[axis] <= bound
    )
    for current in vertices:
        current_inside = (
            current[axis] >= bound if keep_greater else current[axis] <= bound
        )
        if current_inside != previous_inside:
            denominator = current[axis] - previous[axis]
            fraction = (
                0.0 if denominator == 0.0 else (bound - previous[axis]) / denominator
            )
            point = (
                previous[0] + fraction * (current[0] - previous[0]),
                previous[1] + fraction * (current[1] - previous[1]),
            )
            output.append(point)
        if current_inside:
            output.append(current)
        previous = current
        previous_inside = current_inside
    return output


def _pixel_overlap(vertices: list[tuple[float, float]], row: int, column: int) -> float:
    clipped = vertices
    clipped = _clip_axis(clipped, axis=0, bound=column - 0.5, keep_greater=True)
    clipped = _clip_axis(clipped, axis=0, bound=column + 0.5, keep_greater=False)
    clipped = _clip_axis(clipped, axis=1, bound=row - 0.5, keep_greater=True)
    clipped = _clip_axis(clipped, axis=1, bound=row + 0.5, keep_greater=False)
    return _polygon_area(clipped)


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


def _default_footprint(
    row: int, column: int, *, device: torch.device, dtype: torch.dtype
) -> Tensor:
    inset = 8.0 * torch.finfo(dtype).eps
    return torch.tensor(
        [
            [column - 0.5 + inset, row - 0.5 + inset],
            [column + 0.5 - inset, row - 0.5 + inset],
            [column + 0.5 - inset, row + 0.5 - inset],
            [column - 0.5 + inset, row + 0.5 - inset],
        ],
        device=device,
        dtype=dtype,
    )


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
        science = observation.sci.to(device=device, dtype=dtype).reshape(-1)
        science_parts.append(science)
        total_variance = torch.zeros_like(science)
        variance_valid = torch.ones_like(science, dtype=torch.bool)
        for name in component_names:
            if name in observation.variance_components:
                component = (
                    observation.variance_components[name]
                    .to(device=device, dtype=dtype)
                    .reshape(-1)
                )
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
            weight = torch.as_tensor(observation.weight, device=device, dtype=dtype)
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


def loop_plan_ifu_cube_response(
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
    """Reference per-sample loop planner (pre-vectorisation behaviour)."""
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
    if resolved_device.type == "mps" and dtype == torch.float64:
        dtype = torch.float32
    if any(
        observation.sci.dtype != observations[0].sci.dtype
        for observation in observations
    ):
        raise ValueError("IFU observations must share a science dtype")
    science, components, dq, usable, statistical_weight, offsets = _observation_arrays(
        observations, device=resolved_device, dtype=dtype
    )
    target_edges = target.spectral_bin_edges.to(device=resolved_device, dtype=dtype)
    target_centers = 0.5 * (target_edges[:-1] + target_edges[1:])
    depth, height, width = target_centers.numel(), *target.shape
    target_wcs = target.spatial_wcs.to(resolved_device)
    target_ra, target_dec = _target_centers(target, device=resolved_device, dtype=dtype)
    pixel_scale = _target_pixel_scale_arcsec(
        target, device=resolved_device, dtype=dtype
    )

    source_entries: list[int] = []
    target_entries: list[int] = []
    geometry_entries: list[float] = []
    contributor_entries: list[int] = []

    def append(
        source_index: int, target_index: int, coefficient: float, contributor: int
    ) -> None:
        if coefficient <= 0.0 or not math.isfinite(coefficient):
            return
        source_entries.append(source_index)
        target_entries.append(target_index)
        geometry_entries.append(coefficient)
        contributor_entries.append(contributor)
        if max_response_entries is not None and len(source_entries) > int(
            max_response_entries
        ):
            raise MemoryError("IFU response exceeded max_response_entries")

    for observation_index, observation in enumerate(observations):
        transform = observation.detector_to_world
        if callable(getattr(transform, "to", None)):
            transform = transform.to(resolved_device)
        rows, columns = observation.shape
        factor = _spectral_factor(
            observation.spectral_unit, observation.spectral_type, target
        )
        slice_ids = observation.slice_id
        if slice_ids is None:
            slice_ids = getattr(transform, "slice_id_map", None)
        for row in range(rows):
            for column in range(columns):
                source_index = offsets[observation_index] + row * columns + column
                if not bool(usable[source_index].detach().cpu()):
                    continue
                center_x = torch.tensor(
                    float(column), device=resolved_device, dtype=dtype
                )
                center_y = torch.tensor(float(row), device=resolved_device, dtype=dtype)
                ra_center, dec_center, wavelength_center = transform.pixel_to_world(
                    center_x, center_y
                )
                wavelength_center = wavelength_center * factor
                if not bool(
                    (
                        torch.isfinite(ra_center)
                        & torch.isfinite(dec_center)
                        & torch.isfinite(wavelength_center)
                    )
                    .detach()
                    .cpu()
                ):
                    continue
                if slice_ids is not None and int(slice_ids[row, column]) < 0:
                    continue
                if kernel == "drizzle":
                    supplied_footprint = observation.detector_pixel_footprint
                    world_footprint = (
                        supplied_footprint is not None
                        and supplied_footprint.shape[-1] == 3
                    )
                    if world_footprint:
                        assert supplied_footprint is not None
                        world_vertices = supplied_footprint[row, column].to(
                            device=resolved_device, dtype=dtype
                        )
                        x_vertex, y_vertex = target_wcs.world_to_pixel(
                            world_vertices[:, 0], world_vertices[:, 1]
                        )
                        mapped = torch.stack((x_vertex, y_vertex), dim=-1).to(
                            dtype=dtype
                        )
                        vertex_wavelength = world_vertices[:, 2] * factor
                    else:
                        detector_vertices = (
                            _default_footprint(
                                row, column, device=resolved_device, dtype=dtype
                            )
                            if supplied_footprint is None
                            else supplied_footprint[row, column].to(
                                device=resolved_device, dtype=dtype
                            )
                        )
                        inset = 8.0 * torch.finfo(dtype).eps
                        detector_center = detector_vertices.new_tensor(
                            (float(column), float(row))
                        )
                        detector_vertices = detector_center + (1.0 - inset) * (
                            detector_vertices - detector_center
                        )

                        def map_spatial(points: Tensor) -> Tensor:
                            ra, dec, _ = transform.pixel_to_world(
                                points[:, 0], points[:, 1]
                            )
                            x, y = target_wcs.world_to_pixel(ra, dec)
                            return torch.stack((x, y), dim=-1).to(dtype=dtype)

                        mapped = bounded_subdivide_boundary(
                            tuple(detector_vertices),
                            map_points=map_spatial,
                            tolerance=float(footprint_tolerance),
                            max_depth=int(footprint_max_depth),
                        )
                        _, _, vertex_wavelength = transform.pixel_to_world(
                            detector_vertices[:, 0], detector_vertices[:, 1]
                        )
                        vertex_wavelength = vertex_wavelength * factor
                    if pixfrac < 1.0:
                        spatial_center = mapped.mean(dim=0, keepdim=True)
                        mapped = spatial_center + float(pixfrac) * (
                            mapped - spatial_center
                        )
                    polygon = [
                        (float(point[0]), float(point[1]))
                        for point in mapped.detach().cpu().tolist()
                    ]
                    spatial_area = _polygon_area(polygon)
                    if spatial_area <= 0.0:
                        continue
                    finite_wavelength = vertex_wavelength[
                        torch.isfinite(vertex_wavelength)
                    ]
                    if finite_wavelength.numel() == 0:
                        continue
                    lower = float(finite_wavelength.min().detach().cpu())
                    upper = float(finite_wavelength.max().detach().cpu())
                    if not upper > lower:
                        power = _model_value(resolving_power, wavelength_center)
                        if power is None:
                            nearest = torch.searchsorted(
                                target_edges, wavelength_center
                            ).clamp(1, depth)
                            spectral_width = (
                                target_edges[nearest] - target_edges[nearest - 1]
                            )
                        else:
                            spectral_width = wavelength_center.abs() / power
                        half = (
                            0.5
                            * float(spectral_pixfrac)
                            * float(spectral_width.detach().cpu())
                        )
                        lower = float(wavelength_center.detach().cpu()) - half
                        upper = float(wavelength_center.detach().cpu()) + half
                    else:
                        midpoint = 0.5 * (lower + upper)
                        half = 0.5 * (upper - lower) * float(spectral_pixfrac)
                        lower, upper = midpoint - half, midpoint + half
                    spectral_width = upper - lower
                    min_x = max(0, math.ceil(min(x for x, _ in polygon) - 0.5))
                    max_x = min(width - 1, math.floor(max(x for x, _ in polygon) + 0.5))
                    min_y = max(0, math.ceil(min(y for _, y in polygon) - 0.5))
                    max_y = min(
                        height - 1, math.floor(max(y for _, y in polygon) + 0.5)
                    )
                    z_start = max(
                        0,
                        int(
                            torch.searchsorted(
                                target_edges, target_edges.new_tensor(lower), right=True
                            ).item()
                        )
                        - 1,
                    )
                    z_stop = min(
                        depth,
                        int(
                            torch.searchsorted(
                                target_edges,
                                target_edges.new_tensor(upper),
                                right=False,
                            ).item()
                        )
                        + 1,
                    )
                    for z_index in range(z_start, z_stop):
                        spectral_overlap = max(
                            0.0,
                            min(upper, float(target_edges[z_index + 1]))
                            - max(lower, float(target_edges[z_index])),
                        )
                        if spectral_overlap == 0.0:
                            continue
                        for y_index in range(min_y, max_y + 1):
                            for x_index in range(min_x, max_x + 1):
                                area = _pixel_overlap(polygon, y_index, x_index)
                                fraction = (
                                    area
                                    / spatial_area
                                    * spectral_overlap
                                    / spectral_width
                                )
                                append(
                                    source_index,
                                    z_index * height * width
                                    + y_index * width
                                    + x_index,
                                    fraction,
                                    observation_index,
                                )
                else:
                    spatial_limit = _model_value(spatial_roi, wavelength_center)
                    if spatial_limit is None:
                        spatial_limit = wavelength_center.new_tensor(2.5 * pixel_scale)
                    power = _model_value(resolving_power, wavelength_center)
                    spectral_limit = _model_value(spectral_roi, wavelength_center)
                    if spectral_limit is None:
                        if power is None:
                            nearest = torch.searchsorted(
                                target_edges, wavelength_center
                            ).clamp(1, depth)
                            spectral_limit = 2.5 * (
                                target_edges[nearest] - target_edges[nearest - 1]
                            )
                        else:
                            spectral_limit = (
                                2.5 * wavelength_center.abs() / power / 2.355
                            )
                    separation = _angular_distance_arcsec(
                        target_ra, target_dec, ra_center, dec_center
                    )
                    spatial_mask = separation <= spatial_limit
                    spectral_distance = torch.abs(target_centers - wavelength_center)
                    spectral_mask = spectral_distance <= spectral_limit
                    candidates = spatial_mask[None] & spectral_mask[:, None, None]
                    z_index, y_index, x_index = torch.nonzero(candidates, as_tuple=True)
                    if z_index.numel() == 0:
                        continue
                    spatial_sigma = spatial_limit / 2.355
                    spectral_sigma = (
                        wavelength_center.abs() / power / 2.355
                        if power is not None
                        else spectral_limit / 2.355
                    )
                    distance2 = (
                        separation[y_index, x_index] / spatial_sigma
                    ).square() + (spectral_distance[z_index] / spectral_sigma).square()
                    if kernel == "emsm":
                        kernel_weight = torch.exp(
                            -0.5 * distance2 / max(float(emsm_scale), 1.0e-12) ** 2
                        )
                    else:
                        kernel_weight = torch.pow(
                            1.0 + distance2,
                            -max(float(emsm_scale), 1.0e-12),
                        )
                    kernel_weight = kernel_weight / kernel_weight.sum().clamp_min(
                        torch.finfo(dtype).tiny
                    )
                    for entry in range(z_index.numel()):
                        append(
                            source_index,
                            int(z_index[entry]) * height * width
                            + int(y_index[entry]) * width
                            + int(x_index[entry]),
                            float(kernel_weight[entry].detach().cpu()),
                            observation_index,
                        )

    source_index_tensor = torch.tensor(
        source_entries, device=resolved_device, dtype=torch.int64
    )
    target_index_tensor = torch.tensor(
        target_entries, device=resolved_device, dtype=torch.int64
    )
    geometry = torch.tensor(geometry_entries, device=resolved_device, dtype=dtype)
    contributors = torch.tensor(
        contributor_entries, device=resolved_device, dtype=torch.int64
    )
    if semantics == "surface_brightness" and geometry.numel() > 0:
        raw = geometry * statistical_weight[source_index_tensor]
        denominator = torch.zeros(
            depth * height * width, device=resolved_device, dtype=dtype
        )
        denominator.scatter_add_(0, target_index_tensor, raw)
        coefficients = raw / denominator[target_index_tensor].clamp_min(
            torch.finfo(dtype).tiny
        )
        output_weight = denominator.reshape(depth, height, width)
    else:
        coefficients = geometry
        output_weight = torch.zeros(
            depth * height * width, device=resolved_device, dtype=dtype
        )
        if geometry.numel() > 0:
            output_weight.scatter_add_(0, target_index_tensor, geometry)
        output_weight = output_weight.reshape(depth, height, width)
    plan = IndexedResponsePlan(
        DomainSpec((science.numel(),), dtype),
        DomainSpec((depth, height, width), dtype),
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
