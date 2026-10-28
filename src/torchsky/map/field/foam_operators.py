"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from .foam_geometry import (
    _cell_clip_polygon,
    _polygon_disk_intersection_area_moment,
    _polygon_signed_area,
    _validate_csr,
    _validate_sites_and_radii,
    polygon_disk_candidate_csr,
    sparse_cech_graph,
)
from .foam_solvers import PowerPixelOperator

from functools import lru_cache
import math
from typing import Any, Literal

import torch
from torch import Tensor

def bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    candidate_indptr: Tensor | None = None,
    candidate_indices: Tensor | None = None,
    neighbor_indptr: Tensor | None = None,
    neighbor_indices: Tensor | None = None,
    max_candidate_pairs: int = 1_000_000,
) -> PowerPixelOperator:
    """Build an exact sparse operator from bounded power cells to pixels.

    Pixels are finite convex polygons with shape ``(P, V, 2)``. Candidate CSR
    may conservatively include zero-area disk/pixel pairs; those are removed
    after exact radical-half-plane clipping and analytic disk/polygon
    integration. Neighbor CSR may use :func:`sparse_cech_graph`; omitted
    neighbors use every site and are intended only as a small correctness path.
    Curved WCS edges must first be adaptively tessellated to an error budget;
    that controlled approximation is intentionally outside this exact
    straight-edge operator and must not be reported as analytic pixel area.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("bounded_power_pixel_operator requires 2-D sites")
    if (
        not pixel_polygons.is_floating_point()
        or pixel_polygons.ndim != 3
        or pixel_polygons.shape[-1] != 2
        or pixel_polygons.shape[1] < 3
    ):
        raise ValueError("pixel_polygons must have floating shape (P, V>=3, 2)")
    if pixel_polygons.device != sites.device or pixel_polygons.dtype != sites.dtype:
        raise ValueError("pixels, sites, and radii must share dtype and device")
    if not bool(torch.isfinite(pixel_polygons).all()):
        raise ValueError("pixel polygons must be finite")
    signed_pixel_areas = (
        torch.stack([_polygon_signed_area(polygon) for polygon in pixel_polygons])
        if pixel_polygons.shape[0]
        else pixel_polygons.new_empty((0,))
    )
    if pixel_polygons.shape[0]:
        if bool(torch.any(torch.abs(signed_pixel_areas) == 0)):
            raise ValueError("pixel polygons must have non-zero area")
    edge_a = torch.roll(pixel_polygons, shifts=-1, dims=1) - pixel_polygons
    edge_b = torch.roll(edge_a, shifts=-1, dims=1)
    turns = edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0]
    positive = turns > 0
    negative = turns < 0
    if bool(torch.any(positive.any(dim=1) & negative.any(dim=1))):
        raise ValueError("pixel polygons must be convex and consistently ordered")

    pixel_count = pixel_polygons.shape[0]
    cell_count = sites.shape[0]
    candidate_limit = int(max_candidate_pairs)
    if candidate_limit <= 0:
        raise ValueError("max_candidate_pairs must be positive")
    if (candidate_indptr is None) != (candidate_indices is None):
        raise ValueError(
            "candidate_indptr and candidate_indices must be supplied together"
        )
    if candidate_indptr is None:
        candidate_count = pixel_count * cell_count
        if candidate_count > candidate_limit:
            raise ValueError(
                f"dense fallback needs {candidate_count} candidate pairs, exceeding "
                f"max_candidate_pairs={candidate_limit}; supply sparse candidate CSR"
            )
        candidate_indptr = (
            torch.arange(
                pixel_count + 1,
                dtype=torch.long,
                device=sites.device,
            )
            * cell_count
        )
        candidate_indices = torch.arange(
            cell_count,
            dtype=torch.long,
            device=sites.device,
        ).repeat(pixel_count)
    assert candidate_indices is not None
    _validate_csr(
        candidate_indptr,
        candidate_indices,
        rows=pixel_count,
        columns=cell_count,
        name="candidate",
        device=sites.device,
    )
    if candidate_indices.numel() > candidate_limit:
        raise ValueError("candidate CSR exceeds max_candidate_pairs")

    if (neighbor_indptr is None) != (neighbor_indices is None):
        raise ValueError(
            "neighbor_indptr and neighbor_indices must be supplied together"
        )
    if neighbor_indptr is not None:
        assert neighbor_indices is not None
        _validate_csr(
            neighbor_indptr,
            neighbor_indices,
            rows=cell_count,
            columns=cell_count,
            name="neighbor",
            device=sites.device,
        )
    all_cells = torch.arange(cell_count, dtype=torch.long, device=sites.device)

    def neighbors(cell: int) -> Tensor:
        if neighbor_indptr is None:
            return all_cells
        assert neighbor_indices is not None
        return neighbor_indices[neighbor_indptr[cell] : neighbor_indptr[cell + 1]]

    active_cells = torch.unique(candidate_indices, sorted=True)
    cell_areas = sites.new_zeros((cell_count,))
    cell_centered_first_moments = sites.new_zeros((cell_count, 2))
    for raw_cell in active_cells:
        cell = int(raw_cell.item())
        radius = radii[cell]
        lower = sites[cell] - radius
        upper = sites[cell] + radius
        bounds = torch.stack(
            (
                lower,
                torch.stack((upper[0], lower[1])),
                upper,
                torch.stack((lower[0], upper[1])),
            )
        )
        cell_polygon = _cell_clip_polygon(
            bounds,
            cell=cell,
            sites=sites,
            radii=radii,
            neighbors=neighbors(cell),
        )
        cell_area, cell_first_moment = _polygon_disk_intersection_area_moment(
            cell_polygon,
            sites[cell],
            radius,
        )
        cell_areas[cell] = cell_area
        cell_centered_first_moments[cell] = cell_first_moment - sites[cell] * cell_area

    kept_pixels: list[int] = []
    kept_cells: list[int] = []
    kept_areas: list[Tensor] = []
    kept_centered_first_moments: list[Tensor] = []
    row_counts: list[int] = []
    for pixel in range(pixel_count):
        row = torch.unique(
            candidate_indices[candidate_indptr[pixel] : candidate_indptr[pixel + 1]],
            sorted=True,
        )
        before = len(kept_areas)
        for raw_cell in row:
            cell = int(raw_cell.item())
            clipped = _cell_clip_polygon(
                pixel_polygons[pixel],
                cell=cell,
                sites=sites,
                radii=radii,
                neighbors=neighbors(cell),
            )
            area, first_moment = _polygon_disk_intersection_area_moment(
                clipped,
                sites[cell],
                radii[cell],
            )
            if clipped.shape[0] >= 3:
                squared_distance = torch.sum(
                    (clipped - sites[cell]) ** 2,
                    dim=1,
                )
                containment_tolerance = (
                    64.0
                    * torch.finfo(area.dtype).eps
                    * (radii[cell] * radii[cell] + torch.max(squared_distance) + 1.0)
                )
                fully_contained = bool(
                    torch.all(
                        squared_distance
                        <= radii[cell] * radii[cell] + containment_tolerance
                    )
                )
            else:
                fully_contained = False
            area_tolerance = (
                64.0
                * torch.finfo(area.dtype).eps
                * (
                    torch.abs(signed_pixel_areas[pixel])
                    + math.pi * radii[cell] * radii[cell]
                )
            )
            pixel_tolerance = (
                64.0
                * torch.finfo(area.dtype).eps
                * torch.abs(signed_pixel_areas[pixel])
            )
            if bool(
                area > area_tolerance or (fully_contained and area > pixel_tolerance)
            ):
                kept_pixels.append(pixel)
                kept_cells.append(cell)
                kept_areas.append(area)
                kept_centered_first_moments.append(first_moment - sites[cell] * area)
        row_counts.append(len(kept_areas) - before)

    indices = torch.tensor(kept_cells, dtype=torch.long, device=sites.device)
    pixel_ids = torch.tensor(kept_pixels, dtype=torch.long, device=sites.device)
    areas = torch.stack(kept_areas) if kept_areas else sites.new_empty((0,))
    centered_first_moments = (
        torch.stack(kept_centered_first_moments)
        if kept_centered_first_moments
        else sites.new_empty((0, 2))
    )
    counts = torch.tensor(row_counts, dtype=torch.long, device=sites.device)
    indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(counts, dim=0),
        )
    )
    order = torch.argsort(pixel_ids, stable=True)
    order = order[torch.argsort(indices[order], stable=True)]
    transpose_indices = pixel_ids[order]
    transpose_areas = areas[order]
    transpose_counts = torch.bincount(indices, minlength=cell_count)
    transpose_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(transpose_counts, dim=0),
        )
    )
    return PowerPixelOperator(
        indptr=indptr,
        indices=indices,
        pixel_ids=pixel_ids,
        areas=areas,
        centered_first_moments=centered_first_moments,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        cell_centered_first_moments=cell_centered_first_moments,
        pixel_count=pixel_count,
        cell_count=cell_count,
        candidate_pairs=int(candidate_indices.numel()),
    )


def _padded_cell_neighbors(
    cell_ids: Tensor,
    *,
    neighbor_indptr: Tensor | None,
    neighbor_indices: Tensor | None,
    cell_count: int,
    max_neighbors: int,
) -> Tensor:
    if cell_ids.numel() == 0:
        return cell_ids.new_empty((0, max_neighbors))
    if neighbor_indptr is None:
        return torch.arange(
            cell_count,
            dtype=torch.long,
            device=cell_ids.device,
        ).expand(cell_ids.shape[0], -1)
    assert neighbor_indices is not None
    counts = neighbor_indptr[cell_ids + 1] - neighbor_indptr[cell_ids]
    rows = torch.repeat_interleave(
        torch.arange(cell_ids.shape[0], device=cell_ids.device),
        counts,
    )
    starts = torch.repeat_interleave(neighbor_indptr[cell_ids], counts)
    offsets = torch.arange(rows.shape[0], device=cell_ids.device) - (
        torch.repeat_interleave(torch.cumsum(counts, 0) - counts, counts)
    )
    padded = torch.full(
        (cell_ids.shape[0], max_neighbors),
        -1,
        dtype=torch.long,
        device=cell_ids.device,
    )
    padded[rows, offsets] = neighbor_indices[starts + offsets]
    return padded


def _batched_clip_power_cells(
    polygons: Tensor,
    cell_ids: Tensor,
    padded_neighbors: Tensor,
    sites: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    """Clip equal-vertex-count polygons without candidate-wise synchronization."""
    batch, initial_vertices, _ = polygons.shape
    capacity = initial_vertices + padded_neighbors.shape[1]
    vertices = polygons.new_zeros((batch, capacity, 2))
    vertices[:, :initial_vertices] = polygons
    counts = torch.full(
        (batch,),
        initial_vertices,
        dtype=torch.long,
        device=polygons.device,
    )
    positions = torch.arange(capacity, device=polygons.device).expand(batch, -1)
    cell_sites = sites[cell_ids]
    cell_radii = radii[cell_ids]

    for plane in range(padded_neighbors.shape[1]):
        neighbor_ids = padded_neighbors[:, plane]
        present = neighbor_ids >= 0
        safe_neighbors = neighbor_ids.clamp_min(0)
        neighbor_sites = sites[safe_neighbors]
        neighbor_radii = radii[safe_neighbors]
        normals = neighbor_sites - cell_sites
        offsets = 0.5 * (
            torch.sum(neighbor_sites * neighbor_sites, dim=1)
            - torch.sum(cell_sites * cell_sites, dim=1)
            + cell_radii * cell_radii
            - neighbor_radii * neighbor_radii
        )
        coincident = torch.all(normals == 0, dim=1)
        dominated = (
            present
            & coincident
            & ((offsets < 0) | ((offsets == 0) & (neighbor_ids < cell_ids)))
        )
        active = present & ~coincident & (counts > 0)

        safe_counts = counts.clamp_min(1)
        following = torch.remainder(positions + 1, safe_counts[:, None])
        stop = torch.gather(vertices, 1, following[..., None].expand(-1, -1, 2))
        start_values = (
            torch.sum(vertices * normals[:, None, :], dim=2) - offsets[:, None]
        )
        stop_values = torch.sum(stop * normals[:, None, :], dim=2) - offsets[:, None]
        vertex_scale = torch.max(
            torch.linalg.vector_norm(vertices, dim=2), dim=1
        ).values
        tolerance = (
            64.0
            * torch.finfo(vertices.dtype).eps
            * (
                torch.abs(offsets)
                + torch.linalg.vector_norm(normals, dim=1) * vertex_scale
                + 1.0
            )
        )
        valid_edges = positions < counts[:, None]
        start_inside = start_values <= tolerance[:, None]
        stop_inside = stop_values <= tolerance[:, None]
        emit_start = valid_edges & start_inside
        emit_intersection = valid_edges & (start_inside != stop_inside)
        denominator = start_values - stop_values
        fraction = torch.where(
            emit_intersection,
            start_values
            / torch.where(
                denominator != 0,
                denominator,
                torch.ones_like(denominator),
            ),
            torch.zeros_like(start_values),
        ).clamp(0.0, 1.0)
        intersection = vertices + fraction[..., None] * (stop - vertices)
        emitted = torch.stack((vertices, intersection), dim=2).reshape(
            batch,
            2 * capacity,
            2,
        )
        emit = torch.stack((emit_start, emit_intersection), dim=2).reshape(
            batch,
            2 * capacity,
        )
        destinations = (torch.cumsum(emit, dim=1) - 1).clamp_min(0)
        clipped = vertices.new_zeros(vertices.shape)
        clipped.scatter_add_(
            1,
            destinations[..., None].expand(-1, -1, 2),
            emitted * emit[..., None],
        )
        clipped_counts = torch.sum(emit, dim=1)
        vertices = torch.where(active[:, None, None], clipped, vertices)
        counts = torch.where(active, clipped_counts, counts)
        counts = torch.where(dominated, torch.zeros_like(counts), counts)
    return vertices, counts


def _batched_polygon_disk_area_moment(
    vertices: Tensor,
    counts: Tensor,
    centers: Tensor,
    radii: Tensor,
) -> tuple[Tensor, Tensor]:
    """Analytic polygon/disk areas and global first moments for a batch."""
    batch, capacity, _ = vertices.shape
    positions = torch.arange(capacity, device=vertices.device).expand(batch, -1)
    safe_counts = counts.clamp_min(1)
    following = torch.remainder(positions + 1, safe_counts[:, None])
    start = vertices - centers[:, None, :]
    stop = (
        torch.gather(vertices, 1, following[..., None].expand(-1, -1, 2))
        - centers[:, None, :]
    )
    delta = stop - start
    quadratic = torch.sum(delta * delta, dim=2)
    safe_quadratic = torch.where(
        quadratic > 0,
        quadratic,
        torch.ones_like(quadratic),
    )
    linear = 2.0 * torch.sum(start * delta, dim=2)
    constant = torch.sum(start * start, dim=2) - radii[:, None] ** 2
    discriminant = linear * linear - 4.0 * quadratic * constant
    tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (torch.abs(linear * linear) + torch.abs(4.0 * quadratic * constant) + 1.0)
    )
    root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
    lower_root = (-linear - root) / (2.0 * safe_quadratic)
    upper_root = (-linear + root) / (2.0 * safe_quadratic)
    roots_exist = discriminant >= -tolerance
    lower_valid = roots_exist & (lower_root > 0) & (lower_root < 1)
    upper_valid = roots_exist & (upper_root > 0) & (upper_root < 1)
    parameters = torch.stack(
        (
            torch.zeros_like(lower_root),
            torch.where(lower_valid, lower_root, torch.ones_like(lower_root)),
            torch.where(upper_valid, upper_root, torch.ones_like(upper_root)),
            torch.ones_like(lower_root),
        ),
        dim=2,
    )
    parameters = torch.sort(parameters, dim=2).values
    lower = parameters[..., :-1]
    upper = parameters[..., 1:]
    first = start[:, :, None, :] + lower[..., None] * delta[:, :, None, :]
    second = start[:, :, None, :] + upper[..., None] * delta[:, :, None, :]
    midpoint = 0.5 * (first + second)
    cross = first[..., 0] * second[..., 1] - first[..., 1] * second[..., 0]
    triangle = 0.5 * cross
    triangle_moment = triangle[..., None] * (first + second) / 3.0
    sector = (
        0.5
        * radii[:, None, None] ** 2
        * torch.atan2(cross, torch.sum(first * second, dim=3))
    )
    first_norm = torch.linalg.vector_norm(first, dim=3).clamp_min(
        torch.finfo(vertices.dtype).tiny
    )
    second_norm = torch.linalg.vector_norm(second, dim=3).clamp_min(
        torch.finfo(vertices.dtype).tiny
    )
    first_unit = first / first_norm[..., None]
    second_unit = second / second_norm[..., None]
    sector_moment = (radii[:, None, None, None] ** 3 / 3.0) * torch.stack(
        (
            second_unit[..., 1] - first_unit[..., 1],
            first_unit[..., 0] - second_unit[..., 0],
        ),
        dim=3,
    )
    inside = torch.sum(midpoint * midpoint, dim=3) <= (
        radii[:, None, None] ** 2 + tolerance[:, :, None]
    )
    intervals = torch.where(inside, triangle, sector)
    interval_moments = torch.where(
        inside[..., None],
        triangle_moment,
        sector_moment,
    )
    intervals = torch.where(
        upper > lower,
        intervals,
        torch.zeros_like(intervals),
    )
    interval_moments = torch.where(
        (upper > lower)[..., None],
        interval_moments,
        torch.zeros_like(interval_moments),
    )
    valid_edges = (positions < counts[:, None]) & (counts[:, None] >= 3)
    signed_intersection_area = torch.sum(intervals * valid_edges[..., None], dim=(1, 2))
    intersection_orientation = torch.where(
        signed_intersection_area < 0,
        -torch.ones_like(signed_intersection_area),
        torch.ones_like(signed_intersection_area),
    )
    intersection_area = torch.abs(signed_intersection_area)
    intersection_local_moment = intersection_orientation[:, None] * torch.sum(
        interval_moments * valid_edges[..., None, None],
        dim=(1, 2),
    )
    vertex_distance_squared = torch.sum(start * start, dim=2)
    containment_tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (
            radii * radii
            + torch.max(
                torch.where(
                    valid_edges,
                    vertex_distance_squared,
                    torch.zeros_like(vertex_distance_squared),
                ),
                dim=1,
            ).values
            + 1.0
        )
    )
    fully_contained = (counts >= 3) & torch.all(
        (~valid_edges)
        | (
            vertex_distance_squared
            <= radii[:, None] * radii[:, None] + containment_tolerance[:, None]
        ),
        dim=1,
    )
    polygon_signed_area = torch.sum(triangle[..., 0] * valid_edges, dim=1)
    polygon_orientation = torch.where(
        polygon_signed_area < 0,
        -torch.ones_like(polygon_signed_area),
        torch.ones_like(polygon_signed_area),
    )
    polygon_area = torch.abs(polygon_signed_area)
    polygon_local_moment = polygon_orientation[:, None] * torch.sum(
        triangle_moment[..., 0, :] * valid_edges[..., None],
        dim=1,
    )
    area = torch.where(fully_contained, polygon_area, intersection_area)
    local_moment = torch.where(
        fully_contained[:, None],
        polygon_local_moment,
        intersection_local_moment,
    )
    return area, local_moment + centers * area[:, None]


@lru_cache(maxsize=1)
def _compiled_batched_power_kernels() -> tuple[Any, Any]:
    """Create compiled numeric kernels lazily; validation and assembly stay eager."""
    return (
        torch.compile(_batched_clip_power_cells, fullgraph=True),
        torch.compile(_batched_polygon_disk_area_moment, fullgraph=True),
    )


def batched_bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    candidate_indptr: Tensor,
    candidate_indices: Tensor,
    neighbor_indptr: Tensor,
    neighbor_indices: Tensor,
    max_candidate_pairs: int = 1_000_000,
    max_packed_vertices: int = 64,
    candidate_chunk_size: int = 4096,
    kernel_execution: Literal["eager", "compile"] = "eager",
) -> PowerPixelOperator:
    """Build the exact operator with batched Torch clipping and area integration.

    This production candidate requires both sparse CSR directions. The scalar
    :func:`bounded_power_pixel_operator` remains the small-input oracle and dense
    fallback.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("batched_bounded_power_pixel_operator requires 2-D sites")
    if (
        not pixel_polygons.is_floating_point()
        or pixel_polygons.ndim != 3
        or pixel_polygons.shape[-1] != 2
        or pixel_polygons.shape[1] < 3
    ):
        raise ValueError("pixel_polygons must have floating shape (P, V>=3, 2)")
    if pixel_polygons.device != sites.device or pixel_polygons.dtype != sites.dtype:
        raise ValueError("pixels, sites, and radii must share dtype and device")
    if not bool(torch.isfinite(pixel_polygons).all()):
        raise ValueError("pixel polygons must be finite")
    signed_pixel_areas = (
        torch.stack([_polygon_signed_area(polygon) for polygon in pixel_polygons])
        if pixel_polygons.shape[0]
        else pixel_polygons.new_empty((0,))
    )
    if bool(torch.any(torch.abs(signed_pixel_areas) == 0)):
        raise ValueError("pixel polygons must have non-zero area")
    edge_a = torch.roll(pixel_polygons, shifts=-1, dims=1) - pixel_polygons
    edge_b = torch.roll(edge_a, shifts=-1, dims=1)
    turns = edge_a[..., 0] * edge_b[..., 1] - edge_a[..., 1] * edge_b[..., 0]
    if bool(torch.any((turns > 0).any(dim=1) & (turns < 0).any(dim=1))):
        raise ValueError("pixel polygons must be convex and consistently ordered")
    pixel_count = pixel_polygons.shape[0]
    cell_count = sites.shape[0]
    _validate_csr(
        candidate_indptr,
        candidate_indices,
        rows=pixel_count,
        columns=cell_count,
        name="candidate",
        device=sites.device,
    )
    _validate_csr(
        neighbor_indptr,
        neighbor_indices,
        rows=cell_count,
        columns=cell_count,
        name="neighbor",
        device=sites.device,
    )
    if candidate_indices.numel() > int(max_candidate_pairs):
        raise ValueError("candidate CSR exceeds max_candidate_pairs")
    chunk_size = int(candidate_chunk_size)
    if chunk_size <= 0:
        raise ValueError("candidate_chunk_size must be positive")
    if kernel_execution == "eager":
        clip_cells = _batched_clip_power_cells
        polygon_disk_area_moment = _batched_polygon_disk_area_moment
    elif kernel_execution == "compile":
        clip_cells, polygon_disk_area_moment = _compiled_batched_power_kernels()
    else:
        raise ValueError("kernel_execution must be 'eager' or 'compile'")

    candidate_pixels = torch.repeat_interleave(
        torch.arange(pixel_count, device=sites.device),
        candidate_indptr[1:] - candidate_indptr[:-1],
    )
    if cell_count:
        keys = candidate_pixels * cell_count + candidate_indices
        keys = torch.unique(keys, sorted=True)
        candidate_pixels = torch.div(keys, cell_count, rounding_mode="floor")
        candidate_cells = torch.remainder(keys, cell_count)
    else:
        candidate_cells = candidate_indices
    active_cells = torch.unique(candidate_cells, sorted=True)
    neighbor_counts = neighbor_indptr[1:] - neighbor_indptr[:-1]
    max_neighbors = (
        int(neighbor_counts[active_cells].max().item()) if active_cells.numel() else 0
    )
    if pixel_polygons.shape[1] + max_neighbors > int(max_packed_vertices):
        raise ValueError(
            "batched clipping exceeds max_packed_vertices; raise the explicit "
            "limit or use the scalar correctness oracle"
        )

    cell_areas = sites.new_zeros((cell_count,))
    cell_centered_first_moments = sites.new_zeros((cell_count, 2))
    for start in range(0, active_cells.shape[0], chunk_size):
        chunk_cells = active_cells[start : start + chunk_size]
        chunk_radii = radii[chunk_cells]
        lower = sites[chunk_cells] - chunk_radii[:, None]
        upper = sites[chunk_cells] + chunk_radii[:, None]
        bounds = torch.stack(
            (
                lower,
                torch.stack((upper[:, 0], lower[:, 1]), dim=1),
                upper,
                torch.stack((lower[:, 0], upper[:, 1]), dim=1),
            ),
            dim=1,
        )
        chunk_neighbors = _padded_cell_neighbors(
            chunk_cells,
            neighbor_indptr=neighbor_indptr,
            neighbor_indices=neighbor_indices,
            cell_count=cell_count,
            max_neighbors=max_neighbors,
        )
        cell_vertices, cell_counts = clip_cells(
            bounds,
            chunk_cells,
            chunk_neighbors,
            sites,
            radii,
        )
        chunk_areas, chunk_first_moments = polygon_disk_area_moment(
            cell_vertices,
            cell_counts,
            sites[chunk_cells],
            chunk_radii,
        )
        cell_areas[chunk_cells] = chunk_areas
        cell_centered_first_moments[chunk_cells] = (
            chunk_first_moments - sites[chunk_cells] * chunk_areas[:, None]
        )

    kept_pixels: list[Tensor] = []
    kept_cells: list[Tensor] = []
    kept_areas: list[Tensor] = []
    kept_centered_first_moments: list[Tensor] = []
    for start in range(0, candidate_cells.shape[0], chunk_size):
        chunk_cells = candidate_cells[start : start + chunk_size]
        chunk_pixels = candidate_pixels[start : start + chunk_size]
        chunk_neighbors = _padded_cell_neighbors(
            chunk_cells,
            neighbor_indptr=neighbor_indptr,
            neighbor_indices=neighbor_indices,
            cell_count=cell_count,
            max_neighbors=max_neighbors,
        )
        vertices, counts = clip_cells(
            pixel_polygons[chunk_pixels],
            chunk_cells,
            chunk_neighbors,
            sites,
            radii,
        )
        chunk_areas, chunk_first_moments = polygon_disk_area_moment(
            vertices,
            counts,
            sites[chunk_cells],
            radii[chunk_cells],
        )
        area_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * (
                torch.abs(signed_pixel_areas[chunk_pixels])
                + math.pi * radii[chunk_cells] ** 2
            )
        )
        positions = torch.arange(vertices.shape[1], device=vertices.device)
        valid_vertices = positions[None, :] < counts[:, None]
        squared_distance = torch.sum(
            (vertices - sites[chunk_cells, None, :]) ** 2,
            dim=2,
        )
        containment_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * (
                radii[chunk_cells] * radii[chunk_cells]
                + torch.max(
                    torch.where(
                        valid_vertices,
                        squared_distance,
                        torch.zeros_like(squared_distance),
                    ),
                    dim=1,
                ).values
                + 1.0
            )
        )
        fully_contained = (counts >= 3) & torch.all(
            (~valid_vertices)
            | (
                squared_distance
                <= radii[chunk_cells, None] ** 2 + containment_tolerance[:, None]
            ),
            dim=1,
        )
        pixel_tolerance = (
            64.0
            * torch.finfo(chunk_areas.dtype).eps
            * torch.abs(signed_pixel_areas[chunk_pixels])
        )
        keep = (chunk_areas > area_tolerance) | (
            fully_contained & (chunk_areas > pixel_tolerance)
        )
        kept_pixels.append(chunk_pixels[keep])
        kept_cells.append(chunk_cells[keep])
        kept_areas.append(chunk_areas[keep])
        kept_centered_first_moments.append(
            (chunk_first_moments - sites[chunk_cells] * chunk_areas[:, None])[keep]
        )
    pixel_ids = (
        torch.cat(kept_pixels) if kept_pixels else candidate_pixels.new_empty((0,))
    )
    indices = torch.cat(kept_cells) if kept_cells else candidate_cells.new_empty((0,))
    areas = torch.cat(kept_areas) if kept_areas else sites.new_empty((0,))
    centered_first_moments = (
        torch.cat(kept_centered_first_moments)
        if kept_centered_first_moments
        else sites.new_empty((0, 2))
    )
    row_counts = torch.bincount(pixel_ids, minlength=pixel_count)
    indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(row_counts, dim=0),
        )
    )
    order = torch.argsort(pixel_ids, stable=True)
    order = order[torch.argsort(indices[order], stable=True)]
    transpose_indices = pixel_ids[order]
    transpose_areas = areas[order]
    transpose_counts = torch.bincount(indices, minlength=cell_count)
    transpose_indptr = torch.cat(
        (
            torch.zeros(1, dtype=torch.long, device=sites.device),
            torch.cumsum(transpose_counts, dim=0),
        )
    )
    return PowerPixelOperator(
        indptr=indptr,
        indices=indices,
        pixel_ids=pixel_ids,
        areas=areas,
        centered_first_moments=centered_first_moments,
        transpose_indptr=transpose_indptr,
        transpose_indices=transpose_indices,
        transpose_areas=transpose_areas,
        cell_areas=cell_areas,
        cell_centered_first_moments=cell_centered_first_moments,
        pixel_count=pixel_count,
        cell_count=cell_count,
        candidate_pairs=int(candidate_indices.numel()),
    )


def sparse_bounded_power_pixel_operator(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_candidate_pairs: int = 10_000_000,
    max_bin_memberships: int = 50_000_000,
    max_site_bin_memberships: int = 10_000_000,
    max_packed_vertices: int = 64,
    candidate_chunk_size: int = 4096,
    kernel_execution: Literal["eager", "compile"] = "eager",
) -> PowerPixelOperator:
    """Build the production sparse exact operator for arbitrary sky pixels."""
    candidates = polygon_disk_candidate_csr(
        pixel_polygons,
        sites,
        radii,
        bin_size=bin_size,
        max_bin_memberships=max_bin_memberships,
        max_candidate_pairs=max_candidate_pairs,
    )
    neighbors = sparse_cech_graph(
        sites,
        radii,
        bin_size=bin_size,
        max_site_bin_memberships=max_site_bin_memberships,
        max_candidate_pairs=max_candidate_pairs,
    )
    return batched_bounded_power_pixel_operator(
        pixel_polygons,
        sites,
        radii,
        candidate_indptr=candidates.indptr,
        candidate_indices=candidates.indices,
        neighbor_indptr=neighbors.indptr,
        neighbor_indices=neighbors.indices,
        max_candidate_pairs=max_candidate_pairs,
        max_packed_vertices=max_packed_vertices,
        candidate_chunk_size=candidate_chunk_size,
        kernel_execution=kernel_execution,
    )
