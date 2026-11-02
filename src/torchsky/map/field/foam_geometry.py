"""Bounded power-diagram primitives for experimental sky-field models.

These tensor-native kernels implement the geometry underlying Power Foam.  They
are representation primitives, not a complete renderer: exact rendering also
needs sparse candidate construction and pixel/cell intersection integration.
"""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor


@dataclass(frozen=True)
class SparseCechGraph:
    """Experimental exact Čech graph plus its sparse-index diagnostics."""

    candidate_pairs: Tensor
    edges: Tensor
    indptr: Tensor
    indices: Tensor
    total_pairs: int
    site_bin_memberships: int
    occupied_bins: int
    bin_size: float


@dataclass(frozen=True)
class TileDiskCSR:
    """Exact disk candidates for a regular tangent-plane tile grid."""

    indptr: Tensor
    indices: Tensor
    tile_ids: Tensor
    aabb_memberships: int
    total_pairs: int
    tile_shape: tuple[int, int]
    tile_origin: tuple[float, float]
    tile_size: tuple[float, float]


@dataclass(frozen=True)
class PolygonDiskCSR:
    """Conservative disk candidates for arbitrary convex pixel polygons."""

    indptr: Tensor
    indices: Tensor
    polygon_ids: Tensor
    polygon_bin_memberships: int
    site_bin_memberships: int
    raw_candidate_pairs: int
    total_pairs: int
    bin_size: float


def _validate_sites_and_radii(
    sites: Tensor,
    radii: Tensor,
    *,
    allow_empty: bool,
) -> None:
    if not sites.is_floating_point():
        raise TypeError("sites must be a floating-point tensor")
    if sites.ndim != 2:
        raise ValueError("sites must have shape (N, D)")
    if not allow_empty and sites.shape[0] == 0:
        raise ValueError("at least one power site is required")
    if radii.ndim != 1 or radii.shape[0] != sites.shape[0]:
        raise ValueError("radii must have shape (N,) matching sites")
    if sites.device != radii.device:
        raise ValueError("sites and radii must be on the same device")
    if sites.dtype != radii.dtype:
        raise ValueError("sites and radii must have the same dtype")
    if not bool(torch.isfinite(sites).all()):
        raise ValueError("sites must be finite")
    if not bool(torch.isfinite(radii).all()) or bool(torch.any(radii < 0)):
        raise ValueError("radii must be finite and non-negative")


def _validate_power_inputs(points: Tensor, sites: Tensor, radii: Tensor) -> None:
    _validate_sites_and_radii(sites, radii, allow_empty=False)
    if not points.is_floating_point():
        raise TypeError("points must be a floating-point tensor")
    if points.ndim < 1:
        raise ValueError("points must end in coordinates")
    if points.shape[-1] != sites.shape[-1]:
        raise ValueError("points and sites must have the same coordinate dimension")
    if points.device != sites.device:
        raise ValueError("points, sites, and radii must be on the same device")
    if points.dtype != sites.dtype:
        raise ValueError("points, sites, and radii must have the same dtype")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("points must be finite")


def power_distance(points: Tensor, sites: Tensor, radii: Tensor) -> Tensor:
    """Return ``||x - p_i||² - r_i²`` for every point/site pair.

    ``points`` has shape ``(..., D)``, ``sites`` has shape ``(N, D)``, and the
    result has shape ``(..., N)``.  Use :func:`bounded_power_ownership` for
    bounded-memory ownership queries over many points.
    """
    _validate_power_inputs(points, sites, radii)
    delta = points.unsqueeze(-2) - sites
    return torch.sum(delta * delta, dim=-1) - radii * radii


def bounded_power_ownership(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    point_chunk_size: int | None = None,
) -> tuple[Tensor, Tensor]:
    """Return exact bounded-power-cell ids and minimum power distances.

    Cell ``i`` is its ordinary power cell intersected with its radius-``r_i``
    ball.  Points outside the union of those balls receive owner ``-1``.  The
    optional chunk size bounds the temporary point-by-site distance matrix.
    """
    _validate_power_inputs(points, sites, radii)
    flat = points.reshape(-1, points.shape[-1])
    if flat.shape[0] == 0:
        shape = points.shape[:-1]
        return (
            torch.empty(shape, dtype=torch.long, device=points.device),
            torch.empty(shape, dtype=points.dtype, device=points.device),
        )
    chunk_size = flat.shape[0] if point_chunk_size is None else int(point_chunk_size)
    if chunk_size <= 0:
        raise ValueError("point_chunk_size must be positive")

    owner_parts: list[Tensor] = []
    distance_parts: list[Tensor] = []
    for start in range(0, flat.shape[0], chunk_size):
        distances = power_distance(flat[start : start + chunk_size], sites, radii)
        minimum, owners = torch.min(distances, dim=-1)
        owners = torch.where(minimum <= 0, owners, torch.full_like(owners, -1))
        owner_parts.append(owners)
        distance_parts.append(minimum)
    shape = points.shape[:-1]
    return torch.cat(owner_parts).reshape(shape), torch.cat(distance_parts).reshape(
        shape
    )


def soft_bounded_power_weights(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    temperature: float = 10.0,
) -> Tensor:
    """Return a differentiable, compactly supported power-ownership surrogate.

    Weights are a softmax of negative power distance over balls containing each
    query.  They sum to one inside the union of bounds and to zero outside it.
    This is useful during optimization but is not an exact power-cell renderer.
    """
    _validate_power_inputs(points, sites, radii)
    if not math.isfinite(float(temperature)) or float(temperature) <= 0:
        raise ValueError("temperature must be finite and positive")
    distances = power_distance(points, sites, radii)
    support = distances <= 0
    logits = -float(temperature) * distances
    masked_logits = torch.where(support, logits, torch.full_like(logits, -torch.inf))
    any_support = support.any(dim=-1, keepdim=True)
    safe_logits = torch.where(
        any_support, masked_logits, torch.zeros_like(masked_logits)
    )
    weights = torch.softmax(safe_logits, dim=-1)
    return torch.where(any_support, weights, torch.zeros_like(weights))


def smooth_bounded_power_weights(
    points: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    power_temperature: float = 10.0,
    support_temperature: float = 10.0,
) -> Tensor:
    """Return smooth bounded-power weights for geometry optimization.

    The result sums to a smooth probability that each query lies inside the
    union of the site bounds; the missing mass is background. Conditional
    ownership combines power distance with each site's smooth support
    probability. This is a training surrogate, not exact power-cell inference
    or exact pixel-area integration.
    """
    _validate_power_inputs(points, sites, radii)
    power_scale = float(power_temperature)
    support_scale = float(support_temperature)
    if not math.isfinite(power_scale) or power_scale <= 0:
        raise ValueError("power_temperature must be finite and positive")
    if not math.isfinite(support_scale) or support_scale <= 0:
        raise ValueError("support_temperature must be finite and positive")

    distances = power_distance(points, sites, radii)
    log_membership = torch.nn.functional.logsigmoid(-support_scale * distances)
    conditional = torch.softmax(
        -power_scale * distances + log_membership,
        dim=-1,
    )
    log_outside_union = torch.nn.functional.logsigmoid(support_scale * distances).sum(
        dim=-1, keepdim=True
    )
    union_occupancy = -torch.expm1(log_outside_union)
    return conditional * union_occupancy


def cech_adjacency(
    sites: Tensor,
    radii: Tensor,
    *,
    site_chunk_size: int = 1024,
) -> Tensor:
    """Return undirected edges between all overlapping bounding balls.

    The result has shape ``(2, E)`` with each edge stored once as ``i < j``.
    This exact, bounded-memory reference still performs quadratic pair tests;
    survey-scale callers should use a spatial candidate index before this test.
    """
    _validate_power_inputs(sites[:1], sites, radii)
    chunk_size = int(site_chunk_size)
    if chunk_size <= 0:
        raise ValueError("site_chunk_size must be positive")
    edges: list[Tensor] = []
    count = sites.shape[0]
    for start in range(0, count, chunk_size):
        stop = min(start + chunk_size, count)
        delta = sites[start:stop, None, :] - sites[None, :, :]
        distance_squared = torch.sum(delta * delta, dim=-1)
        radius_sum = radii[start:stop, None] + radii[None, :]
        local_i = torch.arange(start, stop, device=sites.device)[:, None]
        global_j = torch.arange(count, device=sites.device)[None, :]
        overlap = (local_i < global_j) & (distance_squared <= radius_sum * radius_sum)
        ij = torch.nonzero(overlap, as_tuple=False)
        if ij.numel() != 0:
            ij[:, 0] += start
            edges.append(ij)
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=sites.device)
    return torch.cat(edges, dim=0).transpose(0, 1).contiguous()


def _empty_sparse_cech_graph(
    sites: Tensor,
    *,
    bin_size: float,
    site_bin_memberships: int = 0,
    occupied_bins: int = 0,
) -> SparseCechGraph:
    empty_pairs = torch.empty((2, 0), dtype=torch.long, device=sites.device)
    return SparseCechGraph(
        candidate_pairs=empty_pairs,
        edges=empty_pairs,
        indptr=torch.zeros(
            sites.shape[0] + 1,
            dtype=torch.long,
            device=sites.device,
        ),
        indices=torch.empty((0,), dtype=torch.long, device=sites.device),
        total_pairs=sites.shape[0] * (sites.shape[0] - 1) // 2,
        site_bin_memberships=site_bin_memberships,
        occupied_bins=occupied_bins,
        bin_size=bin_size,
    )


def sparse_cech_graph(
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_site_bin_memberships: int = 10_000_000,
    max_candidate_pairs: int = 50_000_000,
) -> SparseCechGraph:
    """Build an exact 2-D Čech graph through a Torch-native uniform-grid index.

    Each disk is inserted into every grid bin touched by its axis-aligned
    bounding box. Candidate pairs share at least one bin, then the exact
    disk-overlap predicate is applied. ``edges`` stores each overlap once as
    ``i < j``; ``indptr`` and ``indices`` store the same graph symmetrically.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("sparse_cech_graph currently requires 2-D disk sites")
    membership_limit = int(max_site_bin_memberships)
    candidate_limit = int(max_candidate_pairs)
    if membership_limit <= 0 or candidate_limit <= 0:
        raise ValueError("sparse Čech allocation limits must be positive")

    if bin_size is None:
        max_radius = float(radii.max().item()) if radii.numel() else 0.0
        width = 2.0 * max_radius if max_radius > 0.0 else 1.0
    else:
        width = float(bin_size)
    if not math.isfinite(width) or width <= 0.0:
        raise ValueError("bin_size must be finite and positive")
    if sites.shape[0] < 2:
        return _empty_sparse_cech_graph(sites, bin_size=width)

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype).unsqueeze(-1)
        lower = index_sites - index_radii
        upper = index_sites + index_radii
        origin = lower.amin(dim=0)
        lower_bin = torch.floor((lower - origin) / width).to(dtype=torch.long)
        upper_bin = torch.floor((upper - origin) / width).to(dtype=torch.long)
        lower_bin = lower_bin.clamp_min(0)
        upper_bin = torch.maximum(upper_bin, lower_bin)
        spans = upper_bin - lower_bin + 1
        membership_counts = spans[:, 0] * spans[:, 1]
        membership_count = int(membership_counts.sum().item())
        if membership_count > membership_limit:
            raise ValueError(
                f"uniform-grid index needs {membership_count} site-bin memberships, "
                f"exceeding max_site_bin_memberships={membership_limit}; increase "
                "bin_size or the explicit limit"
            )

        site_ids = torch.repeat_interleave(
            torch.arange(sites.shape[0], device=sites.device),
            membership_counts,
        )
        membership_offsets = torch.cumsum(membership_counts, dim=0) - membership_counts
        local = torch.arange(membership_count, device=sites.device) - (
            torch.repeat_interleave(membership_offsets, membership_counts)
        )
        span_x = spans[site_ids, 0]
        bin_x = lower_bin[site_ids, 0] + torch.remainder(local, span_x)
        bin_y = lower_bin[site_ids, 1] + torch.div(
            local,
            span_x,
            rounding_mode="floor",
        )
        bin_columns = int(upper_bin[:, 0].max().item()) + 1
        max_bin_y = int(upper_bin[:, 1].max().item())
        if max_bin_y > (torch.iinfo(torch.long).max - bin_columns + 1) // bin_columns:
            raise ValueError("uniform-grid key range exceeds int64")
        bin_keys = bin_y * bin_columns + bin_x

        order = torch.argsort(site_ids, stable=True)
        order = order[torch.argsort(bin_keys[order], stable=True)]
        sorted_keys = bin_keys[order]
        sorted_sites = site_ids[order]
        _, bin_counts = torch.unique_consecutive(sorted_keys, return_counts=True)
        occupied_bins = int(bin_counts.numel())
        pair_counts = bin_counts * (bin_counts - 1) // 2
        raw_pair_count = int(pair_counts.sum().item())
        if raw_pair_count > candidate_limit:
            raise ValueError(
                f"uniform-grid bins produce {raw_pair_count} raw candidate pairs, "
                f"exceeding max_candidate_pairs={candidate_limit}; decrease "
                "occupancy with a smaller bin_size or raise the explicit limit"
            )
        if raw_pair_count == 0:
            return _empty_sparse_cech_graph(
                sites,
                bin_size=width,
                site_bin_memberships=membership_count,
                occupied_bins=occupied_bins,
            )

        group_ids = torch.repeat_interleave(
            torch.arange(occupied_bins, device=sites.device),
            pair_counts,
        )
        pair_offsets = torch.cumsum(pair_counts, dim=0) - pair_counts
        pair_rank = torch.arange(raw_pair_count, device=sites.device) - (
            torch.repeat_interleave(pair_offsets, pair_counts)
        )
        group_size = bin_counts[group_ids]
        solve_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        diagonal = (2 * group_size - 1).to(dtype=solve_dtype)
        row = torch.floor(
            0.5
            * (
                diagonal
                - torch.sqrt(
                    diagonal * diagonal - 8.0 * pair_rank.to(dtype=solve_dtype)
                )
            )
        ).to(dtype=torch.long)
        row_start = row * (2 * group_size - row - 1) // 2
        row = torch.where(row_start > pair_rank, row - 1, row)
        row_start = row * (2 * group_size - row - 1) // 2
        next_start = (row + 1) * (2 * group_size - row - 2) // 2
        row = torch.where(next_start <= pair_rank, row + 1, row)
        row_start = row * (2 * group_size - row - 1) // 2
        column = row + 1 + pair_rank - row_start
        group_start = (torch.cumsum(bin_counts, dim=0) - bin_counts)[group_ids]
        pairs = torch.stack(
            (
                sorted_sites[group_start + row],
                sorted_sites[group_start + column],
            ),
            dim=1,
        )
        pairs = torch.unique(torch.sort(pairs, dim=1).values, dim=0, sorted=True)
        pair_order = torch.argsort(pairs[:, 1], stable=True)
        pairs = pairs[pair_order]
        pairs = pairs[torch.argsort(pairs[:, 0], stable=True)]
        candidate_pairs = pairs.transpose(0, 1).contiguous()

        left, right = candidate_pairs
        delta = sites[left] - sites[right]
        radius_sum = radii[left] + radii[right]
        overlap = torch.sum(delta * delta, dim=-1) <= radius_sum * radius_sum
        edges = candidate_pairs[:, overlap]
        if edges.shape[1] == 0:
            indptr = torch.zeros(
                sites.shape[0] + 1,
                dtype=torch.long,
                device=sites.device,
            )
            indices = torch.empty((0,), dtype=torch.long, device=sites.device)
        else:
            rows = torch.cat((edges[0], edges[1]))
            columns = torch.cat((edges[1], edges[0]))
            csr_order = torch.argsort(columns, stable=True)
            csr_order = csr_order[torch.argsort(rows[csr_order], stable=True)]
            rows = rows[csr_order]
            indices = columns[csr_order]
            row_counts = torch.bincount(rows, minlength=sites.shape[0])
            indptr = torch.cat(
                (
                    torch.zeros(1, dtype=torch.long, device=sites.device),
                    torch.cumsum(row_counts, dim=0),
                )
            )

    return SparseCechGraph(
        candidate_pairs=candidate_pairs,
        edges=edges,
        indptr=indptr,
        indices=indices,
        total_pairs=sites.shape[0] * (sites.shape[0] - 1) // 2,
        site_bin_memberships=membership_count,
        occupied_bins=occupied_bins,
        bin_size=width,
    )


def tile_disk_candidate_csr(
    sites: Tensor,
    radii: Tensor,
    *,
    tile_shape: tuple[int, int],
    tile_origin: tuple[float, float] = (0.0, 0.0),
    tile_size: tuple[float, float] = (1.0, 1.0),
    max_aabb_memberships: int = 50_000_000,
) -> TileDiskCSR:
    """Return exact disk-to-tile candidates for a regular 2-D tile grid.

    Coordinates use a local tangent plane with ``(x, y)`` sites, an ``(x, y)``
    lower-left grid origin, an ``(x, y)`` tile size, and a ``(rows, columns)``
    tile shape. Tiles are flattened in row-major order. A disk touching a tile
    boundary is included in every adjacent tile sharing that boundary.

    The index enumerates only the tiles covered by each disk's AABB, then
    applies the exact closed disk/AABB intersection predicate. It never
    allocates a ``number_of_tiles × number_of_sites`` matrix.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("tile_disk_candidate_csr requires 2-D disk sites")
    if len(tile_shape) != 2:
        raise ValueError("tile_shape must contain (rows, columns)")
    rows, columns = (int(tile_shape[0]), int(tile_shape[1]))
    if rows <= 0 or columns <= 0:
        raise ValueError("tile_shape entries must be positive")
    if len(tile_origin) != 2 or len(tile_size) != 2:
        raise ValueError("tile_origin and tile_size must contain (x, y)")
    origin_x, origin_y = (float(tile_origin[0]), float(tile_origin[1]))
    size_x, size_y = (float(tile_size[0]), float(tile_size[1]))
    if not all(math.isfinite(value) for value in (origin_x, origin_y)):
        raise ValueError("tile_origin entries must be finite")
    if not all(math.isfinite(value) and value > 0.0 for value in (size_x, size_y)):
        raise ValueError("tile_size entries must be finite and positive")
    membership_limit = int(max_aabb_memberships)
    if membership_limit <= 0:
        raise ValueError("max_aabb_memberships must be positive")
    tile_count = rows * columns
    if tile_count > torch.iinfo(torch.long).max:
        raise ValueError("tile grid exceeds int64 indexing")

    empty = torch.empty((0,), dtype=torch.long, device=sites.device)
    if sites.shape[0] == 0:
        return TileDiskCSR(
            indptr=torch.zeros(tile_count + 1, dtype=torch.long, device=sites.device),
            indices=empty,
            tile_ids=empty,
            aabb_memberships=0,
            total_pairs=0,
            tile_shape=(rows, columns),
            tile_origin=(origin_x, origin_y),
            tile_size=(size_x, size_y),
        )

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype)
        grid_origin = torch.tensor(
            [origin_x, origin_y],
            dtype=index_dtype,
            device=sites.device,
        )
        grid_size = torch.tensor(
            [size_x, size_y],
            dtype=index_dtype,
            device=sites.device,
        )
        lower_scaled = (index_sites - index_radii[:, None] - grid_origin) / grid_size
        upper_scaled = (index_sites + index_radii[:, None] - grid_origin) / grid_size
        # Closed support includes the tile immediately left/below an exact boundary.
        lower_tile = torch.ceil(lower_scaled).to(torch.long) - 1
        upper_tile = torch.floor(upper_scaled).to(torch.long)
        lower_tile[:, 0].clamp_(0, columns - 1)
        upper_tile[:, 0].clamp_(0, columns - 1)
        lower_tile[:, 1].clamp_(0, rows - 1)
        upper_tile[:, 1].clamp_(0, rows - 1)

        grid_upper = grid_origin + grid_size * torch.tensor(
            [columns, rows],
            dtype=index_dtype,
            device=sites.device,
        )
        outside = torch.any(
            (index_sites + index_radii[:, None] < grid_origin)
            | (index_sites - index_radii[:, None] > grid_upper),
            dim=1,
        )
        spans = upper_tile - lower_tile + 1
        membership_counts = spans[:, 0] * spans[:, 1]
        membership_counts = torch.where(
            outside,
            torch.zeros_like(membership_counts),
            membership_counts,
        )
        membership_count = int(membership_counts.sum().item())
        if membership_count > membership_limit:
            raise ValueError(
                f"disk AABBs cover {membership_count} tile memberships, exceeding "
                f"max_aabb_memberships={membership_limit}; increase tile size or "
                "the explicit limit"
            )
        if membership_count == 0:
            return TileDiskCSR(
                indptr=torch.zeros(
                    tile_count + 1,
                    dtype=torch.long,
                    device=sites.device,
                ),
                indices=empty,
                tile_ids=empty,
                aabb_memberships=0,
                total_pairs=tile_count * sites.shape[0],
                tile_shape=(rows, columns),
                tile_origin=(origin_x, origin_y),
                tile_size=(size_x, size_y),
            )

        site_ids = torch.repeat_interleave(
            torch.arange(sites.shape[0], device=sites.device),
            membership_counts,
        )
        membership_offsets = torch.cumsum(membership_counts, dim=0) - membership_counts
        local = torch.arange(membership_count, device=sites.device) - (
            torch.repeat_interleave(membership_offsets, membership_counts)
        )
        span_x = spans[site_ids, 0]
        tile_x = lower_tile[site_ids, 0] + torch.remainder(local, span_x)
        tile_y = lower_tile[site_ids, 1] + torch.div(
            local,
            span_x,
            rounding_mode="floor",
        )

        tile_lower = grid_origin + grid_size * torch.stack((tile_x, tile_y), dim=1)
        tile_upper = tile_lower + grid_size
        centers = index_sites[site_ids]
        closest = torch.minimum(torch.maximum(centers, tile_lower), tile_upper)
        intersects = torch.sum((centers - closest) ** 2, dim=1) <= (
            index_radii[site_ids] ** 2
        )
        site_ids = site_ids[intersects]
        tile_ids = (tile_y[intersects] * columns + tile_x[intersects]).to(torch.long)

        order = torch.argsort(site_ids, stable=True)
        order = order[torch.argsort(tile_ids[order], stable=True)]
        tile_ids = tile_ids[order]
        indices = site_ids[order]
        tile_counts = torch.bincount(tile_ids, minlength=tile_count)
        indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=sites.device),
                torch.cumsum(tile_counts, dim=0),
            )
        )

    return TileDiskCSR(
        indptr=indptr,
        indices=indices,
        tile_ids=tile_ids,
        aabb_memberships=membership_count,
        total_pairs=tile_count * sites.shape[0],
        tile_shape=(rows, columns),
        tile_origin=(origin_x, origin_y),
        tile_size=(size_x, size_y),
    )


def polygon_disk_candidate_csr(
    pixel_polygons: Tensor,
    sites: Tensor,
    radii: Tensor,
    *,
    bin_size: float | None = None,
    max_bin_memberships: int = 50_000_000,
    max_candidate_pairs: int = 10_000_000,
) -> PolygonDiskCSR:
    """Return conservative disk candidates for arbitrary convex polygons.

    Polygon and disk AABBs are joined through a uniform grid, then candidate
    pairs that fail the exact disk/AABB intersection predicate are removed.
    The result can contain false positives for rotated polygons, but it cannot
    omit a disk that intersects a polygon. ``max_candidate_pairs`` bounds both
    each temporary join and the retained result, not duplicate hits across bins.
    """
    _validate_sites_and_radii(sites, radii, allow_empty=True)
    if sites.shape[1] != 2:
        raise ValueError("polygon_disk_candidate_csr requires 2-D disk sites")
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
    membership_limit = int(max_bin_memberships)
    candidate_limit = int(max_candidate_pairs)
    if membership_limit <= 0 or candidate_limit <= 0:
        raise ValueError("sparse allocation limits must be positive")
    if bin_size is None:
        max_radius = float(radii.max().item()) if radii.numel() else 0.0
        width = max_radius if max_radius > 0.0 else 1.0
    else:
        width = float(bin_size)
    if not math.isfinite(width) or width <= 0:
        raise ValueError("bin_size must be finite and positive")

    polygon_count = pixel_polygons.shape[0]
    site_count = sites.shape[0]
    empty = torch.empty((0,), dtype=torch.long, device=sites.device)
    if polygon_count == 0 or site_count == 0:
        return PolygonDiskCSR(
            indptr=torch.zeros(
                polygon_count + 1, dtype=torch.long, device=sites.device
            ),
            indices=empty,
            polygon_ids=empty,
            polygon_bin_memberships=0,
            site_bin_memberships=0,
            raw_candidate_pairs=0,
            total_pairs=polygon_count * site_count,
            bin_size=width,
        )

    with torch.no_grad():
        index_dtype = torch.float32 if sites.device.type == "mps" else torch.float64
        polygons = pixel_polygons.detach().to(dtype=index_dtype)
        index_sites = sites.detach().to(dtype=index_dtype)
        index_radii = radii.detach().to(dtype=index_dtype)
        polygon_lower = polygons.amin(dim=1)
        polygon_upper = polygons.amax(dim=1)
        site_lower = index_sites - index_radii[:, None]
        site_upper = index_sites + index_radii[:, None]
        origin = torch.minimum(
            polygon_lower.amin(dim=0),
            site_lower.amin(dim=0),
        )

        def memberships(
            lower: Tensor,
            upper: Tensor,
        ) -> tuple[Tensor, Tensor, int]:
            lower_bin = torch.floor((lower - origin) / width).to(torch.long)
            upper_bin = torch.floor((upper - origin) / width).to(torch.long)
            spans = upper_bin - lower_bin + 1
            counts = spans[:, 0] * spans[:, 1]
            count = int(counts.sum().item())
            if count > membership_limit:
                raise ValueError(
                    f"uniform-grid index needs {count} memberships, exceeding "
                    f"max_bin_memberships={membership_limit}"
                )
            object_ids = torch.repeat_interleave(
                torch.arange(lower.shape[0], device=sites.device),
                counts,
            )
            starts = torch.cumsum(counts, dim=0) - counts
            local = torch.arange(count, device=sites.device) - torch.repeat_interleave(
                starts, counts
            )
            span_x = spans[object_ids, 0]
            bin_x = lower_bin[object_ids, 0] + torch.remainder(local, span_x)
            bin_y = lower_bin[object_ids, 1] + torch.div(
                local, span_x, rounding_mode="floor"
            )
            return object_ids, torch.stack((bin_x, bin_y), dim=1), count

        polygon_ids, polygon_bins, polygon_memberships = memberships(
            polygon_lower, polygon_upper
        )
        site_ids, site_bins, site_memberships = memberships(site_lower, site_upper)
        all_bins = torch.cat((polygon_bins, site_bins))
        min_bins = all_bins.amin(dim=0)
        all_bins = all_bins - min_bins
        columns = int(all_bins[:, 0].max().item()) + 1
        polygon_keys = (
            all_bins[:polygon_memberships, 1] * columns
            + all_bins[:polygon_memberships, 0]
        )
        site_keys = (
            all_bins[polygon_memberships:, 1] * columns
            + all_bins[polygon_memberships:, 0]
        )

        polygon_order = torch.argsort(polygon_ids, stable=True)
        polygon_order = polygon_order[
            torch.argsort(polygon_keys[polygon_order], stable=True)
        ]
        site_order = torch.argsort(site_ids, stable=True)
        site_order = site_order[torch.argsort(site_keys[site_order], stable=True)]
        polygon_keys = polygon_keys[polygon_order]
        polygon_ids = polygon_ids[polygon_order]
        site_keys = site_keys[site_order]
        site_ids = site_ids[site_order]
        unique_polygon_keys, polygon_counts = torch.unique_consecutive(
            polygon_keys, return_counts=True
        )
        unique_site_keys, site_counts = torch.unique_consecutive(
            site_keys, return_counts=True
        )
        polygon_starts = torch.cumsum(polygon_counts, dim=0) - polygon_counts
        site_starts = torch.cumsum(site_counts, dim=0) - site_counts
        site_groups = {
            int(key): (int(start), int(count))
            for key, start, count in zip(
                unique_site_keys.tolist(),
                site_starts.tolist(),
                site_counts.tolist(),
                strict=True,
            )
        }
        pair_pixels: list[Tensor] = []
        pair_sites: list[Tensor] = []
        polygon_groups = list(
            zip(
                unique_polygon_keys.tolist(),
                polygon_starts.tolist(),
                polygon_counts.tolist(),
                strict=True,
            )
        )
        raw_count = sum(
            int(count) * site_groups[int(key)][1]
            for key, _, count in polygon_groups
            if int(key) in site_groups
        )
        stream_join = raw_count > candidate_limit
        retained_count = 0
        if stream_join:
            polygon_first_bin = torch.floor((polygon_lower - origin) / width).long()
            site_first_bin = torch.floor((site_lower - origin) / width).long()
        for key, start, count in polygon_groups:
            site_group = site_groups.get(int(key))
            if site_group is None:
                continue
            site_start, site_count_in_bin = site_group
            group_count = int(count) * site_count_in_bin
            if stream_join:
                for offset in range(0, group_count, candidate_limit):
                    local = torch.arange(
                        offset,
                        min(offset + candidate_limit, group_count),
                        device=sites.device,
                    )
                    pixels = polygon_ids[
                        int(start)
                        + torch.div(local, site_count_in_bin, rounding_mode="floor")
                    ]
                    disks = site_ids[
                        site_start + torch.remainder(local, site_count_in_bin)
                    ]
                    # Each AABB pair belongs to its lexically first shared bin.
                    # This removes duplicate bin hits before retaining any pairs.
                    first = (
                        torch.maximum(polygon_first_bin[pixels], site_first_bin[disks])
                        - min_bins
                    )
                    owned = first[:, 1] * columns + first[:, 0] == int(key)
                    pixels, disks = pixels[owned], disks[owned]
                    closest = torch.minimum(
                        torch.maximum(index_sites[disks], polygon_lower[pixels]),
                        polygon_upper[pixels],
                    )
                    intersects = (index_sites[disks] - closest).square().sum(
                        dim=1
                    ) <= index_radii[disks].square()
                    pixels, disks = pixels[intersects], disks[intersects]
                    retained_count += pixels.numel()
                    if retained_count > candidate_limit:
                        raise ValueError(
                            f"polygon/disk result needs {retained_count} candidate pairs, "
                            f"exceeding max_candidate_pairs={candidate_limit}"
                        )
                    if pixels.numel():
                        pair_pixels.append(pixels)
                        pair_sites.append(disks)
                continue
            pair_pixels.append(
                polygon_ids[int(start) : int(start) + int(count)].repeat_interleave(
                    site_count_in_bin
                )
            )
            pair_sites.append(
                site_ids[site_start : site_start + site_count_in_bin].repeat(int(count))
            )
        if not pair_pixels:
            candidate_pixels = empty
            candidate_sites = empty
        else:
            candidate_pixels = torch.cat(pair_pixels)
            candidate_sites = torch.cat(pair_sites)
            keys = torch.unique(
                candidate_pixels * site_count + candidate_sites,
                sorted=True,
            )
            candidate_pixels = torch.div(keys, site_count, rounding_mode="floor")
            candidate_sites = torch.remainder(keys, site_count)
            closest = torch.minimum(
                torch.maximum(
                    index_sites[candidate_sites], polygon_lower[candidate_pixels]
                ),
                polygon_upper[candidate_pixels],
            )
            intersects = (
                torch.sum((index_sites[candidate_sites] - closest) ** 2, dim=1)
                <= index_radii[candidate_sites] ** 2
            )
            candidate_pixels = candidate_pixels[intersects]
            candidate_sites = candidate_sites[intersects]
        counts = torch.bincount(candidate_pixels, minlength=polygon_count)
        indptr = torch.cat(
            (
                torch.zeros(1, dtype=torch.long, device=sites.device),
                torch.cumsum(counts, dim=0),
            )
        )
    return PolygonDiskCSR(
        indptr=indptr,
        indices=candidate_sites,
        polygon_ids=candidate_pixels,
        polygon_bin_memberships=polygon_memberships,
        site_bin_memberships=site_memberships,
        raw_candidate_pairs=raw_count,
        total_pairs=polygon_count * site_count,
        bin_size=width,
    )


def power_face_planes(
    sites: Tensor,
    radii: Tensor,
    edges: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return radical planes ``normal · x = offset`` for site-pair edges."""
    _validate_power_inputs(sites[:1], sites, radii)
    if edges.ndim != 2 or edges.shape[0] != 2 or edges.dtype != torch.long:
        raise ValueError("edges must be a torch.long tensor with shape (2, E)")
    if edges.device != sites.device:
        raise ValueError("edges and sites must be on the same device")
    if edges.numel() and (
        bool(torch.any(edges < 0)) or bool(torch.any(edges >= sites.shape[0]))
    ):
        raise ValueError("edges contain an out-of-range site index")
    left, right = edges[0], edges[1]
    left_sites = sites[left]
    right_sites = sites[right]
    normals = right_sites - left_sites
    offsets = 0.5 * (
        torch.sum(right_sites * right_sites, dim=-1)
        - torch.sum(left_sites * left_sites, dim=-1)
        + radii[left] * radii[left]
        - radii[right] * radii[right]
    )
    return normals, offsets


def _polygon_signed_area(vertices: Tensor) -> Tensor:
    following = torch.roll(vertices, shifts=-1, dims=0)
    return 0.5 * torch.sum(
        vertices[:, 0] * following[:, 1] - vertices[:, 1] * following[:, 0]
    )


def _clip_polygon_half_plane(
    vertices: Tensor,
    normal: Tensor,
    offset: Tensor,
) -> Tensor:
    """Clip one convex polygon by ``normal · x <= offset``."""
    if vertices.shape[0] == 0:
        return vertices
    values = vertices @ normal - offset
    scale = (
        torch.abs(offset)
        + torch.linalg.vector_norm(normal)
        * torch.max(torch.linalg.vector_norm(vertices, dim=1))
        + 1.0
    )
    tolerance = 64.0 * torch.finfo(vertices.dtype).eps * scale
    inside = values <= tolerance
    output: list[Tensor] = []
    for index in range(vertices.shape[0]):
        following = (index + 1) % vertices.shape[0]
        start = vertices[index]
        stop = vertices[following]
        start_inside = bool(inside[index].item())
        stop_inside = bool(inside[following].item())
        if start_inside:
            output.append(start)
        if start_inside != stop_inside:
            denominator = values[index] - values[following]
            fraction = torch.clamp(values[index] / denominator, 0.0, 1.0)
            output.append(start + fraction * (stop - start))
    if not output:
        return vertices.new_empty((0, 2))
    return torch.stack(output)


def _segment_circle_area_moment(
    start: Tensor,
    stop: Tensor,
    radius: Tensor,
) -> tuple[Tensor, Tensor]:
    """Return signed disk-intersection area and first moment for one edge."""
    delta = stop - start
    quadratic = torch.dot(delta, delta)
    if bool(quadratic == 0):
        return start.new_zeros(()), start.new_zeros((2,))
    linear = 2.0 * torch.dot(start, delta)
    constant = torch.dot(start, start) - radius * radius
    discriminant = linear * linear - 4.0 * quadratic * constant
    parameters = [start.new_zeros(()), start.new_ones(())]
    tolerance = (
        64.0
        * torch.finfo(start.dtype).eps
        * (torch.abs(linear * linear) + torch.abs(4.0 * quadratic * constant) + 1.0)
    )
    if bool(discriminant >= -tolerance):
        root = torch.sqrt(torch.clamp_min(discriminant, 0.0))
        for parameter in (
            (-linear - root) / (2.0 * quadratic),
            (-linear + root) / (2.0 * quadratic),
        ):
            if bool((parameter > 0.0) & (parameter < 1.0)):
                parameters.append(parameter)
    parameters.sort(key=lambda value: float(value.item()))

    area = start.new_zeros(())
    moment = start.new_zeros((2,))
    for lower, upper in zip(parameters[:-1], parameters[1:], strict=True):
        first = start + lower * delta
        second = start + upper * delta
        midpoint = 0.5 * (first + second)
        cross = first[0] * second[1] - first[1] * second[0]
        if bool(torch.dot(midpoint, midpoint) <= radius * radius + tolerance):
            triangle_area = 0.5 * cross
            area = area + triangle_area
            moment = moment + triangle_area * (first + second) / 3.0
        else:
            area = area + 0.5 * radius * radius * torch.atan2(
                cross,
                torch.dot(first, second),
            )
            first_unit = first / torch.linalg.vector_norm(first)
            second_unit = second / torch.linalg.vector_norm(second)
            moment = moment + (radius**3 / 3.0) * torch.stack(
                (
                    second_unit[1] - first_unit[1],
                    first_unit[0] - second_unit[0],
                )
            )
    return area, moment


def _polygon_disk_intersection_area_moment(
    vertices: Tensor,
    center: Tensor,
    radius: Tensor,
) -> tuple[Tensor, Tensor]:
    """Analytic area and global first moment of polygon intersected with a disk."""
    if vertices.shape[0] < 3 or bool(radius <= 0):
        return vertices.new_zeros(()), vertices.new_zeros((2,))
    shifted = vertices - center
    squared_distance = torch.sum(shifted * shifted, dim=1)
    containment_tolerance = (
        64.0
        * torch.finfo(vertices.dtype).eps
        * (radius * radius + torch.max(squared_distance) + 1.0)
    )
    area = vertices.new_zeros(())
    local_moment = vertices.new_zeros((2,))
    if bool(torch.all(squared_distance <= radius * radius + containment_tolerance)):
        following = torch.roll(shifted, shifts=-1, dims=0)
        triangle_areas = 0.5 * (
            shifted[:, 0] * following[:, 1] - shifted[:, 1] * following[:, 0]
        )
        area = torch.sum(triangle_areas)
        local_moment = torch.sum(
            triangle_areas[:, None] * (shifted + following) / 3.0,
            dim=0,
        )
    else:
        for index in range(shifted.shape[0]):
            edge_area, edge_moment = _segment_circle_area_moment(
                shifted[index],
                shifted[(index + 1) % shifted.shape[0]],
                radius,
            )
            area = area + edge_area
            local_moment = local_moment + edge_moment
    orientation = torch.where(area < 0, -torch.ones_like(area), torch.ones_like(area))
    positive_area = torch.abs(area)
    return positive_area, orientation * local_moment + center * positive_area


def _cell_clip_polygon(
    vertices: Tensor,
    *,
    cell: int,
    sites: Tensor,
    radii: Tensor,
    neighbors: Tensor,
) -> Tensor:
    clipped = vertices
    for raw_neighbor in neighbors:
        neighbor = int(raw_neighbor.item())
        if neighbor == cell:
            continue
        normal = sites[neighbor] - sites[cell]
        offset = 0.5 * (
            torch.dot(sites[neighbor], sites[neighbor])
            - torch.dot(sites[cell], sites[cell])
            + radii[cell] * radii[cell]
            - radii[neighbor] * radii[neighbor]
        )
        if bool(torch.all(normal == 0)):
            if bool(offset < 0) or (bool(offset == 0) and neighbor < cell):
                return vertices.new_empty((0, 2))
            continue
        clipped = _clip_polygon_half_plane(clipped, normal, offset)
        if clipped.shape[0] == 0:
            break
    return clipped


def _validate_csr(
    indptr: Tensor,
    indices: Tensor,
    *,
    rows: int,
    columns: int,
    name: str,
    device: torch.device,
) -> None:
    if (
        indptr.dtype != torch.long
        or indices.dtype != torch.long
        or indptr.device != device
        or indices.device != device
    ):
        raise ValueError(f"{name} CSR tensors must be torch.long on the input device")
    if indptr.shape != (rows + 1,) or indices.ndim != 1:
        raise ValueError(f"{name} CSR has an invalid shape")
    if (
        int(indptr[0].item()) != 0
        or int(indptr[-1].item()) != indices.numel()
        or bool(torch.any(indptr[1:] < indptr[:-1]))
    ):
        raise ValueError(f"{name} indptr must be monotonic and span indices")
    if indices.numel() and (
        bool(torch.any(indices < 0)) or bool(torch.any(indices >= columns))
    ):
        raise ValueError(f"{name} indices contain an out-of-range entry")
