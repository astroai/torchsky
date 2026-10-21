"""Native inverse-mapped adaptive image reprojection plans."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import torch
from torch import Tensor

from ...domain import DomainSpec
from ...wcs import WCS
from .response import IndexedResponsePlan


AdaptiveValueSemantics = Literal["surface_brightness", "integrated_flux"]


@dataclass(frozen=True, slots=True)
class AdaptiveResampleConfig:
    """Declared accuracy and resource bounds for adaptive reprojection."""

    value_semantics: AdaptiveValueSemantics = "surface_brightness"
    max_edge_error_px: float = 0.02
    max_jacobian_variation: float = 0.02
    max_source_step_px: float = 0.5
    min_abs_jacobian_determinant: float = 1.0e-8
    max_depth: int = 8
    planning_chunk_size: int = 4096
    default_chunk_size: int = 65_536
    max_plan_entries: int = 50_000_000

    def __post_init__(self) -> None:
        if self.value_semantics not in {"surface_brightness", "integrated_flux"}:
            raise ValueError(
                "value_semantics must be 'surface_brightness' or 'integrated_flux'"
            )
        for name in (
            "max_edge_error_px",
            "max_jacobian_variation",
            "max_source_step_px",
            "min_abs_jacobian_determinant",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and positive")
        if int(self.max_depth) < 0:
            raise ValueError("max_depth must be non-negative")
        for name in ("planning_chunk_size", "default_chunk_size", "max_plan_entries"):
            if int(getattr(self, name)) <= 0:
                raise ValueError(f"{name} must be positive")


@dataclass(frozen=True, slots=True)
class _CellMetrics:
    mapped: Tensor
    jacobian: Tensor
    edge_error: Tensor
    jacobian_variation: Tensor
    determinant: Tensor


def _determinant_2x2(matrix: Tensor) -> Tensor:
    return matrix[..., 0, 0] * matrix[..., 1, 1] - matrix[..., 0, 1] * matrix[..., 1, 0]


def _convert_world_between_wcs(
    lon: Tensor,
    lat: Tensor,
    from_wcs: WCS,
    to_wcs: WCS,
) -> tuple[Tensor, Tensor]:
    from_frame = getattr(from_wcs, "celestial_frame", "ICRS")
    to_frame = getattr(to_wcs, "celestial_frame", "ICRS")
    from_equinox = getattr(from_wcs, "celestial_equinox", None)
    to_equinox = getattr(to_wcs, "celestial_equinox", None)
    if from_frame == to_frame and (
        from_equinox == to_equinox or from_frame in {"ICRS", "GALACTIC", "ECLIPTIC"}
    ):
        return lon, lat
    shape = lon.shape
    converted_lon, converted_lat = from_wcs.convert_celestial(
        lon.reshape(-1),
        lat.reshape(-1),
        to_frame,
        to_equinox=to_equinox,
    )
    return converted_lon.reshape(shape), converted_lat.reshape(shape)


def _mapped_points(
    points: Tensor,
    *,
    source_wcs: WCS,
    target_wcs: WCS,
) -> Tensor:
    lon, lat = target_wcs.pixel_to_world(points[:, 0], points[:, 1])
    lon, lat = _convert_world_between_wcs(lon, lat, target_wcs, source_wcs)
    x_source, y_source = source_wcs.world_to_pixel(lon, lat)
    mapped = torch.stack((x_source, y_source), dim=-1).to(
        device=points.device, dtype=points.dtype
    )
    if mapped.shape != points.shape or not bool(torch.isfinite(mapped).all()):
        raise RuntimeError(
            "Adaptive reprojection produced non-finite inverse coordinates"
        )
    return mapped


def _cell_metrics(
    cells: Tensor,
    *,
    source_wcs: WCS,
    target_wcs: WCS,
) -> _CellMetrics:
    x0, x1, y0, y1 = cells.unbind(dim=1)
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    xs = torch.stack((x0, xc, x1, x0, xc, x1, x0, xc, x1), dim=1)
    ys = torch.stack((y0, y0, y0, yc, yc, yc, y1, y1, y1), dim=1)
    points = torch.stack((xs, ys), dim=-1)
    mapped = _mapped_points(
        points.reshape(-1, 2), source_wcs=source_wcs, target_wcs=target_wcs
    ).reshape(points.shape)
    edge_errors = torch.stack(
        (
            torch.linalg.vector_norm(
                mapped[:, 1] - 0.5 * (mapped[:, 0] + mapped[:, 2]), dim=1
            ),
            torch.linalg.vector_norm(
                mapped[:, 7] - 0.5 * (mapped[:, 6] + mapped[:, 8]), dim=1
            ),
            torch.linalg.vector_norm(
                mapped[:, 3] - 0.5 * (mapped[:, 0] + mapped[:, 6]), dim=1
            ),
            torch.linalg.vector_norm(
                mapped[:, 5] - 0.5 * (mapped[:, 2] + mapped[:, 8]), dim=1
            ),
            torch.linalg.vector_norm(
                mapped[:, 4]
                - 0.25 * (mapped[:, 0] + mapped[:, 2] + mapped[:, 6] + mapped[:, 8]),
                dim=1,
            ),
        ),
        dim=1,
    )
    dx = (x1 - x0).unsqueeze(1)
    dy = (y1 - y0).unsqueeze(1)
    jacobian_x = (mapped[:, 5] - mapped[:, 3]) / dx
    jacobian_y = (mapped[:, 7] - mapped[:, 1]) / dy
    jacobian = torch.stack((jacobian_x, jacobian_y), dim=-1)

    quadrant_indices = ((0, 1, 3, 4), (1, 2, 4, 5), (3, 4, 6, 7), (4, 5, 7, 8))
    quadrant_jacobians = []
    for bottom_left, bottom_right, top_left, top_right in quadrant_indices:
        jx = (
            0.5 * (mapped[:, bottom_right] + mapped[:, top_right])
            - 0.5 * (mapped[:, bottom_left] + mapped[:, top_left])
        ) / (0.5 * dx)
        jy = (
            0.5 * (mapped[:, top_left] + mapped[:, top_right])
            - 0.5 * (mapped[:, bottom_left] + mapped[:, bottom_right])
        ) / (0.5 * dy)
        quadrant_jacobians.append(torch.stack((jx, jy), dim=-1))
    local_jacobians = torch.stack(quadrant_jacobians, dim=1)
    scale = torch.linalg.matrix_norm(jacobian, ord="fro").clamp_min(
        torch.finfo(jacobian.dtype).eps
    )
    variation = (
        torch.linalg.matrix_norm(
            local_jacobians - jacobian.unsqueeze(1), ord="fro"
        ).amax(dim=1)
        / scale
    )
    determinants = torch.cat(
        (
            _determinant_2x2(jacobian).unsqueeze(1),
            _determinant_2x2(local_jacobians),
        ),
        dim=1,
    )
    return _CellMetrics(
        mapped=mapped,
        jacobian=jacobian,
        edge_error=edge_errors.amax(dim=1),
        jacobian_variation=variation,
        determinant=determinants,
    )


def _validate_jacobians(
    determinants: Tensor,
    orientation: Tensor,
    *,
    minimum: float,
) -> None:
    if not bool(torch.isfinite(determinants).all()):
        raise RuntimeError("Adaptive reprojection encountered a non-finite Jacobian")
    oriented = determinants * orientation.unsqueeze(1)
    if bool((oriented <= float(minimum)).any()):
        raise RuntimeError(
            "Adaptive reprojection encountered a fold or singular target-to-source Jacobian"
        )


def _subdivide_cells(
    cells: Tensor, target_indices: Tensor, areas: Tensor
) -> tuple[Tensor, Tensor, Tensor]:
    x0, x1, y0, y1 = cells.unbind(dim=1)
    xc = 0.5 * (x0 + x1)
    yc = 0.5 * (y0 + y1)
    children = torch.stack(
        (
            torch.stack((x0, xc, y0, yc), dim=1),
            torch.stack((xc, x1, y0, yc), dim=1),
            torch.stack((x0, xc, yc, y1), dim=1),
            torch.stack((xc, x1, yc, y1), dim=1),
        ),
        dim=1,
    ).reshape(-1, 4)
    return (
        children,
        target_indices.unsqueeze(1).expand(-1, 4).reshape(-1),
        (0.25 * areas).unsqueeze(1).expand(-1, 4).reshape(-1),
    )


def _coalesce_entries(
    source_indices: Tensor,
    target_indices: Tensor,
    coefficients: Tensor,
    *,
    source_numel: int,
) -> tuple[Tensor, Tensor, Tensor]:
    if source_indices.numel() == 0:
        return source_indices, target_indices, coefficients
    keys = target_indices * int(source_numel) + source_indices
    order = torch.argsort(keys, stable=True)
    keys = keys[order]
    source_indices = source_indices[order]
    target_indices = target_indices[order]
    coefficients = coefficients[order]
    unique, inverse = torch.unique_consecutive(keys, return_inverse=True)
    combined = coefficients.new_zeros(unique.numel())
    combined.scatter_add_(0, inverse, coefficients)
    keep = combined != 0.0
    unique = unique[keep]
    return unique % int(source_numel), unique // int(source_numel), combined[keep]


def adaptive_reprojection_response(
    source_wcs: WCS,
    target_wcs: WCS,
    source_shape: tuple[int, int],
    target_shape: tuple[int, int],
    *,
    config: AdaptiveResampleConfig | None = None,
    dtype: torch.dtype = torch.float32,
    device: torch.device | str | None = None,
) -> IndexedResponsePlan:
    """Build one sparse adaptive response for science, variance, DQ, context, and adjoint."""
    cfg = config or AdaptiveResampleConfig()
    if dtype not in {torch.float32, torch.float64}:
        raise TypeError(
            "adaptive reprojection supports float32 and float64 response dtypes"
        )
    source_shape = tuple(int(value) for value in source_shape)
    target_shape = tuple(int(value) for value in target_shape)
    if (
        len(source_shape) != 2
        or len(target_shape) != 2
        or any(value <= 0 for value in (*source_shape, *target_shape))
    ):
        raise ValueError(
            "source_shape and target_shape must contain two positive dimensions"
        )
    resolved_device = torch.device(
        getattr(source_wcs, "device", "cpu") if device is None else device
    )
    for name, wcs in (("source_wcs", source_wcs), ("target_wcs", target_wcs)):
        wcs_device = torch.device(getattr(wcs, "device", resolved_device))
        if wcs_device != resolved_device:
            raise ValueError(f"{name} must be on the adaptive response device")
    coordinate_dtype = torch.float32 if resolved_device.type == "mps" else torch.float64
    source_numel = source_shape[0] * source_shape[1]
    target_numel = target_shape[0] * target_shape[1]
    all_source: list[Tensor] = []
    all_target: list[Tensor] = []
    all_coefficients: list[Tensor] = []
    raw_entries = 0

    for start in range(0, target_numel, int(cfg.planning_chunk_size)):
        stop = min(start + int(cfg.planning_chunk_size), target_numel)
        target_index = torch.arange(
            start, stop, dtype=torch.int64, device=resolved_device
        )
        x_center = (target_index % target_shape[1]).to(dtype=coordinate_dtype)
        y_center = torch.div(target_index, target_shape[1], rounding_mode="floor").to(
            dtype=coordinate_dtype
        )
        root_cells = torch.stack(
            (x_center - 0.5, x_center + 0.5, y_center - 0.5, y_center + 0.5),
            dim=1,
        )
        root_metrics = _cell_metrics(
            root_cells, source_wcs=source_wcs, target_wcs=target_wcs
        )
        root_det = root_metrics.determinant[:, 0]
        orientation = torch.sign(root_det)
        _validate_jacobians(
            root_metrics.determinant,
            orientation,
            minimum=float(cfg.min_abs_jacobian_determinant),
        )
        x_extent = torch.linalg.vector_norm(root_metrics.jacobian[:, :, 0], dim=1)
        y_extent = torch.linalg.vector_norm(root_metrics.jacobian[:, :, 1], dim=1)
        nx = (
            torch.ceil(x_extent / float(cfg.max_source_step_px))
            .to(torch.int64)
            .clamp_min(1)
        )
        ny = (
            torch.ceil(y_extent / float(cfg.max_source_step_px))
            .to(torch.int64)
            .clamp_min(1)
        )

        base_cells: list[Tensor] = []
        base_targets: list[Tensor] = []
        base_areas: list[Tensor] = []
        base_orientations: list[Tensor] = []
        for local_index in range(stop - start):
            count_x = int(nx[local_index].item())
            count_y = int(ny[local_index].item())
            x_edges = torch.linspace(
                root_cells[local_index, 0],
                root_cells[local_index, 1],
                count_x + 1,
                dtype=coordinate_dtype,
                device=resolved_device,
            )
            y_edges = torch.linspace(
                root_cells[local_index, 2],
                root_cells[local_index, 3],
                count_y + 1,
                dtype=coordinate_dtype,
                device=resolved_device,
            )
            yy0, xx0 = torch.meshgrid(y_edges[:-1], x_edges[:-1], indexing="ij")
            yy1, xx1 = torch.meshgrid(y_edges[1:], x_edges[1:], indexing="ij")
            count = count_x * count_y
            base_cells.append(
                torch.stack(
                    (
                        xx0.reshape(-1),
                        xx1.reshape(-1),
                        yy0.reshape(-1),
                        yy1.reshape(-1),
                    ),
                    dim=1,
                )
            )
            base_targets.append(target_index[local_index].expand(count))
            base_areas.append(
                torch.full(
                    (count,),
                    1.0 / float(count),
                    dtype=coordinate_dtype,
                    device=resolved_device,
                )
            )
            base_orientations.append(orientation[local_index].expand(count))

        cells = torch.cat(base_cells, dim=0)
        cell_targets = torch.cat(base_targets)
        cell_areas = torch.cat(base_areas)
        cell_orientations = torch.cat(base_orientations)
        leaf_mapped: list[Tensor] = []
        leaf_targets: list[Tensor] = []
        leaf_weights: list[Tensor] = []
        for depth in range(int(cfg.max_depth) + 1):
            metrics = _cell_metrics(cells, source_wcs=source_wcs, target_wcs=target_wcs)
            _validate_jacobians(
                metrics.determinant,
                cell_orientations,
                minimum=float(cfg.min_abs_jacobian_determinant),
            )
            refine = (metrics.edge_error > float(cfg.max_edge_error_px)) | (
                metrics.jacobian_variation > float(cfg.max_jacobian_variation)
            )
            accept = ~refine
            if bool(accept.any()):
                leaf_mapped.append(metrics.mapped[accept, 4])
                leaf_targets.append(cell_targets[accept])
                weights = cell_areas[accept]
                if cfg.value_semantics == "integrated_flux":
                    weights = weights * torch.abs(metrics.determinant[accept, 0])
                leaf_weights.append(weights)
            if not bool(refine.any()):
                break
            if depth >= int(cfg.max_depth):
                raise RuntimeError(
                    "Adaptive reprojection could not resolve the declared edge/Jacobian bounds"
                )
            cells, cell_targets, cell_areas = _subdivide_cells(
                cells[refine], cell_targets[refine], cell_areas[refine]
            )
            cell_orientations = (
                cell_orientations[refine].unsqueeze(1).expand(-1, 4).reshape(-1)
            )
            if raw_entries + 4 * int(cells.shape[0]) > int(cfg.max_plan_entries):
                raise RuntimeError("Adaptive reprojection exceeded max_plan_entries")

        mapped = torch.cat(leaf_mapped, dim=0)
        sample_targets = torch.cat(leaf_targets)
        sample_weights = torch.cat(leaf_weights).to(dtype=dtype)
        x = mapped[:, 0]
        y = mapped[:, 1]
        x0 = torch.floor(x).to(torch.int64)
        y0 = torch.floor(y).to(torch.int64)
        chunk_source: list[Tensor] = []
        chunk_target: list[Tensor] = []
        chunk_coefficients: list[Tensor] = []
        for offset_y, offset_x in ((0, 0), (0, 1), (1, 0), (1, 1)):
            xi = x0 + offset_x
            yi = y0 + offset_y
            coefficient = (
                (1.0 - torch.abs(x - xi.to(x.dtype)))
                * (1.0 - torch.abs(y - yi.to(y.dtype)))
                * sample_weights.to(dtype=x.dtype)
            )
            valid = (
                (xi >= 0)
                & (xi < source_shape[1])
                & (yi >= 0)
                & (yi < source_shape[0])
                & (coefficient != 0.0)
            )
            chunk_source.append(yi[valid] * source_shape[1] + xi[valid])
            chunk_target.append(sample_targets[valid])
            chunk_coefficients.append(coefficient[valid].to(dtype=dtype))
        source_entries = torch.cat(chunk_source)
        target_entries = torch.cat(chunk_target)
        coefficient_entries = torch.cat(chunk_coefficients)
        raw_entries += int(source_entries.numel())
        if raw_entries > int(cfg.max_plan_entries):
            raise RuntimeError("Adaptive reprojection exceeded max_plan_entries")
        all_source.append(source_entries)
        all_target.append(target_entries)
        all_coefficients.append(coefficient_entries)

    source_indices, target_indices, coefficients = _coalesce_entries(
        torch.cat(all_source),
        torch.cat(all_target),
        torch.cat(all_coefficients),
        source_numel=source_numel,
    )
    if int(source_indices.numel()) > int(cfg.max_plan_entries):
        raise RuntimeError("Adaptive reprojection exceeded max_plan_entries")
    return IndexedResponsePlan(
        DomainSpec(source_shape, dtype),
        DomainSpec(target_shape, dtype),
        source_indices,
        target_indices,
        coefficients,
        default_chunk_size=int(cfg.default_chunk_size),
    )


__all__ = [
    "AdaptiveResampleConfig",
    "AdaptiveValueSemantics",
    "adaptive_reprojection_response",
]
