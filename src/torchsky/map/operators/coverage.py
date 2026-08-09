"""Certified calibration-coverage diagnostics for learned operators."""

from __future__ import annotations

from dataclasses import dataclass
import math

import torch
from torch import Tensor

__all__ = ["SpatialFillDistanceBounds", "spatial_fill_distance_bounds"]


@dataclass(frozen=True)
class SpatialFillDistanceBounds:
    """Certified bounds on distance to the nearest 2D calibration point."""

    lower: float
    upper: float
    domain_bounds: tuple[float, float, float, float]
    grid_shape: tuple[int, int]
    point_count: int

    @property
    def absolute_error_bound(self) -> float:
        return self.upper - self.lower

    @property
    def conservative_minimum_feature_length(self) -> float:
        """Twice the upper fill-distance bound, in the input coordinate unit."""
        return 2.0 * self.upper

    def as_dict(self) -> dict[str, float | int | list[float] | list[int]]:
        return {
            "lower": self.lower,
            "upper": self.upper,
            "absolute_error_bound": self.absolute_error_bound,
            "conservative_minimum_feature_length": (
                self.conservative_minimum_feature_length
            ),
            "domain_bounds": list(self.domain_bounds),
            "grid_shape": list(self.grid_shape),
            "point_count": self.point_count,
        }


def spatial_fill_distance_bounds(
    points_xy: Tensor,
    domain_bounds: tuple[float, float, float, float],
    *,
    rtol: float = 0.01,
    atol: float = 0.0,
    initial_long_axis_cells: int = 16,
    maximum_grid_points: int = 1_048_576,
    chunk_size: int = 65_536,
) -> SpatialFillDistanceBounds:
    """Bound the rectangular-domain fill distance of scattered 2D samples.

    If ``d(x)`` is distance from ``x`` to its nearest calibration point, then
    ``d`` is 1-Lipschitz. Evaluating every grid-cell centre gives a lower bound
    on ``max d``; adding the cell half-diagonal gives an upper bound. The grid
    is refined until that certified interval meets ``atol + rtol * upper``.

    The conservative minimum feature length is a coverage guard, not a
    scattered-sampling Nyquist theorem or a physical correlation length.
    """
    relative_tolerance = float(rtol)
    absolute_tolerance = float(atol)
    initial_cells = int(initial_long_axis_cells)
    maximum_points = int(maximum_grid_points)
    distance_chunk = int(chunk_size)
    if (
        not math.isfinite(relative_tolerance)
        or relative_tolerance < 0.0
        or not math.isfinite(absolute_tolerance)
        or absolute_tolerance < 0.0
        or relative_tolerance == absolute_tolerance == 0.0
    ):
        raise ValueError("fill-distance tolerances must be finite and non-negative")
    if initial_cells < 1 or maximum_points < 1 or distance_chunk < 1:
        raise ValueError("fill-distance grid and chunk sizes must be positive")

    xmin, xmax, ymin, ymax = map(float, domain_bounds)
    if not all(math.isfinite(value) for value in (xmin, xmax, ymin, ymax)):
        raise ValueError("domain bounds must be finite")
    width = xmax - xmin
    height = ymax - ymin
    if width <= 0.0 or height <= 0.0:
        raise ValueError("domain bounds must have positive x and y extent")

    points = (
        torch.as_tensor(points_xy)
        .detach()
        .to(device="cpu", dtype=torch.float64)
    )
    if points.ndim != 2 or points.shape[1] != 2 or points.shape[0] < 1:
        raise ValueError("points_xy must have shape (n, 2) with n >= 1")
    if not bool(torch.isfinite(points).all()):
        raise ValueError("calibration points must be finite")
    inside = (
        (points[:, 0] >= xmin)
        & (points[:, 0] <= xmax)
        & (points[:, 1] >= ymin)
        & (points[:, 1] <= ymax)
    )
    if not bool(inside.all()):
        raise ValueError("calibration points must lie inside domain_bounds")
    points = torch.unique(points, dim=0)

    if width >= height:
        nx = initial_cells
        ny = max(1, math.ceil(initial_cells * height / width))
    else:
        ny = initial_cells
        nx = max(1, math.ceil(initial_cells * width / height))

    while True:
        grid_points = nx * ny
        if grid_points > maximum_points:
            raise RuntimeError(
                "fill-distance tolerance was not reached before maximum_grid_points"
            )
        dx = width / nx
        dy = height / ny
        xs = torch.linspace(
            xmin + 0.5 * dx,
            xmax - 0.5 * dx,
            nx,
            dtype=torch.float64,
        )
        ys = torch.linspace(
            ymin + 0.5 * dy,
            ymax - 0.5 * dy,
            ny,
            dtype=torch.float64,
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        centers = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=1)
        lower = 0.0
        for start in range(0, centers.shape[0], distance_chunk):
            distances = torch.cdist(centers[start : start + distance_chunk], points)
            lower = max(lower, float(distances.min(dim=1).values.max()))
        radius = 0.5 * math.hypot(dx, dy)
        upper = lower + radius
        if radius <= absolute_tolerance + relative_tolerance * upper:
            return SpatialFillDistanceBounds(
                lower=lower,
                upper=upper,
                domain_bounds=(xmin, xmax, ymin, ymax),
                grid_shape=(ny, nx),
                point_count=int(points.shape[0]),
            )
        # ponytail: uniform refinement is predictable and certified; switch to
        # adaptive cells only if calibration manifests make this setup cost real.
        nx *= 2
        ny *= 2
