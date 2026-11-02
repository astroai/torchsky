"""Fail-closed bounded subdivision helpers."""

from __future__ import annotations

from collections.abc import Callable, Sequence

import torch
from torch import Tensor


PointMap = Callable[[Tensor], Tensor]


def bounded_subdivide_segment(
    start: Tensor,
    stop: Tensor,
    mapped_start: Tensor,
    mapped_stop: Tensor,
    *,
    map_points: PointMap,
    tolerance: float,
    max_depth: int,
    depth: int = 0,
) -> list[Tensor]:
    """Return the mapped segment endpoint sequence within a chord-error bound."""
    fractions = start.new_tensor((0.25, 0.5, 0.75))
    points = start.unsqueeze(0) + fractions.unsqueeze(1) * (stop - start).unsqueeze(0)
    mapped = map_points(points)
    if mapped.shape != points.shape or not bool(torch.isfinite(mapped).all()):
        raise RuntimeError("Bounded subdivision produced non-finite mapped coordinates")
    linear = mapped_start.unsqueeze(0) + fractions.unsqueeze(1) * (
        mapped_stop - mapped_start
    ).unsqueeze(0)
    error = torch.linalg.vector_norm(mapped - linear, dim=1).amax()
    if float(error) <= float(tolerance):
        return [mapped_stop]
    if depth >= int(max_depth):
        raise RuntimeError(
            "Could not resolve mapped boundary within the requested tolerance"
        )
    midpoint = points[1]
    mapped_midpoint = mapped[1]
    return bounded_subdivide_segment(
        start,
        midpoint,
        mapped_start,
        mapped_midpoint,
        map_points=map_points,
        tolerance=tolerance,
        max_depth=max_depth,
        depth=depth + 1,
    ) + bounded_subdivide_segment(
        midpoint,
        stop,
        mapped_midpoint,
        mapped_stop,
        map_points=map_points,
        tolerance=tolerance,
        max_depth=max_depth,
        depth=depth + 1,
    )


def bounded_subdivide_boundary(
    corners: Sequence[Tensor],
    *,
    map_points: PointMap,
    tolerance: float,
    max_depth: int,
) -> Tensor:
    """Map a closed polygon boundary with bounded chord error."""
    if len(corners) < 3:
        raise ValueError("bounded subdivision requires at least three corners")
    if not torch.isfinite(torch.tensor(float(tolerance))) or float(tolerance) <= 0.0:
        raise ValueError("tolerance must be finite and positive")
    if int(max_depth) < 0:
        raise ValueError("max_depth must be non-negative")
    corner_tensor = torch.stack(tuple(corners), dim=0)
    mapped_corners = map_points(corner_tensor)
    if mapped_corners.shape != corner_tensor.shape or not bool(
        torch.isfinite(mapped_corners).all()
    ):
        raise RuntimeError("Bounded subdivision produced non-finite corner coordinates")
    output = [mapped_corners[0]]
    for index in range(len(corners)):
        output.extend(
            bounded_subdivide_segment(
                corner_tensor[index],
                corner_tensor[(index + 1) % len(corners)],
                mapped_corners[index],
                mapped_corners[(index + 1) % len(corners)],
                map_points=map_points,
                tolerance=float(tolerance),
                max_depth=int(max_depth),
            )
        )
    return torch.stack(output, dim=0)


__all__ = ["bounded_subdivide_boundary", "bounded_subdivide_segment"]
