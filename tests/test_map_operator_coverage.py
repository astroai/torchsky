from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.operators import spatial_fill_distance_bounds


@pytest.mark.parametrize(
    ("points", "expected"),
    [
        ([[0.5, 0.5]], math.sqrt(0.5)),
        ([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], math.sqrt(0.5)),
    ],
)
def test_spatial_fill_distance_certifies_analytic_square_cases(
    points: list[list[float]], expected: float
) -> None:
    result = spatial_fill_distance_bounds(
        torch.tensor(points),
        (0.0, 1.0, 0.0, 1.0),
        rtol=0.01,
    )

    assert result.lower <= expected <= result.upper
    assert result.absolute_error_bound <= 0.01 * result.upper
    assert result.conservative_minimum_feature_length == 2.0 * result.upper


def test_spatial_fill_distance_is_unit_invariant() -> None:
    points = torch.tensor([[0.2, 0.3], [0.8, 0.7], [0.4, 0.9]])
    base = spatial_fill_distance_bounds(points, (0.0, 1.0, 0.0, 1.0))
    scaled = spatial_fill_distance_bounds(
        17.0 * points,
        (0.0, 17.0, 0.0, 17.0),
    )

    assert scaled.grid_shape == base.grid_shape
    assert scaled.lower == pytest.approx(17.0 * base.lower)
    assert scaled.upper == pytest.approx(17.0 * base.upper)


def test_spatial_fill_distance_fails_closed() -> None:
    with pytest.raises(ValueError, match="inside"):
        spatial_fill_distance_bounds(
            torch.tensor([[2.0, 0.5]]),
            (0.0, 1.0, 0.0, 1.0),
        )
    with pytest.raises(RuntimeError, match="maximum_grid_points"):
        spatial_fill_distance_bounds(
            torch.tensor([[0.5, 0.5]]),
            (0.0, 1.0, 0.0, 1.0),
            rtol=1.0e-9,
            maximum_grid_points=16,
        )
