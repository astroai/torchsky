"""Exact contracts for bounded power-diagram field primitives."""

from __future__ import annotations

import pytest
import torch

from torchsky.map.field.foam import (
    bounded_power_ownership,
    cech_adjacency,
    power_distance,
    power_face_planes,
    soft_bounded_power_weights,
)


def test_equal_radius_ownership_matches_nearest_site_and_bounds() -> None:
    sites = torch.tensor([[-1.0, 0.0], [1.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([2.0, 2.0], dtype=torch.float64)
    points = torch.tensor(
        [[-1.5, 0.0], [-0.2, 0.0], [0.2, 0.0], [4.0, 0.0]],
        dtype=torch.float64,
    )

    owners, minimum = bounded_power_ownership(
        points, sites, radii, point_chunk_size=2
    )

    assert owners.tolist() == [0, 0, 1, -1]
    torch.testing.assert_close(
        minimum, power_distance(points, sites, radii).amin(dim=-1)
    )


def test_learned_radius_moves_the_radical_boundary() -> None:
    sites = torch.tensor([[0.0], [2.0]], dtype=torch.float64)
    radii = torch.tensor([1.0, 2.0], dtype=torch.float64)
    edges = torch.tensor([[0], [1]], dtype=torch.long)
    normals, offsets = power_face_planes(sites, radii, edges)
    boundary = offsets / normals.squeeze(-1)

    torch.testing.assert_close(boundary, torch.tensor([0.25], dtype=torch.float64))
    at_boundary = boundary[:, None]
    distances = power_distance(at_boundary, sites, radii)
    torch.testing.assert_close(distances[:, 0], distances[:, 1])


def test_soft_bounded_weights_partition_union_and_are_differentiable() -> None:
    sites = torch.tensor(
        [[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64, requires_grad=True
    )
    radii = torch.tensor([1.0, 1.0], dtype=torch.float64, requires_grad=True)
    points = torch.tensor(
        [[-0.25, 0.1], [0.25, -0.1], [3.0, 0.0]],
        dtype=torch.float64,
        requires_grad=True,
    )

    weights = soft_bounded_power_weights(points, sites, radii, temperature=7.0)

    torch.testing.assert_close(weights[:2].sum(dim=-1), torch.ones(2, dtype=weights.dtype))
    torch.testing.assert_close(weights[2], torch.zeros(2, dtype=weights.dtype))
    loss = (weights[:2] * torch.tensor([1.0, 2.0])).sum()
    loss.backward()
    for parameter in (points, sites, radii):
        assert parameter.grad is not None
        assert bool(torch.isfinite(parameter.grad).all())


def test_cech_adjacency_is_exact_and_chunk_invariant() -> None:
    sites = torch.tensor([[0.0, 0.0], [1.5, 0.0], [3.1, 0.0], [0.0, 3.0]])
    radii = torch.ones(4)
    expected = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    for chunk_size in (1, 2, 8):
        edges = cech_adjacency(sites, radii, site_chunk_size=chunk_size)
        torch.testing.assert_close(edges, expected)


def test_power_input_validation_fails_closed() -> None:
    sites = torch.zeros((2, 2))
    points = torch.zeros((1, 2))
    with pytest.raises(ValueError, match="non-negative"):
        power_distance(points, sites, torch.tensor([1.0, -1.0]))
    with pytest.raises(ValueError, match="positive"):
        bounded_power_ownership(points, sites, torch.ones(2), point_chunk_size=0)
    with pytest.raises(ValueError, match="temperature"):
        soft_bounded_power_weights(points, sites, torch.ones(2), temperature=0.0)


def test_empty_ownership_query_preserves_shape() -> None:
    points = torch.empty((2, 0, 2), dtype=torch.float64)
    sites = torch.zeros((1, 2), dtype=torch.float64)
    owners, minimum = bounded_power_ownership(
        points, sites, torch.ones(1, dtype=torch.float64)
    )
    assert owners.shape == minimum.shape == (2, 0)
