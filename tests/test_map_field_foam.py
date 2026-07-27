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
    sparse_cech_graph,
    tile_disk_candidate_csr,
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


def test_sparse_cech_graph_matches_dense_oracle_and_builds_symmetric_csr() -> None:
    generator = torch.Generator().manual_seed(41)
    sites = 4.0 * torch.rand((64, 2), generator=generator, dtype=torch.float64)
    radii = 0.05 + 0.2 * torch.rand((64,), generator=generator, dtype=torch.float64)

    graph = sparse_cech_graph(sites, radii)
    repeated = sparse_cech_graph(sites, radii)
    expected = cech_adjacency(sites, radii, site_chunk_size=11)

    torch.testing.assert_close(graph.edges, expected)
    torch.testing.assert_close(repeated.candidate_pairs, graph.candidate_pairs)
    torch.testing.assert_close(repeated.indptr, graph.indptr)
    torch.testing.assert_close(repeated.indices, graph.indices)
    assert graph.candidate_pairs.shape[1] < graph.total_pairs
    assert graph.site_bin_memberships <= 4 * sites.shape[0]
    assert graph.indptr.shape == (sites.shape[0] + 1,)
    for site in range(sites.shape[0]):
        neighbors = graph.indices[graph.indptr[site] : graph.indptr[site + 1]]
        assert neighbors.tolist() == sorted(neighbors.tolist())
        for neighbor in neighbors.tolist():
            reverse = graph.indices[
                graph.indptr[neighbor] : graph.indptr[neighbor + 1]
            ]
            assert site in reverse.tolist()


def test_sparse_cech_graph_handles_adversarial_disk_geometry() -> None:
    sites = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, 0.0],
            [2.0, 0.0],
            [4.1, 0.0],
            [10.0, 10.0],
            [10.0, 13.0],
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([0.0, 0.0, 2.0, 0.1, 1.5, 1.5], dtype=torch.float64)

    dense = cech_adjacency(sites, radii, site_chunk_size=2)
    auto = sparse_cech_graph(sites, radii)
    fine = sparse_cech_graph(sites, radii, bin_size=0.25)

    torch.testing.assert_close(auto.edges, dense)
    torch.testing.assert_close(fine.edges, dense)
    assert [0, 1] in dense.transpose(0, 1).tolist()
    assert [2, 3] in dense.transpose(0, 1).tolist()
    assert [4, 5] in dense.transpose(0, 1).tolist()


def test_sparse_cech_graph_is_edge_invariant_under_similarity_transforms() -> None:
    generator = torch.Generator().manual_seed(7)
    sites = torch.randn((40, 2), generator=generator, dtype=torch.float64)
    radii = 0.1 + torch.rand((40,), generator=generator, dtype=torch.float64)
    angle = torch.tensor(0.63, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    reference = sparse_cech_graph(sites, radii).edges

    transformed = (3.7 * sites) @ rotation.T + torch.tensor([12.0, -5.0])
    actual = sparse_cech_graph(transformed, 3.7 * radii).edges

    torch.testing.assert_close(actual, reference)


def test_sparse_cech_graph_empty_singleton_and_limits() -> None:
    empty = sparse_cech_graph(torch.empty((0, 2)), torch.empty((0,)))
    assert empty.edges.shape == empty.candidate_pairs.shape == (2, 0)
    assert empty.indptr.tolist() == [0]

    singleton = sparse_cech_graph(torch.tensor([[1.0, 2.0]]), torch.tensor([0.0]))
    assert singleton.edges.shape == (2, 0)
    assert singleton.indptr.tolist() == [0, 0]

    sites = torch.tensor([[0.0, 0.0], [1.0, 1.0]])
    radii = torch.ones(2)
    with pytest.raises(ValueError, match="site-bin memberships"):
        sparse_cech_graph(
            sites,
            radii,
            bin_size=0.01,
            max_site_bin_memberships=10,
        )
    with pytest.raises(ValueError, match="raw candidate pairs"):
        sparse_cech_graph(
            torch.zeros((20, 2)),
            torch.ones(20),
            max_candidate_pairs=100,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_sparse_cech_graph_cuda_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(12)
    sites = torch.rand((128, 2), generator=generator)
    radii = 0.02 + 0.08 * torch.rand((128,), generator=generator)
    cpu = sparse_cech_graph(sites, radii)
    cuda = sparse_cech_graph(sites.cuda(), radii.cuda())

    torch.testing.assert_close(cuda.candidate_pairs.cpu(), cpu.candidate_pairs)
    torch.testing.assert_close(cuda.edges.cpu(), cpu.edges)
    torch.testing.assert_close(cuda.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(cuda.indices.cpu(), cpu.indices)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_sparse_cech_graph_mps_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(13)
    sites = torch.rand((128, 2), generator=generator)
    radii = 0.02 + 0.08 * torch.rand((128,), generator=generator)
    cpu = sparse_cech_graph(sites, radii)
    mps = sparse_cech_graph(sites.to("mps"), radii.to("mps"))

    torch.testing.assert_close(mps.candidate_pairs.cpu(), cpu.candidate_pairs)
    torch.testing.assert_close(mps.edges.cpu(), cpu.edges)
    torch.testing.assert_close(mps.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(mps.indices.cpu(), cpu.indices)


def _brute_force_tile_disk_lists(
    sites: torch.Tensor,
    radii: torch.Tensor,
    *,
    tile_shape: tuple[int, int],
    tile_origin: tuple[float, float],
    tile_size: tuple[float, float],
) -> list[list[int]]:
    rows, columns = tile_shape
    origin = torch.tensor(tile_origin, dtype=sites.dtype, device=sites.device)
    size = torch.tensor(tile_size, dtype=sites.dtype, device=sites.device)
    result: list[list[int]] = []
    for row in range(rows):
        for column in range(columns):
            lower = origin + size * torch.tensor(
                [column, row],
                dtype=sites.dtype,
                device=sites.device,
            )
            upper = lower + size
            closest = torch.minimum(torch.maximum(sites, lower), upper)
            intersects = torch.sum((sites - closest) ** 2, dim=1) <= radii**2
            result.append(torch.nonzero(intersects).flatten().tolist())
    return result


def test_tile_disk_candidate_csr_matches_brute_force_and_is_deterministic() -> None:
    generator = torch.Generator().manual_seed(73)
    sites = 8.0 * torch.rand((96, 2), generator=generator, dtype=torch.float64) - 1.0
    radii = 0.05 + 1.0 * torch.rand(
        (96,),
        generator=generator,
        dtype=torch.float64,
    )
    kwargs = {
        "tile_shape": (5, 7),
        "tile_origin": (-0.5, 0.25),
        "tile_size": (0.9, 1.2),
    }

    result = tile_disk_candidate_csr(sites, radii, **kwargs)
    repeated = tile_disk_candidate_csr(sites, radii, **kwargs)
    expected = _brute_force_tile_disk_lists(sites, radii, **kwargs)

    assert result.total_pairs == 5 * 7 * 96
    assert result.indices.numel() < result.total_pairs
    assert result.indptr.shape == (5 * 7 + 1,)
    torch.testing.assert_close(repeated.indptr, result.indptr)
    torch.testing.assert_close(repeated.indices, result.indices)
    torch.testing.assert_close(repeated.tile_ids, result.tile_ids)
    for tile_id, expected_sites in enumerate(expected):
        actual = result.indices[
            result.indptr[tile_id] : result.indptr[tile_id + 1]
        ]
        assert actual.tolist() == expected_sites


def test_tile_disk_candidate_csr_closed_boundaries_and_wide_support() -> None:
    sites = torch.tensor(
        [
            [1.0, 0.5],  # zero-radius point on an internal vertical boundary
            [0.5, 1.0],  # zero-radius point on an internal horizontal boundary
            [1.0, 1.0],  # shared four-tile corner
            [-2.0, -2.0],  # outside
            [1.0, 1.0],  # covers the complete grid
            [3.25, 0.5],  # outside but touches the outer boundary
        ],
        dtype=torch.float64,
    )
    radii = torch.tensor([0.0, 0.0, 0.0, 0.1, 2.0, 1.25], dtype=torch.float64)
    kwargs = {
        "tile_shape": (2, 2),
        "tile_origin": (0.0, 0.0),
        "tile_size": (1.0, 1.0),
    }

    result = tile_disk_candidate_csr(sites, radii, **kwargs)
    expected = _brute_force_tile_disk_lists(sites, radii, **kwargs)

    for tile_id, expected_sites in enumerate(expected):
        actual = result.indices[
            result.indptr[tile_id] : result.indptr[tile_id + 1]
        ]
        assert actual.tolist() == expected_sites
    assert all(4 in candidates for candidates in expected)
    assert all(3 not in candidates for candidates in expected)


def test_tile_disk_candidate_csr_empty_outside_scale_and_limits() -> None:
    empty = tile_disk_candidate_csr(
        torch.empty((0, 2)),
        torch.empty((0,)),
        tile_shape=(2, 3),
    )
    assert empty.indptr.tolist() == [0] * 7
    assert empty.indices.numel() == empty.tile_ids.numel() == 0

    outside = tile_disk_candidate_csr(
        torch.tensor([[-10.0, -10.0]]),
        torch.tensor([0.1]),
        tile_shape=(2, 3),
    )
    assert outside.indptr.tolist() == [0] * 7
    assert outside.indices.numel() == 0

    sites = torch.tensor([[0.2, 0.3], [1.0, 0.8], [1.8, 1.7]])
    radii = torch.tensor([0.2, 0.4, 0.1])
    reference = tile_disk_candidate_csr(
        sites,
        radii,
        tile_shape=(4, 4),
        tile_origin=(0.0, 0.0),
        tile_size=(0.5, 0.5),
    )
    scaled = tile_disk_candidate_csr(
        3.5 * sites + torch.tensor([7.0, -2.0]),
        3.5 * radii,
        tile_shape=(4, 4),
        tile_origin=(7.0, -2.0),
        tile_size=(1.75, 1.75),
    )
    torch.testing.assert_close(scaled.indptr, reference.indptr)
    torch.testing.assert_close(scaled.indices, reference.indices)

    with pytest.raises(ValueError, match="tile memberships"):
        tile_disk_candidate_csr(
            torch.tensor([[1.0, 1.0]]),
            torch.tensor([10.0]),
            tile_shape=(20, 20),
            max_aabb_memberships=100,
        )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_tile_disk_candidate_csr_cuda_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(17)
    sites = torch.rand((128, 2), generator=generator)
    radii = 0.01 + 0.2 * torch.rand((128,), generator=generator)
    cpu = tile_disk_candidate_csr(sites, radii, tile_shape=(8, 8))
    cuda = tile_disk_candidate_csr(
        sites.cuda(),
        radii.cuda(),
        tile_shape=(8, 8),
    )

    torch.testing.assert_close(cuda.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(cuda.indices.cpu(), cpu.indices)


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_tile_disk_candidate_csr_mps_matches_cpu() -> None:
    generator = torch.Generator().manual_seed(19)
    sites = torch.rand((128, 2), generator=generator)
    radii = 0.01 + 0.2 * torch.rand((128,), generator=generator)
    cpu = tile_disk_candidate_csr(sites, radii, tile_shape=(8, 8))
    mps = tile_disk_candidate_csr(
        sites.to("mps"),
        radii.to("mps"),
        tile_shape=(8, 8),
    )

    torch.testing.assert_close(mps.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(mps.indices.cpu(), cpu.indices)


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
