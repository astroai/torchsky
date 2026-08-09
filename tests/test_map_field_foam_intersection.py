"""Exact bounded-power-cell/pixel intersection contracts."""

from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.field import (
    batched_bounded_power_pixel_operator,
    bounded_power_ownership,
    bounded_power_pixel_operator,
    smooth_bounded_power_weights,
    sparse_cech_graph,
    tile_disk_candidate_csr,
)


def _rectangle(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    *,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    return torch.tensor(
        [[x0, y0], [x1, y0], [x1, y1], [x0, y1]],
        dtype=dtype,
    )


def test_single_cell_matches_analytic_circle_and_rectangle_areas() -> None:
    sites = torch.tensor([[0.0, 0.0]], dtype=torch.float64)
    radii = torch.tensor([0.5], dtype=torch.float64)
    pixels = torch.stack(
        (
            _rectangle(-1.0, -1.0, 1.0, 1.0),
            _rectangle(-0.1, -0.2, 0.1, 0.2),
            _rectangle(0.0, -1.0, 1.0, 1.0),
            _rectangle(0.5, 0.0, 1.5, 1.0),
        )
    )

    operator = bounded_power_pixel_operator(pixels, sites, radii)

    torch.testing.assert_close(
        operator.areas,
        torch.tensor(
            [math.pi * 0.25, 0.08, math.pi * 0.125],
            dtype=torch.float64,
        ),
        rtol=2e-13,
        atol=2e-13,
    )
    torch.testing.assert_close(
        operator.cell_areas,
        torch.tensor([math.pi * 0.25], dtype=torch.float64),
        rtol=2e-13,
        atol=2e-13,
    )
    integrated = operator.matvec(
        torch.tensor([7.0], dtype=torch.float64),
        semantics="integrated_flux",
    )
    torch.testing.assert_close(integrated[0], torch.tensor(7.0, dtype=torch.float64))
    assert integrated[-1] == 0


def test_float32_tiny_pixel_inside_large_cell_survives_area_tolerance() -> None:
    sites = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    radii = torch.tensor([10.0], dtype=torch.float32)
    pixels = _rectangle(-0.005, -0.005, 0.005, 0.005, dtype=torch.float32)[None]
    candidates = torch.tensor([0], dtype=torch.long)
    candidate_indptr = torch.tensor([0, 1], dtype=torch.long)
    neighbor_indptr = torch.tensor([0, 0], dtype=torch.long)
    neighbors = torch.empty(0, dtype=torch.long)

    scalar = bounded_power_pixel_operator(pixels, sites, radii)
    batched = batched_bounded_power_pixel_operator(
        pixels,
        sites,
        radii,
        candidate_indptr=candidate_indptr,
        candidate_indices=candidates,
        neighbor_indptr=neighbor_indptr,
        neighbor_indices=neighbors,
    )

    expected = torch.tensor([1.0e-4], dtype=torch.float32)
    torch.testing.assert_close(scalar.areas, expected, rtol=2e-5, atol=1e-8)
    torch.testing.assert_close(batched.areas, expected, rtol=2e-5, atol=1e-8)


def test_smooth_radius_training_hands_off_to_exact_pixel_inference() -> None:
    axis = torch.linspace(-1.0, 1.0, 31, dtype=torch.float64)
    grid_y, grid_x = torch.meshgrid(axis, axis, indexing="ij")
    points = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    site = torch.zeros((1, 2), dtype=torch.float64)
    true_radius = torch.tensor([0.72], dtype=torch.float64)
    target = smooth_bounded_power_weights(
        points,
        site,
        true_radius,
        support_temperature=18.0,
    ).detach()
    learned_radius = torch.tensor([0.3], dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([learned_radius], lr=0.025)

    for _ in range(120):
        optimizer.zero_grad()
        prediction = smooth_bounded_power_weights(
            points,
            site,
            learned_radius,
            support_temperature=18.0,
        )
        torch.mean(torch.square(prediction - target)).backward()
        optimizer.step()

    exact = bounded_power_pixel_operator(
        _rectangle(-1.0, -1.0, 1.0, 1.0).unsqueeze(0),
        site,
        learned_radius.detach(),
    )
    target_area = math.pi * true_radius.square()
    torch.testing.assert_close(learned_radius.detach(), true_radius, atol=2e-3, rtol=0)
    torch.testing.assert_close(exact.areas, target_area, atol=1e-2, rtol=0)


def test_radical_half_planes_partition_pixel_and_break_coincident_ties() -> None:
    pixel = _rectangle(-1.0, -0.5, 1.0, 0.5).unsqueeze(0)
    sites = torch.tensor([[-0.5, 0.0], [0.5, 0.0]], dtype=torch.float64)
    radii = torch.full((2,), 2.0, dtype=torch.float64)

    partition = bounded_power_pixel_operator(pixel, sites, radii)

    assert partition.indices.tolist() == [0, 1]
    torch.testing.assert_close(partition.areas, torch.ones(2, dtype=torch.float64))
    torch.testing.assert_close(
        partition.areas.sum(),
        torch.tensor(2.0, dtype=torch.float64),
    )

    coincident = bounded_power_pixel_operator(
        _rectangle(-1.0, -1.0, 1.0, 1.0).unsqueeze(0),
        torch.zeros((2, 2), dtype=torch.float64),
        torch.ones(2, dtype=torch.float64),
    )
    assert coincident.indices.tolist() == [0]
    torch.testing.assert_close(
        coincident.areas,
        torch.tensor([math.pi], dtype=torch.float64),
    )
    assert coincident.cell_areas[1] == 0


@pytest.mark.parametrize("semantics", ["surface_brightness", "integrated_flux"])
def test_forward_transpose_dot_product_and_tile_chunk_invariance(
    semantics: str,
) -> None:
    sites = torch.tensor(
        [[-0.7, -0.1], [0.2, 0.3], [0.8, -0.2]],
        dtype=torch.float64,
    )
    radii = torch.tensor([0.9, 0.8, 0.7], dtype=torch.float64)
    pixels = torch.stack(
        (
            _rectangle(-1.0, -0.8, 0.0, 0.8),
            _rectangle(0.0, -0.8, 1.0, 0.8),
        )
    )
    graph = sparse_cech_graph(sites, radii)
    operator = bounded_power_pixel_operator(
        pixels,
        sites,
        radii,
        neighbor_indptr=graph.indptr,
        neighbor_indices=graph.indices,
    )
    cell_values = torch.tensor([0.3, -1.2, 2.1], dtype=torch.float64)
    pixel_values = torch.tensor([1.7, -0.4], dtype=torch.float64)

    forward = operator.matvec(cell_values, semantics=semantics)
    transpose = operator.rmatvec(pixel_values, semantics=semantics)

    torch.testing.assert_close(
        torch.dot(forward, pixel_values),
        torch.dot(cell_values, transpose),
        rtol=2e-13,
        atol=2e-13,
    )
    chunked = torch.cat(
        [
            bounded_power_pixel_operator(
                pixels[index : index + 1],
                sites,
                radii,
                neighbor_indptr=graph.indptr,
                neighbor_indices=graph.indices,
            ).matvec(cell_values, semantics=semantics)
            for index in range(2)
        ]
    )
    torch.testing.assert_close(chunked, forward)


def test_sparse_candidates_are_canonical_and_preserve_exact_area() -> None:
    sites = torch.tensor(
        [[-0.5, 0.0], [0.5, 0.0], [8.0, 8.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0, 0.1], dtype=torch.float64)
    pixels = _rectangle(-1.0, -1.0, 1.0, 1.0).unsqueeze(0)
    dense = bounded_power_pixel_operator(pixels, sites, radii)
    sparse = bounded_power_pixel_operator(
        pixels,
        sites,
        radii,
        candidate_indptr=torch.tensor([0, 3], dtype=torch.long),
        candidate_indices=torch.tensor([1, 0, 1], dtype=torch.long),
    )

    assert sparse.candidate_pairs == 3
    assert sparse.indices.tolist() == [0, 1]
    torch.testing.assert_close(sparse.areas, dense.areas)


def test_similarity_and_site_permutation_invariance() -> None:
    sites = torch.tensor(
        [[-0.6, -0.2], [0.4, 0.4], [0.7, -0.5]],
        dtype=torch.float64,
    )
    radii = torch.tensor([0.8, 0.7, 0.6], dtype=torch.float64)
    pixels = torch.stack(
        (
            _rectangle(-1.0, -1.0, 0.0, 1.0),
            _rectangle(0.0, -1.0, 1.0, 1.0),
        )
    )
    values = torch.tensor([2.0, -0.3, 1.2], dtype=torch.float64)
    reference = bounded_power_pixel_operator(pixels, sites, radii)
    reference_render = reference.matvec(values)

    angle = torch.tensor(0.61, dtype=torch.float64)
    rotation = torch.stack(
        (
            torch.stack((torch.cos(angle), -torch.sin(angle))),
            torch.stack((torch.sin(angle), torch.cos(angle))),
        )
    )
    scale = 3.2
    shift = torch.tensor([7.0, -4.0], dtype=torch.float64)
    transformed = bounded_power_pixel_operator(
        scale * pixels @ rotation.T + shift,
        scale * sites @ rotation.T + shift,
        scale * radii,
    )
    torch.testing.assert_close(
        transformed.matvec(values),
        scale**2 * reference_render,
        rtol=2e-12,
        atol=2e-12,
    )

    permutation = torch.tensor([2, 0, 1])
    permuted = bounded_power_pixel_operator(
        pixels,
        sites[permutation],
        radii[permutation],
    )
    torch.testing.assert_close(
        permuted.matvec(values[permutation]),
        reference_render,
        rtol=2e-13,
        atol=2e-13,
    )


def test_randomized_areas_match_independent_dense_ownership_oracle() -> None:
    sites = torch.tensor(
        [[-0.55, -0.15], [0.15, 0.35], [0.55, -0.35]],
        dtype=torch.float64,
    )
    radii = torch.tensor([0.85, 0.75, 0.65], dtype=torch.float64)
    pixel = _rectangle(-1.0, -1.0, 1.0, 1.0).unsqueeze(0)
    operator = bounded_power_pixel_operator(pixel, sites, radii)

    resolution = 1024
    axis = -1.0 + (torch.arange(resolution, dtype=torch.float64) + 0.5) * (
        2.0 / resolution
    )
    y, x = torch.meshgrid(axis, axis, indexing="ij")
    owners, _ = bounded_power_ownership(
        torch.stack((x.flatten(), y.flatten()), dim=1),
        sites,
        radii,
        point_chunk_size=131_072,
    )
    numerical = (
        torch.bincount(owners[owners >= 0], minlength=sites.shape[0]).to(torch.float64)
        * (2.0 / resolution) ** 2
    )
    exact = torch.zeros_like(numerical)
    exact[operator.indices] = operator.areas

    torch.testing.assert_close(exact, numerical, rtol=0.0, atol=3.5e-3)
    assert float(exact.sum()) <= 4.0


def test_operator_validation_and_empty_contract() -> None:
    sites = torch.empty((0, 2), dtype=torch.float64)
    radii = torch.empty((0,), dtype=torch.float64)
    empty = bounded_power_pixel_operator(
        torch.empty((0, 4, 2), dtype=torch.float64),
        sites,
        radii,
    )
    assert empty.indptr.tolist() == [0]
    assert empty.transpose_indptr.tolist() == [0]
    assert empty.areas.numel() == 0

    with pytest.raises(ValueError, match="convex"):
        bounded_power_pixel_operator(
            torch.tensor(
                [[[0.0, 0.0], [1.0, 0.0], [0.2, 0.2], [1.0, 1.0], [0.0, 1.0]]],
                dtype=torch.float64,
            ),
            torch.tensor([[0.0, 0.0]], dtype=torch.float64),
            torch.ones(1, dtype=torch.float64),
        )
    with pytest.raises(ValueError, match="candidate pairs"):
        bounded_power_pixel_operator(
            _rectangle(-1.0, -1.0, 1.0, 1.0).repeat(2, 1, 1),
            torch.zeros((2, 2), dtype=torch.float64),
            torch.ones(2, dtype=torch.float64),
            max_candidate_pairs=3,
        )


def test_batched_operator_matches_scalar_oracle() -> None:
    generator = torch.Generator().manual_seed(602)
    sites = 2.0 * torch.rand((24, 2), generator=generator, dtype=torch.float64)
    radii = 0.2 + 0.55 * torch.rand((24,), generator=generator, dtype=torch.float64)
    pixels = torch.stack(
        [
            _rectangle(float(column), float(row), column + 1.0, row + 1.0)
            for row in range(2)
            for column in range(2)
        ]
    )
    candidates = tile_disk_candidate_csr(sites, radii, tile_shape=(2, 2))
    graph = sparse_cech_graph(sites, radii)
    kwargs = {
        "candidate_indptr": candidates.indptr,
        "candidate_indices": candidates.indices,
        "neighbor_indptr": graph.indptr,
        "neighbor_indices": graph.indices,
    }

    scalar = bounded_power_pixel_operator(pixels, sites, radii, **kwargs)
    batched = batched_bounded_power_pixel_operator(pixels, sites, radii, **kwargs)
    chunked = batched_bounded_power_pixel_operator(
        pixels,
        sites,
        radii,
        candidate_chunk_size=3,
        **kwargs,
    )

    torch.testing.assert_close(batched.indptr, scalar.indptr)
    torch.testing.assert_close(batched.indices, scalar.indices)
    torch.testing.assert_close(batched.pixel_ids, scalar.pixel_ids)
    torch.testing.assert_close(batched.transpose_indptr, scalar.transpose_indptr)
    torch.testing.assert_close(batched.transpose_indices, scalar.transpose_indices)
    torch.testing.assert_close(batched.areas, scalar.areas, rtol=2e-12, atol=2e-12)
    torch.testing.assert_close(
        batched.transpose_areas,
        scalar.transpose_areas,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(
        batched.cell_areas,
        scalar.cell_areas,
        rtol=2e-12,
        atol=2e-12,
    )
    torch.testing.assert_close(chunked.indptr, batched.indptr)
    torch.testing.assert_close(chunked.indices, batched.indices)
    torch.testing.assert_close(chunked.areas, batched.areas)
    values = torch.randn(24, generator=generator, dtype=torch.float64)
    pixel_values = torch.randn(4, generator=generator, dtype=torch.float64)
    for semantics in ("surface_brightness", "integrated_flux"):
        torch.testing.assert_close(
            batched.matvec(values, semantics=semantics),
            scalar.matvec(values, semantics=semantics),
            rtol=2e-12,
            atol=2e-12,
        )
        torch.testing.assert_close(
            batched.rmatvec(pixel_values, semantics=semantics),
            scalar.rmatvec(pixel_values, semantics=semantics),
            rtol=2e-12,
            atol=2e-12,
        )


def test_batched_operator_preserves_coincident_tie_and_zero_radius() -> None:
    sites = torch.tensor(
        [[0.0, 0.0], [0.0, 0.0], [0.5, 0.0]],
        dtype=torch.float64,
    )
    radii = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64)
    pixel = _rectangle(-1.0, -1.0, 1.0, 1.0).unsqueeze(0)
    candidates = torch.tensor([0, 1, 2], dtype=torch.long)
    graph = sparse_cech_graph(sites, radii)

    batched = batched_bounded_power_pixel_operator(
        pixel,
        sites,
        radii,
        candidate_indptr=torch.tensor([0, 3], dtype=torch.long),
        candidate_indices=candidates,
        neighbor_indptr=graph.indptr,
        neighbor_indices=graph.indices,
    )

    assert batched.indices.tolist() == [0]
    torch.testing.assert_close(
        batched.areas,
        torch.tensor([math.pi], dtype=torch.float64),
    )
    assert batched.cell_areas[1] == batched.cell_areas[2] == 0


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_power_pixel_operator_cuda_matches_cpu() -> None:
    sites = torch.tensor([[-0.4, 0.0], [0.4, 0.0]], dtype=torch.float32)
    radii = torch.tensor([0.8, 0.8], dtype=torch.float32)
    pixels = torch.stack(
        (
            _rectangle(-1.0, -1.0, 0.0, 1.0, dtype=torch.float32),
            _rectangle(0.0, -1.0, 1.0, 1.0, dtype=torch.float32),
        )
    )
    cpu = bounded_power_pixel_operator(pixels, sites, radii)
    cuda = bounded_power_pixel_operator(pixels.cuda(), sites.cuda(), radii.cuda())

    torch.testing.assert_close(cuda.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(cuda.indices.cpu(), cpu.indices)
    torch.testing.assert_close(cuda.areas.cpu(), cpu.areas, rtol=2e-5, atol=2e-6)
    torch.testing.assert_close(
        cuda.cell_areas.cpu(),
        cpu.cell_areas,
        rtol=2e-5,
        atol=2e-6,
    )
    candidates = tile_disk_candidate_csr(sites, radii, tile_shape=(2, 1))
    graph = sparse_cech_graph(sites, radii)
    batched_cpu = batched_bounded_power_pixel_operator(
        pixels,
        sites,
        radii,
        candidate_indptr=candidates.indptr,
        candidate_indices=candidates.indices,
        neighbor_indptr=graph.indptr,
        neighbor_indices=graph.indices,
    )
    batched_cuda = batched_bounded_power_pixel_operator(
        pixels.cuda(),
        sites.cuda(),
        radii.cuda(),
        candidate_indptr=candidates.indptr.cuda(),
        candidate_indices=candidates.indices.cuda(),
        neighbor_indptr=graph.indptr.cuda(),
        neighbor_indices=graph.indices.cuda(),
    )
    torch.testing.assert_close(
        batched_cuda.areas.cpu(),
        batched_cpu.areas,
        rtol=2e-5,
        atol=2e-6,
    )


@pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS not available")
def test_power_pixel_operator_mps_matches_cpu() -> None:
    sites = torch.tensor([[-0.4, 0.0], [0.4, 0.0]], dtype=torch.float32)
    radii = torch.tensor([0.8, 0.8], dtype=torch.float32)
    pixels = torch.stack(
        (
            _rectangle(-1.0, -1.0, 0.0, 1.0, dtype=torch.float32),
            _rectangle(0.0, -1.0, 1.0, 1.0, dtype=torch.float32),
        )
    )
    cpu = bounded_power_pixel_operator(pixels, sites, radii)
    mps = bounded_power_pixel_operator(
        pixels.to("mps"), sites.to("mps"), radii.to("mps")
    )

    torch.testing.assert_close(mps.indptr.cpu(), cpu.indptr)
    torch.testing.assert_close(mps.indices.cpu(), cpu.indices)
    torch.testing.assert_close(mps.areas.cpu(), cpu.areas, rtol=2e-5, atol=2e-6)
