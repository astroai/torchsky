#!/usr/bin/env python3
"""Measure all-resident versus host-batched exact spectral Foam solves."""

from __future__ import annotations

import argparse
import json
import statistics
import time

import torch

from torchsky.map.field import (
    sparse_bounded_power_pixel_operator,
    solve_spectral_power_pixel_affine_least_squares,
    spectral_basis_whitening_transform,
)


def _case(side: int, observations: int, basis_count: int, whiten_basis: bool):
    edges = torch.linspace(-1.0, 1.0, side + 1, dtype=torch.float64)
    y0, x0 = torch.meshgrid(edges[:-1], edges[:-1], indexing="ij")
    y1, x1 = torch.meshgrid(edges[1:], edges[1:], indexing="ij")
    pixels = torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=-2,
    ).reshape(-1, 4, 2)
    cell_side = max(2, side // 2)
    centers = torch.linspace(
        -1.0 + 1.0 / cell_side,
        1.0 - 1.0 / cell_side,
        cell_side,
        dtype=torch.float64,
    )
    sy, sx = torch.meshgrid(centers, centers, indexing="ij")
    sites = torch.stack((sx.flatten(), sy.flatten()), dim=-1)
    radii = torch.full((sites.shape[0],), 4.0 / cell_side, dtype=torch.float64)
    operators = []
    for index in range(observations):
        phase = (index + 1) / (observations + 1)
        shifted = pixels + pixels.new_tensor(
            [0.35 * phase / side, -0.2 * phase / side]
        )
        operators.append(sparse_bounded_power_pixel_operator(shifted, sites, radii))
    spectral_coordinate = torch.linspace(
        -1.0, 1.0, observations, dtype=torch.float64
    )
    basis = torch.stack(
        [spectral_coordinate**order for order in range(basis_count)], dim=1
    )
    raw_condition = torch.linalg.cond(basis)
    if whiten_basis:
        basis = basis @ spectral_basis_whitening_transform(basis)
    generator = torch.Generator().manual_seed(101)
    coefficients = torch.randn(
        (sites.shape[0], basis_count), generator=generator, dtype=torch.float64
    )
    gradients = 0.1 * torch.randn(
        (sites.shape[0], basis_count, 2), generator=generator, dtype=torch.float64
    )
    values = [
        operator.matvec_affine(
            coefficients @ row,
            torch.einsum("ckd,k->cd", gradients, row),
        )
        for operator, row in zip(operators, basis, strict=True)
    ]
    weights = [
        torch.linspace(0.5 + 0.05 * index, 1.5, value.numel(), dtype=value.dtype)
        for index, value in enumerate(values)
    ]
    return operators, basis, values, weights, raw_condition


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--side", type=int, default=32)
    parser.add_argument("--observations", type=int, default=6)
    parser.add_argument("--basis-count", type=int, default=4)
    parser.add_argument("--whiten-basis", action="store_true")
    parser.add_argument("--iterations", type=int, default=10)
    parser.add_argument("--runs", type=int, default=5)
    args = parser.parse_args()
    operators, basis, values, weights, raw_condition = _case(
        args.side, args.observations, args.basis_count, args.whiten_basis
    )
    rows = []
    for label, batch_size, preconditioner in (
        ("resident_diagonal", None, "diagonal"),
        ("batch1_diagonal", 1, "diagonal"),
        ("batch1_cell_block", 1, "cell_block"),
    ):
        samples = []
        diagnostics: dict[str, float] = {}
        for run in range(args.runs + 1):
            started = time.perf_counter()
            solve_spectral_power_pixel_affine_least_squares(
                operators,
                basis,
                values,
                weights,
                max_iterations=args.iterations,
                tolerance=1.0e-8,
                observation_batch_size=batch_size,
                preconditioner=preconditioner,
                diagnostics=diagnostics,
            )
            elapsed = time.perf_counter() - started
            if run:
                samples.append(elapsed)
        rows.append(
            {
                "mode": label,
                "median_seconds": statistics.median(samples),
                "min_seconds": min(samples),
                "max_seconds": max(samples),
                **diagnostics,
            }
        )
    print(
        json.dumps(
            {
                "side": args.side,
                "observations": args.observations,
                "basis_count": args.basis_count,
                "whiten_basis": args.whiten_basis,
                "raw_basis_condition": float(raw_condition),
                "solve_basis_condition": float(torch.linalg.cond(basis)),
                "runs_after_warmup": args.runs,
                "rows": rows,
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
