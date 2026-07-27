#!/usr/bin/env python3
"""Benchmark the exact bounded-power-cell/pixel reference operator."""

from __future__ import annotations

import argparse
from functools import partial
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch

from torchsky.map.field import (
    batched_bounded_power_pixel_operator,
    bounded_power_ownership,
    bounded_power_pixel_operator,
    sparse_cech_graph,
    tile_disk_candidate_csr,
)


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=256)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--columns", type=int, default=16)
    parser.add_argument("--radius", type=float, default=1.25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=947)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--implementation",
        choices=("batched", "reference"),
        default="batched",
    )
    parser.add_argument(
        "--execution",
        choices=("eager", "compile", "compile-kernels"),
        default="eager",
        help="Compile the whole builder or only its batched numeric kernels",
    )
    parser.add_argument("--oracle-resolution", type=int, default=256)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_foam_operator.json"),
    )
    args = parser.parse_args()
    if min(args.n_sites, args.rows, args.columns, args.runs) <= 0:
        raise ValueError("site, grid, and run counts must be positive")
    if args.radius < 0.0 or args.oracle_resolution <= 0:
        raise ValueError("radius must be non-negative and oracle resolution positive")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    sites = torch.stack(
        (
            args.columns
            * torch.rand(
                args.n_sites,
                generator=generator,
                dtype=dtype,
            ),
            args.rows
            * torch.rand(
                args.n_sites,
                generator=generator,
                dtype=dtype,
            ),
        ),
        dim=1,
    ).to(device)
    radii = args.radius * (
        0.75
        + 0.5
        * torch.rand(
            args.n_sites,
            generator=generator,
            dtype=dtype,
        )
    ).to(device)
    tile_candidates = tile_disk_candidate_csr(
        sites,
        radii,
        tile_shape=(args.rows, args.columns),
    )
    graph = sparse_cech_graph(sites, radii)
    grid_y, grid_x = torch.meshgrid(
        torch.arange(args.rows, dtype=dtype, device=device),
        torch.arange(args.columns, dtype=dtype, device=device),
        indexing="ij",
    )
    lower = torch.stack((grid_x.flatten(), grid_y.flatten()), dim=1)
    pixels = torch.stack(
        (
            lower,
            lower + lower.new_tensor([1.0, 0.0]),
            lower + 1.0,
            lower + lower.new_tensor([0.0, 1.0]),
        ),
        dim=1,
    )

    builder = (
        batched_bounded_power_pixel_operator
        if args.implementation == "batched"
        else bounded_power_pixel_operator
    )
    if args.execution == "compile-kernels":
        if args.implementation != "batched":
            raise ValueError("compile-kernels requires --implementation batched")
        builder = partial(
            batched_bounded_power_pixel_operator,
            kernel_execution="compile",
        )

    compile_warmup_seconds = None
    operator = None
    if args.execution in ("compile", "compile-kernels"):
        if args.execution == "compile":
            builder = torch.compile(builder)
        _synchronize(device)
        started = time.perf_counter()
        operator = builder(
            pixels,
            sites,
            radii,
            candidate_indptr=tile_candidates.indptr,
            candidate_indices=tile_candidates.indices,
            neighbor_indptr=graph.indptr,
            neighbor_indices=graph.indices,
            max_candidate_pairs=max(tile_candidates.indices.numel(), 1),
        )
        _synchronize(device)
        compile_warmup_seconds = time.perf_counter() - started

    samples: list[float] = []
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.runs):
        _synchronize(device)
        started = time.perf_counter()
        operator = builder(
            pixels,
            sites,
            radii,
            candidate_indptr=tile_candidates.indptr,
            candidate_indices=tile_candidates.indices,
            neighbor_indptr=graph.indptr,
            neighbor_indices=graph.indices,
            max_candidate_pairs=max(tile_candidates.indices.numel(), 1),
        )
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    assert operator is not None
    operator_peak_rss_bytes = _peak_rss_bytes()
    operator_peak_cuda_allocated_bytes = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    resolution = args.oracle_resolution
    oracle_y, oracle_x = torch.meshgrid(
        (torch.arange(resolution, dtype=dtype, device=device) + 0.5)
        * (args.rows / resolution),
        (torch.arange(resolution, dtype=dtype, device=device) + 0.5)
        * (args.columns / resolution),
        indexing="ij",
    )
    owners, _ = bounded_power_ownership(
        torch.stack((oracle_x.flatten(), oracle_y.flatten()), dim=1),
        sites,
        radii,
        point_chunk_size=65_536,
    )
    numerical_union_area = torch.count_nonzero(owners >= 0).to(dtype) * (
        args.rows * args.columns / resolution**2
    )
    exact_union_area = operator.areas.sum()
    numerical_error = float(torch.abs(exact_union_area - numerical_union_area).item())
    median = statistics.median(samples)
    candidates = int(tile_candidates.indices.numel())
    intersections = int(operator.areas.numel())
    payload = {
        "n_sites": args.n_sites,
        "tile_shape": [args.rows, args.columns],
        "seed": args.seed,
        "radius": args.radius,
        "device": str(device),
        "dtype": args.dtype,
        "implementation": args.implementation,
        "execution": args.execution,
        "compile_warmup_seconds": compile_warmup_seconds,
        "runs": args.runs,
        "candidate_pairs": candidates,
        "positive_intersections": intersections,
        "wall_seconds_median": median,
        "wall_seconds_min": min(samples),
        "candidate_pairs_per_second": candidates / median,
        "positive_intersections_per_second": intersections / median,
        "exact_union_area": float(exact_union_area.item()),
        "numerical_oracle_union_area": float(numerical_union_area.item()),
        "numerical_oracle_absolute_error": numerical_error,
        "oracle_resolution": resolution,
        "operator_peak_rss_bytes": operator_peak_rss_bytes,
        "operator_peak_cuda_allocated_bytes": operator_peak_cuda_allocated_bytes,
        "process_peak_rss_bytes_including_oracle": _peak_rss_bytes(),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
