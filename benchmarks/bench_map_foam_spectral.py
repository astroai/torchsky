#!/usr/bin/env python3
"""Benchmark shared-geometry multi-channel Power Foam rendering."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import statistics
import sys
import time

import torch

from torchsky.map.field import (
    batched_bounded_power_pixel_operator,
    render_power_spectral_samples,
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
    parser.add_argument("--n-sites", type=int, default=1000)
    parser.add_argument("--rows", type=int, default=32)
    parser.add_argument("--columns", type=int, default=32)
    parser.add_argument("--radius", type=float, default=1.25)
    parser.add_argument("--channels", type=int, default=6)
    parser.add_argument("--spectral-samples", type=int, default=9)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=751)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_foam_spectral.json"),
    )
    args = parser.parse_args()
    if min(
        args.n_sites,
        args.rows,
        args.columns,
        args.channels,
        args.spectral_samples,
        args.runs,
    ) <= 0:
        raise ValueError("all size and run counts must be positive")
    if args.radius < 0.0:
        raise ValueError("radius must be non-negative")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    sites = torch.stack(
        (
            args.columns * torch.rand(args.n_sites, generator=generator, dtype=dtype),
            args.rows * torch.rand(args.n_sites, generator=generator, dtype=dtype),
        ),
        dim=1,
    ).to(device)
    radii = args.radius * (
        0.75 + 0.5 * torch.rand(args.n_sites, generator=generator, dtype=dtype)
    ).to(device)
    cell_samples = torch.randn(
        (args.spectral_samples, args.n_sites),
        generator=generator,
        dtype=dtype,
    ).to(device)
    responses = torch.rand(
        (args.channels, args.spectral_samples),
        generator=generator,
        dtype=dtype,
    ).to(device)
    responses = responses / responses.sum(dim=1, keepdim=True)

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
    builder_kwargs = {
        "candidate_indptr": tile_candidates.indptr,
        "candidate_indices": tile_candidates.indices,
        "neighbor_indptr": graph.indptr,
        "neighbor_indices": graph.indices,
        "max_candidate_pairs": max(tile_candidates.indices.numel(), 1),
    }

    shared_samples: list[float] = []
    shared_output = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.runs):
        _synchronize(device)
        started = time.perf_counter()
        operator = batched_bounded_power_pixel_operator(
            pixels,
            sites,
            radii,
            **builder_kwargs,
        )
        shared_output = render_power_spectral_samples(
            operator,
            cell_samples,
            responses,
        )
        _synchronize(device)
        shared_samples.append(time.perf_counter() - started)
    shared_peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )

    independent_samples: list[float] = []
    independent_output = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.runs):
        _synchronize(device)
        started = time.perf_counter()
        channel_outputs = []
        for channel in range(args.channels):
            operator = batched_bounded_power_pixel_operator(
                pixels,
                sites,
                radii,
                **builder_kwargs,
            )
            channel_cells = torch.einsum(
                "s,sc->c",
                responses[channel],
                cell_samples,
            )
            channel_outputs.append(operator.matvec(channel_cells))
        independent_output = torch.stack(channel_outputs)
        _synchronize(device)
        independent_samples.append(time.perf_counter() - started)
    independent_peak_cuda = (
        int(torch.cuda.max_memory_allocated(device))
        if device.type == "cuda"
        else None
    )
    assert shared_output is not None and independent_output is not None
    torch.testing.assert_close(shared_output, independent_output)

    shared_median = statistics.median(shared_samples)
    independent_median = statistics.median(independent_samples)
    payload = {
        "n_sites": args.n_sites,
        "tile_shape": [args.rows, args.columns],
        "channels": args.channels,
        "spectral_samples": args.spectral_samples,
        "seed": args.seed,
        "radius": args.radius,
        "device": str(device),
        "dtype": args.dtype,
        "runs": args.runs,
        "candidate_pairs": int(tile_candidates.indices.numel()),
        "positive_intersections": int(operator.areas.numel()),
        "shared_geometry_builds_per_run": 1,
        "independent_geometry_builds_per_run": args.channels,
        "shared_wall_seconds_median": shared_median,
        "independent_wall_seconds_median": independent_median,
        "shared_speedup": independent_median / shared_median,
        "shared_peak_cuda_allocated_bytes": shared_peak_cuda,
        "independent_peak_cuda_allocated_bytes": independent_peak_cuda,
        "process_peak_rss_bytes": _peak_rss_bytes(),
        "outputs_equal": True,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
