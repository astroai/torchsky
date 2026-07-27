#!/usr/bin/env python3
"""Benchmark exact disk-to-tile CSR construction for Power Foam."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import resource
import statistics
import sys
import time

import torch

from torchsky.map.field import tile_disk_candidate_csr


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=1_000_000)
    parser.add_argument("--rows", type=int, default=1024)
    parser.add_argument("--columns", type=int, default=1024)
    parser.add_argument("--radius", type=float, default=0.35)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=321)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--oracle-sites", type=int, default=2_000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_foam_tiles.json"),
    )
    args = parser.parse_args()
    if min(args.n_sites, args.rows, args.columns, args.oracle_sites) < 0:
        raise ValueError("counts must be non-negative")
    if args.rows == 0 or args.columns == 0 or args.runs <= 0:
        raise ValueError("grid dimensions and runs must be positive")
    if not math.isfinite(args.radius) or args.radius < 0.0:
        raise ValueError("radius must be finite and non-negative")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    sites = torch.stack(
        (
            args.columns
            * torch.rand(
                args.n_sites,
                generator=generator,
                device=device,
                dtype=dtype,
            ),
            args.rows
            * torch.rand(
                args.n_sites,
                generator=generator,
                device=device,
                dtype=dtype,
            ),
        ),
        dim=1,
    )
    radii = float(args.radius) * (
        0.75
        + 0.5
        * torch.rand(
            args.n_sites,
            generator=generator,
            device=device,
            dtype=dtype,
        )
    )

    samples: list[float] = []
    result = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.runs):
        _synchronize(device)
        started = time.perf_counter()
        result = tile_disk_candidate_csr(
            sites,
            radii,
            tile_shape=(args.rows, args.columns),
        )
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    assert result is not None

    oracle_count = min(args.n_sites, args.oracle_sites)
    oracle_rows = min(args.rows, 16)
    oracle_columns = min(args.columns, 16)
    oracle = tile_disk_candidate_csr(
        sites[:oracle_count],
        radii[:oracle_count],
        tile_shape=(oracle_rows, oracle_columns),
    )
    tile_y, tile_x = torch.meshgrid(
        torch.arange(oracle_rows, device=device, dtype=dtype),
        torch.arange(oracle_columns, device=device, dtype=dtype),
        indexing="ij",
    )
    lower = torch.stack((tile_x.flatten(), tile_y.flatten()), dim=1)
    upper = lower + 1.0
    closest = torch.minimum(
        torch.maximum(sites[:oracle_count].unsqueeze(0), lower.unsqueeze(1)),
        upper.unsqueeze(1),
    )
    dense = torch.sum(
        (sites[:oracle_count].unsqueeze(0) - closest) ** 2,
        dim=2,
    ) <= radii[:oracle_count].unsqueeze(0) ** 2
    dense_rows, dense_sites = torch.nonzero(dense, as_tuple=True)
    oracle_passed = bool(
        torch.equal(oracle.tile_ids, dense_rows)
        and torch.equal(oracle.indices, dense_sites)
    )

    occupancy = result.indptr[1:] - result.indptr[:-1]
    occupancy_float = occupancy.to(torch.float64)
    quantiles = torch.quantile(
        occupancy_float,
        torch.tensor(
            [0.5, 0.9, 0.99],
            dtype=torch.float64,
            device=device,
        ),
    )
    exact_candidates = int(result.indices.numel())
    total_pairs = result.total_pairs
    payload = {
        "n_sites": args.n_sites,
        "tile_shape": [args.rows, args.columns],
        "tile_count": args.rows * args.columns,
        "radius": args.radius,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "runs": args.runs,
        "wall_seconds_median": statistics.median(samples),
        "wall_seconds_min": min(samples),
        "total_pairs": total_pairs,
        "aabb_memberships": result.aabb_memberships,
        "exact_candidates": exact_candidates,
        "candidate_fraction": exact_candidates / total_pairs if total_pairs else 0.0,
        "candidate_reduction_ratio": (
            1.0 - exact_candidates / total_pairs if total_pairs else 1.0
        ),
        "empty_tiles": int(torch.count_nonzero(occupancy == 0).item()),
        "mean_candidates_per_tile": float(occupancy_float.mean().item()),
        "max_candidates_per_tile": int(occupancy.max().item()),
        "candidate_quantiles": {
            "p50": float(quantiles[0].item()),
            "p90": float(quantiles[1].item()),
            "p99": float(quantiles[2].item()),
        },
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "oracle_sites": oracle_count,
        "oracle_tile_shape": [oracle_rows, oracle_columns],
        "oracle_passed": oracle_passed,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if oracle_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
