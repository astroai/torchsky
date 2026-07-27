#!/usr/bin/env python3
"""Benchmark exact sparse Čech candidate generation for Power Foam disks."""

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

from torchsky.map.field import cech_adjacency, sparse_cech_graph


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-sites", type=int, default=100_000)
    parser.add_argument("--radius", type=float, default=0.25)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=123)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float32")
    parser.add_argument("--oracle-sites", type=int, default=2_000)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_foam_sparse.json"),
    )
    args = parser.parse_args()
    if args.n_sites < 0 or args.runs <= 0 or args.oracle_sites < 0:
        raise ValueError("site and run counts must be non-negative, with runs positive")
    if not math.isfinite(args.radius) or args.radius < 0.0:
        raise ValueError("radius must be finite and non-negative")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator(device=device).manual_seed(args.seed)
    extent = max(
        math.sqrt(max(args.n_sites, 1)) * max(args.radius, 1.0e-3) * 5.0,
        1.0,
    )
    sites = extent * torch.rand(
        (args.n_sites, 2),
        generator=generator,
        device=device,
        dtype=dtype,
    )
    radii = float(args.radius) * (
        0.75
        + 0.5
        * torch.rand(
            (args.n_sites,),
            generator=generator,
            device=device,
            dtype=dtype,
        )
    )

    samples: list[float] = []
    graph = None
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for _ in range(args.runs):
        _synchronize(device)
        started = time.perf_counter()
        graph = sparse_cech_graph(sites, radii)
        _synchronize(device)
        samples.append(time.perf_counter() - started)
    assert graph is not None

    oracle_count = min(args.n_sites, args.oracle_sites)
    oracle_passed = True
    if oracle_count:
        sparse_oracle = sparse_cech_graph(
            sites[:oracle_count],
            radii[:oracle_count],
        ).edges
        dense_oracle = cech_adjacency(
            sites[:oracle_count],
            radii[:oracle_count],
            site_chunk_size=256,
        )
        oracle_passed = bool(torch.equal(sparse_oracle, dense_oracle))
    total_pairs = graph.total_pairs
    candidate_pairs = int(graph.candidate_pairs.shape[1])
    result = {
        "n_sites": args.n_sites,
        "radius": args.radius,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "runs": args.runs,
        "wall_seconds_median": statistics.median(samples),
        "wall_seconds_min": min(samples),
        "total_pairs": total_pairs,
        "candidate_pairs": candidate_pairs,
        "exact_edges": int(graph.edges.shape[1]),
        "candidate_fraction": (
            candidate_pairs / total_pairs if total_pairs else 0.0
        ),
        "candidate_reduction_ratio": (
            1.0 - candidate_pairs / total_pairs if total_pairs else 1.0
        ),
        "site_bin_memberships": graph.site_bin_memberships,
        "occupied_bins": graph.occupied_bins,
        "bin_size": graph.bin_size,
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "oracle_sites": oracle_count,
        "oracle_passed": oracle_passed,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))
    return 0 if oracle_passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
