#!/usr/bin/env python3
"""Measure bounded exact-median and sigma-clipped map reducers."""

from __future__ import annotations

import argparse
import gc
import json
import platform
import resource
import statistics
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map.operators.execution import (  # noqa: E402
    exact_median_scratch_bytes,
    sigma_clipped_scratch_bytes,
    spilled_exact_median_stack,
    spilled_sigma_clipped_stack,
)


_PROFILES = {
    "small": {"size": 256, "n_observations": 9},
    "medium": {"size": 1024, "n_observations": 13},
    "large": {"size": 2048, "n_observations": 17},
}


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _worker(
    *,
    method: str,
    size: int,
    n_observations: int,
    working_memory_mib: int,
) -> dict[str, Any]:
    if method not in {"median", "sigma_clipped"}:
        raise ValueError("method must be median or sigma_clipped")
    if size <= 0 or n_observations < 3 or working_memory_mib <= 0:
        raise ValueError("size/memory must be positive and observations at least 3")
    torch.set_num_threads(1)
    shape = (int(size), int(size))
    row = torch.arange(size, dtype=torch.float32).unsqueeze(1)
    column = torch.arange(size, dtype=torch.float32).unsqueeze(0)
    truth = row * 1.0e-4 + column * 2.0e-4
    del row, column
    support = torch.ones(shape, dtype=torch.float32)
    work_bytes = int(working_memory_mib) * 1024 * 1024

    if method == "median":
        offsets = torch.linspace(-0.04, 0.04, n_observations).tolist()

        def contributions():
            for offset in offsets:
                yield truth + float(offset), support

        required = exact_median_scratch_bytes(n_observations, shape)
    else:
        offsets = torch.linspace(-0.04, 0.04, n_observations - 1).tolist()
        offsets.append(20.0)

        def contributions():
            for offset in offsets:
                yield truth + float(offset), support, support, None

        required = sigma_clipped_scratch_bytes(n_observations, shape)

    with tempfile.TemporaryDirectory(prefix="torchsky-reducer-bench-") as work:
        scratch = Path(work) / "scratch"
        scratch.mkdir()
        gc.collect()
        rss_before = _peak_rss_bytes()
        start = time.perf_counter()
        if method == "median":
            data, weight = spilled_exact_median_stack(
                contributions(),
                n_observations=n_observations,
                shape=shape,
                scratch_dir=scratch,
                output_device="cpu",
                working_memory_bytes=work_bytes,
            )
            accepted_min = float(weight.min().item())
            accepted_max = float(weight.max().item())
        else:
            reduced = spilled_sigma_clipped_stack(
                contributions(),
                n_observations=n_observations,
                shape=shape,
                scratch_dir=scratch,
                output_device="cpu",
                working_memory_bytes=work_bytes,
            )
            data = reduced.data
            accepted_min = float(reduced.weight.min().item())
            accepted_max = float(reduced.weight.max().item())
        elapsed = time.perf_counter() - start
        peak_rss = _peak_rss_bytes()
        scratch_clean = not any(scratch.iterdir())
    maximum_error = float(torch.max(torch.abs(data - truth)).item())
    return {
        "method": method,
        "size": size,
        "n_observations": n_observations,
        "output_pixels": size * size,
        "input_samples": n_observations * size * size,
        "elapsed_seconds": elapsed,
        "input_megapixels_per_second": (
            n_observations * size * size / 1.0e6 / elapsed
        ),
        "rss_before_bytes": rss_before,
        "peak_rss_bytes": peak_rss,
        "peak_rss_delta_bytes": max(peak_rss - rss_before, 0),
        "scratch_required_bytes": required,
        "working_memory_limit_bytes": work_bytes,
        "scratch_clean": scratch_clean,
        "accepted_weight_min": accepted_min,
        "accepted_weight_max": accepted_max,
        "maximum_absolute_error": maximum_error,
        "checksum": float(data.sum().item()),
    }


def _aggregate(samples: list[dict[str, Any]]) -> dict[str, Any]:
    row = dict(samples[0])
    for key in (
        "elapsed_seconds",
        "input_megapixels_per_second",
        "rss_before_bytes",
        "peak_rss_bytes",
        "peak_rss_delta_bytes",
    ):
        row[key] = statistics.median(float(sample[key]) for sample in samples)
    row["scratch_clean"] = all(bool(sample["scratch_clean"]) for sample in samples)
    row["run_samples"] = samples
    return row


def _metadata(*, profile: str, rows: list[dict[str, Any]], runs: int) -> dict[str, Any]:
    commit = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=ROOT,
        check=False,
        capture_output=True,
        text=True,
    ).stdout.strip()
    return {
        "schema": "torchsky.map_reducer_streaming_scorecard/v1",
        "profile": profile,
        "runs": runs,
        "aggregation": "per-metric median of fresh subprocesses",
        "platform": platform.platform(),
        "python": platform.python_version(),
        "torch": torch.__version__,
        "torchsky": torchsky.__version__,
        "git_commit": commit or None,
        "results": rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=tuple(_PROFILES), default="small")
    parser.add_argument("--methods", default="median,sigma_clipped")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--working-memory-mib", type=int, default=64)
    parser.add_argument("--json-out", type=Path)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--method", choices=("median", "sigma_clipped"))
    parser.add_argument("--size", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--n-observations", type=int, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.runs <= 0 or args.working_memory_mib <= 0:
        parser.error("runs and working-memory-mib must be positive")
    if args.worker:
        if args.method is None or args.size is None or args.n_observations is None:
            parser.error("worker requires method, size, and n-observations")
        print(
            json.dumps(
                _worker(
                    method=args.method,
                    size=args.size,
                    n_observations=args.n_observations,
                    working_memory_mib=args.working_memory_mib,
                )
            )
        )
        return 0

    methods = tuple(part.strip() for part in args.methods.split(",") if part.strip())
    if not methods or any(
        method not in {"median", "sigma_clipped"} for method in methods
    ):
        parser.error("methods must contain median and/or sigma_clipped")
    profile = _PROFILES[args.profile]
    rows = []
    for method in methods:
        samples = []
        for _ in range(args.runs):
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--worker",
                "--method",
                method,
                "--size",
                str(profile["size"]),
                "--n-observations",
                str(profile["n_observations"]),
                "--working-memory-mib",
                str(args.working_memory_mib),
            ]
            completed = subprocess.run(
                command, check=True, capture_output=True, text=True
            )
            samples.append(json.loads(completed.stdout))
        rows.append(_aggregate(samples))
    scorecard = _metadata(profile=args.profile, rows=rows, runs=args.runs)
    rendered = json.dumps(scorecard, indent=2, sort_keys=True)
    print(rendered)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
