#!/usr/bin/env python3
"""Lightweight RSS/throughput evidence for multi-exposure operator chunking."""

from __future__ import annotations

import argparse
import json
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.map.operators import ObservationOperatorBundle  # noqa: E402
from torchsky.map.solvers import (  # noqa: E402
    BoundObservationOperator,
    BoundObservationOperatorSet,
)
from torchsky.map.target import TargetRenderSpec  # noqa: E402


def _max_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024 * 1024) if sys.platform == "darwin" else value / 1024


def _operator_set(
    state: torch.Tensor,
    *,
    n_obs: int,
    exposure_chunk_size: int,
) -> BoundObservationOperatorSet:
    target = TargetRenderSpec(
        header={"NAXIS": 2, "NAXIS1": state.shape[1], "NAXIS2": state.shape[0]},
        shape=tuple(state.shape),
    )

    def make_operator(scale: float) -> BoundObservationOperator:
        def apply(scene, target_spec):
            tensor = scene.render_to_target(target_spec)
            return tensor * scale

        def adjoint(tensor):
            return torch.as_tensor(tensor) * scale

        return BoundObservationOperator(
            operators=ObservationOperatorBundle(),
            target=target,
            apply_operator=apply,
            apply_adjoint=adjoint,
        )

    return BoundObservationOperatorSet(
        tuple(make_operator(float(index + 1) / n_obs) for index in range(n_obs)),
        exposure_chunk_size=exposure_chunk_size,
    )


def _worker(
    *,
    mode: str,
    size: int,
    n_obs: int,
    chunk_size: int,
    runs: int,
) -> dict[str, Any]:
    torch.set_num_threads(1)
    state = torch.linspace(-1.0, 1.0, size * size, dtype=torch.float32).reshape(
        size, size
    )
    operator_set = _operator_set(
        state,
        n_obs=n_obs,
        exposure_chunk_size=chunk_size,
    )
    rss_before = _max_rss_mb()
    reference: torch.Tensor | None = None
    deterministic = True
    start = time.perf_counter()
    with torch.inference_mode():
        for _ in range(runs):
            if mode == "eager":
                predictions = operator_set.apply(state)
                result = operator_set.adjoint(
                    operator_set.apply_noise_model(predictions)
                )
                del predictions
            else:
                result = operator_set.normal_operator_action(state)
            if reference is None:
                reference = result
            else:
                deterministic = deterministic and torch.equal(result, reference)
    elapsed = time.perf_counter() - start
    assert reference is not None
    peak_rss_mb = _max_rss_mb()
    live_predictions = n_obs if mode == "eager" else min(chunk_size, n_obs)
    return {
        "mode": mode,
        "size": size,
        "n_obs": n_obs,
        "exposure_chunk_size": chunk_size if mode == "chunked" else n_obs,
        "runs": runs,
        "time_s": elapsed,
        "exposures_per_second": runs * n_obs / elapsed,
        "megapixels_per_second": runs * n_obs * size * size / 1.0e6 / elapsed,
        "rss_before_mb": rss_before,
        "peak_rss_mb": peak_rss_mb,
        "peak_rss_delta_mb": max(peak_rss_mb - rss_before, 0.0),
        "prediction_working_set_bound_mb": (
            live_predictions * state.numel() * state.element_size() / 1024**2
        ),
        "checksum": float(reference.sum().item()),
        "deterministic": deterministic,
    }


def _run_subprocess(
    args: argparse.Namespace,
    mode: str,
    chunk_size: int,
) -> dict[str, Any]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--mode",
        mode,
        "--size",
        str(args.size),
        "--n-obs",
        str(args.n_obs),
        "--chunk-size",
        str(chunk_size),
        "--runs",
        str(args.runs),
    ]
    completed = subprocess.run(command, check=True, capture_output=True, text=True)
    return json.loads(completed.stdout)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=512)
    parser.add_argument("--n-obs", type=int, default=24)
    parser.add_argument("--chunk-sizes", type=int, nargs="+", default=[1, 4])
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--mode", choices=("eager", "chunked"), default="chunked")
    parser.add_argument("--chunk-size", type=int, default=1)
    args = parser.parse_args()
    if args.size < 1 or args.n_obs < 1 or args.runs < 1:
        parser.error("size, n-obs, and runs must be positive")
    if args.chunk_size < 1 or any(size < 1 for size in args.chunk_sizes):
        parser.error("chunk sizes must be positive")

    if args.worker:
        print(
            json.dumps(
                _worker(
                    mode=args.mode,
                    size=args.size,
                    n_obs=args.n_obs,
                    chunk_size=args.chunk_size,
                    runs=args.runs,
                )
            )
        )
        return 0

    rows = [_run_subprocess(args, "eager", args.n_obs)]
    rows.extend(
        _run_subprocess(args, "chunked", chunk_size)
        for chunk_size in args.chunk_sizes
    )
    checksums = {row["checksum"] for row in rows}
    if len(checksums) != 1 or not all(row["deterministic"] for row in rows):
        raise RuntimeError("chunked and eager operator execution diverged")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
