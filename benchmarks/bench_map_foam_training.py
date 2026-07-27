#!/usr/bin/env python3
"""Fit a seeded two-cell spectral Foam and verify exact-pixel handoff."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import resource
import sys
import time

import torch

from torchsky.map.field import SpectralPowerFoam, bounded_power_pixel_operator


def _peak_rss_bytes() -> int:
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == "darwin" else value * 1024


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _pixel_grid(
    rows: int,
    columns: int,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    x_edges = torch.linspace(-1.2, 1.2, columns + 1, device=device, dtype=dtype)
    y_edges = torch.linspace(-0.8, 0.8, rows + 1, device=device, dtype=dtype)
    y0, x0 = torch.meshgrid(y_edges[:-1], x_edges[:-1], indexing="ij")
    y1, x1 = torch.meshgrid(y_edges[1:], x_edges[1:], indexing="ij")
    return torch.stack(
        (
            torch.stack((x0, y0), dim=-1),
            torch.stack((x1, y0), dim=-1),
            torch.stack((x1, y1), dim=-1),
            torch.stack((x0, y1), dim=-1),
        ),
        dim=2,
    ).reshape(-1, 4, 2)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--samples", type=int, default=1500)
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--learning-rate", type=float, default=0.02)
    parser.add_argument("--noise-sigma", type=float, default=0.002)
    parser.add_argument("--seed", type=int, default=3801)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    parser.add_argument("--dtype", choices=("float32", "float64"), default="float64")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_foam_training.json"),
    )
    args = parser.parse_args()
    if args.samples < 10 or args.steps <= 0:
        raise ValueError("samples must be at least 10 and steps must be positive")
    if args.learning_rate <= 0.0 or args.noise_sigma < 0.0:
        raise ValueError("learning rate must be positive and noise non-negative")

    device = torch.device(
        "cuda"
        if args.device == "auto" and torch.cuda.is_available()
        else ("cpu" if args.device == "auto" else args.device)
    )
    dtype = getattr(torch, args.dtype)
    generator = torch.Generator().manual_seed(args.seed)
    points = torch.empty((args.samples, 2), dtype=dtype)
    points[:, 0] = 2.6 * torch.rand(args.samples, generator=generator, dtype=dtype) - 1.3
    points[:, 1] = 1.8 * torch.rand(args.samples, generator=generator, dtype=dtype) - 0.9
    spectral_coordinate = (
        2.0 * torch.rand(args.samples, generator=generator, dtype=dtype) - 1.0
    )
    basis = torch.stack(
        (torch.ones_like(spectral_coordinate), spectral_coordinate),
        dim=1,
    )
    truth = SpectralPowerFoam(
        torch.tensor([[-0.55, -0.05], [0.5, 0.08]], dtype=dtype),
        torch.tensor([0.78, 0.72], dtype=dtype),
        torch.tensor([[1.0, 0.55], [1.7, -0.45]], dtype=dtype),
        minimum_radius=0.05,
    )
    with torch.no_grad():
        target = truth(
            points,
            basis,
            power_temperature=24.0,
            support_temperature=24.0,
        )
        target += args.noise_sigma * torch.randn(
            target.shape,
            generator=generator,
            dtype=dtype,
        )

    points = points.to(device)
    basis = basis.to(device)
    target = target.to(device)
    truth = truth.to(device)
    validation = torch.arange(args.samples, device=device) % 5 == 0
    train = ~validation
    model = SpectralPowerFoam(
        torch.tensor([[-0.4, 0.08], [0.38, -0.04]], dtype=dtype, device=device),
        torch.tensor([0.62, 0.6], dtype=dtype, device=device),
        torch.tensor([[0.8, 0.0], [1.4, 0.0]], dtype=dtype, device=device),
        minimum_radius=0.05,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)

    _synchronize(device)
    started = time.perf_counter()
    initial_validation_mse = None
    for step in range(args.steps):
        progress = step / max(args.steps - 1, 1)
        temperature = 4.0 * (24.0 / 4.0) ** progress
        optimizer.zero_grad()
        prediction = model(
            points[train],
            basis[train],
            power_temperature=temperature,
            support_temperature=temperature,
        )
        loss = torch.mean(torch.square(prediction - target[train]))
        loss.backward()
        optimizer.step()
        if step == 0:
            with torch.no_grad():
                initial_validation_mse = torch.mean(
                    torch.square(
                        model(
                            points[validation],
                            basis[validation],
                            power_temperature=24.0,
                            support_temperature=24.0,
                        )
                        - target[validation]
                    )
                )
    _synchronize(device)
    training_seconds = time.perf_counter() - started
    assert initial_validation_mse is not None

    with torch.no_grad():
        validation_prediction = model(
            points[validation],
            basis[validation],
            power_temperature=24.0,
            support_temperature=24.0,
        )
        validation_mse = torch.mean(
            torch.square(validation_prediction - target[validation])
        )
        pixels = _pixel_grid(8, 12, device=device, dtype=dtype)
        truth_operator = bounded_power_pixel_operator(
            pixels,
            truth.sites.detach(),
            truth.radii.detach(),
        )
        learned_operator = bounded_power_pixel_operator(
            pixels,
            model.sites.detach(),
            model.radii.detach(),
        )
        basis_samples = torch.tensor(
            [[1.0, -0.75], [1.0, 0.0], [1.0, 0.75]],
            dtype=dtype,
            device=device,
        )
        responses = torch.tensor(
            [[0.5, 0.5, 0.0], [0.0, 0.5, 0.5]],
            dtype=dtype,
            device=device,
        )
        truth_pixels = truth.render_exact(
            truth_operator,
            basis_samples,
            responses,
        )
        learned_pixels = model.render_exact(
            learned_operator,
            basis_samples,
            responses,
        )
        exact_pixel_rmse = torch.sqrt(
            torch.mean(torch.square(learned_pixels - truth_pixels))
        )
        exact_pixel_peak = torch.max(torch.abs(truth_pixels))
        validation_threshold = max(25.0 * args.noise_sigma**2, 1.0e-4)
        passed = (
            float(validation_mse.item()) <= validation_threshold
            and float((initial_validation_mse / validation_mse).item()) >= 100.0
            and float(
                (exact_pixel_rmse / torch.clamp(exact_pixel_peak, min=1e-12)).item()
            )
            <= 0.02
        )

    payload = {
        "samples": args.samples,
        "training_samples": int(train.sum().item()),
        "validation_samples": int(validation.sum().item()),
        "steps": args.steps,
        "learning_rate": args.learning_rate,
        "noise_sigma": args.noise_sigma,
        "seed": args.seed,
        "device": str(device),
        "dtype": args.dtype,
        "training_seconds": training_seconds,
        "initial_validation_mse": float(initial_validation_mse.item()),
        "final_validation_mse": float(validation_mse.item()),
        "validation_improvement": float(
            (initial_validation_mse / validation_mse).item()
        ),
        "exact_pixel_rmse": float(exact_pixel_rmse.item()),
        "exact_pixel_relative_rmse": float(
            (exact_pixel_rmse / torch.clamp(exact_pixel_peak, min=1e-12)).item()
        ),
        "validation_mse_threshold": validation_threshold,
        "truth_sites": truth.sites.detach().cpu().tolist(),
        "learned_sites": model.sites.detach().cpu().tolist(),
        "truth_radii": truth.radii.detach().cpu().tolist(),
        "learned_radii": model.radii.detach().cpu().tolist(),
        "truth_coefficients": truth.cell_coefficients.detach().cpu().tolist(),
        "learned_coefficients": model.cell_coefficients.detach().cpu().tolist(),
        "peak_rss_bytes": _peak_rss_bytes(),
        "peak_cuda_allocated_bytes": (
            int(torch.cuda.max_memory_allocated(device))
            if device.type == "cuda"
            else None
        ),
        "passed": passed,
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
