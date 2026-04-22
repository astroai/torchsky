#!/usr/bin/env python3
"""Focused hyperparameter sweeps for exact-band field-fit diagnostics."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from torchsky.map import (  # noqa: E402
    SkyEncoderMLPField,
    SpectralBasisField,
    estimate_field_domain,
    evaluate_map_product,
    fit_field_model,
    plan_map_product,
    render_field_product,
    stack_map_product,
)

import bench_map_field_fit as field_bench  # noqa: E402


@dataclass(frozen=True)
class SweepConfig:
    steps: int
    hidden_dim: int
    hidden_layers: int
    num_spatial_frequencies: int
    num_spectral_frequencies: int


def _parse_int_list(text: str) -> list[int]:
    values = [value.strip() for value in text.split(",") if value.strip()]
    if not values:
        raise argparse.ArgumentTypeError("expected at least one integer")
    return [int(value) for value in values]


def _markdown_table(rows: list[dict[str, Any]]) -> str:
    header = (
        "| case | model | steps | hidden | layers | spatial_freq | spectral_freq | "
        "rmse | support_rmse | snr_db | flux_err | fit_time_ms |\n"
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |\n"
    )
    lines = []
    for row in rows:
        lines.append(
            "| {case} | {model} | {steps} | {hidden_dim} | {hidden_layers} | "
            "{num_spatial_frequencies} | {num_spectral_frequencies} | "
            "{reconstruction_rmse:.6f} | {support_reconstruction_rmse:.6f} | {signal_to_residual_db:.3f} | "
            "{total_flux_relative_error:.6f} | {fit_time_ms:.3f} |".format(**row)
        )
    return header + "\n".join(lines) + ("\n" if lines else "")


def _baseline_row(
    *,
    case: str,
    size: int,
    seed: int,
) -> dict[str, Any]:
    row = field_bench.evaluate_case(
        case=case,
        model_kind="coadd_bilinear",
        size=size,
        seed=seed,
        steps=1,
        runs=1,
    )
    return {
        "case": case,
        "model": "coadd_bilinear",
        "steps": 0,
        "hidden_dim": 0,
        "hidden_layers": 0,
        "num_spatial_frequencies": 0,
        "num_spectral_frequencies": 0,
        "reconstruction_rmse": row["reconstruction_rmse"],
        "support_reconstruction_rmse": row["support_reconstruction_rmse"],
        "signal_to_residual_db": row["signal_to_residual_db"],
        "total_flux_relative_error": row["total_flux_relative_error"],
        "fit_time_ms": row["fit_time_ms"],
    }


def _evaluate_field_model(
    *,
    model_kind: str,
    case: str,
    size: int,
    seed: int,
    config: SweepConfig,
) -> dict[str, Any]:
    spec = field_bench._build_case_specs(size)[case]
    with tempfile.TemporaryDirectory(prefix="torchsky-field-fit-sweep-") as tmpdir_str:
        observations, image_observations, truth_cube = (
            field_bench._simulate_observations(
                spec,
                seed=seed,
                tmpdir=Path(tmpdir_str),
            )
        )
        product = plan_map_product(
            image_observations,
            header_overrides=spec.target_header,
            spectral_axis=spec.spectral_axis,
            spectral_bin_edges=spec.spectral_bin_edges,
        )
        support_product = stack_map_product(
            image_observations,
            product,
            resampling_kernel="bilinear",
        )
        domain = estimate_field_domain(observations)
        if model_kind == "spectral_basis_field":
            model = SpectralBasisField(
                domain,
                hidden_dim=int(config.hidden_dim),
                hidden_layers=int(config.hidden_layers),
                num_spatial_frequencies=int(config.num_spatial_frequencies),
                num_spectral_frequencies=int(config.num_spectral_frequencies),
            )
        else:
            model = SkyEncoderMLPField(
                domain,
                hidden_dim=int(config.hidden_dim),
                hidden_layers=int(config.hidden_layers),
                num_fourier_frequencies=int(config.num_spatial_frequencies),
                spectral_frequencies=int(config.num_spectral_frequencies),
            )
        fit_kwargs = field_bench._field_fit_kwargs_for_model(model_kind, case=case)
        torch.manual_seed(int(seed) + 999)
        warmup_model = field_bench._make_field_model_for_case(
            model_kind,
            domain,
            case=case,
        )
        fit_field_model(
            observations,
            warmup_model,
            steps=int(config.steps),
            **fit_kwargs,
        )
        torch.manual_seed(int(seed) + 1000)
        start = time.perf_counter()
        result = fit_field_model(
            observations,
            model,
            steps=int(config.steps),
            **fit_kwargs,
        )
        fit_time_ms = (time.perf_counter() - start) * 1000.0
        product_pred = render_field_product(result.model, product)
        metrics = evaluate_map_product(product_pred, truth_cube)
        support_metrics = evaluate_map_product(
            product_pred.data,
            truth_cube,
            weight=support_product.weight,
            header=product_pred.header,
            value_semantics=product_pred.product.value_semantics,
        )
        signal_rms = float(torch.sqrt(torch.mean(torch.square(truth_cube))).item())
        residual_rms = float(metrics["reconstruction_rmse"])
        return {
            "case": case,
            "model": model_kind,
            "steps": int(config.steps),
            "hidden_dim": int(config.hidden_dim),
            "hidden_layers": int(config.hidden_layers),
            "num_spatial_frequencies": int(config.num_spatial_frequencies),
            "num_spectral_frequencies": int(config.num_spectral_frequencies),
            "reconstruction_rmse": residual_rms,
            "support_reconstruction_rmse": float(
                support_metrics["reconstruction_rmse"]  # type: ignore[arg-type]
            ),
            "signal_to_residual_db": 20.0
            * math.log10(signal_rms / max(residual_rms, 1.0e-8)),
            "total_flux_relative_error": float(metrics["total_flux_relative_error"]),
            "fit_time_ms": fit_time_ms,
            "final_loss": float(result.final_loss),
        }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default="tan_field_fit_exact_transmission,tpv_field_fit_deep_multiband_exact",
        help="Comma-separated benchmark cases.",
    )
    parser.add_argument(
        "--model-kind",
        choices=("spectral_basis_field", "sky_encoder_mlp"),
        default="spectral_basis_field",
    )
    parser.add_argument("--size", type=int, default=24)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=_parse_int_list, default=[32, 64])
    parser.add_argument("--hidden-dims", type=_parse_int_list, default=[96, 128])
    parser.add_argument("--hidden-layers", type=_parse_int_list, default=[2])
    parser.add_argument("--spatial-frequencies", type=_parse_int_list, default=[6, 8])
    parser.add_argument("--spectral-frequencies", type=_parse_int_list, default=[4, 6])
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    cases = [case.strip() for case in args.cases.split(",") if case.strip()]
    configs = [
        SweepConfig(*combo)
        for combo in product(
            args.steps,
            args.hidden_dims,
            args.hidden_layers,
            args.spatial_frequencies,
            args.spectral_frequencies,
        )
    ]
    rows: list[dict[str, Any]] = []
    for case in cases:
        rows.append(_baseline_row(case=case, size=args.size, seed=args.seed))
        for config in configs:
            rows.append(
                _evaluate_field_model(
                    model_kind=args.model_kind,
                    case=case,
                    size=int(args.size),
                    seed=int(args.seed),
                    config=config,
                )
            )
    rows.sort(
        key=lambda row: (
            row["case"],
            row["support_reconstruction_rmse"],
            row["fit_time_ms"],
        )
    )
    payload = {"rows": rows}
    if args.json_out is not None:
        args.json_out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.write_text(_markdown_table(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
