#!/usr/bin/env python3
"""Sweep bilinear edge-support policies on the large-union TAN weighted case."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCHMARKS = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import bench_map_ground_truth as truth_bench  # noqa: E402
import bench_map_compat as compat_bench  # noqa: E402
import torchsky.map.image_plane as image_plane_mod  # noqa: E402
from torchsky.map import ImageObservation, coadd_image_plane  # noqa: E402
import torchsky.map.operators.resampling as resampling_mod  # noqa: E402
from torchsky.map.operators import validate_image_plane_2d  # noqa: E402


def _is_tpv(header: dict[str, object]) -> bool:
    return "TPV" in str(header.get("CTYPE1", "")) or "TPV" in str(
        header.get("CTYPE2", "")
    )


def _strip_observation(
    obs: ImageObservation,
    *,
    keep_weight: bool,
    keep_mask: bool,
) -> ImageObservation:
    return ImageObservation(
        path=obs.path,
        hdu=obs.hdu,
        weight=obs.weight,
        photometric_scale=obs.photometric_scale,
        background_offset=obs.background_offset,
        wavelength=obs.wavelength,
        spectral_min=obs.spectral_min,
        spectral_max=obs.spectral_max,
        transmission_center=obs.transmission_center,
        transmission_sigma=obs.transmission_sigma,
        band_name=obs.band_name,
        weight_path=obs.weight_path if keep_weight else None,
        weight_hdu=obs.weight_hdu,
        mask_path=obs.mask_path if keep_mask else None,
        mask_hdu=obs.mask_hdu,
    )


def _parse_values(text: str) -> list[float]:
    values: list[float] = []
    for part in text.split(","):
        item = part.strip()
        if not item:
            continue
        values.append(float(item))
    if not values:
        raise ValueError("at least one sweep value is required")
    return values


def _make_boundary_gate_wrapper(margin: float):
    original = image_plane_mod._resample_weighted_to_target

    def wrapped(
        image,
        source_weight,
        source_wcs,
        target,
        *,
        resampling_kernel,
        value_semantics,
    ):
        if resampling_kernel != "bilinear" or value_semantics != "surface_brightness":
            return original(
                image,
                source_weight,
                source_wcs,
                target,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
            )
        image = validate_image_plane_2d(image).to(dtype=torch.float32)
        source_weight = validate_image_plane_2d(source_weight).to(
            dtype=torch.float32
        )
        if (
            bool(getattr(source_wcs, "_is_tpv", False))
            or getattr(source_wcs, "sip", None) is not None
        ):
            return original(
                image,
                source_weight,
                source_wcs,
                target,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
            )

        h_src, w_src = image.shape
        weighted_image = image * source_weight
        weighted_weight = source_weight
        flat_ra = target.ra_deg.reshape(-1)
        flat_dec = target.dec_deg.reshape(-1)
        x_src, y_src = source_wcs.world_to_pixel(flat_ra, flat_dec)
        x_src = x_src.reshape_as(target.ra_deg)
        y_src = y_src.reshape_as(target.dec_deg)
        x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
        y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
        stacked = torch.stack([weighted_image, weighted_weight], dim=0)
        sampled = resampling_mod.sample_image_plane_batched_channels(
            stacked,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel="bilinear",
        )
        sampled_numer = sampled[0]
        sampled_weight = sampled[1]
        in_bounds = (
            (x_src >= float(margin))
            & (x_src <= float(w_src - 1) - float(margin))
            & (y_src >= float(margin))
            & (y_src <= float(h_src - 1) - float(margin))
        )
        sampled_numer = torch.where(
            in_bounds, sampled_numer, torch.zeros_like(sampled_numer)
        )
        sampled_weight = torch.where(
            in_bounds, sampled_weight, torch.zeros_like(sampled_weight)
        )
        sampled_weight = torch.where(
            sampled_weight > 0.0, sampled_weight, torch.zeros_like(sampled_weight)
        )
        sampled_numer = torch.where(
            sampled_weight > 0.0, sampled_numer, torch.zeros_like(sampled_numer)
        )
        return sampled_numer, sampled_weight

    return wrapped


def _make_coverage_gate_wrapper(min_coverage: float):
    original = image_plane_mod._resample_weighted_to_target

    def wrapped(
        image,
        source_weight,
        source_wcs,
        target,
        *,
        resampling_kernel,
        value_semantics,
    ):
        if resampling_kernel != "bilinear" or value_semantics != "surface_brightness":
            return original(
                image,
                source_weight,
                source_wcs,
                target,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
            )
        image = validate_image_plane_2d(image).to(dtype=torch.float32)
        source_weight = validate_image_plane_2d(source_weight).to(
            dtype=torch.float32
        )
        if (
            bool(getattr(source_wcs, "_is_tpv", False))
            or getattr(source_wcs, "sip", None) is not None
        ):
            return original(
                image,
                source_weight,
                source_wcs,
                target,
                resampling_kernel=resampling_kernel,
                value_semantics=value_semantics,
            )

        h_src, w_src = image.shape
        weighted_image = image * source_weight
        weighted_weight = source_weight
        support_image = torch.ones_like(source_weight, dtype=torch.float32)
        flat_ra = target.ra_deg.reshape(-1)
        flat_dec = target.dec_deg.reshape(-1)
        x_src, y_src = source_wcs.world_to_pixel(flat_ra, flat_dec)
        x_src = x_src.reshape_as(target.ra_deg)
        y_src = y_src.reshape_as(target.dec_deg)
        x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
        y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
        grid = torch.stack([x_norm, y_norm], dim=-1).to(dtype=torch.float32)
        stacked = torch.stack([weighted_image, weighted_weight, support_image], dim=0)
        sampled = resampling_mod.sample_image_plane_batched_channels(
            stacked,
            x_src=x_src,
            y_src=y_src,
            grid=grid,
            kernel="bilinear",
        )
        sampled_numer = sampled[0]
        sampled_weight = sampled[1]
        sampled_support = sampled[2]
        coverage_ok = sampled_support >= float(min_coverage)
        sampled_numer = torch.where(
            coverage_ok, sampled_numer, torch.zeros_like(sampled_numer)
        )
        sampled_weight = torch.where(
            coverage_ok, sampled_weight, torch.zeros_like(sampled_weight)
        )
        sampled_weight = torch.where(
            sampled_weight > 0.0, sampled_weight, torch.zeros_like(sampled_weight)
        )
        sampled_numer = torch.where(
            sampled_weight > 0.0, sampled_numer, torch.zeros_like(sampled_numer)
        )
        return sampled_numer, sampled_weight

    return wrapped


def _format_markdown(rows: list[dict[str, object]]) -> str:
    headers = [
        "policy",
        "value",
        "rmse_vs_truth",
        "mae_vs_truth",
        "coverage_fraction",
        "mean_weight",
        "max_abs_data_diff_vs_swarp",
        "max_abs_weight_diff_vs_swarp",
    ]
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        vals: list[str] = []
        for key in headers:
            value = row.get(key)
            if isinstance(value, float):
                vals.append(f"{value:.6g}")
            else:
                vals.append(str(value))
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="tan_random10_large_union_rect_weighted_masked"
    )
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scenario",
        choices=("full", "no_mask", "plain"),
        default="no_mask",
    )
    parser.add_argument(
        "--policy",
        choices=("margin", "coverage"),
        default="margin",
        help="Benchmark-only edge-support policy family to sweep",
    )
    parser.add_argument(
        "--values",
        default=None,
        help="Comma-separated sweep values; defaults depend on --policy",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    if args.values is None:
        if args.policy == "margin":
            values = [0.0, 0.0001, 0.001, 0.01, 0.1, 0.5]
        else:
            values = [0.0, 0.25, 0.5, 0.75, 0.9, 0.99, 0.999]
    else:
        values = _parse_values(args.values)
    original = image_plane_mod._resample_weighted_to_target

    with tempfile.TemporaryDirectory(prefix="torchsky_boundary_support_") as tmp:
        tmpdir = Path(tmp)
        observations, target_header, truth = truth_bench.make_case(
            tmpdir,
            case=args.case,
            size=args.size,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        tan_items = [item for item in observations if not _is_tpv(item.header)]
        if args.scenario == "full":
            obs_list = [
                _strip_observation(item.observation, keep_weight=True, keep_mask=True)
                for item in tan_items
            ]
        elif args.scenario == "no_mask":
            obs_list = [
                _strip_observation(item.observation, keep_weight=True, keep_mask=False)
                for item in tan_items
            ]
        else:
            obs_list = [
                _strip_observation(item.observation, keep_weight=False, keep_mask=False)
                for item in tan_items
            ]

        swarp_data_np, swarp_weight_np = compat_bench._coadd_with_swarp(
            obs_list,
            target_header=target_header,
            resampling_type="BILINEAR",
        )
        swarp_data = torch.from_numpy(swarp_data_np).to(dtype=torch.float32)
        swarp_weight = torch.from_numpy(swarp_weight_np).to(dtype=torch.float32)

        rows: list[dict[str, object]] = []
        for value in values:
            if args.policy == "margin":
                image_plane_mod._resample_weighted_to_target = (
                    _make_boundary_gate_wrapper(value)
                )
            else:
                image_plane_mod._resample_weighted_to_target = (
                    _make_coverage_gate_wrapper(value)
                )
            try:
                result = coadd_image_plane(
                    obs_list,
                    target_header=target_header,
                    resampling_kernel="bilinear",
                    value_semantics="surface_brightness",
                )
            finally:
                image_plane_mod._resample_weighted_to_target = original
            metrics = truth_bench._truth_metrics(
                result.data.to(dtype=torch.float32),
                result.weight.to(dtype=torch.float32),
                truth,
                target_header=target_header,
                value_semantics="surface_brightness",
            )
            pair = compat_bench._pairwise_backend_row(
                f"{args.case}_{args.width}x{args.height}",
                {
                    "library": "torchsky",
                    "variant": f"{args.policy}:{value}",
                    "time_ms": 1.0,
                },
                result.data.to(dtype=torch.float32),
                result.weight.to(dtype=torch.float32),
                {"library": "swarp", "variant": "bilinear", "time_ms": 1.0},
                swarp_data,
                swarp_weight,
            )
            rows.append(
                {
                    "policy": args.policy,
                    "value": float(value),
                    "rmse_vs_truth": float(metrics["reconstruction_rmse"]),
                    "mae_vs_truth": float(metrics["reconstruction_mae"]),
                    "coverage_fraction": float(metrics["coverage_fraction"]),
                    "mean_weight": float(result.weight.mean().item()),
                    "max_abs_data_diff_vs_swarp": float(pair["max_abs_data_diff"]),
                    "max_abs_weight_diff_vs_swarp": float(pair["max_abs_weight_diff"]),
                }
            )

    text = json.dumps(rows, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    markdown = _format_markdown(rows)
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(markdown)
    if args.json_out is not None:
        print(args.json_out)
    if args.markdown_out is not None:
        print(args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
