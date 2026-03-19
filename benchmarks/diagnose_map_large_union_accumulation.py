#!/usr/bin/env python3
"""Diagnose large-union accumulation differences across grouped and per-observation outputs."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
from pathlib import Path
from typing import Any

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
from torchsky.map import ImageObservation, MapResult, coadd_image_plane  # noqa: E402


def _is_tpv(header: dict[str, object]) -> bool:
    return "TPV" in str(header.get("CTYPE1", "")) or "TPV" in str(header.get("CTYPE2", ""))


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


def _squeeze2(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 3 and int(x.shape[0]) == 1:
        return x.squeeze(0)
    return x


def _torchsky_grouped(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
) -> MapResult:
    result = coadd_image_plane(
        observations,
        target_header=target_header,
        resampling_kernel="bilinear",
        value_semantics="surface_brightness",
    )
    return MapResult(
        data=_squeeze2(result.data.to(dtype=torch.float32)),
        weight=_squeeze2(result.weight.to(dtype=torch.float32)),
        header=result.header,
        overlap_count=_squeeze2(result.overlap_count) if result.overlap_count is not None else None,
        coverage_moc=result.coverage_moc,
        observation_mocs=result.observation_mocs,
    )


def _torchsky_per_observation_sum(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    numer_sum: torch.Tensor | None = None
    denom_sum: torch.Tensor | None = None
    for obs in observations:
        result = _torchsky_grouped([obs], target_header)
        data = result.data
        weight = result.weight
        if numer_sum is None or denom_sum is None:
            numer_sum = torch.zeros_like(data)
            denom_sum = torch.zeros_like(weight)
        numer_sum = numer_sum + data * weight
        denom_sum = denom_sum + weight
    if numer_sum is None or denom_sum is None:
        raise RuntimeError("at least one observation is required")
    data = torch.where(denom_sum > 0.0, numer_sum / denom_sum, torch.zeros_like(numer_sum))
    data = torch.where(denom_sum > 0.0, data, torch.zeros_like(data))
    return data, denom_sum


def _swarp_grouped(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    data_np, weight_np = compat_bench._coadd_with_swarp(
        observations,
        target_header=target_header,
        resampling_type="BILINEAR",
    )
    return torch.from_numpy(data_np).to(dtype=torch.float32), torch.from_numpy(weight_np).to(dtype=torch.float32)


def _swarp_per_observation_sum(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
) -> tuple[torch.Tensor, torch.Tensor]:
    numer_sum: torch.Tensor | None = None
    denom_sum: torch.Tensor | None = None
    for obs in observations:
        data, weight = _swarp_grouped([obs], target_header)
        if numer_sum is None or denom_sum is None:
            numer_sum = torch.zeros_like(data)
            denom_sum = torch.zeros_like(weight)
        numer_sum = numer_sum + data * weight
        denom_sum = denom_sum + weight
    if numer_sum is None or denom_sum is None:
        raise RuntimeError("at least one observation is required")
    data = torch.where(denom_sum > 0.0, numer_sum / denom_sum, torch.zeros_like(numer_sum))
    data = torch.where(denom_sum > 0.0, data, torch.zeros_like(data))
    return data, denom_sum


def _pairwise_stats(
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
) -> dict[str, float]:
    overlap = (left_weight > 0.0) & (right_weight > 0.0)
    data_diff = torch.abs(left_data - right_data)
    weight_diff = torch.abs(left_weight - right_weight)
    return {
        "coverage_left": float((left_weight > 0.0).to(torch.float32).mean().item()),
        "coverage_right": float((right_weight > 0.0).to(torch.float32).mean().item()),
        "max_abs_data_diff": float(data_diff.max().item()),
        "max_abs_weight_diff": float(weight_diff.max().item()),
        "mean_abs_data_diff_overlap": (
            float(data_diff[overlap].mean().item()) if bool(overlap.any()) else 0.0
        ),
        "mean_abs_weight_diff_overlap": (
            float(weight_diff[overlap].mean().item()) if bool(overlap.any()) else 0.0
        ),
    }


def _masked_scalar_stats(values: torch.Tensor, mask: torch.Tensor) -> dict[str, float]:
    if not bool(mask.any()):
        return {"count": 0.0, "mean": 0.0, "p95": 0.0, "max": 0.0}
    selected = values[mask].to(dtype=torch.float32)
    return {
        "count": float(selected.numel()),
        "mean": float(selected.mean().item()),
        "p95": float(torch.quantile(selected, 0.95).item()),
        "max": float(selected.max().item()),
    }


def _residual_stats(
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
    *,
    mask: torch.Tensor,
) -> dict[str, dict[str, float]]:
    numer_left = left_data * left_weight
    numer_right = right_data * right_weight
    return {
        "abs_data_diff": _masked_scalar_stats(torch.abs(left_data - right_data), mask),
        "abs_weight_diff": _masked_scalar_stats(torch.abs(left_weight - right_weight), mask),
        "abs_numer_diff": _masked_scalar_stats(torch.abs(numer_left - numer_right), mask),
    }


def _overlap_histogram(overlap_count: torch.Tensor) -> dict[str, float]:
    out: dict[str, float] = {}
    total = max(int(overlap_count.numel()), 1)
    max_overlap = int(overlap_count.max().item()) if overlap_count.numel() > 0 else 0
    for level in range(max_overlap + 1):
        frac = float((overlap_count == level).to(torch.float32).sum().item()) / float(total)
        out[str(level)] = frac
    return out


def _residuals_by_overlap(
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
    overlap_count: torch.Tensor,
) -> dict[str, dict[str, dict[str, float]]]:
    result: dict[str, dict[str, dict[str, float]]] = {}
    max_overlap = int(overlap_count.max().item()) if overlap_count.numel() > 0 else 0
    overlap_mask = (left_weight > 0.0) & (right_weight > 0.0)
    for level in range(1, max_overlap + 1):
        result[f"eq_{level}"] = _residual_stats(
            left_data,
            left_weight,
            right_data,
            right_weight,
            mask=overlap_mask & (overlap_count == level),
        )
    for level in range(2, max_overlap + 1):
        result[f"ge_{level}"] = _residual_stats(
            left_data,
            left_weight,
            right_data,
            right_weight,
            mask=overlap_mask & (overlap_count >= level),
        )
    return result


def _hotspot(
    values: torch.Tensor,
    *,
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
    overlap_count: torch.Tensor,
) -> dict[str, float | int]:
    flat_idx = int(torch.argmax(values).item())
    y = flat_idx // int(values.shape[1])
    x = flat_idx % int(values.shape[1])
    return {
        "x": x,
        "y": y,
        "value": float(values[y, x].item()),
        "overlap_count": int(overlap_count[y, x].item()),
        "left_data": float(left_data[y, x].item()),
        "right_data": float(right_data[y, x].item()),
        "left_weight": float(left_weight[y, x].item()),
        "right_weight": float(right_weight[y, x].item()),
        "left_numer": float((left_data[y, x] * left_weight[y, x]).item()),
        "right_numer": float((right_data[y, x] * right_weight[y, x]).item()),
    }


def _obs_metadata(item: truth_bench.SimulatedObservation) -> dict[str, float | int]:
    header = item.header
    cd11 = float(header["CD1_1"])
    cd12 = float(header["CD1_2"])
    cd21 = float(header["CD2_1"])
    cd22 = float(header["CD2_2"])
    return {
        "width": int(header["NAXIS1"]),
        "height": int(header["NAXIS2"]),
        "scale_deg": math.sqrt(abs(cd11 * cd22 - cd12 * cd21)),
        "rotation_deg": math.degrees(math.atan2(cd12, -cd11)),
        "crval1": float(header["CRVAL1"]),
        "crval2": float(header["CRVAL2"]),
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="tan_random10_large_union_rect_weighted_masked")
    parser.add_argument("--size", type=int, default=2048)
    parser.add_argument("--width", type=int, default=2048)
    parser.add_argument("--height", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--scenario",
        choices=("full", "no_mask", "plain"),
        default="no_mask",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    with tempfile.TemporaryDirectory(prefix="torchsky_diag_accum_") as tmp:
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
            obs_list = [_strip_observation(item.observation, keep_weight=True, keep_mask=True) for item in tan_items]
        elif args.scenario == "no_mask":
            obs_list = [_strip_observation(item.observation, keep_weight=True, keep_mask=False) for item in tan_items]
        else:
            obs_list = [_strip_observation(item.observation, keep_weight=False, keep_mask=False) for item in tan_items]

        torchsky_grouped = _torchsky_grouped(obs_list, target_header)
        torchsky_grouped_data = torchsky_grouped.data
        torchsky_grouped_weight = torchsky_grouped.weight
        torchsky_sum_data, torchsky_sum_weight = _torchsky_per_observation_sum(obs_list, target_header)
        swarp_grouped_data, swarp_grouped_weight = _swarp_grouped(obs_list, target_header)
        swarp_sum_data, swarp_sum_weight = _swarp_per_observation_sum(obs_list, target_header)
        if torchsky_grouped.overlap_count is None:
            raise RuntimeError("torchsky grouped diagnostic requires overlap_count")
        overlap_count = torchsky_grouped.overlap_count.to(dtype=torch.int32)
        grouped_data_diff = torch.abs(torchsky_grouped_data - swarp_grouped_data)
        grouped_weight_diff = torch.abs(torchsky_grouped_weight - swarp_grouped_weight)

        result = {
            "case": args.case,
            "scenario": args.scenario,
            "size": args.size,
            "width": args.width,
            "height": args.height,
            "n_tan_observations": len(obs_list),
            "overlap_histogram": _overlap_histogram(overlap_count),
            "torchsky_grouped_vs_sum": _pairwise_stats(
                torchsky_grouped_data,
                torchsky_grouped_weight,
                torchsky_sum_data,
                torchsky_sum_weight,
            ),
            "swarp_grouped_vs_sum": _pairwise_stats(
                swarp_grouped_data,
                swarp_grouped_weight,
                swarp_sum_data,
                swarp_sum_weight,
            ),
            "torchsky_grouped_vs_swarp_grouped": _pairwise_stats(
                torchsky_grouped_data,
                torchsky_grouped_weight,
                swarp_grouped_data,
                swarp_grouped_weight,
            ),
            "torchsky_vs_swarp_by_overlap": _residuals_by_overlap(
                torchsky_grouped_data,
                torchsky_grouped_weight,
                swarp_grouped_data,
                swarp_grouped_weight,
                overlap_count,
            ),
            "max_abs_data_diff_hotspot": _hotspot(
                grouped_data_diff,
                left_data=torchsky_grouped_data,
                left_weight=torchsky_grouped_weight,
                right_data=swarp_grouped_data,
                right_weight=swarp_grouped_weight,
                overlap_count=overlap_count,
            ),
            "max_abs_weight_diff_hotspot": _hotspot(
                grouped_weight_diff,
                left_data=torchsky_grouped_data,
                left_weight=torchsky_grouped_weight,
                right_data=swarp_grouped_data,
                right_weight=swarp_grouped_weight,
                overlap_count=overlap_count,
            ),
            "torchsky_truth": truth_bench._truth_metrics(
                torchsky_grouped_data,
                torchsky_grouped_weight,
                truth,
                target_header=target_header,
                value_semantics="surface_brightness",
            ),
            "swarp_truth": truth_bench._truth_metrics(
                swarp_grouped_data,
                swarp_grouped_weight,
                truth,
                target_header=target_header,
                value_semantics="surface_brightness",
            ),
            "per_observation": [],
        }
        for idx, item in enumerate(tan_items):
            obs = obs_list[idx]
            ts_result = _torchsky_grouped([obs], target_header)
            ts_data = ts_result.data
            ts_weight = ts_result.weight
            sw_data, sw_weight = _swarp_grouped([obs], target_header)
            entry = {
                "obs_index": idx,
                **_obs_metadata(item),
                "torchsky_truth": truth_bench._truth_metrics(
                    ts_data,
                    ts_weight,
                    truth,
                    target_header=target_header,
                    value_semantics="surface_brightness",
                ),
                "swarp_truth": truth_bench._truth_metrics(
                    sw_data,
                    sw_weight,
                    truth,
                    target_header=target_header,
                    value_semantics="surface_brightness",
                ),
                "torchsky_vs_swarp": _pairwise_stats(ts_data, ts_weight, sw_data, sw_weight),
            }
            result["per_observation"].append(entry)

    text = json.dumps(result, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
