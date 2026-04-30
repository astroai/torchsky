#!/usr/bin/env python3
"""Diagnose per-observation reprojection differences on large-union truth cases."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
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

import bench_map_compat as compat_bench  # noqa: E402
import bench_map_ground_truth as truth_bench  # noqa: E402
import torchsky  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.map import (  # noqa: E402
    ImageObservation,
    evaluate_map_product,
    plan_map_product,
    stack_map_product,
)
import torchsky.map.image_plane as image_plane_mod  # noqa: E402

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover - native bench env requirement
    raise RuntimeError(
        "diagnose_map_large_union_reprojection.py requires torchsky.cpp; "
        "run it from a native-extension environment"
    ) from exc


def _time_many(fn, runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _target_setup(
    target_header: dict[str, Any], device: torch.device
) -> tuple[tuple[int, int], image_plane_mod._PreparedImagePlaneTarget]:
    target_shape = image_plane_mod._target_shape_from_header(
        target_header,
        fallback=(int(target_header["NAXIS2"]), int(target_header["NAXIS1"])),
    )
    prepared_target = image_plane_mod._prepare_target(
        header=target_header,
        target_wcs=WCS(target_header),
        target_shape=target_shape,
        device=device,
        include_pixel_area=False,
        include_quadrature=False,
    )
    return target_shape, prepared_target


def _torchsky_weighted_observation(
    obs: truth_bench.SimulatedObservation,
    *,
    target_header: dict[str, Any],
    prepared_target: image_plane_mod._PreparedImagePlaneTarget,
    target_shape: tuple[int, int],
    kernel: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    observation = obs.observation
    image, header = torchsky.read(
        observation.path, hdu=observation.hdu, return_header=True
    )
    image = image_plane_mod._validate_image_plane(image)
    header_dict_obs = dict(header)
    source_wcs = None
    if not (
        image_plane_mod._headers_equivalent_for_direct_copy(
            header_dict_obs, target_header
        )
        and tuple(image.shape) == target_shape
    ):
        source_wcs = WCS(header_dict_obs)
    numer, weight, _, _ = image_plane_mod._project_observation_to_target(
        observation,
        image=image,
        header_dict_obs=header_dict_obs,
        source_wcs=source_wcs,
        prepared_target=prepared_target,
        target_header=target_header,
        target_shape=target_shape,
        device=image.device,
        footprint_order=None,
        resampling_kernel=kernel,
        value_semantics="surface_brightness",
    )
    return numer, weight


def _coadd_from_terms(numer: torch.Tensor, weight: torch.Tensor) -> torch.Tensor:
    return torch.where(weight > 0.0, numer / weight, torch.zeros_like(numer))


def _observation_truth_metrics(
    data: torch.Tensor,
    weight: torch.Tensor,
    truth: torch.Tensor,
    *,
    target_header: dict[str, Any],
) -> dict[str, float | None]:
    return evaluate_map_product(
        data,
        truth,
        weight=weight,
        header=target_header,
        value_semantics="surface_brightness",
    )


def _row_for_observation(
    *,
    case: str,
    obs_index: int,
    backend: str,
    variant: str,
    numer: torch.Tensor,
    weight: torch.Tensor,
    truth: torch.Tensor,
    target_header: dict[str, Any],
    obs: truth_bench.SimulatedObservation,
    time_ms: float,
) -> dict[str, Any]:
    data = _coadd_from_terms(numer, weight)
    metrics = _observation_truth_metrics(
        data, weight, truth, target_header=target_header
    )
    return {
        "case": case,
        "row_type": "observation",
        "observation_index": obs_index,
        "backend": backend,
        "variant": variant,
        "time_ms": time_ms,
        "header_projection": str(obs.header.get("CTYPE1", "")),
        "input_width": int(obs.header["NAXIS1"]),
        "input_height": int(obs.header["NAXIS2"]),
        "coverage_fraction": float(
            (weight > 0.0).to(dtype=torch.float32).mean().item()
        ),
        "mean_weight": float(weight.mean().item()),
        "min_weight": float(weight.min().item()),
        "max_weight": float(weight.max().item()),
        "explicit_mask_fraction_input": float(obs.explicit_mask_fraction),
        "zero_weight_fraction_input": float(obs.zero_weight_fraction),
        **metrics,
    }


def _pair_row(
    *,
    case: str,
    obs_index: int,
    left_backend: str,
    left_variant: str,
    left_data: torch.Tensor,
    left_weight: torch.Tensor,
    right_backend: str,
    right_variant: str,
    right_data: torch.Tensor,
    right_weight: torch.Tensor,
) -> dict[str, Any]:
    return {
        "case": case,
        "row_type": "observation_pairwise",
        "observation_index": obs_index,
        "left_backend": left_backend,
        "left_variant": left_variant,
        "right_backend": right_backend,
        "right_variant": right_variant,
        **compat_bench._diff_summary(
            left_data,
            left_weight,
            right_data,
            right_weight,
            suffix="",
        ),
    }


def _stack_rows(
    *,
    case: str,
    obs_list: list[ImageObservation],
    target_header: dict[str, Any],
    truth: torch.Tensor,
    torchsky_kernel: str,
    dfreproject_kernel: str,
    runs: int,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    product = plan_map_product(
        obs_list,
        header_overrides=target_header,
        value_semantics="surface_brightness",
    )
    torchsky_dt = _time_many(
        lambda: stack_map_product(obs_list, product, resampling_kernel=torchsky_kernel),
        runs,
    )
    torchsky_result = stack_map_product(
        obs_list, product, resampling_kernel=torchsky_kernel
    )
    torchsky_metrics = evaluate_map_product(
        torchsky_result.data,
        truth,
        weight=torchsky_result.weight,
        header=target_header,
        value_semantics="surface_brightness",
    )
    rows.append(
        {
            "case": case,
            "row_type": "stack",
            "backend": "torchsky",
            "variant": torchsky_kernel,
            "time_ms": torchsky_dt * 1000.0,
            **torchsky_metrics,
        }
    )

    dfreproject_dt = _time_many(
        lambda: compat_bench._coadd_with_dfreproject(
            obs_list,
            target_header=target_header,
            order=dfreproject_kernel,
        ),
        runs,
    )
    df_data_np, df_weight_np = compat_bench._coadd_with_dfreproject(
        obs_list,
        target_header=target_header,
        order=dfreproject_kernel,
    )
    df_data = torch.from_numpy(df_data_np)
    df_weight = torch.from_numpy(df_weight_np)
    df_metrics = evaluate_map_product(
        df_data,
        truth,
        weight=df_weight,
        header=target_header,
        value_semantics="surface_brightness",
    )
    rows.append(
        {
            "case": case,
            "row_type": "stack",
            "backend": "dfreproject",
            "variant": dfreproject_kernel,
            "time_ms": dfreproject_dt * 1000.0,
            **df_metrics,
        }
    )
    rows.append(
        {
            "case": case,
            "row_type": "stack_pairwise",
            "left_backend": "torchsky",
            "left_variant": torchsky_kernel,
            "right_backend": "dfreproject",
            "right_variant": dfreproject_kernel,
            **compat_bench._diff_summary(
                torchsky_result.data.squeeze(0),
                torchsky_result.weight.squeeze(0),
                df_data,
                df_weight,
                suffix="",
            ),
        }
    )
    return rows


def _build_markdown(rows: list[dict[str, Any]]) -> str:
    obs_rows = [row for row in rows if row["row_type"] == "observation"]
    pair_rows = [row for row in rows if row["row_type"] == "observation_pairwise"]
    stack_rows = [row for row in rows if row["row_type"] == "stack"]
    stack_pair_rows = [row for row in rows if row["row_type"] == "stack_pairwise"]

    lines = ["# Large Union Reprojection Diagnostics", ""]
    if stack_rows:
        lines.extend(
            [
                "## Stack Summary",
                "",
                "| backend | variant | time_ms | reconstruction_rmse | signal_to_residual_db | total_flux_relative_error | coverage_fraction |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in stack_rows:
            lines.append(
                "| {backend} | {variant} | {time_ms:.6g} | {reconstruction_rmse:.6g} | {signal_to_residual_db:.6g} | {total_flux_relative_error:.6g} | {coverage_fraction:.6g} |".format(
                    **row
                )
            )
        lines.append("")
    if stack_pair_rows:
        row = stack_pair_rows[0]
        lines.extend(
            [
                "## Stack Pairwise",
                "",
                "| left | right | max_abs_data_diff | max_abs_data_diff_interior | max_abs_weight_diff | max_abs_weight_diff_interior |",
                "| --- | --- | --- | --- | --- | --- |",
                (
                    f"| {row['left_backend']}:{row['left_variant']} | "
                    f"{row['right_backend']}:{row['right_variant']} | "
                    f"{row['max_abs_data_diff']:.6g} | {row['max_abs_data_diff_interior']:.6g} | "
                    f"{row['max_abs_weight_diff']:.6g} | {row['max_abs_weight_diff_interior']:.6g} |"
                ),
                "",
            ]
        )
    if obs_rows:
        lines.extend(
            [
                "## Per-Observation",
                "",
                "| obs | backend | variant | proj | input_shape | time_ms | rmse | snr_db | flux_rel_err | coverage |",
                "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in obs_rows:
            lines.append(
                f"| {row['observation_index']} | {row['backend']} | {row['variant']} | "
                f"{row['header_projection']} | {row['input_width']}x{row['input_height']} | "
                f"{row['time_ms']:.6g} | {row['reconstruction_rmse']:.6g} | "
                f"{row['signal_to_residual_db']:.6g} | {row['total_flux_relative_error']:.6g} | "
                f"{row['coverage_fraction']:.6g} |"
            )
        lines.append("")
    if pair_rows:
        lines.extend(
            [
                "## Per-Observation Pairwise",
                "",
                "| obs | left | right | max_abs_data_diff | max_abs_data_diff_interior | max_abs_weight_diff | max_abs_weight_diff_interior |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
        )
        for row in pair_rows:
            lines.append(
                f"| {row['observation_index']} | {row['left_backend']}:{row['left_variant']} | "
                f"{row['right_backend']}:{row['right_variant']} | {row['max_abs_data_diff']:.6g} | "
                f"{row['max_abs_data_diff_interior']:.6g} | {row['max_abs_weight_diff']:.6g} | "
                f"{row['max_abs_weight_diff_interior']:.6g} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case", default="tan_random10_large_union_rect_weighted_masked"
    )
    parser.add_argument("--size", type=int, default=1024)
    parser.add_argument("--width", type=int, default=1024)
    parser.add_argument("--height", type=int, default=768)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--torchsky-kernel", default="bilinear")
    parser.add_argument("--dfreproject-kernel", default="bilinear")
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/map_large_union_reprojection_diagnostics.json"),
    )
    parser.add_argument(
        "--markdown-out",
        type=Path,
        default=Path("benchmark_results/map_large_union_reprojection_diagnostics.md"),
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_large_union_diag_") as tmp:
        tmpdir = Path(tmp)
        observations, target_header, truth = truth_bench.make_case(
            tmpdir,
            case=args.case,
            size=args.size,
            seed=args.seed,
            width=args.width,
            height=args.height,
        )
        first_image, _ = torchsky.read(
            observations[0].observation.path,
            hdu=observations[0].observation.hdu,
            return_header=True,
        )
        target_shape, prepared_target = _target_setup(
            target_header, device=first_image.device
        )

        for obs_index, obs in enumerate(observations):
            torchsky_dt = _time_many(
                lambda: _torchsky_weighted_observation(
                    obs,
                    target_header=target_header,
                    prepared_target=prepared_target,
                    target_shape=target_shape,
                    kernel=args.torchsky_kernel,
                ),
                args.runs,
            )
            torchsky_numer, torchsky_weight = _torchsky_weighted_observation(
                obs,
                target_header=target_header,
                prepared_target=prepared_target,
                target_shape=target_shape,
                kernel=args.torchsky_kernel,
            )
            rows.append(
                _row_for_observation(
                    case=args.case,
                    obs_index=obs_index,
                    backend="torchsky",
                    variant=args.torchsky_kernel,
                    numer=torchsky_numer,
                    weight=torchsky_weight,
                    truth=truth,
                    target_header=target_header,
                    obs=obs,
                    time_ms=torchsky_dt * 1000.0,
                )
            )

            dfreproject_dt = _time_many(
                lambda: compat_bench._dfreproject_weighted_observation(
                    obs.observation,
                    target_header=target_header,
                    order=args.dfreproject_kernel,
                ),
                args.runs,
            )
            df_numer_np, df_weight_np = compat_bench._dfreproject_weighted_observation(
                obs.observation,
                target_header=target_header,
                order=args.dfreproject_kernel,
            )
            df_numer = torch.from_numpy(df_numer_np)
            df_weight = torch.from_numpy(df_weight_np)
            rows.append(
                _row_for_observation(
                    case=args.case,
                    obs_index=obs_index,
                    backend="dfreproject",
                    variant=args.dfreproject_kernel,
                    numer=df_numer,
                    weight=df_weight,
                    truth=truth,
                    target_header=target_header,
                    obs=obs,
                    time_ms=dfreproject_dt * 1000.0,
                )
            )
            rows.append(
                _pair_row(
                    case=args.case,
                    obs_index=obs_index,
                    left_backend="torchsky",
                    left_variant=args.torchsky_kernel,
                    left_data=_coadd_from_terms(torchsky_numer, torchsky_weight),
                    left_weight=torchsky_weight,
                    right_backend="dfreproject",
                    right_variant=args.dfreproject_kernel,
                    right_data=_coadd_from_terms(df_numer, df_weight),
                    right_weight=df_weight,
                )
            )

        rows.extend(
            _stack_rows(
                case=args.case,
                obs_list=[item.observation for item in observations],
                target_header=target_header,
                truth=truth,
                torchsky_kernel=args.torchsky_kernel,
                dfreproject_kernel=args.dfreproject_kernel,
                runs=args.runs,
            )
        )

    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_build_markdown(rows), encoding="utf-8")
    print(args.json_out)
    print(args.markdown_out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
