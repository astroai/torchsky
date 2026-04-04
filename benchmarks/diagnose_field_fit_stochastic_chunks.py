#!/usr/bin/env python3
"""Assemble a large product from many locally fit stochastic sky chunks."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import bench_map_field_fit as field_bench  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.map import MapProductResult, evaluate_map_product, plan_map_product, stack_map_product  # noqa: E402


def _hann2d(height: int, width: int, *, device: torch.device) -> torch.Tensor:
    def _axis_window(length: int) -> torch.Tensor:
        if length <= 1:
            return torch.ones((length,), dtype=torch.float32, device=device)
        return torch.hann_window(length, periodic=False, dtype=torch.float32, device=device)

    wy = _axis_window(height)
    wx = _axis_window(width)
    window = wy[:, None] * wx[None, :]
    return torch.clamp(window, min=1.0e-3)


def _sample_chunk_origins(
    *,
    width: int,
    height: int,
    chunk_size: int,
    stride: int,
    passes: int,
    jitter: int,
    seed: int,
) -> list[tuple[int, int]]:
    chunk = max(int(chunk_size), 1)
    stride = max(int(stride), 1)
    jitter = max(int(jitter), 0)
    passes = max(int(passes), 1)
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))

    def _positions(limit: int, *, offset: int) -> list[int]:
        starts = list(range(offset, max(limit - chunk + 1, 1), stride))
        starts.append(max(limit - chunk, 0))
        starts.append(0)
        ordered = sorted({min(max(pos, 0), max(limit - chunk, 0)) for pos in starts})
        return ordered

    origins: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for pass_idx in range(passes):
        base_x = 0 if pass_idx == 0 else int(torch.randint(stride, (1,), generator=gen).item())
        base_y = 0 if pass_idx == 0 else int(torch.randint(stride, (1,), generator=gen).item())
        xs = _positions(width, offset=base_x)
        ys = _positions(height, offset=base_y)
        for y0 in ys:
            for x0 in xs:
                if jitter > 0 and pass_idx > 0:
                    dx = int(torch.randint(2 * jitter + 1, (1,), generator=gen).item()) - jitter
                    dy = int(torch.randint(2 * jitter + 1, (1,), generator=gen).item()) - jitter
                    x = min(max(x0 + dx, 0), max(width - chunk, 0))
                    y = min(max(y0 + dy, 0), max(height - chunk, 0))
                else:
                    x, y = x0, y0
                key = (x, y)
                if key not in seen:
                    origins.append(key)
                    seen.add(key)
    return origins


def _header_overlap_score(source_header: dict[str, Any], target_header: dict[str, Any]) -> bool:
    source_wcs = WCS(source_header)
    target_wcs = WCS(target_header)
    source_h = int(source_header.get("NAXIS2", 0))
    source_w = int(source_header.get("NAXIS1", 0))
    target_h = int(target_header.get("NAXIS2", 0))
    target_w = int(target_header.get("NAXIS1", 0))
    if source_h <= 0 or source_w <= 0 or target_h <= 0 or target_w <= 0:
        return False
    sample_points = torch.tensor(
        [
            [0.0, 0.0],
            [0.0, float(target_h - 1)],
            [float(target_w - 1), 0.0],
            [float(target_w - 1), float(target_h - 1)],
            [0.5 * float(target_w - 1), 0.5 * float(target_h - 1)],
        ],
        dtype=torch.float64,
    )
    ra_deg, dec_deg = target_wcs.pixel_to_world(sample_points[:, 0], sample_points[:, 1])
    x_pix, y_pix = source_wcs.world_to_pixel(ra_deg, dec_deg)
    inside = (
        (x_pix >= 0.0)
        & (x_pix <= float(source_w - 1))
        & (y_pix >= 0.0)
        & (y_pix <= float(source_h - 1))
    )
    return bool(torch.any(inside))


def _assemble_stochastic_chunk_product(
    *,
    case: str,
    model_kind: str,
    size: int,
    seed: int,
    steps: int,
    runs: int,
    chunk_size: int,
    stride: int,
    passes: int,
    jitter: int,
    local_size_hint: int | None,
    max_chunks: int | None,
) -> dict[str, Any]:
    case_spec = field_bench._build_case_specs(size)[case]
    with tempfile.TemporaryDirectory(prefix="torchsky-field-fit-chunks-") as tmpdir_str:
        observations, image_observations, truth_cube = field_bench._simulate_observations(
            case_spec,
            seed=seed,
            tmpdir=Path(tmpdir_str),
        )
        product = plan_map_product(
            image_observations,
            header_overrides=case_spec.target_header,
            spectral_axis=case_spec.spectral_axis,
            spectral_bin_edges=case_spec.spectral_bin_edges,
        )
        support_reference_product = stack_map_product(
            image_observations,
            product,
            resampling_kernel="bilinear",
        )
        height = int(product.target_header["NAXIS2"])
        width = int(product.target_header["NAXIS1"])
        origins = _sample_chunk_origins(
            width=width,
            height=height,
            chunk_size=chunk_size,
            stride=stride,
            passes=passes,
            jitter=jitter,
            seed=seed,
        )
        if max_chunks is not None and max_chunks > 0:
            origins = origins[: int(max_chunks)]
        assembled_data = torch.zeros_like(support_reference_product.data, dtype=torch.float32)
        assembled_weight = torch.zeros_like(support_reference_product.weight, dtype=torch.float32)
        fit_times: list[float] = []
        used_chunks = 0
        for chunk_index, (x0, y0) in enumerate(origins):
            tile_w = min(int(chunk_size), width - int(x0))
            tile_h = min(int(chunk_size), height - int(y0))
            chunk_header = field_bench._copy_header_with_crop(
                dict(product.target_header),
                x0=int(x0),
                y0=int(y0),
                width=int(tile_w),
                height=int(tile_h),
            )
            selected_indices = [
                idx
                for idx, observation in enumerate(observations)
                if _header_overlap_score(observation.header, chunk_header)
            ]
            if not selected_indices:
                continue
            local_observations = [observations[idx] for idx in selected_indices]
            local_image_observations = [image_observations[idx] for idx in selected_indices]
            local_product = plan_map_product(
                local_image_observations,
                header_overrides=chunk_header,
                spectral_axis=case_spec.spectral_axis,
                spectral_bin_edges=case_spec.spectral_bin_edges,
            )
            local_domain = field_bench.estimate_field_domain(
                local_observations,
                coord_mode=case_spec.domain.coord_mode,
            )
            fit_kwargs = field_bench._field_fit_kwargs_for_model(
                model_kind,
                case=case_spec.case,
                size=int(local_size_hint) if local_size_hint is not None else chunk_size,
                benchmark_steps=steps,
            )
            model = field_bench._make_field_model_for_case(
                model_kind,
                local_domain,
                case=case_spec.case,
                size=int(local_size_hint) if local_size_hint is not None else chunk_size,
            )
            start = field_bench.time.perf_counter()
            result = field_bench.fit_field_model(
                local_observations,
                model,
                steps=steps,
                **fit_kwargs,
            )
            fit_times.append((field_bench.time.perf_counter() - start) * 1000.0)
            local_pred = field_bench._render_field_product_chunked(
                result.model,
                local_product,
                spatial_chunk_size=None,
            )
            taper = _hann2d(tile_h, tile_w, device=local_pred.data.device)
            local_support = stack_map_product(
                local_image_observations,
                local_product,
                resampling_kernel="bilinear",
            )
            support_mask = local_support.weight.gt(0.0).to(dtype=torch.float32)
            blend_weight = taper.unsqueeze(0) * (0.25 + 0.75 * support_mask)
            assembled_data[:, y0 : y0 + tile_h, x0 : x0 + tile_w] += local_pred.data.to(dtype=torch.float32) * blend_weight
            assembled_weight[:, y0 : y0 + tile_h, x0 : x0 + tile_w] += blend_weight
            used_chunks += 1
        normalized = assembled_data / torch.clamp(assembled_weight, min=1.0e-6)
        stitched = MapProductResult(
            product=product,
            data=normalized,
            weight=assembled_weight,
            header=dict(product.target_header),
            overlap_count=(assembled_weight > 0.0).to(dtype=torch.int32),
            coverage_mocs=None,
        )
        metrics = evaluate_map_product(stitched, truth_cube)
        support_metrics = evaluate_map_product(
            stitched.data,
            truth_cube,
            weight=support_reference_product.weight,
            header=stitched.header,
            value_semantics=stitched.product.value_semantics,
        )
    return {
        "case": case,
        "model_kind": model_kind,
        "size": int(size),
        "local_steps": int(steps),
        "chunk_size": int(chunk_size),
        "stride": int(stride),
        "passes": int(passes),
        "jitter": int(jitter),
        "used_chunks": int(used_chunks),
        "fit_time_ms_mean": float(sum(fit_times) / max(len(fit_times), 1)),
        "fit_time_ms_total": float(sum(fit_times)),
        "reconstruction_rmse": float(metrics["reconstruction_rmse"]),
        "support_reconstruction_rmse": float(support_metrics["reconstruction_rmse"]),
        "signal_to_residual_db": float(metrics["signal_to_residual_db"]),
    }


def _markdown(rows: list[dict[str, Any]]) -> str:
    header = (
        "| variant | rmse | support_rmse | snr_db | fit_time_ms | details |\n"
        "| --- | ---: | ---: | ---: | ---: | --- |\n"
    )
    lines = []
    for row in rows:
        label = row.get("label", row.get("model_kind", "variant"))
        details = ", ".join(
            f"{key}={row[key]}"
            for key in ("chunk_size", "stride", "passes", "jitter", "used_chunks")
            if key in row
        )
        lines.append(
            f"| {label} | {row['reconstruction_rmse']:.6f} | {row['support_reconstruction_rmse']:.6f} | "
            f"{row['signal_to_residual_db']:.3f} | {row.get('fit_time_ms', row.get('fit_time_ms_total', 0.0)):.3f} | {details} |"
        )
    return header + "\n".join(lines) + ("\n" if lines else "")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--case", default="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap")
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--steps", type=int, default=12, help="Local chunk fit steps.")
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--model", default="sky_lambda_feature_field")
    parser.add_argument("--chunk-size", type=int, default=32)
    parser.add_argument("--stride", type=int, default=32)
    parser.add_argument("--passes", type=int, default=1)
    parser.add_argument("--jitter", type=int, default=0)
    parser.add_argument(
        "--local-size-hint",
        type=int,
        default=None,
        help="Optional size hint for choosing local fit/model defaults; use 128 to force the lean large-profile contract on local chunks.",
    )
    parser.add_argument(
        "--max-chunks",
        type=int,
        default=None,
        help="Optional cap on the number of local chunks to fit.",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    coadd_row = field_bench.evaluate_case(
        case=args.case,
        model_kind="coadd_bilinear",
        size=args.size,
        seed=args.seed,
        steps=1,
        runs=args.runs,
    )
    coadd_row["label"] = "coadd_bilinear"
    rows.append(coadd_row)

    global_row = field_bench.evaluate_case(
        case=args.case,
        model_kind=args.model,
        size=args.size,
        seed=args.seed,
        steps=args.steps,
        runs=args.runs,
    )
    global_row["label"] = f"{args.model}_global"
    rows.append(global_row)

    chunk_row = _assemble_stochastic_chunk_product(
        case=args.case,
        model_kind=args.model,
        size=args.size,
        seed=args.seed,
        steps=args.steps,
        runs=args.runs,
        chunk_size=args.chunk_size,
        stride=args.stride,
        passes=args.passes,
        jitter=args.jitter,
        local_size_hint=args.local_size_hint,
        max_chunks=args.max_chunks,
    )
    chunk_row["label"] = f"{args.model}_chunk_ensemble"
    rows.append(chunk_row)

    if args.json_out is not None:
        args.json_out.write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.write_text(_markdown(rows), encoding="utf-8")
    print(json.dumps(rows, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
