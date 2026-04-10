#!/usr/bin/env python3
"""Joint multiband real-data field-fit benchmark on JWST SMACS J0723 products."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_field_fit import _field_fit_kwargs_for_model, _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    _PerExposureScalarBias,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _robust_intensity_scale,
)
from bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    _approx_overlap_grid,
    _best_overlap_cutout,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _field_observation_kwargs_for_band,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _reference_path_for_band,
    _sci_header,
)

DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_joint.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_joint.md"
DEFAULT_EXPORT_DIR = DATA / "benchmark_results" / "smacs0723_field_fit_joint_exports"

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ForwardContext,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_observation,
)


REAL_JOINT_CASE = "tpv_field_fit_deep_multiband_broad"


def _parse_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _safe_slug(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "item"


def _preview_uint8(image: np.ndarray) -> np.ndarray:
    finite = np.isfinite(image)
    if not np.any(finite):
        return np.zeros(image.shape, dtype=np.uint8)
    values = image[finite].astype(np.float64, copy=False)
    lo = float(np.percentile(values, 1.0))
    hi = float(np.percentile(values, 99.0))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        lo = float(np.min(values))
        hi = float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        return np.zeros(image.shape, dtype=np.uint8)
    clipped = np.clip((image.astype(np.float64) - lo) / (hi - lo), 0.0, 1.0)
    clipped = np.where(finite, clipped, 0.0)
    return np.rint(255.0 * clipped).astype(np.uint8)


def _write_export_products(
    export_dir: Path,
    *,
    band: str,
    model: str,
    header: dict[str, Any],
    reference: np.ndarray,
    prediction: np.ndarray,
    support_mask: np.ndarray,
    source_mask: np.ndarray,
) -> None:
    model_dir = export_dir / _safe_slug(band) / _safe_slug(model)
    model_dir.mkdir(parents=True, exist_ok=True)
    residual = prediction - reference
    torchsky.write(
        str(model_dir / "reference.fits"),
        torch.from_numpy(reference.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchsky.write(
        str(model_dir / "prediction.fits"),
        torch.from_numpy(prediction.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchsky.write(
        str(model_dir / "residual.fits"),
        torch.from_numpy(residual.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchsky.write(
        str(model_dir / "support_mask.fits"),
        torch.from_numpy(support_mask.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    torchsky.write(
        str(model_dir / "source_mask.fits"),
        torch.from_numpy(source_mask.astype(np.float32, copy=False)),
        header=dict(header),
        overwrite=True,
    )
    Image.fromarray(_preview_uint8(reference), mode="L").save(model_dir / "reference.png")
    Image.fromarray(_preview_uint8(prediction), mode="L").save(model_dir / "prediction.png")
    Image.fromarray(_preview_uint8(residual), mode="L").save(model_dir / "residual.png")
    Image.fromarray(support_mask.astype(np.uint8) * 255, mode="L").save(model_dir / "support_mask.png")
    Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").save(model_dir / "source_mask.png")


def _source_metric_mask(
    pred: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> np.ndarray:
    base = _metric_mask(pred, reference, support)
    if not np.any(base):
        return base
    ref_values = reference[base].astype(np.float64, copy=False)
    median = float(np.median(ref_values))
    mad = float(np.median(np.abs(ref_values - median)))
    robust_sigma = 1.4826 * mad
    percentile_threshold = float(np.percentile(ref_values, 90.0))
    if not np.isfinite(robust_sigma) or robust_sigma <= 0.0:
        threshold = percentile_threshold
    else:
        threshold = min(median + 3.0 * robust_sigma, percentile_threshold)
    source = base & (reference >= threshold)
    if np.any(source):
        return source
    source = base & (reference >= float(np.percentile(ref_values, 80.0)))
    if np.any(source):
        return source
    return base


def _background_subtracted_source_stats(
    pred: np.ndarray,
    reference: np.ndarray,
    support: np.ndarray,
) -> dict[str, float]:
    base = _metric_mask(pred, reference, support)
    if not np.any(base):
        return {
            "source_rmse": float("nan"),
            "source_contrast_rmse": float("nan"),
            "source_contrast_slope": float("nan"),
            "source_flux_ratio": float("nan"),
            "source_frac": 0.0,
        }
    source = _source_metric_mask(pred, reference, support)
    pred64 = pred.astype(np.float64, copy=False)
    ref64 = reference.astype(np.float64, copy=False)
    source_pred = pred64[source]
    source_ref = ref64[source]
    source_diff = source_pred - source_ref
    background = base & ~source
    if np.any(background):
        pred_bg = float(np.median(pred64[background]))
        ref_bg = float(np.median(ref64[background]))
    else:
        pred_bg = float(np.median(pred64[base]))
        ref_bg = float(np.median(ref64[base]))
    source_pred_contrast = source_pred - pred_bg
    source_ref_contrast = source_ref - ref_bg
    contrast_diff = source_pred_contrast - source_ref_contrast
    if source_ref_contrast.size:
        design = source_ref_contrast.reshape(-1, 1)
        slope_arr, *_ = np.linalg.lstsq(design, source_pred_contrast, rcond=None)
        contrast_slope = float(slope_arr[0])
    else:
        contrast_slope = float("nan")
    source_flux_ratio = (
        float(np.sum(source_pred_contrast) / np.sum(source_ref_contrast))
        if source_ref_contrast.size and np.sum(source_ref_contrast) != 0.0
        else float("nan")
    )
    return {
        "source_rmse": float(np.sqrt(np.mean(source_diff * source_diff)))
        if source_diff.size
        else float("nan"),
        "source_contrast_rmse": float(np.sqrt(np.mean(contrast_diff * contrast_diff)))
        if contrast_diff.size
        else float("nan"),
        "source_contrast_slope": contrast_slope,
        "source_flux_ratio": source_flux_ratio,
        "source_frac": float(source.mean()),
    }


def _joint_real_fit_kwargs(
    model_kind: str,
    *,
    crop_size: int,
    steps: int,
    num_observations: int,
) -> dict[str, Any]:
    kwargs = _field_fit_kwargs_for_model(
        model_kind,
        case=REAL_JOINT_CASE,
        size=int(crop_size),
        benchmark_steps=int(steps),
    )
    if model_kind == "coadd_bilinear":
        return kwargs
    kwargs["observations_per_step"] = min(max(num_observations, 1), 12)
    kwargs["observation_sampling"] = "band_balanced"
    kwargs["training_patch_size"] = None
    kwargs["pixel_samples_per_observation"] = max(256, min(int(crop_size) * int(crop_size), 1024))
    kwargs["loss_kind"] = "mse"
    kwargs["loss_reduction"] = "global"
    kwargs["grad_clip_norm"] = 1.0
    kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 1.0e-3)
    kwargs["lr_warmup_steps"] = max(int(kwargs.get("lr_warmup_steps", 0)), 4)
    kwargs["context_observations_per_step"] = min(
        int(kwargs.get("context_observations_per_step", 0)),
        1,
    )
    kwargs["full_observations_per_step"] = 0
    if model_kind in {"spectral_basis_field", "sky_encoder_resmlp", "sky_encoder_mlp"}:
        kwargs["num_spectral_samples"] = min(int(kwargs.get("num_spectral_samples", 9)), 5)
    if model_kind == "sky_lambda_feature_field":
        kwargs["num_spectral_samples"] = min(int(kwargs.get("num_spectral_samples", 9)), 5)
        kwargs["spectral_anchor_observations_per_step"] = min(
            int(kwargs.get("spectral_anchor_observations_per_step", 0)),
            1,
        )
    if model_kind == "operator_constrained_feature_field":
        kwargs["num_spectral_samples"] = min(int(kwargs.get("num_spectral_samples", 7)), 5)
        kwargs["context_loss_weight"] = min(float(kwargs.get("context_loss_weight", 0.0)), 0.05)
    if model_kind == "spectral_basis_field":
        kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 5.0e-4)
    return kwargs


def _summarize_by_model(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row["status"] == "OK":
            grouped.setdefault(str(row["model"]), []).append(row)
    summary: list[dict[str, Any]] = []
    for model, model_rows in grouped.items():
        summary.append(
            {
                "model": model,
                "bands": len(model_rows),
                "mean_support_rmse": float(np.mean([float(r["support_rmse"]) for r in model_rows])),
                "mean_source_rmse": float(np.mean([float(r["source_rmse"]) for r in model_rows])),
                "mean_source_contrast_rmse": float(
                    np.mean([float(r["source_contrast_rmse"]) for r in model_rows])
                ),
                "mean_uniform_rmse": float(np.mean([float(r["uniform_rmse"]) for r in model_rows])),
                "mean_aligned_rmse": float(np.mean([float(r["aligned_rmse"]) for r in model_rows])),
                "mean_affine_slope": float(np.mean([float(r["affine_slope"]) for r in model_rows])),
                "mean_source_contrast_slope": float(
                    np.mean([float(r["source_contrast_slope"]) for r in model_rows])
                ),
                "mean_support_flux_ratio": float(
                    np.mean([float(r["support_flux_ratio"]) for r in model_rows])
                ),
                "mean_source_flux_ratio": float(
                    np.mean([float(r["source_flux_ratio"]) for r in model_rows])
                ),
                "total_fit_time_ms": float(np.max([float(r["fit_time_ms"]) for r in model_rows])),
            }
        )
    summary.sort(key=lambda row: row["mean_source_contrast_rmse"])
    return summary


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    ok_rows = [row for row in rows if row["status"] == "OK"]
    summary = _summarize_by_model(ok_rows)
    lines = [
        "# SMACS J0723 Joint Multiband Field-Fit",
        "",
        "## Model Summary",
        "",
        "| model | bands | mean source contrast rmse | mean source rmse | mean support rmse | mean aligned rmse | mean source slope | fit ms |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            f"| {row['model']} | {row['bands']} | {row['mean_source_contrast_rmse']:.6f} | "
            f"{row['mean_source_rmse']:.6f} | {row['mean_support_rmse']:.6f} | "
            f"{row['mean_aligned_rmse']:.6f} | {row['mean_source_contrast_slope']:.4f} | "
            f"{row['total_fit_time_ms']:.2f} |"
        )
    lines.extend(
        [
            "",
            "## Per-Band Rows",
            "",
            "| band | model | status | n exposures | source contrast rmse | source rmse | support rmse | uniform rmse | source slope | support flux ratio | source flux ratio | support frac | source frac | fit ms |",
            "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['band']} | {row['model']} | {row['status']} | {row.get('n_exposures', 0)} | nan | nan | nan | nan | nan | nan | nan | nan | nan | nan |"
            )
            continue
        lines.append(
            f"| {row['band']} | {row['model']} | OK | {row['n_exposures']} | "
            f"{row['source_contrast_rmse']:.6f} | {row['source_rmse']:.6f} | {row['support_rmse']:.6f} | {row['uniform_rmse']:.6f} | "
            f"{row['source_contrast_slope']:.4f} | {row['support_flux_ratio']:.4f} | {row['source_flux_ratio']:.4f} | "
            f"{row['support_frac']:.4f} | {row['source_frac']:.4f} | "
            f"{row['fit_time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F090W,F277W,F356W,F444W")
    parser.add_argument("--module", default="b")
    parser.add_argument("--input-product", choices=("auto", "crf", "cal"), default="cal")
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crop-strategy", choices=("center", "best_overlap"), default="best_overlap")
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument(
        "--models",
        default="coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--forward-context",
        action="store_true",
        help=(
            "Per-exposure ForwardContext for fit_field_model; wraps INR with "
            "_PerExposureScalarBias (see bench_map_smacs0723_field_fit.py)."
        ),
    )
    args = parser.parse_args()

    bands = _parse_csv(args.bands)
    model_kinds = [part.strip() for part in str(args.models).split(",") if part.strip()]
    rows: list[dict[str, Any]] = []

    band_entries: list[dict[str, Any]] = []
    all_raw_field_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
    all_field_observations: list[FieldObservation] = []
    all_forward_contexts: list[ForwardContext] = []

    with tempfile.TemporaryDirectory(prefix="torchsky_smacs_joint_") as tmp:
        tmpdir = Path(tmp)
        for band in bands:
            reference_path = _reference_path_for_band(band)
            exposure_paths = _discover_exposure_paths(
                DATA / "jwst_smacs0723",
                input_product=args.input_product,
                band=band,
                module=args.module,
                filename_contains="nrcblong" if band in {"F277W", "F356W", "F444W"} else None,
            )
            if not reference_path.exists() or not exposure_paths:
                message = (
                    f"missing reference={reference_path.exists()} "
                    f"n_exposures={len(exposure_paths)}"
                )
                if args.skip_missing:
                    for model_kind in model_kinds:
                        rows.append(
                            {
                                "band": band,
                                "model": model_kind,
                                "status": "MISSING",
                                "message": message,
                                "n_exposures": len(exposure_paths),
                                "forward_context": bool(args.forward_context),
                            }
                        )
                    continue
                raise FileNotFoundError(message)

            ref_data = _read_ext(reference_path, "SCI")
            ref_header = _sci_header(reference_path)
            overlap_headers = [_sci_header(path) for path in exposure_paths]
            if args.crop_strategy == "best_overlap":
                coarse_step = 4
                overlap_count = _approx_overlap_grid(
                    overlap_headers,
                    ref_header,
                    coarse_step=coarse_step,
                )
                coarse_ref = ref_data[::coarse_step, ::coarse_step]
                coarse_size = max(1, int(np.ceil(int(args.crop_size) / coarse_step)))
                _, coarse_x0, coarse_y0 = _best_overlap_cutout(
                    coarse_ref,
                    overlap_count,
                    coarse_size,
                )
                x0 = min(int(coarse_x0 * coarse_step), max(0, ref_data.shape[1] - int(args.crop_size)))
                y0 = min(int(coarse_y0 * coarse_step), max(0, ref_data.shape[0] - int(args.crop_size)))
                ref_crop = ref_data[y0 : y0 + int(args.crop_size), x0 : x0 + int(args.crop_size)]
            else:
                ref_crop, x0, y0 = _center_cutout(ref_data, int(args.crop_size))
            target_header = _cutout_header(ref_header, x0, y0, int(args.crop_size))
            target_observation = _field_target_observation(target_header, band_name=band)

            image_observations: list[ImageObservation] = []
            raw_field_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
            for exposure_path in exposure_paths:
                science_path, weight_path, mask_path = _prepare_aux_files(
                    exposure_path,
                    tmpdir,
                    linearize_sip=False,
                )
                image_observations.append(
                    ImageObservation(
                        str(science_path),
                        hdu=0,
                        weight_path=str(weight_path),
                        weight_hdu=0,
                        mask_path=str(mask_path),
                        mask_hdu=0,
                    )
                )
                science = _read_ext(exposure_path, "SCI")
                err = _read_ext(exposure_path, "ERR")
                dq = _read_ext(exposure_path, "DQ")
                valid = np.isfinite(err) & (err > 0) & np.isfinite(dq) & (dq == 0)
                weight = np.zeros_like(err, dtype=np.float32)
                weight[valid] = 1.0 / np.square(err[valid])
                raw_field_inputs.append(
                    (science, weight, valid.astype(np.float32, copy=False), dict(_sci_header(exposure_path)))
                )
            field_observations: list[FieldObservation] = []
            for exposure_path, (image, weight, mask, header) in zip(
                exposure_paths, raw_field_inputs, strict=True
            ):
                cropped = _crop_field_inputs_to_target(
                    image=image,
                    weight=weight,
                    mask=mask,
                    header=header,
                    target_header=target_header,
                )
                if cropped is None:
                    continue
                crop_image, crop_weight, crop_mask, crop_header = cropped
                obs_index = len(all_field_observations) + len(field_observations)
                field_observations.append(
                    FieldObservation(
                        image=torch.from_numpy(crop_image),
                        header=crop_header,
                        weight=torch.from_numpy(crop_weight),
                        mask=torch.from_numpy(crop_mask),
                        **_field_observation_kwargs_for_band(band),
                    )
                )
                all_forward_contexts.append(
                    ForwardContext(
                        survey=str(band),
                        exposure_id=f"{band}_{exposure_path.stem}",
                        extra={
                            "exposure_index": obs_index,
                            "source_path": str(exposure_path),
                        },
                    )
                )
            if not field_observations:
                if args.skip_missing:
                    for model_kind in model_kinds:
                        rows.append(
                            {
                                "band": band,
                                "model": model_kind,
                                "status": "MISSING",
                                "message": "no overlapping field observations",
                                "n_exposures": len(exposure_paths),
                                "forward_context": bool(args.forward_context),
                            }
                        )
                    continue
                raise RuntimeError(f"no overlapping field observations for {band}")

            coadd_reference = coadd_image_plane(
                image_observations,
                target_header=dict(target_header),
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
                stacking_method="weighted_mean",
            )
            coadd_weight = coadd_reference.weight.cpu().numpy()
            coadd_data = coadd_reference.data.cpu().numpy().astype(np.float32, copy=False)
            support_mask = coadd_weight > 0
            if not np.any(support_mask):
                support_mask = np.isfinite(ref_crop) & np.isfinite(coadd_data)

            band_entries.append(
                {
                    "band": band,
                    "reference_crop": ref_crop.astype(np.float32, copy=False),
                    "target_header": dict(target_header),
                    "target_observation": target_observation,
                    "support_mask": support_mask.astype(bool, copy=False),
                    "coadd_pred": coadd_data,
                    "n_exposures": len(exposure_paths),
                }
            )
            all_raw_field_inputs.extend(raw_field_inputs)
            all_field_observations.extend(field_observations)

        if not band_entries:
            if args.skip_missing:
                payload = {
                    "rows": rows,
                    "summary": [],
                    "status": "skipped",
                    "reason": "no bands available (missing reference or exposures)",
                }
                text = json.dumps(payload, indent=2)
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(text, encoding="utf-8")
                args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
                args.markdown_out.write_text(_format_markdown(rows), encoding="utf-8")
                print(text)
                return 0
            raise RuntimeError("no bands available for joint fit")

        intensity_scale = _robust_intensity_scale(all_raw_field_inputs)
        normalized_observations: list[FieldObservation] = []
        for observation in all_field_observations:
            normalized_observations.append(
                FieldObservation(
                    image=observation.image / intensity_scale,
                    header=dict(observation.header),
                    weight=(
                        observation.weight * (intensity_scale * intensity_scale)
                        if observation.weight is not None
                        else None
                    ),
                    mask=observation.mask.clone() if observation.mask is not None else None,
                    wavelength=observation.wavelength,
                    spectral_min=observation.spectral_min,
                    spectral_max=observation.spectral_max,
                    transmission_center=observation.transmission_center,
                    transmission_sigma=observation.transmission_sigma,
                    psf_sigma_pix=observation.psf_sigma_pix,
                    band_name=observation.band_name,
                )
            )

        domain = estimate_field_domain(normalized_observations, coord_mode="sphere_xyz")
        forward_ctx_tuple: tuple[ForwardContext | None, ...] | None = None
        if args.forward_context:
            if len(all_forward_contexts) != len(normalized_observations):
                raise RuntimeError("internal: joint forward context list length mismatch")
            forward_ctx_tuple = tuple(all_forward_contexts)

        fitted_models: dict[str, tuple[Any, float, dict[str, Any]]] = {}
        for model_kind in model_kinds:
            if model_kind == "coadd_bilinear":
                continue
            model_kwargs = _field_model_kwargs_for_case(
                model_kind,
                case=REAL_JOINT_CASE,
                size=int(args.crop_size),
            )
            fit_kwargs = _joint_real_fit_kwargs(
                model_kind,
                crop_size=int(args.crop_size),
                steps=int(args.steps),
                num_observations=len(normalized_observations),
            )
            if forward_ctx_tuple is not None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["observation_forward_contexts"] = forward_ctx_tuple
            model = make_field_model(model_kind, domain, **model_kwargs)
            if args.forward_context:
                model = _PerExposureScalarBias(model, len(normalized_observations))
            t0 = time.perf_counter()
            result = fit_field_model(
                normalized_observations,
                model,
                steps=int(args.steps),
                **fit_kwargs,
            )
            fit_time_ms = (time.perf_counter() - t0) * 1000.0
            fitted_models[model_kind] = (result.model, fit_time_ms, fit_kwargs)

        for entry in band_entries:
            band = str(entry["band"])
            ref_crop = entry["reference_crop"]
            target_header = entry["target_header"]
            support_mask = entry["support_mask"]
            target_observation = entry["target_observation"]
            for model_kind in model_kinds:
                row = {
                    "band": band,
                    "model": model_kind,
                    "n_exposures": int(entry["n_exposures"]),
                    "forward_context": bool(args.forward_context),
                    "status": "OK",
                }
                try:
                    if model_kind == "coadd_bilinear":
                        pred = entry["coadd_pred"]
                        fit_time_ms = 0.0
                    else:
                        result_model, fit_time_ms, fit_kwargs = fitted_models[model_kind]
                        render_num_spectral_samples = int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        )
                        render_ctx = (
                            ForwardContext(survey=str(band))
                            if args.forward_context
                            else None
                        )
                        pred = (
                            render_field_observation(
                                result_model,
                                target_observation,
                                num_spectral_samples=render_num_spectral_samples,
                                forward_context=render_ctx,
                            )
                            .detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        ) * intensity_scale
                    support_metric_mask = _metric_mask(pred, ref_crop, support_mask)
                    uniform_metric_mask = np.isfinite(pred) & np.isfinite(ref_crop)
                    if not np.any(uniform_metric_mask):
                        raise RuntimeError("render produced no finite comparison pixels")
                    support_diff = pred[support_metric_mask] - ref_crop[support_metric_mask]
                    uniform_diff = pred[uniform_metric_mask] - ref_crop[uniform_metric_mask]
                    source_stats = _background_subtracted_source_stats(
                        pred,
                        ref_crop,
                        support_mask,
                    )

                    # Robust affine fit to ignore background offsets
                    source_metric_mask = _source_metric_mask(pred, ref_crop, support_mask)
                    ref_pts = ref_crop[source_metric_mask].astype(np.float64)
                    pred_pts = pred[source_metric_mask].astype(np.float64)
                    design = np.stack([ref_pts, np.ones_like(ref_pts)], axis=1)
                    coeffs, *_ = np.linalg.lstsq(design, pred_pts, rcond=None)
                    affine_slope = float(coeffs[0])
                    affine_intercept = float(coeffs[1])
                    aligned_pred = (pred_pts - affine_intercept) / max(affine_slope, 1e-6) if affine_slope > 0.5 else pred_pts
                    aligned_diff = aligned_pred - ref_pts

                    row.update(
                        {
                            "source_rmse": source_stats["source_rmse"],
                            "source_contrast_rmse": source_stats["source_contrast_rmse"],
                            "source_contrast_slope": source_stats["source_contrast_slope"],
                            "support_rmse": float(np.sqrt(np.mean(support_diff * support_diff)))
                            if support_diff.size
                            else float("nan"),
                            "uniform_rmse": float(np.sqrt(np.mean(uniform_diff * uniform_diff)))
                            if uniform_diff.size
                            else float("nan"),
                            "aligned_rmse": float(np.sqrt(np.mean(aligned_diff * aligned_diff)))
                            if aligned_diff.size
                            else float("nan"),
                            "affine_slope": affine_slope,
                            "support_flux_ratio": (
                                float(np.sum(pred_pts) / np.sum(ref_pts))
                                if ref_pts.size and np.sum(ref_pts) != 0.0
                                else float("nan")
                            ),
                            "source_flux_ratio": source_stats["source_flux_ratio"],
                            "support_frac": float(support_metric_mask.mean()),
                            "source_frac": source_stats["source_frac"],
                            "fit_time_ms": float(fit_time_ms),
                        }
                    )
                    _write_export_products(
                        args.export_dir,
                        band=band,
                        model=model_kind,
                        header=target_header,
                        reference=ref_crop,
                        prediction=pred,
                        support_mask=support_mask,
                        source_mask=source_metric_mask,
                    )
                except Exception as exc:
                    row["status"] = "FAIL"
                    row["message"] = str(exc)
                rows.append(row)

    ok_rows = [row for row in rows if row["status"] == "OK"]
    fail_rows = [row for row in rows if row["status"] != "OK"]
    ok_rows.sort(key=lambda row: (str(row["model"]), float(row["support_rmse"]), str(row["band"])))
    fail_rows.sort(key=lambda row: (str(row["model"]), str(row["band"])))
    payload = {"rows": ok_rows + fail_rows, "summary": _summarize_by_model(ok_rows)}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_format_markdown(payload["rows"]), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
