#!/usr/bin/env python3
"""Real-data field-fit smoke on JWST SMACS J0723 F277W exposures."""

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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
DATA = ROOT / "data"
DEFAULT_REFERENCE_PATH = DATA / "updec_webb" / "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
DEFAULT_EXPOSURE_DIR = DATA / "jwst_smacs0723"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_field_fit.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_field_fit.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_field_fit import _field_fit_kwargs_for_model, _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_real import (  # noqa: E402
    _best_overlap_cutout_from_observations,
    _field_observation_kwargs_for_band,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _linearize_sip_header,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _sci_header,
)
import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_observation,
)


REAL_CASE = "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit"


def _time_many(fn, runs: int) -> float:
    if runs <= 1:
        t0 = time.perf_counter()
        fn()
        return float(time.perf_counter() - t0)
    fn()
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _fit_with_timing(
    observations: list[FieldObservation],
    model: torch.nn.Module,
    *,
    steps: int,
    runs: int,
    fit_kwargs: dict[str, Any],
) -> tuple[Any, float]:
    if runs <= 1:
        t0 = time.perf_counter()
        result = fit_field_model(
            observations,
            model,
            steps=steps,
            **fit_kwargs,
        )
        return result, float((time.perf_counter() - t0) * 1000.0)
    fit_time_ms = (
        _time_many(
            lambda: fit_field_model(
                observations,
                model,
                steps=steps,
                **fit_kwargs,
            ),
            runs,
        )
        * 1000.0
    )
    result = fit_field_model(
        observations,
        model,
        steps=steps,
        **fit_kwargs,
    )
    return result, fit_time_ms


def _field_target_observation(
    header: dict[str, Any],
    *,
    band_name: str,
    observation_mode: str = "bandpass",
) -> FieldObservation:
    image = torch.zeros((int(header["NAXIS2"]), int(header["NAXIS1"])), dtype=torch.float32)
    return FieldObservation(
        image=image,
        header=dict(header),
        **_field_observation_kwargs_for_band(band_name, observation_mode=observation_mode),
    )


def _copy_header_with_crop(
    header: dict[str, Any],
    *,
    x0: int,
    y0: int,
    width: int,
    height: int,
) -> dict[str, Any]:
    cropped = dict(header)
    cropped["NAXIS1"] = int(width)
    cropped["NAXIS2"] = int(height)
    if "CRPIX1" in cropped:
        cropped["CRPIX1"] = float(cropped["CRPIX1"]) - float(x0)
    if "CRPIX2" in cropped:
        cropped["CRPIX2"] = float(cropped["CRPIX2"]) - float(y0)
    return cropped


def _crop_field_inputs_to_target(
    *,
    image: np.ndarray,
    weight: np.ndarray,
    mask: np.ndarray,
    header: dict[str, Any],
    target_header: dict[str, Any],
    margin: int = 32,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]] | None:
    target_wcs = torchsky.WCS(target_header)
    source_wcs = torchsky.WCS(header)
    height = int(target_header["NAXIS2"])
    width = int(target_header["NAXIS1"])
    tx = torch.tensor(
        [0.0, float(width - 1), float(width - 1), 0.0, 0.5 * float(width - 1)],
        dtype=torch.float32,
    )
    ty = torch.tensor(
        [0.0, 0.0, float(height - 1), float(height - 1), 0.5 * float(height - 1)],
        dtype=torch.float32,
    )
    ra, dec = target_wcs.pixel_to_world(tx, ty)
    sx, sy = source_wcs.world_to_pixel(ra, dec)
    sx_np = sx.detach().cpu().numpy()
    sy_np = sy.detach().cpu().numpy()
    valid = np.isfinite(sx_np) & np.isfinite(sy_np)
    if not np.any(valid):
        return None
    x0 = max(int(np.floor(np.min(sx_np[valid]))) - int(margin), 0)
    y0 = max(int(np.floor(np.min(sy_np[valid]))) - int(margin), 0)
    x1 = min(int(np.ceil(np.max(sx_np[valid]))) + int(margin) + 1, image.shape[-1])
    y1 = min(int(np.ceil(np.max(sy_np[valid]))) + int(margin) + 1, image.shape[-2])
    if x1 <= x0 or y1 <= y0:
        return None
    return (
        image[y0:y1, x0:x1].copy(),
        weight[y0:y1, x0:x1].copy(),
        mask[y0:y1, x0:x1].copy(),
        _copy_header_with_crop(header, x0=x0, y0=y0, width=x1 - x0, height=y1 - y0),
    )


def _uniform_mask(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.isfinite(pred) & np.isfinite(reference)


def _robust_intensity_scale(
    field_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
) -> float:
    weighted_power = 0.0
    weighted_count = 0.0
    for image, weight, mask, _header in field_inputs:
        valid = np.isfinite(image) & np.isfinite(weight) & (weight > 0) & (mask > 0)
        if not np.any(valid):
            continue
        image_valid = image[valid].astype(np.float64, copy=False)
        weight_valid = weight[valid].astype(np.float64, copy=False)
        weighted_power += float(np.sum(weight_valid * np.square(image_valid)))
        weighted_count += float(np.sum(weight_valid))
    if weighted_count <= 0.0:
        return 1.0
    scale = float(np.sqrt(weighted_power / weighted_count))
    if not np.isfinite(scale) or scale <= 0.0:
        return 1.0
    return scale


def _scaled_field_observation(
    observation: FieldObservation,
    *,
    intensity_scale: float,
) -> FieldObservation:
    scale = float(intensity_scale)
    return FieldObservation(
        image=observation.image / scale,
        header=dict(observation.header),
        weight=(
            observation.weight * (scale * scale)
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


def _real_fit_kwargs(
    model_kind: str,
    *,
    crop_size: int,
    steps: int,
    observation_mode: str,
    num_observations: int,
) -> dict[str, Any]:
    kwargs = _field_fit_kwargs_for_model(
        model_kind,
        case=REAL_CASE,
        size=int(crop_size),
        benchmark_steps=int(steps),
    )
    mode = str(observation_mode).lower()
    if mode == "exact_wavelength":
        kwargs["num_spectral_samples"] = 1
        kwargs["training_spectral_sampling"] = "deterministic_quadrature"
        kwargs["spectral_anchor_observations_per_step"] = 0
        kwargs["spectral_anchor_loss_weight"] = 0.0
        kwargs["training_patch_size"] = None
        kwargs["training_patches_per_observation"] = 1
        kwargs["pixel_samples_per_observation"] = 512
        kwargs["observations_per_step"] = 1
        kwargs["observation_sampling"] = "uniform"
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
        kwargs["grad_clip_norm"] = 1.0
        if model_kind == "spectral_basis_field":
            kwargs["lr"] = 3.0e-3
        elif model_kind == "sky_encoder_resmlp":
            kwargs["lr"] = 2.0e-3
        elif model_kind == "hashgrid_mlp":
            kwargs["lr"] = 2.0e-3
    return kwargs


def _summarize_row_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Real Field-Fit Smoke",
        "",
        "| model | support rmse | uniform rmse | support flux ratio | support frac | crop mode | fit ms |",
        "| --- | ---: | ---: | ---: | ---: | --- | ---: |",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['support_rmse']:.6f} | {row['uniform_rmse']:.6f} | "
            f"{row['support_flux_ratio']:.4f} | {row['support_frac']:.4f} | "
            f"{row['crop_mode']} | {row['fit_time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--exposures", default="")
    parser.add_argument("--crop-size", type=int, default=256)
    parser.add_argument("--crop-mode", choices=("center", "best_overlap"), default="best_overlap")
    parser.add_argument(
        "--overlap-step",
        type=int,
        default=8,
        help="Coarse pixel step used by footprint-based best-overlap crop selection",
    )
    parser.add_argument(
        "--linearize-target-sip",
        action="store_true",
        help="Strip SIP terms from the target crop header for field rendering only",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal"),
        default="auto",
        help="Which local exposure product family to use when --exposures is not given",
    )
    parser.add_argument(
        "--filename-contains",
        default="",
        help="Optional case-insensitive filename substring filter applied to auto-discovered exposures",
    )
    parser.add_argument("--band-name", default="F277W")
    parser.add_argument(
        "--observation-mode",
        choices=("bandpass", "exact_wavelength"),
        default="bandpass",
        help="Treat each exposure as a finite bandpass observation or as an exact-wavelength 2D control.",
    )
    parser.add_argument("--wavelength-angstrom", type=float, default=27700.0)
    parser.add_argument(
        "--models",
        default="coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field",
    )
    parser.add_argument(
        "--linearize-sip",
        action="store_true",
        help="Strip SIP terms from the calibrated exposure headers as a fallback",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    model_kinds = [part.strip() for part in args.models.split(",") if part.strip()]
    if args.exposures.strip():
        exposure_paths = [Path(part.strip()) for part in args.exposures.split(",") if part.strip()]
    else:
        exposure_paths = _discover_exposure_paths(
            DEFAULT_EXPOSURE_DIR,
            input_product=args.input_product,
            filename_contains=args.filename_contains or None,
        )
    if not exposure_paths:
        raise FileNotFoundError(
            "No calibrated exposures found. Pass --exposures or populate "
            f"{DEFAULT_EXPOSURE_DIR}"
        )
    ref_data = _read_ext(args.reference_path, "SCI")
    ref_header = _sci_header(args.reference_path)
    tmp_root = DATA / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix="torchsky_smacs0723_field_fit_", dir=str(tmp_root)) as tmp:
        tmpdir = Path(tmp)
        image_observations: list[ImageObservation] = []
        raw_field_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]] = []
        for exposure_path in exposure_paths:
            science_path, weight_path, mask_path = _prepare_aux_files(
                exposure_path,
                tmpdir,
                linearize_sip=args.linearize_sip,
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
                (
                    science,
                    weight,
                    valid.astype(np.float32),
                    dict(_sci_header(exposure_path)),
                )
            )
        if args.crop_mode == "best_overlap":
            ref_crop, x0, y0 = _best_overlap_cutout_from_observations(
                ref_data,
                image_observations,
                ref_header,
                size=args.crop_size,
                coarse_step=args.overlap_step,
                resampling_kernel="bilinear",
                value_semantics="flux",
                stacking_method="weighted_mean",
            )
        else:
            ref_crop, x0, y0 = _center_cutout(ref_data, args.crop_size)
        target_header = _cutout_header(ref_header, x0, y0, args.crop_size)
        field_target_header = _linearize_sip_header(target_header) if args.linearize_target_sip else dict(target_header)

        field_observations: list[FieldObservation] = []
        for image, weight, mask, header in raw_field_inputs:
            cropped = _crop_field_inputs_to_target(
                image=image,
                weight=weight,
                mask=mask,
                header=header,
                target_header=field_target_header,
            )
            if cropped is None:
                continue
            crop_image, crop_weight, crop_mask, crop_header = cropped
            field_observations.append(
                FieldObservation(
                    image=torch.from_numpy(crop_image),
                    header=crop_header,
                    weight=torch.from_numpy(crop_weight),
                    mask=torch.from_numpy(crop_mask),
                    **_field_observation_kwargs_for_band(
                        args.band_name,
                        observation_mode=args.observation_mode,
                    ),
                )
            )

        coadd_reference = coadd_image_plane(
            image_observations,
            target_header=dict(target_header),
            resampling_kernel="bilinear",
            value_semantics="flux",
            stacking_method="weighted_mean",
        )
        support_mask = coadd_reference.weight.cpu().numpy() > 0
        target_observation = _field_target_observation(
            field_target_header,
            band_name=args.band_name,
            observation_mode=args.observation_mode,
        )
        if not field_observations:
            raise RuntimeError("No field observations overlap the requested target crop")
        domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
        intensity_scale = _robust_intensity_scale(raw_field_inputs)

        normalized_field_observations: list[FieldObservation] = []
        for observation in field_observations:
            normalized_field_observations.append(
                _scaled_field_observation(
                    observation,
                    intensity_scale=intensity_scale,
                )
            )

        rows: list[dict[str, Any]] = []
        for model_kind in model_kinds:
            if model_kind == "coadd_bilinear":
                pred = coadd_reference.data.cpu().numpy().astype(np.float32)
                fit_time_ms = 0.0
            else:
                model_kwargs = _field_model_kwargs_for_case(
                    model_kind,
                    case=REAL_CASE,
                    size=args.crop_size,
                )
                fit_kwargs = _real_fit_kwargs(
                    model_kind,
                    crop_size=int(args.crop_size),
                    steps=int(args.steps),
                    observation_mode=args.observation_mode,
                    num_observations=len(normalized_field_observations),
                )
                model = make_field_model(model_kind, domain, **model_kwargs)
                result, fit_time_ms = _fit_with_timing(
                    normalized_field_observations,
                    model,
                    steps=int(args.steps),
                    runs=int(args.runs),
                    fit_kwargs=fit_kwargs,
                )
                render_num_spectral_samples = int(
                    fit_kwargs.get("num_spectral_samples", 1)
                )
                pred = (
                    render_field_observation(
                        result.model,
                        target_observation,
                        num_spectral_samples=render_num_spectral_samples,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pred = pred * intensity_scale

            support_metric_mask = _metric_mask(pred, ref_crop, support_mask)
            uniform_metric_mask = _uniform_mask(pred, ref_crop)
            support_diff = pred[support_metric_mask] - ref_crop[support_metric_mask]
            uniform_diff = pred[uniform_metric_mask] - ref_crop[uniform_metric_mask]
            rows.append(
                {
                    "model": model_kind,
                    "n_exposures": len(exposure_paths),
                    "crop_size": args.crop_size,
                    "crop_mode": args.crop_mode,
                    "crop_x0": int(x0),
                    "crop_y0": int(y0),
                    "support_rmse": float(np.sqrt(np.mean(support_diff * support_diff)))
                    if support_diff.size
                    else float("nan"),
                    "uniform_rmse": float(np.sqrt(np.mean(uniform_diff * uniform_diff)))
                    if uniform_diff.size
                    else float("nan"),
                    "support_flux_ratio": (
                        float(np.sum(pred[support_metric_mask]) / np.sum(ref_crop[support_metric_mask]))
                        if support_diff.size
                        else float("nan")
                    ),
                    "support_frac": float(support_metric_mask.mean()),
                    "fit_time_ms": float(fit_time_ms),
                }
            )

    payload = {"rows": rows}
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_summarize_row_markdown(rows))
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
