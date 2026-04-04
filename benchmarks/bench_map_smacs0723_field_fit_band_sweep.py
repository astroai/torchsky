#!/usr/bin/env python3
"""Per-band real-data field-fit sweep on JWST SMACS J0723 module-B products."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
DATA = ROOT / "data"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_band_sweep.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_field_fit_band_sweep.md"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_smacs0723_field_fit import (  # noqa: E402
    REAL_CASE,
    _crop_field_inputs_to_target,
    _fit_with_timing,
    _field_target_observation,
    _real_fit_kwargs,
    _robust_intensity_scale,
    _scaled_field_observation,
)
from bench_map_field_fit import _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_real import (  # noqa: E402
    _approx_overlap_grid,
    _best_overlap_cutout,
    _field_observation_kwargs_for_band,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _reference_path_for_band,
    _sci_header,
    _band_spec,
)
import numpy as np  # noqa: E402
import torch  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    make_field_model,
    render_field_observation,
)


def _parse_csv(value: str) -> list[str]:
    return [part.strip() for part in str(value).split(",") if part.strip()]


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Real Field-Fit Band Sweep",
        "",
        "| band | model | status | support rmse | aligned rmse | slope | support frac | fit ms |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['band']} | {row['model']} | {row['status']} | nan | nan | nan | nan | nan |"
            )
            continue
        lines.append(
            f"| {row['band']} | {row['model']} | OK | {row['support_rmse']:.6f} | "
            f"{row['aligned_rmse']:.6f} | {row['affine_slope']:.4f} | "
            f"{row['support_frac']:.4f} | {row['fit_time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bands", default="F090W,F150W,F200W,F277W,F356W,F444W")
    parser.add_argument("--module", default="b")
    parser.add_argument("--input-product", choices=("auto", "crf", "cal"), default="cal")
    parser.add_argument(
        "--observation-mode",
        choices=("bandpass", "exact_wavelength"),
        default="bandpass",
        help="Treat each exposure as a finite bandpass observation or as an exact-wavelength 2D control.",
    )
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument("--crop-strategy", choices=("center", "best_overlap"), default="best_overlap")
    parser.add_argument("--steps", type=int, default=2)
    parser.add_argument(
        "--models",
        default="coadd_bilinear,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    args = parser.parse_args()

    bands = [band.upper() for band in _parse_csv(args.bands)]
    model_kinds = _parse_csv(args.models)
    rows: list[dict[str, Any]] = []

    for band in bands:
        reference_path = _reference_path_for_band(band)
        _band_spec(band)
        try:
            exposure_paths = _discover_exposure_paths(
                DATA / "jwst_smacs0723",
                input_product=args.input_product,
                band=band,
                module=args.module,
            )
            if not exposure_paths:
                raise FileNotFoundError("no local exposures for requested band/module")
            if not reference_path.exists():
                raise FileNotFoundError(f"missing reference product {reference_path.name}")
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

            import tempfile
            tmp_root = DATA / "_tmp"
            tmp_root.mkdir(parents=True, exist_ok=True)
            with tempfile.TemporaryDirectory(prefix=f"torchsky_smacs_{band.lower()}_", dir=str(tmp_root)) as tmp:
                tmpdir = Path(tmp)
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
                        (science, weight, valid.astype(np.float32), dict(_sci_header(exposure_path)))
                    )

                field_observations: list[FieldObservation] = []
                for image, weight, mask, header in raw_field_inputs:
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
                    field_observations.append(
                        FieldObservation(
                            image=torch.from_numpy(crop_image),
                            header=crop_header,
                            weight=torch.from_numpy(crop_weight),
                            mask=torch.from_numpy(crop_mask),
                            **_field_observation_kwargs_for_band(
                                band,
                                observation_mode=args.observation_mode,
                            ),
                        )
                    )
                if not field_observations:
                    raise RuntimeError("no overlapping field observations")

                coadd_reference = coadd_image_plane(
                    image_observations,
                    target_header=dict(target_header),
                    resampling_kernel="bilinear",
                    value_semantics="flux",
                    stacking_method="weighted_mean",
                )
                coadd_weight = coadd_reference.weight.cpu().numpy()
                coadd_data = coadd_reference.data.cpu().numpy()
                support_mask = coadd_weight > 0
                if not np.any(support_mask):
                    support_mask = np.isfinite(ref_crop) & np.isfinite(coadd_data)
                target_observation = _field_target_observation(
                    target_header,
                    band_name=band,
                    observation_mode=args.observation_mode,
                )
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

                for model_kind in model_kinds:
                    row = {
                        "band": band,
                        "model": model_kind,
                        "n_exposures": len(exposure_paths),
                        "status": "OK",
                    }
                    try:
                        if model_kind == "coadd_bilinear":
                            pred = coadd_reference.data.cpu().numpy().astype(np.float32)
                            fit_time_ms = 0.0
                        else:
                            model_kwargs = _field_model_kwargs_for_case(
                                model_kind,
                                case=REAL_CASE,
                                size=int(args.crop_size),
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
                                runs=1,
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
                            ) * intensity_scale
                        support_metric_mask = _metric_mask(pred, ref_crop, support_mask)
                        uniform_metric_mask = np.isfinite(pred) & np.isfinite(ref_crop)
                        support_diff = pred[support_metric_mask] - ref_crop[support_metric_mask]
                        uniform_diff = pred[uniform_metric_mask] - ref_crop[uniform_metric_mask]

                        # Robust affine fit to ignore background offsets
                        ref_pts = ref_crop[support_metric_mask].astype(np.float64)
                        pred_pts = pred[support_metric_mask].astype(np.float64)
                        design = np.stack([ref_pts, np.ones_like(ref_pts)], axis=1)
                        coeffs, *_ = np.linalg.lstsq(design, pred_pts, rcond=None)
                        affine_slope = float(coeffs[0])
                        affine_intercept = float(coeffs[1])
                        aligned_pred = (pred_pts - affine_intercept) / max(affine_slope, 1e-6) if affine_slope > 0.5 else pred_pts
                        aligned_diff = aligned_pred - ref_pts

                        row.update(
                            {
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
                                    if support_diff.size and np.sum(ref_pts) != 0.0
                                    else float("nan")
                                ),
                                "support_frac": float(support_metric_mask.mean()),
                                "fit_time_ms": float(fit_time_ms),
                            }
                        )
                    except Exception as exc:
                        row["status"] = "FAIL"
                        row["message"] = str(exc)
                    rows.append(row)
        except Exception as exc:
            for model_kind in model_kinds:
                rows.append(
                    {
                        "band": band,
                        "model": model_kind,
                        "status": "FAIL",
                        "message": str(exc),
                    }
                )

    ok_rows = [row for row in rows if row["status"] == "OK"]
    fail_rows = [row for row in rows if row["status"] != "OK"]
    ok_rows.sort(key=lambda row: (row["band"], float(row["support_rmse"])))
    fail_rows.sort(key=lambda row: (row["band"], row["model"]))
    payload = {"rows": ok_rows + fail_rows}
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_format_markdown(payload["rows"]), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
