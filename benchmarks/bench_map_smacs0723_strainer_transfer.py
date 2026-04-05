#!/usr/bin/env python3
"""Real-data Strainer trunk transfer smoke (JWST SMACS J0723).

Compares ``strainer_sky_encoder_mlp`` with random init vs multi-noise synthetic pretrain
plus ``reset_decoder_head()`` on the same cropped exposures as
``bench_map_smacs0723_field_fit.py``. Reports training losses and support RMSE vs the
drizzle reference crop.

If reference FITS or exposures are missing, prints a JSON ``skipped`` payload and exits 0
(so CI and clone checkouts without ``data/`` do not fail).
"""

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
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_field_fit import _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    REAL_CASE,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _real_fit_kwargs,
    _robust_intensity_scale,
    _scaled_field_observation,
)
from bench_map_smacs0723_real import (  # noqa: E402
    _best_overlap_cutout_from_observations,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _field_observation_kwargs_for_band,
    _linearize_sip_header,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _sci_header,
)
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    pretrain_strainer_trunk_synthetic,
    render_field_observation,
)


def _skip_payload(reason: str) -> dict[str, Any]:
    return {"suite": "smacs0723_strainer_transfer", "status": "skipped", "reason": reason}


def _fit_ms(fn: Any) -> tuple[Any, float]:
    t0 = time.perf_counter()
    out = fn()
    return out, float((time.perf_counter() - t0) * 1000.0)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=DEFAULT_REFERENCE_PATH)
    parser.add_argument("--exposures", default="")
    parser.add_argument("--crop-size", type=int, default=128)
    parser.add_argument("--crop-mode", choices=("center", "best_overlap"), default="best_overlap")
    parser.add_argument("--overlap-step", type=int, default=8)
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal"),
        default="auto",
    )
    parser.add_argument("--filename-contains", default="")
    parser.add_argument("--band-name", default="F277W")
    parser.add_argument(
        "--observation-mode",
        choices=("bandpass", "exact_wavelength"),
        default="bandpass",
    )
    parser.add_argument("--steps", type=int, default=12)
    parser.add_argument("--lr", type=float, default=2.5e-3)
    parser.add_argument("--synthetic-pretrain-steps", type=int, default=80)
    parser.add_argument("--pretrain-batch-size", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument(
        "--linearize-sip",
        action="store_true",
        help="Strip SIP terms from calibrated exposure headers (passed to aux prep)",
    )
    parser.add_argument(
        "--linearize-target-sip",
        action="store_true",
        help="Strip SIP from target header used for field rendering",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    if not args.reference_path.is_file():
        payload = _skip_payload(f"missing reference: {args.reference_path}")
        print(json.dumps(payload, indent=2))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2))
        return 0

    if args.exposures.strip():
        exposure_paths = [Path(p.strip()) for p in args.exposures.split(",") if p.strip()]
    else:
        exposure_paths = _discover_exposure_paths(
            DEFAULT_EXPOSURE_DIR,
            input_product=args.input_product,
            filename_contains=args.filename_contains or None,
        )
    if not exposure_paths:
        payload = _skip_payload(
            f"no exposures under {DEFAULT_EXPOSURE_DIR} (pass --exposures or add data)"
        )
        print(json.dumps(payload, indent=2))
        if args.json_out is not None:
            args.json_out.parent.mkdir(parents=True, exist_ok=True)
            args.json_out.write_text(json.dumps(payload, indent=2))
        return 0

    ref_data = _read_ext(args.reference_path, "SCI")
    ref_header = _sci_header(args.reference_path)
    tmp_root = DATA / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)

    with tempfile.TemporaryDirectory(
        prefix="torchsky_smacs0723_strainer_transfer_", dir=str(tmp_root)
    ) as tmp:
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
                (science, weight, valid.astype(np.float32), dict(_sci_header(exposure_path)))
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
        field_target_header = (
            _linearize_sip_header(target_header) if args.linearize_target_sip else dict(target_header)
        )

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
            payload = _skip_payload("no field observations overlap the target crop")
            print(json.dumps(payload, indent=2))
            if args.json_out is not None:
                args.json_out.parent.mkdir(parents=True, exist_ok=True)
                args.json_out.write_text(json.dumps(payload, indent=2))
            return 0

        domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
        intensity_scale = _robust_intensity_scale(raw_field_inputs)
        normalized_field_observations = [
            _scaled_field_observation(o, intensity_scale=intensity_scale)
            for o in field_observations
        ]

        fit_kwargs = _real_fit_kwargs(
            "strainer_sky_encoder_mlp",
            crop_size=int(args.crop_size),
            steps=int(args.steps),
            observation_mode=args.observation_mode,
            num_observations=len(normalized_field_observations),
        )
        fit_kwargs["lr"] = float(args.lr)
        model_kw = _field_model_kwargs_for_case(
            "strainer_sky_encoder_mlp",
            case=REAL_CASE,
            size=int(args.crop_size),
        )
        render_num_spectral_samples = int(fit_kwargs.get("num_spectral_samples", 1))

        rows: list[dict[str, Any]] = []

        def run_one(
            *,
            name: str,
            synthetic_pretrain_steps: int,
            model_seed: int,
        ) -> dict[str, Any]:
            torch.manual_seed(int(model_seed))
            model = make_field_model("strainer_sky_encoder_mlp", domain, **model_kw)
            if synthetic_pretrain_steps > 0:
                pretrain_strainer_trunk_synthetic(
                    model,
                    steps=int(synthetic_pretrain_steps),
                    lr=float(args.lr),
                    noise_kinds=("gaussian", "uniform", "spectrum"),
                    batch_size=int(args.pretrain_batch_size),
                    seed=int(model_seed) + 11,
                )
                model.reset_decoder_head()

            def do_fit():
                return fit_field_model(
                    normalized_field_observations,
                    model,
                    steps=int(args.steps),
                    **fit_kwargs,
                )

            result, fit_time_ms = _fit_ms(do_fit)
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
            support_diff = pred[support_metric_mask] - ref_crop[support_metric_mask]
            support_rmse = (
                float(np.sqrt(np.mean(support_diff * support_diff)))
                if support_diff.size
                else float("nan")
            )
            return {
                "name": name,
                "synthetic_pretrain_steps": int(synthetic_pretrain_steps),
                "initial_training_loss": float(result.initial_loss),
                "final_training_loss": float(result.final_loss),
                "training_loss_after_step_1": float(result.training_loss_after_step(1)),
                "support_rmse": support_rmse,
                "fit_time_ms": fit_time_ms,
                "model_seed": int(model_seed),
            }

        base = int(args.seed)
        rows.append(
            run_one(
                name="strainer_baseline",
                synthetic_pretrain_steps=0,
                model_seed=base + 10_000,
            )
        )
        rows.append(
            run_one(
                name="strainer_pretrain_reset",
                synthetic_pretrain_steps=int(args.synthetic_pretrain_steps),
                model_seed=base + 10_000 + 997,
            )
        )

    payload = {
        "suite": "smacs0723_strainer_transfer",
        "status": "ok",
        "reference_path": str(args.reference_path),
        "n_exposures": len(exposure_paths),
        "crop_size": int(args.crop_size),
        "crop_mode": args.crop_mode,
        "crop_x0": int(x0),
        "crop_y0": int(y0),
        "fit_steps": int(args.steps),
        "rows": rows,
    }
    print(json.dumps(payload, indent=2))
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
