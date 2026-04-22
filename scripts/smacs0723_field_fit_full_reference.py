#!/usr/bin/env python3
"""JWST SMACS J0723 field fit on the full UPdec reference grid (or a capped crop).

The usual ``bench_map_smacs0723_field_fit.py`` uses a small ``--crop-size`` cutout. This
script uses the full reference ``SCI`` array (``NAXIS1`` × ``NAXIS2``) as the map target,
so memory and runtime scale with mosaic area. Use ``--max-side`` for a centered debug
crop without changing the rest of the pipeline.

Examples::

  PYTHONPATH=src:benchmarks uv run python scripts/smacs0723_field_fit_full_reference.py \\
    --band F277W --max-side 256 --steps 4 --models spectral_basis_field

  PYTHONPATH=src:benchmarks uv run python scripts/smacs0723_field_fit_full_reference.py \\
    --band F277W --steps 12 --forward-context
"""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
DATA = ROOT / "data"
for p in (SRC, BENCH):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

from bench_map_field_fit import _field_model_kwargs_for_case  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    REAL_CASE,
    _PerExposureScalarBias,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _fit_with_timing,
    _real_fit_kwargs,
    _robust_intensity_scale,
    _scaled_field_observation,
)
from bench_map_smacs0723_real import (  # noqa: E402
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
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ForwardContext,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    make_field_model,
    render_field_observation,
)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--reference-path", type=Path, default=None)
    parser.add_argument("--exposure-dir", type=Path, default=DATA / "jwst_smacs0723")
    parser.add_argument(
        "--input-product", choices=("auto", "crf", "cal"), default="auto"
    )
    parser.add_argument("--filename-contains", default="")
    parser.add_argument(
        "--max-side",
        type=int,
        default=None,
        help="Center-crop reference to at most this size per axis (omit for full mosaic).",
    )
    parser.add_argument("--steps", type=int, default=8)
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument(
        "--models",
        default="spectral_basis_field",
        help="Comma-separated INR model kinds.",
    )
    parser.add_argument("--forward-context", action="store_true")
    parser.add_argument(
        "--observation-mode",
        choices=("bandpass", "exact_wavelength"),
        default="bandpass",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    args = parser.parse_args()

    band = str(args.band).upper()
    ref_path = args.reference_path or _reference_path_for_band(band)
    if not ref_path.exists():
        print(f"Missing reference: {ref_path}", file=sys.stderr)
        return 1

    exposure_paths = _discover_exposure_paths(
        args.exposure_dir,
        input_product=args.input_product,
        band=band,
        filename_contains=args.filename_contains or None,
    )
    if not exposure_paths:
        print(f"No exposures under {args.exposure_dir}", file=sys.stderr)
        return 1

    ref_data = _read_ext(ref_path, "SCI")
    ref_header = _sci_header(ref_path)
    h, w = int(ref_data.shape[0]), int(ref_data.shape[1])
    x0, y0 = 0, 0
    if args.max_side is not None:
        side = min(int(args.max_side), h, w)
        ref_work, x0, y0 = _center_cutout(ref_data, side)
        target_header = dict(_cutout_header(ref_header, x0, y0, side))
    else:
        ref_work = ref_data
        target_header = dict(ref_header)
        target_header["NAXIS1"] = w
        target_header["NAXIS2"] = h

    field_target_header = dict(target_header)
    crop_span = max(int(ref_work.shape[0]), int(ref_work.shape[1]))

    model_kinds = [p.strip() for p in args.models.split(",") if p.strip()]

    tmp_root = DATA / "_tmp"
    tmp_root.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="torchsky_smacs_fullref_", dir=str(tmp_root)
    ) as tmp:
        tmpdir = Path(tmp)
        image_observations: list[ImageObservation] = []
        raw_field_inputs: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
        ] = []
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
                (
                    science,
                    weight,
                    valid.astype(np.float32, copy=False),
                    dict(_sci_header(exposure_path)),
                )
            )

        field_observations: list[FieldObservation] = []
        forward_contexts: list[ForwardContext] = []
        for exposure_path, (image, weight, mask, header) in zip(
            exposure_paths, raw_field_inputs, strict=True
        ):
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
            obs_index = len(field_observations)
            field_observations.append(
                FieldObservation(
                    image=torch.from_numpy(crop_image),
                    header=crop_header,
                    weight=torch.from_numpy(crop_weight),
                    mask=torch.from_numpy(crop_mask),
                    **_field_observation_kwargs_for_band(
                        band, observation_mode=args.observation_mode
                    ),
                )
            )
            forward_contexts.append(
                ForwardContext(
                    survey=band,
                    exposure_id=exposure_path.stem,
                    extra={
                        "exposure_index": obs_index,
                        "source_path": str(exposure_path),
                    },
                )
            )

        if not field_observations:
            print("No exposures overlap the reference footprint.", file=sys.stderr)
            return 1

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
            band_name=band,
            observation_mode=args.observation_mode,
        )

        domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
        intensity_scale = _robust_intensity_scale(raw_field_inputs)
        normalized: list[FieldObservation] = []
        for observation in field_observations:
            normalized.append(
                _scaled_field_observation(observation, intensity_scale=intensity_scale)
            )

        forward_ctx_tuple: tuple[ForwardContext | None, ...] | None = None
        if args.forward_context:
            if len(forward_contexts) != len(normalized):
                raise RuntimeError("forward context / observation count mismatch")
            forward_ctx_tuple = tuple(forward_contexts)

        rows: list[dict[str, Any]] = []
        for model_kind in model_kinds:
            if model_kind == "coadd_bilinear":
                pred = coadd_reference.data.cpu().numpy().astype(np.float32)
                fit_time_ms = 0.0
            else:
                model_kwargs = _field_model_kwargs_for_case(
                    model_kind,
                    case=REAL_CASE,
                    size=int(crop_span),
                )
                fit_kwargs = _real_fit_kwargs(
                    model_kind,
                    crop_size=int(crop_span),
                    steps=int(args.steps),
                    observation_mode=args.observation_mode,
                    num_observations=len(normalized),
                )
                model = make_field_model(model_kind, domain, **model_kwargs)
                if args.forward_context:
                    model = _PerExposureScalarBias(model, len(normalized))
                result, fit_time_ms = _fit_with_timing(
                    normalized,
                    model,
                    steps=int(args.steps),
                    runs=int(args.runs),
                    fit_kwargs=fit_kwargs,
                    observation_forward_contexts=forward_ctx_tuple,
                )
                render_n = int(fit_kwargs.get("num_spectral_samples", 9))
                render_ctx = (
                    ForwardContext(survey=band) if args.forward_context else None
                )
                pred = (
                    render_field_observation(
                        result.model,
                        target_observation,
                        num_spectral_samples=render_n,
                        forward_context=render_ctx,
                    )
                    .detach()
                    .cpu()
                    .numpy()
                    .astype(np.float32)
                )
                pred = pred * intensity_scale

            support_metric_mask = _metric_mask(pred, ref_work, support_mask)
            support_diff = pred[support_metric_mask] - ref_work[support_metric_mask]
            rows.append(
                {
                    "model": model_kind,
                    "band": band,
                    "reference_shape": list(ref_work.shape),
                    "crop_origin_xy": [int(x0), int(y0)],
                    "full_reference_shape": [h, w],
                    "max_side_cap": args.max_side,
                    "n_exposures_used": len(field_observations),
                    "forward_context": bool(args.forward_context),
                    "support_rmse": float(np.sqrt(np.mean(support_diff * support_diff)))
                    if support_diff.size
                    else float("nan"),
                    "support_frac": float(support_metric_mask.mean()),
                    "fit_time_ms": float(fit_time_ms),
                }
            )

    out = {"rows": rows}
    text = json.dumps(out, indent=2)
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(text, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
