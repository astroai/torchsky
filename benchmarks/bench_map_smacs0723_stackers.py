#!/usr/bin/env python3
"""Compare legacy stackers, torchsky kernels, and INR models on real SMACS J0723 data."""

from __future__ import annotations

import argparse
import json
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import torch
from astropy.wcs import WCS
import torchsky  # noqa: E402

try:  # noqa: E402
    from reproject import reproject_adaptive, reproject_exact  # type: ignore
except Exception:  # pragma: no cover - optional comparator in test envs
    reproject_adaptive = None
    reproject_exact = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
BENCH = ROOT / "benchmarks"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
if str(BENCH) not in sys.path:
    sys.path.insert(0, str(BENCH))

from bench_map_compat import (  # noqa: E402
    _as_fits_header,
    _align_array_to_target,
    _coadd_with_dfreproject,
    _coadd_with_montage,
    _coadd_with_reproject,
    _coadd_with_swarp,
    _geometry_family_for_torchsky_kernel,
    _load_weight_map,
    geometry_family_for_library_variant,
)
from torchsky.map.operators import headers_equivalent_for_direct_copy  # noqa: E402
from bench_map_field_fit import (
    _field_fit_kwargs_for_model,
    _field_model_kwargs_for_case,
)  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    REAL_CASE,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _robust_intensity_scale,
)
from bench_map_smacs0723_reproduce import (
    _coadd_with_drizzle as _coadd_with_real_drizzle,
)  # noqa: E402
from bench_multiband_depth import inject_common_world_exposures  # noqa: E402
from bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    _best_overlap_cutout_from_observations,
    _band_spec,
    _field_observation_kwargs_for_band,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _linearize_sip_header,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _reference_path_for_band,
    _sci_header,
)

DEFAULT_REFERENCE_PATH = DATA / "updec_webb" / "F277Wb_UPDC-v2_iter0_i2d.fits.fz"
DEFAULT_EXPOSURE_DIR = DATA / "jwst_smacs0723"
DEFAULT_JSON_OUT = DATA / "benchmark_results" / "smacs0723_stackers.json"
DEFAULT_MARKDOWN_OUT = DATA / "benchmark_results" / "smacs0723_stackers.md"

from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    coadd_image_plane,
    estimate_field_domain,
    fit_field_model,
    make_field_model,
    render_field_observation,
)


def _uniform_mask(pred: np.ndarray, reference: np.ndarray) -> np.ndarray:
    return np.isfinite(pred) & np.isfinite(reference)


def _fixed_cutout(
    image: np.ndarray,
    *,
    size: int,
    x0: int,
    y0: int,
) -> tuple[np.ndarray, int, int]:
    if x0 < 0 or y0 < 0:
        raise ValueError("fixed crop origin must be non-negative")
    cutout = image[y0 : y0 + size, x0 : x0 + size]
    if cutout.shape != (size, size):
        raise ValueError("fixed crop extends beyond the reference image")
    return cutout, x0, y0


def _time_once(fn: Callable[[], Any]) -> tuple[float, Any]:
    t0 = time.perf_counter()
    result = fn()
    return float(time.perf_counter() - t0), result


def _build_method_specs(include_inr: bool) -> list[dict[str, Any]]:
    specs: list[dict[str, Any]] = [
        {
            "name": "torchsky_bilinear",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel("bilinear"),
            "variant": "torchsky:bilinear:surface_brightness",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "bilinear",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
        },
        {
            "name": "torchsky_lanczos3",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel("lanczos3"),
            "variant": "torchsky:lanczos3:surface_brightness",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "lanczos3",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
        },
        {
            "name": "torchsky_splat_box",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel("splat_box"),
            "variant": "torchsky:splat_box:surface_brightness:pixfrac=1.000",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "splat_box",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
            "drizzle_pixfrac": 1.0,
        },
        {
            "name": "torchsky_drizzle_square",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel("drizzle_square"),
            "variant": "torchsky:drizzle_square:surface_brightness:pixfrac=1.000",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "drizzle_square",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
            "drizzle_pixfrac": 1.0,
        },
        {
            "name": "torchsky_drizzle_point",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel("drizzle_point"),
            "variant": "torchsky:drizzle_point:surface_brightness:pixfrac=1.000",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "drizzle_point",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
            "drizzle_pixfrac": 1.0,
        },
        {
            "name": "torchsky_conservative_polygon",
            "family": "torchsky",
            "geometry_family": _geometry_family_for_torchsky_kernel(
                "conservative_polygon"
            ),
            "variant": "torchsky:conservative_polygon:surface_brightness",
            "target_mode": "published_native",
            "kind": "coadd",
            "kernel": "conservative_polygon",
            "value_semantics": "surface_brightness",
            "stacking_method": "weighted_mean",
        },
        {
            "name": "drizzle_square",
            "family": "drizzle",
            "geometry_family": geometry_family_for_library_variant("drizzle", "square"),
            "variant": "drizzle:square:pixfrac=1.000",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "drizzle",
            "kernel": "square",
            "pixfrac": 1.0,
        },
        {
            "name": "drizzle_point",
            "family": "drizzle",
            "geometry_family": geometry_family_for_library_variant("drizzle", "point"),
            "variant": "drizzle:point:pixfrac=1.000",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "drizzle",
            "kernel": "point",
            "pixfrac": 1.0,
        },
        {
            "name": "reproject_bilinear",
            "family": "reproject",
            "geometry_family": geometry_family_for_library_variant(
                "reproject_interp",
                "bilinear",
            ),
            "variant": "reproject_interp:bilinear",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "reproject",
            "order": "bilinear",
        },
        {
            "name": "reproject_bicubic",
            "family": "reproject",
            "geometry_family": geometry_family_for_library_variant(
                "reproject_interp",
                "bicubic",
            ),
            "variant": "reproject_interp:bicubic",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "reproject",
            "order": "bicubic",
        },
        {
            "name": "reproject_exact",
            "family": "reproject",
            "geometry_family": geometry_family_for_library_variant(
                "reproject_exact", None
            ),
            "variant": "reproject_exact",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "reproject_exact",
        },
        {
            "name": "reproject_adaptive",
            "family": "reproject",
            "geometry_family": geometry_family_for_library_variant(
                "reproject_adaptive",
                "gaussian",
            ),
            "variant": "reproject_adaptive:gaussian",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "reproject_adaptive",
        },
        {
            "name": "dfreproject_bilinear",
            "family": "dfreproject",
            "geometry_family": geometry_family_for_library_variant(
                "dfreproject", "bilinear"
            ),
            "variant": "dfreproject:bilinear",
            "target_mode": "published_native",
            "kind": "legacy",
            "runner": "dfreproject",
            "order": "bilinear",
        },
        {
            "name": "montage",
            "family": "montage",
            "geometry_family": geometry_family_for_library_variant(
                "montage",
                "mProject_or_mProjectPP",
            ),
            "variant": "montage:mProject_or_mProjectPP",
            "target_mode": "published_linearized",
            "kind": "legacy",
            "runner": "montage",
        },
        {
            "name": "montage_tan",
            "family": "montage",
            "geometry_family": geometry_family_for_library_variant(
                "montage",
                "mProject_or_mProjectPP:north_up_tan",
            ),
            "variant": "montage:mProject_or_mProjectPP:north_up_tan",
            "target_mode": "published_swarp_tan",
            "kind": "legacy",
            "runner": "montage",
        },
        {
            "name": "swarp_bilinear",
            "family": "swarp",
            "geometry_family": geometry_family_for_library_variant("swarp", "BILINEAR"),
            "variant": "swarp:BILINEAR",
            "target_mode": "published_swarp_tan",
            "kind": "legacy",
            "runner": "swarp",
            "resampling_type": "BILINEAR",
        },
        {
            "name": "swarp_lanczos3",
            "family": "swarp",
            "geometry_family": geometry_family_for_library_variant("swarp", "LANCZOS3"),
            "variant": "swarp:LANCZOS3",
            "target_mode": "published_swarp_tan",
            "kind": "legacy",
            "runner": "swarp",
            "resampling_type": "LANCZOS3",
        },
    ]
    if include_inr:
        specs.extend(
            [
                {
                    "name": "inr_spectral_basis_field",
                    "family": "inr",
                    "geometry_family": "latent_scene",
                    "variant": "inr:spectral_basis_field",
                    "target_mode": "published_linearized",
                    "kind": "inr",
                    "model_kind": "spectral_basis_field",
                },
                {
                    "name": "inr_sky_encoder_resmlp",
                    "family": "inr",
                    "geometry_family": "latent_scene",
                    "variant": "inr:sky_encoder_resmlp",
                    "target_mode": "published_linearized",
                    "kind": "inr",
                    "model_kind": "sky_encoder_resmlp",
                },
                {
                    "name": "inr_sky_lambda_feature_field",
                    "family": "inr",
                    "geometry_family": "latent_scene",
                    "variant": "inr:sky_lambda_feature_field",
                    "target_mode": "published_linearized",
                    "kind": "inr",
                    "model_kind": "sky_lambda_feature_field",
                },
                {
                    "name": "inr_operator_constrained_feature_field",
                    "family": "inr",
                    "geometry_family": "latent_scene",
                    "variant": "inr:operator_constrained_feature_field",
                    "target_mode": "published_linearized",
                    "kind": "inr",
                    "model_kind": "operator_constrained_feature_field",
                },
            ]
        )
    return specs


def _build_validation_method_specs(
    drizzle_pixfracs: tuple[float, ...] = (0.6, 0.8, 1.0),
) -> list[dict[str, Any]]:
    """Return the pre-registered classical validation sweep."""

    if not drizzle_pixfracs or any(value <= 0.0 for value in drizzle_pixfracs):
        raise ValueError("drizzle pixfracs must be positive")
    base = {spec["name"]: spec for spec in _build_method_specs(include_inr=False)}
    specs = [
        dict(base["torchsky_bilinear"]),
        dict(base["torchsky_lanczos3"]),
    ]
    for pixfrac in drizzle_pixfracs:
        suffix = f"{pixfrac:.3f}".replace(".", "p")
        torchsky_spec = dict(base["torchsky_drizzle_square"])
        torchsky_spec.update(
            {
                "name": f"torchsky_drizzle_square_pf{suffix}",
                "variant": (
                    f"torchsky:drizzle_square:surface_brightness:pixfrac={pixfrac:.3f}"
                ),
                "drizzle_pixfrac": float(pixfrac),
            }
        )
        upstream_spec = dict(base["drizzle_square"])
        upstream_spec.update(
            {
                "name": f"drizzle_square_pf{suffix}",
                "variant": f"drizzle:square:pixfrac={pixfrac:.3f}",
                "pixfrac": float(pixfrac),
            }
        )
        specs.extend((torchsky_spec, upstream_spec))
    specs.extend(
        (
            dict(base["swarp_bilinear"]),
            dict(base["swarp_lanczos3"]),
        )
    )
    for spec in specs:
        spec["target_mode"] = "published_swarp_tan"
    return specs


def _filter_exposure_paths_by_patterns(
    paths: list[Path],
    pattern_numbers: tuple[int, ...],
) -> list[Path]:
    """Select complete JWST dither identifiers without opening FITS files."""

    if any(pattern < 1 or pattern > 9 for pattern in pattern_numbers):
        raise ValueError("PATT_NUM selections must be integers from 1 through 9")
    tokens = tuple(f"_{pattern:05d}_" for pattern in pattern_numbers)
    return [path for path in paths if any(token in path.name for token in tokens)]


def _injection_manifest_tensors(
    manifest: dict[str, Any],
    *,
    band: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Validate JSON injection arrays and restore the empty coordinate shape."""

    band_flux = manifest.get("band_flux_jy", {}).get(band)
    if band_flux is None:
        raise ValueError(f"injection manifest has no band_flux_jy entry for {band}")
    world = torch.as_tensor(manifest.get("world_radec_deg", []), dtype=torch.float64)
    if world.numel() == 0:
        world = world.reshape(0, 2)
    flux = torch.as_tensor(band_flux, dtype=torch.float64)
    if world.ndim != 2 or world.shape[1] != 2 or flux.shape != (world.shape[0],):
        raise ValueError("injection manifest coordinates/flux have inconsistent shapes")
    return world, flux


def _resolve_target_header(
    mode: str,
    *,
    published_target_header: dict[str, Any],
    linearized_target_header: dict[str, Any],
    swarp_target_header: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if mode == "published_native":
        return dict(published_target_header), "published_native"
    if mode == "published_linearized":
        return dict(linearized_target_header), "published_linearized"
    if mode == "published_swarp_tan":
        return dict(swarp_target_header), "published_swarp_tan"
    raise ValueError(f"unsupported target mode {mode!r}")


def _resolve_benchmark_target_header(
    mode: str,
    *,
    crop_size: int,
    explicit_header: dict[str, Any] | None,
    published_target_header: dict[str, Any],
    linearized_target_header: dict[str, Any],
    swarp_target_header: dict[str, Any],
) -> tuple[dict[str, Any], str]:
    if explicit_header is None:
        return _resolve_target_header(
            mode,
            published_target_header=published_target_header,
            linearized_target_header=linearized_target_header,
            swarp_target_header=swarp_target_header,
        )
    if (
        int(explicit_header.get("NAXIS1", -1)) != crop_size
        or int(explicit_header.get("NAXIS2", -1)) != crop_size
    ):
        raise ValueError("explicit target header shape must match --crop-size")
    return dict(explicit_header), "explicit_fits"


def _north_up_tan_header(header: dict[str, Any]) -> dict[str, Any]:
    out = dict(_linearize_sip_header(header))
    wcs = WCS(_as_fits_header(out))
    matrix = np.asarray(wcs.pixel_scale_matrix, dtype=np.float64)
    scale_x = float(np.sqrt(np.square(matrix[:, 0]).sum()))
    scale_y = float(np.sqrt(np.square(matrix[:, 1]).sum()))
    scale = 0.5 * (abs(scale_x) + abs(scale_y))
    width = int(out["NAXIS1"])
    height = int(out["NAXIS2"])
    center_pix = np.array(
        [[0.5 * float(width - 1), 0.5 * float(height - 1)]], dtype=np.float64
    )
    center_world = wcs.all_pix2world(center_pix, 0)[0]
    out["CTYPE1"] = "RA---TAN"
    out["CTYPE2"] = "DEC--TAN"
    out["CRPIX1"] = 0.5 * float(width - 1) + 1.0
    out["CRPIX2"] = 0.5 * float(height - 1) + 1.0
    out["CRVAL1"] = float(center_world[0] % 360.0)
    out["CRVAL2"] = float(center_world[1])
    for key in list(out.keys()):
        if str(key).startswith("PC") or key in {"CDELT1", "CDELT2", "CROTA1", "CROTA2"}:
            del out[key]
    out["CD1_1"] = -float(scale)
    out["CD1_2"] = 0.0
    out["CD2_1"] = 0.0
    out["CD2_2"] = float(scale)
    for key in list(out.keys()):
        if str(key).startswith("PV"):
            del out[key]
    return out


def _support_mask_from_weight(weight: np.ndarray) -> np.ndarray:
    return np.isfinite(weight) & (weight > 0.0)


def _product_and_metric_arrays(
    pred: np.ndarray,
    weight: np.ndarray,
    *,
    target_mode_used: str,
    target_header: dict[str, Any],
    published_target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    product_pred = np.asarray(pred, dtype=np.float32)
    product_weight = np.asarray(weight, dtype=np.float32)
    if target_mode_used == "published_native":
        return product_pred, product_weight, product_pred, product_weight
    metric_pred = _align_array_to_target(
        product_pred,
        header=target_header,
        target_header=published_target_header,
        order="bilinear",
    )
    metric_weight = _align_array_to_target(
        product_weight,
        header=target_header,
        target_header=published_target_header,
        order="nearest-neighbor",
    )
    return product_pred, product_weight, metric_pred, metric_weight


def _radial_bin_masks(
    shape: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    height, width = shape
    yy, xx = np.mgrid[0:height, 0:width]
    cx = 0.5 * float(width - 1)
    cy = 0.5 * float(height - 1)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_rr = float(rr.max()) if rr.size else 1.0
    if max_rr <= 0.0:
        max_rr = 1.0
    rr = rr / max_rr
    return rr <= 0.33, (rr > 0.33) & (rr <= 0.66), rr > 0.66


def _safe_stat_mask(
    values: np.ndarray, mask: np.ndarray, fn: Callable[[np.ndarray], float]
) -> float:
    if not np.any(mask):
        return float("nan")
    return float(fn(values[mask]))


def _compute_region_metrics(
    pred: np.ndarray,
    reference: np.ndarray,
    support_mask: np.ndarray,
) -> dict[str, float]:
    metric_mask = _metric_mask(pred, reference, support_mask)
    uniform_mask = _uniform_mask(pred, reference)
    support_residual = pred - reference
    center_mask, mid_mask, edge_mask = _radial_bin_masks(reference.shape)

    support_reference = reference[metric_mask]
    support_prediction = pred[metric_mask]
    support_diff = support_prediction - support_reference
    uniform_diff = pred[uniform_mask] - reference[uniform_mask]

    metrics: dict[str, float] = {
        "support_rmse": float(np.sqrt(np.mean(np.square(support_diff))))
        if support_diff.size
        else float("nan"),
        "support_mae": float(np.mean(np.abs(support_diff)))
        if support_diff.size
        else float("nan"),
        "support_mean_residual": float(np.mean(support_diff))
        if support_diff.size
        else float("nan"),
        "support_median_residual": float(np.median(support_diff))
        if support_diff.size
        else float("nan"),
        "support_flux_ratio": (
            float(np.sum(support_prediction) / np.sum(support_reference))
            if support_diff.size and float(np.sum(support_reference)) != 0.0
            else float("nan")
        ),
        "support_frac": float(metric_mask.mean()),
        "uniform_rmse": float(np.sqrt(np.mean(np.square(uniform_diff))))
        if uniform_diff.size
        else float("nan"),
        "uniform_mae": float(np.mean(np.abs(uniform_diff)))
        if uniform_diff.size
        else float("nan"),
        "uniform_mean_residual": float(np.mean(uniform_diff))
        if uniform_diff.size
        else float("nan"),
        "uniform_median_residual": float(np.median(uniform_diff))
        if uniform_diff.size
        else float("nan"),
        "uniform_frac": float(uniform_mask.mean()),
    }

    for label, region in (
        ("center", center_mask),
        ("mid", mid_mask),
        ("edge", edge_mask),
    ):
        region_mask = metric_mask & region
        metrics[f"{label}_support_rmse"] = _safe_stat_mask(
            np.square(support_residual),
            region_mask,
            lambda x: float(np.sqrt(np.mean(x))),
        )
        metrics[f"{label}_support_mean_residual"] = _safe_stat_mask(
            support_residual,
            region_mask,
            lambda x: float(np.mean(x)),
        )

    if support_diff.size:
        finite_ref = support_reference[np.isfinite(support_reference)]
        if finite_ref.size:
            signal_threshold = float(np.median(finite_ref))
            low_signal_mask = metric_mask & (reference <= signal_threshold)
            high_signal_mask = metric_mask & (reference > signal_threshold)
            metrics["low_signal_support_rmse"] = _safe_stat_mask(
                np.square(support_residual),
                low_signal_mask,
                lambda x: float(np.sqrt(np.mean(x))),
            )
            metrics["high_signal_support_rmse"] = _safe_stat_mask(
                np.square(support_residual),
                high_signal_mask,
                lambda x: float(np.sqrt(np.mean(x))),
            )

            x = support_reference.astype(np.float64)
            y = support_prediction.astype(np.float64)
            design = np.stack([x, np.ones_like(x)], axis=1)
            coeffs, *_ = np.linalg.lstsq(design, y, rcond=None)
            affine_pred = coeffs[0] * x + coeffs[1]
            metrics["affine_fit_slope"] = float(coeffs[0])
            metrics["affine_fit_intercept"] = float(coeffs[1])
            metrics["support_rmse_after_affine_fit"] = float(
                np.sqrt(np.mean(np.square(affine_pred - y)))
            )
    return metrics


def _format_markdown(rows: list[dict[str, Any]]) -> str:
    lines = [
        "# SMACS J0723 Stacker Comparison",
        "",
        "| variant | family | geometry | status | target | support rmse | uniform rmse | flux ratio | center rmse | edge rmse | affine-corrected rmse | time ms |",
        "| --- | --- | --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['variant']} | {row['family']} | {row.get('geometry_family', '')} | {row['status']} | {row.get('target_mode_used', '')} | "
                f"nan | nan | nan | nan | nan | nan | {row.get('time_ms', float('nan')):.2f} |"
            )
            continue
        lines.append(
            f"| {row['variant']} | {row['family']} | {row.get('geometry_family', '')} | {row['status']} | {row.get('target_mode_used', '')} | "
            f"{row['support_rmse']:.6f} | {row['uniform_rmse']:.6f} | {row['support_flux_ratio']:.4f} | "
            f"{row['center_support_rmse']:.6f} | {row['edge_support_rmse']:.6f} | "
            f"{row.get('support_rmse_after_affine_fit', float('nan')):.6f} | {row['time_ms']:.2f} |"
        )
    return "\n".join(lines) + "\n"


def _run_coadd_method(
    observations: list[ImageObservation],
    spec: dict[str, Any],
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    result = coadd_image_plane(
        observations,
        target_header=dict(target_header),
        resampling_kernel=str(spec["kernel"]),
        value_semantics=str(spec["value_semantics"]),
        stacking_method=str(spec["stacking_method"]),
        drizzle_pixfrac=float(spec.get("drizzle_pixfrac", 1.0)),
    )
    extra = {
        "max_overlap": (
            int(result.overlap_count.max().item())
            if result.overlap_count is not None
            else 0
        ),
    }
    return (
        result.data.cpu().numpy().astype(np.float32, copy=False),
        result.weight.cpu().numpy().astype(np.float32, copy=False),
        extra,
    )


def _run_legacy_method(
    observations: list[ImageObservation],
    spec: dict[str, Any],
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    runner = str(spec["runner"])
    if runner == "drizzle":
        data, weight = _coadd_with_real_drizzle(
            observations,
            target_header=dict(target_header),
            kernel=str(spec["kernel"]),
            pixfrac=float(spec.get("pixfrac", 1.0)),
            value_semantics="surface_brightness",
        )
        return data, weight, {}
    if runner == "reproject":
        data, weight = _coadd_with_reproject(
            observations,
            target_header=dict(target_header),
            order=str(spec["order"]),
        )
        return data, weight, {}
    if runner == "reproject_exact":
        data, weight = _coadd_with_reproject_exact(
            observations,
            target_header=dict(target_header),
        )
        return data, weight, {}
    if runner == "reproject_adaptive":
        data, weight = _coadd_with_reproject_adaptive(
            observations,
            target_header=dict(target_header),
        )
        return data, weight, {}
    if runner == "dfreproject":
        data, weight = _coadd_with_dfreproject(
            observations,
            target_header=dict(target_header),
            order=str(spec["order"]),
        )
        return data, weight, {}
    if runner == "montage":
        data, weight, mode, backend_path = _coadd_with_montage(
            observations,
            target_header=dict(target_header),
        )
        return data, weight, {"montage_mode": mode, "backend_path": backend_path}
    if runner == "swarp":
        data, weight = _coadd_with_swarp(
            observations,
            target_header=dict(target_header),
            resampling_type=str(spec["resampling_type"]),
        )
        return data, weight, {}
    raise ValueError(f"unsupported legacy runner {runner!r}")


def _coadd_with_reproject_exact(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if reproject_exact is None:
        raise RuntimeError("reproject_exact is unavailable")
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer = np.zeros(target_shape, dtype=np.float32)
    denom = np.zeros(target_shape, dtype=np.float32)
    target_header_astropy = _as_fits_header(target_header)
    for obs in observations:
        image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
        image = image.to(dtype=torch.float32)
        weight = _load_weight_map(obs, image)
        if (
            headers_equivalent_for_direct_copy(dict(header), target_header)
            and tuple(int(v) for v in image.shape) == target_shape
        ):
            obs_numer = (
                torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
            )
            numer += obs_numer.cpu().numpy()
            denom += weight.cpu().numpy()
            continue
        obs_numer, _ = reproject_exact(
            (
                (torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight)
                .cpu()
                .numpy(),
                _as_fits_header(dict(header)),
            ),
            target_header_astropy,
            shape_out=target_shape,
            return_footprint=True,
        )
        obs_denom, _ = reproject_exact(
            (weight.cpu().numpy(), _as_fits_header(dict(header))),
            target_header_astropy,
            shape_out=target_shape,
            return_footprint=True,
        )
        numer += np.nan_to_num(
            np.asarray(obs_numer, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        denom += np.nan_to_num(
            np.asarray(obs_denom, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
    data = numer / np.clip(denom, 1e-8, None)
    data = np.where(denom > 0.0, data, 0.0)
    return data.astype(np.float32, copy=False), denom.astype(np.float32, copy=False)


def _coadd_with_reproject_adaptive(
    observations: list[ImageObservation],
    *,
    target_header: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    if reproject_adaptive is None:
        raise RuntimeError("reproject_adaptive is unavailable")
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    numer = np.zeros(target_shape, dtype=np.float32)
    denom = np.zeros(target_shape, dtype=np.float32)
    target_header_astropy = _as_fits_header(target_header)
    for obs in observations:
        image, header = torchsky.read(obs.path, hdu=obs.hdu, return_header=True)
        image = image.to(dtype=torch.float32)
        weight = _load_weight_map(obs, image)
        if (
            headers_equivalent_for_direct_copy(dict(header), target_header)
            and tuple(int(v) for v in image.shape) == target_shape
        ):
            obs_numer = (
                torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight
            )
            numer += obs_numer.cpu().numpy()
            denom += weight.cpu().numpy()
            continue
        obs_numer, _ = reproject_adaptive(
            (
                (torch.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0) * weight)
                .cpu()
                .numpy(),
                _as_fits_header(dict(header)),
            ),
            target_header_astropy,
            shape_out=target_shape,
            return_footprint=True,
            conserve_flux=False,
            kernel="gaussian",
        )
        obs_denom, _ = reproject_adaptive(
            (weight.cpu().numpy(), _as_fits_header(dict(header))),
            target_header_astropy,
            shape_out=target_shape,
            return_footprint=True,
            conserve_flux=False,
            kernel="gaussian",
        )
        numer += np.nan_to_num(
            np.asarray(obs_numer, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
        denom += np.nan_to_num(
            np.asarray(obs_denom, dtype=np.float32), nan=0.0, posinf=0.0, neginf=0.0
        )
    data = numer / np.clip(denom, 1e-8, None)
    data = np.where(denom > 0.0, data, 0.0)
    return data.astype(np.float32, copy=False), denom.astype(np.float32, copy=False)


def _run_inr_method(
    raw_field_inputs: list[tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]],
    spec: dict[str, Any],
    target_header: dict[str, Any],
    *,
    band: str,
    crop_size: int,
    steps: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
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
                **_field_observation_kwargs_for_band(band),
            )
        )
    if not field_observations:
        raise RuntimeError("no field observations overlap the requested target crop")
    domain = estimate_field_domain(field_observations, coord_mode="sphere_xyz")
    intensity_scale = _robust_intensity_scale(raw_field_inputs)
    normalized_observations: list[FieldObservation] = []
    for observation in field_observations:
        norm_weight = (
            observation.weight * (intensity_scale * intensity_scale)
            if observation.weight is not None
            else None
        )
        normalized_observations.append(
            FieldObservation(
                image=observation.image / intensity_scale,
                header=dict(observation.header),
                weight=norm_weight,
                mask=observation.mask.clone() if observation.mask is not None else None,
                wavelength=observation.wavelength,
                band_name=observation.band_name,
            )
        )
    model_kind = str(spec["model_kind"])
    model_kwargs = _field_model_kwargs_for_case(
        model_kind, case=REAL_CASE, size=crop_size
    )
    fit_kwargs = _field_fit_kwargs_for_model(
        model_kind,
        case=REAL_CASE,
        size=crop_size,
        benchmark_steps=steps,
    )
    model = make_field_model(model_kind, domain, **model_kwargs)
    result = fit_field_model(normalized_observations, model, steps=steps, **fit_kwargs)
    target_observation = _field_target_observation(
        target_header,
        band_name=band,
    )
    render_num_spectral_samples = int(fit_kwargs.get("num_spectral_samples", 1))
    pred = (
        render_field_observation(
            result.model,
            target_observation,
            num_spectral_samples=render_num_spectral_samples,
        )
        .detach()
        .cpu()
        .numpy()
        .astype(np.float32, copy=False)
    )
    pred = pred * intensity_scale
    weight = np.isfinite(pred).astype(np.float32, copy=False)
    return pred, weight, {}


def run_stacker_benchmark(
    *,
    band: str,
    reference_path: Path,
    exposure_paths: list[Path],
    specs: list[dict[str, Any]],
    crop_size: int,
    crop_mode: str,
    overlap_step: int,
    input_product: str,
    filename_contains: str,
    linearize_sip: bool,
    inr_steps: int,
    crop_xy0: tuple[int, int] | None = None,
    science_overrides: dict[Path, np.ndarray] | None = None,
    target_header_override: dict[str, Any] | None = None,
    product_out_dir: Path | None = None,
) -> list[dict[str, Any]]:
    ref_data = _read_ext(reference_path, "SCI")
    ref_header = _sci_header(reference_path)

    with tempfile.TemporaryDirectory(prefix="torchsky_smacs0723_stackers_") as tmp:
        tmpdir = Path(tmp)
        observations: list[ImageObservation] = []
        raw_field_inputs: list[
            tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
        ] = []
        for exposure_path in exposure_paths:
            science_override = (
                science_overrides.get(exposure_path)
                if science_overrides is not None
                else None
            )
            science_path, weight_path, mask_path = _prepare_aux_files(
                exposure_path,
                tmpdir,
                linearize_sip=linearize_sip,
                science_override=science_override,
            )
            observations.append(
                ImageObservation(
                    str(science_path),
                    hdu=0,
                    weight_path=str(weight_path),
                    weight_hdu=0,
                    mask_path=str(mask_path),
                    mask_hdu=0,
                )
            )
            science = (
                _read_ext(exposure_path, "SCI")
                if science_override is None
                else np.asarray(science_override)
            )
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
        if crop_xy0 is not None:
            ref_crop, x0, y0 = _fixed_cutout(
                ref_data,
                size=crop_size,
                x0=int(crop_xy0[0]),
                y0=int(crop_xy0[1]),
            )
            resolved_crop_mode = "fixed"
        elif crop_mode == "best_overlap":
            ref_crop, x0, y0 = _best_overlap_cutout_from_observations(
                ref_data,
                observations,
                ref_header,
                size=crop_size,
                coarse_step=overlap_step,
                resampling_kernel="drizzle_square",
                value_semantics="surface_brightness",
                stacking_method="weighted_mean",
            )
            resolved_crop_mode = crop_mode
        else:
            ref_crop, x0, y0 = _center_cutout(ref_data, crop_size)
            resolved_crop_mode = crop_mode

        published_target_header = dict(_cutout_header(ref_header, x0, y0, crop_size))
        linearized_target_header = dict(_linearize_sip_header(published_target_header))
        swarp_target_header = _north_up_tan_header(published_target_header)

        rows: list[dict[str, Any]] = []
        for spec in specs:
            target_header, target_mode_used = _resolve_benchmark_target_header(
                str(spec["target_mode"]),
                crop_size=crop_size,
                explicit_header=target_header_override,
                published_target_header=published_target_header,
                linearized_target_header=linearized_target_header,
                swarp_target_header=swarp_target_header,
            )
            row: dict[str, Any] = {
                "name": spec["name"],
                "variant": spec["variant"],
                "family": spec["family"],
                "geometry_family": spec["geometry_family"],
                "status": "OK",
                "target_mode_used": target_mode_used,
                "crop_size": crop_size,
                "crop_mode": resolved_crop_mode,
                "crop_x0": int(x0),
                "crop_y0": int(y0),
                "n_exposures": len(exposure_paths),
                "input_product": input_product,
                "filename_contains": filename_contains,
            }
            try:
                if spec["kind"] == "coadd":
                    elapsed, result = _time_once(
                        lambda: _run_coadd_method(observations, spec, target_header)
                    )
                elif spec["kind"] == "legacy":
                    elapsed, result = _time_once(
                        lambda: _run_legacy_method(observations, spec, target_header)
                    )
                elif spec["kind"] == "inr":
                    elapsed, result = _time_once(
                        lambda: _run_inr_method(
                            raw_field_inputs,
                            spec,
                            target_header,
                            band=band,
                            crop_size=crop_size,
                            steps=inr_steps,
                        )
                    )
                else:
                    raise ValueError(f"unsupported method kind {spec['kind']!r}")
                raw_pred, raw_weight, extra = result
                product_pred, product_weight, pred, weight = _product_and_metric_arrays(
                    raw_pred,
                    raw_weight,
                    target_mode_used=target_mode_used,
                    target_header=target_header,
                    published_target_header=published_target_header,
                )
                support_mask = _support_mask_from_weight(weight)
                if not np.any(support_mask):
                    raise RuntimeError(
                        "stacker produced no finite supported pixels on the target grid"
                    )
                if not np.all(np.isfinite(pred[support_mask])):
                    raise RuntimeError(
                        "predicted stack contains non-finite supported pixels"
                    )
                row.update(_compute_region_metrics(pred, ref_crop, support_mask))
                row.update(extra)
                row["time_ms"] = elapsed * 1000.0
                if product_out_dir is not None:
                    product_out_dir.mkdir(parents=True, exist_ok=True)
                    product_path = product_out_dir / f"{spec['name']}_sci.fits"
                    weight_path = product_out_dir / f"{spec['name']}_wht.fits"
                    torchsky.write(
                        str(product_path),
                        torch.from_numpy(product_pred),
                        header=target_header,
                        overwrite=True,
                    )
                    torchsky.write(
                        str(weight_path),
                        torch.from_numpy(product_weight),
                        header=target_header,
                        overwrite=True,
                    )
                    row["product_path"] = str(product_path.resolve())
                    row["product_weight_path"] = str(weight_path.resolve())
            except Exception as exc:
                row["status"] = "FAIL"
                row["message"] = str(exc)
                row["time_ms"] = float("nan")
            rows.append(row)
    ok_rows = [row for row in rows if row["status"] == "OK"]
    fail_rows = [row for row in rows if row["status"] != "OK"]
    ok_rows.sort(key=lambda row: float(row["support_rmse"]))
    return ok_rows + fail_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference-path", type=Path, default=None)
    parser.add_argument("--band", default="F277W")
    parser.add_argument("--module", default="b")
    parser.add_argument("--exposure-dir", type=Path, default=DEFAULT_EXPOSURE_DIR)
    parser.add_argument("--exposures", default="")
    parser.add_argument(
        "--patt-nums",
        default="",
        help="Comma-separated PATT_NUM identifiers, for example 3,8 for validation",
    )
    parser.add_argument("--crop-size", type=int, default=512)
    parser.add_argument(
        "--crop-mode", choices=("center", "best_overlap"), default="center"
    )
    parser.add_argument("--overlap-step", type=int, default=8)
    parser.add_argument("--crop-x0", type=int, default=None)
    parser.add_argument("--crop-y0", type=int, default=None)
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal", "aligned"),
        default="cal",
    )
    parser.add_argument("--filename-contains", default="")
    parser.add_argument("--linearize-sip", action="store_true")
    parser.add_argument("--include-inr", action="store_true")
    parser.add_argument("--inr-steps", type=int, default=4)
    parser.add_argument(
        "--validation-classical",
        action="store_true",
        help="Run the frozen Torchsky/Drizzle/SWarp classical validation sweep",
    )
    parser.add_argument("--drizzle-pixfracs", default="0.6,0.8,1.0")
    parser.add_argument(
        "--fail-on-method-error",
        action="store_true",
        help="Exit nonzero after writing artifacts when any selected method failed",
    )
    parser.add_argument(
        "--methods",
        default="",
        help="Comma-separated method names to run; defaults to the full legacy/torchsky set",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument(
        "--injection-manifest",
        type=Path,
        default=None,
        help="JSON common-world catalogue with world_radec_deg and band_flux_jy",
    )
    parser.add_argument(
        "--product-out-dir",
        type=Path,
        default=None,
        help="Persist each reconstructed SCI/weight product on its method target grid",
    )
    parser.add_argument(
        "--target-header-path",
        type=Path,
        default=None,
        help="FITS image supplying one exact target WCS for every selected method",
    )
    args = parser.parse_args()

    band = str(args.band).upper()
    _band_spec(band)
    reference_path = args.reference_path or _reference_path_for_band(band)
    if args.exposures.strip():
        exposure_paths = [
            Path(part.strip()) for part in args.exposures.split(",") if part.strip()
        ]
    else:
        exposure_paths = _discover_exposure_paths(
            args.exposure_dir,
            input_product=args.input_product,
            band=band,
            module=args.module,
            filename_contains=args.filename_contains or None,
        )
    if args.patt_nums.strip():
        pattern_numbers = tuple(
            int(part.strip()) for part in args.patt_nums.split(",") if part.strip()
        )
        exposure_paths = _filter_exposure_paths_by_patterns(
            exposure_paths,
            pattern_numbers,
        )
    if not exposure_paths:
        raise FileNotFoundError("No local exposures found for the requested selection")

    science_overrides = None
    injection_audit = None
    if args.injection_manifest is not None:
        injection_manifest = json.loads(
            args.injection_manifest.read_text(encoding="utf-8")
        )
        injection_world, injection_flux = _injection_manifest_tensors(
            injection_manifest,
            band=band,
        )
        science_overrides, injection_audit = inject_common_world_exposures(
            exposure_paths,
            band=band,
            world_radec_deg=injection_world,
            flux_jy=injection_flux,
        )

    if (args.crop_x0 is None) != (args.crop_y0 is None):
        raise ValueError("--crop-x0 and --crop-y0 must be provided together")
    if args.validation_classical:
        pixfracs = tuple(
            float(part.strip())
            for part in args.drizzle_pixfracs.split(",")
            if part.strip()
        )
        specs = _build_validation_method_specs(pixfracs)
    else:
        specs = _build_method_specs(include_inr=args.include_inr)
    if args.methods.strip():
        requested = {part.strip() for part in args.methods.split(",") if part.strip()}
        specs = [spec for spec in specs if spec["name"] in requested]
        if not specs:
            raise ValueError(
                "none of the requested methods matched the available specs"
            )

    rows = run_stacker_benchmark(
        band=band,
        reference_path=reference_path,
        exposure_paths=exposure_paths,
        specs=specs,
        crop_size=args.crop_size,
        crop_mode=args.crop_mode,
        overlap_step=args.overlap_step,
        input_product=args.input_product,
        filename_contains=args.filename_contains,
        linearize_sip=args.linearize_sip,
        inr_steps=args.inr_steps,
        crop_xy0=(
            (args.crop_x0, args.crop_y0)
            if args.crop_x0 is not None and args.crop_y0 is not None
            else None
        ),
        science_overrides=science_overrides,
        target_header_override=(
            dict(torchsky.read(str(args.target_header_path), return_header=True)[1])
            if args.target_header_path is not None
            else None
        ),
        product_out_dir=args.product_out_dir,
    )
    for row in rows:
        row["band"] = band
        row["module"] = args.module
        row["reference_path"] = str(reference_path)
    payload = {"rows": rows}
    if injection_audit is not None:
        payload["injection_manifest"] = str(args.injection_manifest.resolve())
        payload["injection_audit"] = injection_audit
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        args.markdown_out.write_text(_format_markdown(rows), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.fail_on_method_error and any(row["status"] != "OK" for row in rows):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
