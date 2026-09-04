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
    _check_acceptance_contracts,
    _coadd_with_dfreproject,
    _coadd_with_montage,
    _coadd_with_reproject,
    _coadd_with_swarp,
    _geometry_family_for_torchsky_kernel,
    _load_weight_map,
    _software_manifest,
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
    _filter_exposure_paths_by_patterns,
    _injection_manifest_sign,
    _injection_manifest_tensors,
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
DEFAULT_FULL_MOSAIC_ACCEPTANCE = (
    ROOT / "benchmarks" / "baselines" / "smacs_full_mosaic_parity.json"
)

SMACS_FULL_MOSAIC_PARITY_PAIRS: tuple[dict[str, str], ...] = (
    {
        "torchsky": "torchsky_drizzle_square_pf0p800",
        "official": "drizzle_square_pf0p800",
        "library": "drizzle",
        "variant": "square:pixfrac=0.800",
    },
    {
        "torchsky": "torchsky_lanczos3",
        "official": "swarp_lanczos3",
        "library": "swarp",
        "variant": "lanczos3",
    },
    {
        "torchsky": "torchsky_conservative_polygon",
        "official": "montage_tan",
        "library": "montage",
        "variant": "mProject_or_mProjectPP:north_up_tan",
    },
)

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


def _wcs_overlap_probe(
    observations: list[ImageObservation],
    target_header: dict[str, Any],
) -> list[dict[str, Any]]:
    """Compare sparse Torchsky/Astropy source-to-target mappings."""
    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    target_torch = torchsky.WCS(target_header)
    target_astropy = WCS(_as_fits_header(target_header))
    rows: list[dict[str, Any]] = []
    for observation in observations:
        image, source_header = torchsky.read(
            observation.path,
            hdu=observation.hdu,
            return_header=True,
        )
        height, width = (int(image.shape[0]), int(image.shape[1]))
        ys, xs = torch.meshgrid(
            torch.linspace(0.0, float(height - 1), 5, dtype=torch.float64),
            torch.linspace(0.0, float(width - 1), 5, dtype=torch.float64),
            indexing="ij",
        )
        source_torch = torchsky.WCS(dict(source_header))
        ra, dec = source_torch.pixel_to_world(xs.reshape(-1), ys.reshape(-1))
        tx, ty = target_torch.world_to_pixel(ra, dec)
        torch_pixels = torch.stack((tx, ty), dim=1).detach().cpu().numpy()

        source_astropy = WCS(_as_fits_header(dict(source_header)))
        source_pixels = np.stack(
            (xs.reshape(-1).numpy(), ys.reshape(-1).numpy()), axis=1
        )
        astropy_world = source_astropy.all_pix2world(source_pixels, 0)
        astropy_pixels = target_astropy.all_world2pix(astropy_world, 0)
        finite_torch = np.isfinite(torch_pixels).all(axis=1)
        finite_astropy = np.isfinite(astropy_pixels).all(axis=1)
        common = finite_torch & finite_astropy
        inside_torch = (
            finite_torch
            & (torch_pixels[:, 0] >= -0.5)
            & (torch_pixels[:, 0] <= target_shape[1] - 0.5)
            & (torch_pixels[:, 1] >= -0.5)
            & (torch_pixels[:, 1] <= target_shape[0] - 0.5)
        )
        inside_astropy = (
            finite_astropy
            & (astropy_pixels[:, 0] >= -0.5)
            & (astropy_pixels[:, 0] <= target_shape[1] - 0.5)
            & (astropy_pixels[:, 1] >= -0.5)
            & (astropy_pixels[:, 1] <= target_shape[0] - 0.5)
        )
        rows.append(
            {
                "path": str(Path(observation.path).resolve()),
                "source_shape": [height, width],
                "source_has_sip": source_torch.sip is not None,
                "source_has_tpv": source_torch.tpv is not None,
                "torch_finite_samples": int(finite_torch.sum()),
                "astropy_finite_samples": int(finite_astropy.sum()),
                "torch_inside_samples": int(inside_torch.sum()),
                "astropy_inside_samples": int(inside_astropy.sum()),
                "max_target_pixel_disagreement": (
                    float(np.max(np.abs(torch_pixels[common] - astropy_pixels[common])))
                    if np.any(common)
                    else None
                ),
                "torch_target_x_range": (
                    [
                        float(torch_pixels[finite_torch, 0].min()),
                        float(torch_pixels[finite_torch, 0].max()),
                    ]
                    if np.any(finite_torch)
                    else None
                ),
                "torch_target_y_range": (
                    [
                        float(torch_pixels[finite_torch, 1].min()),
                        float(torch_pixels[finite_torch, 1].max()),
                    ]
                    if np.any(finite_torch)
                    else None
                ),
            }
        )
    return rows


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


def _build_full_mosaic_parity_specs() -> list[dict[str, Any]]:
    """Return matched strongest-kernel pairs on one Montage/SWarp-safe grid."""

    base = {spec["name"]: spec for spec in _build_method_specs(include_inr=False)}
    torchsky_drizzle = dict(base["torchsky_drizzle_square"])
    torchsky_drizzle.update(
        {
            "name": "torchsky_drizzle_square_pf0p800",
            "variant": "torchsky:drizzle_square:surface_brightness:pixfrac=0.800",
            "drizzle_pixfrac": 0.8,
        }
    )
    official_drizzle = dict(base["drizzle_square"])
    official_drizzle.update(
        {
            "name": "drizzle_square_pf0p800",
            "variant": "drizzle:square:pixfrac=0.800",
            "pixfrac": 0.8,
        }
    )
    specs = [
        torchsky_drizzle,
        official_drizzle,
        dict(base["torchsky_lanczos3"]),
        dict(base["swarp_lanczos3"]),
        dict(base["torchsky_conservative_polygon"]),
        dict(base["montage_tan"]),
    ]
    for spec in specs:
        spec["target_mode"] = "published_swarp_tan"
    return specs


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


def _compute_pairwise_parity_metrics(
    torchsky_data: np.ndarray,
    torchsky_weight: np.ndarray,
    comparator_data: np.ndarray,
    comparator_weight: np.ndarray,
) -> dict[str, float | int]:
    """Compare products on the exact same grid with scale-free diagnostics."""

    arrays = tuple(
        np.asarray(value, dtype=np.float32)
        for value in (
            torchsky_data,
            torchsky_weight,
            comparator_data,
            comparator_weight,
        )
    )
    if len({array.shape for array in arrays}) != 1:
        raise ValueError("parity products must have identical shapes")
    ts_data, ts_weight, ext_data, ext_weight = arrays
    ts_support = np.isfinite(ts_data) & np.isfinite(ts_weight) & (ts_weight > 0.0)
    ext_support = (
        np.isfinite(ext_data) & np.isfinite(ext_weight) & (ext_weight > 0.0)
    )
    common = ts_support & ext_support
    union = ts_support | ext_support
    common_count = int(common.sum())
    union_count = int(union.sum())
    if common_count == 0:
        raise ValueError("parity products have no common finite support")

    ts_values = ts_data[common]
    ext_values = ext_data[common]
    residual = ext_values - ts_values
    reference_rms = float(
        np.sqrt(np.mean(np.square(ts_values), dtype=np.float64))
    )
    scale = max(reference_rms, float(np.finfo(np.float32).eps))

    ts_mean = float(np.mean(ts_values, dtype=np.float64))
    ext_mean = float(np.mean(ext_values, dtype=np.float64))
    centered_ts = ts_values - ts_mean
    centered_ext = ext_values - ext_mean
    variance = float(np.mean(centered_ts * centered_ts, dtype=np.float64))
    slope = (
        float(np.mean(centered_ts * centered_ext, dtype=np.float64) / variance)
        if variance > 0.0
        else float("nan")
    )
    intercept = ext_mean - slope * ts_mean if np.isfinite(slope) else float("nan")

    ts_weights = ts_weight[common]
    ext_weights = ext_weight[common]
    ts_weight_mean = float(np.mean(ts_weights, dtype=np.float64))
    ext_weight_mean = float(np.mean(ext_weights, dtype=np.float64))
    if ts_weight_mean <= 0.0 or ext_weight_mean <= 0.0:
        raise ValueError("parity products have non-positive mean weight")
    normalized_weight_residual = (
        ext_weights / ext_weight_mean - ts_weights / ts_weight_mean
    )

    ts_flux = float(np.sum(ts_values, dtype=np.float64))
    ext_flux = float(np.sum(ext_values, dtype=np.float64))
    return {
        "common_support_pixel_count": common_count,
        "union_support_pixel_count": union_count,
        "comparison_support_frac_vs_torchsky": common_count / float(ts_data.size),
        "support_iou_vs_torchsky": common_count / float(union_count),
        "support_symmetric_difference_frac_vs_torchsky": (
            int(np.logical_xor(ts_support, ext_support).sum()) / float(ts_data.size)
        ),
        "rmse_data_diff_vs_torchsky": float(
            np.sqrt(np.mean(np.square(residual), dtype=np.float64))
        ),
        "mae_data_diff_vs_torchsky": float(
            np.mean(np.abs(residual), dtype=np.float64)
        ),
        "max_abs_data_diff_vs_torchsky": float(np.max(np.abs(residual))),
        "torchsky_data_rms": reference_rms,
        "normalized_rmse_data_diff_vs_torchsky": float(
            np.sqrt(np.mean(np.square(residual), dtype=np.float64)) / scale
        ),
        "normalized_abs_mean_data_diff_vs_torchsky": float(
            abs(np.mean(residual, dtype=np.float64)) / scale
        ),
        "flux_ratio_vs_torchsky": ext_flux / ts_flux if ts_flux != 0.0 else float("nan"),
        "affine_slope_vs_torchsky": slope,
        "affine_slope_abs_error_vs_torchsky": (
            abs(slope - 1.0) if np.isfinite(slope) else float("nan")
        ),
        "affine_intercept_vs_torchsky": intercept,
        "normalized_rmse_weight_diff_vs_torchsky": float(
            np.sqrt(
                np.mean(
                    np.square(normalized_weight_residual),
                    dtype=np.float64,
                )
            )
        ),
    }


def _score_product_parity_pairs(
    rows: list[dict[str, Any]],
    pairs: tuple[dict[str, str], ...] = SMACS_FULL_MOSAIC_PARITY_PAIRS,
) -> list[dict[str, Any]]:
    """Read one persisted pair at a time and emit strict common-grid rows."""

    by_name = {str(row["name"]): row for row in rows}
    parity_rows: list[dict[str, Any]] = []
    for pair in pairs:
        native_name = pair["torchsky"]
        official_name = pair["official"]
        parity: dict[str, Any] = {
            "case": "smacs_full_mosaic",
            "library": pair["library"],
            "variant": pair["variant"],
            "status": "OK",
            "torchsky_method": native_name,
            "official_method": official_name,
        }
        try:
            native = by_name[native_name]
            official = by_name[official_name]
            if native.get("status") != "OK" or official.get("status") != "OK":
                raise RuntimeError(
                    f"input method failed: {native_name}={native.get('status')}, "
                    f"{official_name}={official.get('status')}"
                )
            native_data, native_header = torchsky.read(
                str(native["product_path"]), return_header=True
            )
            native_weight, native_weight_header = torchsky.read(
                str(native["product_weight_path"]), return_header=True
            )
            official_data, official_header = torchsky.read(
                str(official["product_path"]), return_header=True
            )
            official_weight, official_weight_header = torchsky.read(
                str(official["product_weight_path"]), return_header=True
            )
            headers = (
                dict(native_header),
                dict(native_weight_header),
                dict(official_header),
                dict(official_weight_header),
            )
            if not all(
                headers_equivalent_for_direct_copy(headers[0], header)
                for header in headers[1:]
            ):
                raise RuntimeError("parity products do not share one exact target grid")
            parity.update(
                {
                    "comparison_reference_variant": str(native["variant"]),
                    "torchsky_time_ms": float(native["time_ms"]),
                    "official_time_ms": float(official["time_ms"]),
                    "speed_ratio_torchsky_vs_official": float(native["time_ms"])
                    / max(float(official["time_ms"]), 1.0e-12),
                    "target_header_match": True,
                    **_compute_pairwise_parity_metrics(
                        native_data.detach().cpu().numpy(),
                        native_weight.detach().cpu().numpy(),
                        official_data.detach().cpu().numpy(),
                        official_weight.detach().cpu().numpy(),
                    ),
                }
            )
            del native_data, native_weight, official_data, official_weight
        except Exception as exc:
            parity["status"] = "FAIL"
            parity["message"] = str(exc)
        parity_rows.append(parity)
    return parity_rows


def _validate_expected_exposure_count(
    paths: list[Path],
    *,
    expected: int | None,
    band: str,
    module: str,
) -> None:
    if expected is None:
        return
    if expected <= 0:
        raise ValueError("expected exposure count must be positive")
    if len(paths) != expected:
        raise RuntimeError(
            f"{band} module {module} selected {len(paths)} exposures; expected {expected}"
        )


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


def _format_parity_markdown(
    rows: list[dict[str, Any]], acceptance_errors: list[str]
) -> str:
    lines = [
        "",
        "## Full-mosaic direct parity",
        "",
        "| official comparator | kernel | status | support IoU | normalized data RMSE | normalized weight RMSE | slope error |",
        "| --- | --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for row in rows:
        if row["status"] != "OK":
            lines.append(
                f"| {row['library']} | {row['variant']} | FAIL | nan | nan | nan | nan |"
            )
            continue
        lines.append(
            f"| {row['library']} | {row['variant']} | OK | "
            f"{row['support_iou_vs_torchsky']:.6f} | "
            f"{row['normalized_rmse_data_diff_vs_torchsky']:.6g} | "
            f"{row['normalized_rmse_weight_diff_vs_torchsky']:.6g} | "
            f"{row['affine_slope_abs_error_vs_torchsky']:.6g} |"
        )
    lines.extend(
        [
            "",
            f"Acceptance: {'FAIL' if acceptance_errors else 'PASS'}",
        ]
    )
    if acceptance_errors:
        lines.extend(["", *[f"- {message}" for message in acceptance_errors]])
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
    product_only: bool = False,
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
        overlap_probes: dict[str, list[dict[str, Any]]] = {}

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
            probe_key = json.dumps(target_header, sort_keys=True, default=str)
            if probe_key not in overlap_probes:
                overlap_probes[probe_key] = _wcs_overlap_probe(
                    observations,
                    target_header,
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
                "wcs_overlap_probe": overlap_probes[probe_key],
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
                row.update(extra)
                row["time_ms"] = elapsed * 1000.0
                if product_only:
                    product_pred = np.asarray(raw_pred, dtype=np.float32)
                    product_weight = np.asarray(raw_weight, dtype=np.float32)
                    pred, weight = product_pred, product_weight
                else:
                    product_pred, product_weight, pred, weight = (
                        _product_and_metric_arrays(
                            raw_pred,
                            raw_weight,
                            target_mode_used=target_mode_used,
                            target_header=target_header,
                            published_target_header=published_target_header,
                        )
                    )
                product_support = _support_mask_from_weight(product_weight)
                row["product_support_pixel_count"] = int(product_support.sum())
                if not np.any(product_support):
                    raise RuntimeError(
                        "stacker produced no finite supported pixels on its product grid"
                    )
                if not np.all(np.isfinite(product_pred[product_support])):
                    raise RuntimeError(
                        "predicted stack contains non-finite supported product pixels"
                    )
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
                if product_only:
                    row["reference_metrics_skipped"] = True
                    row.update(
                        {
                            "support_rmse": float("nan"),
                            "uniform_rmse": float("nan"),
                            "support_flux_ratio": float("nan"),
                            "center_support_rmse": float("nan"),
                            "edge_support_rmse": float("nan"),
                            "support_rmse_after_affine_fit": float("nan"),
                        }
                    )
                    rows.append(row)
                    continue
                support_mask = _support_mask_from_weight(weight)
                row["reference_support_pixel_count"] = int(support_mask.sum())
                if not np.any(support_mask):
                    raise RuntimeError(
                        "product grid has no overlap with the reference metric grid"
                    )
                if not np.all(np.isfinite(pred[support_mask])):
                    raise RuntimeError(
                        "predicted stack contains non-finite supported pixels"
                    )
                row.update(_compute_region_metrics(pred, ref_crop, support_mask))
            except Exception as exc:
                row["status"] = "FAIL"
                row["message"] = str(exc)
                row.setdefault("time_ms", float("nan"))
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
    parser.add_argument(
        "--full-mosaic-parity",
        action="store_true",
        help=(
            "Run the fail-closed full-size Drizzle/SWarp/Montage strongest-kernel "
            "comparison on one common TAN grid"
        ),
    )
    parser.add_argument(
        "--expected-exposures",
        type=int,
        default=None,
        help="Require this exact post-filter exposure count before running",
    )
    parser.add_argument(
        "--full-mosaic-acceptance",
        type=Path,
        default=DEFAULT_FULL_MOSAIC_ACCEPTANCE,
        help="Versioned numerical contract for --full-mosaic-parity",
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
        "--product-only",
        action="store_true",
        help="Validate and persist products without comparing to the reference grid",
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
    _validate_expected_exposure_count(
        exposure_paths,
        expected=args.expected_exposures,
        band=band,
        module=args.module,
    )

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
            injection_sign=_injection_manifest_sign(injection_manifest),
        )

    if (args.crop_x0 is None) != (args.crop_y0 is None):
        raise ValueError("--crop-x0 and --crop-y0 must be provided together")
    if args.full_mosaic_parity:
        if args.validation_classical or args.methods.strip():
            raise ValueError(
                "--full-mosaic-parity owns the exact method matrix; do not combine "
                "it with --validation-classical or --methods"
            )
        if args.product_out_dir is None:
            raise ValueError("--full-mosaic-parity requires --product-out-dir")
        if args.expected_exposures is None:
            raise ValueError("--full-mosaic-parity requires --expected-exposures")
        specs = _build_full_mosaic_parity_specs()
    elif args.validation_classical:
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
        product_only=args.product_only or args.full_mosaic_parity,
    )
    for row in rows:
        row["band"] = band
        row["module"] = args.module
        row["reference_path"] = str(reference_path)
    parity_rows: list[dict[str, Any]] = []
    acceptance_errors: list[str] = []
    software = _software_manifest()
    if args.full_mosaic_parity:
        parity_rows = _score_product_parity_pairs(rows)
        comparator_versions = {
            "drizzle": software["drizzle"],
            "swarp": software["swarp_source_commit"],
            "montage": software["montage_source_commit"],
        }
        for row in parity_rows:
            row["band"] = band
            row["module"] = args.module
            row["comparator_version"] = comparator_versions.get(row["library"])
        contracts = json.loads(
            args.full_mosaic_acceptance.read_text(encoding="utf-8")
        )
        if not isinstance(contracts, list) or not all(
            isinstance(contract, dict) for contract in contracts
        ):
            raise ValueError("full-mosaic acceptance JSON must be a list of objects")
        acceptance_errors = _check_acceptance_contracts(parity_rows, contracts)
    payload: dict[str, Any] = {
        "rows": rows,
        "software": software,
        "input_count": len(exposure_paths),
        "inputs": [str(path.resolve()) for path in exposure_paths],
    }
    if args.full_mosaic_parity:
        payload.update(
            {
                "parity_rows": parity_rows,
                "acceptance_contract": str(
                    args.full_mosaic_acceptance.resolve()
                ),
                "acceptance_status": "FAIL" if acceptance_errors else "PASS",
                "acceptance_errors": acceptance_errors,
            }
        )
    if injection_audit is not None:
        payload["injection_manifest"] = str(args.injection_manifest.resolve())
        payload["injection_audit"] = injection_audit
    if args.json_out is not None:
        args.json_out.parent.mkdir(parents=True, exist_ok=True)
        args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    if args.markdown_out is not None:
        args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
        markdown = _format_markdown(rows)
        if args.full_mosaic_parity:
            markdown += _format_parity_markdown(parity_rows, acceptance_errors)
        args.markdown_out.write_text(markdown, encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.fail_on_method_error and any(row["status"] != "OK" for row in rows):
        return 1
    if args.full_mosaic_parity and acceptance_errors:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
