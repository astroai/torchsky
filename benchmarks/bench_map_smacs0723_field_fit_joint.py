#!/usr/bin/env python3
"""Joint multiband real-data field-fit benchmark on JWST SMACS J0723 products."""

from __future__ import annotations

import argparse
import json
import os
import math
import sys
import tempfile
import time
from pathlib import Path
from collections.abc import Callable, Sequence
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

from bench_map_field_fit import (
    _field_fit_kwargs_for_model,
    _field_model_kwargs_for_case,
)  # noqa: E402
from bench_map_smacs0723_field_fit import (  # noqa: E402
    _PerExposureScalarBias,
    _crop_field_inputs_to_target,
    _field_target_observation,
    _robust_intensity_scale,
)
from bench_map_smacs0723_real import (  # noqa: E402
    DATA,
    DEFAULT_EXPOSURE_DIR,
    _approx_overlap_grid,
    _best_overlap_cutout,
    _center_cutout,
    _cutout_header,
    _discover_exposure_paths,
    _field_observation_kwargs_for_band,
    _filter_exposure_paths_by_patterns,
    _injection_manifest_tensors,
    _jwst_dq_support_and_contamination_prior,
    _metric_mask,
    _prepare_aux_files,
    _read_ext,
    _read_optional_ext,
    _reference_path_for_band,
    _sci_header,
)
from bench_multiband_depth import (  # noqa: E402
    gaussian_psf_contract,
    inject_common_world_exposures,
)
import fit_smacs0723_operator_calibration as calibration_fit  # noqa: E402

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
    fit_structured_noise_crossfit,
    make_field_model,
    render_field_observation,
    render_spectral_power_foam_observation_exact,
)
from torchsky.map.field import (  # noqa: E402
    solve_power_pixel_affine_least_squares,
    solve_power_pixel_least_squares,
    solve_spectral_cell_coefficients,
    solve_spectral_power_pixel_affine_least_squares,
    spectral_basis_whitening_transform,
)
from torchsky.map.field.bandpass import bandpass_quadrature  # noqa: E402
from torchsky.instrument import nircam_imaging_band  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    apply_learned_psf_spec_adjoint,
    apply_psf_spec,
    pixel_area_sr_at,
)
from torchsky.calibration_evidence import load_operator_calibration_split  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    ConditionedBackgroundField,
    LearnedBackgroundOp,
    LearnedPSFOp,
    LowRankChromaticPSF,
    ObservationOperatorBundle,
    StructuredDetectorNoise,
    StructuredNoiseOp,
    VAR_FLAT,
    VAR_POISSON,
    VAR_RNOISE,
    apply_operator_calibration_pruning,
    field_observation_operator_bundle,
    profile_amplifier_row_offsets,
)
from torchsky.map.field.types import PSFSpec  # noqa: E402


REAL_JOINT_CASE = "tpv_field_fit_deep_multiband_broad"
FOAM_FOOTPRINT_EXTENT = 0.55
FOAM_MAX_SMOOTH_PAIRS = 64 * 1024 * 1024
FOAM_TEMPERATURE_START_CELL_UNITS = 1.0
FOAM_TEMPERATURE_END_CELL_UNITS = 16.0


def _fitted_gaussian_psf_sigma_pix(
    band: str,
    header: dict[str, Any],
) -> float:
    """Convert the frozen fitted PSF FWHM to the local image pixel scale."""

    width = int(header.get("NAXIS1", 0))
    height = int(header.get("NAXIS2", 0))
    if width <= 0 or height <= 0:
        raise ValueError("PSF conversion requires positive NAXIS1/NAXIS2")
    wcs = torchsky.WCS(header)
    area_sr = pixel_area_sr_at(
        wcs,
        torch.tensor(0.5 * (width - 1), dtype=torch.float64),
        torch.tensor(0.5 * (height - 1), dtype=torch.float64),
    )
    pixel_scale_arcsec = float(torch.sqrt(area_sr) * (180.0 / torch.pi) * 3600.0)
    contract = gaussian_psf_contract(band)
    sigma_pix = contract.fitted_fwhm_arcsec / 2.354820045 / pixel_scale_arcsec
    if not math.isfinite(sigma_pix) or sigma_pix <= 0.0:
        raise ValueError("fitted PSF has a non-finite image-pixel width")
    return sigma_pix


def _parse_csv(value: str) -> list[str]:
    return [part.strip().upper() for part in str(value).split(",") if part.strip()]


def _safe_slug(value: str) -> str:
    text = str(value).strip().lower()
    return "".join(ch if ch.isalnum() else "_" for ch in text).strip("_") or "item"


def _calibration_exposure_lookup(payload: dict[str, Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for exposure in payload["exposures"]:
        lookup[str(exposure["path"])] = exposure
        lookup[str(exposure["resolved_path"])] = exposure
        lookup[str(exposure["filename"])] = exposure
    return lookup


def _calibration_exposure(
    lookup: dict[str, dict[str, Any]], path: Path
) -> dict[str, Any]:
    exposure = lookup.get(str(path)) or lookup.get(str(path.resolve()))
    exposure = exposure or lookup.get(path.name)
    if exposure is None:
        raise ValueError(f"calibration manifest does not contain {path}")
    return exposure


def _load_calibrated_operator_models(
    manifest: Path,
    fit_report: Path,
    *,
    detectors: set[str],
    intensity_scale: float,
) -> tuple[dict[str, dict[str, torch.nn.Module]], dict[str, Any]]:
    manifest = manifest.resolve()
    report = json.loads(fit_report.read_text(encoding="utf-8"))
    if report.get("manifest_sha256") != calibration_fit._sha256(manifest):
        raise ValueError("operator fit report does not match the calibration manifest")
    indexed = {
        (
            str(group["detector"]).upper(),
            str(group["role"]),
            str(group["family"]),
        ): group
        for group in report["groups"]
    }
    models: dict[str, dict[str, torch.nn.Module]] = {}
    audit = []
    for detector in sorted(detectors):
        models[detector] = {}
        for role in ("background", "psf"):
            group = indexed.get((detector, role, "adaptive"))
            if group is None:
                raise ValueError(f"fit report has no adaptive {detector}/{role} group")
            checkpoint_path = Path(group["checkpoint"])
            if calibration_fit._sha256(checkpoint_path) != group["checkpoint_sha256"]:
                raise ValueError(f"{detector}/{role} checkpoint hash mismatch")
            arrays = load_operator_calibration_split(
                manifest,
                split="train",
                role=role,
                detector=detector,
                verify_hashes=True,
            )
            checkpoint = torch.load(
                checkpoint_path, map_location="cpu", weights_only=True
            )
            model, _initializer = calibration_fit._make_model(
                arrays,
                role=role,
                family="adaptive",
                device=torch.device("cpu"),
                psf_model=str(checkpoint.get("psf_model") or "fitted_gaussian_kernel"),
            )
            if (
                checkpoint.get("schema_version") != 1
                or checkpoint.get("detector") != detector
                or checkpoint.get("role") != role
                or checkpoint.get("family") != "adaptive"
            ):
                raise ValueError(f"invalid {detector}/{role} checkpoint metadata")
            model.load_state_dict(checkpoint["state_dict"])
            if isinstance(
                getattr(model, "encoder", None),
                AdaptiveSpatioSpectralTemporalEncoder,
            ):
                pruning = apply_operator_calibration_pruning(
                    model, group["validation_pruning"]
                )
            else:
                pruning = {
                    "pruned_bands": [],
                    "note": "model has no adaptive gates to prune",
                }
            _freeze_calibrated_operator_model(
                model, role=role, intensity_scale=intensity_scale
            )
            models[detector][role] = model
            audit.append(
                {
                    "detector": detector,
                    "role": role,
                    "checkpoint": str(checkpoint_path),
                    "checkpoint_sha256": group["checkpoint_sha256"],
                    **pruning,
                }
            )
    return models, {
        "manifest": str(manifest),
        "manifest_sha256": report["manifest_sha256"],
        "fit_report": str(fit_report.resolve()),
        "intensity_scale": float(intensity_scale),
        "models": audit,
    }


def _freeze_calibrated_operator_model(
    model: torch.nn.Module, *, role: str, intensity_scale: float
) -> None:
    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0.0:
        raise ValueError("calibrated operator intensity_scale must be positive")
    if role == "background":
        if not isinstance(model, ConditionedBackgroundField):
            raise TypeError("background checkpoint produced the wrong model")
        output = model.network[-1]
        if not isinstance(output, torch.nn.Linear):
            raise TypeError("background model must end with a linear layer")
        with torch.no_grad():
            output.weight.div_(scale)
            output.bias.div_(scale)
    elif role != "psf":
        raise ValueError("calibrated operator role must be background or psf")
    model.requires_grad_(False)
    model.eval()


def _calibrated_operator_bundles(
    observations: list[FieldObservation],
    contexts: list[ForwardContext],
    models: dict[str, dict[str, torch.nn.Module]],
) -> tuple[ObservationOperatorBundle, ...]:
    if len(observations) != len(contexts):
        raise ValueError("calibrated observations and contexts must have equal length")
    bundles = []
    for observation, context in zip(observations, contexts, strict=True):
        detector = str(context.detector or "").upper()
        if detector not in models:
            raise ValueError(f"no calibrated operators for detector {detector!r}")
        bundles.append(
            field_observation_operator_bundle(observation).with_updates(
                psf=LearnedPSFOp(models[detector]["psf"]),
                background=LearnedBackgroundOp(
                    models[detector]["background"], coordinate_frame="detector"
                ),
            )
        )
    return tuple(bundles)


def _apply_learned_psf_arm(
    observations: list[FieldObservation],
    bands: list[str],
    *,
    grid_side: int = 4,
    spectral_degree: int = 2,
) -> dict[str, float]:
    """Swap fitted Gaussian PSF specs for shared per-band learned models.

    One :class:`LowRankChromaticPSF` per band (base width from the fitted
    Gaussian contract); every exposure evaluates it at its own detector
    control-cell centers. ponytail: in-band spectral coordinate is the
    band index normalized to [0, 1]; upgrade path is true bandpass-weighted
    central wavelengths.
    """
    models: dict[str, LowRankChromaticPSF] = {}
    # True in-band spectral coordinate: band pivot wavelengths normalized
    # across the run's band set (not a bare band index).
    pivots = {
        band: float(nircam_imaging_band(band).pivot_wavelength_angstrom)
        for band in bands
    }
    pivot_lo, pivot_hi = min(pivots.values()), max(pivots.values())
    for observation in observations:
        band = str(observation.band_name)
        if band not in bands:
            raise ValueError(f"observation band {band!r} not in {bands}")
        model = models.get(band)
        if model is None:
            model = LowRankChromaticPSF(
                grid_side=grid_side,
                spectral_degree=spectral_degree,
                base_sigma_pix=float(observation.psf_sigma_pix),
            )
            # Calibration-anchored nuisance: frozen during scene fitting so
            # renders through it build no autograd graph.
            model.requires_grad_(False)
            models[band] = model
        height, width = (int(v) for v in observation.image.shape[-2:])
        xs = torch.linspace(0.0, width - 1.0, grid_side) + float(
            observation.detector_x0_pix
        )
        ys = torch.linspace(0.0, height - 1.0, grid_side) + float(
            observation.detector_y0_pix
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        cell_u_pix = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
        lam_norm = (pivots[band] - pivot_lo) / max(pivot_hi - pivot_lo, 1e-9)
        lam_norm = min(max(lam_norm, 0.0), 1.0)
        observation.psf_spec = PSFSpec.learned(
            model,
            base_sigma_pix=float(observation.psf_sigma_pix),
            cell_u_pix=cell_u_pix,
            cell_lam_norm=lam_norm,
        )
    return {band: float(model.base_sigma_pix) for band, model in models.items()}


def _regular_foam_model_kwargs(
    *,
    grid_side: int,
    basis_count: int,
    affine_spatial_basis: bool = False,
    extent: float = 1.0,
    bounds_xy: tuple[tuple[float, float], tuple[float, float]] | None = None,
    log_intensity_spectra: bool = False,
    initial_intensity: float = 1.0e-2,
) -> dict[str, Any]:
    """Deterministic Power-Foam initialization independent of science pixels."""
    side = int(grid_side)
    basis = int(basis_count)
    limit = float(extent)
    if side < 2:
        raise ValueError("grid_side must be at least 2")
    if basis < 1:
        raise ValueError("basis_count must be positive")
    if not math.isfinite(limit) or limit <= 0:
        raise ValueError("extent must be finite and positive")
    if not math.isfinite(initial_intensity) or initial_intensity <= 0:
        raise ValueError("initial_intensity must be finite and positive")
    bounds = bounds_xy or ((-limit, limit), (-limit, limit))
    (x_min, x_max), (y_min, y_max) = bounds
    if not all(math.isfinite(value) for pair in bounds for value in pair):
        raise ValueError("bounds_xy must be finite")
    if x_min >= x_max or y_min >= y_max:
        raise ValueError("bounds_xy must have increasing x/y limits")
    step_x = (x_max - x_min) / float(side)
    step_y = (y_max - y_min) / float(side)
    centers_x = torch.linspace(
        x_min + 0.5 * step_x,
        x_max - 0.5 * step_x,
        steps=side,
        dtype=torch.float32,
    )
    centers_y = torch.linspace(
        y_min + 0.5 * step_y,
        y_max - 0.5 * step_y,
        steps=side,
        dtype=torch.float32,
    )
    yy, xx = torch.meshgrid(centers_y, centers_x, indexing="ij")
    sites = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    radii = torch.full(
        (side * side,),
        math.hypot(step_x, step_y),
        dtype=torch.float32,
    )
    coefficients = torch.zeros((side * side, basis), dtype=torch.float32)
    if log_intensity_spectra:
        coefficients[:, 0] = math.log(float(initial_intensity))
    inverse_cell_area = 1.0 / (step_x * step_y)
    result = {
        "sites": sites,
        "radii": radii,
        "cell_coefficients": coefficients,
        "minimum_radius": 0.25 * min(step_x, step_y),
        "temperature_start": FOAM_TEMPERATURE_START_CELL_UNITS * inverse_cell_area,
        "temperature_end": FOAM_TEMPERATURE_END_CELL_UNITS * inverse_cell_area,
        "log_intensity_spectra": bool(log_intensity_spectra),
    }
    if affine_spatial_basis:
        result["cell_gradient_coefficients"] = torch.zeros(
            (side * side, basis, 2), dtype=torch.float32
        )
    return result


def _foam_target_footprint_bounds(
    domain: Any,
    band_entries: list[dict[str, Any]],
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return target-WCS bounds in the Foam's normalized tangent plane."""
    if not band_entries:
        raise ValueError("band_entries must not be empty")
    boundary_world: list[torch.Tensor] = []
    for entry in band_entries:
        header = dict(entry["target_header"])
        width = int(header.get("NAXIS1", 0))
        height = int(header.get("NAXIS2", 0))
        if width <= 0 or height <= 0:
            raise ValueError("target WCS requires positive NAXIS1/NAXIS2")
        # ponytail: 17 samples per edge bound ordinary distorted imaging WCS;
        # upgrade to adaptive extrema only if a high-order WCS violates the box.
        x = torch.linspace(-0.5, width - 0.5, 17, dtype=torch.float64)
        y = torch.linspace(-0.5, height - 0.5, 17, dtype=torch.float64)
        boundary = torch.cat(
            (
                torch.stack((x, torch.full_like(x, -0.5)), dim=-1),
                torch.stack((torch.full_like(y, width - 0.5), y), dim=-1),
                torch.stack((x, torch.full_like(x, height - 0.5)), dim=-1),
                torch.stack((torch.full_like(y, -0.5), y), dim=-1),
            )
        )
        boundary_world.append(torchsky.WCS(header).pixel_to_world(boundary))
    world = torch.cat(boundary_world)
    delta_ra = (
        torch.remainder(
            world[:, 0] - float(domain.center_ra_deg) + 180.0,
            360.0,
        )
        - 180.0
    )
    x = (
        delta_ra
        * math.cos(math.radians(float(domain.center_dec_deg)))
        / float(domain.spatial_scale_deg)
    )
    y = (world[:, 1] - float(domain.center_dec_deg)) / float(domain.spatial_scale_deg)
    return (
        (float(torch.min(x)), float(torch.max(x))),
        (float(torch.min(y)), float(torch.max(y))),
    )


def _foam_training_tile_side(*, crop_size: int, grid_side: int) -> int:
    """Bound dense smooth memberships while increasing fixed-grid capacity."""
    crop = int(crop_size)
    side = int(grid_side)
    if crop <= 0:
        raise ValueError("crop_size must be positive")
    if side < 2:
        raise ValueError("grid_side must be at least 2")
    pair_limited = math.isqrt(FOAM_MAX_SMOOTH_PAIRS // (side * side))
    if pair_limited < 8:
        raise ValueError(
            "Foam grid exceeds the dense smooth-training capacity at the "
            f"{FOAM_MAX_SMOOTH_PAIRS:,}-pair limit"
        )
    return min(crop, 128, pair_limited)


def _set_foam_geometry_trainable(model: torch.nn.Module, trainable: bool) -> None:
    """Select fixed-grid or adaptive-geometry fitting without freezing appearance."""
    foam = getattr(model, "foam", model)
    sites = getattr(foam, "sites", None)
    raw_radii = getattr(foam, "raw_radii", None)
    if not isinstance(sites, torch.nn.Parameter) or not isinstance(
        raw_radii, torch.nn.Parameter
    ):
        raise TypeError("model must expose trainable Foam sites and raw_radii")
    sites.requires_grad_(bool(trainable))
    raw_radii.requires_grad_(bool(trainable))


def _initialize_foam_from_band_coadds(
    model: torch.nn.Module,
    band_entries: list[dict[str, Any]],
    *,
    intensity_scale: float,
    num_spectral_samples: int,
    operator_max_candidate_pairs: int = 10_000_000,
    whiten_spectral_basis: bool = False,
) -> dict[str, float]:
    """Warm-start fixed Foam appearance from exact observed pixel footprints."""
    first_parameter = next(model.parameters(), None)
    if first_parameter is None:
        raise TypeError("model must have parameters")
    # TORCHSKY_DEVICE=cpu opts out of CUDA entirely -- e.g. on MIG slices
    # where the caching allocator's NVML queries can assert (torch 2.10).
    requested_device = os.environ.get("TORCHSKY_DEVICE", "").strip().lower()
    if requested_device:
        initialization_device = torch.device(requested_device)
    else:
        initialization_device = torch.device(
            "cuda" if torch.cuda.is_available() else first_parameter.device
        )
    model.to(initialization_device)
    sites = getattr(model, "sites", None)
    coefficients = getattr(model, "cell_coefficients", None)
    gradient_coefficients = getattr(model, "cell_gradient_coefficients", None)
    spectral_basis = getattr(model, "spectral_basis", None)
    normalized_spatial_coordinates = getattr(
        model,
        "normalized_spatial_coordinates",
        None,
    )
    exact_operator = getattr(model, "exact_operator", None)
    if (
        not isinstance(sites, torch.Tensor)
        or not isinstance(coefficients, torch.nn.Parameter)
        or not callable(spectral_basis)
        or not callable(normalized_spatial_coordinates)
        or not callable(exact_operator)
    ):
        raise TypeError(
            "model must expose Foam sites, coefficients, basis, and operator"
        )
    scale = float(intensity_scale)
    if not math.isfinite(scale) or scale <= 0:
        raise ValueError("intensity_scale must be finite and positive")
    if not band_entries:
        raise ValueError("band_entries must not be empty")
    candidate_limit = int(operator_max_candidate_pairs)
    if candidate_limit <= 0:
        raise ValueError("operator_max_candidate_pairs must be positive")

    cell_values: list[torch.Tensor] = []
    cell_weights: list[torch.Tensor] = []
    response_basis: list[torch.Tensor] = []
    pixel_systems: list[
        tuple[Any, torch.Tensor, torch.Tensor, torch.Tensor, Any, torch.Tensor | None]
    ] = []
    for entry in band_entries:
        image = torch.as_tensor(
            entry["coadd_pred"],
            dtype=sites.dtype,
            device=sites.device,
        )
        support = torch.as_tensor(
            entry["support_mask"],
            dtype=torch.bool,
            device=sites.device,
        )
        if image.ndim != 2 or support.shape != image.shape:
            raise ValueError("coadd image and support mask must share 2-D shape")
        valid = support & torch.isfinite(image)
        height, width = image.shape
        grid_device = (
            torch.device("cpu") if sites.device.type == "mps" else sites.device
        )
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=grid_device),
            torch.arange(width, dtype=torch.float64, device=grid_device),
            indexing="ij",
        )
        offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float64,
            device=grid_device,
        )
        wcs = torchsky.WCS(dict(entry["target_header"]))
        wcs.to(grid_device)
        ra_deg, dec_deg = wcs.pixel_to_world(
            x[..., None] + offsets[:, 0],
            y[..., None] + offsets[:, 1],
        )
        polygons = normalized_spatial_coordinates(
            ra_deg.to(device=sites.device),
            dec_deg.to(device=sites.device),
        ).reshape(height * width, 4, 2)
        operator = exact_operator(
            polygons,
            max_candidate_pairs=candidate_limit,
        )
        next_vertices = torch.roll(polygons, shifts=-1, dims=-2)
        pixel_areas = 0.5 * torch.abs(
            torch.sum(
                polygons[..., 0] * next_vertices[..., 1]
                - polygons[..., 1] * next_vertices[..., 0],
                dim=-1,
            )
        )
        if bool(torch.any(pixel_areas <= 0)):
            raise ValueError("target WCS produced a degenerate pixel polygon")
        normalized = torch.where(
            valid,
            image / scale,
            torch.zeros_like(image),
        ).reshape(-1)
        pixel_weight = valid.to(dtype=sites.dtype).reshape(-1)
        integrated = normalized * pixel_areas
        psf_spec = entry["target_observation"].psf_spec
        if getattr(psf_spec, "mode", "none") not in (
            "none",
            "gaussian_sigma_pix",
            "learned_lowrank",
        ):
            raise ValueError("Foam initialization requires a symmetric target PSF")

        def psf_forward(
            values: torch.Tensor,
            areas: torch.Tensor = pixel_areas,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            with torch.no_grad():
                surface_brightness = values / areas
                return (
                    apply_psf_spec(
                        surface_brightness.reshape(shape),
                        spec,
                    ).reshape(values.shape)
                    * areas
                )

        def psf_adjoint(
            values: torch.Tensor,
            areas: torch.Tensor = pixel_areas,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            with torch.no_grad():
                if getattr(spec, "mode", "none") == "learned_lowrank":
                    core = apply_learned_psf_spec_adjoint(
                        (values * areas).reshape(shape),
                        spec,
                    )
                else:
                    core = apply_psf_spec(
                        (values * areas).reshape(shape),
                        spec,
                    )
                return core.reshape(values.shape) / areas

        solved_gradients = None
        if isinstance(gradient_coefficients, torch.nn.Parameter):
            solved_cells, solved_gradients = solve_power_pixel_affine_least_squares(
                operator,
                integrated,
                pixel_weight,
                ridge=1.0e-6,
                max_iterations=50,
                tolerance=1.0e-5,
                pixel_matvec=psf_forward,
                pixel_rmatvec=psf_adjoint,
            )
        else:
            solved_cells = solve_power_pixel_least_squares(
                operator,
                integrated,
                pixel_weight,
                ridge=1.0e-6,
                max_iterations=50,
                tolerance=1.0e-5,
                pixel_matvec=psf_forward,
                pixel_rmatvec=psf_adjoint,
            )
        coverage = torch.clamp(
            operator.rmatvec(pixel_weight, semantics="integrated_flux"),
            min=0.0,
            max=1.0,
        )
        cell_values.append(solved_cells)
        cell_weights.append(coverage)
        pixel_systems.append(
            (
                operator,
                pixel_areas,
                normalized,
                pixel_weight,
                psf_forward,
                solved_gradients,
            )
        )

        wavelengths, response = bandpass_quadrature(
            entry["target_observation"],
            num_spectral_samples=int(num_spectral_samples),
            device=sites.device,
        )
        basis_samples = spectral_basis(wavelengths).to(dtype=sites.dtype)
        response_basis.append(
            torch.sum(response.to(dtype=sites.dtype)[:, None] * basis_samples, dim=0)
        )

    values = torch.stack(cell_values)
    weights = torch.stack(cell_weights)
    basis = torch.stack(response_basis)
    raw_basis_condition = torch.linalg.cond(basis)
    raw_basis_rank = torch.linalg.matrix_rank(basis)
    if whiten_spectral_basis:
        set_basis_transform = getattr(model, "set_spectral_basis_transform", None)
        if not callable(set_basis_transform):
            raise TypeError("model must expose set_spectral_basis_transform")
        transform = spectral_basis_whitening_transform(basis)
        set_basis_transform(transform)
        basis = basis @ transform
    solved = solve_spectral_cell_coefficients(basis, values, weights)
    with torch.no_grad():
        coefficients.copy_(solved)
    solved_gradient_coefficients = None
    if isinstance(gradient_coefficients, torch.nn.Parameter):
        band_gradients = torch.stack(
            [
                gradient
                for _, _, _, _, _, gradient in pixel_systems
                if gradient is not None
            ]
        )
        solved_gradient_coefficients = torch.stack(
            (
                solve_spectral_cell_coefficients(
                    basis, band_gradients[..., 0], weights
                ),
                solve_spectral_cell_coefficients(
                    basis, band_gradients[..., 1], weights
                ),
            ),
            dim=-1,
        )
        with torch.no_grad():
            gradient_coefficients.copy_(solved_gradient_coefficients)
    reconstructed = basis @ solved.mT
    reconstructed_gradients = (
        torch.einsum("bk,ckd->bcd", basis, solved_gradient_coefficients)
        if solved_gradient_coefficients is not None
        else None
    )
    initial_cell_variations: list[torch.Tensor] = []
    residual_square_sum = sites.new_zeros(())
    observed_count = sites.new_zeros(())
    for band_index, (band_cells, pixel_system) in enumerate(
        zip(reconstructed, pixel_systems, strict=True)
    ):
        operator, pixel_areas, normalized, pixel_weight, psf_forward, _ = pixel_system
        integrated_prediction = (
            operator.matvec_affine(
                band_cells,
                reconstructed_gradients[band_index],
            )
            if reconstructed_gradients is not None
            else operator.matvec(band_cells)
        )
        if reconstructed_gradients is not None:
            observed_cells = weights[band_index] > 0
            cell_variation = torch.sqrt(operator.cell_areas) * torch.linalg.vector_norm(
                reconstructed_gradients[band_index],
                dim=-1,
            )
            observed_variation = cell_variation[observed_cells]
            if observed_variation.numel():
                initial_cell_variations.append(observed_variation)
        prediction = psf_forward(integrated_prediction) / pixel_areas
        residual_square_sum += torch.sum(
            pixel_weight * torch.square(prediction - normalized)
        )
        observed_count += torch.sum(pixel_weight)
    residual_rmse = torch.sqrt(
        residual_square_sum / torch.clamp(observed_count, min=1.0)
    )
    coverage_count = torch.sum(weights > 0, dim=0)
    variation = (
        torch.cat(initial_cell_variations)
        if initial_cell_variations
        else sites.new_zeros((1,))
    )
    return {
        "foam_initialized_cell_fraction": float(
            torch.mean((coverage_count > 0).float())
        ),
        "foam_fully_covered_cell_fraction": float(
            torch.mean((coverage_count == len(band_entries)).float())
        ),
        "foam_initialization_rmse": float(residual_rmse),
        "foam_initial_coefficient_abs_max": float(torch.max(torch.abs(solved))),
        "foam_initial_gradient_coefficient_abs_max": (
            float(torch.max(torch.abs(solved_gradient_coefficients)))
            if solved_gradient_coefficients is not None
            else 0.0
        ),
        "foam_spectral_basis_condition_number": float(torch.linalg.cond(basis)),
        "foam_spectral_basis_raw_condition_number": float(raw_basis_condition),
        "foam_spectral_basis_rank": float(raw_basis_rank),
        "foam_initial_cell_variation_abs_p95": float(torch.quantile(variation, 0.95)),
        "foam_initial_cell_variation_abs_max": float(torch.max(variation)),
    }


def _refine_foam_from_observations_exact(
    model: torch.nn.Module,
    observations: list[FieldObservation],
    *,
    num_spectral_samples: int,
    operator_max_candidate_pairs: int = 10_000_000,
    max_iterations: int = 50,
    operator_storage_device: str = "model",
    observation_batch_size: int | str | None = None,
    preconditioner: str = "auto",
) -> dict[str, float]:
    """Refit fixed spectral-affine appearance through native detector pixels."""
    coefficients = getattr(model, "cell_coefficients", None)
    gradient_coefficients = getattr(model, "cell_gradient_coefficients", None)
    spectral_basis = getattr(model, "spectral_basis", None)
    normalized_coordinates = getattr(model, "normalized_spatial_coordinates", None)
    exact_operator = getattr(model, "exact_operator", None)
    if (
        not isinstance(coefficients, torch.nn.Parameter)
        or not isinstance(gradient_coefficients, torch.nn.Parameter)
        or not callable(spectral_basis)
        or not callable(normalized_coordinates)
        or not callable(exact_operator)
    ):
        raise TypeError("exact observation refinement requires affine spectral Foam")
    if not observations:
        raise ValueError("observations must not be empty")
    device = coefficients.device
    dtype = coefficients.dtype
    storage_device = (
        device
        if str(operator_storage_device).strip().lower() == "model"
        else torch.device(operator_storage_device)
    )
    operators = []
    response_basis = []
    integrated_values = []
    integrated_weights = []
    pixel_areas_by_observation = []
    forwards = []
    adjoints = []
    observed_pixel_count = 0
    for observation in observations:
        image = observation.image.to(device=device, dtype=dtype)
        height, width = image.shape
        grid_device = torch.device("cpu") if device.type == "mps" else device
        y, x = torch.meshgrid(
            torch.arange(height, dtype=torch.float64, device=grid_device),
            torch.arange(width, dtype=torch.float64, device=grid_device),
            indexing="ij",
        )
        offsets = torch.tensor(
            [[-0.5, -0.5], [0.5, -0.5], [0.5, 0.5], [-0.5, 0.5]],
            dtype=torch.float64,
            device=grid_device,
        )
        wcs = torchsky.WCS(dict(observation.header))
        wcs.to(grid_device)
        ra_deg, dec_deg = wcs.pixel_to_world(
            x[..., None] + offsets[:, 0],
            y[..., None] + offsets[:, 1],
        )
        polygons = normalized_coordinates(
            ra_deg.to(device=device), dec_deg.to(device=device)
        ).reshape(height * width, 4, 2)
        operator = exact_operator(
            polygons,
            max_candidate_pairs=int(operator_max_candidate_pairs),
        )
        next_vertices = torch.roll(polygons, shifts=-1, dims=-2)
        pixel_areas = 0.5 * torch.abs(
            torch.sum(
                polygons[..., 0] * next_vertices[..., 1]
                - polygons[..., 1] * next_vertices[..., 0],
                dim=-1,
            )
        )
        if bool(torch.any(pixel_areas <= 0)):
            raise ValueError("observation WCS produced a degenerate pixel polygon")
        mask = (
            observation.mask.to(device=device, dtype=torch.bool)
            if observation.mask is not None
            else torch.ones_like(image, dtype=torch.bool)
        )
        surface_weight = (
            observation.weight.to(device=device, dtype=dtype)
            if observation.weight is not None
            else torch.ones_like(image)
        )
        valid = (
            mask
            & torch.isfinite(image)
            & torch.isfinite(surface_weight)
            & (surface_weight > 0)
        ).reshape(-1)
        area = pixel_areas.reshape(-1)
        stored_area = area.to(storage_device)
        integrated_values.append(
            torch.where(valid, image.reshape(-1) * area, torch.zeros_like(area)).to(
                storage_device
            )
        )
        integrated_weights.append(
            torch.where(
                valid,
                surface_weight.reshape(-1) / area.square(),
                torch.zeros_like(area),
            ).to(storage_device)
        )
        pixel_areas_by_observation.append(stored_area)
        observed_pixel_count += int(torch.sum(valid))
        psf_spec = observation.psf_spec
        if getattr(psf_spec, "mode", "none") not in (
            "none",
            "gaussian_sigma_pix",
            "learned_lowrank",
        ):
            raise ValueError("exact observation refinement requires a symmetric PSF")

        def psf_forward(
            values: torch.Tensor,
            areas: torch.Tensor = stored_area,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            areas = areas.to(device=values.device)
            with torch.no_grad():
                return (
                    apply_psf_spec((values / areas).reshape(shape), spec).reshape(-1)
                    * areas
                )

        def psf_adjoint(
            values: torch.Tensor,
            areas: torch.Tensor = stored_area,
            spec: Any = psf_spec,
            shape: tuple[int, int] = (height, width),
        ) -> torch.Tensor:
            areas = areas.to(device=values.device)
            with torch.no_grad():
                if getattr(spec, "mode", "none") == "learned_lowrank":
                    core = apply_learned_psf_spec_adjoint(
                        (values * areas).reshape(shape), spec
                    )
                else:
                    core = apply_psf_spec((values * areas).reshape(shape), spec)
                return core.reshape(-1) / areas

        wavelengths, response = bandpass_quadrature(
            observation,
            num_spectral_samples=int(num_spectral_samples),
            device=device,
        )
        basis_samples = spectral_basis(wavelengths).to(dtype=dtype)
        response_basis.append(
            torch.sum(response.to(dtype=dtype)[:, None] * basis_samples, dim=0)
            .to(storage_device)
        )
        operators.append(operator.to(storage_device))
        forwards.append(psf_forward)
        adjoints.append(psf_adjoint)

    initial_coefficients = coefficients.detach().clone()
    initial_gradients = gradient_coefficients.detach().clone()
    solve_diagnostics: dict[str, float] = {}
    solved, solved_gradients = solve_spectral_power_pixel_affine_least_squares(
        operators,
        torch.stack(response_basis),
        integrated_values,
        integrated_weights,
        ridge=1.0e-6,
        max_iterations=int(max_iterations),
        tolerance=1.0e-5,
        pixel_matvecs=forwards,
        pixel_rmatvecs=adjoints,
        initial_coefficients=initial_coefficients,
        initial_gradient_coefficients=initial_gradients,
        solve_device=device,
        observation_batch_size=observation_batch_size,
        preconditioner=preconditioner,
        diagnostics=solve_diagnostics,
    )
    with torch.no_grad():
        coefficients.copy_(solved)
        gradient_coefficients.copy_(solved_gradients)

    weighted_square_sum = coefficients.new_zeros(())
    initial_weighted_square_sum = coefficients.new_zeros(())
    weight_sum = coefficients.new_zeros(())
    for operator, basis, values, weight, area, forward in zip(
        operators,
        response_basis,
        integrated_values,
        integrated_weights,
        pixel_areas_by_observation,
        forwards,
        strict=True,
    ):
        solve_operator = operator.to(device)
        solve_basis = basis.to(device)
        solve_area = area.to(device)
        solve_values = values.to(device)
        solve_weight = weight.to(device)
        prediction = forward(
            solve_operator.matvec_affine(
                solved @ solve_basis,
                torch.einsum("ckd,k->cd", solved_gradients, solve_basis),
            )
        )
        initial_prediction = forward(
            solve_operator.matvec_affine(
                initial_coefficients @ solve_basis,
                torch.einsum("ckd,k->cd", initial_gradients, solve_basis),
            )
        )
        surface_weight = solve_weight * solve_area.square()
        weighted_square_sum += torch.sum(
            surface_weight * torch.square((prediction - solve_values) / solve_area)
        )
        initial_weighted_square_sum += torch.sum(
            surface_weight
            * torch.square((initial_prediction - solve_values) / solve_area)
        )
        weight_sum += torch.sum(surface_weight)
    return {
        "foam_exact_refinement_initial_rmse": float(
            torch.sqrt(
                initial_weighted_square_sum / torch.clamp(weight_sum, min=1.0)
            )
        ),
        "foam_exact_refinement_rmse": float(
            torch.sqrt(weighted_square_sum / torch.clamp(weight_sum, min=1.0))
        ),
        "foam_exact_refinement_observations": float(len(observations)),
        "foam_exact_refinement_pixels": float(observed_pixel_count),
        "foam_exact_refinement_max_iterations": float(max_iterations),
        "foam_exact_refinement_iterations": solve_diagnostics["iterations"],
        "foam_exact_refinement_normal_operator_applications": solve_diagnostics[
            "normal_operator_applications"
        ],
        "foam_exact_refinement_initial_relative_residual": solve_diagnostics[
            "initial_relative_residual"
        ],
        "foam_exact_refinement_ridge_centered_on_initial": solve_diagnostics[
            "ridge_centered_on_initial"
        ],
        "foam_exact_refinement_relative_residual": solve_diagnostics[
            "relative_residual"
        ],
        "foam_exact_refinement_converged": solve_diagnostics["converged"],
        "foam_exact_refinement_preconditioner_setup_seconds": solve_diagnostics[
            "preconditioner_setup_seconds"
        ],
        "foam_exact_refinement_rhs_seconds": solve_diagnostics["rhs_seconds"],
        "foam_exact_refinement_initial_residual_seconds": solve_diagnostics[
            "initial_residual_seconds"
        ],
        "foam_exact_refinement_pcg_seconds": solve_diagnostics["pcg_seconds"],
        "foam_exact_refinement_preconditioner_cell_block": solve_diagnostics[
            "preconditioner_cell_block"
        ],
        "foam_exact_refinement_preconditioner_block_fallback_cells": (
            solve_diagnostics["preconditioner_block_fallback_cells"]
        ),
        "foam_exact_refinement_preconditioner_storage_bytes": solve_diagnostics[
            "preconditioner_storage_bytes"
        ],
        "foam_exact_refinement_preconditioner_auto": solve_diagnostics[
            "preconditioner_auto"
        ],
        "foam_exact_refinement_preconditioner_response_condition": (
            solve_diagnostics["preconditioner_response_condition"]
        ),
        "foam_exact_refinement_preconditioner_auto_condition_threshold": (
            solve_diagnostics["preconditioner_auto_condition_threshold"]
        ),
        "foam_exact_refinement_preconditioner_auto_budget_bytes": solve_diagnostics[
            "preconditioner_auto_budget_bytes"
        ],
        "foam_exact_refinement_preconditioner_block_construction_peak_bytes": (
            solve_diagnostics["preconditioner_block_construction_peak_bytes"]
        ),
        "foam_exact_refinement_observation_batch_size": float(
            solve_diagnostics["observation_batch_size"]
        ),
        "foam_exact_refinement_observation_batch_auto": solve_diagnostics[
            "observation_batch_auto"
        ],
        "foam_exact_refinement_observation_batch_transfer_budget_bytes": (
            solve_diagnostics["observation_batch_transfer_budget_bytes"]
        ),
        "foam_exact_refinement_max_observation_transfer_bytes": (
            solve_diagnostics["max_observation_transfer_bytes"]
        ),
        "foam_exact_refinement_operator_storage_bytes": float(
            solve_diagnostics["operator_storage_bytes"]
        ),
        "foam_exact_refinement_peak_operator_batch_bytes": solve_diagnostics[
            "peak_operator_batch_bytes"
        ],
        "foam_exact_refinement_host_offload": float(storage_device.type == "cpu"),
    }


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
    psf_mode: str = "fitted_gaussian",
    header: dict[str, Any],
    reference: np.ndarray,
    prediction: np.ndarray,
    support_mask: np.ndarray,
    source_mask: np.ndarray,
) -> None:
    model_dir = (
        export_dir / _safe_slug(psf_mode) / _safe_slug(band) / _safe_slug(model)
    )
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
    Image.fromarray(_preview_uint8(reference), mode="L").save(
        model_dir / "reference.png"
    )
    Image.fromarray(_preview_uint8(prediction), mode="L").save(
        model_dir / "prediction.png"
    )
    Image.fromarray(_preview_uint8(residual), mode="L").save(model_dir / "residual.png")
    Image.fromarray(support_mask.astype(np.uint8) * 255, mode="L").save(
        model_dir / "support_mask.png"
    )
    Image.fromarray(source_mask.astype(np.uint8) * 255, mode="L").save(
        model_dir / "source_mask.png"
    )


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
    foam_grid_side: int = 16,
    foam_lr: float = 2.0e-2,
    foam_fixed_geometry: bool = False,
) -> dict[str, Any]:
    kwargs = _field_fit_kwargs_for_model(
        model_kind,
        case=REAL_JOINT_CASE,
        size=int(crop_size),
        benchmark_steps=int(steps),
    )
    if model_kind == "coadd_bilinear":
        return kwargs
    if model_kind == "spectral_power_foam":
        foam_learning_rate = float(foam_lr)
        if not math.isfinite(foam_learning_rate) or foam_learning_rate <= 0.0:
            raise ValueError("foam_lr must be finite and positive")
        tile_side = (
            min(int(crop_size), 128)
            if foam_fixed_geometry
            else _foam_training_tile_side(
                crop_size=int(crop_size),
                grid_side=int(foam_grid_side),
            )
        )
        kwargs["lr"] = foam_learning_rate
        kwargs["curriculum"] = "feature_scale_ramp"
        kwargs["num_spectral_samples"] = 5
        kwargs["training_spectral_sampling"] = "deterministic_quadrature"
        kwargs["spectral_anchor_observations_per_step"] = 0
        kwargs["spectral_anchor_loss_weight"] = 0.0
        kwargs["training_patch_size"] = tile_side
        kwargs["streaming_accumulation"] = True
        kwargs["diagnostic_spatial_tile_hw"] = (min(tile_side, 64),) * 2
    kwargs["observations_per_step"] = min(max(num_observations, 1), 12)
    kwargs["observation_sampling"] = "band_balanced"
    if model_kind != "spectral_power_foam":
        kwargs["training_patch_size"] = None
    kwargs["pixel_samples_per_observation"] = max(
        256, min(int(crop_size) * int(crop_size), 1024)
    )
    if model_kind == "spectral_power_foam":
        kwargs["pixel_samples_per_observation"] = None
        kwargs["shared_sky_patch_sampling"] = True
    kwargs["loss_kind"] = "mse"
    kwargs["loss_reduction"] = "global"
    kwargs["spatial_sampling"] = "gauss_legendre_2x2"
    kwargs["grad_clip_norm"] = 1.0
    if model_kind != "spectral_power_foam":
        kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 1.0e-3)
    kwargs["lr_warmup_steps"] = max(int(kwargs.get("lr_warmup_steps", 0)), 4)
    kwargs["context_observations_per_step"] = min(
        int(kwargs.get("context_observations_per_step", 0)),
        1,
    )
    kwargs["full_observations_per_step"] = 0
    if model_kind in {"spectral_basis_field", "sky_encoder_resmlp", "sky_encoder_mlp"}:
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 9)), 5
        )
    if model_kind == "sky_lambda_feature_field":
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 9)), 5
        )
        kwargs["spectral_anchor_observations_per_step"] = min(
            int(kwargs.get("spectral_anchor_observations_per_step", 0)),
            1,
        )
    if model_kind == "operator_constrained_feature_field":
        kwargs["num_spectral_samples"] = min(
            int(kwargs.get("num_spectral_samples", 7)), 5
        )
        kwargs["context_loss_weight"] = min(
            float(kwargs.get("context_loss_weight", 0.0)), 0.05
        )
    if model_kind == "spectral_basis_field":
        kwargs["lr"] = min(float(kwargs.get("lr", 1.0e-3)), 5.0e-4)
    return kwargs


def _structured_scene_fit_kwargs(
    base: dict[str, Any], *, num_observations: int
) -> dict[str, Any]:
    """Make a full-exposure fit compatible with the structured likelihood."""
    kwargs = dict(base)
    kwargs.update(
        loss_kind="structured_nll",
        loss_reduction="global",
        training_spectral_sampling="deterministic_quadrature",
        observations_per_step=max(int(num_observations), 1),
        training_patch_size=None,
        pixel_samples_per_observation=None,
        shared_sky_patch_sampling=False,
        diagnostic_spatial_tile_hw=None,
        gradient_residual_weight=0.0,
        spectral_anchor_observations_per_step=0,
        component_anchor_terms=None,
        component_centroid_anchor_terms=None,
        context_observations_per_step=0,
    )
    return kwargs


def _cross_fitted_scene_predictions(
    observations: Sequence[FieldObservation],
    *,
    model_factory: Callable[[], torch.nn.Module],
    steps: int,
    fit_kwargs: dict[str, Any],
    render_kwargs: dict[str, Any],
    seed: int,
) -> tuple[tuple[torch.Tensor, ...], tuple[dict[str, float | int], ...]]:
    """Fit one scene per held-out exposure and render only that exposure."""
    if len(observations) < 2:
        raise ValueError("cross-fitting requires at least two exposures")
    if int(steps) < 1:
        raise ValueError("cross-fit scene steps must be positive")
    predictions: list[torch.Tensor] = []
    diagnostics: list[dict[str, float | int]] = []
    for held_out_index, held_out in enumerate(observations):
        training = tuple(
            observation
            for index, observation in enumerate(observations)
            if index != held_out_index
        )
        torch.manual_seed(int(seed) + held_out_index)
        result = fit_field_model(
            training,
            model_factory(),
            steps=int(steps),
            **fit_kwargs,
        )
        result.model.eval()
        with torch.no_grad():
            prediction = render_field_observation(
                result.model,
                held_out,
                **render_kwargs,
            )
        if prediction.shape != held_out.image.shape:
            raise RuntimeError("held-out render shape does not match exposure")
        predictions.append(prediction.detach().cpu())
        diagnostics.append(
            {
                "held_out_index": held_out_index,
                "training_exposures": len(training),
                "initial_loss": float(result.initial_loss),
                "final_loss": float(result.final_loss),
            }
        )
    return tuple(predictions), tuple(diagnostics)


def _structured_noise_bundles(
    observations: Sequence[FieldObservation],
    models: Sequence[StructuredDetectorNoise],
) -> tuple[ObservationOperatorBundle, ...]:
    if len(observations) != len(models):
        raise ValueError("noise models must match observations")
    return tuple(
        field_observation_operator_bundle(observation).with_updates(
            noise=StructuredNoiseOp(model, frozen=True)
        )
        for observation, model in zip(observations, models, strict=True)
    )


@torch.no_grad()
def _structured_noise_diagnostics(
    observations: Sequence[FieldObservation],
    predictions: Sequence[torch.Tensor],
    models: Sequence[StructuredDetectorNoise],
    losses: Sequence[float],
) -> list[dict[str, Any]]:
    if not (
        len(observations) == len(predictions) == len(models) == len(losses)
    ):
        raise ValueError("noise diagnostics inputs must have equal length")
    rows: list[dict[str, Any]] = []
    for index, (observation, prediction, model, loss) in enumerate(
        zip(observations, predictions, models, losses, strict=True)
    ):
        calibration = model.variance_calibration
        row_noise = model.row_noise
        residual = prediction.to(dtype=torch.float32) - observation.image.to(
            dtype=torch.float32
        )
        support = torch.isfinite(residual)
        components: dict[str, torch.Tensor] = {}
        for key, value in (
            (VAR_POISSON, observation.var_poisson),
            (VAR_RNOISE, observation.var_rnoise),
            (VAR_FLAT, observation.var_flat),
        ):
            if value is None:
                continue
            plane = value.to(dtype=residual.dtype)
            plane_valid = torch.isfinite(plane) & (plane >= 0.0)
            support &= plane_valid
            components[key] = torch.where(
                plane_valid, plane, torch.zeros_like(plane)
            )
        err = (
            None
            if observation.err is None
            else observation.err.to(dtype=residual.dtype)
        )
        if not components and err is None:
            if observation.weight is None:
                raise ValueError("noise diagnostics require variance, ERR, or weight")
            weight = observation.weight.to(dtype=residual.dtype)
            err = torch.where(
                weight > 0.0,
                weight.clamp_min(torch.finfo(weight.dtype).tiny).rsqrt(),
                torch.ones_like(weight),
            )
        elif not components and err is not None:
            err_valid = torch.isfinite(err) & (err >= 0.0)
            support &= err_valid
            err = torch.where(err_valid, err, torch.ones_like(err))
        variance = calibration(components or None, err=err).to(dtype=residual.dtype)
        if variance.shape != residual.shape:
            variance = torch.broadcast_to(variance, residual.shape)
        support &= torch.isfinite(variance) & (variance > 0)
        if observation.mask is not None:
            support &= observation.mask.to(dtype=torch.bool)
        safe_residual = torch.where(support, residual, torch.zeros_like(residual))
        amplifier_index = (
            model.default_amplifier_index(tuple(residual.shape), residual.device)
            if observation.amplifier_index is None
            else observation.amplifier_index.to(device=residual.device)
        )
        _, corrected = profile_amplifier_row_offsets(
            safe_residual,
            variance,
            amplifier_index,
            valid=support,
            num_amplifiers=model.num_amplifiers,
        )
        inlier_probability, _ = model.contamination_responsibilities(
            safe_residual,
            components=components or None,
            err=err,
            valid=support,
            amplifier_index=amplifier_index,
            contamination_prior=observation.contamination_prior,
        )
        z_abs = torch.abs(corrected) / torch.sqrt(variance)
        inlier_weight = torch.where(
            support, inlier_probability, torch.zeros_like(inlier_probability)
        )
        inlier_weight_sum = inlier_weight.sum().clamp_min(
            torch.finfo(inlier_weight.dtype).tiny
        )
        active_amplifiers = torch.unique(amplifier_index[support]).detach().cpu()
        rows.append(
            {
                "exposure_index": index,
                "crossfit_nll": float(loss),
                "poisson_scale": float(calibration.poisson_scale.detach().cpu()),
                "rnoise_scale": float(calibration.rnoise_scale.detach().cpu()),
                "flat_scale": float(calibration.flat_scale.detach().cpu()),
                "err_scale": float(calibration.err_scale.detach().cpu()),
                "variance_floor": float(
                    calibration.variance_floor.detach().cpu()
                ),
                "row_white_psd": row_noise.white_psd.detach().cpu().tolist(),
                "row_knee_frequency": (
                    row_noise.knee_frequency.detach().cpu().tolist()
                ),
                "row_exponent": row_noise.exponent.detach().cpu().tolist(),
                "outlier_probability": float(
                    model.outlier_probability.detach().cpu()
                ),
                "outlier_scale": float(model.outlier_scale.detach().cpu()),
                "active_amplifiers": active_amplifiers.tolist(),
                "valid_pixels": int(support.sum()),
                "posterior_weighted_coverage_68": float(
                    torch.sum(inlier_weight * (z_abs <= 1.0)) / inlier_weight_sum
                ),
                "posterior_weighted_coverage_95": float(
                    torch.sum(inlier_weight * (z_abs <= 1.959963984540054))
                    / inlier_weight_sum
                ),
            }
        )
    return rows


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
                "mean_support_rmse": float(
                    np.mean([float(r["support_rmse"]) for r in model_rows])
                ),
                "mean_source_rmse": float(
                    np.mean([float(r["source_rmse"]) for r in model_rows])
                ),
                "mean_source_contrast_rmse": float(
                    np.mean([float(r["source_contrast_rmse"]) for r in model_rows])
                ),
                "mean_uniform_rmse": float(
                    np.mean([float(r["uniform_rmse"]) for r in model_rows])
                ),
                "mean_aligned_rmse": float(
                    np.mean([float(r["aligned_rmse"]) for r in model_rows])
                ),
                "mean_affine_slope": float(
                    np.mean([float(r["affine_slope"]) for r in model_rows])
                ),
                "mean_source_contrast_slope": float(
                    np.mean([float(r["source_contrast_slope"]) for r in model_rows])
                ),
                "mean_support_flux_ratio": float(
                    np.mean([float(r["support_flux_ratio"]) for r in model_rows])
                ),
                "mean_source_flux_ratio": float(
                    np.mean([float(r["source_flux_ratio"]) for r in model_rows])
                ),
                "total_fit_time_ms": float(
                    np.max([float(r["fit_time_ms"]) for r in model_rows])
                ),
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
    exact_rows: dict[str, dict[str, Any]] = {}
    for row in ok_rows:
        if "foam_exact_refinement_rmse" in row:
            exact_rows.setdefault(str(row["model"]), row)
    if exact_rows:
        lines.extend(
            [
                "",
                "## Exact Refinement",
                "",
                "| model | initial rmse | final rmse | iterations | residual | setup s | rhs s | pcg s | peak operator MiB | ridge center |",
                "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- |",
            ]
        )
        for model, row in exact_rows.items():
            peak_mib = float(row["foam_exact_refinement_peak_operator_batch_bytes"]) / 2**20
            ridge_center = (
                "initial"
                if float(row["foam_exact_refinement_ridge_centered_on_initial"])
                else "zero"
            )
            lines.append(
                f"| {model} | {row['foam_exact_refinement_initial_rmse']:.6f} | "
                f"{row['foam_exact_refinement_rmse']:.6f} | "
                f"{row['foam_exact_refinement_iterations']:.0f} | "
                f"{row['foam_exact_refinement_relative_residual']:.3e} | "
                f"{row['foam_exact_refinement_preconditioner_setup_seconds']:.3f} | "
                f"{row['foam_exact_refinement_rhs_seconds']:.3f} | "
                f"{row['foam_exact_refinement_pcg_seconds']:.3f} | "
                f"{peak_mib:.2f} | {ridge_center} |"
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
    parser.add_argument(
        "--exposure-dir",
        type=Path,
        default=DEFAULT_EXPOSURE_DIR,
        help="Directory containing native CRF/CAL products (aligned only for explicit comparator runs)",
    )
    parser.add_argument(
        "--input-product",
        choices=("auto", "crf", "cal", "aligned"),
        default="auto",
        help="Native training defaults to CRF with per-exposure CAL fallback",
    )
    parser.add_argument("--crop-size", type=int, default=32)
    parser.add_argument(
        "--crop-strategy", choices=("center", "best_overlap"), default="best_overlap"
    )
    parser.add_argument("--crop-x0", type=int, default=None)
    parser.add_argument("--crop-y0", type=int, default=None)
    parser.add_argument(
        "--target-header-path",
        type=Path,
        default=None,
        help="FITS image supplying one exact target WCS for all output bands",
    )
    parser.add_argument(
        "--injection-manifest",
        type=Path,
        default=None,
        help="Frozen common-world detector-level injection manifest",
    )
    parser.add_argument(
        "--patt-nums",
        default="",
        help="Comma-separated PATT_NUM identifiers, for example 3,8 for validation",
    )
    parser.add_argument("--steps", type=int, default=4)
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--foam-grid-side", type=int, default=32)
    parser.add_argument(
        "--foam-exact-max-candidate-pairs",
        type=int,
        default=10_000_000,
        help="Explicit allocation ceiling for exact Foam sparse geometry",
    )
    parser.add_argument("--foam-basis-count", type=int, default=4)
    parser.add_argument(
        "--foam-affine-spatial-basis",
        action="store_true",
        help="Learn cell-local x/y slopes and integrate them by exact first moments",
    )
    parser.add_argument("--foam-lr", type=float, default=2.0e-2)
    parser.add_argument(
        "--foam-log-intensity-spectra",
        action="store_true",
        help="Parameterize each Foam cell spectrum as exp(Legendre modes)",
    )
    parser.add_argument("--foam-initial-intensity", type=float, default=1.0e-2)
    parser.add_argument(
        "--foam-train-geometry",
        action="store_true",
        help="Train sites and radii; off by default for the fixed-grid control",
    )
    parser.add_argument(
        "--foam-initialize-from-coadd",
        action="store_true",
        help="Warm-start fixed cell spectra from observed per-band coadds",
    )
    parser.add_argument(
        "--foam-fit-target-footprint",
        action="store_true",
        help="Place the fixed grid over target-WCS bounds instead of the full domain",
    )
    parser.add_argument(
        "--foam-whiten-spectral-basis",
        action="store_true",
        help=(
            "Orthonormalize passband-integrated basis columns before fitting; "
            "requires at least as many bands as basis modes"
        ),
    )
    parser.add_argument(
        "--foam-exact-observation-refine",
        action="store_true",
        help=(
            "Finish fixed affine Foam with a joint matrix-free solve through "
            "native detector WCS pixels, weights, bandpasses, and symmetric PSFs"
        ),
    )
    parser.add_argument(
        "--foam-exact-operator-storage",
        choices=("model", "cpu"),
        default="model",
        help="Store all exact operators with the model or offload them to CPU",
    )
    parser.add_argument(
        "--foam-exact-observation-batch-size",
        default="auto",
        help="Maximum exact observations resident on solve device; auto or positive integer",
    )
    parser.add_argument(
        "--foam-exact-max-iterations",
        type=int,
        default=50,
        help="Maximum PCG iterations for exact detector-pixel refinement",
    )
    parser.add_argument(
        "--foam-exact-preconditioner",
        choices=("diagonal", "cell_block", "auto"),
        default="auto",
        help="Exact-solve preconditioner; auto applies a device-memory guard",
    )
    parser.add_argument(
        "--foam-extent",
        type=float,
        default=FOAM_FOOTPRINT_EXTENT,
        help=(
            "Half-width of the deterministic normalized Foam grid. The field "
            "domain maps the exposure-footprint union inside +/-0.5."
        ),
    )
    parser.add_argument(
        "--models",
        default="coadd_bilinear,spectral_basis_field,sky_encoder_resmlp,sky_lambda_feature_field,operator_constrained_feature_field",
    )
    parser.add_argument("--skip-missing", action="store_true")
    parser.add_argument(
        "--fail-on-method-error",
        action="store_true",
        help="Exit nonzero when any requested model fails or is missing",
    )
    parser.add_argument("--json-out", type=Path, default=DEFAULT_JSON_OUT)
    parser.add_argument("--markdown-out", type=Path, default=DEFAULT_MARKDOWN_OUT)
    parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    parser.add_argument(
        "--operator-calibration-manifest",
        type=Path,
        default=None,
        help="135-CRF calibration evidence manifest for frozen learned operators",
    )
    parser.add_argument(
        "--operator-calibration-fit-report",
        type=Path,
        default=None,
        help="Fit report containing adaptive checkpoints and validation pruning",
    )
    parser.add_argument(
        "--forward-context",
        action="store_true",
        help=(
            "Per-exposure ForwardContext for fit_field_model; wraps INR with "
            "_PerExposureScalarBias (see bench_map_smacs0723_field_fit.py)."
        ),
    )
    parser.add_argument(
        "--structured-noise-crossfit",
        action="store_true",
        help=(
            "Calibrate one frozen structured-noise model per exposure from "
            "leave-one-exposure-out Foam predictions, then use structured NLL"
        ),
    )
    parser.add_argument("--crossfit-scene-steps", type=int, default=4)
    parser.add_argument("--structured-noise-steps", type=int, default=50)
    parser.add_argument("--structured-noise-lr", type=float, default=1.0e-2)
    parser.add_argument(
        "--foam-initialized-inr",
        action="store_true",
        help="Fit a positive zero-correction INR initialized from the fitted Foam",
    )
    parser.add_argument("--inr-steps", type=int, default=None)
    parser.add_argument("--inr-lr", type=float, default=1.0e-3)
    parser.add_argument("--inr-hidden-dim", type=int, default=64)
    parser.add_argument(
        "--tile-ra",
        type=float,
        default=None,
        help="COSMOS-scale tiling skeleton: tile center R.A. [deg]. "
        "Recorded in outputs this pass; per-tile FieldDomain partitioning "
        "is a later run.",
    )
    parser.add_argument(
        "--tile-dec",
        type=float,
        default=None,
        help="COSMOS-scale tiling skeleton: tile center Dec [deg].",
    )
    parser.add_argument(
        "--tile-size-deg",
        type=float,
        default=0.25,
        help="COSMOS-scale tiling skeleton: tile edge length [deg].",
    )
    parser.add_argument("--inr-hidden-layers", type=int, default=3)
    parser.add_argument(
        "--psf-model",
        choices=("fitted_gaussian", "learned_lowrank"),
        default="fitted_gaussian",
        help="Operator arm: frozen fitted Gaussian (default) or learned "
        "low-rank chromatic PSF anchored on calibration evidence.",
    )
    parser.add_argument(
        "--background-model",
        choices=("plane", "learned"),
        default="plane",
        help="Nuisance arm: scalar/plane background (default) or the learned "
        "ConditionedBackgroundField from the operator calibration fit.",
    )
    args = parser.parse_args()
    if args.foam_initialize_from_coadd and args.foam_train_geometry:
        parser.error("coadd initialization currently requires fixed Foam geometry")
    if args.foam_whiten_spectral_basis and not args.foam_initialize_from_coadd:
        parser.error(
            "--foam-whiten-spectral-basis requires --foam-initialize-from-coadd"
        )
    if args.foam_exact_observation_refine and (
        args.foam_train_geometry or not args.foam_affine_spatial_basis
    ):
        parser.error(
            "--foam-exact-observation-refine requires fixed affine Foam geometry"
        )
    if args.foam_log_intensity_spectra and (
        args.foam_affine_spatial_basis
        or args.foam_initialize_from_coadd
        or args.foam_exact_observation_refine
    ):
        parser.error(
            "log-intensity Foam currently excludes affine spectra, coadd initialization, "
            "and the linear exact refiner"
        )
    exact_batch: int | str
    if str(args.foam_exact_observation_batch_size).strip().lower() == "auto":
        exact_batch = "auto"
    else:
        try:
            exact_batch = int(args.foam_exact_observation_batch_size)
        except ValueError:
            parser.error(
                "--foam-exact-observation-batch-size must be auto or a positive integer"
            )
        if exact_batch <= 0:
            parser.error(
                "--foam-exact-observation-batch-size must be auto or a positive integer"
            )
    if args.foam_exact_max_iterations <= 0:
        parser.error("--foam-exact-max-iterations must be positive")
    if args.foam_exact_max_candidate_pairs <= 0:
        parser.error("--foam-exact-max-candidate-pairs must be positive")
    if (args.operator_calibration_manifest is None) != (
        args.operator_calibration_fit_report is None
    ):
        parser.error(
            "--operator-calibration-manifest and "
            "--operator-calibration-fit-report must be provided together"
        )
    if args.structured_noise_crossfit:
        if args.crossfit_scene_steps <= 0 or args.structured_noise_steps <= 0:
            parser.error("cross-fit scene and structured-noise steps must be positive")
        if not math.isfinite(args.structured_noise_lr) or args.structured_noise_lr <= 0:
            parser.error("--structured-noise-lr must be finite and positive")
        if args.forward_context or args.operator_calibration_manifest is not None:
            parser.error(
                "structured-noise cross-fitting does not yet support conditioned operators"
            )
        if args.foam_initialize_from_coadd:
            parser.error(
                "structured-noise cross-fitting forbids full-data coadd initialization"
            )
        if args.foam_exact_observation_refine:
            parser.error(
                "structured-noise cross-fitting cannot be followed by the MSE exact refiner"
            )
    if args.foam_initialized_inr:
        if args.inr_steps is not None and args.inr_steps <= 0:
            parser.error("--inr-steps must be positive")
        if not math.isfinite(args.inr_lr) or args.inr_lr <= 0:
            parser.error("--inr-lr must be finite and positive")
        if int(args.inr_hidden_dim) <= 0 or int(args.inr_hidden_layers) <= 0:
            parser.error("INR hidden dimensions must be positive")
        if not args.foam_log_intensity_spectra:
            parser.error(
                "--foam-initialized-inr requires --foam-log-intensity-spectra"
            )
    if (args.tile_ra is None) != (args.tile_dec is None):
        parser.error("--tile-ra and --tile-dec must be given together")
    if args.tile_ra is not None:
        for name, value in (
            ("--tile-ra", args.tile_ra),
            ("--tile-dec", args.tile_dec),
        ):
            if not math.isfinite(value) or not -180.0 <= value <= 360.0:
                parser.error(f"{name} must be a finite angle in [-180, 360] deg")
        if (
            not math.isfinite(args.tile_size_deg)
            or args.tile_size_deg <= 0.0
            or args.tile_size_deg > 2.0
        ):
            parser.error("--tile-size-deg must be in (0, 2] deg")
    tile_center = (
        None
        if args.tile_ra is None
        else (float(args.tile_ra), float(args.tile_dec))
    )
    calibrated_operators = args.operator_calibration_manifest is not None
    if args.background_model == "learned" and not calibrated_operators:
        parser.error(
            "--background-model learned requires --operator-calibration-manifest"
        )
    if (
        not math.isfinite(args.foam_initial_intensity)
        or args.foam_initial_intensity <= 0
    ):
        parser.error("--foam-initial-intensity must be finite and positive")

    bands = _parse_csv(args.bands)
    if args.foam_whiten_spectral_basis and args.foam_basis_count > len(bands):
        parser.error(
            "--foam-whiten-spectral-basis requires at least as many bands as modes"
        )
    pattern_numbers = tuple(int(value) for value in _parse_csv(args.patt_nums))
    model_kinds = [part.strip() for part in str(args.models).split(",") if part.strip()]
    if (args.structured_noise_crossfit or args.foam_initialized_inr) and (
        "spectral_power_foam" not in model_kinds
    ):
        parser.error("structured noise and Foam-initialized INR require spectral_power_foam")
    if args.foam_initialized_inr and "foam_initialized_inr" not in model_kinds:
        foam_index = model_kinds.index("spectral_power_foam")
        model_kinds.insert(foam_index + 1, "foam_initialized_inr")
    rows: list[dict[str, Any]] = []

    band_entries: list[dict[str, Any]] = []
    all_raw_field_inputs: list[
        tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
    ] = []
    all_field_observations: list[FieldObservation] = []
    all_forward_contexts: list[ForwardContext] = []
    injection_rows: list[dict[str, Any]] = []
    injection_manifest = (
        json.loads(args.injection_manifest.read_text(encoding="utf-8"))
        if args.injection_manifest is not None
        else None
    )
    calibration_manifest_payload = (
        json.loads(args.operator_calibration_manifest.read_text(encoding="utf-8"))
        if args.operator_calibration_manifest is not None
        else None
    )
    calibration_lookup = (
        _calibration_exposure_lookup(calibration_manifest_payload)
        if calibration_manifest_payload is not None
        else {}
    )
    explicit_target_header = (
        dict(torchsky.read(str(args.target_header_path), return_header=True)[1])
        if args.target_header_path is not None
        else None
    )
    if (args.crop_x0 is None) != (args.crop_y0 is None):
        raise ValueError("--crop-x0 and --crop-y0 must be provided together")
    if explicit_target_header is not None and (
        int(explicit_target_header.get("NAXIS1", -1)) != int(args.crop_size)
        or int(explicit_target_header.get("NAXIS2", -1)) != int(args.crop_size)
    ):
        raise ValueError("explicit target header shape must match --crop-size")

    with tempfile.TemporaryDirectory(prefix="torchsky_smacs_joint_") as tmp:
        tmpdir = Path(tmp)
        for band in bands:
            reference_path = _reference_path_for_band(band)
            exposure_paths = _discover_exposure_paths(
                args.exposure_dir,
                input_product=args.input_product,
                band=band,
                module=args.module,
                filename_contains="nrcblong"
                if band in {"F277W", "F356W", "F444W"}
                else None,
            )
            if pattern_numbers:
                exposure_paths = _filter_exposure_paths_by_patterns(
                    exposure_paths,
                    pattern_numbers,
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
            if args.crop_x0 is not None:
                x0 = int(args.crop_x0)
                y0 = int(args.crop_y0)
                if (
                    x0 < 0
                    or y0 < 0
                    or x0 + int(args.crop_size) > ref_data.shape[1]
                    or y0 + int(args.crop_size) > ref_data.shape[0]
                ):
                    raise ValueError("fixed crop lies outside the reference image")
                ref_crop = ref_data[
                    y0 : y0 + int(args.crop_size),
                    x0 : x0 + int(args.crop_size),
                ]
            elif args.crop_strategy == "best_overlap":
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
                x0 = min(
                    int(coarse_x0 * coarse_step),
                    max(0, ref_data.shape[1] - int(args.crop_size)),
                )
                y0 = min(
                    int(coarse_y0 * coarse_step),
                    max(0, ref_data.shape[0] - int(args.crop_size)),
                )
                ref_crop = ref_data[
                    y0 : y0 + int(args.crop_size), x0 : x0 + int(args.crop_size)
                ]
            else:
                ref_crop, x0, y0 = _center_cutout(ref_data, int(args.crop_size))
            target_header = (
                dict(explicit_target_header)
                if explicit_target_header is not None
                else _cutout_header(ref_header, x0, y0, int(args.crop_size))
            )
            psf_contract = gaussian_psf_contract(band)
            target_psf_sigma_pix = _fitted_gaussian_psf_sigma_pix(
                band,
                target_header,
            )
            target_observation = _field_target_observation(
                target_header,
                band_name=band,
                psf_sigma_pix=target_psf_sigma_pix,
            )
            science_overrides = None
            if injection_manifest is not None:
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
                injection_rows.extend({"band": band, **row} for row in injection_audit)

            image_observations: list[ImageObservation] = []
            raw_field_inputs: list[
                tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]
            ] = []
            raw_noise_inputs: dict[
                Path,
                tuple[
                    np.ndarray,
                    np.ndarray | None,
                    np.ndarray | None,
                    np.ndarray | None,
                    np.ndarray,
                ],
            ] = {}
            for exposure_path in exposure_paths:
                science_override = (
                    science_overrides.get(exposure_path)
                    if science_overrides is not None
                    else None
                )
                science_path, weight_path, mask_path = _prepare_aux_files(
                    exposure_path,
                    tmpdir,
                    linearize_sip=False,
                    science_override=science_override,
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
                science = (
                    _read_ext(exposure_path, "SCI")
                    if science_override is None
                    else np.asarray(science_override)
                )
                err = _read_ext(exposure_path, "ERR")
                dq = _read_ext(exposure_path, "DQ")
                dq_support, contamination_prior = (
                    _jwst_dq_support_and_contamination_prior(dq)
                )
                valid = np.isfinite(err) & (err > 0) & dq_support
                weight = np.zeros_like(err, dtype=np.float32)
                weight[valid] = 1.0 / np.square(err[valid])
                raw_noise_inputs[exposure_path] = (
                    err,
                    _read_optional_ext(exposure_path, "VAR_POISSON"),
                    _read_optional_ext(exposure_path, "VAR_RNOISE"),
                    _read_optional_ext(exposure_path, "VAR_FLAT"),
                    contamination_prior,
                )
                raw_field_inputs.append(
                    (
                        science,
                        weight,
                        valid.astype(np.float32, copy=False),
                        dict(_sci_header(exposure_path)),
                    )
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
                calibration_exposure = (
                    _calibration_exposure(calibration_lookup, exposure_path)
                    if calibrated_operators
                    else None
                )
                detector_x0 = float(crop_header["SUBSTRT1"]) - 1.0
                detector_y0 = float(crop_header["SUBSTRT2"]) - 1.0
                source_x0 = int(crop_header["SUBSTRT1"]) - int(
                    header.get("SUBSTRT1", 1)
                )
                source_y0 = int(crop_header["SUBSTRT2"]) - int(
                    header.get("SUBSTRT2", 1)
                )
                if calibration_exposure is not None:
                    detector_x0 += float(
                        calibration_exposure["detector_bounds_xy"][0]
                    ) - (float(header.get("SUBSTRT1", 1)) - 1.0)
                    detector_y0 += float(
                        calibration_exposure["detector_bounds_xy"][2]
                    ) - (float(header.get("SUBSTRT2", 1)) - 1.0)
                    crop_header["SUBSTRT1"] = int(detector_x0) + 1
                    crop_header["SUBSTRT2"] = int(detector_y0) + 1
                source_slice = (
                    slice(source_y0, source_y0 + crop_image.shape[0]),
                    slice(source_x0, source_x0 + crop_image.shape[1]),
                )
                err, var_poisson, var_rnoise, var_flat, contamination_prior = (
                    raw_noise_inputs[exposure_path]
                )
                amplifier_columns = torch.clamp(
                    (
                        torch.arange(crop_image.shape[1], dtype=torch.int64)
                        + int(detector_x0)
                    )
                    // 512,
                    min=0,
                    max=3,
                )
                field_observations.append(
                    FieldObservation(
                        image=torch.from_numpy(crop_image),
                        header=crop_header,
                        weight=torch.from_numpy(crop_weight),
                        mask=torch.from_numpy(crop_mask),
                        err=torch.from_numpy(err[source_slice].copy()),
                        var_poisson=(
                            None
                            if var_poisson is None
                            else torch.from_numpy(var_poisson[source_slice].copy())
                        ),
                        var_rnoise=(
                            None
                            if var_rnoise is None
                            else torch.from_numpy(var_rnoise[source_slice].copy())
                        ),
                        var_flat=(
                            None
                            if var_flat is None
                            else torch.from_numpy(var_flat[source_slice].copy())
                        ),
                        contamination_prior=torch.from_numpy(
                            contamination_prior[source_slice].copy()
                        ),
                        amplifier_index=amplifier_columns[None, :].expand(
                            crop_image.shape[0], crop_image.shape[1]
                        ),
                        detector_x0_pix=detector_x0,
                        detector_y0_pix=detector_y0,
                        psf_sigma_pix=_fitted_gaussian_psf_sigma_pix(
                            band,
                            crop_header,
                        ),
                        **_field_observation_kwargs_for_band(band),
                    )
                )
                all_forward_contexts.append(
                    ForwardContext(
                        survey=str(band),
                        instrument="NIRCAM" if calibration_exposure else None,
                        detector=(
                            str(calibration_exposure["detector"])
                            if calibration_exposure
                            else None
                        ),
                        exposure_id=f"{band}_{exposure_path.stem}",
                        time_mjd=(
                            float(calibration_exposure["timing"]["midpoint_mjd_utc"])
                            if calibration_exposure
                            else None
                        ),
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
            coadd_data = (
                coadd_reference.data.cpu().numpy().astype(np.float32, copy=False)
            )
            support_mask = coadd_weight > 0
            if not np.any(support_mask):
                support_mask = np.isfinite(ref_crop) & np.isfinite(coadd_data)

            band_entries.append(
                {
                    "band": band,
                    "reference_crop": (
                        coadd_data
                        if explicit_target_header is not None
                        else ref_crop.astype(np.float32, copy=False)
                    ),
                    "target_header": dict(target_header),
                    "target_observation": target_observation,
                    "truth_psf_fwhm_arcsec": psf_contract.truth_fwhm_arcsec,
                    "fitted_psf_fwhm_arcsec": psf_contract.fitted_fwhm_arcsec,
                    "input_psf_sigma_pix_min": min(
                        observation.psf_sigma_pix for observation in field_observations
                    ),
                    "input_psf_sigma_pix_max": max(
                        observation.psf_sigma_pix for observation in field_observations
                    ),
                    "target_psf_sigma_pix": target_psf_sigma_pix,
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
                    mask=observation.mask.clone()
                    if observation.mask is not None
                    else None,
                    err=(
                        observation.err / intensity_scale
                        if observation.err is not None
                        else None
                    ),
                    var_poisson=(
                        observation.var_poisson / (intensity_scale * intensity_scale)
                        if observation.var_poisson is not None
                        else None
                    ),
                    var_rnoise=(
                        observation.var_rnoise / (intensity_scale * intensity_scale)
                        if observation.var_rnoise is not None
                        else None
                    ),
                    var_flat=(
                        observation.var_flat / (intensity_scale * intensity_scale)
                        if observation.var_flat is not None
                        else None
                    ),
                    contamination_prior=observation.contamination_prior,
                    amplifier_index=observation.amplifier_index,
                    wavelength=observation.wavelength,
                    spectral_min=observation.spectral_min,
                    spectral_max=observation.spectral_max,
                    transmission_center=observation.transmission_center,
                    transmission_sigma=observation.transmission_sigma,
                    transmission_wavelength=observation.transmission_wavelength,
                    transmission_throughput=observation.transmission_throughput,
                    psf_sigma_pix=observation.psf_sigma_pix,
                    psf_spec=observation.psf_spec,
                    band_name=observation.band_name,
                    spectral_quantity=observation.spectral_quantity,
                    detector_x0_pix=observation.detector_x0_pix,
                    detector_y0_pix=observation.detector_y0_pix,
                    detector_to_sky=observation.detector_to_sky,
                )
            )

        psf_arm_audit: dict[str, float] | None = None
        if args.psf_model == "learned_lowrank":
            psf_arm_audit = _apply_learned_psf_arm(normalized_observations, bands)
        spherical_domain = estimate_field_domain(
            normalized_observations,
            coord_mode="sphere_xyz",
        )
        local_domain = estimate_field_domain(
            normalized_observations,
            coord_mode="local_tangent",
        )
        forward_ctx_tuple: tuple[ForwardContext | None, ...] | None = None
        if args.forward_context or calibrated_operators:
            if len(all_forward_contexts) != len(normalized_observations):
                raise RuntimeError(
                    "internal: joint forward context list length mismatch"
                )
            forward_ctx_tuple = tuple(all_forward_contexts)

        calibrated_bundles: tuple[ObservationOperatorBundle, ...] | None = None
        calibration_audit: dict[str, Any] | None = None
        if calibrated_operators:
            assert args.operator_calibration_manifest is not None
            assert args.operator_calibration_fit_report is not None
            detectors = {
                str(context.detector).upper() for context in all_forward_contexts
            }
            calibrated_models, calibration_audit = _load_calibrated_operator_models(
                args.operator_calibration_manifest,
                args.operator_calibration_fit_report,
                detectors=detectors,
                intensity_scale=float(intensity_scale),
            )
            calibrated_bundles = _calibrated_operator_bundles(
                normalized_observations, all_forward_contexts, calibrated_models
            )
            if args.background_model == "plane":
                # Plane arm: drop the learned background component, keep the
                # calibrated PSF; the scalar observation background stays.
                calibrated_bundles = tuple(
                    bundle.with_updates(background=None)
                    for bundle in calibrated_bundles
                )

        foam_bounds = (
            _foam_target_footprint_bounds(local_domain, band_entries)
            if args.foam_fit_target_footprint
            else None
        )
        foam_model_kwargs = _regular_foam_model_kwargs(
            grid_side=int(args.foam_grid_side),
            basis_count=int(args.foam_basis_count),
            affine_spatial_basis=bool(args.foam_affine_spatial_basis),
            extent=float(args.foam_extent),
            bounds_xy=foam_bounds,
            log_intensity_spectra=bool(args.foam_log_intensity_spectra),
            initial_intensity=float(args.foam_initial_intensity),
        )
        foam_model_kwargs["compact_fixed_geometry"] = not bool(
            args.foam_train_geometry
        )

        def make_foam_model() -> torch.nn.Module:
            model = make_field_model(
                "spectral_power_foam", local_domain, **foam_model_kwargs
            )
            _set_foam_geometry_trainable(model, args.foam_train_geometry)
            return model

        structured_bundles: tuple[ObservationOperatorBundle, ...] | None = None
        noise_crossfit_audit: dict[str, Any] | None = None
        if args.structured_noise_crossfit:
            crossfit_kwargs = _joint_real_fit_kwargs(
                "spectral_power_foam",
                crop_size=int(args.crop_size),
                steps=int(args.crossfit_scene_steps),
                num_observations=max(len(normalized_observations) - 1, 1),
                foam_grid_side=int(args.foam_grid_side),
                foam_lr=float(args.foam_lr),
                foam_fixed_geometry=not bool(args.foam_train_geometry),
            )
            crossfit_predictions, crossfit_diagnostics = (
                _cross_fitted_scene_predictions(
                    normalized_observations,
                    model_factory=make_foam_model,
                    steps=int(args.crossfit_scene_steps),
                    fit_kwargs=crossfit_kwargs,
                    render_kwargs={
                        "num_spectral_samples": int(
                            crossfit_kwargs.get("num_spectral_samples", 1)
                        ),
                        "spatial_sampling": str(
                            crossfit_kwargs.get("spatial_sampling", "center")
                        ),
                    },
                    seed=int(args.seed),
                )
            )
            noise_models, crossfit_noise_losses = fit_structured_noise_crossfit(
                normalized_observations,
                crossfit_predictions,
                steps=int(args.structured_noise_steps),
                lr=float(args.structured_noise_lr),
                device="cpu",
            )
            structured_bundles = _structured_noise_bundles(
                normalized_observations, noise_models
            )
            noise_crossfit_audit = {
                "scene_folds": list(crossfit_diagnostics),
                "noise_models": _structured_noise_diagnostics(
                    normalized_observations,
                    crossfit_predictions,
                    noise_models,
                    crossfit_noise_losses,
                ),
            }

        fitted_models: dict[
            str,
            tuple[Any, float, dict[str, Any], dict[str, Any]],
        ] = {}
        for model_kind in model_kinds:
            if model_kind == "coadd_bilinear":
                continue
            if model_kind == "spectral_power_foam":
                model_kwargs = foam_model_kwargs
                model_steps = int(args.steps)
            elif model_kind == "foam_initialized_inr":
                foam_reference = fitted_models["spectral_power_foam"][0]
                foam_reference = getattr(foam_reference, "inner", foam_reference)
                model_kwargs = {
                    "foam": foam_reference,
                    "hidden_dim": int(args.inr_hidden_dim),
                    "hidden_layers": int(args.inr_hidden_layers),
                }
                model_steps = int(args.inr_steps or args.steps)
            else:
                model_kwargs = _field_model_kwargs_for_case(
                    model_kind,
                    case=REAL_JOINT_CASE,
                    size=int(args.crop_size),
                )
                model_steps = int(args.steps)
            fit_kwargs = _joint_real_fit_kwargs(
                model_kind,
                crop_size=int(args.crop_size),
                steps=model_steps,
                num_observations=len(normalized_observations),
                foam_grid_side=int(args.foam_grid_side),
                foam_lr=float(args.foam_lr),
                foam_fixed_geometry=not bool(args.foam_train_geometry),
            )
            if forward_ctx_tuple is not None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["observation_forward_contexts"] = forward_ctx_tuple
            if calibrated_bundles is not None:
                fit_kwargs = dict(fit_kwargs)
                fit_kwargs["observation_operator_bundles"] = calibrated_bundles
            if structured_bundles is not None:
                fit_kwargs = _structured_scene_fit_kwargs(
                    fit_kwargs, num_observations=len(normalized_observations)
                )
                fit_kwargs["observation_operator_bundles"] = structured_bundles
            if model_kind == "foam_initialized_inr":
                fit_kwargs["lr"] = float(args.inr_lr)
            domain = (
                local_domain
                if model_kind in {"spectral_power_foam", "foam_initialized_inr"}
                else spherical_domain
            )
            torch.manual_seed(int(args.seed))
            model = (
                make_foam_model()
                if model_kind == "spectral_power_foam"
                else make_field_model(model_kind, domain, **model_kwargs)
            )
            initialization_diagnostics: dict[str, float] = {}
            if model_kind == "spectral_power_foam":
                if args.foam_initialize_from_coadd:
                    initialization_diagnostics = _initialize_foam_from_band_coadds(
                        model,
                        band_entries,
                        intensity_scale=float(intensity_scale),
                        num_spectral_samples=int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        ),
                        operator_max_candidate_pairs=int(
                            args.foam_exact_max_candidate_pairs
                        ),
                        whiten_spectral_basis=bool(args.foam_whiten_spectral_basis),
                    )
            if args.forward_context:
                model = _PerExposureScalarBias(model, len(normalized_observations))
            t0 = time.perf_counter()
            result = fit_field_model(
                normalized_observations,
                model,
                steps=model_steps,
                **fit_kwargs,
            )
            fit_time_ms = (time.perf_counter() - t0) * 1000.0
            exact_refinement_diagnostics: dict[str, float] = {}
            if model_kind == "spectral_power_foam" and args.foam_exact_observation_refine:
                if calibrated_bundles is not None or args.forward_context:
                    raise ValueError(
                        "exact Foam observation refinement does not yet support "
                        "conditioned operators or per-exposure bias"
                    )
                hard_model = getattr(result.model, "inner", result.model)
                refine_t0 = time.perf_counter()
                exact_refinement_diagnostics = _refine_foam_from_observations_exact(
                    hard_model,
                    normalized_observations,
                    num_spectral_samples=int(
                        fit_kwargs.get("num_spectral_samples", 1)
                    ),
                    operator_max_candidate_pairs=int(
                        args.foam_exact_max_candidate_pairs
                    ),
                    operator_storage_device=str(args.foam_exact_operator_storage),
                    observation_batch_size=exact_batch,
                    max_iterations=int(args.foam_exact_max_iterations),
                    preconditioner=str(args.foam_exact_preconditioner),
                )
                fit_time_ms += (time.perf_counter() - refine_t0) * 1000.0
            fitted_models[model_kind] = (
                result.model,
                fit_time_ms,
                fit_kwargs,
                {
                    **dict(result.diagnostics or {}),
                    "initial_loss": float(result.initial_loss),
                    "final_loss": float(result.final_loss),
                    "last_training_loss": float(result.loss_history[-1]),
                    **initialization_diagnostics,
                    **exact_refinement_diagnostics,
                },
            )

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
                    "truth_psf_fwhm_arcsec": float(entry["truth_psf_fwhm_arcsec"]),
                    "fitted_psf_fwhm_arcsec": float(entry["fitted_psf_fwhm_arcsec"]),
                    "input_psf_sigma_pix_min": float(entry["input_psf_sigma_pix_min"]),
                    "input_psf_sigma_pix_max": float(entry["input_psf_sigma_pix_max"]),
                    "target_psf_sigma_pix": float(entry["target_psf_sigma_pix"]),
                    "forward_context": bool(args.forward_context),
                    "calibrated_operators": bool(
                        calibrated_operators and model_kind != "coadd_bilinear"
                    ),
                    "structured_noise_crossfit": bool(
                        args.structured_noise_crossfit
                        and model_kind != "coadd_bilinear"
                    ),
                    "optimizer_steps": (
                        int(args.inr_steps or args.steps)
                        if model_kind == "foam_initialized_inr"
                        else int(args.steps)
                        if model_kind != "coadd_bilinear"
                        else 0
                    ),
                    "status": "OK",
                }
                try:
                    if model_kind == "coadd_bilinear":
                        pred = entry["coadd_pred"]
                        fit_time_ms = 0.0
                    else:
                        (
                            result_model,
                            fit_time_ms,
                            fit_kwargs,
                            fit_diagnostics,
                        ) = fitted_models[model_kind]
                        row["parameter_count"] = int(
                            sum(
                                parameter.numel()
                                for parameter in result_model.parameters()
                            )
                        )
                        row["trainable_parameter_count"] = int(
                            sum(
                                parameter.numel()
                                for parameter in result_model.parameters()
                                if parameter.requires_grad
                            )
                        )
                        row["spectral_samples"] = int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        )
                        row["spatial_sampling"] = str(
                            fit_kwargs.get("spatial_sampling", "center")
                        )
                        row["intensity_scale"] = float(intensity_scale)
                        row["initial_loss"] = float(fit_diagnostics["initial_loss"])
                        row["final_loss"] = float(fit_diagnostics["final_loss"])
                        row["last_training_loss"] = float(
                            fit_diagnostics["last_training_loss"]
                        )
                        row["foam_initialized_from_coadd"] = bool(
                            model_kind == "spectral_power_foam"
                            and args.foam_initialize_from_coadd
                        )
                        for diagnostic_name in (
                            "foam_initialized_cell_fraction",
                            "foam_fully_covered_cell_fraction",
                            "foam_initialization_rmse",
                            "foam_initial_coefficient_abs_max",
                            "foam_initial_gradient_coefficient_abs_max",
                            "foam_spectral_basis_condition_number",
                            "foam_spectral_basis_raw_condition_number",
                            "foam_spectral_basis_rank",
                            "foam_initial_cell_variation_abs_p95",
                            "foam_initial_cell_variation_abs_max",
                            "foam_exact_refinement_rmse",
                            "foam_exact_refinement_initial_rmse",
                            "foam_exact_refinement_observations",
                            "foam_exact_refinement_pixels",
                            "foam_exact_refinement_max_iterations",
                            "foam_exact_refinement_iterations",
                            "foam_exact_refinement_normal_operator_applications",
                            "foam_exact_refinement_initial_relative_residual",
                            "foam_exact_refinement_ridge_centered_on_initial",
                            "foam_exact_refinement_relative_residual",
                            "foam_exact_refinement_converged",
                            "foam_exact_refinement_preconditioner_setup_seconds",
                            "foam_exact_refinement_rhs_seconds",
                            "foam_exact_refinement_initial_residual_seconds",
                            "foam_exact_refinement_pcg_seconds",
                            "foam_exact_refinement_preconditioner_cell_block",
                            "foam_exact_refinement_preconditioner_block_fallback_cells",
                            "foam_exact_refinement_preconditioner_storage_bytes",
                            "foam_exact_refinement_preconditioner_auto",
                            "foam_exact_refinement_preconditioner_response_condition",
                            "foam_exact_refinement_preconditioner_auto_condition_threshold",
                            "foam_exact_refinement_preconditioner_auto_budget_bytes",
                            "foam_exact_refinement_preconditioner_block_construction_peak_bytes",
                            "foam_exact_refinement_observation_batch_size",
                            "foam_exact_refinement_observation_batch_auto",
                            "foam_exact_refinement_observation_batch_transfer_budget_bytes",
                            "foam_exact_refinement_max_observation_transfer_bytes",
                            "foam_exact_refinement_operator_storage_bytes",
                            "foam_exact_refinement_peak_operator_batch_bytes",
                            "foam_exact_refinement_host_offload",
                        ):
                            if diagnostic_name in fit_diagnostics:
                                row[diagnostic_name] = float(
                                    fit_diagnostics[diagnostic_name]
                                )
                        row["training_patch_size"] = int(
                            fit_kwargs.get("training_patch_size") or args.crop_size
                        )
                        row["optimizer_lr"] = float(fit_kwargs["lr"])
                        performance = fit_diagnostics.get("performance_profile", {})
                        row["peak_accelerator_memory_bytes"] = int(
                            performance.get("max_memory_allocated_bytes", 0)
                        )
                        render_num_spectral_samples = int(
                            fit_kwargs.get("num_spectral_samples", 1)
                        )
                        render_ctx = (
                            ForwardContext(survey=str(band))
                            if args.forward_context
                            else None
                        )
                        if model_kind == "spectral_power_foam":
                            hard_model = getattr(result_model, "inner", result_model)
                            coefficients = hard_model.cell_coefficients.detach()
                            sites = hard_model.sites.detach()
                            radii = hard_model.radii.detach()
                            probe = torch.tensor(
                                [
                                    [0.0, 0.0],
                                    [float(args.crop_size - 1), 0.0],
                                    [0.0, float(args.crop_size - 1)],
                                    [
                                        float(args.crop_size - 1),
                                        float(args.crop_size - 1),
                                    ],
                                    [
                                        0.5 * float(args.crop_size - 1),
                                        0.5 * float(args.crop_size - 1),
                                    ],
                                ],
                                dtype=torch.float64,
                                device=sites.device,
                            )
                            world = (
                                torchsky.WCS(target_header)
                                .to(sites.device)
                                .pixel_to_world(probe)
                            )
                            target_coordinates = (
                                hard_model.normalized_spatial_coordinates(
                                    world[:, 0], world[:, 1]
                                )
                            )
                            row["foam_coefficient_abs_max"] = float(
                                coefficients.abs().max()
                            )
                            row["foam_log_intensity_spectra"] = bool(
                                hard_model.foam.log_intensity_spectra
                            )
                            gradients = hard_model.cell_gradient_coefficients
                            row["foam_affine_spatial_basis"] = gradients is not None
                            x_centers = torch.unique(sites[:, 0])
                            y_centers = torch.unique(sites[:, 1])
                            if x_centers.numel() < 2 or y_centers.numel() < 2:
                                raise ValueError(
                                    "regular Foam diagnostics require a 2-D grid"
                                )
                            cell_width_x = float(
                                (x_centers.max() - x_centers.min())
                                / (x_centers.numel() - 1)
                            )
                            cell_width_y = float(
                                (y_centers.max() - y_centers.min())
                                / (y_centers.numel() - 1)
                            )
                            cell_scale = math.sqrt(cell_width_x * cell_width_y)
                            if gradients is not None:
                                row["foam_gradient_coefficient_abs_max"] = float(
                                    gradients.detach().abs().max()
                                )
                                wavelengths, response = bandpass_quadrature(
                                    target_observation,
                                    num_spectral_samples=render_num_spectral_samples,
                                    device=sites.device,
                                )
                                band_basis = torch.sum(
                                    response.to(dtype=sites.dtype)[:, None]
                                    * hard_model.spectral_basis(wavelengths).to(
                                        dtype=sites.dtype
                                    ),
                                    dim=0,
                                )
                                band_gradients = torch.einsum(
                                    "k,ckd->cd",
                                    band_basis,
                                    gradients.detach(),
                                )
                                cell_variation = cell_scale * torch.linalg.vector_norm(
                                    band_gradients,
                                    dim=-1,
                                )
                                row["foam_band_cell_variation_abs_p95"] = float(
                                    torch.quantile(cell_variation, 0.95)
                                )
                                row["foam_band_cell_variation_abs_max"] = float(
                                    torch.max(cell_variation)
                                )
                            row["foam_site_x_range"] = [
                                float(sites[:, 0].min()),
                                float(sites[:, 0].max()),
                            ]
                            row["foam_site_y_range"] = [
                                float(sites[:, 1].min()),
                                float(sites[:, 1].max()),
                            ]
                            row["foam_radius_range"] = [
                                float(radii.min()),
                                float(radii.max()),
                            ]
                            row["foam_extent"] = float(args.foam_extent)
                            row["foam_grid_cell_width_x"] = cell_width_x
                            row["foam_grid_cell_width_y"] = cell_width_y
                            row["foam_grid_cell_scale"] = cell_scale
                            row["foam_temperature_start"] = float(
                                hard_model.temperature_start
                            )
                            row["foam_temperature_end"] = float(
                                hard_model.temperature_end
                            )
                            row["foam_geometry_trainable"] = bool(
                                hard_model.sites.requires_grad
                                or hard_model.foam.raw_radii.requires_grad
                            )
                            row["foam_compact_fixed_geometry"] = bool(
                                hard_model.compact_fixed_geometry
                            )
                            row["foam_target_x_range"] = [
                                float(target_coordinates[:, 0].min()),
                                float(target_coordinates[:, 0].max()),
                            ]
                            row["foam_target_y_range"] = [
                                float(target_coordinates[:, 1].min()),
                                float(target_coordinates[:, 1].max()),
                            ]
                            rendered = render_spectral_power_foam_observation_exact(
                                hard_model,
                                target_observation,
                                num_spectral_samples=render_num_spectral_samples,
                                operator_kwargs={
                                    "max_candidate_pairs": int(
                                        args.foam_exact_max_candidate_pairs
                                    )
                                },
                            )
                            row["render_mode"] = "hard_pixel_footprint"
                        else:
                            rendered = render_field_observation(
                                result_model,
                                target_observation,
                                num_spectral_samples=render_num_spectral_samples,
                                forward_context=render_ctx,
                            )
                            row["render_mode"] = "pixel_center"
                        pred = (
                            rendered.detach()
                            .cpu()
                            .numpy()
                            .astype(np.float32, copy=False)
                        ) * intensity_scale
                        row["prediction_abs_max"] = float(np.max(np.abs(pred)))
                        row["prediction_nonzero_fraction"] = float(
                            np.count_nonzero(pred) / pred.size
                        )
                    support_metric_mask = _metric_mask(pred, ref_crop, support_mask)
                    uniform_metric_mask = np.isfinite(pred) & np.isfinite(ref_crop)
                    if not np.any(uniform_metric_mask):
                        raise RuntimeError(
                            "render produced no finite comparison pixels"
                        )
                    support_diff = (
                        pred[support_metric_mask] - ref_crop[support_metric_mask]
                    )
                    uniform_diff = (
                        pred[uniform_metric_mask] - ref_crop[uniform_metric_mask]
                    )
                    source_stats = _background_subtracted_source_stats(
                        pred,
                        ref_crop,
                        support_mask,
                    )

                    # Robust affine fit to ignore background offsets
                    source_metric_mask = _source_metric_mask(
                        pred, ref_crop, support_mask
                    )
                    ref_pts = ref_crop[source_metric_mask].astype(np.float64)
                    pred_pts = pred[source_metric_mask].astype(np.float64)
                    design = np.stack([ref_pts, np.ones_like(ref_pts)], axis=1)
                    coeffs, *_ = np.linalg.lstsq(design, pred_pts, rcond=None)
                    affine_slope = float(coeffs[0])
                    affine_intercept = float(coeffs[1])
                    aligned_pred = (
                        (pred_pts - affine_intercept) / max(affine_slope, 1e-6)
                        if affine_slope > 0.5
                        else pred_pts
                    )
                    aligned_diff = aligned_pred - ref_pts

                    row.update(
                        {
                            "source_rmse": source_stats["source_rmse"],
                            "source_contrast_rmse": source_stats[
                                "source_contrast_rmse"
                            ],
                            "source_contrast_slope": source_stats[
                                "source_contrast_slope"
                            ],
                            "support_rmse": float(
                                np.sqrt(np.mean(support_diff * support_diff))
                            )
                            if support_diff.size
                            else float("nan"),
                            "uniform_rmse": float(
                                np.sqrt(np.mean(uniform_diff * uniform_diff))
                            )
                            if uniform_diff.size
                            else float("nan"),
                            "aligned_rmse": float(
                                np.sqrt(np.mean(aligned_diff * aligned_diff))
                            )
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
                        psf_mode=str(args.psf_model),
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
    ok_rows.sort(
        key=lambda row: (
            str(row["model"]),
            float(row["support_rmse"]),
            str(row["band"]),
        )
    )
    fail_rows.sort(key=lambda row: (str(row["model"]), str(row["band"])))
    payload = {
        "experiment": {
            "input_product": str(args.input_product),
            "exposure_dir": str(args.exposure_dir),
            "bands": bands,
            "module": str(args.module),
            "psf_model": str(args.psf_model),
            "background_model": str(args.background_model),
            "psf_arm_audit": psf_arm_audit,
            "tile_center_deg": tile_center,
            "tile_size_deg": (
                float(args.tile_size_deg) if tile_center is not None else None
            ),
            "crop_size": int(args.crop_size),
            "optimizer_steps": int(args.steps),
            "seed": int(args.seed),
            "foam_grid_side": int(args.foam_grid_side),
            "foam_affine_spatial_basis": bool(args.foam_affine_spatial_basis),
            "foam_exact_max_candidate_pairs": int(args.foam_exact_max_candidate_pairs),
            "foam_basis_count": int(args.foam_basis_count),
            "foam_extent": float(args.foam_extent),
            "foam_lr": float(args.foam_lr),
            "foam_log_intensity_spectra": bool(args.foam_log_intensity_spectra),
            "foam_initial_intensity": float(args.foam_initial_intensity),
            "foam_train_geometry": bool(args.foam_train_geometry),
            "foam_initialize_from_coadd": bool(args.foam_initialize_from_coadd),
            "foam_fit_target_footprint": bool(args.foam_fit_target_footprint),
            "foam_whiten_spectral_basis": bool(args.foam_whiten_spectral_basis),
            "foam_exact_observation_refine": bool(
                args.foam_exact_observation_refine
            ),
            "foam_exact_operator_storage": str(args.foam_exact_operator_storage),
            "foam_exact_observation_batch_size": exact_batch,
            "foam_exact_max_iterations": int(args.foam_exact_max_iterations),
            "foam_exact_preconditioner": str(args.foam_exact_preconditioner),
            "structured_noise_crossfit": bool(args.structured_noise_crossfit),
            "crossfit_scene_steps": int(args.crossfit_scene_steps),
            "structured_noise_steps": int(args.structured_noise_steps),
            "structured_noise_lr": float(args.structured_noise_lr),
            "foam_initialized_inr": bool(args.foam_initialized_inr),
            "inr_steps": int(args.inr_steps or args.steps),
            "inr_lr": float(args.inr_lr),
            "inr_hidden_dim": int(args.inr_hidden_dim),
            "inr_hidden_layers": int(args.inr_hidden_layers),
            "intensity_scale": float(intensity_scale),
            "patt_nums": list(pattern_numbers),
            "crop_x0": args.crop_x0,
            "crop_y0": args.crop_y0,
            "target_header_path": (
                str(args.target_header_path.resolve())
                if args.target_header_path is not None
                else None
            ),
            "injection_manifest": (
                str(args.injection_manifest.resolve())
                if args.injection_manifest is not None
                else None
            ),
            "operator_calibration_manifest": (
                str(args.operator_calibration_manifest.resolve())
                if args.operator_calibration_manifest is not None
                else None
            ),
            "operator_calibration_fit_report": (
                str(args.operator_calibration_fit_report.resolve())
                if args.operator_calibration_fit_report is not None
                else None
            ),
        },
        "injection_audit": injection_rows,
        "exposures": [
            {
                "exposure_index": index,
                "exposure_id": context.exposure_id,
                "band": context.survey,
                "source_path": (context.extra or {}).get("source_path"),
            }
            for index, context in enumerate(all_forward_contexts)
        ],
        "operator_calibration": calibration_audit,
        "psf_arm_audit": psf_arm_audit,
        "structured_noise_crossfit": noise_crossfit_audit,
        "rows": ok_rows + fail_rows,
        "summary": _summarize_by_model(ok_rows),
    }
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    args.markdown_out.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_out.write_text(_format_markdown(payload["rows"]), encoding="utf-8")
    print(json.dumps(payload, indent=2))
    if args.fail_on_method_error and fail_rows:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
