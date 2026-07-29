#!/usr/bin/env python3
"""Gate-0 helpers for the pre-registered multi-band depth experiment.

This module is intentionally benchmark-local.  It provides deterministic truth,
rendering, detection, association, and catalogue scoring without committing a
source-extraction API to :mod:`torchsky`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import hashlib
import json
import math
import numpy as np
from pathlib import Path
import tempfile
import time
from typing import Any, Sequence

import torch
import torch.nn.functional as F

import torchsky
from torchsky.map import ImageObservation, coadd_image_plane
from torchsky.map.operators import fit_background_plane, pixel_area_sr_at
from torchsky.map.sky import MeshSkyEstimator


_SOURCE_KINDS = (
    "continuum_point",
    "compact_galaxy",
    "emission_line",
    "blue_dropout",
    "red_dropout",
)

_NIRCAM_GAUSSIAN_PSF_FWHM_ARCSEC = {
    "F090W": 0.033,
    "F150W": 0.049,
    "F200W": 0.066,
    "F277W": 0.091,
    "F356W": 0.116,
    "F444W": 0.145,
}
_UPDEC_PATT_SPLIT = {
    1: "train",
    2: "train",
    3: "validation",
    4: "train",
    5: "train",
    6: "test",
    7: "train",
    8: "validation",
    9: "test",
}
_UPDEC_BANDS = ("F090W", "F150W", "F200W", "F277W", "F356W", "F444W")


@dataclass(frozen=True)
class SeedManifest:
    """Independent random streams required by the experiment protocol."""

    scene: int
    noise: int
    pointing: int
    nuisance: int


@dataclass(frozen=True)
class SplitManifest:
    """Disjoint complete identifiers used by the benchmark."""

    train: tuple[str, ...]
    validation: tuple[str, ...]
    test: tuple[str, ...]


@dataclass
class TruthCatalog:
    """Synthetic catalogue in target-image pixel coordinates."""

    xy: torch.Tensor
    band_flux: torch.Tensor
    spectral_coordinate: torch.Tensor
    spectral_flux_density: torch.Tensor
    band_response: torch.Tensor
    sigma_px: torch.Tensor
    kind: tuple[str, ...]
    source_id: tuple[str, ...]

    def __post_init__(self) -> None:
        count = self.xy.shape[0]
        if self.xy.ndim != 2 or self.xy.shape[1] != 2:
            raise ValueError("xy must have shape (sources, 2)")
        if self.band_flux.ndim != 2 or self.band_flux.shape[0] != count:
            raise ValueError("band_flux must have shape (sources, bands)")
        if self.spectral_coordinate.ndim != 1:
            raise ValueError("spectral_coordinate must be one-dimensional")
        spectral_samples = self.spectral_coordinate.shape[0]
        if self.spectral_flux_density.shape != (count, spectral_samples):
            raise ValueError(
                "spectral_flux_density must have shape (sources, spectral samples)"
            )
        if self.band_response.shape != (self.band_flux.shape[1], spectral_samples):
            raise ValueError(
                "band_response must have shape (bands, spectral samples)"
            )
        if self.sigma_px.shape != (count,):
            raise ValueError("sigma_px must have shape (sources,)")
        if len(self.kind) != count or len(self.source_id) != count:
            raise ValueError("kind and source_id must match the source count")
        if not (
            torch.isfinite(self.xy).all()
            and torch.isfinite(self.band_flux).all()
            and torch.isfinite(self.spectral_coordinate).all()
            and torch.isfinite(self.spectral_flux_density).all()
            and torch.isfinite(self.band_response).all()
            and torch.isfinite(self.sigma_px).all()
        ):
            raise ValueError("catalogue tensors must be finite")
        if (
            torch.any(self.band_flux < 0)
            or torch.any(self.spectral_flux_density < 0)
            or torch.any(self.band_response < 0)
            or torch.any(self.sigma_px <= 0)
        ):
            raise ValueError(
                "fluxes/responses must be non-negative and sigma_px positive"
            )


@dataclass(frozen=True)
class MatchResult:
    """Maximum-cardinality one-to-one truth/detection association."""

    pairs: tuple[tuple[int, int], ...]
    distances_px: tuple[float, ...]
    unmatched_truth: tuple[int, ...]
    unmatched_detections: tuple[int, ...]


@dataclass
class SyntheticExposureSet:
    """Dithered detector images and their explicit WCS/noise contract."""

    images: torch.Tensor
    headers: tuple[dict[str, object], ...]
    target_header: dict[str, object]
    noise_sigma: torch.Tensor
    dither_xy_px: torch.Tensor

    def __post_init__(self) -> None:
        if self.images.ndim != 4:
            raise ValueError("images must have shape (exposures, bands, height, width)")
        exposures, bands = self.images.shape[:2]
        if len(self.headers) != exposures:
            raise ValueError("one header is required per exposure")
        if self.noise_sigma.shape != (bands,):
            raise ValueError("noise_sigma must have shape (bands,)")
        if self.dither_xy_px.shape != (exposures, 2):
            raise ValueError("dither_xy_px must have shape (exposures, 2)")
        if (
            not torch.isfinite(self.images).all()
            or not torch.isfinite(self.noise_sigma).all()
            or not torch.isfinite(self.dither_xy_px).all()
            or torch.any(self.noise_sigma <= 0)
        ):
            raise ValueError("exposure tensors must be finite and noise positive")


@dataclass(frozen=True)
class GaussianPSFContract:
    """Explicit truth and fitted Gaussian PSF proxies for the sprint smoke."""

    band: str
    truth_fwhm_arcsec: float
    fitted_fwhm_arcsec: float
    spatial_oversample: int = 4

    def __post_init__(self) -> None:
        if self.band not in _NIRCAM_GAUSSIAN_PSF_FWHM_ARCSEC:
            raise ValueError(f"unsupported NIRCam band {self.band!r}")
        if self.truth_fwhm_arcsec <= 0 or self.fitted_fwhm_arcsec <= 0:
            raise ValueError("PSF FWHM values must be positive")
        if self.spatial_oversample < 1:
            raise ValueError("spatial_oversample must be positive")


@dataclass
class DetectorInjectionResult:
    """Detector image plus sparse, unit-audited point-source injection."""

    image: torch.Tensor
    delta: torch.Tensor
    source_xy: torch.Tensor
    accepted: torch.Tensor
    rendered_flux_jy: torch.Tensor


def inject_common_world_exposures(
    exposure_paths: Sequence[Path],
    *,
    band: str,
    world_radec_deg: torch.Tensor,
    flux_jy: torch.Tensor,
    device: torch.device | str = "cpu",
) -> tuple[dict[Path, np.ndarray], list[dict[str, object]]]:
    """Create transient SCI overrides for one common sky catalogue.

    Archived detector products are only read. A source may fall outside an
    individual detector, but every requested source must be covered by at least
    one selected exposure.
    """

    if not exposure_paths:
        raise ValueError("at least one exposure path is required")
    if (
        world_radec_deg.ndim != 2
        or world_radec_deg.shape[1] != 2
        or flux_jy.shape != (world_radec_deg.shape[0],)
    ):
        raise ValueError("world_radec_deg must be (sources, 2) and flux_jy (sources,)")
    from bench_map_smacs0723_real import _read_ext, _sci_header

    target_device = torch.device(device)
    world = world_radec_deg.to(device=target_device, dtype=torch.float64)
    flux = flux_jy.to(device=target_device, dtype=torch.float64)
    contract = gaussian_psf_contract(band)
    overrides: dict[Path, np.ndarray] = {}
    audit: list[dict[str, object]] = []
    accepted_rows: list[torch.Tensor] = []
    for exposure_path in exposure_paths:
        science_array = _read_ext(exposure_path, "SCI")
        science = torch.from_numpy(science_array.copy()).to(target_device)
        if world.shape[0] == 0:
            injected = DetectorInjectionResult(
                image=science,
                delta=torch.zeros_like(science),
                source_xy=torch.empty(
                    (0, 2), dtype=torch.float64, device=target_device
                ),
                accepted=torch.empty(
                    (0,), dtype=torch.bool, device=target_device
                ),
                rendered_flux_jy=torch.empty(
                    (0,), dtype=torch.float64, device=target_device
                ),
            )
        else:
            injected = inject_gaussian_sources_mjysr(
                science,
                header=dict(_sci_header(exposure_path)),
                world_radec_deg=world,
                flux_jy=flux,
                psf_fwhm_arcsec=contract.truth_fwhm_arcsec,
                spatial_oversample=contract.spatial_oversample,
            )
        accepted_rows.append(injected.accepted)
        overrides[exposure_path] = (
            injected.image.detach().cpu().numpy().astype(science_array.dtype, copy=False)
        )
        audit.append(
            {
                "path": str(exposure_path.resolve()),
                "accepted": injected.accepted.detach().cpu().tolist(),
                "source_xy": injected.source_xy.detach().cpu().tolist(),
                "rendered_flux_jy": (
                    injected.rendered_flux_jy.detach().cpu().tolist()
                ),
                "input_unchanged": bool(
                    torch.allclose(
                        science.detach().cpu(),
                        torch.from_numpy(science_array),
                        equal_nan=True,
                    )
                ),
            }
        )
    if world.shape[0] and not torch.stack(accepted_rows).any(dim=0).all():
        missing = torch.nonzero(
            ~torch.stack(accepted_rows).any(dim=0), as_tuple=False
        ).flatten()
        raise ValueError(
            "common-world sources lack detector coverage: "
            + ",".join(str(int(index)) for index in missing)
        )
    return overrides, audit


def make_updec_common_world_manifest(
    *,
    target_header: dict[str, object],
    bands: Sequence[str] = _UPDEC_BANDS,
    source_count: int = 28,
    seed: int = 101,
    ab_magnitudes: Sequence[float] = (27.0, 27.5, 28.0, 28.5, 29.0, 29.5, 30.0),
    margin_px: float = 12.0,
    min_separation_px: float = 8.0,
    batch_id: str = "validation-000",
    valid_mask: torch.Tensor | None = None,
) -> dict[str, object]:
    """Generate a deterministic four-quadrant smooth-continuum injection batch."""

    width = int(target_header["NAXIS1"])
    height = int(target_header["NAXIS2"])
    if valid_mask is not None and (
        valid_mask.dtype is not torch.bool
        or valid_mask.device.type != "cpu"
        or valid_mask.shape != (height, width)
    ):
        raise ValueError("valid_mask must be a CPU bool tensor matching the target")
    normalized_bands = tuple(str(band).upper() for band in bands)
    if (
        source_count < 0
        or seed < 0
        or not normalized_bands
        or not ab_magnitudes
        or margin_px <= 0
        or min_separation_px < 0
        or width <= 3 * margin_px
        or height <= 3 * margin_px
    ):
        raise ValueError("invalid source, seed, band, magnitude, or target-grid contract")
    from bench_map_smacs0723_real import _band_spec

    wavelengths = {
        band: float(_band_spec(band)["wavelength_angstrom"])
        for band in normalized_bands
    }
    generator = torch.Generator().manual_seed(seed)
    positions: list[torch.Tensor] = []
    quadrants: list[int] = []
    quadrant_bounds: list[tuple[float, float, float, float]] = []
    for quadrant in range(4):
        left = quadrant % 2 == 0
        bottom = quadrant < 2
        x_min = margin_px if left else 0.5 * width + 0.5 * margin_px
        x_max = 0.5 * width - 0.5 * margin_px if left else width - margin_px
        y_min = margin_px if bottom else 0.5 * height + 0.5 * margin_px
        y_max = 0.5 * height - 0.5 * margin_px if bottom else height - margin_px
        quadrant_bounds.append((x_min, x_max, y_min, y_max))
    if valid_mask is not None:
        candidate_pools: list[torch.Tensor] = []
        valid_y, valid_x = torch.nonzero(valid_mask, as_tuple=True)
        for x_min, x_max, y_min, y_max in quadrant_bounds:
            inside = (
                (valid_x >= math.ceil(x_min))
                & (valid_x <= math.floor(x_max))
                & (valid_y >= math.ceil(y_min))
                & (valid_y <= math.floor(y_max))
            )
            pool = torch.stack((valid_x[inside], valid_y[inside]), dim=1)
            if pool.shape[0]:
                pool = pool[
                    torch.randperm(pool.shape[0], generator=generator)
                ]
            candidate_pools.append(pool)
        pool_offsets = [0, 0, 0, 0]
        while len(positions) < source_count:
            quadrant = len(positions) % 4
            pool = candidate_pools[quadrant]
            accepted = False
            while pool_offsets[quadrant] < pool.shape[0]:
                pixel = pool[pool_offsets[quadrant]].to(torch.float64)
                pool_offsets[quadrant] += 1
                jitter = (
                    torch.rand((2,), generator=generator, dtype=torch.float64) - 0.5
                ) * 0.9
                candidate = pixel + jitter
                if positions and min(
                    float(torch.linalg.vector_norm(candidate - position))
                    for position in positions
                ) < min_separation_px:
                    continue
                positions.append(candidate)
                quadrants.append(quadrant)
                accepted = True
                break
            if not accepted:
                raise ValueError(
                    "cannot place requested sources in every support quadrant"
                )
    else:
        attempts = 0
        while len(positions) < source_count:
            attempts += 1
            if attempts > max(1000, 200 * source_count):
                raise ValueError(
                    "cannot place requested sources at the frozen separation"
                )
            quadrant = len(positions) % 4
            x_min, x_max, y_min, y_max = quadrant_bounds[quadrant]
            sample = torch.rand((2,), generator=generator, dtype=torch.float64)
            candidate = torch.tensor(
                [
                    x_min + float(sample[0]) * (x_max - x_min),
                    y_min + float(sample[1]) * (y_max - y_min),
                ],
                dtype=torch.float64,
            )
            if positions and min(
                float(torch.linalg.vector_norm(candidate - position))
                for position in positions
            ) < min_separation_px:
                continue
            positions.append(candidate)
            quadrants.append(quadrant)
    target_xy = (
        torch.stack(positions)
        if positions
        else torch.empty((0, 2), dtype=torch.float64)
    )
    world = torchsky.WCS(target_header).pixel_to_world(target_xy)
    magnitudes = torch.tensor(
        [
            float(ab_magnitudes[index % len(ab_magnitudes)])
            for index in range(source_count)
        ],
        dtype=torch.float64,
    )
    reference_flux_jy = 3631.0 * torch.pow(10.0, -0.4 * magnitudes)
    slopes = (
        torch.rand((source_count,), generator=generator, dtype=torch.float64) - 0.5
    )
    pivot = float(wavelengths["F200W"] if "F200W" in wavelengths else next(iter(wavelengths.values())))
    band_flux = {
        band: (
            reference_flux_jy * (wavelength / pivot) ** slopes
        ).tolist()
        for band, wavelength in wavelengths.items()
    }
    canonical_header = {
        str(key): (
            value
            if value is None or isinstance(value, (bool, int, float, str))
            else str(value)
        )
        for key, value in target_header.items()
    }
    header_sha256 = hashlib.sha256(
        json.dumps(
            canonical_header, sort_keys=True, separators=(",", ":")
        ).encode("utf-8")
    ).hexdigest()
    return {
        "schema_version": 1,
        "profile": "updec-adass-common-world-manifest",
        "depth_claim": False,
        "batch_id": str(batch_id),
        "seed": seed,
        "source_count": source_count,
        "source_kind": "smooth_continuum_point",
        "source_id": [
            f"{batch_id}-source-{index:04d}" for index in range(source_count)
        ],
        "quadrant": quadrants,
        "target_xy": target_xy.tolist(),
        "world_radec_deg": world.tolist(),
        "reference_ab_magnitude": magnitudes.tolist(),
        "spectral_slope_fnu": slopes.tolist(),
        "band_wavelength_angstrom": wavelengths,
        "band_flux_jy": band_flux,
        "target_header_sha256": header_sha256,
        "placement": {
            "margin_px": margin_px,
            "min_separation_px": min_separation_px,
            "quadrant_round_robin": True,
            "valid_mask_pixel_count": (
                int(valid_mask.sum()) if valid_mask is not None else None
            ),
            "valid_mask_sha256": (
                hashlib.sha256(valid_mask.numpy().tobytes()).hexdigest()
                if valid_mask is not None
                else None
            ),
            "sampling": (
                "shuffled_valid_pixels"
                if valid_mask is not None
                else "continuous_rejection"
            ),
        },
        "psf_contract": {
            band: asdict(gaussian_psf_contract(band))
            for band in normalized_bands
        },
    }


def common_support_mask(
    weights: Sequence[torch.Tensor],
    *,
    threshold: float = 0.0,
    erode_radius_px: int = 0,
    min_footprint_fraction: float = 1.0,
) -> torch.Tensor:
    """Require common center support and local footprint coverage in every map."""

    if not weights:
        raise ValueError("at least one weight map is required")
    shape = weights[0].shape
    if (
        len(shape) != 2
        or not math.isfinite(threshold)
        or not isinstance(erode_radius_px, int)
        or erode_radius_px < 0
        or not math.isfinite(min_footprint_fraction)
        or not 0 < min_footprint_fraction <= 1
        or any(weight.shape != shape for weight in weights)
    ):
        raise ValueError("weight maps must be matching 2-D tensors")
    masks: list[torch.Tensor] = []
    for weight in weights:
        values = weight.detach().to(device="cpu")
        masks.append(torch.isfinite(values) & (values > threshold))
    support = torch.stack(masks).all(dim=0)
    if erode_radius_px and min_footprint_fraction == 1.0:
        invalid = (~support).to(dtype=torch.float32)[None, None]
        invalid = F.pad(
            invalid,
            (erode_radius_px,) * 4,
            mode="constant",
            value=1.0,
        )
        support = (
            F.max_pool2d(
                invalid,
                kernel_size=2 * erode_radius_px + 1,
                stride=1,
            )[0, 0]
            == 0.0
        )
    elif erode_radius_px:
        kernel_size = 2 * erode_radius_px + 1
        required = min_footprint_fraction - 0.5 / kernel_size**2
        support = torch.ones(shape, dtype=torch.bool)
        for mask in masks:
            padded = F.pad(
                mask.to(torch.float32)[None, None],
                (erode_radius_px,) * 4,
                mode="constant",
                value=0.0,
            )
            local_fraction = F.avg_pool2d(
                padded,
                kernel_size=kernel_size,
                stride=1,
            )[0, 0]
            support &= mask & (local_fraction >= required)
    return support


def exclude_catalogue_neighborhoods(
    valid_mask: torch.Tensor,
    catalogue_xy: torch.Tensor,
    *,
    radius_px: float,
) -> torch.Tensor:
    """Remove circular neighborhoods around a frozen control catalogue."""

    if (
        valid_mask.ndim != 2
        or valid_mask.dtype is not torch.bool
        or valid_mask.device.type != "cpu"
        or catalogue_xy.ndim != 2
        or catalogue_xy.shape[1] != 2
        or catalogue_xy.device.type != "cpu"
        or not catalogue_xy.is_floating_point()
        or not torch.isfinite(catalogue_xy).all()
        or not math.isfinite(radius_px)
        or radius_px < 0
    ):
        raise ValueError("invalid CPU mask, catalogue_xy, or exclusion radius")
    result = valid_mask.clone()
    height, width = result.shape
    for position in catalogue_xy:
        x_min = max(0, math.floor(float(position[0] - radius_px)))
        x_max = min(width, math.ceil(float(position[0] + radius_px)) + 1)
        y_min = max(0, math.floor(float(position[1] - radius_px)))
        y_max = min(height, math.ceil(float(position[1] + radius_px)) + 1)
        if x_min >= x_max or y_min >= y_max:
            continue
        yy, xx = torch.meshgrid(
            torch.arange(y_min, y_max, dtype=position.dtype),
            torch.arange(x_min, x_max, dtype=position.dtype),
            indexing="ij",
        )
        disk = (
            (xx - position[0]).square() + (yy - position[1]).square()
            <= radius_px**2
        )
        result[y_min:y_max, x_min:x_max] &= ~disk
    return result


def gaussian_psf_contract(
    band: str,
    *,
    fitted_scale: float = 1.05,
    spatial_oversample: int = 4,
) -> GaussianPSFContract:
    """Return the frozen band-dependent Gaussian proxy and mismatch control."""

    band = str(band).upper()
    if fitted_scale <= 0:
        raise ValueError("fitted_scale must be positive")
    truth = _NIRCAM_GAUSSIAN_PSF_FWHM_ARCSEC.get(band)
    if truth is None:
        raise ValueError(f"unsupported NIRCam band {band!r}")
    return GaussianPSFContract(
        band=band,
        truth_fwhm_arcsec=truth,
        fitted_fwhm_arcsec=truth * fitted_scale,
        spatial_oversample=spatial_oversample,
    )


def updec_patt_split(patt_num: int) -> str:
    """Return the pre-registered split for a complete JWST dither identifier."""

    try:
        return _UPDEC_PATT_SPLIT[int(patt_num)]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("PATT_NUM must be an integer from 1 through 9") from exc


def _validate_mjysr_image(
    image: torch.Tensor,
    header: dict[str, object],
) -> None:
    if image.ndim != 2 or not image.is_floating_point():
        raise ValueError("detector image must be a floating-point 2D tensor")
    unit = str(header.get("BUNIT", "")).strip().lower().replace(" ", "")
    if unit not in {"mjy/sr", "mjypersteradian"}:
        raise ValueError("detector injection currently requires BUNIT='MJy/sr'")


def _oversampled_gaussian_patch(
    *,
    center_xy: torch.Tensor,
    sigma_px: float,
    x_min: int,
    x_max: int,
    y_min: int,
    y_max: int,
    oversample: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = center_xy.dtype
    device = center_xy.device
    yy, xx = torch.meshgrid(
        torch.arange(y_min, y_max, dtype=dtype, device=device),
        torch.arange(x_min, x_max, dtype=dtype, device=device),
        indexing="ij",
    )
    offsets = (
        torch.arange(oversample, dtype=dtype, device=device) + 0.5
    ) / oversample - 0.5
    profile = torch.zeros_like(xx)
    for dy in offsets:
        for dx in offsets:
            radius2 = (xx + dx - center_xy[0]).square() + (
                yy + dy - center_xy[1]
            ).square()
            profile += torch.exp(-0.5 * radius2 / sigma_px**2)
    profile /= oversample**2
    profile /= profile.sum()
    return profile, xx, yy


def inject_gaussian_sources_mjysr(
    image: torch.Tensor,
    *,
    header: dict[str, object],
    world_radec_deg: torch.Tensor,
    flux_jy: torch.Tensor,
    psf_fwhm_arcsec: float,
    spatial_oversample: int = 4,
    truncate_sigma: float = 5.0,
) -> DetectorInjectionResult:
    """Inject flux-conserving point sources through a detector WCS.

    The output remains in ``MJy/sr``.  Every sparse footprint uses its local
    WCS Jacobian, so summing ``delta * pixel_area_sr * 1e6`` returns Jy.
    Sources whose full truncated footprint is not on the detector are rejected.
    """

    _validate_mjysr_image(image, header)
    if (
        world_radec_deg.ndim != 2
        or world_radec_deg.shape[1] != 2
        or flux_jy.shape != (world_radec_deg.shape[0],)
    ):
        raise ValueError("world_radec_deg must be (sources, 2) and flux_jy (sources,)")
    if (
        world_radec_deg.device != image.device
        or flux_jy.device != image.device
        or not torch.isfinite(world_radec_deg).all()
        or not torch.isfinite(flux_jy).all()
    ):
        raise ValueError("image, coordinates, and finite fluxes must share a device")
    if torch.any(flux_jy < 0):
        raise ValueError("flux_jy must be non-negative")
    if psf_fwhm_arcsec <= 0 or spatial_oversample < 1 or truncate_sigma <= 0:
        raise ValueError("PSF, oversampling, and truncation values must be positive")

    wcs = torchsky.WCS(header)
    x, y = wcs.world_to_pixel(world_radec_deg[:, 0], world_radec_deg[:, 1])
    source_xy = torch.stack((x, y), dim=-1).to(
        device=image.device, dtype=torch.float64
    )
    delta = torch.zeros_like(image)
    accepted = torch.zeros(world_radec_deg.shape[0], dtype=torch.bool, device=image.device)
    rendered_flux = torch.zeros_like(flux_jy)
    height, width = image.shape
    sigma_arcsec = psf_fwhm_arcsec / 2.354820045

    for index, position in enumerate(source_xy):
        center_area = pixel_area_sr_at(wcs, position[0], position[1])
        pixel_scale_arcsec = float(
            torch.sqrt(center_area) * (180.0 / torch.pi) * 3600.0
        )
        sigma_px = sigma_arcsec / pixel_scale_arcsec
        radius = max(1, math.ceil(truncate_sigma * sigma_px + 0.5))
        x_min = math.floor(float(position[0])) - radius
        x_max = math.floor(float(position[0])) + radius + 1
        y_min = math.floor(float(position[1])) - radius
        y_max = math.floor(float(position[1])) + radius + 1
        if x_min < 0 or y_min < 0 or x_max > width or y_max > height:
            continue
        profile, xx, yy = _oversampled_gaussian_patch(
            center_xy=position,
            sigma_px=sigma_px,
            x_min=x_min,
            x_max=x_max,
            y_min=y_min,
            y_max=y_max,
            oversample=spatial_oversample,
        )
        area_sr = pixel_area_sr_at(wcs, xx, yy)
        contribution = (
            flux_jy[index].to(torch.float64) * profile / (1.0e6 * area_sr)
        ).to(image.dtype)
        delta[y_min:y_max, x_min:x_max] += contribution
        accepted[index] = True
        rendered_flux[index] = (
            contribution.to(torch.float64) * area_sr * 1.0e6
        ).sum().to(flux_jy.dtype)

    return DetectorInjectionResult(
        image=image + delta,
        delta=delta,
        source_xy=source_xy,
        accepted=accepted,
        rendered_flux_jy=rendered_flux,
    )


def measure_gaussian_source_flux_jy(
    image_mjysr: torch.Tensor,
    *,
    header: dict[str, object],
    world_radec_deg: torch.Tensor,
    psf_fwhm_arcsec: float,
    spatial_oversample: int = 4,
    truncate_sigma: float = 5.0,
) -> float:
    """Measure one source with a fixed Gaussian template in integrated Jy."""

    _validate_mjysr_image(image_mjysr, header)
    if world_radec_deg.shape != (2,) or not torch.isfinite(world_radec_deg).all():
        raise ValueError("world_radec_deg must contain one finite (RA, Dec) pair")
    wcs = torchsky.WCS(header)
    x, y = wcs.world_to_pixel(world_radec_deg[0], world_radec_deg[1])
    position = torch.stack((x, y)).to(
        device=image_mjysr.device, dtype=torch.float64
    )
    area_center = pixel_area_sr_at(wcs, position[0], position[1])
    scale_arcsec = float(torch.sqrt(area_center) * (180.0 / torch.pi) * 3600.0)
    sigma_px = psf_fwhm_arcsec / 2.354820045 / scale_arcsec
    radius = max(1, math.ceil(truncate_sigma * sigma_px + 0.5))
    x_min = math.floor(float(position[0])) - radius
    x_max = math.floor(float(position[0])) + radius + 1
    y_min = math.floor(float(position[1])) - radius
    y_max = math.floor(float(position[1])) + radius + 1
    if x_min < 0 or y_min < 0 or x_max > image_mjysr.shape[1] or y_max > image_mjysr.shape[0]:
        raise ValueError("source PSF footprint falls outside the detector")
    profile, xx, yy = _oversampled_gaussian_patch(
        center_xy=position,
        sigma_px=sigma_px,
        x_min=x_min,
        x_max=x_max,
        y_min=y_min,
        y_max=y_max,
        oversample=spatial_oversample,
    )
    area_sr = pixel_area_sr_at(wcs, xx, yy)
    data_jy = (
        image_mjysr[y_min:y_max, x_min:x_max].to(torch.float64)
        * area_sr
        * 1.0e6
    )
    return float((profile * data_jy).sum() / profile.square().sum())


def make_seed_manifest(root_seed: int) -> SeedManifest:
    """Derive independent, serialized random streams from one root seed."""

    if root_seed < 0:
        raise ValueError("root_seed must be non-negative")
    generator = torch.Generator().manual_seed(root_seed)
    values = torch.randint(0, 2**31 - 1, (4,), generator=generator).tolist()
    return SeedManifest(*(int(value) for value in values))


def split_identifiers(
    identifiers: Sequence[str],
    *,
    seed: int,
    validation_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> SplitManifest:
    """Split complete identifiers deterministically without leakage."""

    ordered = tuple(str(value) for value in identifiers)
    if len(set(ordered)) != len(ordered):
        raise ValueError("identifiers must be unique")
    if not ordered:
        raise ValueError("at least one identifier is required")
    if validation_fraction < 0 or test_fraction < 0:
        raise ValueError("split fractions must be non-negative")
    if validation_fraction + test_fraction >= 1:
        raise ValueError("validation_fraction + test_fraction must be below one")

    permutation = torch.randperm(
        len(ordered), generator=torch.Generator().manual_seed(seed)
    ).tolist()
    n_validation = int(round(len(ordered) * validation_fraction))
    n_test = int(round(len(ordered) * test_fraction))
    if len(ordered) >= 3:
        n_validation = max(n_validation, 1)
        n_test = max(n_test, 1)
    if n_validation + n_test >= len(ordered):
        raise ValueError("split leaves no training identifiers")
    validation_idx = set(permutation[:n_validation])
    test_idx = set(permutation[n_validation : n_validation + n_test])
    return SplitManifest(
        train=tuple(value for i, value in enumerate(ordered) if i not in validation_idx | test_idx),
        validation=tuple(value for i, value in enumerate(ordered) if i in validation_idx),
        test=tuple(value for i, value in enumerate(ordered) if i in test_idx),
    )


def generate_truth_catalog(
    *,
    height: int,
    width: int,
    bands: int,
    source_count: int,
    seed: int,
    spectral_samples: int = 129,
    include_close_pair: bool = True,
    device: torch.device | str = "cpu",
    dtype: torch.dtype = torch.float64,
) -> TruthCatalog:
    """Generate deterministic continuum, line, dropout, galaxy, and pair truth."""

    if height < 16 or width < 16:
        raise ValueError("height and width must be at least 16 pixels")
    if bands < 3:
        raise ValueError("at least three bands are required")
    if spectral_samples < 33:
        raise ValueError("spectral_samples must be at least 33")
    if source_count < 0:
        raise ValueError("source_count must be non-negative")
    wavelength = torch.linspace(-1.0, 1.0, spectral_samples, dtype=dtype)
    band_centers = torch.linspace(-0.85, 0.85, bands, dtype=dtype)
    band_response = torch.exp(
        -0.5 * ((wavelength[None] - band_centers[:, None]) / 0.24).square()
    )
    band_response /= torch.trapezoid(
        band_response, wavelength, dim=-1
    )[:, None]
    if source_count == 0:
        empty = torch.empty
        return TruthCatalog(
            xy=empty((0, 2), device=device, dtype=dtype),
            band_flux=empty((0, bands), device=device, dtype=dtype),
            spectral_coordinate=wavelength.to(device),
            spectral_flux_density=empty(
                (0, spectral_samples), device=device, dtype=dtype
            ),
            band_response=band_response.to(device),
            sigma_px=empty((0,), device=device, dtype=dtype),
            kind=(),
            source_id=(),
        )

    generator = torch.Generator().manual_seed(seed)
    margin = 5.0
    xy = torch.rand((source_count, 2), generator=generator, dtype=dtype)
    xy[:, 0] = margin + xy[:, 0] * (width - 1 - 2 * margin)
    xy[:, 1] = margin + xy[:, 1] * (height - 1 - 2 * margin)
    amplitude = torch.exp(
        torch.empty(source_count, dtype=dtype).uniform_(
            -0.7, 1.4, generator=generator
        )
    )
    kinds = tuple(
        _SOURCE_KINDS[index % len(_SOURCE_KINDS)]
        for index in range(source_count)
    )
    spectra: list[torch.Tensor] = []
    sigma: list[float] = []
    for index, kind in enumerate(kinds):
        slope = float(
            torch.empty((), dtype=dtype).uniform_(-0.5, 0.5, generator=generator)
        )
        spectrum = torch.exp(slope * wavelength)
        if kind == "emission_line":
            spectrum = 0.08 * spectrum
            line_center = band_centers[(2 * index + 1) % bands]
            spectrum += 2.5 * torch.exp(
                -0.5 * ((wavelength - line_center) / 0.018).square()
            )
        elif kind == "blue_dropout":
            spectrum = spectrum * (wavelength <= 0.0)
        elif kind == "red_dropout":
            spectrum = spectrum * (wavelength >= 0.0)
        spectra.append(amplitude[index] * spectrum)
        sigma.append(1.05 if kind == "continuum_point" else 1.8)

    if include_close_pair and source_count >= 2:
        # The pair spans 1.25 nominal point-source FWHM and remains inside support.
        xy[-1] = xy[-2]
        xy[-1, 0] += 1.25 * 2.354820045 * 1.05
        xy[-1, 0].clamp_(max=width - 1 - margin)

    spectral_flux_density = torch.stack(spectra)
    band_flux = torch.trapezoid(
        spectral_flux_density[:, None] * band_response[None],
        wavelength,
        dim=-1,
    )
    return TruthCatalog(
        xy=xy.to(device),
        band_flux=band_flux.to(device),
        spectral_coordinate=wavelength.to(device),
        spectral_flux_density=spectral_flux_density.to(device),
        band_response=band_response.to(device),
        sigma_px=torch.tensor(sigma, dtype=dtype, device=device),
        kind=kinds,
        source_id=tuple(f"source-{index:05d}" for index in range(source_count)),
    )


def render_truth(
    catalog: TruthCatalog,
    *,
    height: int,
    width: int,
    spatial_oversample: int = 4,
) -> torch.Tensor:
    """Render a flux-conserving Gaussian truth cube with subpixel integration."""

    if spatial_oversample < 1:
        raise ValueError("spatial_oversample must be positive")
    bands = catalog.band_flux.shape[1]
    cube = torch.zeros(
        (bands, height, width),
        dtype=catalog.band_flux.dtype,
        device=catalog.band_flux.device,
    )
    if catalog.xy.shape[0] == 0:
        return cube
    profiles = _gaussian_profiles(
        catalog,
        height=height,
        width=width,
        spatial_oversample=spatial_oversample,
    )
    return torch.einsum("nb,nhw->bhw", catalog.band_flux, profiles)


def simple_tan_header(
    height: int,
    width: int,
    *,
    pixel_scale_arcsec: float = 0.04,
    dither_xy_px: tuple[float, float] = (0.0, 0.0),
) -> dict[str, object]:
    """Return the small deterministic TAN grid used by synthetic detector data."""

    if height < 1 or width < 1:
        raise ValueError("image dimensions must be positive")
    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be positive")
    scale_deg = pixel_scale_arcsec / 3600.0
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRVAL1": 110.0,
        "CRVAL2": -73.0,
        "CRPIX1": (width + 1.0) / 2.0 + dither_xy_px[0],
        "CRPIX2": (height + 1.0) / 2.0 + dither_xy_px[1],
        "CD1_1": -scale_deg,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale_deg,
    }


def _sample_target_cube(
    cube: torch.Tensor,
    *,
    target_header: dict[str, object],
    detector_header: dict[str, object],
) -> torch.Tensor:
    """Sample a target-plane cube at detector pixel centres through Torchsky WCS."""

    height, width = cube.shape[-2:]
    y, x = torch.meshgrid(
        torch.arange(height, device=cube.device, dtype=torch.float64),
        torch.arange(width, device=cube.device, dtype=torch.float64),
        indexing="ij",
    )
    ra, dec = torchsky.WCS(detector_header).pixel_to_world(x, y)
    target_x, target_y = torchsky.WCS(target_header).world_to_pixel(ra, dec)
    grid = torch.stack(
        (
            target_x.to(cube.dtype) / max(width - 1, 1) * 2.0 - 1.0,
            target_y.to(cube.dtype) / max(height - 1, 1) * 2.0 - 1.0,
        ),
        dim=-1,
    )
    return F.grid_sample(
        cube[None],
        grid[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[0]


def simulate_dithered_exposures(
    truth: torch.Tensor,
    *,
    seed: int,
    exposure_count: int = 4,
    noise_sigma: torch.Tensor | float = 0.05,
    pixel_scale_arcsec: float = 0.04,
) -> SyntheticExposureSet:
    """Create independent noisy detector images with subpixel TAN dithers."""

    if truth.ndim != 3 or not truth.is_floating_point():
        raise ValueError("truth must be floating point with shape (bands, height, width)")
    if exposure_count < 1:
        raise ValueError("exposure_count must be positive")
    bands, height, width = truth.shape
    sigma = torch.as_tensor(noise_sigma, dtype=truth.dtype, device=truth.device)
    if sigma.ndim == 0:
        sigma = sigma.expand(bands).clone()
    if sigma.shape != (bands,) or torch.any(sigma <= 0):
        raise ValueError("noise_sigma must be positive and scalar or one per band")
    generator = torch.Generator().manual_seed(seed)
    dithers = torch.empty((exposure_count, 2), dtype=torch.float64).uniform_(
        -0.45, 0.45, generator=generator
    )
    dithers[0].zero_()
    target_header = simple_tan_header(
        height, width, pixel_scale_arcsec=pixel_scale_arcsec
    )
    headers: list[dict[str, object]] = []
    images: list[torch.Tensor] = []
    for dither in dithers:
        header = simple_tan_header(
            height,
            width,
            pixel_scale_arcsec=pixel_scale_arcsec,
            dither_xy_px=(float(dither[0]), float(dither[1])),
        )
        sampled = _sample_target_cube(
            truth, target_header=target_header, detector_header=header
        )
        noise = torch.randn(
            sampled.shape,
            generator=generator,
            dtype=sampled.dtype,
            device="cpu",
        ).to(sampled.device)
        images.append(sampled + sigma[:, None, None] * noise)
        headers.append(header)
    return SyntheticExposureSet(
        images=torch.stack(images),
        headers=tuple(headers),
        target_header=target_header,
        noise_sigma=sigma,
        dither_xy_px=dithers.to(truth.device),
    )


def coadd_synthetic_exposures(
    exposures: SyntheticExposureSet,
    *,
    resampling_kernel: str,
    drizzle_pixfrac: float = 1.0,
    device: torch.device | str = "cpu",
) -> tuple[torch.Tensor, torch.Tensor]:
    """Run the production Torchsky/Torchfits map boundary for every band."""

    if resampling_kernel not in ("bilinear", "drizzle_square"):
        raise ValueError("resampling_kernel must be 'bilinear' or 'drizzle_square'")
    bands = exposures.images.shape[1]
    products: list[torch.Tensor] = []
    weights: list[torch.Tensor] = []
    with tempfile.TemporaryDirectory(prefix="torchsky-multiband-") as directory:
        root = Path(directory)
        for band in range(bands):
            observations: list[ImageObservation] = []
            for exposure, header in enumerate(exposures.headers):
                path = root / f"exposure-{exposure:02d}-band-{band:02d}.fits"
                torchsky.write(
                    str(path),
                    exposures.images[exposure, band].to("cpu"),
                    header=header,
                    overwrite=True,
                )
                observations.append(
                    ImageObservation(
                        str(path),
                        weight=float(exposures.noise_sigma[band].reciprocal().square()),
                    )
                )
            result = coadd_image_plane(
                observations,
                target_header=exposures.target_header,
                resampling_kernel=resampling_kernel,
                value_semantics="surface_brightness",
                stacking_method="weighted_mean",
                drizzle_pixfrac=drizzle_pixfrac,
                device=device,
            )
            products.append(result.data)
            weights.append(result.weight)
    return torch.stack(products), torch.stack(weights)


def _gaussian_profiles(
    catalog: TruthCatalog,
    *,
    height: int,
    width: int,
    spatial_oversample: int,
) -> torch.Tensor:
    """Build normalized benchmark truth/measurement profiles."""

    if height < 1 or width < 1 or spatial_oversample < 1:
        raise ValueError("image dimensions and spatial_oversample must be positive")
    offsets = (
        torch.arange(
            spatial_oversample,
            dtype=catalog.xy.dtype,
            device=catalog.xy.device,
        )
        + 0.5
    ) / spatial_oversample - 0.5
    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=catalog.xy.dtype, device=catalog.xy.device),
        torch.arange(width, dtype=catalog.xy.dtype, device=catalog.xy.device),
        indexing="ij",
    )
    profiles: list[torch.Tensor] = []
    for position, sigma in zip(catalog.xy, catalog.sigma_px, strict=True):
        profile = torch.zeros_like(xx)
        for dy in offsets:
            for dx in offsets:
                radius2 = (xx + dx - position[0]).square() + (
                    yy + dy - position[1]
                ).square()
                profile += torch.exp(-0.5 * radius2 / sigma.square())
        profile /= spatial_oversample**2
        profile /= profile.sum()
        profiles.append(profile)
    return torch.stack(profiles)


def forced_linear_flux(
    images: torch.Tensor,
    catalog: TruthCatalog,
    *,
    spatial_oversample: int = 4,
) -> torch.Tensor:
    """Jointly fit source amplitudes in every band at all truth positions."""

    if images.ndim != 3 or images.shape[0] != catalog.band_flux.shape[1]:
        raise ValueError("images must have shape (catalog bands, height, width)")
    if images.device != catalog.xy.device or images.dtype != catalog.xy.dtype:
        raise ValueError("images and catalogue must share device and dtype")
    if catalog.xy.shape[0] == 0:
        return torch.empty_like(catalog.band_flux)
    profiles = _gaussian_profiles(
        catalog,
        height=images.shape[1],
        width=images.shape[2],
        spatial_oversample=spatial_oversample,
    )
    design = profiles.reshape(profiles.shape[0], -1).mT
    return torch.linalg.lstsq(design, images.reshape(images.shape[0], -1).mT).solution


def flux_color_metrics(
    measured_flux: torch.Tensor,
    truth_flux: torch.Tensor,
) -> dict[str, list[float]]:
    """Report all-injection fractional-flux and adjacent-color bias."""

    if measured_flux.shape != truth_flux.shape or measured_flux.ndim != 2:
        raise ValueError("measured_flux and truth_flux must share (sources, bands)")
    if torch.any(truth_flux < 0):
        raise ValueError("truth_flux must be non-negative")
    fractional_bias: list[float] = []
    fractional_nmad: list[float] = []
    for band in range(truth_flux.shape[1]):
        valid = truth_flux[:, band] > 0
        if not torch.any(valid):
            fractional_bias.append(float("nan"))
            fractional_nmad.append(float("nan"))
            continue
        residual = (
            measured_flux[valid, band] / truth_flux[valid, band] - 1.0
        )
        median = torch.median(residual)
        fractional_bias.append(float(median))
        fractional_nmad.append(
            float(1.4826 * torch.median(torch.abs(residual - median)))
        )

    color_bias_mag: list[float] = []
    for band in range(truth_flux.shape[1] - 1):
        valid = (
            (truth_flux[:, band] > 0)
            & (truth_flux[:, band + 1] > 0)
            & (measured_flux[:, band] > 0)
            & (measured_flux[:, band + 1] > 0)
        )
        if not torch.any(valid):
            color_bias_mag.append(float("nan"))
            continue
        measured_color = -2.5 * torch.log10(
            measured_flux[valid, band] / measured_flux[valid, band + 1]
        )
        truth_color = -2.5 * torch.log10(
            truth_flux[valid, band] / truth_flux[valid, band + 1]
        )
        color_bias_mag.append(float(torch.median(measured_color - truth_color)))
    return {
        "median_fractional_flux_bias": fractional_bias,
        "fractional_flux_nmad": fractional_nmad,
        "median_adjacent_color_bias_mag": color_bias_mag,
    }


def gaussian_nll(
    observation: torch.Tensor,
    prediction: torch.Tensor,
    noise_sigma: torch.Tensor | float,
) -> float:
    """Mean held-out detector Gaussian negative log likelihood, up to a constant."""

    if observation.shape != prediction.shape:
        raise ValueError("observation and prediction must have matching shapes")
    sigma = torch.as_tensor(
        noise_sigma, device=observation.device, dtype=observation.dtype
    )
    if sigma.ndim == 1 and observation.ndim >= 1 and sigma.shape[0] == observation.shape[0]:
        sigma = sigma.reshape((-1,) + (1,) * (observation.ndim - 1))
    if torch.any(sigma <= 0):
        raise ValueError("noise_sigma must be positive")
    standardized = (observation - prediction) / sigma
    return float(torch.mean(0.5 * standardized.square() + torch.log(sigma)))


def gaussian_template_bank(
    sigmas_px: Sequence[float],
    *,
    device: torch.device | str,
    dtype: torch.dtype,
) -> tuple[torch.Tensor, ...]:
    """Return unit-energy odd Gaussian kernels for matched filtering."""

    kernels: list[torch.Tensor] = []
    for sigma in sigmas_px:
        if sigma <= 0:
            raise ValueError("template sigmas must be positive")
        radius = max(1, int(round(4 * sigma)))
        coordinates = torch.arange(-radius, radius + 1, device=device, dtype=dtype)
        yy, xx = torch.meshgrid(coordinates, coordinates, indexing="ij")
        kernel = torch.exp(-0.5 * (xx.square() + yy.square()) / sigma**2)
        kernels.append(kernel / torch.linalg.vector_norm(kernel))
    if not kernels:
        raise ValueError("at least one template sigma is required")
    return tuple(kernels)


def weighted_matched_filter_snr(
    image: torch.Tensor,
    inverse_variance: torch.Tensor,
    template: torch.Tensor,
) -> torch.Tensor:
    """Return the spatially weighted matched-filter signal-to-noise map."""

    if (
        image.ndim != 2
        or inverse_variance.shape != image.shape
        or not image.is_floating_point()
        or not inverse_variance.is_floating_point()
        or image.device != inverse_variance.device
        or image.dtype != inverse_variance.dtype
    ):
        raise ValueError(
            "image and inverse_variance must be matching floating-point 2-D tensors"
        )
    if (
        template.ndim != 2
        or not template.is_floating_point()
        or template.device != image.device
        or template.dtype != image.dtype
        or template.shape[0] % 2 != 1
        or template.shape[1] % 2 != 1
        or not torch.isfinite(template).all()
        or not bool(torch.any(template != 0))
    ):
        raise ValueError(
            "template must be a finite, nonzero, odd-shaped tensor matching image dtype/device"
        )
    if not torch.isfinite(inverse_variance).all() or torch.any(inverse_variance < 0):
        raise ValueError("inverse_variance must be finite and non-negative")
    supported = inverse_variance > 0
    if not torch.isfinite(image[supported]).all():
        raise ValueError("image must be finite wherever inverse_variance is positive")
    safe_image = torch.where(supported, image, 0.0)
    kernel = template[None, None]
    padding = (template.shape[0] // 2, template.shape[1] // 2)
    numerator = F.conv2d(
        (safe_image * inverse_variance)[None, None],
        kernel,
        padding=padding,
    )[0, 0]
    denominator2 = F.conv2d(
        inverse_variance[None, None],
        kernel.square(),
        padding=padding,
    )[0, 0]
    return torch.where(
        denominator2 > 0,
        numerator / torch.sqrt(torch.clamp_min(denominator2, 0.0)),
        0.0,
    )


def weighted_matched_filter_response_cube(
    images: torch.Tensor,
    inverse_variances: torch.Tensor,
    psf_templates: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Return one spatially weighted matched-filter response map per band."""

    if (
        images.ndim != 3
        or inverse_variances.shape != images.shape
        or len(psf_templates) != images.shape[0]
    ):
        raise ValueError(
            "images/inverse_variances must be (bands, height, width) with one PSF per band"
        )
    return torch.stack(
        [
            weighted_matched_filter_snr(
                images[band],
                inverse_variances[band],
                template,
            )
            for band, template in enumerate(psf_templates)
        ]
    )


def weighted_chi_detection_statistic(
    images: torch.Tensor,
    inverse_variances: torch.Tensor,
    psf_templates: Sequence[torch.Tensor],
) -> torch.Tensor:
    """Combine positive per-band matched-filter S/N maps in quadrature."""

    responses = weighted_matched_filter_response_cube(
        images,
        inverse_variances,
        psf_templates,
    )
    positive = torch.clamp_min(responses, 0.0)
    return torch.sqrt(positive.square().sum(dim=0))


def calibrate_matched_filter_whitening(
    responses: torch.Tensor,
    support_mask: torch.Tensor,
    *,
    clip_sigma: float = 5.0,
    max_iterations: int = 3,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Estimate robust per-band response centers/scales from blank controls."""

    if (
        responses.ndim != 3
        or support_mask.shape != responses.shape[-2:]
        or support_mask.dtype is not torch.bool
        or support_mask.device != responses.device
        or not responses.is_floating_point()
        or not math.isfinite(clip_sigma)
        or clip_sigma <= 0
        or max_iterations < 0
    ):
        raise ValueError("invalid response cube, support mask, or clipping contract")
    centers: list[torch.Tensor] = []
    scales: list[torch.Tensor] = []
    for response in responses:
        values = response[support_mask & torch.isfinite(response)]
        if values.numel() < 16:
            raise ValueError("response whitening needs at least 16 supported pixels")
        keep = torch.ones_like(values, dtype=torch.bool)
        center = torch.median(values)
        scale = 1.4826 * torch.median(torch.abs(values - center))
        for _ in range(max_iterations):
            if not torch.isfinite(scale) or scale <= 0:
                raise ValueError("response whitening scale must be positive")
            keep = torch.abs(values - center) <= clip_sigma * scale
            if int(keep.sum()) < 16:
                raise ValueError("response whitening clipping retained too few pixels")
            center = torch.median(values[keep])
            scale = 1.4826 * torch.median(torch.abs(values[keep] - center))
        if not torch.isfinite(scale) or scale <= 0:
            raise ValueError("response whitening scale must be positive")
        centers.append(center)
        scales.append(scale)
    return torch.stack(centers), torch.stack(scales)


def combine_whitened_chi_responses(
    responses: torch.Tensor,
    centers: torch.Tensor,
    scales: torch.Tensor,
) -> torch.Tensor:
    """Center, empirically whiten, and positively combine matched-filter bands."""

    if (
        responses.ndim != 3
        or centers.shape != (responses.shape[0],)
        or scales.shape != (responses.shape[0],)
        or centers.device != responses.device
        or scales.device != responses.device
        or centers.dtype != responses.dtype
        or scales.dtype != responses.dtype
        or not torch.isfinite(centers).all()
        or not torch.isfinite(scales).all()
        or torch.any(scales <= 0)
    ):
        raise ValueError("centers/scales must be finite positive per-band tensors")
    whitened = (
        responses - centers[:, None, None]
    ) / scales[:, None, None]
    positive = torch.clamp_min(whitened, 0.0)
    return torch.sqrt(positive.square().sum(dim=0))


def stack_effective_psf(
    difference_image: torch.Tensor,
    source_xy: torch.Tensor,
    *,
    radius_px: int = 9,
) -> torch.Tensor:
    """Register and robustly stack isolated injected-minus-blank PSF responses."""

    if (
        difference_image.ndim != 2
        or not difference_image.is_floating_point()
        or not torch.isfinite(difference_image).all()
        or source_xy.ndim != 2
        or source_xy.shape[1] != 2
        or source_xy.device != difference_image.device
        or source_xy.dtype != difference_image.dtype
        or not torch.isfinite(source_xy).all()
        or source_xy.shape[0] == 0
        or not isinstance(radius_px, int)
        or radius_px < 1
    ):
        raise ValueError(
            "difference_image and nonempty source_xy must be finite matching tensors"
        )
    height, width = difference_image.shape
    if (
        torch.any(source_xy[:, 0] < radius_px)
        or torch.any(source_xy[:, 0] > width - 1 - radius_px)
        or torch.any(source_xy[:, 1] < radius_px)
        or torch.any(source_xy[:, 1] > height - 1 - radius_px)
    ):
        raise ValueError("effective-PSF stamps must lie fully inside the image")
    offsets = torch.arange(
        -radius_px,
        radius_px + 1,
        device=difference_image.device,
        dtype=difference_image.dtype,
    )
    yy, xx = torch.meshgrid(offsets, offsets, indexing="ij")
    grids = []
    for position in source_xy:
        grids.append(
            torch.stack(
                (
                    (position[0] + xx) / max(width - 1, 1) * 2.0 - 1.0,
                    (position[1] + yy) / max(height - 1, 1) * 2.0 - 1.0,
                ),
                dim=-1,
            )
        )
    # ponytail: bilinear phase registration is the small validation baseline;
    # replace it with an oversampled spatial ePSF when field variation is material.
    stamps = F.grid_sample(
        difference_image[None, None].expand(source_xy.shape[0], -1, -1, -1),
        torch.stack(grids),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    )[:, 0]
    border = torch.cat(
        (
            stamps[:, 0],
            stamps[:, -1],
            stamps[:, 1:-1, 0],
            stamps[:, 1:-1, -1],
        ),
        dim=1,
    )
    stamps = stamps - torch.median(border, dim=1).values[:, None, None]
    totals = stamps.sum(dim=(-2, -1))
    if torch.any(~torch.isfinite(totals)) or torch.any(totals <= 0):
        raise ValueError("effective-PSF stamps must have positive finite total flux")
    normalized = stamps / totals[:, None, None]
    effective = torch.median(normalized, dim=0).values
    total = effective.sum()
    if not torch.isfinite(total) or total <= 0:
        raise ValueError("stacked effective PSF must have positive finite total flux")
    return effective / total


def effective_psf_metrics(psf: torch.Tensor) -> dict[str, float]:
    """Summarize a normalized effective PSF, including resampling side lobes."""

    if (
        psf.ndim != 2
        or not psf.is_floating_point()
        or not torch.isfinite(psf).all()
        or psf.shape[0] % 2 != 1
        or psf.shape[1] % 2 != 1
    ):
        raise ValueError("psf must be a finite odd-shaped floating-point tensor")
    positive = torch.clamp_min(psf, 0.0)
    positive_total = positive.sum()
    if positive_total <= 0:
        raise ValueError("psf must have positive support")
    positive = positive / positive_total
    yy, xx = torch.meshgrid(
        torch.arange(psf.shape[0], dtype=psf.dtype, device=psf.device),
        torch.arange(psf.shape[1], dtype=psf.dtype, device=psf.device),
        indexing="ij",
    )
    center_x = (positive * xx).sum()
    center_y = (positive * yy).sum()
    radius2 = (xx - center_x).square() + (yy - center_y).square()
    sigma = torch.sqrt(0.5 * (positive * radius2).sum())
    negative_fraction = torch.clamp_min(-psf, 0.0).sum() / positive_total
    return {
        "centroid_x_px": float(center_x),
        "centroid_y_px": float(center_y),
        "gaussian_equivalent_fwhm_px": float(2.354820045 * sigma),
        "negative_lobe_fraction": float(negative_fraction),
        "peak": float(psf.max()),
        "sum": float(psf.sum()),
    }


def detection_statistic(
    images: torch.Tensor,
    noise_sigma: torch.Tensor | float,
    *,
    template_sigmas_px: Sequence[float] = (1.05, 1.8),
) -> torch.Tensor:
    """Whiten bands and return the common PSF-bank chi detection statistic."""

    if images.ndim != 3:
        raise ValueError("images must have shape (bands, height, width)")
    sigma = torch.as_tensor(noise_sigma, dtype=images.dtype, device=images.device)
    if sigma.ndim == 0:
        sigma = sigma.expand(images.shape[0])
    if sigma.shape != (images.shape[0],) or torch.any(sigma <= 0):
        raise ValueError("noise_sigma must be positive and scalar or one per band")
    whitened = images / sigma[:, None, None]
    templates = gaussian_template_bank(
        template_sigmas_px, device=images.device, dtype=images.dtype
    )
    responses: list[torch.Tensor] = []
    for kernel in templates:
        response = F.conv2d(
            whitened[:, None],
            kernel[None, None],
            padding=kernel.shape[-1] // 2,
        )[:, 0]
        responses.append(response)
    per_band = torch.stack(responses).amax(dim=0)
    return torch.sqrt(torch.clamp_min(per_band, 0.0).square().sum(dim=0))


def independent_union_statistic(
    images: torch.Tensor,
    noise_sigma: torch.Tensor | float,
    *,
    template_sigmas_px: Sequence[float] = (1.05, 1.8),
) -> torch.Tensor:
    """Return the best positive single-band/template response at every pixel."""

    if images.ndim != 3:
        raise ValueError("images must have shape (bands, height, width)")
    sigma = torch.as_tensor(noise_sigma, dtype=images.dtype, device=images.device)
    if sigma.ndim == 0:
        sigma = sigma.expand(images.shape[0])
    if sigma.shape != (images.shape[0],) or torch.any(sigma <= 0):
        raise ValueError("noise_sigma must be positive and scalar or one per band")
    whitened = images / sigma[:, None, None]
    responses = [
        F.conv2d(
            whitened[:, None],
            kernel[None, None],
            padding=kernel.shape[-1] // 2,
        )[:, 0]
        for kernel in gaussian_template_bank(
            template_sigmas_px, device=images.device, dtype=images.dtype
        )
    ]
    return torch.clamp_min(torch.stack(responses), 0.0).amax(dim=(0, 1))


def find_detections(
    statistic: torch.Tensor,
    *,
    threshold: float,
    nms_radius_px: float | None = 2.0,
    nms_radius_arcsec: float | None = None,
    pixel_scale_arcsec: float | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Find local maxima and apply deterministic Euclidean non-maximum suppression."""

    if statistic.ndim != 2 or not torch.isfinite(statistic).all():
        raise ValueError("statistic must be a finite 2-D tensor")
    if nms_radius_arcsec is not None:
        if nms_radius_px not in (None, 2.0):
            raise ValueError("set either pixel or angular NMS radius, not both")
        if pixel_scale_arcsec is None or pixel_scale_arcsec <= 0:
            raise ValueError("positive pixel_scale_arcsec is required")
        nms_radius_px = nms_radius_arcsec / pixel_scale_arcsec
    if nms_radius_px is None or nms_radius_px < 0:
        raise ValueError("nms_radius_px must be non-negative")
    pool_radius = int(math.ceil(nms_radius_px))
    window = 2 * pool_radius + 1
    pooled = F.max_pool2d(
        statistic[None, None], kernel_size=window, stride=1, padding=pool_radius
    )[0, 0]
    candidates = torch.nonzero(
        (statistic >= threshold) & (statistic == pooled), as_tuple=False
    )
    if candidates.shape[0] == 0:
        return (
            torch.empty((0, 2), dtype=statistic.dtype, device=statistic.device),
            torch.empty((0,), dtype=statistic.dtype, device=statistic.device),
        )
    scores = statistic[candidates[:, 0], candidates[:, 1]]
    # Stable tie order is score descending, then y/x ascending.
    order = sorted(
        range(candidates.shape[0]),
        key=lambda i: (
            -float(scores[i].detach().cpu()),
            int(candidates[i, 0]),
            int(candidates[i, 1]),
        ),
    )
    kept: list[int] = []
    radius2 = nms_radius_px**2
    for candidate_index in order:
        candidate = candidates[candidate_index]
        if all(
            int(torch.sum((candidate - candidates[kept_index]).square()).item())
            > radius2
            for kept_index in kept
        ):
            kept.append(candidate_index)
    kept_index = torch.tensor(kept, dtype=torch.long, device=statistic.device)
    yx = candidates[kept_index]
    return torch.stack((yx[:, 1], yx[:, 0]), dim=1).to(statistic.dtype), scores[kept_index]


def calibrate_purity_threshold(
    scores: torch.Tensor,
    is_true_detection: torch.Tensor,
    *,
    target_purity: float = 0.95,
) -> float:
    """Choose the lowest validation threshold whose measured purity reaches target."""

    if scores.ndim != 1 or is_true_detection.shape != scores.shape:
        raise ValueError("scores and labels must be matching 1-D tensors")
    if scores.numel() == 0 or not torch.isfinite(scores).all():
        raise ValueError("at least one finite score is required")
    if not 0 < target_purity <= 1:
        raise ValueError("target_purity must be in (0, 1]")
    labels = is_true_detection.to(torch.bool)
    order = torch.argsort(scores, descending=True, stable=True)
    sorted_scores = scores[order]
    cumulative_true = torch.cumsum(labels[order].to(torch.int64), dim=0)
    count = torch.arange(1, scores.numel() + 1, device=scores.device)
    purity = cumulative_true / count
    group_end = torch.ones_like(sorted_scores, dtype=torch.bool)
    group_end[:-1] = sorted_scores[:-1] != sorted_scores[1:]
    valid = torch.nonzero(
        (purity >= target_purity) & group_end, as_tuple=False
    ).flatten()
    if valid.numel() == 0:
        return float("inf")
    # Extend to the largest accepted prefix: this is the lowest valid threshold.
    index = int(valid[-1].item())
    return float(sorted_scores[index].detach().cpu())


def maximum_cardinality_match(
    truth_xy: torch.Tensor,
    detection_xy: torch.Tensor,
    *,
    radius_px: float,
) -> MatchResult:
    """Return an exact maximum-cardinality bipartite match within ``radius_px``."""

    if truth_xy.ndim != 2 or truth_xy.shape[1:] != (2,):
        raise ValueError("truth_xy must have shape (truth, 2)")
    if detection_xy.ndim != 2 or detection_xy.shape[1:] != (2,):
        raise ValueError("detection_xy must have shape (detections, 2)")
    if radius_px < 0:
        raise ValueError("radius_px must be non-negative")
    if truth_xy.device != detection_xy.device:
        raise ValueError("truth and detections must share a device")
    if not torch.isfinite(truth_xy).all() or not torch.isfinite(detection_xy).all():
        raise ValueError("coordinates must be finite")

    distances = torch.cdist(truth_xy.to(torch.float64), detection_xy.to(torch.float64))
    adjacency = [
        sorted(
            torch.nonzero(distances[truth] <= radius_px, as_tuple=False)
            .flatten()
            .tolist(),
            key=lambda detection: (float(distances[truth, detection]), detection),
        )
        for truth in range(truth_xy.shape[0])
    ]
    detection_owner = [-1] * detection_xy.shape[0]

    # ponytail: O(VE) is exact and adequate for benchmark catalogues; replace with
    # Hopcroft-Karp only if catalogue-scale profiling shows matching dominates.
    def augment(truth: int, seen: set[int]) -> bool:
        for detection in adjacency[truth]:
            if detection in seen:
                continue
            seen.add(detection)
            owner = detection_owner[detection]
            if owner < 0 or augment(owner, seen):
                detection_owner[detection] = truth
                return True
        return False

    truth_order = sorted(
        range(truth_xy.shape[0]),
        key=lambda truth: (len(adjacency[truth]), truth),
    )
    for truth in truth_order:
        augment(truth, set())

    pairs = tuple(
        sorted(
            (
                (truth, detection)
                for detection, truth in enumerate(detection_owner)
                if truth >= 0
            )
        )
    )
    paired_truth = {truth for truth, _ in pairs}
    paired_detection = {detection for _, detection in pairs}
    return MatchResult(
        pairs=pairs,
        distances_px=tuple(
            float(distances[truth, detection].detach().cpu())
            for truth, detection in pairs
        ),
        unmatched_truth=tuple(
            truth for truth in range(truth_xy.shape[0]) if truth not in paired_truth
        ),
        unmatched_detections=tuple(
            detection
            for detection in range(detection_xy.shape[0])
            if detection not in paired_detection
        ),
    )


def maximum_cardinality_angular_match(
    truth_xy: torch.Tensor,
    detection_xy: torch.Tensor,
    *,
    radius_arcsec: float,
    pixel_scale_arcsec: float,
) -> MatchResult:
    """Associate target-plane coordinates using an explicit WCS angular scale."""

    if pixel_scale_arcsec <= 0:
        raise ValueError("pixel_scale_arcsec must be positive")
    return maximum_cardinality_match(
        truth_xy,
        detection_xy,
        radius_px=radius_arcsec / pixel_scale_arcsec,
    )


def score_detections(
    truth_xy: torch.Tensor,
    detection_xy: torch.Tensor,
    *,
    radius_px: float,
    area_square_arcmin: float,
) -> dict[str, float | int]:
    """Score catalogue completeness, purity, false density, and astrometry."""

    if area_square_arcmin <= 0:
        raise ValueError("area_square_arcmin must be positive")
    match = maximum_cardinality_match(
        truth_xy, detection_xy, radius_px=radius_px
    )
    matched = len(match.pairs)
    truth_count = truth_xy.shape[0]
    detection_count = detection_xy.shape[0]
    distances = torch.tensor(match.distances_px, dtype=torch.float64)
    return {
        "truth_count": truth_count,
        "detection_count": detection_count,
        "matched_count": matched,
        "completeness": matched / truth_count if truth_count else 1.0,
        "purity": matched / detection_count if detection_count else 1.0,
        "false_detections_per_square_arcmin": len(match.unmatched_detections)
        / area_square_arcmin,
        "astrometric_median_px": float(distances.median()) if matched else float("nan"),
        "astrometric_p95_px": (
            float(torch.quantile(distances, 0.95)) if matched else float("nan")
        ),
    }


def _gate0_artifact(*, seed: int, size: int, bands: int, sources: int) -> dict[str, object]:
    manifest = make_seed_manifest(seed)
    split = split_identifiers(
        tuple(f"scene-{index:03d}" for index in range(10)),
        seed=manifest.pointing,
    )
    catalog = generate_truth_catalog(
        height=size,
        width=size,
        bands=bands,
        source_count=sources,
        seed=manifest.scene,
    )
    truth = render_truth(catalog, height=size, width=size)
    measured_flux = forced_linear_flux(truth, catalog)
    noise_sigma = 0.05
    noise = noise_sigma * torch.randn(
        truth.shape,
        generator=torch.Generator().manual_seed(manifest.noise),
        dtype=truth.dtype,
    )
    photometry = flux_color_metrics(measured_flux, catalog.band_flux)
    # ponytail: the noiseless contract is 1e-11; canonicalize sub-tolerance
    # LAPACK roundoff so deterministic JSON does not depend on the CPU driver.
    photometry = {
        name: [0.0 if abs(value) < 1e-12 else value for value in values]
        for name, values in photometry.items()
    }
    return {
        "profile": "gate0",
        "seeds": asdict(manifest),
        "shape": [bands, size, size],
        "source_count": sources,
        "source_kinds": list(catalog.kind),
        "band_flux_sum": truth.sum(dim=(-2, -1)).tolist(),
        "split": asdict(split),
        "spatial_oversample": 4,
        "spectral_samples": catalog.spectral_coordinate.numel(),
        "noiseless_forced_photometry": photometry,
        "heldout_truth_gaussian_nll": gaussian_nll(
            truth + noise, truth, noise_sigma
        ),
    }


def _classical_method_row(
    *,
    name: str,
    product: torch.Tensor,
    null_product: torch.Tensor,
    catalog: TruthCatalog,
    truth: torch.Tensor,
    product_noise_sigma: torch.Tensor,
    pixel_scale_arcsec: float,
    statistic_kind: str,
    runtime_seconds: float,
    peak_cuda_bytes: int | None,
) -> dict[str, object]:
    if statistic_kind == "independent_union":
        statistic = independent_union_statistic(product, product_noise_sigma)
        null_statistic = independent_union_statistic(
            null_product, product_noise_sigma
        )
    elif statistic_kind == "chi2":
        statistic = detection_statistic(product, product_noise_sigma)
        null_statistic = detection_statistic(null_product, product_noise_sigma)
    else:
        raise ValueError("unknown statistic_kind")
    threshold = float(null_statistic.max()) + 16 * torch.finfo(statistic.dtype).eps
    detections, _ = find_detections(
        statistic,
        threshold=threshold,
        nms_radius_px=None,
        nms_radius_arcsec=0.12,
        pixel_scale_arcsec=pixel_scale_arcsec,
    )
    height, width = product.shape[-2:]
    score = score_detections(
        catalog.xy,
        detections,
        radius_px=min(1.0, 0.5 * 2.354820045 * 1.05),
        area_square_arcmin=(
            height * pixel_scale_arcsec / 60.0
        ) * (width * pixel_scale_arcsec / 60.0),
    )
    measured_flux = forced_linear_flux(
        product.to(dtype=catalog.xy.dtype),
        catalog,
    )
    return {
        "method": name,
        "representation": (
            "classical_coadd" if statistic_kind == "independent_union"
            else "classical_detection"
        ),
        "statistic": statistic_kind,
        "threshold_from_validation_null": threshold,
        "null_detections": int(
            find_detections(
                null_statistic,
                threshold=threshold,
                nms_radius_px=None,
                nms_radius_arcsec=0.12,
                pixel_scale_arcsec=pixel_scale_arcsec,
            )[0].shape[0]
        ),
        "detection": score,
        "forced_photometry": flux_color_metrics(
            measured_flux, catalog.band_flux
        ),
        "heldout_gaussian_nll": gaussian_nll(
            product.to(truth.dtype), truth, product_noise_sigma.to(truth.dtype)
        ),
        "image_rmse": float(
            torch.sqrt(torch.mean((product.to(truth.dtype) - truth).square()))
        ),
        "runtime_seconds": runtime_seconds,
        "peak_cuda_bytes": peak_cuda_bytes,
    }


def _classical_smoke_artifact(
    *,
    seed: int,
    size: int,
    bands: int,
    sources: int,
    device: torch.device | str,
    exposure_count: int = 4,
) -> dict[str, object]:
    """Run the first production-operator classical floor without a depth claim."""

    target_device = torch.device(device)
    manifest = make_seed_manifest(seed)
    catalog = generate_truth_catalog(
        height=size,
        width=size,
        bands=bands,
        source_count=sources,
        seed=manifest.scene,
        device=target_device,
    )
    truth = render_truth(catalog, height=size, width=size)
    noise_sigma = torch.linspace(
        0.045,
        0.065,
        bands,
        device=target_device,
        dtype=truth.dtype,
    )
    exposures = simulate_dithered_exposures(
        truth,
        seed=manifest.noise,
        exposure_count=exposure_count,
        noise_sigma=noise_sigma,
    )
    null_exposures = simulate_dithered_exposures(
        torch.zeros_like(truth),
        seed=manifest.nuisance,
        exposure_count=exposure_count,
        noise_sigma=noise_sigma,
    )
    product_noise_sigma = noise_sigma / math.sqrt(exposure_count)
    rows: list[dict[str, object]] = []
    products: dict[str, tuple[torch.Tensor, torch.Tensor, float, int | None]] = {}
    for name, kernel in (
        ("weighted_bilinear", "bilinear"),
        ("drizzle_square", "drizzle_square"),
    ):
        if target_device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(target_device)
        started = time.perf_counter()
        product, _ = coadd_synthetic_exposures(
            exposures,
            resampling_kernel=kernel,
            device=target_device,
        )
        null_product, _ = coadd_synthetic_exposures(
            null_exposures,
            resampling_kernel=kernel,
            device=target_device,
        )
        if target_device.type == "cuda":
            torch.cuda.synchronize(target_device)
        runtime_seconds = time.perf_counter() - started
        peak_cuda_bytes = (
            int(torch.cuda.max_memory_allocated(target_device))
            if target_device.type == "cuda"
            else None
        )
        products[name] = (
            product,
            null_product,
            runtime_seconds,
            peak_cuda_bytes,
        )
        rows.append(
            _classical_method_row(
                name=name,
                product=product,
                null_product=null_product,
                catalog=catalog,
                truth=truth,
                product_noise_sigma=product_noise_sigma,
                pixel_scale_arcsec=0.04,
                statistic_kind="independent_union",
                runtime_seconds=runtime_seconds,
                peak_cuda_bytes=peak_cuda_bytes,
            )
        )
    bilinear, null_bilinear, coadd_runtime, peak_cuda_bytes = products[
        "weighted_bilinear"
    ]
    started = time.perf_counter()
    chi2_row = _classical_method_row(
        name="chi2_coadd",
        product=bilinear,
        null_product=null_bilinear,
        catalog=catalog,
        truth=truth,
        product_noise_sigma=product_noise_sigma,
        pixel_scale_arcsec=0.04,
        statistic_kind="chi2",
        runtime_seconds=0.0,
        peak_cuda_bytes=peak_cuda_bytes,
    )
    if target_device.type == "cuda":
        torch.cuda.synchronize(target_device)
    chi2_row["runtime_seconds"] = coadd_runtime + time.perf_counter() - started
    chi2_row["reuses_product"] = "weighted_bilinear"
    rows.append(chi2_row)
    return {
        "profile": "classical-smoke",
        "depth_claim": False,
        "seeds": asdict(manifest),
        "shape": [bands, size, size],
        "source_count": sources,
        "exposure_count": exposure_count,
        "pixel_scale_arcsec": 0.04,
        "dither_xy_px": exposures.dither_xy_px.tolist(),
        "input_noise_sigma": noise_sigma.tolist(),
        "propagated_product_noise_sigma": product_noise_sigma.tolist(),
        "methods": rows,
    }


def _updec_adass_smoke_artifact(
    *,
    input_root: Path,
    bands: Sequence[str],
    device: torch.device | str,
    exposures_per_band: int,
    cutout_size: int,
    flux_jy: float,
) -> dict[str, object]:
    """Audit real SCI/ERR/DQ injection boundaries without making a depth claim."""

    if exposures_per_band < 1 or cutout_size < 16 or flux_jy <= 0:
        raise ValueError("positive exposures/flux and cutout_size >= 16 are required")
    from bench_map_smacs0723_real import (
        _center_cutout,
        _cutout_header,
        _discover_exposure_paths,
        _exposure_filter_band,
        _read_ext,
        _sci_header,
    )

    target_device = torch.device(device)
    normalized_bands = tuple(str(band).upper() for band in bands)
    rows: list[dict[str, Any]] = []
    for band in normalized_bands:
        paths = _discover_exposure_paths(input_root, input_product="cal", band=band)
        if not paths:
            paths = [
                path
                for path in sorted(input_root.rglob("*_cal.fits"))
                if _exposure_filter_band(path) == band
            ]
        if len(paths) < exposures_per_band:
            raise FileNotFoundError(
                f"need {exposures_per_band} {band} calibrated exposures under {input_root}"
            )
        contract = gaussian_psf_contract(band)
        for path in paths[:exposures_per_band]:
            science_array = _read_ext(path, "SCI")
            error_array = _read_ext(path, "ERR")
            dq_array = _read_ext(path, "DQ")
            science_cutout, x0, y0 = _center_cutout(science_array, cutout_size)
            error_cutout = error_array[
                y0 : y0 + science_cutout.shape[0],
                x0 : x0 + science_cutout.shape[1],
            ]
            dq_cutout = dq_array[
                y0 : y0 + science_cutout.shape[0],
                x0 : x0 + science_cutout.shape[1],
            ]
            header = _cutout_header(
                _sci_header(path),
                x0,
                y0,
                science_cutout.shape[0],
            )
            if tuple(science_cutout.shape) != (cutout_size, cutout_size):
                raise ValueError(f"{path} is smaller than the requested cutout")
            science = torch.from_numpy(science_cutout.copy()).to(target_device)
            original = science.clone()
            error = torch.from_numpy(error_cutout.copy()).to(target_device)
            dq = torch.from_numpy(dq_cutout.copy()).to(target_device)
            valid = (
                torch.isfinite(science)
                & torch.isfinite(error)
                & (error > 0)
                & (dq == 0)
            )
            inverse_variance = torch.where(
                valid,
                error.square().reciprocal(),
                torch.zeros_like(error),
            )
            center = torch.tensor(
                [(cutout_size - 1) * 0.5, (cutout_size - 1) * 0.5],
                dtype=torch.float64,
                device=target_device,
            )
            ra, dec = torchsky.WCS(header).pixel_to_world(center[0], center[1])
            world = torch.stack((ra, dec))
            injection = inject_gaussian_sources_mjysr(
                science,
                header=header,
                world_radec_deg=world[None],
                flux_jy=torch.tensor(
                    [flux_jy], dtype=torch.float64, device=target_device
                ),
                psf_fwhm_arcsec=contract.truth_fwhm_arcsec,
                spatial_oversample=contract.spatial_oversample,
            )
            yy, xx = torch.meshgrid(
                torch.arange(cutout_size, device=target_device),
                torch.arange(cutout_size, device=target_device),
                indexing="ij",
            )
            pixel_scale_arcsec = float(
                torch.sqrt(
                    pixel_area_sr_at(
                        torchsky.WCS(header),
                        injection.source_xy[0, 0],
                        injection.source_xy[0, 1],
                    )
                )
                * (180.0 / torch.pi)
                * 3600.0
            )
            source_mask = (
                (xx - injection.source_xy[0, 0]).square()
                + (yy - injection.source_xy[0, 1]).square()
            ) > (2.0 * contract.truth_fwhm_arcsec / pixel_scale_arcsec) ** 2
            background_mask = valid & source_mask
            control_background = fit_background_plane(
                science,
                background_mask,
                weight=inverse_variance,
                sigma_clip=4.0,
            )
            injected_background = fit_background_plane(
                injection.image,
                background_mask,
                weight=inverse_variance,
                sigma_clip=4.0,
            )
            recovered_delta = (
                injection.image
                - injected_background.model
                - (science - control_background.model)
            )
            exact_flux = measure_gaussian_source_flux_jy(
                recovered_delta,
                header=header,
                world_radec_deg=world,
                psf_fwhm_arcsec=contract.truth_fwhm_arcsec,
                spatial_oversample=contract.spatial_oversample,
            )
            perturbed_flux = measure_gaussian_source_flux_jy(
                recovered_delta,
                header=header,
                world_radec_deg=world,
                psf_fwhm_arcsec=contract.fitted_fwhm_arcsec,
                spatial_oversample=contract.spatial_oversample,
            )
            support_y, support_x = torch.nonzero(
                injection.delta != 0,
                as_tuple=True,
            )
            support_area = pixel_area_sr_at(
                torchsky.WCS(header),
                support_x.to(torch.float64),
                support_y.to(torch.float64),
            )
            support_flux = (
                injection.delta[support_y, support_x].to(torch.float64)
                * support_area
                * 1.0e6
            )
            centroid = torch.stack(
                (
                    (support_x * support_flux).sum() / support_flux.sum(),
                    (support_y * support_flux).sum() / support_flux.sum(),
                )
            )
            position_error_px = float(
                torch.linalg.vector_norm(centroid - injection.source_xy[0])
            )
            primary_header = dict(torchsky.get_header(str(path), 0))
            patt_num = int(primary_header["PATT_NUM"])
            rows.append(
                {
                    "band": band,
                    "path": str(path.resolve()),
                    "patt_num": patt_num,
                    "split": updec_patt_split(patt_num),
                    "cutout_xy0": [x0, y0],
                    "shape": list(science.shape),
                    "bunit": header.get("BUNIT"),
                    "valid_fraction": float(valid.float().mean()),
                    "median_inverse_variance": float(
                        torch.median(inverse_variance[valid])
                    ),
                    "input_unchanged": bool(
                        torch.allclose(science, original, equal_nan=True)
                    ),
                    "source_xy": injection.source_xy[0].tolist(),
                    "recovered_centroid_xy": centroid.tolist(),
                    "position_error_px": position_error_px,
                    "source_accepted": bool(injection.accepted[0]),
                    "injected_flux_jy": flux_jy,
                    "rendered_flux_jy": float(injection.rendered_flux_jy[0]),
                    "exact_psf_flux_jy": exact_flux,
                    "exact_psf_fractional_error": exact_flux / flux_jy - 1.0,
                    "perturbed_psf_flux_jy": perturbed_flux,
                    "perturbed_psf_fractional_error": perturbed_flux / flux_jy - 1.0,
                    "psf_contract": asdict(contract),
                    "background": {
                        "model": "weighted_sigma_clipped_plane",
                        "sigma_clip": 4.0,
                        "control_coeffs": control_background.coeffs.tolist(),
                        "injected_coeffs": injected_background.coeffs.tolist(),
                        "coefficient_shift": (
                            injected_background.coeffs
                            - control_background.coeffs
                        ).tolist(),
                    },
                }
            )
    return {
        "profile": "updec-adass-smoke",
        "depth_claim": False,
        "input_root": str(input_root.resolve()),
        "bands": list(normalized_bands),
        "exposures_per_band": exposures_per_band,
        "cutout_size": cutout_size,
        "device": str(target_device),
        "patt_split": {
            name: [
                patt_num
                for patt_num, split_name in _UPDEC_PATT_SPLIT.items()
                if split_name == name
            ]
            for name in ("train", "validation", "test")
        },
        "rows": rows,
    }


def _updec_effective_psf_artifact(
    *,
    positive_product_root: Path,
    blank_product_root: Path,
    injection_manifest_path: Path,
    output_dir: Path,
    bands: Sequence[str],
    methods: Sequence[str],
    radius_px: int,
) -> dict[str, object]:
    """Measure method/band effective PSFs from injected-minus-blank products."""

    if radius_px < 1 or not bands or not methods:
        raise ValueError("positive radius, bands, and methods are required")
    manifest = json.loads(injection_manifest_path.read_text())
    world = torch.as_tensor(manifest["world_radec_deg"], dtype=torch.float64)
    if world.ndim != 2 or world.shape[1] != 2 or world.shape[0] == 0:
        raise ValueError("effective-PSF manifest needs nonempty world_radec_deg")
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    for band in (str(value).upper() for value in bands):
        for method in methods:
            relative = Path(band) / "products" / f"{method}_sci.fits"
            positive_path = (positive_product_root / relative).resolve()
            blank_path = (blank_product_root / relative).resolve()
            positive, positive_header = torchsky.read(
                str(positive_path),
                return_header=True,
            )
            blank, blank_header = torchsky.read(
                str(blank_path),
                return_header=True,
            )
            if (
                positive.shape != blank.shape
                or positive.ndim != 2
                or positive.dtype != blank.dtype
            ):
                raise ValueError(f"positive/blank product mismatch for {band}/{method}")
            source_xy = torchsky.WCS(dict(positive_header)).world_to_pixel(world)
            blank_xy = torchsky.WCS(dict(blank_header)).world_to_pixel(world)
            if not torch.allclose(source_xy, blank_xy, rtol=0.0, atol=1e-6):
                raise ValueError(f"positive/blank WCS mismatch for {band}/{method}")
            effective = stack_effective_psf(
                positive.to(torch.float64) - blank.to(torch.float64),
                source_xy,
                radius_px=radius_px,
            ).to(torch.float32)
            output_path = output_dir / f"{band}_{method}_effective_psf.fits"
            torchsky.write(
                str(output_path),
                effective,
                header={
                    "BUNIT": "1",
                    "PSFTYPE": "EFFECTIVE",
                    "FILTER": band,
                    "METHOD": str(method),
                    "RADIUS": radius_px,
                },
                overwrite=True,
            )
            rows.append(
                {
                    "band": band,
                    "method": str(method),
                    "positive_path": str(positive_path),
                    "blank_path": str(blank_path),
                    "effective_psf_path": str(output_path.resolve()),
                    "effective_psf_sha256": hashlib.sha256(
                        output_path.read_bytes()
                    ).hexdigest(),
                    **effective_psf_metrics(effective),
                }
            )
    return {
        "schema_version": 1,
        "profile": "updec-adass-effective-psf",
        "depth_claim": False,
        "injection_manifest": str(injection_manifest_path.resolve()),
        "source_count": int(world.shape[0]),
        "radius_px": radius_px,
        "rows": rows,
    }


def _updec_classical_detection_artifact(
    *,
    product_root: Path,
    effective_psf_root: Path,
    support_mask_path: Path,
    bands: Sequence[str],
    method: str,
    detection_threshold: float,
    background_block_size: int,
    statistic_out: Path,
    whitening_path: Path | None = None,
) -> dict[str, object]:
    """Run one common background/matched-filter detector on classical products."""

    if (
        not bands
        or not method
        or not math.isfinite(detection_threshold)
        or detection_threshold <= 0
        or background_block_size < 8
    ):
        raise ValueError("invalid classical detection contract")
    support = torchsky.read(str(support_mask_path)).to(torch.bool)
    if support.ndim != 2 or not bool(support.any()):
        raise ValueError("support mask must be a nonempty 2-D FITS image")
    estimator = MeshSkyEstimator(
        block_size=background_block_size,
        block_stride=max(1, background_block_size // 2),
        statistic="median",
        mesh_filter="median",
        interpolate="bicubic",
        self_mask_sigma=3.0,
    )
    images: list[torch.Tensor] = []
    inverse_variances: list[torch.Tensor] = []
    psfs: list[torch.Tensor] = []
    band_rows: list[dict[str, object]] = []
    target_header: dict[str, object] | None = None
    for band_value in bands:
        band = str(band_value).upper()
        science_path = (
            product_root / band / "products" / f"{method}_sci.fits"
        ).resolve()
        weight_path = (
            product_root / band / "products" / f"{method}_wht.fits"
        ).resolve()
        psf_path = (
            effective_psf_root / f"{band}_{method}_effective_psf.fits"
        ).resolve()
        image, header = torchsky.read(str(science_path), return_header=True)
        weight = torchsky.read(str(weight_path))
        psf = torchsky.read(str(psf_path))
        if (
            image.shape != support.shape
            or weight.shape != support.shape
            or image.dtype != weight.dtype
            or psf.ndim != 2
        ):
            raise ValueError(f"product/support/PSF mismatch for {band}/{method}")
        valid = (
            support
            & torch.isfinite(image)
            & torch.isfinite(weight)
            & (weight > 0)
        )
        if int(valid.sum()) < 16:
            raise ValueError(f"insufficient supported pixels for {band}/{method}")
        safe_image = torch.where(valid, image, 0.0)
        sky = estimator(safe_image, ~valid)
        residual = torch.where(valid, safe_image - sky, 0.0)
        images.append(residual)
        inverse_variances.append(torch.where(valid, weight, 0.0))
        psfs.append(psf.to(dtype=image.dtype))
        residual_values = residual[valid]
        residual_center = torch.median(residual_values)
        residual_nmad = 1.4826 * torch.median(
            torch.abs(residual_values - residual_center)
        )
        band_rows.append(
            {
                "band": band,
                "science_path": str(science_path),
                "weight_path": str(weight_path),
                "effective_psf_path": str(psf_path),
                "supported_pixel_count": int(valid.sum()),
                "residual_median": float(residual_center),
                "residual_nmad": float(residual_nmad),
            }
        )
        if target_header is None:
            target_header = dict(header)
        else:
            probe = torch.tensor(
                [
                    [0.0, 0.0],
                    [support.shape[1] - 1.0, support.shape[0] - 1.0],
                    [
                        0.5 * (support.shape[1] - 1.0),
                        0.5 * (support.shape[0] - 1.0),
                    ],
                ],
                dtype=torch.float64,
            )
            if not torch.allclose(
                torchsky.WCS(target_header).pixel_to_world(probe),
                torchsky.WCS(dict(header)).pixel_to_world(probe),
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError("classical detection products do not share one WCS")
    image_cube = torch.stack(images)
    inverse_variance_cube = torch.stack(inverse_variances)
    response_cube = weighted_matched_filter_response_cube(
        image_cube,
        inverse_variance_cube,
        psfs,
    )
    if whitening_path is None:
        centers, scales = calibrate_matched_filter_whitening(
            response_cube,
            support,
        )
        whitening_source = "self_blank"
    else:
        whitening = json.loads(whitening_path.read_text())
        centers = torch.as_tensor(
            whitening["whitening"]["centers"],
            dtype=response_cube.dtype,
        )
        scales = torch.as_tensor(
            whitening["whitening"]["scales"],
            dtype=response_cube.dtype,
        )
        whitening_source = str(whitening_path.resolve())
    statistic = combine_whitened_chi_responses(
        response_cube,
        centers,
        scales,
    )
    statistic = torch.where(support, statistic, 0.0)
    detections, scores = find_detections(
        statistic,
        threshold=detection_threshold,
        nms_radius_px=2.0,
    )
    assert target_header is not None
    world = torchsky.WCS(target_header).pixel_to_world(detections)
    statistic_out = statistic_out.expanduser().resolve()
    statistic_out.parent.mkdir(parents=True, exist_ok=True)
    torchsky.write(
        str(statistic_out),
        statistic,
        header=target_header,
        overwrite=True,
    )
    return {
        "schema_version": 1,
        "profile": "updec-adass-classical-detection",
        "depth_claim": False,
        "method": method,
        "bands": [str(value).upper() for value in bands],
        "support_mask_path": str(support_mask_path.resolve()),
        "background": {
            "model": "mesh",
            "block_size": background_block_size,
            "block_stride": max(1, background_block_size // 2),
            "statistic": "median",
            "mesh_filter": "median",
            "self_mask_sigma": 3.0,
        },
        "whitening": {
            "source": whitening_source,
            "centers": centers.tolist(),
            "scales": scales.tolist(),
        },
        "detection_threshold": detection_threshold,
        "nms_radius_px": 2.0,
        "detection_count": int(detections.shape[0]),
        "detection_xy": detections.tolist(),
        "world_radec_deg": world.tolist(),
        "scores": scores.tolist(),
        "statistic_path": str(statistic_out),
        "statistic_sha256": hashlib.sha256(
            statistic_out.read_bytes()
        ).hexdigest(),
        "band_rows": band_rows,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--profile",
        choices=(
            "gate0",
            "classical-smoke",
            "updec-adass-smoke",
            "updec-adass-manifest",
            "updec-adass-effective-psf",
            "updec-adass-classical-detection",
        ),
        default="gate0",
    )
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--exposures", type=int, default=4)
    parser.add_argument("--input-root", type=Path, default=Path("data/jwst_smacs0723"))
    parser.add_argument("--updec-bands", default=None)
    parser.add_argument("--cutout-size", type=int, default=None)
    parser.add_argument("--flux-jy", type=float, default=1.0e-7)
    parser.add_argument("--reference-path", type=Path, default=None)
    parser.add_argument("--crop-x0", type=int, default=112)
    parser.add_argument("--crop-y0", type=int, default=0)
    parser.add_argument("--batch-sources", type=int, default=28)
    parser.add_argument("--batch-id", default="validation-000")
    parser.add_argument("--margin-px", type=float, default=12.0)
    parser.add_argument("--min-separation-px", type=float, default=8.0)
    parser.add_argument(
        "--ab-magnitudes",
        default="27,27.5,28,28.5,29,29.5,30",
    )
    parser.add_argument(
        "--support-weight-paths",
        default=None,
        help="Comma-separated final weight maps whose positive support is intersected",
    )
    parser.add_argument("--support-weight-threshold", type=float, default=0.0)
    parser.add_argument("--support-erode-radius-px", type=int, default=9)
    parser.add_argument("--support-min-footprint-fraction", type=float, default=1.0)
    parser.add_argument("--support-footprint-fraction-grid", default="")
    parser.add_argument("--support-mask-out", type=Path, default=None)
    parser.add_argument("--avoidance-catalogue", type=Path, default=None)
    parser.add_argument("--avoidance-radius-px", type=float, default=0.0)
    parser.add_argument(
        "--target-mode",
        choices=("published_native", "published_swarp_tan"),
        default="published_swarp_tan",
    )
    parser.add_argument("--positive-product-root", type=Path, default=None)
    parser.add_argument("--blank-product-root", type=Path, default=None)
    parser.add_argument("--product-root", type=Path, default=None)
    parser.add_argument("--effective-psf-root", type=Path, default=None)
    parser.add_argument("--support-mask-path", type=Path, default=None)
    parser.add_argument("--injection-manifest", type=Path, default=None)
    parser.add_argument("--methods", default="")
    parser.add_argument("--method", default="")
    parser.add_argument("--psf-radius-px", type=int, default=9)
    parser.add_argument(
        "--psf-out-dir",
        type=Path,
        default=Path("benchmark_results/effective_psfs"),
    )
    parser.add_argument("--whitening-json", type=Path, default=None)
    parser.add_argument("--detection-threshold", type=float, default=5.0)
    parser.add_argument("--background-block-size", type=int, default=64)
    parser.add_argument(
        "--statistic-out",
        type=Path,
        default=Path("benchmark_results/classical_detection_statistic.fits"),
    )
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/multiband_depth_gate0.json"),
    )
    args = parser.parse_args()
    if args.profile == "gate0":
        artifact = _gate0_artifact(
            seed=args.seed, size=args.size, bands=args.bands, sources=args.sources
        )
    elif args.profile == "classical-smoke":
        artifact = _classical_smoke_artifact(
            seed=args.seed,
            size=args.size,
            bands=args.bands,
            sources=args.sources,
            device=args.device,
            exposure_count=args.exposures,
        )
    elif args.profile == "updec-adass-smoke":
        smoke_bands = args.updec_bands or "F150W,F277W"
        artifact = _updec_adass_smoke_artifact(
            input_root=args.input_root,
            bands=tuple(
                value.strip()
                for value in smoke_bands.split(",")
                if value.strip()
            ),
            device=args.device,
            exposures_per_band=args.exposures,
            cutout_size=args.cutout_size or 32,
            flux_jy=args.flux_jy,
        )
    elif args.profile == "updec-adass-effective-psf":
        if (
            args.positive_product_root is None
            or args.blank_product_root is None
            or args.injection_manifest is None
        ):
            raise ValueError(
                "effective-PSF profile requires positive/blank roots and a manifest"
            )
        artifact = _updec_effective_psf_artifact(
            positive_product_root=args.positive_product_root,
            blank_product_root=args.blank_product_root,
            injection_manifest_path=args.injection_manifest,
            output_dir=args.psf_out_dir,
            bands=tuple(
                value.strip()
                for value in (args.updec_bands or ",".join(_UPDEC_BANDS)).split(",")
                if value.strip()
            ),
            methods=tuple(
                value.strip() for value in args.methods.split(",") if value.strip()
            ),
            radius_px=args.psf_radius_px,
        )
    elif args.profile == "updec-adass-classical-detection":
        if (
            args.product_root is None
            or args.effective_psf_root is None
            or args.support_mask_path is None
        ):
            raise ValueError(
                "classical detection requires product, PSF, and support roots"
            )
        artifact = _updec_classical_detection_artifact(
            product_root=args.product_root,
            effective_psf_root=args.effective_psf_root,
            support_mask_path=args.support_mask_path,
            bands=tuple(
                value.strip()
                for value in (args.updec_bands or ",".join(_UPDEC_BANDS)).split(",")
                if value.strip()
            ),
            method=args.method,
            detection_threshold=args.detection_threshold,
            background_block_size=args.background_block_size,
            statistic_out=args.statistic_out,
            whitening_path=args.whitening_json,
        )
    else:
        if args.reference_path is None:
            raise ValueError("--reference-path is required for updec-adass-manifest")
        from bench_map_smacs0723_real import _cutout_header, _sci_header
        from bench_map_smacs0723_stackers import _north_up_tan_header

        target_header = dict(
            _cutout_header(
                _sci_header(args.reference_path),
                args.crop_x0,
                args.crop_y0,
                args.cutout_size or 512,
            )
        )
        if args.target_mode == "published_swarp_tan":
            target_header = _north_up_tan_header(target_header)
        support_paths = tuple(
            Path(value.strip()).expanduser().resolve()
            for value in (args.support_weight_paths or "").split(",")
            if value.strip()
        )
        support_weights: list[torch.Tensor] = []
        target_shape = (
            int(target_header["NAXIS2"]),
            int(target_header["NAXIS1"]),
        )
        target_wcs = torchsky.WCS(target_header)
        target_probe = torch.tensor(
            [
                [0.0, 0.0],
                [target_shape[1] - 1.0, 0.0],
                [0.0, target_shape[0] - 1.0],
                [target_shape[1] - 1.0, target_shape[0] - 1.0],
                [0.5 * (target_shape[1] - 1.0), 0.5 * (target_shape[0] - 1.0)],
            ],
            dtype=torch.float64,
        )
        target_world = target_wcs.pixel_to_world(target_probe)
        for path in support_paths:
            weight, header = torchsky.read(
                str(path),
                return_header=True,
            )
            if weight.shape != target_shape:
                raise ValueError(f"support weight map has wrong shape: {path}")
            weight_world = torchsky.WCS(dict(header)).pixel_to_world(target_probe)
            if not torch.allclose(
                weight_world,
                target_world,
                rtol=0.0,
                atol=1e-9,
            ):
                raise ValueError(f"support weight map has a different target WCS: {path}")
            support_weights.append(weight)
        valid_mask = (
            common_support_mask(
                support_weights,
                threshold=args.support_weight_threshold,
                erode_radius_px=args.support_erode_radius_px,
                min_footprint_fraction=args.support_min_footprint_fraction,
            )
            if support_weights
            else None
        )
        raw_support = (
            common_support_mask(
                support_weights,
                threshold=args.support_weight_threshold,
            )
            if support_weights
            else None
        )
        pre_avoidance_pixel_count = (
            int(valid_mask.sum()) if valid_mask is not None else None
        )
        avoidance_catalogue_path: Path | None = None
        avoidance_source_count = 0
        if args.avoidance_catalogue is not None:
            if valid_mask is None or args.avoidance_radius_px <= 0:
                raise ValueError(
                    "avoidance catalogue requires support weights and a positive radius"
                )
            avoidance_catalogue_path = (
                args.avoidance_catalogue.expanduser().resolve()
            )
            avoidance_catalogue = json.loads(
                avoidance_catalogue_path.read_text()
            )
            avoidance_world = torch.as_tensor(
                avoidance_catalogue["world_radec_deg"],
                dtype=torch.float64,
            )
            if avoidance_world.numel() == 0:
                avoidance_world = avoidance_world.reshape(0, 2)
            if avoidance_world.ndim != 2 or avoidance_world.shape[1] != 2:
                raise ValueError(
                    "avoidance catalogue world_radec_deg must have shape (N, 2)"
                )
            avoidance_xy = target_wcs.world_to_pixel(avoidance_world)
            valid_mask = exclude_catalogue_neighborhoods(
                valid_mask,
                avoidance_xy,
                radius_px=args.avoidance_radius_px,
            )
            avoidance_source_count = int(avoidance_xy.shape[0])
        footprint_fraction_grid = tuple(
            float(value.strip())
            for value in args.support_footprint_fraction_grid.split(",")
            if value.strip()
        )
        footprint_fraction_sweep: list[dict[str, object]] = []
        for fraction in footprint_fraction_grid:
            candidate_mask = common_support_mask(
                support_weights,
                threshold=args.support_weight_threshold,
                erode_radius_px=args.support_erode_radius_px,
                min_footprint_fraction=fraction,
            )
            split_y = target_shape[0] // 2
            split_x = target_shape[1] // 2
            footprint_fraction_sweep.append(
                {
                    "min_footprint_fraction": fraction,
                    "pixel_count": int(candidate_mask.sum()),
                    "fraction": float(candidate_mask.to(torch.float64).mean()),
                    "quadrant_pixel_count": [
                        int(candidate_mask[:split_y, :split_x].sum()),
                        int(candidate_mask[:split_y, split_x:].sum()),
                        int(candidate_mask[split_y:, :split_x].sum()),
                        int(candidate_mask[split_y:, split_x:].sum()),
                    ],
                }
            )
        support_mask_path: Path | None = None
        support_mask_sha256: str | None = None
        if valid_mask is not None and args.support_mask_out is not None:
            support_mask_path = args.support_mask_out.expanduser().resolve()
            support_mask_path.parent.mkdir(parents=True, exist_ok=True)
            torchsky.write(
                str(support_mask_path),
                valid_mask.to(torch.uint8),
                header=target_header,
                overwrite=True,
            )
            support_mask_sha256 = hashlib.sha256(
                support_mask_path.read_bytes()
            ).hexdigest()
        artifact = make_updec_common_world_manifest(
            target_header=target_header,
            bands=tuple(
                value.strip()
                for value in (args.updec_bands or ",".join(_UPDEC_BANDS)).split(",")
                if value.strip()
            ),
            source_count=args.batch_sources,
            seed=args.seed,
            ab_magnitudes=tuple(
                float(value.strip())
                for value in args.ab_magnitudes.split(",")
                if value.strip()
            ),
            batch_id=args.batch_id,
            margin_px=args.margin_px,
            min_separation_px=args.min_separation_px,
            valid_mask=valid_mask,
        )
        artifact["reference_path"] = str(args.reference_path.resolve())
        artifact["target_crop"] = {
            "size": args.cutout_size or 512,
            "x0": args.crop_x0,
            "y0": args.crop_y0,
            "target_mode": args.target_mode,
        }
        artifact["common_support"] = (
            {
                "weight_paths": [str(path) for path in support_paths],
                "weight_threshold": args.support_weight_threshold,
                "erode_radius_px": args.support_erode_radius_px,
                "min_footprint_fraction": args.support_min_footprint_fraction,
                "pre_avoidance_pixel_count": pre_avoidance_pixel_count,
                "avoidance_catalogue": (
                    str(avoidance_catalogue_path)
                    if avoidance_catalogue_path is not None
                    else None
                ),
                "avoidance_source_count": avoidance_source_count,
                "avoidance_radius_px": args.avoidance_radius_px,
                "pixel_count": int(valid_mask.sum()),
                "fraction": float(valid_mask.to(torch.float64).mean()),
                "raw_intersection_pixel_count": int(raw_support.sum()),
                "raw_intersection_fraction": float(
                    raw_support.to(torch.float64).mean()
                ),
                "individual": [
                    {
                        "path": str(path),
                        "pixel_count": int(
                            (
                                torch.isfinite(weight)
                                & (weight > args.support_weight_threshold)
                            ).sum()
                        ),
                    }
                    for path, weight in zip(
                        support_paths, support_weights, strict=True
                    )
                ],
                "footprint_fraction_sweep": footprint_fraction_sweep,
                "quadrant_pixel_count": [
                    int(
                        valid_mask[
                            : target_shape[0] // 2,
                            : target_shape[1] // 2,
                        ].sum()
                    ),
                    int(
                        valid_mask[
                            : target_shape[0] // 2,
                            target_shape[1] // 2 :,
                        ].sum()
                    ),
                    int(
                        valid_mask[
                            target_shape[0] // 2 :,
                            : target_shape[1] // 2,
                        ].sum()
                    ),
                    int(
                        valid_mask[
                            target_shape[0] // 2 :,
                            target_shape[1] // 2 :,
                        ].sum()
                    ),
                ],
                "mask_path": (
                    str(support_mask_path) if support_mask_path is not None else None
                ),
                "mask_sha256": support_mask_sha256,
            }
            if valid_mask is not None
            else None
        )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
