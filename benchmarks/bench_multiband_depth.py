#!/usr/bin/env python3
"""Gate-0 helpers for the pre-registered multi-band depth experiment.

This module is intentionally benchmark-local.  It provides deterministic truth,
rendering, detection, association, and catalogue scoring without committing a
source-extraction API to :mod:`torchsky`.
"""

from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import math
from pathlib import Path
from typing import Sequence

import torch
import torch.nn.functional as F


_SOURCE_KINDS = (
    "continuum_point",
    "compact_galaxy",
    "emission_line",
    "blue_dropout",
    "red_dropout",
)


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


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--profile", choices=("gate0",), default="gate0")
    parser.add_argument("--seed", type=int, default=101)
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--bands", type=int, default=6)
    parser.add_argument("--sources", type=int, default=24)
    parser.add_argument(
        "--json-out",
        type=Path,
        default=Path("benchmark_results/multiband_depth_gate0.json"),
    )
    args = parser.parse_args()
    artifact = _gate0_artifact(
        seed=args.seed, size=args.size, bands=args.bands, sources=args.sources
    )
    args.json_out.parent.mkdir(parents=True, exist_ok=True)
    args.json_out.write_text(json.dumps(artifact, indent=2, sort_keys=True) + "\n")
    print(json.dumps(artifact, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
