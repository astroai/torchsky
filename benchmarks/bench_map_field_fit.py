#!/usr/bin/env python3
"""Benchmark differentiable latent-field fitting against synthetic truth."""

from __future__ import annotations

import argparse
import json
import math
import resource
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import Tensor, nn

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import torchsky  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldDomain,
    FieldObservation,
    ImageObservation,
    LearnedSpectralBasisField,
    MapProductResult,
    SkyEncoderBandpassField,
    SkyEncoderMLPField,
    SkyEncoderResMLPField,
    SineImplicitField,
    SpectralBasisField,
    estimate_map_inverse_variance,
    estimate_field_domain,
    evaluate_map_frontier_regions,
    evaluate_map_uncertainty,
    evaluate_map_product,
    fit_field_model,
    make_field_model,
    check_map_benchmark_regressions,
    render_field_cube,
    render_field_observation,
    render_field_observation_distribution,
    render_field_product,
    plan_map_product,
    spectral_data_budget,
    stack_map_product,
    suggest_num_spectral_samples,
    suggest_spectral_model_capacity,
    write_map_benchmark_artifacts,
)
from torchsky.map.field import _BaseFieldModel  # noqa: E402
from torchsky.spectral import SpectralAxis  # noqa: E402


DEFAULT_CASES = (
    "tan_sip_field_fit_multicolor",
    "tan_sip_field_fit_deep_multiband_broad",
    "tan_field_fit_exact_transmission",
    "tpv_field_fit_deep_multiband_broad",
    "tpv_field_fit_deep_multiband_exact",
    "tan_field_fit",
    "tan_field_fit_analytic",
    "tan_field_fit_low_overlap_depth",
    "tan_field_fit_psf_mismatch",
    "tpv_field_fit",
)
DEFAULT_MODEL_KINDS = (
    "sine_implicit",
    "wire_implicit",
    "gaussian_basis",
    "spectral_basis_field",
    "bandpass_conditioned_field",
    "learned_spectral_basis_field",
    "sky_encoder_bandpass_field",
    "sky_lambda_planes_field",
    "sky_lambda_feature_field",
    "sky_lambda_feature_field_hetero",
    "sky_lambda_feature_atlas_field",
    "conditioned_spectral_field",
    "hashgrid_conditioned_field",
    "hashgrid_spectral_field",
    "hashgrid_resmlp",
    "sky_encoder_mlp",
    "sky_encoder_resmlp",
    "coadd_bilinear",
    "coadd_bicubic",
)

STEP_PROFILES: dict[str, dict[str, int]] = {
    "sparse_mosaic_frontier": {
        "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit": 32,
        "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap": 16,
    },
    "sparse_mosaic_large_frontier": {
        "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit": 32,
        "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap": 12,
    },
}


def _process_peak_rss_bytes() -> int:
    peak_rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return peak_rss if sys.platform == "darwin" else peak_rss * 1024


@dataclass(frozen=True)
class ObservationSpec:
    header: dict[str, Any]
    spectral_min: float
    spectral_max: float
    transmission_center: float
    transmission_sigma: float | None
    psf_sigma_pix: float
    noise_sigma: float
    fit_psf_sigma_pix: float | None = None
    gap: tuple[slice, slice] | None = None


@dataclass(frozen=True)
class CaseSpec:
    case: str
    projection: str
    description: str
    truth_model: str
    domain: FieldDomain
    target_header: dict[str, Any]
    spectral_axis: SpectralAxis
    spectral_bin_edges: Tensor | None
    observations: tuple[ObservationSpec, ...]
    family: str = "field_fit"


def _sample_feature_volume(volume: Tensor, coords_unit: Tensor) -> Tensor:
    flat = coords_unit.reshape(-1, 3).to(dtype=volume.dtype, device=volume.device)
    grid = torch.stack(
        [
            2.0 * flat[:, 0] - 1.0,
            2.0 * flat[:, 1] - 1.0,
            2.0 * flat[:, 2] - 1.0,
        ],
        dim=-1,
    ).view(1, -1, 1, 1, 3)
    sampled = F.grid_sample(
        volume.unsqueeze(0),
        grid,
        mode="bilinear",
        padding_mode="border",
        align_corners=True,
    )
    return sampled.squeeze(0).squeeze(0).reshape(*coords_unit.shape[:-1])


class ExplicitSpectralGridField(_BaseFieldModel):
    """Benchmark-local multiframe pixel-grid baseline."""

    def __init__(
        self,
        domain: FieldDomain,
        *,
        spatial_resolution: int = 48,
        spectral_resolution: int = 8,
    ) -> None:
        super().__init__(domain)
        self.spatial_resolution = int(spatial_resolution)
        self.spectral_resolution = int(spectral_resolution)
        self.grid = nn.Parameter(
            1.0e-3
            * torch.randn(
                1,
                self.spectral_resolution,
                self.spatial_resolution,
                self.spatial_resolution,
            )
        )

    def _grid_world_coords(self) -> tuple[Tensor, Tensor, Tensor]:
        device = self.grid.device
        dtype = self.grid.dtype
        xs = torch.linspace(
            -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        ys = torch.linspace(
            -1.0, 1.0, steps=self.spatial_resolution, device=device, dtype=dtype
        )
        ls = torch.linspace(
            -1.0, 1.0, steps=self.spectral_resolution, device=device, dtype=dtype
        )
        yy, xx = torch.meshgrid(ys, xs, indexing="ij")
        wavelengths = float(self.domain.wavelength_center) + ls * float(
            self.domain.wavelength_scale
        )
        cos_dec = math.cos(math.radians(float(self.domain.center_dec_deg)))
        dra = xx * float(self.domain.spatial_scale_deg) / max(cos_dec, 1.0e-6)
        dec = yy * float(self.domain.spatial_scale_deg) + float(
            self.domain.center_dec_deg
        )
        ra = dra + float(self.domain.center_ra_deg)
        return ra, dec, wavelengths

    def initialize_from_product(self, product: MapProductResult) -> None:
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "ExplicitSpectralGridField initialization requires local_tangent"
            )
        target_wcs = torchsky.WCS(product.header)
        ra_grid, dec_grid, wavelengths = self._grid_world_coords()
        px, py = target_wcs.world_to_pixel(ra_grid, dec_grid)
        target_width = max(int(product.header["NAXIS1"]) - 1, 1)
        target_height = max(int(product.header["NAXIS2"]) - 1, 1)
        x_norm = 2.0 * (px / target_width) - 1.0
        y_norm = 2.0 * (py / target_height) - 1.0
        sample_grid = (
            torch.stack([x_norm, y_norm], dim=-1).unsqueeze(0).to(dtype=self.grid.dtype)
        )
        band_centers = []
        for band in product.product.bands:
            if band.transmission_center is not None:
                band_centers.append(float(band.transmission_center))
            elif band.spectral_min is not None and band.spectral_max is not None:
                band_centers.append(
                    0.5 * (float(band.spectral_min) + float(band.spectral_max))
                )
            else:
                raise ValueError(
                    "MapProductSpec band requires center or bounds for initialization"
                )
        centers = torch.tensor(
            band_centers, device=self.grid.device, dtype=self.grid.dtype
        )
        data = product.data.to(device=self.grid.device, dtype=self.grid.dtype)
        sampled_planes = []
        for plane in data:
            sampled = (
                F.grid_sample(
                    plane.unsqueeze(0).unsqueeze(0),
                    sample_grid,
                    mode="bilinear",
                    padding_mode="border",
                    align_corners=True,
                )
                .squeeze(0)
                .squeeze(0)
            )
            sampled_planes.append(sampled)
        sampled_cube = torch.stack(sampled_planes, dim=0)
        interp_planes = []
        for wavelength in wavelengths:
            if wavelength <= centers[0]:
                interp_planes.append(sampled_cube[0])
                continue
            if wavelength >= centers[-1]:
                interp_planes.append(sampled_cube[-1])
                continue
            upper_idx = int(torch.searchsorted(centers, wavelength).item())
            lower_idx = max(upper_idx - 1, 0)
            lam0 = centers[lower_idx]
            lam1 = centers[upper_idx]
            alpha = (wavelength - lam0) / torch.clamp(lam1 - lam0, min=1.0e-6)
            interp_planes.append(
                (1.0 - alpha) * sampled_cube[lower_idx]
                + alpha * sampled_cube[upper_idx]
            )
        init_cube = torch.stack(interp_planes, dim=0).unsqueeze(0)
        with torch.no_grad():
            self.grid.copy_(
                torch.log(
                    torch.clamp(
                        torch.expm1(torch.clamp(init_cube, min=1.0e-4)), min=1.0e-6
                    )
                )
            )

    def _coords_unit(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        if self.coord_mode != "local_tangent":
            raise ValueError(
                "ExplicitSpectralGridField requires local_tangent coordinates"
            )
        return self._unit_box_coords(ra_deg, dec_deg, wavelength_angstrom).to(
            dtype=self.grid.dtype
        )

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        coords_unit = self._coords_unit(ra_deg, dec_deg, wavelength_angstrom)
        sampled = _sample_feature_volume(self.grid, coords_unit)
        return F.softplus(sampled) + 1.0e-4

    def render_bandpass(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelengths: Tensor,
        weights: Tensor,
    ) -> Tensor:
        norm_weights = weights.to(dtype=self.grid.dtype)
        norm_weights = norm_weights / torch.clamp(norm_weights.sum(), min=1.0e-8)
        rendered = [
            self.forward(
                ra_deg,
                dec_deg,
                torch.as_tensor(
                    float(wavelength.item()),
                    device=ra_deg.device,
                    dtype=self.grid.dtype,
                ),
            )
            for wavelength in wavelengths
        ]
        return torch.sum(
            torch.stack(rendered, dim=0) * norm_weights.view(-1, *([1] * ra_deg.ndim)),
            dim=0,
        )

    family: str = "field_fit"


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


@torch.no_grad()
def _render_field_product_chunked(
    model: torch.nn.Module,
    product,
    *,
    num_spectral_samples: int = 9,
    spatial_chunk_size: int | None = None,
):
    if spatial_chunk_size is None:
        return render_field_product(
            model,
            product,
            num_spectral_samples=num_spectral_samples,
        )
    chunk = max(int(spatial_chunk_size), 1)
    height = int(product.target_header["NAXIS2"])
    width = int(product.target_header["NAXIS1"])
    shape = (height, width)
    device = next(model.parameters()).device
    planes: list[Tensor] = []
    ivar_planes: list[Tensor] = []
    for band in product.bands:
        center = (
            float(band.transmission_center)
            if band.transmission_center is not None
            else (float(band.spectral_min) + float(band.spectral_max)) / 2.0
        )
        lower = float(band.spectral_min) if band.spectral_min is not None else center
        upper = float(band.spectral_max) if band.spectral_max is not None else center
        plane = torch.empty(shape, dtype=torch.float32, device=device)
        ivar = torch.empty(shape, dtype=torch.float32, device=device)
        for y0 in range(0, height, chunk):
            for x0 in range(0, width, chunk):
                tile_h = min(chunk, height - y0)
                tile_w = min(chunk, width - x0)
                field_observation = FieldObservation(
                    image=torch.zeros(
                        (tile_h, tile_w), dtype=torch.float32, device=device
                    ),
                    header=_copy_header_with_crop(
                        dict(product.target_header),
                        x0=x0,
                        y0=y0,
                        width=tile_w,
                        height=tile_h,
                    ),
                    wavelength=center,
                    spectral_min=lower if lower != upper else None,
                    spectral_max=upper if lower != upper else None,
                    transmission_center=band.transmission_center,
                    transmission_sigma=band.transmission_sigma,
                    band_name=band.label,
                )
                if callable(
                    getattr(model, "render_bandpass_distribution", None)
                ) or callable(getattr(model, "forward_distribution", None)):
                    rendered, logvar = render_field_observation_distribution(
                        model,
                        field_observation,
                        num_spectral_samples=num_spectral_samples,
                    )
                    ivar[y0 : y0 + tile_h, x0 : x0 + tile_w] = torch.exp(-logvar).to(
                        dtype=torch.float32
                    )
                else:
                    rendered = render_field_observation(
                        model,
                        field_observation,
                        num_spectral_samples=num_spectral_samples,
                    )
                    ivar[y0 : y0 + tile_h, x0 : x0 + tile_w] = 1.0
                plane[y0 : y0 + tile_h, x0 : x0 + tile_w] = rendered.to(
                    dtype=torch.float32
                )
        planes.append(plane)
        ivar_planes.append(ivar)
    return MapProductResult(
        product=product,
        data=torch.stack(planes, dim=0),
        weight=torch.stack(ivar_planes, dim=0),
        header=dict(product.target_header),
        overlap_count=torch.ones(
            (len(planes), height, width), dtype=torch.int32, device=device
        ),
        coverage_mocs=None,
    )


def _uses_low_memory_sparse_profile(*, case: str | None, size: int | None) -> bool:
    return bool(
        case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap"
        and size is not None
        and int(size) >= 128
    )


def _field_fit_kwargs_for_model(
    model_kind: str,
    *,
    case: str | None = None,
    size: int | None = None,
    benchmark_steps: int | None = None,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lr": 5.0e-3,
        "num_spectral_samples": 9,
        "loss_kind": "charbonnier",
        "loss_reduction": "per_observation",
        "lr_schedule": "cosine",
        "grad_clip_norm": 1.0,
        "curriculum": "none",
    }
    if case in {
        "tan_field_fit_exact_transmission",
        "tpv_field_fit_deep_multiband_exact",
    }:
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
        }:
            kwargs["loss_kind"] = "mse"
            kwargs["loss_reduction"] = "global"
    if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
    if case in {
        "tan_sip_field_fit_deep_multiband_broad",
        "tpv_field_fit_deep_multiband_broad",
        "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
    }:
        kwargs["loss_kind"] = "charbonnier"
        kwargs["loss_reduction"] = "global"
        kwargs["grad_clip_norm"] = 1.0
        kwargs["training_spectral_sampling"] = "mixture_mc"
        kwargs["training_spectral_sampling_grid_size"] = 65
        kwargs["spectral_anchor_observations_per_step"] = 1
        kwargs["spectral_anchor_num_samples"] = 17
        kwargs["spectral_anchor_loss_weight"] = 0.1
    if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
        kwargs["loss_kind"] = "charbonnier"
        kwargs["loss_reduction"] = "global"
        kwargs["grad_clip_norm"] = 1.0
    if model_kind == "wire_implicit":
        kwargs["lr"] = 4.0e-3
    elif model_kind == "gaussian_basis":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 0.75
    elif model_kind == "spectral_basis_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "bandpass_conditioned_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "learned_spectral_basis_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "sky_encoder_bandpass_field":
        kwargs["lr"] = 2.5e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "sky_lambda_planes_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "sky_lambda_kplanes_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
        if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
            kwargs["loss_kind"] = "mse"
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        }:
            kwargs["lr"] = 2.5e-3
            kwargs["loss_reduction"] = "per_band"
    elif model_kind == "sky_lambda_feature_field":
        kwargs["lr"] = 2.0e-3
        kwargs["grad_clip_norm"] = 1.0
        kwargs["lr_warmup_steps"] = 4
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
        }:
            kwargs["loss_kind"] = "mse"
            kwargs["loss_reduction"] = "band_global_mix"
            kwargs["lr"] = 1.5e-3
            kwargs["lr_warmup_steps"] = 6
            kwargs["weight_decay"] = 1.0e-5
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            kwargs["loss_kind"] = "mse"
            kwargs["loss_reduction"] = "global"
            kwargs["lr"] = 1.5e-3
            kwargs["lr_warmup_steps"] = 6
            kwargs["weight_decay"] = 1.0e-5
            kwargs["observations_per_step"] = 9
            kwargs["observation_sampling"] = "band_balanced"
            kwargs["training_patch_size"] = 24
            kwargs["training_patches_per_observation"] = 1
        if case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap":
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "band_global_mix"
            kwargs["lr"] = 1.5e-3
            kwargs["lr_warmup_steps"] = 6
            kwargs["weight_decay"] = 1.0e-5
            kwargs["observations_per_step"] = 9
            kwargs["observation_sampling"] = "band_balanced"
            kwargs["training_patch_size"] = 24
            kwargs["training_patches_per_observation"] = 1
            if _uses_low_memory_sparse_profile(case=case, size=size):
                kwargs["loss_kind"] = "mse"
                kwargs["loss_reduction"] = "global"
                kwargs["training_spectral_sampling"] = "deterministic_quadrature"
                kwargs["num_spectral_samples"] = 3
                kwargs["spectral_anchor_observations_per_step"] = 0
                kwargs["spectral_anchor_loss_weight"] = 0.0
                kwargs["observations_per_step"] = 1
                kwargs["training_patch_size"] = 12
                kwargs["lr"] = (
                    3.0e-4
                    if (benchmark_steps is not None and int(benchmark_steps) >= 12)
                    else 1.0e-3
                )
                kwargs["lr_warmup_steps"] = 2
    elif model_kind == "operator_constrained_feature_field":
        kwargs["lr"] = 1.5e-3
        kwargs["grad_clip_norm"] = 1.0
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
        kwargs["lr_warmup_steps"] = 6
        kwargs["observations_per_step"] = 9
        kwargs["observation_sampling"] = "band_balanced"
        kwargs["training_patch_size"] = 24
        kwargs["training_patches_per_observation"] = 1
        kwargs["context_observations_per_step"] = 1
        kwargs["context_downsample_factor"] = 4
        kwargs["context_loss_weight"] = 0.1
        kwargs["spectral_observability_weight"] = 1.0e-3
        kwargs["spectral_observability_probe_grid_size"] = 8
        kwargs["residual_head_penalty_weight"] = 1.0e-5
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        }:
            kwargs["lr"] = 1.0e-3
            kwargs["loss_reduction"] = "band_global_mix"
            kwargs["context_downsample_factor"] = 2
            kwargs["spectral_observability_weight"] = 5.0e-4
            kwargs["residual_head_penalty_weight"] = 3.0e-6
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            kwargs["context_observations_per_step"] = 0
            kwargs["context_loss_weight"] = 0.0
        if case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap":
            kwargs["loss_reduction"] = "band_global_mix"
            kwargs["spectral_observability_weight"] = 5.0e-4
            kwargs["residual_head_penalty_weight"] = 3.0e-6
            kwargs["num_spectral_samples"] = 7
            if _uses_low_memory_sparse_profile(case=case, size=size):
                kwargs["training_spectral_sampling"] = "deterministic_quadrature"
                kwargs["num_spectral_samples"] = 3
                kwargs["observations_per_step"] = 1
                kwargs["training_patch_size"] = 12
                kwargs["lr"] = (
                    3.0e-4
                    if (benchmark_steps is not None and int(benchmark_steps) >= 12)
                    else 8.0e-4
                )
                kwargs["lr_warmup_steps"] = 2
                kwargs["context_observations_per_step"] = 1
                kwargs["context_downsample_factor"] = 4
                kwargs["context_loss_weight"] = 0.05
    elif model_kind == "explicit_spectral_grid_field":
        kwargs["lr"] = 2.5e-2
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
        kwargs["grad_clip_norm"] = 1.0
        kwargs["lr_schedule"] = "cosine"
        kwargs["lr_warmup_steps"] = 2
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
            "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
        }:
            kwargs["observations_per_step"] = 9
            kwargs["observation_sampling"] = "band_balanced"
            kwargs["training_patch_size"] = 24
            kwargs["training_patches_per_observation"] = 1
            if _uses_low_memory_sparse_profile(case=case, size=size):
                kwargs["observations_per_step"] = 1
                kwargs["training_patch_size"] = 12
                kwargs["lr"] = 1.0e-2
    elif model_kind == "sky_lambda_feature_field_hetero":
        kwargs["lr"] = 2.0e-3
        kwargs["grad_clip_norm"] = 1.0
        kwargs["lr_warmup_steps"] = 4
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        }:
            kwargs["loss_kind"] = "gaussian_nll"
            kwargs["loss_reduction"] = "per_band"
            kwargs["lr"] = 1.5e-3
            kwargs["lr_warmup_steps"] = 6
            kwargs["weight_decay"] = 1.0e-5
    elif model_kind == "sky_lambda_feature_atlas_field":
        kwargs["lr"] = 1.5e-3
        kwargs["grad_clip_norm"] = 1.0
        kwargs["lr_warmup_steps"] = 6
        kwargs["weight_decay"] = 1.0e-5
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            kwargs["loss_kind"] = "mse"
            kwargs["loss_reduction"] = "global"
    elif model_kind == "conditioned_spectral_field":
        kwargs["lr"] = 3.5e-3
        kwargs["grad_clip_norm"] = 0.9
    elif model_kind == "hashgrid_conditioned_field":
        kwargs["lr"] = 3.5e-3
        kwargs["grad_clip_norm"] = 0.85
    elif model_kind == "hashgrid_spectral_field":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.8
    elif model_kind == "hashgrid_resmlp":
        kwargs["lr"] = 2.5e-3
        kwargs["grad_clip_norm"] = 0.65
        if case in {
            "tan_sip_field_fit_deep_multiband_broad",
            "tpv_field_fit_deep_multiband_broad",
        }:
            kwargs["lr"] = 2.5e-3
            kwargs["grad_clip_norm"] = 1.0
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "global"
    elif model_kind == "strainer_sky_encoder_mlp":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.75
    elif model_kind == "sky_encoder_mlp":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.75
        if case == "tpv_field_fit_deep_multiband_exact":
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "global"
            kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "sky_encoder_resmlp":
        kwargs["lr"] = 2.5e-3
        kwargs["grad_clip_norm"] = 0.6
        if case in {
            "tan_sip_field_fit_deep_multiband_broad",
            "tpv_field_fit_deep_multiband_broad",
        }:
            kwargs["lr"] = 2.5e-3
            kwargs["grad_clip_norm"] = 1.0
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "global"
        if case == "tpv_field_fit_deep_multiband_exact":
            kwargs["lr"] = 3.0e-3
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "global"
            kwargs["grad_clip_norm"] = 1.0
    return kwargs


def _steps_for_case(
    case: str, *, base_steps: int, step_profile: str | None = None
) -> int:
    if not step_profile:
        return int(base_steps)
    profile = STEP_PROFILES.get(step_profile)
    if profile is None:
        raise ValueError(f"unknown step profile {step_profile!r}")
    return int(profile.get(case, base_steps))


def _warmup_steps_for_case(case: str, *, benchmark_steps: int) -> int:
    steps = max(int(benchmark_steps), 1)
    if "sparse_large_mosaic" in case:
        return min(2, steps)
    return min(4, steps)


def _field_model_kwargs_for_case(
    model_kind: str,
    *,
    case: str | None = None,
    size: int | None = None,
) -> dict[str, Any]:
    if model_kind == "spectral_basis_field":
        if case == "tan_field_fit_exact_transmission":
            return {
                "hidden_dim": 128,
                "hidden_layers": 2,
                "num_spatial_frequencies": 8,
                "num_spectral_frequencies": 6,
            }
        if case == "tpv_field_fit_deep_multiband_exact":
            return {
                "hidden_dim": 128,
                "hidden_layers": 2,
                "num_spatial_frequencies": 6,
                "num_spectral_frequencies": 4,
            }
    if model_kind == "learned_spectral_basis_field":
        if case in {
            "tan_field_fit_exact_transmission",
            "tpv_field_fit_deep_multiband_exact",
        }:
            return {
                "hidden_dim": 128,
                "hidden_layers": 2,
                "num_spatial_frequencies": 6,
                "basis_dim": 16,
                "spectral_hidden_dim": 32,
            }
    if model_kind == "sky_encoder_bandpass_field":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 160,
                "hidden_layers": 3,
                "num_fourier_frequencies": 8,
                "spectral_frequencies": 6,
                "num_bandpass_frequencies": 8,
                "bandpass_hidden_dim": 64,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 192,
                "hidden_layers": 3,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 6,
                "num_bandpass_frequencies": 8,
                "bandpass_hidden_dim": 64,
            }
        if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
            return {
                "hidden_dim": 160,
                "hidden_layers": 3,
                "num_fourier_frequencies": 8,
                "spectral_frequencies": 6,
                "num_bandpass_frequencies": 8,
                "bandpass_hidden_dim": 64,
            }
    if model_kind == "sky_encoder_mlp":
        if case == "tan_field_fit_exact_transmission":
            return {
                "hidden_dim": 128,
                "hidden_layers": 2,
                "num_fourier_frequencies": 6,
                "spectral_frequencies": 6,
            }
        if case == "tpv_field_fit_deep_multiband_exact":
            return {
                "hidden_dim": 192,
                "hidden_layers": 3,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 6,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact":
            return {
                "hidden_dim": 192,
                "hidden_layers": 3,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 6,
            }
    if model_kind == "sky_lambda_kplanes_field":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "base_resolution": 32,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
            return {
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
    if model_kind == "sky_lambda_feature_field":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 16,
                "scale_hidden_dim": 128,
                "scale_num_blocks": 3,
                "decoder_hidden_dim": 128,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 64,
                "bandpass_hidden_dim": 64,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "base_resolution": 32,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 160,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 160,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 64,
                "bandpass_hidden_dim": 96,
            }
        if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
            return {
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 128,
                "scale_num_blocks": 3,
                "decoder_hidden_dim": 128,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 64,
                "bandpass_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 96,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact_densebands":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 128,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 128,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap":
            if _uses_low_memory_sparse_profile(case=case, size=size):
                return {
                    "base_resolution": 24,
                    "num_scales": 3,
                    "features_per_plane": 12,
                    "scale_hidden_dim": 96,
                    "scale_num_blocks": 2,
                    "decoder_hidden_dim": 96,
                    "decoder_num_blocks": 2,
                    "refine_hidden_dim": 48,
                    "bandpass_hidden_dim": 64,
                }
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 128,
            }
    if model_kind == "operator_constrained_feature_field":
        if case == "tpv_field_fit_sparse_large_mosaic_exact":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "scale_hidden_dim": 128,
                "scale_num_blocks": 3,
                "spectral_basis_dim": 10,
                "decoder_hidden_dim": 128,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 64,
                "bandpass_hidden_dim": 64,
                "residual_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact_densebands":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "scale_hidden_dim": 128,
                "scale_num_blocks": 3,
                "spectral_basis_dim": 12,
                "decoder_hidden_dim": 128,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 64,
                "bandpass_hidden_dim": 80,
                "residual_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "scale_hidden_dim": 160,
                "scale_num_blocks": 3,
                "spectral_basis_dim": 4,
                "decoder_hidden_dim": 160,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 80,
                "bandpass_hidden_dim": 64,
                "spectral_context_dim": 96,
                "residual_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap":
            if _uses_low_memory_sparse_profile(case=case, size=size):
                return {
                    "base_resolution": 24,
                    "num_scales": 3,
                    "features_per_plane": 12,
                    "scale_hidden_dim": 96,
                    "scale_num_blocks": 2,
                    "spectral_basis_dim": 6,
                    "decoder_hidden_dim": 96,
                    "decoder_num_blocks": 2,
                    "refine_hidden_dim": 48,
                    "bandpass_hidden_dim": 64,
                    "spectral_context_dim": 64,
                    "residual_hidden_dim": 48,
                }
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "scale_hidden_dim": 160,
                "scale_num_blocks": 3,
                "spectral_basis_dim": 8,
                "decoder_hidden_dim": 160,
                "decoder_num_blocks": 3,
                "refine_hidden_dim": 80,
                "bandpass_hidden_dim": 80,
                "spectral_context_dim": 128,
                "residual_hidden_dim": 64,
            }
    if model_kind == "explicit_spectral_grid_field":
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        }:
            return {
                "spatial_resolution": max(int(size or 48), 32),
                "spectral_resolution": 8,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact_densebands":
            return {
                "spatial_resolution": max(int(size or 48), 32),
                "spectral_resolution": 12,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap":
            spatial_resolution = (
                32
                if _uses_low_memory_sparse_profile(case=case, size=size)
                else max(int(size or 48), 32)
            )
            return {
                "spatial_resolution": spatial_resolution,
                "spectral_resolution": 12,
            }
    if model_kind == "sky_lambda_feature_field_hetero":
        if case in {
            "tpv_field_fit_sparse_large_mosaic_exact",
            "tpv_field_fit_sparse_large_mosaic_exact_densebands",
            "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
        }:
            return {
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 96,
            }
    if model_kind == "sky_lambda_feature_atlas_field":
        if case == "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit":
            return {
                "atlas_side": 2,
                "atlas_sigma": 0.2,
                "topk_experts": 2,
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 24,
                "scale_hidden_dim": 192,
                "scale_num_blocks": 4,
                "decoder_hidden_dim": 192,
                "decoder_num_blocks": 4,
                "refine_hidden_dim": 96,
                "bandpass_hidden_dim": 128,
            }
    if model_kind == "sky_lambda_atlas_field":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "atlas_side": 3,
                "atlas_sigma": 0.16,
                "topk_experts": 2,
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "atlas_side": 3,
                "atlas_sigma": 0.16,
                "topk_experts": 2,
                "base_resolution": 32,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tpv_field_fit_sparse_large_mosaic_exact":
            return {
                "atlas_side": 2,
                "atlas_sigma": 0.2,
                "base_resolution": 48,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
        if case == "tan_sip_field_fit_sparse_large_mosaic_broad":
            return {
                "base_resolution": 24,
                "num_scales": 4,
                "features_per_plane": 16,
                "hidden_dim": 128,
                "hidden_layers": 3,
                "refine_hidden_dim": 64,
            }
    if model_kind == "sky_encoder_resmlp":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 224,
                "num_blocks": 5,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 8,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 224,
                "num_blocks": 5,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 8,
                "multiscale_readout": True,
                "readout_hidden_dim": 192,
            }
        if case == "tpv_field_fit_deep_multiband_exact":
            return {
                "hidden_dim": 160,
                "num_blocks": 4,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 4,
            }
    if model_kind == "hashgrid_resmlp":
        if case == "tan_sip_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 128,
                "num_res_blocks": 4,
                "num_levels": 8,
                "features_per_level": 4,
                "finest_resolution": 256,
                "log2_hashmap_size": tuple(range(11, 19)),
                "hash_interpolation": "smoothstep",
                "hash_level_gain": True,
            }
        if case == "tpv_field_fit_deep_multiband_broad":
            return {
                "hidden_dim": 144,
                "num_res_blocks": 4,
                "num_levels": 8,
                "features_per_level": 4,
                "finest_resolution": 256,
                "log2_hashmap_size": tuple(range(11, 19)),
                "hash_interpolation": "smoothstep",
                "hash_level_gain": True,
            }
    return {}


def _make_field_model_for_case(
    model_kind: str,
    domain: FieldDomain,
    *,
    case: str | None = None,
    size: int | None = None,
):
    kwargs = _field_model_kwargs_for_case(model_kind, case=case, size=size)
    if model_kind == "explicit_spectral_grid_field":
        if domain.coord_mode == "local_tangent":
            local_domain = domain
        else:
            local_domain = FieldDomain(
                center_ra_deg=domain.center_ra_deg,
                center_dec_deg=domain.center_dec_deg,
                spatial_scale_deg=domain.spatial_scale_deg,
                wavelength_center=domain.wavelength_center,
                wavelength_scale=domain.wavelength_scale,
                coord_mode="local_tangent",
            )
        return ExplicitSpectralGridField(local_domain, **kwargs)
    if model_kind == "spectral_basis_field":
        return SpectralBasisField(domain, **kwargs)
    if model_kind == "learned_spectral_basis_field":
        return LearnedSpectralBasisField(domain, **kwargs)
    if model_kind == "sky_encoder_bandpass_field":
        return SkyEncoderBandpassField(domain, **kwargs)
    if model_kind == "sky_encoder_mlp":
        return SkyEncoderMLPField(domain, **kwargs)
    if model_kind == "sky_encoder_resmlp":
        return SkyEncoderResMLPField(domain, **kwargs)
    if model_kind == "sky_lambda_kplanes_field":
        return make_field_model(model_kind, domain, **kwargs)
    if model_kind == "sky_lambda_feature_field":
        return make_field_model(model_kind, domain, **kwargs)
    if model_kind == "operator_constrained_feature_field":
        return make_field_model(model_kind, domain, **kwargs)
    if model_kind == "sky_lambda_feature_field_hetero":
        return make_field_model(model_kind, domain, **kwargs)
    if model_kind == "sky_lambda_feature_atlas_field":
        return make_field_model(model_kind, domain, **kwargs)
    return make_field_model(model_kind, domain)


def _rotated_cd(
    scale_deg: float, rotation_deg: float
) -> tuple[float, float, float, float]:
    theta = math.radians(float(rotation_deg))
    cos_t = math.cos(theta)
    sin_t = math.sin(theta)
    return (
        -scale_deg * cos_t,
        scale_deg * sin_t,
        scale_deg * sin_t,
        scale_deg * cos_t,
    )


def _tan_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": size / 2.0,
        "CRPIX2": size / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": cd11,
        "CD1_2": cd12,
        "CD2_1": cd21,
        "CD2_2": cd22,
    }


def _tpv_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
    pv_scale: float = 1.0,
) -> dict[str, Any]:
    header = _tan_header(
        size,
        crval1=crval1,
        crval2=crval2,
        scale_deg=scale_deg,
        rotation_deg=rotation_deg,
    )
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_4"] = 2.0e-4 * pv_scale
    header["PV1_5"] = -3.0e-4 * pv_scale
    header["PV1_6"] = 1.5e-4 * pv_scale
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.0
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4 * pv_scale
    header["PV2_5"] = 2.5e-4 * pv_scale
    header["PV2_6"] = -2.0e-4 * pv_scale
    return header


def _sip_header(
    size: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
    distortion_scale: float = 1.0,
) -> dict[str, Any]:
    header = _tan_header(
        size,
        crval1=crval1,
        crval2=crval2,
        scale_deg=scale_deg,
        rotation_deg=rotation_deg,
    )
    header["CTYPE1"] = "RA---TAN-SIP"
    header["CTYPE2"] = "DEC--TAN-SIP"
    header["A_ORDER"] = 3
    header["B_ORDER"] = 3
    header["AP_ORDER"] = 3
    header["BP_ORDER"] = 3
    header["A_0_2"] = 1.2e-5 * distortion_scale
    header["A_1_1"] = -1.5e-5 * distortion_scale
    header["A_2_0"] = 1.0e-5 * distortion_scale
    header["B_0_2"] = -1.0e-5 * distortion_scale
    header["B_1_1"] = 1.4e-5 * distortion_scale
    header["B_2_0"] = -1.1e-5 * distortion_scale
    header["AP_0_2"] = -1.1e-5 * distortion_scale
    header["AP_1_1"] = 1.3e-5 * distortion_scale
    header["AP_2_0"] = -0.9e-5 * distortion_scale
    header["BP_0_2"] = 0.9e-5 * distortion_scale
    header["BP_1_1"] = -1.2e-5 * distortion_scale
    header["BP_2_0"] = 1.0e-5 * distortion_scale
    return header


def _build_case_specs(size: int) -> dict[str, CaseSpec]:
    spectral_axis = SpectralAxis(
        values=torch.tensor([4800.0, 5600.0, 6400.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    spectral_bin_edges = torch.tensor(
        [4500.0, 5200.0, 6000.0, 6800.0],
        dtype=torch.float64,
    )
    tan_domain = FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.04,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    tpv_domain = FieldDomain(
        center_ra_deg=210.0,
        center_dec_deg=-6.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
    )
    multicolor_spectral_axis = SpectralAxis(
        values=torch.tensor([4450.0, 5200.0, 6100.0, 7050.0], dtype=torch.float64),
        unit="Angstrom",
        type="WAVE",
    )
    multicolor_spectral_bin_edges = torch.tensor(
        [4200.0, 4850.0, 5550.0, 6550.0, 7450.0],
        dtype=torch.float64,
    )
    multicolor_domain = FieldDomain(
        center_ra_deg=34.0,
        center_dec_deg=-12.0,
        spatial_scale_deg=0.07,
        wavelength_center=5750.0,
        wavelength_scale=1700.0,
    )
    deep_exact_axis = SpectralAxis(
        values=torch.tensor(
            [6060.0, 8140.0, 11500.0, 15000.0, 20000.0],
            dtype=torch.float64,
        ),
        unit="Angstrom",
        type="WAVE",
    )
    deep_exact_dense_axis = SpectralAxis(
        values=torch.tensor(
            [
                5200.0,
                6060.0,
                7200.0,
                8140.0,
                9800.0,
                11500.0,
                14000.0,
                17000.0,
                20000.0,
            ],
            dtype=torch.float64,
        ),
        unit="Angstrom",
        type="WAVE",
    )
    deep_exact_domain = FieldDomain(
        center_ra_deg=210.0,
        center_dec_deg=-6.0,
        spatial_scale_deg=0.06,
        wavelength_center=14000.0,
        wavelength_scale=9000.0,
    )
    deep_broad_axis = SpectralAxis(
        values=torch.tensor(
            [4700.0, 6200.0, 8200.0, 11000.0, 15000.0],
            dtype=torch.float64,
        ),
        unit="Angstrom",
        type="WAVE",
    )
    deep_broad_dense_axis = SpectralAxis(
        values=torch.tensor(
            [
                4700.0,
                5600.0,
                6500.0,
                7600.0,
                9000.0,
                11000.0,
                13000.0,
                15500.0,
                18000.0,
            ],
            dtype=torch.float64,
        ),
        unit="Angstrom",
        type="WAVE",
    )
    deep_broad_edges = torch.tensor(
        [4200.0, 5400.0, 7200.0, 9500.0, 12500.0, 18000.0],
        dtype=torch.float64,
    )
    deep_broad_dense_edges = torch.tensor(
        [
            4200.0,
            5150.0,
            6050.0,
            7050.0,
            8300.0,
            10000.0,
            12000.0,
            14250.0,
            16750.0,
            19500.0,
        ],
        dtype=torch.float64,
    )
    deep_broad_tan_sip_domain = FieldDomain(
        center_ra_deg=34.0,
        center_dec_deg=-12.0,
        spatial_scale_deg=0.08,
        wavelength_center=10500.0,
        wavelength_scale=8000.0,
    )

    def _sparse_mosaic_offsets() -> tuple[tuple[float, float], ...]:
        return (
            (-0.82, -0.76),
            (-0.46, -0.58),
            (-0.12, -0.72),
            (0.28, -0.64),
            (0.67, -0.51),
            (-0.73, -0.18),
            (-0.34, -0.12),
            (0.02, -0.06),
            (0.44, -0.21),
            (0.78, -0.08),
            (-0.69, 0.24),
            (-0.25, 0.18),
            (0.16, 0.10),
            (0.55, 0.27),
            (0.86, 0.18),
            (-0.78, 0.67),
            (-0.41, 0.56),
            (-0.04, 0.74),
            (0.36, 0.63),
            (0.74, 0.72),
        )

    def _repeated_sparse_mosaic_offsets(
        count: int,
        *,
        jitter: float = 0.06,
    ) -> tuple[tuple[float, float], ...]:
        base = _sparse_mosaic_offsets()
        repeated: list[tuple[float, float]] = []
        for idx in range(count):
            x, y = base[idx % len(base)]
            cycle = idx // len(base)
            if cycle == 0:
                repeated.append((x, y))
                continue
            dx = jitter * ((idx * 3 + cycle) % 5 - 2)
            dy = jitter * ((idx * 5 + 2 * cycle) % 5 - 2)
            repeated.append(
                (
                    max(min(x + dx, 0.95), -0.95),
                    max(min(y + dy, 0.95), -0.95),
                )
            )
        return tuple(repeated)

    def _make_sparse_large_tan_sip_broad_case() -> CaseSpec:
        target_scale_deg = 0.72 / 3600.0
        obs_size = max(24, min(64, size // 4))
        target_header = _sip_header(
            size,
            crval1=34.0,
            crval2=-12.0,
            scale_deg=target_scale_deg,
            distortion_scale=1.0,
        )
        half_span_deg = 0.5 * max(size - obs_size, 1) * target_scale_deg
        cos_dec = math.cos(math.radians(-12.0))
        broad_edges = deep_broad_edges.detach().cpu().tolist()
        scale_arcsec = [0.62, 0.74, 0.88, 1.05]
        rotations = [-12.0, -5.0, 4.0, 11.0]
        distortion_scales = [0.86, 0.97, 1.08, 1.22]
        noise_sigmas = [0.008, 0.011, 0.014, 0.018]
        pointings = _sparse_mosaic_offsets()
        observations: list[ObservationSpec] = []
        for idx, (fx, fy) in enumerate(pointings):
            band_idx = idx % 5
            visit_idx = idx // 5
            dra_deg = float(fx) * half_span_deg
            ddec_deg = float(fy) * half_span_deg
            crval1 = 34.0 + dra_deg / max(cos_dec, 1.0e-6)
            crval2 = -12.0 + ddec_deg
            gap = None
            if idx % 4 == 1:
                gap = (
                    slice(obs_size // 5, obs_size // 5 + max(3, obs_size // 10)),
                    slice(obs_size // 2, obs_size - obs_size // 6),
                )
            observations.append(
                ObservationSpec(
                    header=_sip_header(
                        obs_size,
                        crval1=crval1,
                        crval2=crval2,
                        scale_deg=scale_arcsec[visit_idx] / 3600.0,
                        rotation_deg=rotations[(idx + band_idx) % len(rotations)],
                        distortion_scale=distortion_scales[
                            (idx + visit_idx) % len(distortion_scales)
                        ],
                    ),
                    spectral_min=float(broad_edges[band_idx]),
                    spectral_max=float(broad_edges[band_idx + 1]),
                    transmission_center=float(deep_broad_axis.values[band_idx].item()),
                    transmission_sigma=[190.0, 280.0, 360.0, 520.0, 900.0][band_idx],
                    psf_sigma_pix=0.70,
                    noise_sigma=noise_sigmas[
                        (band_idx + visit_idx) % len(noise_sigmas)
                    ],
                    gap=gap,
                )
            )
        mosaic_domain = FieldDomain(
            center_ra_deg=34.0,
            center_dec_deg=-12.0,
            spatial_scale_deg=max(0.08, 0.5 * size * target_scale_deg),
            wavelength_center=10500.0,
            wavelength_scale=8000.0,
            coord_mode="sphere_xyz",
        )
        return CaseSpec(
            case="tan_sip_field_fit_sparse_large_mosaic_broad",
            projection="TAN-SIP",
            description="Sparse large TAN-SIP broad-band mosaic with many small visits over one larger field, mixed depth/scale/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=mosaic_domain,
            target_header=target_header,
            spectral_axis=deep_broad_axis,
            spectral_bin_edges=deep_broad_edges,
            observations=tuple(observations),
        )

    def _make_sparse_large_tpv_exact_case() -> CaseSpec:
        target_scale_deg = 0.75 / 3600.0
        obs_size = max(24, min(64, size // 4))
        target_header = _tpv_header(
            size,
            crval1=210.0,
            crval2=-6.0,
            scale_deg=target_scale_deg,
            rotation_deg=2.0,
            pv_scale=1.08,
        )
        half_span_deg = 0.5 * max(size - obs_size, 1) * target_scale_deg
        cos_dec = math.cos(math.radians(-6.0))
        exact_values = deep_exact_axis.values.detach().cpu().tolist()
        scale_arcsec = [0.66, 0.79, 0.94, 1.18]
        rotations = [-10.0, -3.0, 6.0, 14.0]
        pv_scales = [0.90, 1.02, 1.18, 1.32]
        noise_sigmas = [0.007, 0.010, 0.014, 0.018]
        pointings = _sparse_mosaic_offsets()
        observations: list[ObservationSpec] = []
        for idx, (fx, fy) in enumerate(pointings):
            band_idx = idx % len(exact_values)
            visit_idx = idx // len(exact_values)
            dra_deg = float(fx) * half_span_deg
            ddec_deg = float(fy) * half_span_deg
            crval1 = 210.0 + dra_deg / max(cos_dec, 1.0e-6)
            crval2 = -6.0 + ddec_deg
            gap = None
            if idx % 5 == 2:
                gap = (
                    slice(obs_size // 3, obs_size - obs_size // 5),
                    slice(obs_size // 4, obs_size // 4 + max(3, obs_size // 10)),
                )
            observations.append(
                ObservationSpec(
                    header=_tpv_header(
                        obs_size,
                        crval1=crval1,
                        crval2=crval2,
                        scale_deg=scale_arcsec[visit_idx] / 3600.0,
                        rotation_deg=rotations[(idx + band_idx) % len(rotations)],
                        pv_scale=pv_scales[(idx + visit_idx) % len(pv_scales)],
                    ),
                    spectral_min=float(exact_values[band_idx]),
                    spectral_max=float(exact_values[band_idx]),
                    transmission_center=float(exact_values[band_idx]),
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=noise_sigmas[
                        (band_idx + 2 * visit_idx) % len(noise_sigmas)
                    ],
                    gap=gap,
                )
            )
        mosaic_domain = FieldDomain(
            center_ra_deg=210.0,
            center_dec_deg=-6.0,
            spatial_scale_deg=max(0.06, 0.5 * size * target_scale_deg),
            wavelength_center=14000.0,
            wavelength_scale=9000.0,
            coord_mode="sphere_xyz",
        )
        return CaseSpec(
            case="tpv_field_fit_sparse_large_mosaic_exact",
            projection="TPV",
            description="Sparse large TPV exact-band mosaic with many small visits over one larger field, mixed depth/scale/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=mosaic_domain,
            target_header=target_header,
            spectral_axis=deep_exact_axis,
            spectral_bin_edges=None,
            observations=tuple(observations),
        )

    def _make_sparse_large_tpv_exact_densebands_case() -> CaseSpec:
        target_scale_deg = 0.75 / 3600.0
        obs_size = max(24, min(64, size // 4))
        target_header = _tpv_header(
            size,
            crval1=210.0,
            crval2=-6.0,
            scale_deg=target_scale_deg,
            rotation_deg=2.0,
            pv_scale=1.08,
        )
        half_span_deg = 0.5 * max(size - obs_size, 1) * target_scale_deg
        cos_dec = math.cos(math.radians(-6.0))
        exact_values = deep_exact_dense_axis.values.detach().cpu().tolist()
        scale_arcsec = [0.70, 0.96]
        rotations = [-9.0, -2.0, 5.0, 11.0]
        pv_scales = [0.92, 1.04, 1.16, 1.28]
        noise_sigmas = [0.0065, 0.0085, 0.011, 0.014, 0.017]
        pointings = _sparse_mosaic_offsets()[: 2 * len(exact_values)]
        observations: list[ObservationSpec] = []
        for idx, (fx, fy) in enumerate(pointings):
            band_idx = idx % len(exact_values)
            visit_idx = idx // len(exact_values)
            dra_deg = float(fx) * half_span_deg
            ddec_deg = float(fy) * half_span_deg
            crval1 = 210.0 + dra_deg / max(cos_dec, 1.0e-6)
            crval2 = -6.0 + ddec_deg
            gap = None
            if idx % 6 == 3:
                gap = (
                    slice(obs_size // 3, obs_size - obs_size // 6),
                    slice(obs_size // 5, obs_size // 5 + max(3, obs_size // 12)),
                )
            observations.append(
                ObservationSpec(
                    header=_tpv_header(
                        obs_size,
                        crval1=crval1,
                        crval2=crval2,
                        scale_deg=scale_arcsec[visit_idx] / 3600.0,
                        rotation_deg=rotations[(idx + band_idx) % len(rotations)],
                        pv_scale=pv_scales[(idx + visit_idx) % len(pv_scales)],
                    ),
                    spectral_min=float(exact_values[band_idx]),
                    spectral_max=float(exact_values[band_idx]),
                    transmission_center=float(exact_values[band_idx]),
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=noise_sigmas[
                        (band_idx + 2 * visit_idx) % len(noise_sigmas)
                    ],
                    gap=gap,
                )
            )
        mosaic_domain = FieldDomain(
            center_ra_deg=210.0,
            center_dec_deg=-6.0,
            spatial_scale_deg=max(0.06, 0.5 * size * target_scale_deg),
            wavelength_center=12500.0,
            wavelength_scale=9500.0,
            coord_mode="sphere_xyz",
        )
        return CaseSpec(
            case="tpv_field_fit_sparse_large_mosaic_exact_densebands",
            projection="TPV",
            description="Sparse large TPV exact-band mosaic with denser spectral sampling across many small visits over one larger field, mixed depth/scale/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=mosaic_domain,
            target_header=target_header,
            spectral_axis=deep_exact_dense_axis,
            spectral_bin_edges=None,
            observations=tuple(observations),
        )

    def _make_sparse_large_tpv_exact_3band_multivisit_case() -> CaseSpec:
        target_scale_deg = 0.75 / 3600.0
        obs_size = max(24, min(64, size // 4))
        target_header = _tpv_header(
            size,
            crval1=210.0,
            crval2=-6.0,
            scale_deg=target_scale_deg,
            rotation_deg=2.0,
            pv_scale=1.08,
        )
        half_span_deg = 0.5 * max(size - obs_size, 1) * target_scale_deg
        cos_dec = math.cos(math.radians(-6.0))
        exact_values = [6200.0, 11500.0, 20000.0]
        spectral_axis = SpectralAxis(
            values=torch.tensor(exact_values, dtype=torch.float64),
            unit="Angstrom",
            type="WAVE",
        )
        scale_arcsec = [0.60, 0.72, 0.88, 1.04, 1.26]
        rotations = [-12.0, -5.0, 2.0, 8.0, 15.0]
        pv_scales = [0.88, 0.98, 1.08, 1.18, 1.30]
        noise_sigmas = [0.0065, 0.0085, 0.0115, 0.0150, 0.0190]
        pointings = _repeated_sparse_mosaic_offsets(15 * len(exact_values))
        observations: list[ObservationSpec] = []
        for idx, (fx, fy) in enumerate(pointings):
            band_idx = idx % len(exact_values)
            visit_idx = idx // len(exact_values)
            dra_deg = float(fx) * half_span_deg
            ddec_deg = float(fy) * half_span_deg
            crval1 = 210.0 + dra_deg / max(cos_dec, 1.0e-6)
            crval2 = -6.0 + ddec_deg
            gap = None
            if idx % 4 == 2:
                gap = (
                    slice(obs_size // 3, obs_size - obs_size // 5),
                    slice(obs_size // 4, obs_size // 4 + max(3, obs_size // 10)),
                )
            observations.append(
                ObservationSpec(
                    header=_tpv_header(
                        obs_size,
                        crval1=crval1,
                        crval2=crval2,
                        scale_deg=scale_arcsec[visit_idx % len(scale_arcsec)] / 3600.0,
                        rotation_deg=rotations[(idx + band_idx) % len(rotations)],
                        pv_scale=pv_scales[(idx + visit_idx) % len(pv_scales)],
                    ),
                    spectral_min=float(exact_values[band_idx]),
                    spectral_max=float(exact_values[band_idx]),
                    transmission_center=float(exact_values[band_idx]),
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=noise_sigmas[
                        (band_idx + 2 * visit_idx) % len(noise_sigmas)
                    ],
                    gap=gap,
                )
            )
        mosaic_domain = FieldDomain(
            center_ra_deg=210.0,
            center_dec_deg=-6.0,
            spatial_scale_deg=max(0.06, 0.5 * size * target_scale_deg),
            wavelength_center=12000.0,
            wavelength_scale=9000.0,
            coord_mode="sphere_xyz",
        )
        return CaseSpec(
            case="tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit",
            projection="TPV",
            description="Sparse large TPV exact-band mosaic with 3 nearly disjoint bands and 15 heterogeneous visits per band over one larger field",
            truth_model="analytic_basis",
            domain=mosaic_domain,
            target_header=target_header,
            spectral_axis=spectral_axis,
            spectral_bin_edges=None,
            observations=tuple(observations),
        )

    def _make_sparse_large_tpv_broad_denseoverlap_case() -> CaseSpec:
        target_scale_deg = 0.75 / 3600.0
        obs_size = max(24, min(64, size // 4))
        target_header = _tpv_header(
            size,
            crval1=210.0,
            crval2=-6.0,
            scale_deg=target_scale_deg,
            rotation_deg=2.0,
            pv_scale=1.08,
        )
        half_span_deg = 0.5 * max(size - obs_size, 1) * target_scale_deg
        cos_dec = math.cos(math.radians(-6.0))
        centers = deep_broad_dense_axis.values.detach().cpu().tolist()
        sigmas = [220.0, 240.0, 260.0, 300.0, 360.0, 460.0, 560.0, 720.0, 900.0]
        scale_arcsec = [0.70, 0.96]
        rotations = [-9.0, -2.0, 5.0, 11.0]
        pv_scales = [0.92, 1.04, 1.16, 1.28]
        noise_sigmas = [0.0065, 0.0085, 0.011, 0.014, 0.017]
        pointings = _sparse_mosaic_offsets()[: 2 * len(centers)]
        observations: list[ObservationSpec] = []
        for idx, (fx, fy) in enumerate(pointings):
            band_idx = idx % len(centers)
            visit_idx = idx // len(centers)
            dra_deg = float(fx) * half_span_deg
            ddec_deg = float(fy) * half_span_deg
            crval1 = 210.0 + dra_deg / max(cos_dec, 1.0e-6)
            crval2 = -6.0 + ddec_deg
            center = float(centers[band_idx])
            sigma = float(sigmas[band_idx])
            gap = None
            if idx % 6 == 4:
                gap = (
                    slice(obs_size // 3, obs_size - obs_size // 6),
                    slice(obs_size // 5, obs_size // 5 + max(3, obs_size // 12)),
                )
            observations.append(
                ObservationSpec(
                    header=_tpv_header(
                        obs_size,
                        crval1=crval1,
                        crval2=crval2,
                        scale_deg=scale_arcsec[visit_idx] / 3600.0,
                        rotation_deg=rotations[(idx + band_idx) % len(rotations)],
                        pv_scale=pv_scales[(idx + visit_idx) % len(pv_scales)],
                    ),
                    spectral_min=center - 3.0 * sigma,
                    spectral_max=center + 3.0 * sigma,
                    transmission_center=center,
                    transmission_sigma=sigma,
                    psf_sigma_pix=0.70,
                    noise_sigma=noise_sigmas[
                        (band_idx + 2 * visit_idx) % len(noise_sigmas)
                    ],
                    gap=gap,
                )
            )
        mosaic_domain = FieldDomain(
            center_ra_deg=210.0,
            center_dec_deg=-6.0,
            spatial_scale_deg=max(0.06, 0.5 * size * target_scale_deg),
            wavelength_center=12500.0,
            wavelength_scale=9500.0,
            coord_mode="sphere_xyz",
        )
        return CaseSpec(
            case="tpv_field_fit_sparse_large_mosaic_broad_denseoverlap",
            projection="TPV",
            description="Sparse large TPV broad-band mosaic with denser overlapping spectral coverage across many small visits over one larger field, mixed depth/scale/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=mosaic_domain,
            target_header=target_header,
            spectral_axis=deep_broad_dense_axis,
            spectral_bin_edges=deep_broad_dense_edges,
            observations=tuple(observations),
        )

    return {
        "tan_sip_field_fit_multicolor": CaseSpec(
            case="tan_sip_field_fit_multicolor",
            projection="TAN-SIP",
            description="Multicolor TAN-SIP field with varied depth, throughput, resolution, and four-band spectral target",
            truth_model="analytic_basis",
            domain=multicolor_domain,
            target_header=_sip_header(
                size,
                crval1=34.0,
                crval2=-12.0,
                scale_deg=1.1 / 3600.0,
            ),
            spectral_axis=multicolor_spectral_axis,
            spectral_bin_edges=multicolor_spectral_bin_edges,
            observations=(
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0,
                        crval2=-12.0,
                        scale_deg=0.75 / 3600.0,
                        distortion_scale=0.8,
                    ),
                    spectral_min=4225.0,
                    spectral_max=4700.0,
                    transmission_center=4450.0,
                    transmission_sigma=95.0,
                    psf_sigma_pix=0.42,
                    noise_sigma=0.008,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0028,
                        crval2=-12.0018,
                        scale_deg=1.35 / 3600.0,
                        rotation_deg=9.0,
                        distortion_scale=1.2,
                    ),
                    spectral_min=4920.0,
                    spectral_max=5480.0,
                    transmission_center=5200.0,
                    transmission_sigma=110.0,
                    psf_sigma_pix=0.95,
                    noise_sigma=0.017,
                    gap=(slice(size // 5, size // 2), slice(size // 2, size // 2 + 7)),
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=33.9972,
                        crval2=-12.0035,
                        scale_deg=1.55 / 3600.0,
                        rotation_deg=-13.0,
                        distortion_scale=1.05,
                    ),
                    spectral_min=5750.0,
                    spectral_max=6420.0,
                    transmission_center=6090.0,
                    transmission_sigma=155.0,
                    psf_sigma_pix=1.25,
                    noise_sigma=0.026,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0015,
                        crval2=-11.9978,
                        scale_deg=0.92 / 3600.0,
                        rotation_deg=4.0,
                        distortion_scale=0.95,
                    ),
                    spectral_min=6620.0,
                    spectral_max=7360.0,
                    transmission_center=7040.0,
                    transmission_sigma=170.0,
                    psf_sigma_pix=0.62,
                    noise_sigma=0.013,
                ),
            ),
        ),
        "tan_sip_field_fit_deep_multiband_broad": CaseSpec(
            case="tan_sip_field_fit_deep_multiband_broad",
            projection="TAN-SIP",
            description="Same-field TAN-SIP deep broad-band benchmark with repeated visits, known transmissions, mixed scale/depth/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=deep_broad_tan_sip_domain,
            target_header=_sip_header(
                size,
                crval1=34.0,
                crval2=-12.0,
                scale_deg=0.72 / 3600.0,
                distortion_scale=1.0,
            ),
            spectral_axis=deep_broad_axis,
            spectral_bin_edges=deep_broad_edges,
            observations=(
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0000,
                        crval2=-12.0000,
                        scale_deg=0.68 / 3600.0,
                        rotation_deg=-3.0,
                        distortion_scale=0.85,
                    ),
                    spectral_min=4200.0,
                    spectral_max=5400.0,
                    transmission_center=4700.0,
                    transmission_sigma=190.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.010,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0010,
                        crval2=-11.9991,
                        scale_deg=1.05 / 3600.0,
                        rotation_deg=10.0,
                        distortion_scale=1.18,
                    ),
                    spectral_min=4200.0,
                    spectral_max=5400.0,
                    transmission_center=4700.0,
                    transmission_sigma=210.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.018,
                    gap=(
                        slice(size // 4, size // 2),
                        slice(size // 2, size - size // 6),
                    ),
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=33.9994,
                        crval2=-12.0005,
                        scale_deg=0.80 / 3600.0,
                        rotation_deg=4.0,
                        distortion_scale=0.92,
                    ),
                    spectral_min=5400.0,
                    spectral_max=7200.0,
                    transmission_center=6200.0,
                    transmission_sigma=280.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.011,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0008,
                        crval2=-12.0010,
                        scale_deg=1.22 / 3600.0,
                        rotation_deg=-9.0,
                        distortion_scale=1.25,
                    ),
                    spectral_min=5400.0,
                    spectral_max=7200.0,
                    transmission_center=6200.0,
                    transmission_sigma=320.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.020,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0003,
                        crval2=-12.0008,
                        scale_deg=0.62 / 3600.0,
                        rotation_deg=7.0,
                        distortion_scale=1.05,
                    ),
                    spectral_min=7200.0,
                    spectral_max=9500.0,
                    transmission_center=8200.0,
                    transmission_sigma=360.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.009,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=33.9989,
                        crval2=-11.9994,
                        scale_deg=0.98 / 3600.0,
                        rotation_deg=-12.0,
                        distortion_scale=1.10,
                    ),
                    spectral_min=7200.0,
                    spectral_max=9500.0,
                    transmission_center=8200.0,
                    transmission_sigma=390.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.016,
                    gap=(
                        slice(size // 5, size // 5 + 5),
                        slice(size // 3, size - size // 4),
                    ),
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0011,
                        crval2=-12.0002,
                        scale_deg=0.74 / 3600.0,
                        rotation_deg=5.0,
                        distortion_scale=0.98,
                    ),
                    spectral_min=9500.0,
                    spectral_max=12500.0,
                    transmission_center=11000.0,
                    transmission_sigma=520.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.010,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=33.9992,
                        crval2=-12.0012,
                        scale_deg=1.28 / 3600.0,
                        rotation_deg=14.0,
                        distortion_scale=1.22,
                    ),
                    spectral_min=9500.0,
                    spectral_max=12500.0,
                    transmission_center=11000.0,
                    transmission_sigma=580.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.019,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=34.0005,
                        crval2=-11.9998,
                        scale_deg=0.70 / 3600.0,
                        rotation_deg=-4.0,
                        distortion_scale=0.90,
                    ),
                    spectral_min=12500.0,
                    spectral_max=18000.0,
                    transmission_center=15000.0,
                    transmission_sigma=900.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.012,
                ),
                ObservationSpec(
                    header=_sip_header(
                        size,
                        crval1=33.9990,
                        crval2=-12.0009,
                        scale_deg=1.34 / 3600.0,
                        rotation_deg=-15.0,
                        distortion_scale=1.28,
                    ),
                    spectral_min=12500.0,
                    spectral_max=18000.0,
                    transmission_center=15000.0,
                    transmission_sigma=980.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.021,
                    gap=(
                        slice(size // 3, size - size // 5),
                        slice(size // 4, size // 4 + 6),
                    ),
                ),
            ),
        ),
        "tan_sip_field_fit_sparse_large_mosaic_broad": _make_sparse_large_tan_sip_broad_case(),
        "tan_field_fit_exact_transmission": CaseSpec(
            case="tan_field_fit_exact_transmission",
            projection="TAN",
            description="Exact-transmission TAN field with same-field repeated delta-band coverage, shared PSF, and mixed scale/depth",
            truth_model="analytic_basis",
            domain=tan_domain,
            target_header=_tan_header(
                size,
                crval1=150.0,
                crval2=2.0,
                scale_deg=0.8 / 3600.0,
            ),
            spectral_axis=SpectralAxis(
                values=torch.tensor([4800.0, 5600.0, 6400.0], dtype=torch.float64),
                unit="Angstrom",
                type="WAVE",
            ),
            spectral_bin_edges=None,
            observations=(
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0,
                        crval2=2.0,
                        scale_deg=0.85 / 3600.0,
                    ),
                    spectral_min=4800.0,
                    spectral_max=4800.0,
                    transmission_center=4800.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.018,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0009,
                        crval2=2.0007,
                        scale_deg=1.12 / 3600.0,
                        rotation_deg=8.0,
                    ),
                    spectral_min=4800.0,
                    spectral_max=4800.0,
                    transmission_center=4800.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.028,
                    gap=(slice(size // 4, size // 2), slice(size // 2, size // 2 + 4)),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.9993,
                        crval2=1.9995,
                        scale_deg=0.96 / 3600.0,
                        rotation_deg=-6.0,
                    ),
                    spectral_min=5600.0,
                    spectral_max=5600.0,
                    transmission_center=5600.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.014,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0012,
                        crval2=2.0008,
                        scale_deg=1.10 / 3600.0,
                        rotation_deg=10.0,
                    ),
                    spectral_min=5600.0,
                    spectral_max=5600.0,
                    transmission_center=5600.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.022,
                    gap=(
                        slice(size // 5, size // 5 + 5),
                        slice(size // 3, size - size // 4),
                    ),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.9991,
                        crval2=2.0009,
                        scale_deg=0.90 / 3600.0,
                        rotation_deg=-9.0,
                    ),
                    spectral_min=6400.0,
                    spectral_max=6400.0,
                    transmission_center=6400.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.016,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0008,
                        crval2=1.9994,
                        scale_deg=1.04 / 3600.0,
                        rotation_deg=5.0,
                    ),
                    spectral_min=6400.0,
                    spectral_max=6400.0,
                    transmission_center=6400.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.026,
                    gap=(
                        slice(size // 3, size - size // 5),
                        slice(size // 4, size // 4 + 4),
                    ),
                ),
            ),
        ),
        "tpv_field_fit_deep_multiband_broad": CaseSpec(
            case="tpv_field_fit_deep_multiband_broad",
            projection="TPV",
            description="Same-field TPV deep broad-band benchmark with repeated visits, known transmissions, mixed scale/depth/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=deep_exact_domain,
            target_header=_tpv_header(
                size,
                crval1=210.0,
                crval2=-6.0,
                scale_deg=0.72 / 3600.0,
                rotation_deg=2.0,
                pv_scale=1.08,
            ),
            spectral_axis=deep_broad_axis,
            spectral_bin_edges=deep_broad_edges,
            observations=(
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0000,
                        crval2=-6.0000,
                        scale_deg=0.69 / 3600.0,
                        rotation_deg=-2.0,
                        pv_scale=0.92,
                    ),
                    spectral_min=4200.0,
                    spectral_max=5400.0,
                    transmission_center=4700.0,
                    transmission_sigma=190.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.011,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9992,
                        crval2=-6.0008,
                        scale_deg=1.08 / 3600.0,
                        rotation_deg=9.0,
                        pv_scale=1.22,
                    ),
                    spectral_min=4200.0,
                    spectral_max=5400.0,
                    transmission_center=4700.0,
                    transmission_sigma=220.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.019,
                    gap=(
                        slice(size // 4, size // 2),
                        slice(size // 3, size - size // 6),
                    ),
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0008,
                        crval2=-5.9992,
                        scale_deg=0.78 / 3600.0,
                        rotation_deg=5.0,
                        pv_scale=0.98,
                    ),
                    spectral_min=5400.0,
                    spectral_max=7200.0,
                    transmission_center=6200.0,
                    transmission_sigma=280.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.012,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9987,
                        crval2=-6.0011,
                        scale_deg=1.24 / 3600.0,
                        rotation_deg=-11.0,
                        pv_scale=1.30,
                    ),
                    spectral_min=5400.0,
                    spectral_max=7200.0,
                    transmission_center=6200.0,
                    transmission_sigma=320.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.021,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0004,
                        crval2=-6.0007,
                        scale_deg=0.63 / 3600.0,
                        rotation_deg=6.0,
                        pv_scale=1.04,
                    ),
                    spectral_min=7200.0,
                    spectral_max=9500.0,
                    transmission_center=8200.0,
                    transmission_sigma=360.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.009,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9990,
                        crval2=-5.9996,
                        scale_deg=0.97 / 3600.0,
                        rotation_deg=-8.0,
                        pv_scale=1.14,
                    ),
                    spectral_min=7200.0,
                    spectral_max=9500.0,
                    transmission_center=8200.0,
                    transmission_sigma=390.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.016,
                    gap=(
                        slice(size // 5, size // 5 + 5),
                        slice(size // 2, size - size // 7),
                    ),
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0011,
                        crval2=-5.9998,
                        scale_deg=0.75 / 3600.0,
                        rotation_deg=4.0,
                        pv_scale=1.00,
                    ),
                    spectral_min=9500.0,
                    spectral_max=12500.0,
                    transmission_center=11000.0,
                    transmission_sigma=520.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.010,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9988,
                        crval2=-6.0013,
                        scale_deg=1.31 / 3600.0,
                        rotation_deg=13.0,
                        pv_scale=1.27,
                    ),
                    spectral_min=9500.0,
                    spectral_max=12500.0,
                    transmission_center=11000.0,
                    transmission_sigma=580.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.020,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0006,
                        crval2=-5.9991,
                        scale_deg=0.73 / 3600.0,
                        rotation_deg=-5.0,
                        pv_scale=0.95,
                    ),
                    spectral_min=12500.0,
                    spectral_max=18000.0,
                    transmission_center=15000.0,
                    transmission_sigma=900.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.012,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9985,
                        crval2=-6.0009,
                        scale_deg=1.36 / 3600.0,
                        rotation_deg=-14.0,
                        pv_scale=1.34,
                    ),
                    spectral_min=12500.0,
                    spectral_max=18000.0,
                    transmission_center=15000.0,
                    transmission_sigma=980.0,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.022,
                    gap=(
                        slice(size // 3, size - size // 5),
                        slice(size // 4, size // 4 + 6),
                    ),
                ),
            ),
        ),
        "tpv_field_fit_deep_multiband_exact": CaseSpec(
            case="tpv_field_fit_deep_multiband_exact",
            projection="TPV",
            description="Deep-field TPV exact-band benchmark with repeated multiband visits, mixed scale/depth/coverage, and shared PSF",
            truth_model="analytic_basis",
            domain=deep_exact_domain,
            target_header=_tpv_header(
                size,
                crval1=210.0,
                crval2=-6.0,
                scale_deg=0.75 / 3600.0,
                rotation_deg=3.0,
                pv_scale=1.1,
            ),
            spectral_axis=deep_exact_axis,
            spectral_bin_edges=None,
            observations=(
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0000,
                        crval2=-6.0000,
                        scale_deg=0.72 / 3600.0,
                        rotation_deg=-2.0,
                        pv_scale=0.9,
                    ),
                    spectral_min=6060.0,
                    spectral_max=6060.0,
                    transmission_center=6060.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.010,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9986,
                        crval2=-6.0011,
                        scale_deg=1.10 / 3600.0,
                        rotation_deg=11.0,
                        pv_scale=1.25,
                    ),
                    spectral_min=6060.0,
                    spectral_max=6060.0,
                    transmission_center=6060.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.018,
                    gap=(
                        slice(size // 4, size // 2),
                        slice(size // 3, size - size // 6),
                    ),
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0018,
                        crval2=-5.9988,
                        scale_deg=0.78 / 3600.0,
                        rotation_deg=4.0,
                        pv_scale=0.95,
                    ),
                    spectral_min=8140.0,
                    spectral_max=8140.0,
                    transmission_center=8140.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.011,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9967,
                        crval2=-6.0024,
                        scale_deg=1.35 / 3600.0,
                        rotation_deg=-13.0,
                        pv_scale=1.35,
                    ),
                    spectral_min=8140.0,
                    spectral_max=8140.0,
                    transmission_center=8140.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.022,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0009,
                        crval2=-6.0015,
                        scale_deg=0.64 / 3600.0,
                        rotation_deg=6.0,
                        pv_scale=1.05,
                    ),
                    spectral_min=11500.0,
                    spectral_max=11500.0,
                    transmission_center=11500.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.008,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9991,
                        crval2=-6.0004,
                        scale_deg=0.94 / 3600.0,
                        rotation_deg=-7.0,
                        pv_scale=1.15,
                    ),
                    spectral_min=11500.0,
                    spectral_max=11500.0,
                    transmission_center=11500.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.015,
                    gap=(
                        slice(size // 5, size // 5 + 6),
                        slice(size // 2, size - size // 8),
                    ),
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0025,
                        crval2=-5.9995,
                        scale_deg=0.83 / 3600.0,
                        rotation_deg=9.0,
                        pv_scale=1.0,
                    ),
                    spectral_min=15000.0,
                    spectral_max=15000.0,
                    transmission_center=15000.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.010,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9978,
                        crval2=-6.0030,
                        scale_deg=1.42 / 3600.0,
                        rotation_deg=-16.0,
                        pv_scale=1.40,
                    ),
                    spectral_min=15000.0,
                    spectral_max=15000.0,
                    transmission_center=15000.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.021,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0012,
                        crval2=-5.9982,
                        scale_deg=0.76 / 3600.0,
                        rotation_deg=5.0,
                        pv_scale=1.08,
                    ),
                    spectral_min=20000.0,
                    spectral_max=20000.0,
                    transmission_center=20000.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.012,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.9958,
                        crval2=-6.0018,
                        scale_deg=1.18 / 3600.0,
                        rotation_deg=-10.0,
                        pv_scale=1.30,
                    ),
                    spectral_min=20000.0,
                    spectral_max=20000.0,
                    transmission_center=20000.0,
                    transmission_sigma=None,
                    psf_sigma_pix=0.70,
                    noise_sigma=0.019,
                    gap=(
                        slice(size // 3, size - size // 5),
                        slice(size // 4, size // 4 + 7),
                    ),
                ),
            ),
        ),
        "tpv_field_fit_sparse_large_mosaic_exact": _make_sparse_large_tpv_exact_case(),
        "tpv_field_fit_sparse_large_mosaic_exact_densebands": _make_sparse_large_tpv_exact_densebands_case(),
        "tpv_field_fit_sparse_large_mosaic_exact_3band_multivisit": _make_sparse_large_tpv_exact_3band_multivisit_case(),
        "tpv_field_fit_sparse_large_mosaic_broad_denseoverlap": _make_sparse_large_tpv_broad_denseoverlap_case(),
        "tan_field_fit": CaseSpec(
            case="tan_field_fit",
            projection="TAN",
            description="Shifted rotated TAN exposures with Gaussian bandpasses and PSF blur",
            truth_model="sine_implicit",
            domain=tan_domain,
            target_header=_tan_header(
                size,
                crval1=150.0,
                crval2=2.0,
                scale_deg=0.9 / 3600.0,
            ),
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            observations=(
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0,
                        crval2=2.0,
                        scale_deg=0.9 / 3600.0,
                    ),
                    spectral_min=4600.0,
                    spectral_max=5050.0,
                    transmission_center=4820.0,
                    transmission_sigma=120.0,
                    psf_sigma_pix=0.55,
                    noise_sigma=0.015,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.012,
                        crval2=2.008,
                        scale_deg=1.1 / 3600.0,
                        rotation_deg=19.0,
                    ),
                    spectral_min=5300.0,
                    spectral_max=5850.0,
                    transmission_center=5580.0,
                    transmission_sigma=150.0,
                    psf_sigma_pix=0.75,
                    noise_sigma=0.02,
                    gap=(slice(size // 4, size // 2), slice(size // 3, size // 3 + 6)),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.992,
                        crval2=1.994,
                        scale_deg=0.85 / 3600.0,
                        rotation_deg=-11.0,
                    ),
                    spectral_min=6100.0,
                    spectral_max=6700.0,
                    transmission_center=6400.0,
                    transmission_sigma=180.0,
                    psf_sigma_pix=0.6,
                    noise_sigma=0.018,
                ),
            ),
        ),
        "tan_field_fit_analytic": CaseSpec(
            case="tan_field_fit_analytic",
            projection="TAN",
            description="Analytic latent field with stronger PSF variation and overlapping Gaussian filters",
            truth_model="analytic_basis",
            domain=tan_domain,
            target_header=_tan_header(
                size,
                crval1=150.0,
                crval2=2.0,
                scale_deg=0.9 / 3600.0,
            ),
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            observations=(
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0,
                        crval2=2.0,
                        scale_deg=0.95 / 3600.0,
                    ),
                    spectral_min=4580.0,
                    spectral_max=5060.0,
                    transmission_center=4810.0,
                    transmission_sigma=130.0,
                    psf_sigma_pix=0.45,
                    noise_sigma=0.012,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.016,
                        crval2=2.012,
                        scale_deg=1.15 / 3600.0,
                        rotation_deg=14.0,
                    ),
                    spectral_min=5260.0,
                    spectral_max=5880.0,
                    transmission_center=5580.0,
                    transmission_sigma=180.0,
                    psf_sigma_pix=1.05,
                    noise_sigma=0.018,
                    gap=(slice(size // 5, size // 2), slice(size // 2, size // 2 + 5)),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.986,
                        crval2=1.99,
                        scale_deg=0.82 / 3600.0,
                        rotation_deg=-17.0,
                    ),
                    spectral_min=6080.0,
                    spectral_max=6720.0,
                    transmission_center=6390.0,
                    transmission_sigma=185.0,
                    psf_sigma_pix=0.7,
                    noise_sigma=0.015,
                ),
            ),
        ),
        "tan_field_fit_psf_mismatch": CaseSpec(
            case="tan_field_fit_psf_mismatch",
            projection="TAN",
            description="Shifted TAN field fit with deliberate PSF metadata mismatch at fit time",
            truth_model="sine_implicit",
            domain=tan_domain,
            target_header=_tan_header(
                size,
                crval1=150.0,
                crval2=2.0,
                scale_deg=0.9 / 3600.0,
            ),
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            observations=(
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0,
                        crval2=2.0,
                        scale_deg=0.9 / 3600.0,
                    ),
                    spectral_min=4600.0,
                    spectral_max=5050.0,
                    transmission_center=4820.0,
                    transmission_sigma=120.0,
                    psf_sigma_pix=0.55,
                    fit_psf_sigma_pix=0.72,
                    noise_sigma=0.015,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.012,
                        crval2=2.008,
                        scale_deg=1.1 / 3600.0,
                        rotation_deg=19.0,
                    ),
                    spectral_min=5300.0,
                    spectral_max=5850.0,
                    transmission_center=5580.0,
                    transmission_sigma=150.0,
                    psf_sigma_pix=0.75,
                    fit_psf_sigma_pix=0.55,
                    noise_sigma=0.02,
                    gap=(slice(size // 4, size // 2), slice(size // 3, size // 3 + 6)),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.992,
                        crval2=1.994,
                        scale_deg=0.85 / 3600.0,
                        rotation_deg=-11.0,
                    ),
                    spectral_min=6100.0,
                    spectral_max=6700.0,
                    transmission_center=6400.0,
                    transmission_sigma=180.0,
                    psf_sigma_pix=0.6,
                    fit_psf_sigma_pix=0.82,
                    noise_sigma=0.018,
                ),
            ),
        ),
        "tan_field_fit_low_overlap_depth": CaseSpec(
            case="tan_field_fit_low_overlap_depth",
            projection="TAN",
            description="Low-overlap TAN stack++ case with mixed depth, strong resolution ratios, and red-sensitive depth variation",
            truth_model="analytic_basis",
            domain=tan_domain,
            target_header=_tan_header(
                size,
                crval1=150.0,
                crval2=2.0,
                scale_deg=1.0 / 3600.0,
            ),
            spectral_axis=SpectralAxis(
                values=torch.tensor(
                    [4700.0, 5550.0, 6400.0, 7150.0], dtype=torch.float64
                ),
                unit="Angstrom",
                type="WAVE",
            ),
            spectral_bin_edges=torch.tensor(
                [4400.0, 5100.0, 6000.0, 6750.0, 7450.0],
                dtype=torch.float64,
            ),
            observations=(
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.997,
                        crval2=1.998,
                        scale_deg=0.70 / 3600.0,
                        rotation_deg=-6.0,
                    ),
                    spectral_min=4480.0,
                    spectral_max=4920.0,
                    transmission_center=4700.0,
                    transmission_sigma=90.0,
                    psf_sigma_pix=0.38,
                    noise_sigma=0.007,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0028,
                        crval2=2.0032,
                        scale_deg=1.60 / 3600.0,
                        rotation_deg=21.0,
                    ),
                    spectral_min=5300.0,
                    spectral_max=5800.0,
                    transmission_center=5550.0,
                    transmission_sigma=120.0,
                    psf_sigma_pix=1.35,
                    noise_sigma=0.030,
                    gap=(
                        slice(size // 4, size // 4 + 7),
                        slice(size // 3, size - size // 5),
                    ),
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=149.994,
                        crval2=2.0018,
                        scale_deg=1.10 / 3600.0,
                        rotation_deg=-18.0,
                    ),
                    spectral_min=6120.0,
                    spectral_max=6680.0,
                    transmission_center=6400.0,
                    transmission_sigma=130.0,
                    psf_sigma_pix=0.92,
                    noise_sigma=0.016,
                ),
                ObservationSpec(
                    header=_tan_header(
                        size,
                        crval1=150.0015,
                        crval2=1.9965,
                        scale_deg=0.82 / 3600.0,
                        rotation_deg=11.0,
                    ),
                    spectral_min=6920.0,
                    spectral_max=7380.0,
                    transmission_center=7150.0,
                    transmission_sigma=100.0,
                    psf_sigma_pix=0.55,
                    noise_sigma=0.010,
                ),
            ),
        ),
        "tpv_field_fit": CaseSpec(
            case="tpv_field_fit",
            projection="TPV",
            description="Distorted TPV exposures with lower overlap and stronger PSF variation",
            truth_model="sine_implicit",
            domain=tpv_domain,
            target_header=_tpv_header(
                size,
                crval1=210.0,
                crval2=-6.0,
                scale_deg=1.0 / 3600.0,
                pv_scale=1.0,
            ),
            spectral_axis=spectral_axis,
            spectral_bin_edges=spectral_bin_edges,
            observations=(
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.0,
                        crval2=-6.0,
                        scale_deg=1.0 / 3600.0,
                        rotation_deg=0.0,
                        pv_scale=1.0,
                    ),
                    spectral_min=4600.0,
                    spectral_max=5050.0,
                    transmission_center=4810.0,
                    transmission_sigma=110.0,
                    psf_sigma_pix=0.65,
                    noise_sigma=0.018,
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=210.018,
                        crval2=-5.982,
                        scale_deg=1.25 / 3600.0,
                        rotation_deg=27.0,
                        pv_scale=1.15,
                    ),
                    spectral_min=5300.0,
                    spectral_max=5850.0,
                    transmission_center=5580.0,
                    transmission_sigma=160.0,
                    psf_sigma_pix=0.95,
                    noise_sigma=0.023,
                    gap=(
                        slice(size // 3, size // 3 + 7),
                        slice(size // 2, size - size // 5),
                    ),
                ),
                ObservationSpec(
                    header=_tpv_header(
                        size,
                        crval1=209.986,
                        crval2=-6.016,
                        scale_deg=0.9 / 3600.0,
                        rotation_deg=-21.0,
                        pv_scale=0.9,
                    ),
                    spectral_min=6100.0,
                    spectral_max=6700.0,
                    transmission_center=6400.0,
                    transmission_sigma=170.0,
                    psf_sigma_pix=0.7,
                    noise_sigma=0.02,
                ),
            ),
        ),
    }


class _AnalyticField(torch.nn.Module):
    def __init__(self, domain: FieldDomain) -> None:
        super().__init__()
        self.domain = domain
        self.register_buffer("_anchor", torch.tensor(0.0), persistent=False)

    def forward(
        self,
        ra_deg: Tensor,
        dec_deg: Tensor,
        wavelength_angstrom: Tensor | float,
    ) -> Tensor:
        wavelength = torch.as_tensor(
            wavelength_angstrom,
            device=ra_deg.device,
            dtype=ra_deg.dtype,
        )
        if wavelength.shape != ra_deg.shape:
            wavelength = torch.broadcast_to(wavelength, ra_deg.shape)
        lam = (wavelength - float(self.domain.wavelength_center)) / float(
            self.domain.wavelength_scale
        )
        dra = ((ra_deg - float(self.domain.center_ra_deg) + 180.0) % 360.0) - 180.0
        dra = dra * math.cos(math.radians(self.domain.center_dec_deg))
        ddec = dec_deg - float(self.domain.center_dec_deg)
        g1 = (1.0 + 0.40 * lam) * torch.exp(
            -((dra + 0.010) ** 2 + (ddec - 0.006) ** 2) / 0.00018
        )
        g2 = (0.8 - 0.18 * lam) * torch.exp(
            -((dra - 0.014) ** 2 + (ddec + 0.008) ** 2) / 0.00011
        )
        ridge = (
            0.18
            * torch.sin(130.0 * dra + 0.5 * lam)
            * torch.cos(110.0 * ddec - 0.25 * lam)
        )
        diffuse = 0.22 + 1.8 * dra + 1.1 * ddec + 0.12 * lam + 0.08 * lam * dra
        line = (0.12 + 0.05 * torch.cos(3.0 * lam)) * torch.exp(
            -torch.square(ddec + 0.008 * torch.sin(2.5 * lam)) / 0.00024
        )
        out = diffuse + g1 + g2 + ridge + line
        return torch.nn.functional.softplus(out) + 1.0e-4


def _make_truth_model(case_spec: CaseSpec, *, seed: int) -> torch.nn.Module:
    domain = case_spec.domain
    if case_spec.truth_model == "analytic_basis":
        model = _AnalyticField(domain)
        model.eval()
        return model
    torch.manual_seed(int(seed))
    model = SineImplicitField(domain, hidden_dim=48, hidden_layers=2, omega0=6.0)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    model.eval()
    return model


def _simulate_observations(
    case_spec: CaseSpec,
    *,
    seed: int,
    tmpdir: Path,
    num_spectral_samples: int = 9,
) -> tuple[list[FieldObservation], list[ImageObservation], Tensor]:
    truth_model = _make_truth_model(case_spec, seed=seed)
    truth_cube = render_field_cube(
        truth_model,
        target_header=case_spec.target_header,
        spectral_axis=case_spec.spectral_axis,
    ).detach()
    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed) + 17)
    observations: list[FieldObservation] = []
    image_observations: list[ImageObservation] = []
    for spec in case_spec.observations:
        blank = torch.zeros(
            int(spec.header["NAXIS2"]),
            int(spec.header["NAXIS1"]),
            dtype=torch.float32,
        )
        field_observation = FieldObservation(
            image=blank,
            header=spec.header,
            spectral_min=spec.spectral_min,
            spectral_max=spec.spectral_max,
            transmission_center=spec.transmission_center,
            transmission_sigma=spec.transmission_sigma,
            psf_sigma_pix=spec.psf_sigma_pix,
            band_name=f"{int(spec.transmission_center)}A",
        )
        clean = render_field_observation(
            truth_model,
            field_observation,
            num_spectral_samples=int(num_spectral_samples),
        ).detach()
        noise = spec.noise_sigma * torch.randn(
            clean.shape,
            generator=generator,
            dtype=clean.dtype,
        )
        image = clean + noise
        weight = torch.full_like(image, 1.0 / max(spec.noise_sigma**2, 1.0e-6))
        mask = torch.ones_like(image, dtype=torch.bool)
        if spec.gap is not None:
            mask[spec.gap] = False
            weight[spec.gap] = 0.0
        band_name = f"{int(spec.transmission_center)}A"
        observations.append(
            FieldObservation(
                image=image,
                header=spec.header,
                weight=weight,
                mask=mask,
                spectral_min=spec.spectral_min,
                spectral_max=spec.spectral_max,
                transmission_center=spec.transmission_center,
                transmission_sigma=spec.transmission_sigma,
                psf_sigma_pix=(
                    spec.psf_sigma_pix
                    if spec.fit_psf_sigma_pix is None
                    else spec.fit_psf_sigma_pix
                ),
                band_name=band_name,
            )
        )
        obs_index = len(observations) - 1
        image_path = tmpdir / f"{case_spec.case}_{obs_index}.fits"
        weight_path = tmpdir / f"{case_spec.case}_{obs_index}_wht.fits"
        torchsky.write(str(image_path), image, header=spec.header, overwrite=True)
        torchsky.write(str(weight_path), weight, header=spec.header, overwrite=True)
        image_observations.append(
            ImageObservation(
                str(image_path),
                weight_path=str(weight_path),
                wavelength=0.5 * (spec.spectral_min + spec.spectral_max),
                spectral_min=spec.spectral_min,
                spectral_max=spec.spectral_max,
                transmission_center=spec.transmission_center,
                transmission_sigma=spec.transmission_sigma,
                band_name=band_name,
            )
        )
    return observations, image_observations, truth_cube


def _pixel_scale_arcsec(header: dict[str, Any]) -> float:
    cd11 = float(header.get("CD1_1", 0.0))
    cd12 = float(header.get("CD1_2", 0.0))
    cd21 = float(header.get("CD2_1", 0.0))
    cd22 = float(header.get("CD2_2", 0.0))
    scale_deg = max(
        math.hypot(cd11, cd12),
        math.hypot(cd21, cd22),
        1.0e-12,
    )
    return 3600.0 * scale_deg


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if not rows:
        return []
    metric_name = (
        "support_reconstruction_rmse"
        if all("support_reconstruction_rmse" in row for row in rows)
        else "reconstruction_rmse"
    )
    best_rmse = min(float(row[metric_name]) for row in rows)
    within = [row for row in rows if float(row[metric_name]) <= 1.05 * best_rmse]
    default = min(within, key=lambda row: float(row["fit_time_ms"]))
    best = min(rows, key=lambda row: float(row[metric_name]))
    return [
        {
            "library": "torchsky",
            "type": "field_recommendation",
            "case": default["case"],
            "profile": "default_general",
            "recommended_model": default["model_kind"],
            "criterion": f"fastest_within_5pct_{metric_name}",
            "reconstruction_rmse": default["reconstruction_rmse"],
            metric_name: default[metric_name],
            "fit_time_ms": default["fit_time_ms"],
        },
        {
            "library": "torchsky",
            "type": "field_recommendation",
            "case": best["case"],
            "profile": "high_accuracy",
            "recommended_model": best["model_kind"],
            "criterion": f"lowest_{metric_name}",
            "reconstruction_rmse": best["reconstruction_rmse"],
            metric_name: best[metric_name],
            "fit_time_ms": best["fit_time_ms"],
        },
    ]


def _placeholder_field_observations_for_spectral_budget(
    case_spec: CaseSpec,
) -> list[FieldObservation]:
    """Minimal :class:`FieldObservation` rows for :func:`spectral_data_budget`."""
    rows: list[FieldObservation] = []
    for spec in case_spec.observations:
        rows.append(
            FieldObservation(
                image=torch.zeros(1, 1, dtype=torch.float32),
                header=spec.header,
                spectral_min=spec.spectral_min,
                spectral_max=spec.spectral_max,
                transmission_center=spec.transmission_center,
                transmission_sigma=spec.transmission_sigma,
                band_name=f"{int(spec.transmission_center)}A",
            )
        )
    return rows


def evaluate_case(
    *,
    case: str,
    model_kind: str,
    size: int,
    seed: int,
    steps: int,
    runs: int,
    use_spectral_budget: bool = False,
    training_patch_size_override: int = -1,
    diagnostic_tile_size_override: int = -1,
) -> dict[str, Any]:
    case_spec = _build_case_specs(size)[case]
    spectral_budget_extras: dict[str, Any] = {}
    simulate_num_spectral = 9
    if use_spectral_budget:
        budget = spectral_data_budget(
            _placeholder_field_observations_for_spectral_budget(case_spec)
        )
        n_suggest = suggest_num_spectral_samples(budget)
        hints = suggest_spectral_model_capacity(budget)
        simulate_num_spectral = n_suggest
        spectral_budget_extras = {
            "use_spectral_budget": True,
            "suggested_num_spectral_samples": n_suggest,
            "spectral_wavelength_span_angstrom": budget.wavelength_span,
            "spectral_min_variation_scale_angstrom": budget.min_variation_scale,
            "spectral_n_distinct_band_specs": budget.n_distinct_band_specs,
            "spectral_budget_observations": [
                {
                    "lower": s.lower,
                    "upper": s.upper,
                    "span": s.span,
                    "transmission_sigma": s.transmission_sigma,
                    "variation_scale": s.variation_scale,
                    "band_name": s.band_name,
                }
                for s in budget.observations
            ],
            "spectral_capacity_hints": {
                "spectroscopicity": hints.spectroscopicity,
                "profile": hints.profile,
                "spectral_frequencies": hints.spectral_frequencies,
                "num_spectral_frequencies_basis": hints.num_spectral_frequencies_basis,
                "recommended_model_kinds": list(hints.recommended_model_kinds),
                "rationale": hints.rationale,
            },
        }
    with tempfile.TemporaryDirectory(prefix="torchsky-field-fit-") as tmpdir_str:
        observations, image_observations, truth_cube = _simulate_observations(
            case_spec,
            seed=seed,
            tmpdir=Path(tmpdir_str),
            num_spectral_samples=simulate_num_spectral,
        )
        product = plan_map_product(
            image_observations,
            header_overrides=case_spec.target_header,
            spectral_axis=case_spec.spectral_axis,
            spectral_bin_edges=case_spec.spectral_bin_edges,
        )
        render_chunk_size = 64 if max(product.shape) >= 128 else None
        support_reference_product = stack_map_product(
            image_observations,
            product,
            resampling_kernel="bilinear",
        )
        domain = estimate_field_domain(
            observations,
            coord_mode=case_spec.domain.coord_mode,
        )
        fit_times = []
        result = None
        product_pred = None
        observation_residuals = []
        model_kwargs: dict[str, Any] = {}
        training_patch_size: int | None = None
        diagnostic_spatial_tile_hw: tuple[int, int] | None = None
        training_performance: dict[str, Any] = {}
        fit_process_peak_rss = _process_peak_rss_bytes()
        if model_kind.startswith("coadd_"):
            kernel = model_kind.split("_", 1)[1]
            coadd_times = []
            result_product = None
            stack_map_product(
                image_observations,
                product,
                resampling_kernel=kernel,
            )
            for _ in range(int(runs)):
                start = time.perf_counter()
                result_product = stack_map_product(
                    image_observations,
                    product,
                    resampling_kernel=kernel,
                )
                coadd_times.append((time.perf_counter() - start) * 1000.0)
            assert result_product is not None
            fit_times = coadd_times
            product_pred = result_product
        else:
            fit_kwargs = _field_fit_kwargs_for_model(
                model_kind,
                case=case_spec.case,
                size=size,
                benchmark_steps=steps,
            )
            if use_spectral_budget:
                fit_kwargs["num_spectral_samples"] = spectral_budget_extras[
                    "suggested_num_spectral_samples"
                ]
            if training_patch_size_override >= 0:
                fit_kwargs["training_patch_size"] = (
                    None
                    if training_patch_size_override == 0
                    else int(training_patch_size_override)
                )
            if diagnostic_tile_size_override >= 0:
                fit_kwargs["diagnostic_spatial_tile_hw"] = (
                    None
                    if diagnostic_tile_size_override == 0
                    else (
                        int(diagnostic_tile_size_override),
                        int(diagnostic_tile_size_override),
                    )
                )
            training_patch_size = fit_kwargs.get("training_patch_size")
            diagnostic_spatial_tile_hw = fit_kwargs.get("diagnostic_spatial_tile_hw")
            num_spectral_render = int(fit_kwargs["num_spectral_samples"])
            model_kwargs = _field_model_kwargs_for_case(
                model_kind,
                case=case_spec.case,
                size=size,
            )
            warmup_steps = _warmup_steps_for_case(case_spec.case, benchmark_steps=steps)
            if warmup_steps > 0:
                torch.manual_seed(int(seed) + 999)
                warmup_model = _make_field_model_for_case(
                    model_kind,
                    domain,
                    case=case_spec.case,
                    size=size,
                )
                if model_kind == "explicit_spectral_grid_field":
                    assert isinstance(warmup_model, ExplicitSpectralGridField)
                    warmup_model.initialize_from_product(support_reference_product)
                fit_field_model(
                    observations,
                    warmup_model,
                    steps=warmup_steps,
                    **fit_kwargs,
                )
            for run_idx in range(int(runs)):
                torch.manual_seed(int(seed) + 1000 + run_idx)
                model = _make_field_model_for_case(
                    model_kind,
                    domain,
                    case=case_spec.case,
                    size=size,
                )
                if model_kind == "explicit_spectral_grid_field":
                    assert isinstance(model, ExplicitSpectralGridField)
                    model.initialize_from_product(support_reference_product)
                start = time.perf_counter()
                result = fit_field_model(
                    observations,
                    model,
                    steps=steps,
                    **fit_kwargs,
                )
                fit_times.append((time.perf_counter() - start) * 1000.0)
            assert result is not None
            if result.diagnostics is not None:
                training_performance = dict(
                    result.diagnostics.get("performance_profile", {})
                )
            fit_process_peak_rss = _process_peak_rss_bytes()
            product_pred = _render_field_product_chunked(
                result.model,
                product,
                spatial_chunk_size=render_chunk_size,
                num_spectral_samples=num_spectral_render,
            )
            with torch.no_grad():
                for observation in observations:
                    pred = render_field_observation(
                        result.model,
                        observation,
                        num_spectral_samples=num_spectral_render,
                    )
                    observation_residuals.append(
                        torch.sqrt(torch.mean(torch.square(pred - observation.image)))
                    )
        assert product_pred is not None
        metrics = evaluate_map_product(product_pred, truth_cube)
        support_metrics = evaluate_map_product(
            product_pred.data,
            truth_cube,
            weight=support_reference_product.weight,
            header=product_pred.header,
            value_semantics=product_pred.product.value_semantics,
        )
        uniform_metrics = evaluate_map_product(
            product_pred.data,
            truth_cube,
            weight=torch.ones_like(product_pred.data, dtype=torch.float32),
            header=product_pred.header,
            value_semantics=product_pred.product.value_semantics,
        )
        has_native_uncertainty = bool(
            torch.any(
                torch.abs(
                    product_pred.weight.to(dtype=torch.float32)
                    - torch.ones_like(product_pred.weight, dtype=torch.float32)
                )
                > 1.0e-6
            )
        )
        uncertainty_mode = (
            "native_inverse_variance"
            if result is None or has_native_uncertainty
            else "support_density_proxy"
        )
        uncertainty_inverse_variance = estimate_map_inverse_variance(
            product_pred,
            mode=uncertainty_mode,
            support_weight=support_reference_product.weight,
        )
        uncertainty_metrics = evaluate_map_uncertainty(
            uncertainty_inverse_variance,
            product_pred,
            truth_cube,
            support_weight=support_reference_product.weight,
        )
        frontier_region_metrics = evaluate_map_frontier_regions(
            product_pred,
            truth_cube,
            support_weight=support_reference_product.weight,
            overlap_count=support_reference_product.overlap_count,
        )
        cube_pred = product_pred.data
    signal_rms = float(torch.sqrt(torch.mean(torch.square(truth_cube))).item())
    residual_rms = float(metrics["reconstruction_rmse"])
    initial_loss = result.initial_loss if result is not None else None
    final_loss = result.final_loss if result is not None else None
    loss_improvement = (
        result.initial_loss / max(result.final_loss, 1.0e-8)
        if result is not None
        else None
    )
    fit_times_sorted = sorted(float(value) for value in fit_times)
    fit_time_ms = fit_times_sorted[len(fit_times_sorted) // 2]
    return {
        "library": "torchsky",
        "type": "field_fit" if result is not None else "cube_baseline",
        "case": case_spec.case,
        "family": case_spec.family,
        "projection": case_spec.projection,
        "description": case_spec.description,
        "truth_model": case_spec.truth_model,
        "model_kind": model_kind,
        "model_kwargs": model_kwargs if result is not None else {},
        "fit_method": "weighted_mse_adam"
        if result is not None
        else "weighted_cube_coadd",
        "num_observations": len(observations),
        "num_bands": int(case_spec.spectral_axis.values.numel()),
        "steps": int(steps),
        "training_patch_size": training_patch_size,
        "diagnostic_spatial_tile_hw": diagnostic_spatial_tile_hw,
        "fit_time_ms": fit_time_ms,
        "training_step_duration_mean_ms": training_performance.get(
            "step_duration_mean_ms"
        ),
        "training_throughput_pixels_per_sec": training_performance.get(
            "throughput_pixels_per_sec"
        ),
        "training_max_accelerator_memory_bytes": training_performance.get(
            "max_memory_allocated_bytes"
        ),
        "fit_process_peak_rss_bytes": fit_process_peak_rss,
        "process_peak_rss_bytes": _process_peak_rss_bytes(),
        "mpix_s": float(cube_pred.numel()) / max(fit_time_ms * 1.0e-3, 1.0e-12) / 1.0e6,
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "loss_improvement_ratio": loss_improvement,
        **metrics,
        "support_reference_model_kind": "coadd_bilinear",
        "support_reference_coverage_fraction": float(
            support_reference_product.weight.gt(0.0)
            .to(dtype=torch.float32)
            .mean()
            .item()
        ),
        "uniform_reconstruction_rmse": float(
            uniform_metrics["reconstruction_rmse"]  # type: ignore[arg-type]
        ),
        "uniform_reconstruction_mae": float(
            uniform_metrics["reconstruction_mae"]  # type: ignore[arg-type]
        ),
        "uniform_signal_to_residual_db": float(
            uniform_metrics["signal_to_residual_db"]  # type: ignore[arg-type]
        ),
        "uniform_total_flux_relative_error": float(
            uniform_metrics["total_flux_relative_error"]  # type: ignore[arg-type]
        ),
        "support_reconstruction_rmse": float(
            support_metrics["reconstruction_rmse"]  # type: ignore[arg-type]
        ),
        "support_band_reconstruction_rmse": list(
            support_metrics["band_reconstruction_rmse"]  # type: ignore[arg-type]
        ),
        "support_reconstruction_mae": float(
            support_metrics["reconstruction_mae"]  # type: ignore[arg-type]
        ),
        "support_reconstruction_p95_abs_err": float(
            support_metrics["reconstruction_p95_abs_err"]  # type: ignore[arg-type]
        ),
        "support_reconstruction_max_abs_err": float(
            support_metrics["reconstruction_max_abs_err"]  # type: ignore[arg-type]
        ),
        "support_signal_to_residual_db": float(
            support_metrics["signal_to_residual_db"]  # type: ignore[arg-type]
        ),
        "support_total_flux_relative_error": float(
            support_metrics["total_flux_relative_error"]  # type: ignore[arg-type]
        ),
        "support_band_flux_relative_error": list(
            support_metrics["band_flux_relative_error"]  # type: ignore[arg-type]
        ),
        "uncertainty_kind": uncertainty_mode,
        **uncertainty_metrics,
        **frontier_region_metrics,
        "band_centers_angstrom": [
            float(value)
            for value in case_spec.spectral_axis.to_wavelength().detach().cpu().tolist()
        ],
        "signal_rms_truth": signal_rms,
        "residual_rms": residual_rms,
        "signal_to_residual_ratio": signal_rms / max(residual_rms, 1.0e-8),
        "signal_to_residual_db": 20.0
        * math.log10(signal_rms / max(residual_rms, 1.0e-8)),
        "observation_rmse": (
            float(torch.stack(observation_residuals).mean().item())
            if observation_residuals
            else None
        ),
        "mean_noise_sigma": float(
            sum(spec.noise_sigma for spec in case_spec.observations)
            / len(case_spec.observations)
        ),
        "min_noise_sigma": float(
            min(spec.noise_sigma for spec in case_spec.observations)
        ),
        "max_noise_sigma": float(
            max(spec.noise_sigma for spec in case_spec.observations)
        ),
        "min_psf_sigma_pix": float(
            min(spec.psf_sigma_pix for spec in case_spec.observations)
        ),
        "max_psf_sigma_pix": float(
            max(spec.psf_sigma_pix for spec in case_spec.observations)
        ),
        "min_resolution_arcsec_pix": float(
            min(_pixel_scale_arcsec(spec.header) for spec in case_spec.observations)
        ),
        "max_resolution_arcsec_pix": float(
            max(_pixel_scale_arcsec(spec.header) for spec in case_spec.observations)
        ),
        "min_bandwidth_angstrom": float(
            min(
                spec.spectral_max - spec.spectral_min for spec in case_spec.observations
            )
        ),
        "max_bandwidth_angstrom": float(
            max(
                spec.spectral_max - spec.spectral_min for spec in case_spec.observations
            )
        ),
        **spectral_budget_extras,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument(
        "--step-profile",
        type=str,
        default="",
        choices=("", *sorted(STEP_PROFILES)),
        help="Optional per-case step-budget profile for focused frontier runs.",
    )
    parser.add_argument("--runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--cases",
        type=str,
        default=",".join(DEFAULT_CASES),
    )
    parser.add_argument(
        "--models",
        type=str,
        default=",".join(DEFAULT_MODEL_KINDS),
    )
    parser.add_argument("--json-out", type=str, default="")
    parser.add_argument("--markdown-out", type=str, default="")
    parser.add_argument("--baseline-json", type=str, default="")
    parser.add_argument(
        "--training-patch-size",
        type=int,
        default=-1,
        help=(
            "Override the model/case training core size for profiling: -1 keeps "
            "the benchmark default, 0 fits full observations, and a positive "
            "value uses PSF-support-aware patches."
        ),
    )
    parser.add_argument(
        "--diagnostic-tile-size",
        type=int,
        default=-1,
        help=(
            "Override pre/post-fit diagnostic rendering: -1 keeps the benchmark "
            "default, 0 renders whole observations, and a positive value renders "
            "square coordinate tiles."
        ),
    )
    parser.add_argument(
        "--use-spectral-budget",
        action="store_true",
        help=(
            "Derive num_spectral_samples from observation bandpasses; use the same "
            "quadrature for simulation, fit, cube render, and per-observation residuals."
        ),
    )
    args = parser.parse_args()

    cases = tuple(item for item in args.cases.split(",") if item)
    models = tuple(item for item in args.models.split(",") if item)
    rows = []
    for case in cases:
        case_steps = _steps_for_case(
            case,
            base_steps=args.steps,
            step_profile=args.step_profile or None,
        )
        case_rows = [
            evaluate_case(
                case=case,
                model_kind=model_kind,
                size=args.size,
                seed=args.seed,
                steps=case_steps,
                runs=args.runs,
                use_spectral_budget=args.use_spectral_budget,
                training_patch_size_override=args.training_patch_size,
                diagnostic_tile_size_override=args.diagnostic_tile_size,
            )
            for model_kind in models
        ]
        rows.extend(case_rows)
        rows.extend(_recommendation_rows(case_rows))
    if args.json_out:
        output_path = Path(args.json_out)
        markdown_path = (
            Path(args.markdown_out)
            if args.markdown_out
            else output_path.with_suffix(".md")
        )
        markdown = write_map_benchmark_artifacts(
            rows,
            json_out=output_path,
            markdown_out=markdown_path,
        )
        print(markdown)
        print(f"JSON: {output_path}")
        print(f"Markdown: {markdown_path}")
    elif args.markdown_out:
        markdown_path = Path(args.markdown_out)
        markdown = write_map_benchmark_artifacts(
            rows,
            json_out=None,
            markdown_out=markdown_path,
        )
        print(markdown)
        print(f"Markdown: {markdown_path}")
    else:
        print(write_map_benchmark_artifacts(rows))
    if args.baseline_json:
        baseline_rows = json.loads(Path(args.baseline_json).read_text(encoding="utf-8"))
        errors = check_map_benchmark_regressions(rows, baseline_rows)
        if errors:
            raise SystemExit("\n".join(errors))


if __name__ == "__main__":
    main()
