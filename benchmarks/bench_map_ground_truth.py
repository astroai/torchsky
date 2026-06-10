#!/usr/bin/env python3
"""Benchmark torchsky mapmaking against synthetic ground truth."""

from __future__ import annotations

import argparse
import json
import math
import sys
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import torchsky  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.sphere import healpix as hp  # noqa: E402
from torchsky.map import (  # noqa: E402
    check_map_benchmark_regressions,
    FieldObservation,
    ImageObservation,
    MapBandpassSpec,
    MapProductSpec,
    estimate_field_domain,
    evaluate_map_product,
    fit_field_model,
    make_field_model,
    plan_map_product,
    stack_map_product,
    write_map_benchmark_artifacts,
)
import torchsky.map.image_plane as image_plane_mod  # noqa: E402
from torchsky.map.product import render_scene_product  # noqa: E402
from torchsky.map.target import TargetRenderSpec  # noqa: E402
from torchsky.map.operators import (  # noqa: E402
    headers_equivalent_for_direct_copy,
    ObservationOperatorBundle,
    make_minimal_learned_operator_scaffold,
    spherical_spectral_operator_bundle,
    swap_bundle_operator_families,
    validate_image_plane_2d,
)
from torchsky.map.scenes import DiscreteTargetImage, HealpixMapScene  # noqa: E402
from torchsky.map.solvers import (  # noqa: E402
    build_solver_hooks,
    build_spherical_harmonic_solver_hooks,
    build_spherical_spin_harmonic_solver_hooks,
    build_state_subspace_operator,
    build_teb_cls_prior_operator,
    compose_bound_operators,
    make_diagonal_preconditioner,
)
import bench_map_compat as map_compat  # noqa: E402
from torchsky.map.operators.background import fit_background_plane  # noqa: E402

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover - native bench env requirement
    raise RuntimeError(
        "bench_map_ground_truth.py requires torchsky.cpp; "
        "run it from a native-extension environment"
    ) from exc


DEFAULT_CASES = (
    "tan_aligned_low_noise",
    "tan_aligned_high_noise",
    "tan_multires_noise",
    "tan_multires_extreme",
    "tan_random10_large_union",
    "tan_low_overlap_exptime_transmission",
    "tpv_multires_noise",
    "tan_flux_multires",
    "tpv_flux_multires_mild",
    "tpv_flux_multires",
)
DEFAULT_TORCHSKY_KERNELS = (
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
    "lanczos5",
    "mitchell_netravali",
    "catmull_rom",
    "conservative_polygon",
)
_INTERPOLATORY_KERNELS = {
    "nearest",
    "bilinear",
    "bicubic",
    "lanczos3",
    "lanczos5",
    "catmull_rom",
}
_SMOOTH_RECONSTRUCTION_KERNELS = {"mitchell_netravali"}


@dataclass
class SimulatedObservation:
    observation: ImageObservation
    header: dict[str, Any]
    exposure_time: float
    transmission: float
    psf_sigma_pix: float
    noise_sigma_flux: float
    zero_weight_fraction: float
    explicit_mask_fraction: float
    used_explicit_mask: bool


@dataclass(frozen=True)
class _CaseSpec:
    case: str
    projection: str
    description: str
    base_noise_flux: float
    value_semantics: str
    noise_model: str
    specs: tuple[
        tuple[dict[str, Any], float, float, float, tuple[slice, slice] | None], ...
    ]
    explicit_masks: bool = False
    weight_gradient: bool = False


def _count_warped_inputs(observations: list[SimulatedObservation]) -> int:
    return sum(
        1
        for item in observations
        if "TPV" in str(item.header.get("CTYPE1", ""))
        or "TPV" in str(item.header.get("CTYPE2", ""))
    )


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
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4 * pv_scale
    header["PV1_5"] = -3.0e-4 * pv_scale
    header["PV1_6"] = 1.5e-4 * pv_scale
    header["PV1_7"] = 2.0e-6 * pv_scale
    header["PV1_39"] = 2.0e-11 * pv_scale
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4 * pv_scale
    header["PV2_5"] = 2.5e-4 * pv_scale
    header["PV2_6"] = -2.0e-4 * pv_scale
    header["PV2_7"] = -1.5e-6 * pv_scale
    header["PV2_39"] = -1.0e-11 * pv_scale
    return header


def _shift_header_center(
    header: dict[str, Any],
    *,
    dra_deg: float = 0.0,
    ddec_deg: float = 0.0,
) -> dict[str, Any]:
    shifted = dict(header)
    shifted["CRVAL1"] = float(header["CRVAL1"]) + float(dra_deg)
    shifted["CRVAL2"] = float(header["CRVAL2"]) + float(ddec_deg)
    return shifted


def _benchmark_healpix_nside_for_target(
    target_header: dict[str, Any],
    *,
    oversample: float = 2.0,
) -> int:
    target_pixels = int(target_header["NAXIS1"]) * int(target_header["NAXIS2"])
    required = max(int(math.ceil(float(oversample) * target_pixels)), 12)
    nside = 1
    while 12 * nside * nside < required:
        nside *= 2
    return nside


def _sample_image_on_lonlat_grid(
    image: torch.Tensor,
    *,
    source_header: dict[str, Any],
    lon_deg: torch.Tensor,
    lat_deg: torch.Tensor,
) -> torch.Tensor:
    src = image.to(dtype=torch.float32)
    h_src, w_src = int(src.shape[0]), int(src.shape[1])
    source_wcs = WCS(dict(source_header))
    x_src, y_src = source_wcs.world_to_pixel(
        lon_deg.reshape(-1).to(torch.float64),
        lat_deg.reshape(-1).to(torch.float64),
    )
    x_src = x_src.reshape_as(lon_deg).to(torch.float32)
    y_src = y_src.reshape_as(lat_deg).to(torch.float32)
    x_norm = (2.0 * x_src / max(w_src - 1, 1)) - 1.0
    y_norm = (2.0 * y_src / max(h_src - 1, 1)) - 1.0
    grid = torch.stack([x_norm.reshape(-1), y_norm.reshape(-1)], dim=-1).view(
        1,
        x_norm.numel(),
        1,
        2,
    )
    sampled = F.grid_sample(
        src.view(1, 1, h_src, w_src),
        grid,
        mode="bilinear",
        padding_mode="zeros",
        align_corners=True,
    ).reshape_as(lon_deg)
    in_bounds = (
        (x_src >= 0.0)
        & (x_src <= float(w_src - 1))
        & (y_src >= 0.0)
        & (y_src <= float(h_src - 1))
    )
    return torch.where(in_bounds, sampled, torch.zeros_like(sampled))


def _truth_image_to_healpix_map(
    truth: torch.Tensor,
    *,
    target_header: dict[str, Any],
    nside: int,
) -> torch.Tensor:
    npix = hp.nside2npix(int(nside))
    ipix = torch.arange(npix, dtype=torch.int64)
    lon, lat = hp.pix2ang(int(nside), ipix, nest=False, lonlat=True)
    return _sample_image_on_lonlat_grid(
        truth,
        source_header=target_header,
        lon_deg=lon,
        lat_deg=lat,
    ).reshape(-1)


def _tan_rect_header(
    width: int,
    height: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
) -> dict[str, Any]:
    cd11, cd12, cd21, cd22 = _rotated_cd(scale_deg, rotation_deg)
    return {
        "NAXIS": 2,
        "NAXIS1": int(width),
        "NAXIS2": int(height),
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": float(width) / 2.0,
        "CRPIX2": float(height) / 2.0,
        "CRVAL1": crval1,
        "CRVAL2": crval2,
        "CD1_1": cd11,
        "CD1_2": cd12,
        "CD2_1": cd21,
        "CD2_2": cd22,
    }


def _tpv_rect_header(
    width: int,
    height: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
    pv_scale: float = 1.0,
) -> dict[str, Any]:
    header = _tan_rect_header(
        width,
        height,
        crval1=crval1,
        crval2=crval2,
        scale_deg=scale_deg,
        rotation_deg=rotation_deg,
    )
    header["CTYPE1"] = "RA---TPV"
    header["CTYPE2"] = "DEC--TPV"
    header["PV1_0"] = 0.0
    header["PV1_1"] = 1.0
    header["PV1_2"] = -0.02
    header["PV1_4"] = 2.0e-4 * pv_scale
    header["PV1_5"] = -3.0e-4 * pv_scale
    header["PV1_6"] = 1.5e-4 * pv_scale
    header["PV1_7"] = 2.0e-6 * pv_scale
    header["PV1_39"] = 2.0e-11 * pv_scale
    header["PV2_0"] = 0.0
    header["PV2_1"] = 0.02
    header["PV2_2"] = 1.0
    header["PV2_4"] = -1.0e-4 * pv_scale
    header["PV2_5"] = 2.5e-4 * pv_scale
    header["PV2_6"] = -2.0e-4 * pv_scale
    header["PV2_7"] = -1.5e-6 * pv_scale
    header["PV2_39"] = -1.0e-11 * pv_scale
    return header


def _weight_gradient_map(width: int, height: int) -> torch.Tensor:
    x = torch.linspace(0.85, 1.35, width, dtype=torch.float32)
    y = torch.linspace(0.9, 1.4, height, dtype=torch.float32)
    yy, xx = torch.meshgrid(y, x, indexing="ij")
    gradient = xx * yy
    return gradient / torch.clamp(torch.mean(gradient), min=1e-8)


def _random_large_rect_specs(
    *,
    seed: int,
    include_warps: bool = False,
) -> tuple[tuple[dict[str, Any], float, float, float, tuple[slice, slice] | None], ...]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    center_offsets = (
        (0.0, 0.0),
        (1.4, -0.5),
        (-2.0, 1.1),
        (3.6, -1.3),
        (-4.1, 1.5),
        (6.5, -0.8),
        (-7.0, 0.7),
        (0.8, 4.1),
        (-0.9, -3.9),
        (10.2, 4.6),
    )
    specs: list[
        tuple[dict[str, Any], float, float, float, tuple[slice, slice] | None]
    ] = []
    for idx, (offset_ra, offset_dec) in enumerate(center_offsets):
        width = int(torch.randint(500, 4001, (1,), generator=gen).item())
        height = int(torch.randint(500, 4001, (1,), generator=gen).item())
        scale_deg = float(torch.empty(1).uniform_(0.0030, 0.0068, generator=gen).item())
        rotation_deg = float(torch.empty(1).uniform_(-24.0, 24.0, generator=gen).item())
        crval1 = (
            12.0
            + float(offset_ra)
            + float(torch.empty(1).uniform_(-0.35, 0.35, generator=gen).item())
        )
        crval2 = (
            -3.0
            + float(offset_dec)
            + float(torch.empty(1).uniform_(-0.25, 0.25, generator=gen).item())
        )
        exposure_time = float(
            torch.empty(1).uniform_(45.0, 260.0, generator=gen).item()
        )
        transmission = float(torch.empty(1).uniform_(0.68, 1.08, generator=gen).item())
        psf_sigma_pix = float(torch.empty(1).uniform_(0.45, 1.55, generator=gen).item())
        if include_warps and idx % 2 == 1:
            pv_scale = float(torch.empty(1).uniform_(0.12, 0.45, generator=gen).item())
            header = _tpv_rect_header(
                width,
                height,
                crval1=crval1,
                crval2=crval2,
                scale_deg=scale_deg,
                rotation_deg=rotation_deg,
                pv_scale=pv_scale,
            )
        else:
            header = _tan_rect_header(
                width,
                height,
                crval1=crval1,
                crval2=crval2,
                scale_deg=scale_deg,
                rotation_deg=rotation_deg,
            )
        gap = None
        if idx % 4 == 1:
            gap = (
                slice(max(1, height // 5), max(2, min(height, height // 2))),
                slice(
                    max(1, width // 3),
                    max(2, min(width, width // 3 + max(8, width // 40))),
                ),
            )
        specs.append(
            (
                header,
                exposure_time,
                transmission,
                psf_sigma_pix,
                gap,
            )
        )
    return tuple(specs)


def _gaussian_kernel1d(sigma_pix: float, *, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * float(sigma_pix))))
    x = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel = torch.exp(-0.5 * (x / float(sigma_pix)) ** 2)
    kernel = kernel / kernel.sum()
    return kernel


def _gaussian_blur(image: torch.Tensor, sigma_pix: float) -> torch.Tensor:
    if sigma_pix <= 0.0:
        return image
    kernel = _gaussian_kernel1d(sigma_pix, device=image.device)
    kernel_x = kernel.view(1, 1, 1, -1)
    kernel_y = kernel.view(1, 1, -1, 1)
    out = image.unsqueeze(0).unsqueeze(0)
    out = torch.nn.functional.conv2d(out, kernel_x, padding=(0, kernel.numel() // 2))
    out = torch.nn.functional.conv2d(out, kernel_y, padding=(kernel.numel() // 2, 0))
    return out.squeeze(0).squeeze(0)


def _analytic_sky(
    ra_deg: torch.Tensor,
    dec_deg: torch.Tensor,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> torch.Tensor:
    dra = (ra_deg - float(center_ra_deg)) * math.cos(
        math.radians(float(center_dec_deg))
    )
    ddec = dec_deg - float(center_dec_deg)
    g1 = 1.4 * torch.exp(-((dra + 0.09) ** 2 + (ddec - 0.04) ** 2) / 0.0018)
    g2 = 0.9 * torch.exp(-((dra - 0.06) ** 2 + (ddec + 0.05) ** 2) / 0.0010)
    g3 = 0.6 * torch.exp(-((dra + 0.01) ** 2 + (ddec + 0.10) ** 2) / 0.00055)
    ripple = 0.12 * torch.sin(55.0 * dra) * torch.cos(48.0 * ddec)
    slope = 0.25 + 0.7 * (dra + 0.18) + 0.35 * (ddec - 0.03)
    return (slope + g1 + g2 + g3 + ripple).to(dtype=torch.float32)


def build_truth_image(
    header: dict[str, Any],
    *,
    value_semantics: str = "surface_brightness",
    flux_area_norm_sr: float | None = None,
) -> torch.Tensor:
    wcs = WCS(header)
    height = int(header["NAXIS2"])
    width = int(header["NAXIS1"])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    truth = _analytic_sky(
        ra_deg.to(dtype=torch.float32),
        dec_deg.to(dtype=torch.float32),
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
    )
    if value_semantics == "flux":
        if flux_area_norm_sr is None:
            raise ValueError("flux_area_norm_sr is required for flux truth images")
        area_sr = image_plane_mod._pixel_area_sr_map(
            wcs,
            (height, width),
            device=truth.device,
        )
        truth = truth * (area_sr / float(flux_area_norm_sr))
    return truth


def _simulate_observation(
    tmpdir: Path,
    *,
    name: str,
    header: dict[str, Any],
    exposure_time: float,
    transmission: float,
    psf_sigma_pix: float,
    base_noise_flux: float,
    seed: int,
    value_semantics: str,
    noise_model: str,
    flux_area_norm_sr: float | None = None,
    gap: tuple[slice, slice] | None = None,
    explicit_mask: bool = False,
    weight_gradient: bool = False,
) -> SimulatedObservation:
    wcs = WCS(header)
    height = int(header["NAXIS2"])
    width = int(header["NAXIS1"])
    y, x = torch.meshgrid(
        torch.arange(height, dtype=torch.float64),
        torch.arange(width, dtype=torch.float64),
        indexing="ij",
    )
    ra_deg, dec_deg = wcs.pixel_to_world(x, y)
    signal = _analytic_sky(
        ra_deg.to(dtype=torch.float32),
        dec_deg.to(dtype=torch.float32),
        center_ra_deg=float(header["CRVAL1"]),
        center_dec_deg=float(header["CRVAL2"]),
    )
    signal = _gaussian_blur(signal, psf_sigma_pix)
    if value_semantics == "flux":
        if flux_area_norm_sr is None:
            raise ValueError("flux_area_norm_sr is required for flux observations")
        area_sr = image_plane_mod._pixel_area_sr_map(
            wcs,
            (height, width),
            device=signal.device,
        )
        signal = signal * (area_sr / float(flux_area_norm_sr))

    scale = float(exposure_time) * float(transmission)
    if noise_model == "flux_constant":
        noise_sigma_flux_map = torch.full_like(
            signal,
            float(base_noise_flux) / math.sqrt(max(scale, 1e-8)),
        )
    elif noise_model == "flux_area_scaled":
        if value_semantics != "flux":
            raise ValueError("flux_area_scaled noise_model requires flux semantics")
        if flux_area_norm_sr is None:
            raise ValueError("flux_area_norm_sr is required for flux_area_scaled noise")
        area_sr = image_plane_mod._pixel_area_sr_map(
            wcs,
            (height, width),
            device=signal.device,
        )
        noise_sigma_flux_map = (
            float(base_noise_flux)
            * (area_sr / float(flux_area_norm_sr))
            / math.sqrt(max(scale, 1e-8))
        )
    else:
        raise ValueError(
            "noise_model must be one of {'flux_constant', 'flux_area_scaled'}"
        )
    gen = torch.Generator(device=signal.device)
    gen.manual_seed(int(seed))
    noisy_flux = signal + noise_sigma_flux_map * torch.randn(
        signal.shape,
        generator=gen,
        dtype=torch.float32,
    )
    weight = 1.0 / torch.clamp(torch.square(noise_sigma_flux_map), min=1e-20)
    if weight_gradient:
        weight = weight * _weight_gradient_map(width, height).to(device=weight.device)
    mask_tensor: torch.Tensor | None = None
    if explicit_mask:
        mask_tensor = torch.ones_like(weight, dtype=torch.float32)
        if gap is not None:
            mask_tensor[gap] = 0.0
        else:
            band = max(4, width // 28)
            row = max(4, height // 24)
            x0 = min(max(1, width // 3), max(1, width - band))
            y0 = min(max(1, height // 2), max(1, height - row))
            mask_tensor[:, x0 : x0 + band] = 0.0
            mask_tensor[y0 : y0 + row, :] = 0.0
    elif gap is not None:
        noisy_flux[gap] = 0.0
        weight[gap] = 0.0

    image_path = tmpdir / f"{name}.fits"
    weight_path = tmpdir / f"{name}_wht.fits"
    torchsky.write(str(image_path), noisy_flux, header=header, overwrite=True)
    torchsky.write(str(weight_path), weight, header=header, overwrite=True)
    mask_path = None
    if mask_tensor is not None:
        mask_path = tmpdir / f"{name}_msk.fits"
        torchsky.write(str(mask_path), mask_tensor, header=header, overwrite=True)
    zero_weight_fraction = float((weight <= 0.0).to(torch.float32).mean().item())
    explicit_mask_fraction = (
        float((mask_tensor <= 0.0).to(torch.float32).mean().item())
        if mask_tensor is not None
        else 0.0
    )
    return SimulatedObservation(
        observation=ImageObservation(
            str(image_path),
            weight_path=str(weight_path),
            mask_path=str(mask_path) if mask_path is not None else None,
        ),
        header=header,
        exposure_time=float(exposure_time),
        transmission=float(transmission),
        psf_sigma_pix=float(psf_sigma_pix),
        noise_sigma_flux=float(noise_sigma_flux_map.mean().item()),
        zero_weight_fraction=zero_weight_fraction,
        explicit_mask_fraction=explicit_mask_fraction,
        used_explicit_mask=mask_tensor is not None,
    )


def _case_spec(case: str, *, size: int) -> _CaseSpec:
    tan_aligned_specs = (
        (
            _tan_header(
                size,
                crval1=12.0,
                crval2=-3.0,
                scale_deg=0.0045,
                rotation_deg=0.0,
            ),
            30.0,
            0.72,
            0.55,
            None,
        ),
        (
            _tan_header(
                size,
                crval1=12.0,
                crval2=-3.0,
                scale_deg=0.0045,
                rotation_deg=0.0,
            ),
            90.0,
            0.93,
            0.8,
            None,
        ),
        (
            _tan_header(
                size,
                crval1=12.0,
                crval2=-3.0,
                scale_deg=0.0045,
                rotation_deg=0.0,
            ),
            180.0,
            0.98,
            1.0,
            None,
        ),
    )
    tan_multires_specs = (
        (
            _tan_header(
                size,
                crval1=12.0,
                crval2=-3.0,
                scale_deg=0.0045,
                rotation_deg=0.0,
            ),
            60.0,
            0.88,
            0.55,
            (slice(None), slice(size // 3, size // 3 + max(2, size // 18))),
        ),
        (
            _tan_header(
                size,
                crval1=12.018,
                crval2=-3.012,
                scale_deg=0.0058,
                rotation_deg=7.0,
            ),
            95.0,
            0.94,
            0.95,
            None,
        ),
        (
            _tan_header(
                size,
                crval1=11.992,
                crval2=-2.986,
                scale_deg=0.0039,
                rotation_deg=-11.0,
            ),
            140.0,
            0.73,
            1.35,
            (slice(size // 2 - 1, size // 2 + 2), slice(None)),
        ),
    )
    cases = {
        "tan_aligned_low_noise": _CaseSpec(
            case="tan_aligned_low_noise",
            projection="TAN",
            description="Low-noise aligned TAN coadd for noise-floor validation",
            base_noise_flux=0.02,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=tan_aligned_specs,
        ),
        "tan_aligned_high_noise": _CaseSpec(
            case="tan_aligned_high_noise",
            projection="TAN",
            description="High-noise aligned TAN coadd for noise-floor validation",
            base_noise_flux=0.25,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=tan_aligned_specs,
        ),
        "tan_multires_noise": _CaseSpec(
            case="tan_multires_noise",
            projection="TAN",
            description="Reference multiresolution TAN coadd with rotation and chip gaps",
            base_noise_flux=0.035,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=tan_multires_specs,
        ),
        "tan_multires_extreme": _CaseSpec(
            case="tan_multires_extreme",
            projection="TAN",
            description="Stronger TAN resolution-ratio and rotation stress case",
            base_noise_flux=0.05,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=(
                (
                    _tan_header(
                        size,
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0030,
                        rotation_deg=0.0,
                    ),
                    55.0,
                    0.90,
                    0.45,
                    None,
                ),
                (
                    _tan_header(
                        size,
                        crval1=12.028,
                        crval2=-3.024,
                        scale_deg=0.0068,
                        rotation_deg=18.0,
                    ),
                    110.0,
                    0.82,
                    1.15,
                    (slice(None), slice(size // 2, size // 2 + max(2, size // 14))),
                ),
                (
                    _tan_header(
                        size,
                        crval1=11.968,
                        crval2=-2.972,
                        scale_deg=0.0026,
                        rotation_deg=-21.0,
                    ),
                    180.0,
                    0.74,
                    1.55,
                    (slice(size // 3, size // 3 + max(2, size // 16)), slice(None)),
                ),
            ),
        ),
        "tan_random10_large_union": _CaseSpec(
            case="tan_random10_large_union",
            projection="TAN",
            description=(
                "Ten-image large TAN union mosaic with random rectangular input sizes "
                "between 500 and 4000 pixels, mixed resolution, rotation, and partial non-overlap"
            ),
            base_noise_flux=0.05,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=_random_large_rect_specs(seed=1234 + size, include_warps=False),
        ),
        "tan_random10_large_union_rect_weighted_masked": _CaseSpec(
            case="tan_random10_large_union_rect_weighted_masked",
            projection="TAN+TPV",
            description=(
                "Ten-image large TAN union mosaic on a rectangular target with random "
                "rectangular input sizes, mixed resolution, rotation, explicit per-pixel "
                "weight maps, explicit mask planes, and mixed TAN/TPV warped inputs"
            ),
            base_noise_flux=0.05,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            explicit_masks=True,
            weight_gradient=True,
            specs=_random_large_rect_specs(seed=2234 + size, include_warps=True),
        ),
        "tan_random10_large_union_rect_weighted_masked_tan": _CaseSpec(
            case="tan_random10_large_union_rect_weighted_masked_tan",
            projection="TAN",
            description=(
                "Ten-image large TAN union mosaic on a rectangular target with random "
                "rectangular input sizes, mixed resolution, rotation, explicit per-pixel "
                "weight maps, explicit mask planes, and TAN-only inputs"
            ),
            base_noise_flux=0.05,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            explicit_masks=True,
            weight_gradient=True,
            specs=_random_large_rect_specs(seed=3234 + size, include_warps=False),
        ),
        "tan_low_overlap_exptime_transmission": _CaseSpec(
            case="tan_low_overlap_exptime_transmission",
            projection="TAN",
            description=(
                "Low-overlap TAN coadd with exposure-time and transmission variation"
            ),
            base_noise_flux=0.045,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=(
                (
                    _tan_header(
                        size,
                        crval1=12.0,
                        crval2=-3.0,
                        scale_deg=0.0048,
                        rotation_deg=0.0,
                    ),
                    40.0,
                    0.65,
                    0.7,
                    None,
                ),
                (
                    _tan_header(
                        size,
                        crval1=12.060,
                        crval2=-3.018,
                        scale_deg=0.0052,
                        rotation_deg=6.0,
                    ),
                    130.0,
                    0.92,
                    1.1,
                    (slice(None), slice(size // 2, size // 2 + max(2, size // 16))),
                ),
                (
                    _tan_header(
                        size,
                        crval1=11.946,
                        crval2=-2.962,
                        scale_deg=0.0060,
                        rotation_deg=-9.0,
                    ),
                    220.0,
                    0.58,
                    1.45,
                    None,
                ),
            ),
        ),
        "tpv_multires_noise": _CaseSpec(
            case="tpv_multires_noise",
            projection="TPV",
            description="Distorted TPV multiresolution coadd with rotation and chip gaps",
            base_noise_flux=0.04,
            value_semantics="surface_brightness",
            noise_model="flux_constant",
            specs=(
                (
                    _tan_header(
                        size,
                        crval1=12.0,
                        crval2=-3.0,
                        scale_deg=0.0045,
                        rotation_deg=0.0,
                    ),
                    70.0,
                    0.87,
                    0.6,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=12.014,
                        crval2=-3.020,
                        scale_deg=0.0051,
                        rotation_deg=8.0,
                    ),
                    100.0,
                    0.91,
                    1.0,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=11.985,
                        crval2=-2.978,
                        scale_deg=0.0040,
                        rotation_deg=-12.0,
                    ),
                    135.0,
                    0.76,
                    1.3,
                    (slice(size // 3, size // 3 + max(2, size // 20)), slice(None)),
                ),
            ),
        ),
        "tan_flux_multires": _CaseSpec(
            case="tan_flux_multires",
            projection="TAN",
            description="Flux-per-pixel TAN multiresolution case for conservation checks",
            base_noise_flux=0.03,
            value_semantics="flux",
            noise_model="flux_area_scaled",
            specs=tan_multires_specs,
        ),
        "tpv_flux_multires_mild": _CaseSpec(
            case="tpv_flux_multires_mild",
            projection="TPV",
            description="Mild flux-per-pixel TPV multiresolution case for conservation checks",
            base_noise_flux=0.03,
            value_semantics="flux",
            noise_model="flux_area_scaled",
            specs=(
                (
                    _tan_header(
                        size,
                        crval1=12.0,
                        crval2=-3.0,
                        scale_deg=0.0042,
                        rotation_deg=0.0,
                    ),
                    80.0,
                    0.90,
                    0.6,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=12.010,
                        crval2=-3.015,
                        scale_deg=0.0049,
                        rotation_deg=8.0,
                        pv_scale=0.15,
                    ),
                    95.0,
                    0.92,
                    0.95,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=11.992,
                        crval2=-2.988,
                        scale_deg=0.0039,
                        rotation_deg=-9.0,
                        pv_scale=0.08,
                    ),
                    125.0,
                    0.84,
                    1.25,
                    (slice(size // 3, size // 3 + max(2, size // 20)), slice(None)),
                ),
            ),
        ),
        "tpv_flux_multires": _CaseSpec(
            case="tpv_flux_multires",
            projection="TPV",
            description="Flux-per-pixel TPV multiresolution case for conservation checks",
            base_noise_flux=0.035,
            value_semantics="flux",
            noise_model="flux_area_scaled",
            specs=(
                (
                    _tan_header(
                        size,
                        crval1=12.0,
                        crval2=-3.0,
                        scale_deg=0.0036,
                        rotation_deg=0.0,
                    ),
                    70.0,
                    0.87,
                    0.6,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=12.016,
                        crval2=-3.024,
                        scale_deg=0.0059,
                        rotation_deg=15.0,
                    ),
                    105.0,
                    0.89,
                    1.0,
                    None,
                ),
                (
                    _tpv_header(
                        size,
                        crval1=11.982,
                        crval2=-2.972,
                        scale_deg=0.0029,
                        rotation_deg=-17.0,
                    ),
                    150.0,
                    0.72,
                    1.4,
                    (slice(size // 3, size // 3 + max(2, size // 18)), slice(None)),
                ),
            ),
        ),
    }
    try:
        return cases[case]
    except KeyError as exc:
        raise ValueError(
            f"unknown case {case!r}; expected one of {', '.join(sorted(cases))}"
        ) from exc


def make_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    width: int | None = None,
    height: int | None = None,
) -> tuple[list[SimulatedObservation], dict[str, Any], torch.Tensor]:
    case_spec = _case_spec(case, size=size)
    target_width = int(width) if width is not None else int(size)
    if height is not None:
        target_height = int(height)
    elif case in {
        "tan_random10_large_union_rect_weighted_masked",
        "tan_random10_large_union_rect_weighted_masked_tan",
    }:
        target_height = max(32, int(round(float(size) * 0.75)))
    else:
        target_height = int(size)
    truth_header = _tan_rect_header(
        target_width,
        target_height,
        crval1=12.0,
        crval2=-3.0,
        scale_deg=0.0045,
        rotation_deg=0.0,
    )
    flux_area_norm_sr = None
    if case_spec.value_semantics == "flux":
        flux_area_norm_sr = float(
            torch.median(
                image_plane_mod._pixel_area_sr_map(
                    WCS(truth_header),
                    (target_height, target_width),
                    device=torch.device("cpu"),
                )
            ).item()
        )
    truth = build_truth_image(
        truth_header,
        value_semantics=case_spec.value_semantics,
        flux_area_norm_sr=flux_area_norm_sr,
    )
    observations = [
        _simulate_observation(
            tmpdir,
            name=f"{case}_{idx}",
            header=header,
            exposure_time=exptime,
            transmission=transmission,
            psf_sigma_pix=psf_sigma,
            base_noise_flux=case_spec.base_noise_flux,
            seed=seed + idx,
            value_semantics=case_spec.value_semantics,
            noise_model=case_spec.noise_model,
            flux_area_norm_sr=flux_area_norm_sr,
            gap=gap,
            explicit_mask=case_spec.explicit_masks,
            weight_gradient=case_spec.weight_gradient,
        )
        for idx, (header, exptime, transmission, psf_sigma, gap) in enumerate(
            case_spec.specs
        )
    ]
    return observations, truth_header, truth


def _time_many(fn, runs: int) -> float:
    fn()
    samples: list[float] = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _truth_metrics(
    data: torch.Tensor,
    weight: torch.Tensor,
    truth: torch.Tensor,
    *,
    target_header: dict[str, Any],
    value_semantics: str,
    region_masks: dict[str, torch.Tensor] | None = None,
) -> dict[str, float]:
    metrics = evaluate_map_product(
        data,
        truth,
        weight=weight,
        header=target_header,
        value_semantics=value_semantics,
    )
    mask = weight > 0.0
    residual = data - truth
    signal = truth[mask]
    signal_rms = (
        float(torch.sqrt(torch.mean(signal**2)).item()) if bool(mask.any()) else 0.0
    )
    if bool(mask.any()):
        residual_supported = residual[mask]
        weight_supported = weight[mask].to(dtype=torch.float32)
        weight_sum = torch.clamp(weight_supported.sum(), min=1.0e-8)
        weighted_rmse = float(
            torch.sqrt(
                torch.sum(
                    weight_supported
                    * torch.square(residual_supported.to(dtype=torch.float32))
                )
                / weight_sum
            ).item()
        )
        background_fit = fit_background_plane(residual, mask)
        background_rms = background_fit.model_rms
        detrended_rms = background_fit.detrended_rms
        residual_bias = background_fit.residual_bias
        background_leakage_fraction = background_fit.leakage_fraction
    else:
        weighted_rmse = 0.0
        background_rms = 0.0
        detrended_rms = 0.0
        residual_bias = 0.0
        background_leakage_fraction = 0.0
    metrics.update(
        {
            "signal_rms_truth": signal_rms,
            "residual_rms": float(metrics["reconstruction_rmse"]),
            "weighted_residual_rms": weighted_rmse,
            "background_residual_rms": background_rms,
            "detrended_residual_rms": detrended_rms,
            "background_leakage_fraction": background_leakage_fraction,
            "residual_bias": residual_bias,
            "output_snr": float(metrics["signal_to_residual_ratio"]),
            "output_snr_db": float(metrics["signal_to_residual_db"]),
            "coverage_frac": float(metrics["coverage_fraction"]),
            "rmse_vs_truth": float(metrics["reconstruction_rmse"]),
            "mae_vs_truth": float(metrics["reconstruction_mae"]),
            "p95_abs_err_vs_truth": float(metrics["reconstruction_p95_abs_err"]),
            "max_abs_err_vs_truth": float(metrics["reconstruction_max_abs_err"]),
            "total_flux_abs_err": float(metrics["total_flux_absolute_error"]),
            "total_flux_rel_err": float(metrics["total_flux_relative_error"]),
        }
    )
    if region_masks:
        for name, region_mask in region_masks.items():
            region_mask = region_mask.to(device=weight.device, dtype=torch.bool)
            region_weight = torch.where(region_mask, weight, torch.zeros_like(weight))
            region_metrics = evaluate_map_product(
                data,
                truth,
                weight=region_weight,
                header=target_header,
                value_semantics=value_semantics,
            )
            metrics.update(
                {
                    f"{name}_coverage_frac": float(region_metrics["coverage_fraction"]),
                    f"{name}_rmse_vs_truth": float(
                        region_metrics["reconstruction_rmse"]
                    ),
                    f"{name}_mae_vs_truth": float(region_metrics["reconstruction_mae"]),
                }
            )
    return metrics


def _clone_observation(
    obs: ImageObservation,
    *,
    keep_weight: bool = True,
    keep_mask: bool = True,
) -> ImageObservation:
    return ImageObservation(
        path=obs.path,
        hdu=obs.hdu,
        weight=obs.weight,
        photometric_scale=obs.photometric_scale,
        background_offset=obs.background_offset,
        wavelength=obs.wavelength,
        spectral_min=obs.spectral_min,
        spectral_max=obs.spectral_max,
        transmission_center=obs.transmission_center,
        transmission_sigma=obs.transmission_sigma,
        band_name=obs.band_name,
        weight_path=obs.weight_path if keep_weight else None,
        weight_hdu=obs.weight_hdu,
        mask_path=obs.mask_path if keep_mask else None,
        mask_hdu=obs.mask_hdu,
    )


def _target_mask_region_masks(
    observations: list[SimulatedObservation],
    *,
    target_header: dict[str, Any],
) -> dict[str, torch.Tensor]:
    if not any(item.observation.mask_path is not None for item in observations):
        return {}

    target_shape = (int(target_header["NAXIS2"]), int(target_header["NAXIS1"]))
    device = torch.device("cpu")
    prepared_target = image_plane_mod._prepare_target(
        header=target_header,
        target_wcs=WCS(target_header),
        target_shape=target_shape,
        device=device,
        include_pixel_area=False,
        include_quadrature=False,
    )
    mask_affected = torch.zeros(target_shape, dtype=torch.bool, device=device)
    for item in observations:
        obs_with_mask = item.observation
        if obs_with_mask.mask_path is None:
            continue
        obs_without_mask = _clone_observation(
            obs_with_mask, keep_weight=True, keep_mask=False
        )
        image, header = torchsky.read(
            obs_with_mask.path, hdu=obs_with_mask.hdu, return_header=True
        )
        image = validate_image_plane_2d(image).to(device=device)
        header_dict_obs = dict(header)
        source_wcs = None
        if not (
            headers_equivalent_for_direct_copy(
                header_dict_obs, target_header
            )
            and tuple(int(v) for v in image.shape) == target_shape
        ):
            source_wcs = WCS(header_dict_obs)
        _numer_masked, weight_masked, _overlap_masked, _moc_masked = (
            image_plane_mod._project_observation_to_target(
                obs_with_mask,
                image=image,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
            )
        )
        _numer_nomask, weight_nomask, _overlap_nomask, _moc_nomask = (
            image_plane_mod._project_observation_to_target(
                obs_without_mask,
                image=image,
                header_dict_obs=header_dict_obs,
                source_wcs=source_wcs,
                prepared_target=prepared_target,
                target_header=target_header,
                target_shape=target_shape,
                device=device,
                footprint_order=None,
                resampling_kernel="bilinear",
                value_semantics="surface_brightness",
            )
        )
        mask_affected |= torch.abs(weight_nomask - weight_masked) > 1.0e-6
    return {
        "mask_affected": mask_affected,
        "mask_clean": ~mask_affected,
    }


def evaluate_torchsky_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    resampling_kernel: str = "bilinear",
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    observations, target_header, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    obs_list = [item.observation for item in observations]
    region_masks = _target_mask_region_masks(observations, target_header=target_header)
    product = plan_map_product(
        obs_list,
        header_overrides=target_header,
        value_semantics=case_spec.value_semantics,
    )
    dt = _time_many(
        lambda: stack_map_product(
            obs_list,
            product,
            resampling_kernel=resampling_kernel,
        ),
        runs,
    )
    result = stack_map_product(
        obs_list,
        product,
        resampling_kernel=resampling_kernel,
    )
    return _build_product_benchmark_row(
        case=case,
        library="torchsky",
        method="coadd",
        variant=resampling_kernel,
        geometry_family=map_compat._geometry_family_for_torchsky_kernel(
            resampling_kernel
        ),
        case_spec=case_spec,
        observations=observations,
        target_header=target_header,
        truth=truth,
        region_masks=region_masks,
        result=result,
        size=size,
        time_ms=dt * 1000.0,
        extra={
            "interpolation_kernel": resampling_kernel,
            "resampling_method": "resample_weighted_numerator_and_denominator",
            "stacking_method": "weighted_mean",
            "goal": "reconstruct_union_field_on_target_grid",
        },
    )


def evaluate_torchsky_scene_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    operator_mode: str = "explicit",
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    observations, target_header, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    obs_list = [item.observation for item in observations]
    region_masks = _target_mask_region_masks(observations, target_header=target_header)
    product = plan_map_product(
        obs_list,
        header_overrides=target_header,
        value_semantics=case_spec.value_semantics,
    )
    scene = DiscreteTargetImage(truth)
    solver_hooks = None
    if operator_mode == "learned_shell":
        learned = make_minimal_learned_operator_scaffold()
        solver_hooks = build_solver_hooks(
            swap_bundle_operator_families(
                ObservationOperatorBundle(),
                learned,
            )
        )
    elif operator_mode != "explicit":
        raise ValueError("operator_mode must be 'explicit' or 'learned_shell'")
    dt = _time_many(
        lambda: render_scene_product(
            scene,
            product,
            solver_hooks=solver_hooks,
        ),
        runs,
    )
    result = render_scene_product(
        scene,
        product,
        solver_hooks=solver_hooks,
    )
    return _build_product_benchmark_row(
        case=case,
        library="torchsky_scene",
        method="scene_render",
        variant=operator_mode,
        geometry_family="scene_target_render",
        case_spec=case_spec,
        observations=observations,
        target_header=target_header,
        truth=truth,
        region_masks=region_masks,
        result=result,
        size=size,
        time_ms=dt * 1000.0,
        extra={
            "operator_mode": operator_mode,
            "goal": "render_latent_scene_on_target_grid",
        },
    )


def evaluate_torchsky_field_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    field_model_kind: str = "wire_implicit",
    fit_steps: int = 12,
    fit_lr: float = 1.0e-2,
    max_observations: int = 3,
    wavelength_angstrom: float = 5500.0,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    observations, target_header, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    region_masks = _target_mask_region_masks(observations, target_header=target_header)
    field_observations = _field_observations_from_simulated_inputs(
        observations,
        wavelength_angstrom=wavelength_angstrom,
        max_observations=max_observations,
    )
    product = MapProductSpec(
        target_header=target_header,
        bands=(MapBandpassSpec(label="mono", transmission_center=wavelength_angstrom),),
        value_semantics=case_spec.value_semantics,
    )

    def _fit_and_render():
        torch.manual_seed(int(seed))
        fit_domain = estimate_field_domain(field_observations)
        model = make_field_model(field_model_kind, fit_domain)
        fit = fit_field_model(
            field_observations,
            model,
            steps=int(fit_steps),
            lr=float(fit_lr),
            num_spectral_samples=1,
            loss_kind="mse",
            loss_reduction="global",
        )
        rendered = render_scene_product(
            fit.model,
            product,
            num_spectral_samples=1,
        )
        return fit, rendered

    dt = _time_many(_fit_and_render, runs)
    fit, result = _fit_and_render()
    return _build_product_benchmark_row(
        case=case,
        library="torchsky_field",
        method="field_fit",
        variant=field_model_kind,
        geometry_family="latent_scene_field",
        case_spec=case_spec,
        observations=observations,
        target_header=target_header,
        truth=truth,
        region_masks=region_masks,
        result=result,
        size=size,
        time_ms=dt * 1000.0,
        extra={
            "field_model_kind": field_model_kind,
            "fit_steps": int(fit_steps),
            "fit_lr": float(fit_lr),
            "fit_observations": len(field_observations),
            "goal": "infer_latent_scene_then_render_target_grid",
            "field_initial_loss": float(fit.initial_loss),
            "field_final_loss": float(fit.final_loss),
        },
    )


def evaluate_torchsky_spherical_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    spherical_mode: str = "harmonic_direct",
    healpix_oversample: float = 2.0,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    observations, target_header, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    region_masks = _target_mask_region_masks(observations, target_header=target_header)
    nside = _benchmark_healpix_nside_for_target(
        target_header,
        oversample=healpix_oversample,
    )
    hp_map = _truth_image_to_healpix_map(
        truth,
        target_header=target_header,
        nside=nside,
    )
    bundle = spherical_spectral_operator_bundle(mode=spherical_mode)
    if spherical_mode not in {"harmonic_direct", "healpix_bridge"}:
        raise ValueError("spherical_mode must be 'harmonic_direct' or 'healpix_bridge'")
    geometry_family = (
        "spherical_harmonic_direct"
        if spherical_mode == "harmonic_direct"
        else "spherical_spectral_healpix_bridge"
    )
    product = MapProductSpec(
        target_header=target_header,
        bands=(MapBandpassSpec(label="mono"),),
        value_semantics=case_spec.value_semantics,
    )
    scene = HealpixMapScene(
        hp_map,
        nside=nside,
        nest=False,
        lmax=min(3 * int(nside) - 1, 32),
    )
    solver_hooks = build_solver_hooks(bundle)

    def render_fn():
        return render_scene_product(
            scene,
            product,
            solver_hooks=solver_hooks,
        )

    dt = _time_many(render_fn, runs)
    result = render_fn()
    return _build_product_benchmark_row(
        case=case,
        library="torchsky_spherical",
        method="spherical_reproject",
        variant=spherical_mode,
        geometry_family=geometry_family,
        case_spec=case_spec,
        observations=observations,
        target_header=target_header,
        truth=truth,
        region_masks=region_masks,
        result=result,
        size=size,
        time_ms=dt * 1000.0,
        extra={
            "healpix_nside": int(nside),
            "operator_bundle_geometry": bundle.geometry.name
            if bundle.geometry is not None
            else None,
            "spherical_mode": spherical_mode,
            "goal": "reconstruct_union_field_on_target_grid_via_spherical_spectral_path",
        },
    )


def evaluate_torchsky_joint_teb_case(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    lmax: int = 3,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Any]:
    observations, target_header, _truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    torch.manual_seed(seed)
    nalm = (int(lmax) + 1) * (int(lmax) + 2) // 2

    teb_truth = torch.zeros((3, nalm), dtype=torch.complex64)
    teb_truth[0, 0] = complex(math.sqrt(4.0 * math.pi) * 0.35, 0.0)
    teb_truth[0, 1] = complex(0.08, 0.0)
    teb_truth[0, 2] = complex(-0.04, 0.02)
    if nalm > 5:
        teb_truth[1, 2] = complex(0.06, 0.02)
        teb_truth[1, 5] = complex(-0.03, 0.01)
    if nalm > 7:
        teb_truth[2, 4] = complex(0.04, -0.01)
        teb_truth[2, 7] = complex(-0.02, 0.01)

    temp_hooks = build_spherical_harmonic_solver_hooks(lmax=lmax)
    pol_hooks = build_spherical_spin_harmonic_solver_hooks(spin=2, lmax=lmax)
    temp_targets = (
        TargetRenderSpec.from_header(
            _shift_header_center(target_header, dra_deg=0.0, ddec_deg=0.0),
            band_label="temp",
        ),
        TargetRenderSpec.from_header(
            _shift_header_center(target_header, dra_deg=1.5, ddec_deg=-0.6),
            band_label="temp",
        ),
    )
    pol_targets = (
        TargetRenderSpec.from_header(
            _shift_header_center(target_header, dra_deg=0.0, ddec_deg=0.0),
            band_label="pol",
        ),
        TargetRenderSpec.from_header(
            _shift_header_center(target_header, dra_deg=-1.0, ddec_deg=0.8),
            band_label="pol",
        ),
    )
    temp_block = build_state_subspace_operator(
        teb_truth,
        temp_hooks.bind_targets(temp_targets),
        selector=0,
    )
    pol_block = build_state_subspace_operator(
        teb_truth,
        pol_hooks.bind_targets(pol_targets),
        selector=[1, 2],
    )
    prior_block = build_teb_cls_prior_operator(
        teb_truth,
        lmax=lmax,
        cls_tt=torch.tensor([10.0, 2.0, 0.8, 0.3], dtype=torch.float64)[: lmax + 1],
        cls_ee=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64)[: lmax + 1],
        cls_bb=torch.tensor([5.0, 1.2, 0.5, 0.2], dtype=torch.float64)[: lmax + 1],
        cls_te=torch.tensor([0.0, 0.4, 0.15, 0.05], dtype=torch.float64)[: lmax + 1],
        apply_noise=lambda tensor: tensor * 0.08,
    )
    operator_set = compose_bound_operators(temp_block, pol_block, prior_block)

    clean_temp = temp_block.apply(teb_truth)
    clean_pol = pol_block.apply(teb_truth)
    observed_temp = tuple(
        cutout + 0.01 * torch.randn_like(cutout) for cutout in clean_temp
    )
    observed_pol = tuple(maps + 0.005 * torch.randn_like(maps) for maps in clean_pol)
    data = (*observed_temp, *observed_pol, torch.zeros_like(teb_truth))
    zero_solution = torch.zeros_like(teb_truth)
    initial_outputs = operator_set.apply(zero_solution)
    initial_residual_norm = math.sqrt(
        sum(
            float(torch.sum(torch.abs(pred - obs) ** 2).item())
            for pred, obs in zip(initial_outputs, data, strict=True)
        )
    )

    def solve_fn():
        diagonal = operator_set.estimate_normal_operator_diagonal(
            data,
            initial_probe=torch.zeros_like(teb_truth),
            max_entries=64,
            damping=1.0e-4,
        )
        return operator_set.solve_normal_equation_cg(
            data,
            initial_solution=torch.zeros_like(teb_truth),
            iterations=10,
            damping=1.0e-4,
            preconditioner=make_diagonal_preconditioner(diagonal),
        )

    dt = _time_many(solve_fn, runs)
    fit = solve_fn()
    reconstructed_temp = temp_block.apply(fit.solution)
    reconstructed_pol = pol_block.apply(fit.solution)
    temp_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_temp, observed_temp, strict=True)
    ]
    pol_rmses = [
        math.sqrt(float(torch.mean(torch.square(pred - obs)).item()))
        for pred, obs in zip(reconstructed_pol, observed_pol, strict=True)
    ]
    prior_norm = float(torch.linalg.vector_norm(prior_block.apply(fit.solution)).item())
    final_residual_norm = (
        float(fit.residual_norm_history[-1])
        if fit.residual_norm_history
        else initial_residual_norm
    )
    return {
        "case": case,
        "library": "torchsky_spherical",
        "method": "spherical_inverse",
        "variant": "joint_teb_cl_prior",
        "geometry_family": "spherical_harmonic_joint_teb",
        "projection": case_spec.projection,
        "description": case_spec.description,
        "value_semantics": case_spec.value_semantics,
        "noise_model": case_spec.noise_model,
        "size": size,
        "n_obs": len(observations),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "time_ms": dt * 1000.0,
        "temp_observations": len(observed_temp),
        "pol_observations": len(observed_pol),
        "lmax": int(lmax),
        "nalm": int(nalm),
        "operator_bundle_geometry": temp_hooks.operators.geometry.name,
        "temp_rmse_mean": float(sum(temp_rmses) / max(len(temp_rmses), 1)),
        "temp_rmse_max": float(max(temp_rmses)),
        "pol_rmse_mean": float(sum(pol_rmses) / max(len(pol_rmses), 1)),
        "pol_rmse_max": float(max(pol_rmses)),
        "residual_norm_initial": float(initial_residual_norm),
        "residual_norm_final": float(final_residual_norm),
        "teb_cross_spectrum_prior_norm": prior_norm,
        "goal": "fit_shared_spherical_teb_state_from_temperature_and_polarized_cutouts",
    }


def _field_observations_from_simulated_inputs(
    observations: list[SimulatedObservation],
    *,
    wavelength_angstrom: float,
    max_observations: int,
) -> list[FieldObservation]:
    selected = observations[: max(int(max_observations), 1)]
    field_observations: list[FieldObservation] = []
    for item in selected:
        image = torchsky.read(
            item.observation.path,
            hdu=item.observation.hdu,
            device="cpu",
        ).to(dtype=torch.float32)
        weight = None
        if item.observation.weight_path is not None:
            weight = torchsky.read(
                item.observation.weight_path,
                device="cpu",
            ).to(dtype=torch.float32)
        mask = None
        if item.observation.mask_path is not None:
            mask = torchsky.read(
                item.observation.mask_path,
                device="cpu",
            ).to(dtype=torch.float32)
        field_observations.append(
            FieldObservation(
                image=image,
                header=dict(item.header),
                weight=weight,
                mask=mask,
                wavelength=float(wavelength_angstrom),
                transmission_center=float(wavelength_angstrom),
                band_name="mono",
            )
        )
    return field_observations


def _build_product_benchmark_row(
    *,
    case: str,
    library: str,
    method: str,
    variant: str,
    geometry_family: str,
    case_spec: _CaseSpec,
    observations: list[SimulatedObservation],
    target_header: dict[str, Any],
    truth: torch.Tensor,
    region_masks: dict[str, torch.Tensor],
    result,
    size: int,
    time_ms: float,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    return {
        "case": case,
        "library": library,
        "method": method,
        "variant": variant,
        "geometry_family": geometry_family,
        "projection": case_spec.projection,
        "description": case_spec.description,
        "value_semantics": case_spec.value_semantics,
        "noise_model": case_spec.noise_model,
        "size": size,
        "n_obs": len(observations),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(item.header["NAXIS1"]) for item in observations),
        "max_input_width": max(int(item.header["NAXIS1"]) for item in observations),
        "min_input_height": min(int(item.header["NAXIS2"]) for item in observations),
        "max_input_height": max(int(item.header["NAXIS2"]) for item in observations),
        "goal": "reconstruct_union_field_on_target_grid",
        "product_geometry": (
            "rectangular"
            if int(target_header["NAXIS1"]) != int(target_header["NAXIS2"])
            else "square"
        ),
        "n_warped_inputs": _count_warped_inputs(observations),
        "n_weight_maps": sum(
            1 for item in observations if item.observation.weight_path is not None
        ),
        "n_mask_maps": sum(
            1 for item in observations if item.observation.mask_path is not None
        ),
        "mean_zero_weight_fraction_input": float(
            sum(item.zero_weight_fraction for item in observations)
            / max(len(observations), 1)
        ),
        "mean_explicit_mask_fraction_input": float(
            sum(item.explicit_mask_fraction for item in observations)
            / max(len(observations), 1)
        ),
        "time_ms": time_ms,
        "mpix_s": (float(result.data.numel()) / max(time_ms / 1000.0, 1e-12)) / 1e6,
        **_truth_metrics(
            result.data.squeeze(0),
            result.weight.squeeze(0),
            truth,
            target_header=target_header,
            value_semantics=case_spec.value_semantics,
            region_masks=region_masks,
        ),
        "mean_noise_sigma_flux": float(
            sum(item.noise_sigma_flux for item in observations)
            / max(len(observations), 1)
        ),
        "mean_exposure_time": float(
            sum(item.exposure_time for item in observations) / max(len(observations), 1)
        ),
        "mean_transmission": float(
            sum(item.transmission for item in observations) / max(len(observations), 1)
        ),
        "mean_psf_sigma_pix": float(
            sum(item.psf_sigma_pix for item in observations) / max(len(observations), 1)
        ),
        **({} if extra is None else extra),
    }


def evaluate_external_methods(
    tmpdir: Path,
    *,
    case: str,
    size: int,
    seed: int,
    runs: int,
    width: int | None = None,
    height: int | None = None,
) -> list[dict[str, Any]]:
    observations, target_header, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        seed=seed,
        width=width,
        height=height,
    )
    case_spec = _case_spec(case, size=size)
    region_masks = _target_mask_region_masks(observations, target_header=target_header)

    methods: list[tuple[str, str | None]] = []
    methods.extend(
        ("reproject_interp", variant) for _, variant in map_compat._REPROJECT_ORDERS
    )
    methods.extend(
        ("dfreproject", variant) for _, variant in map_compat._DFREPROJECT_ORDERS
    )
    methods.extend(("drizzle", variant) for variant in map_compat._DRIZZLE_KERNELS)
    methods.append(("montage", None))
    methods.extend(
        ("swarp", variant.lower()) for variant in map_compat._SWARP_RESAMPLING_TYPES
    )
    methods.append(("casutools", None))

    baseline_row = {
        "library": "torchsky",
        "variant": "bilinear",
        "time_ms": 1.0,
    }
    baseline_data = truth
    baseline_weight = torch.ones_like(truth, dtype=torch.float32)
    rows: list[dict[str, Any]] = []
    for library, variant in methods:
        row, data, weight = map_compat._comparator_row(
            library,
            case,
            [item.observation for item in observations],
            target_header=target_header,
            runs=runs,
            torchsky_row=baseline_row,
            torchsky_data=baseline_data,
            torchsky_weight=baseline_weight,
            variant=variant,
        )
        for key in list(row):
            if key.endswith("_vs_torchsky") or key.startswith(
                "speed_ratio_torchsky_vs_"
            ):
                row.pop(key)
        row.update(
            {
                "method": row.get("method", library),
                "variant": row.get("variant", variant or "default"),
                "projection": case_spec.projection,
                "description": case_spec.description,
                "value_semantics": case_spec.value_semantics,
                "noise_model": case_spec.noise_model,
                "size": size,
                "n_obs": len(observations),
                "target_width": int(target_header["NAXIS1"]),
                "target_height": int(target_header["NAXIS2"]),
                "min_input_width": min(
                    int(item.header["NAXIS1"]) for item in observations
                ),
                "max_input_width": max(
                    int(item.header["NAXIS1"]) for item in observations
                ),
                "min_input_height": min(
                    int(item.header["NAXIS2"]) for item in observations
                ),
                "max_input_height": max(
                    int(item.header["NAXIS2"]) for item in observations
                ),
                "mean_noise_sigma_flux": float(
                    sum(item.noise_sigma_flux for item in observations)
                    / max(len(observations), 1)
                ),
                "mean_exposure_time": float(
                    sum(item.exposure_time for item in observations)
                    / max(len(observations), 1)
                ),
                "mean_transmission": float(
                    sum(item.transmission for item in observations)
                    / max(len(observations), 1)
                ),
                "mean_psf_sigma_pix": float(
                    sum(item.psf_sigma_pix for item in observations)
                    / max(len(observations), 1)
                ),
                "goal": "reconstruct_union_field_on_target_grid",
                "product_geometry": (
                    "rectangular"
                    if int(target_header["NAXIS1"]) != int(target_header["NAXIS2"])
                    else "square"
                ),
                "n_warped_inputs": _count_warped_inputs(observations),
                "n_weight_maps": sum(
                    1
                    for item in observations
                    if item.observation.weight_path is not None
                ),
                "n_mask_maps": sum(
                    1 for item in observations if item.observation.mask_path is not None
                ),
                "mean_zero_weight_fraction_input": float(
                    sum(item.zero_weight_fraction for item in observations)
                    / max(len(observations), 1)
                ),
                "mean_explicit_mask_fraction_input": float(
                    sum(item.explicit_mask_fraction for item in observations)
                    / max(len(observations), 1)
                ),
            }
        )
        if row.get("status") == "OK" and data is not None and weight is not None:
            row.update(
                _truth_metrics(
                    data.to(dtype=torch.float32),
                    weight.to(dtype=torch.float32),
                    truth,
                    target_header=target_header,
                    value_semantics=case_spec.value_semantics,
                    region_masks=region_masks,
                )
            )
        rows.append(row)
    return rows


def _kernel_recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    torch_rows = [
        row
        for row in rows
        if row.get("library") == "torchsky" and row.get("status", "OK") == "OK"
    ]
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in torch_rows:
        by_case.setdefault(str(row["case"]), []).append(row)

    recommendation_rows: list[dict[str, Any]] = []
    for case, case_rows in by_case.items():
        best_rmse_row = min(
            case_rows,
            key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
        )
        rmse_limit = float(best_rmse_row["reconstruction_rmse"]) * 1.01
        near_best_rows = [
            row for row in case_rows if float(row["reconstruction_rmse"]) <= rmse_limit
        ]
        fastest_near_best = min(near_best_rows, key=lambda row: row["time_ms"])
        interpolatory_rows = [
            row for row in case_rows if row.get("variant") in _INTERPOLATORY_KERNELS
        ]
        best_interpolatory = min(
            interpolatory_rows,
            key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
        )
        smooth_rows = [
            row
            for row in case_rows
            if row.get("variant") in _SMOOTH_RECONSTRUCTION_KERNELS
        ]

        def make_row(profile: str, basis: dict[str, Any], note: str) -> dict[str, Any]:
            return {
                "case": case,
                "library": "torchsky_recommendation",
                "method": "torchsky_recommendation",
                "variant": str(basis["variant"]),
                "profile": profile,
                "recommended_variant": str(basis["variant"]),
                "recommended_time_ms": float(basis["time_ms"]),
                "recommended_reconstruction_rmse": float(basis["reconstruction_rmse"]),
                "recommended_signal_to_residual_db": float(
                    basis["signal_to_residual_db"]
                ),
                "coverage_fraction": float(basis["coverage_fraction"]),
                "value_semantics": basis.get("value_semantics"),
                "projection": basis.get("projection"),
                "noise_model": basis.get("noise_model"),
                "note": note,
                "status": "INFO",
            }

        recommendation_rows.append(
            make_row(
                "default_general",
                fastest_near_best,
                "Fastest torchsky kernel within 1% of the best RMSE on this case",
            )
        )
        recommendation_rows.append(
            make_row(
                "high_accuracy_interpolatory",
                best_interpolatory,
                "Lowest-RMSE interpolatory torchsky kernel on this case",
            )
        )
        if smooth_rows:
            best_smooth = min(
                smooth_rows,
                key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
            )
            recommendation_rows.append(
                make_row(
                    "smooth_reconstruction",
                    best_smooth,
                    "Best smooth, non-cardinal reconstruction kernel on this case",
                )
            )
    return recommendation_rows


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--cases",
        default=",".join(DEFAULT_CASES),
        help="Comma-separated case list",
    )
    parser.add_argument(
        "--torchsky-kernels",
        default=",".join(DEFAULT_TORCHSKY_KERNELS),
        help="Comma-separated torchsky resampling kernels",
    )
    parser.add_argument(
        "--include-external",
        action="store_true",
        help="Also score available external mapmaking backends against the synthetic truth",
    )
    parser.add_argument(
        "--include-spherical-inverse",
        action="store_true",
        help="Also score the native joint spherical T/E/B inverse benchmark row",
    )
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument("--baseline-json", type=Path, default=None)
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_map_truth_") as tmp:
        tmpdir = Path(tmp)
        kernels = [
            part.strip() for part in args.torchsky_kernels.split(",") if part.strip()
        ]
        for case in [part.strip() for part in args.cases.split(",") if part.strip()]:
            for kernel in kernels:
                rows.append(
                    evaluate_torchsky_case(
                        tmpdir,
                        case=case,
                        size=args.size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=args.runs,
                        resampling_kernel=kernel,
                    )
                )
            if args.include_external:
                rows.extend(
                    evaluate_external_methods(
                        tmpdir,
                        case=case,
                        size=args.size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=max(2, args.runs),
                    )
                )
            if args.include_spherical_inverse:
                rows.append(
                    evaluate_torchsky_joint_teb_case(
                        tmpdir,
                        case=case,
                        size=args.size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=args.runs,
                    )
                )
    rows.extend(_kernel_recommendation_rows(rows))

    markdown_path = args.markdown_out
    if markdown_path is None and args.json_out is not None:
        markdown_path = args.json_out.with_suffix(".md")
    markdown = write_map_benchmark_artifacts(
        rows,
        json_out=args.json_out,
        markdown_out=markdown_path,
    )
    print(markdown)
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    if markdown_path is not None:
        print(f"Markdown: {markdown_path}")
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errors = check_map_benchmark_regressions(rows, baseline_rows)
        if errors:
            raise SystemExit("\n".join(errors))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
