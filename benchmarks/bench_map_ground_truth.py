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

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))
BENCHMARKS = ROOT / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

import torchsky  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.map import (  # noqa: E402
    check_map_benchmark_regressions,
    ImageObservation,
    evaluate_map_product,
    plan_map_product,
    stack_map_product,
    write_map_benchmark_artifacts,
)
import torchsky.map.image_plane as image_plane_mod  # noqa: E402
import bench_map_compat as map_compat  # noqa: E402

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
        if "TPV" in str(item.header.get("CTYPE1", "")) or "TPV" in str(item.header.get("CTYPE2", ""))
    )


def _rotated_cd(scale_deg: float, rotation_deg: float) -> tuple[float, float, float, float]:
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
    specs: list[tuple[dict[str, Any], float, float, float, tuple[slice, slice] | None]] = []
    for idx, (offset_ra, offset_dec) in enumerate(center_offsets):
        width = int(torch.randint(500, 4001, (1,), generator=gen).item())
        height = int(torch.randint(500, 4001, (1,), generator=gen).item())
        scale_deg = float(torch.empty(1).uniform_(0.0030, 0.0068, generator=gen).item())
        rotation_deg = float(torch.empty(1).uniform_(-24.0, 24.0, generator=gen).item())
        crval1 = 12.0 + float(offset_ra) + float(torch.empty(1).uniform_(-0.35, 0.35, generator=gen).item())
        crval2 = -3.0 + float(offset_dec) + float(torch.empty(1).uniform_(-0.25, 0.25, generator=gen).item())
        exposure_time = float(torch.empty(1).uniform_(45.0, 260.0, generator=gen).item())
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
                slice(max(1, width // 3), max(2, min(width, width // 3 + max(8, width // 40)))),
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


def _analytic_sky(ra_deg: torch.Tensor, dec_deg: torch.Tensor, *, center_ra_deg: float, center_dec_deg: float) -> torch.Tensor:
    dra = (ra_deg - float(center_ra_deg)) * math.cos(math.radians(float(center_dec_deg)))
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
            raise ValueError(
                "flux_area_norm_sr is required for flux_area_scaled noise"
            )
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
                    _tan_header(size, crval1=12.000, crval2=-3.000, scale_deg=0.0030, rotation_deg=0.0),
                    55.0,
                    0.90,
                    0.45,
                    None,
                ),
                (
                    _tan_header(size, crval1=12.028, crval2=-3.024, scale_deg=0.0068, rotation_deg=18.0),
                    110.0,
                    0.82,
                    1.15,
                    (slice(None), slice(size // 2, size // 2 + max(2, size // 14))),
                ),
                (
                    _tan_header(size, crval1=11.968, crval2=-2.972, scale_deg=0.0026, rotation_deg=-21.0),
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
                    _tan_header(size, crval1=12.0, crval2=-3.0, scale_deg=0.0042, rotation_deg=0.0),
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
                    _tan_header(size, crval1=12.0, crval2=-3.0, scale_deg=0.0036, rotation_deg=0.0),
                    70.0,
                    0.87,
                    0.6,
                    None,
                ),
                (
                    _tpv_header(size, crval1=12.016, crval2=-3.024, scale_deg=0.0059, rotation_deg=15.0),
                    105.0,
                    0.89,
                    1.0,
                    None,
                ),
                (
                    _tpv_header(size, crval1=11.982, crval2=-2.972, scale_deg=0.0029, rotation_deg=-17.0),
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
    elif case == "tan_random10_large_union_rect_weighted_masked":
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
) -> dict[str, float]:
    metrics = evaluate_map_product(
        data,
        truth,
        weight=weight,
        header=target_header,
        value_semantics=value_semantics,
    )
    mask = weight > 0.0
    signal = truth[mask]
    signal_rms = (
        float(torch.sqrt(torch.mean(signal**2)).item()) if bool(mask.any()) else 0.0
    )
    metrics.update(
        {
            "signal_rms_truth": signal_rms,
            "residual_rms": float(metrics["reconstruction_rmse"]),
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
    return metrics


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
    return {
        "case": case,
        "library": "torchsky",
        "method": "coadd",
        "variant": resampling_kernel,
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
        "interpolation_kernel": resampling_kernel,
        "resampling_method": "resample_weighted_numerator_and_denominator",
        "stacking_method": "weighted_mean",
        "goal": "reconstruct_union_field_on_target_grid",
        "product_geometry": (
            "rectangular"
            if int(target_header["NAXIS1"]) != int(target_header["NAXIS2"])
            else "square"
        ),
        "n_warped_inputs": _count_warped_inputs(observations),
        "n_weight_maps": sum(1 for item in observations if item.observation.weight_path is not None),
        "n_mask_maps": sum(1 for item in observations if item.observation.mask_path is not None),
        "mean_zero_weight_fraction_input": float(
            sum(item.zero_weight_fraction for item in observations) / max(len(observations), 1)
        ),
        "mean_explicit_mask_fraction_input": float(
            sum(item.explicit_mask_fraction for item in observations) / max(len(observations), 1)
        ),
        "time_ms": dt * 1000.0,
        "mpix_s": (float(result.data.numel()) / dt) / 1e6,
        **_truth_metrics(
            result.data.squeeze(0),
            result.weight.squeeze(0),
            truth,
            target_header=target_header,
            value_semantics=case_spec.value_semantics,
        ),
        "mean_noise_sigma_flux": float(
            sum(item.noise_sigma_flux for item in observations) / max(len(observations), 1)
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

    methods: list[tuple[str, str | None]] = []
    methods.extend(("reproject_interp", variant) for _, variant in map_compat._REPROJECT_ORDERS)
    methods.extend(("dfreproject", variant) for _, variant in map_compat._DFREPROJECT_ORDERS)
    methods.extend(("drizzle", variant) for variant in map_compat._DRIZZLE_KERNELS)
    methods.append(("montage", None))
    methods.extend(("swarp", variant.lower()) for variant in map_compat._SWARP_RESAMPLING_TYPES)
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
            if key.endswith("_vs_torchsky") or key.startswith("speed_ratio_torchsky_vs_"):
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
                "min_input_width": min(int(item.header["NAXIS1"]) for item in observations),
                "max_input_width": max(int(item.header["NAXIS1"]) for item in observations),
                "min_input_height": min(int(item.header["NAXIS2"]) for item in observations),
                "max_input_height": max(int(item.header["NAXIS2"]) for item in observations),
                "mean_noise_sigma_flux": float(
                    sum(item.noise_sigma_flux for item in observations) / max(len(observations), 1)
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
                    sum(item.zero_weight_fraction for item in observations) / max(len(observations), 1)
                ),
                "mean_explicit_mask_fraction_input": float(
                    sum(item.explicit_mask_fraction for item in observations) / max(len(observations), 1)
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
                "recommended_reconstruction_rmse": float(
                    basis["reconstruction_rmse"]
                ),
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
        kernels = [part.strip() for part in args.torchsky_kernels.split(",") if part.strip()]
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
