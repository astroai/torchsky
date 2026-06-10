#!/usr/bin/env python3
"""Benchmark multiwavelength latent-field mapmaking against known cube truth."""

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
import bench_map_compat as map_compat  # noqa: E402
from torchsky import WCS  # noqa: E402
from torchsky.sphere import healpix as hp  # noqa: E402
from torchsky.map import (  # noqa: E402
    FieldObservation,
    ImageObservation,
    MapBandpassSpec,
    MapProductSpec,
    check_map_benchmark_regressions,
    estimate_field_domain,
    evaluate_map_product,
    fit_field_model,
    make_field_model,
    plan_map_product,
    stack_map_product,
    write_map_benchmark_artifacts,
)
from torchsky.map.product import render_scene_product  # noqa: E402
from torchsky.map.operators.background import fit_background_plane  # noqa: E402
from torchsky.map.operators import spherical_spectral_operator_bundle  # noqa: E402
from torchsky.map.scenes import HealpixMapScene  # noqa: E402
from torchsky.map.solvers import (  # noqa: E402
    build_solver_hooks,
    build_spherical_harmonic_solver_hooks,
    build_state_kronecker_operator,
    build_state_subspace_operator,
    compose_bound_operators,
    harmonic_cl_diagonal,
    make_diagonal_preconditioner,
)
from torchsky.map.target import TargetRenderSpec  # noqa: E402
from torchsky.sphere.spectral import map2alm  # noqa: E402
from torchsky.spectral import SpectralAxis  # noqa: E402

try:  # noqa: E402
    import torchsky.cpp  # noqa: F401
except Exception as exc:  # pragma: no cover - native bench env requirement
    raise RuntimeError(
        "bench_map_multifield_truth.py requires torchsky.cpp; "
        "run it from a native-extension environment"
    ) from exc


DEFAULT_CASES = (
    "tan_multifield_cube",
    "tan_multifield_cube_inr",
    "tpv_multifield_cube",
    "tan_multifield_rectangular_scale",
    "tpv_multifield_rectangular_scale",
    "mixed_multifield_random10",
)
DEFAULT_TORCHSKY_KERNELS = (
    "bilinear",
    "bicubic",
    "catmull_rom",
    "lanczos3",
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
    wavelength: float


@dataclass(frozen=True)
class ObservationSpec:
    header: dict[str, Any]
    exposure_time: float
    transmission: float
    psf_sigma_pix: float
    spectral_min: float
    spectral_max: float
    band_name: str
    transmission_center: float | None = None
    transmission_sigma: float | None = None
    gap: tuple[slice, slice] | None = None


@dataclass(frozen=True)
class CaseSpec:
    case: str
    projection: str
    description: str
    field_model: str
    base_noise_sb: float
    target_header: dict[str, Any]
    target_centers: tuple[float, ...]
    target_edges: tuple[float, ...]
    observations: tuple[ObservationSpec, ...]


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


def _tpv_header(
    width: int,
    height: int,
    *,
    crval1: float,
    crval2: float,
    scale_deg: float,
    rotation_deg: float = 0.0,
    pv_scale: float = 1.0,
) -> dict[str, Any]:
    header = _tan_header(
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


def _truth_cube_to_healpix_maps(
    truth: torch.Tensor,
    *,
    target_header: dict[str, Any],
    nside: int,
) -> torch.Tensor:
    npix = hp.nside2npix(int(nside))
    ipix = torch.arange(npix, dtype=torch.int64)
    lon, lat = hp.pix2ang(int(nside), ipix, nest=False, lonlat=True)
    maps = [
        _sample_image_on_lonlat_grid(
            truth[idx],
            source_header=target_header,
            lon_deg=lon,
            lat_deg=lat,
        ).reshape(-1)
        for idx in range(int(truth.shape[0]))
    ]
    return torch.stack(maps, dim=0)


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


@dataclass(frozen=True)
class _FrozenImplicitField:
    w1: torch.Tensor
    b1: torch.Tensor
    w2: torch.Tensor
    b2: torch.Tensor
    w3: torch.Tensor
    b3: torch.Tensor

    def __call__(
        self,
        ra_deg: torch.Tensor,
        dec_deg: torch.Tensor,
        wavelength_angstrom: float,
        *,
        center_ra_deg: float,
        center_dec_deg: float,
    ) -> torch.Tensor:
        dra = (ra_deg - float(center_ra_deg)) * math.cos(
            math.radians(float(center_dec_deg))
        )
        ddec = dec_deg - float(center_dec_deg)
        lam = torch.full_like(dra, (float(wavelength_angstrom) - 6000.0) / 1400.0)
        coords = torch.stack([8.0 * dra, 8.0 * ddec, lam], dim=-1)
        h1 = torch.sin(torch.matmul(coords, self.w1.T) + self.b1)
        h2 = torch.sin(torch.matmul(h1, self.w2.T) + self.b2)
        out = torch.matmul(h2, self.w3.T).squeeze(-1) + self.b3.squeeze(-1)
        return torch.nn.functional.softplus(out) + 0.05


def _make_field_model(field_model: str, *, seed: int):
    if field_model == "analytic_basis":
        return _latent_field
    if field_model == "implicit_siren":
        gen = torch.Generator(device="cpu")
        gen.manual_seed(int(seed))
        return _FrozenImplicitField(
            w1=0.6 * torch.randn((32, 3), generator=gen, dtype=torch.float32),
            b1=0.2 * torch.randn((32,), generator=gen, dtype=torch.float32),
            w2=0.35 * torch.randn((32, 32), generator=gen, dtype=torch.float32),
            b2=0.2 * torch.randn((32,), generator=gen, dtype=torch.float32),
            w3=0.15 * torch.randn((1, 32), generator=gen, dtype=torch.float32),
            b3=0.05 * torch.randn((1,), generator=gen, dtype=torch.float32),
        )
    raise ValueError("field_model must be one of {'analytic_basis', 'implicit_siren'}")


def _transmission_weights(
    samples: torch.Tensor,
    *,
    spectral_min: float,
    spectral_max: float,
    transmission_center: float | None,
    transmission_sigma: float | None,
) -> torch.Tensor:
    center = (
        float(transmission_center)
        if transmission_center is not None
        else 0.5 * (float(spectral_min) + float(spectral_max))
    )
    sigma = (
        float(transmission_sigma)
        if transmission_sigma is not None
        else max(float(spectral_max) - float(spectral_min), 1.0) / 4.0
    )
    weights = torch.exp(-0.5 * torch.square((samples - center) / sigma))
    return weights / torch.clamp(weights.sum(), min=1e-8)


def _latent_field(
    ra_deg: torch.Tensor,
    dec_deg: torch.Tensor,
    wavelength_angstrom: float,
    *,
    center_ra_deg: float,
    center_dec_deg: float,
) -> torch.Tensor:
    lam = (float(wavelength_angstrom) - 6000.0) / 1400.0
    dra = (ra_deg - float(center_ra_deg)) * math.cos(
        math.radians(float(center_dec_deg))
    )
    ddec = dec_deg - float(center_dec_deg)
    g1 = (1.1 + 0.40 * lam) * torch.exp(
        -((dra + 0.08) ** 2 + (ddec - 0.03) ** 2) / 0.0016
    )
    g2 = (0.7 - 0.25 * lam) * torch.exp(
        -((dra - 0.05) ** 2 + (ddec + 0.06) ** 2) / 0.0009
    )
    ridge = (
        0.18 * torch.sin(40.0 * dra + 0.35 * lam) * torch.cos(36.0 * ddec - 0.2 * lam)
    )
    diffuse = (
        0.25 + 0.6 * (dra + 0.15) + 0.25 * (ddec - 0.02) + 0.12 * lam + 0.08 * lam * dra
    )
    line = (
        0.15 + 0.10 * torch.cos(torch.tensor(lam, dtype=torch.float32))
    ) * torch.exp(
        -((ddec + 0.10 * torch.sin(torch.tensor(lam, dtype=torch.float32))) ** 2)
        / 0.0030
    )
    return (diffuse + g1 + g2 + ridge + line).to(dtype=torch.float32)


def _band_samples(
    spectral_min: float, spectral_max: float, *, n: int = 5
) -> torch.Tensor:
    if math.isclose(spectral_min, spectral_max):
        return torch.tensor([spectral_min], dtype=torch.float32)
    return torch.linspace(
        float(spectral_min), float(spectral_max), n, dtype=torch.float32
    )


def _resolve_dims(
    *,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
) -> tuple[int, int]:
    if size is not None:
        if width is not None or height is not None:
            raise ValueError("size cannot be combined with width/height")
        return int(size), int(size)
    if width is None or height is None:
        raise ValueError("either size or both width and height are required")
    return int(width), int(height)


def _scaled_dims(
    target_width: int,
    target_height: int,
    *,
    width_scale: float,
    height_scale: float,
) -> tuple[int, int]:
    return (
        max(16, int(round(float(target_width) * float(width_scale)))),
        max(16, int(round(float(target_height) * float(height_scale)))),
    )


def _random_large_rect_observations(
    target_width: int,
    target_height: int,
    *,
    seed: int,
) -> tuple[ObservationSpec, ...]:
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed))
    band_defs = (
        ("blue_0", 4300.0, 5050.0, 4680.0, 130.0),
        ("green_1", 4920.0, 5750.0, 5340.0, 155.0),
        ("orange_2", 5600.0, 6500.0, 6060.0, 180.0),
        ("red_3", 6400.0, 7400.0, 6910.0, 210.0),
    )
    center_offsets = (
        (0.0, 0.0),
        (1.6, -0.5),
        (-2.2, 1.1),
        (3.8, -1.4),
        (-4.4, 1.7),
        (6.9, -0.6),
        (-7.3, 0.8),
        (0.9, 4.4),
        (-0.7, -4.2),
        (10.8, 4.9),
    )
    specs: list[ObservationSpec] = []
    for idx in range(10):
        width = int(torch.randint(500, 4001, (1,), generator=gen).item())
        height = int(torch.randint(500, 4001, (1,), generator=gen).item())
        scale_deg = float(torch.empty(1).uniform_(0.0032, 0.0068, generator=gen).item())
        rotation_deg = float(torch.empty(1).uniform_(-24.0, 24.0, generator=gen).item())
        jitter_ra = float(torch.empty(1).uniform_(-0.35, 0.35, generator=gen).item())
        jitter_dec = float(torch.empty(1).uniform_(-0.25, 0.25, generator=gen).item())
        base_offset_ra, base_offset_dec = center_offsets[idx]
        crval1 = 12.0 + float(base_offset_ra) + jitter_ra
        crval2 = -3.0 + float(base_offset_dec) + jitter_dec
        exposure_time = float(
            torch.empty(1).uniform_(45.0, 260.0, generator=gen).item()
        )
        transmission = float(torch.empty(1).uniform_(0.68, 1.08, generator=gen).item())
        psf_sigma_pix = float(torch.empty(1).uniform_(0.45, 1.55, generator=gen).item())
        (
            band_name,
            spectral_min,
            spectral_max,
            transmission_center,
            transmission_sigma,
        ) = band_defs[idx % len(band_defs)]
        if idx % 3 == 0:
            header = _tpv_header(
                width,
                height,
                crval1=crval1,
                crval2=crval2,
                scale_deg=scale_deg,
                rotation_deg=rotation_deg,
                pv_scale=float(
                    torch.empty(1).uniform_(0.8, 1.45, generator=gen).item()
                ),
            )
        else:
            header = _tan_header(
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
            ObservationSpec(
                header,
                exposure_time,
                transmission,
                psf_sigma_pix,
                spectral_min,
                spectral_max,
                band_name,
                transmission_center,
                transmission_sigma,
                gap,
            )
        )
    return tuple(specs)


def _integrated_field_image(
    header: dict[str, Any],
    *,
    spectral_min: float,
    spectral_max: float,
    field_model,
    transmission_center: float | None = None,
    transmission_sigma: float | None = None,
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
    samples = _band_samples(spectral_min, spectral_max)
    sample_weights = _transmission_weights(
        samples,
        spectral_min=spectral_min,
        spectral_max=spectral_max,
        transmission_center=transmission_center,
        transmission_sigma=transmission_sigma,
    )
    planes = [
        field_model(
            ra_deg.to(dtype=torch.float32),
            dec_deg.to(dtype=torch.float32),
            float(wavelength.item()),
            center_ra_deg=float(header["CRVAL1"]),
            center_dec_deg=float(header["CRVAL2"]),
        )
        for wavelength in samples
    ]
    return torch.sum(
        torch.stack(planes, dim=0)
        * sample_weights.view(-1, 1, 1).to(dtype=torch.float32),
        dim=0,
    )


def _simulate_observation(
    tmpdir: Path,
    *,
    name: str,
    spec: ObservationSpec,
    base_noise_sb: float,
    seed: int,
    field_model,
) -> SimulatedObservation:
    signal = _integrated_field_image(
        spec.header,
        spectral_min=spec.spectral_min,
        spectral_max=spec.spectral_max,
        field_model=field_model,
        transmission_center=spec.transmission_center,
        transmission_sigma=spec.transmission_sigma,
    )
    signal = _gaussian_blur(signal, spec.psf_sigma_pix)
    scale = float(spec.exposure_time) * float(spec.transmission)
    noise_sigma = float(base_noise_sb) / math.sqrt(max(scale, 1e-8))
    gen = torch.Generator(device=signal.device)
    gen.manual_seed(int(seed))
    noisy = signal + noise_sigma * torch.randn(
        signal.shape, generator=gen, dtype=torch.float32
    )
    weight = torch.full_like(noisy, 1.0 / max(noise_sigma**2, 1e-8))
    if spec.gap is not None:
        noisy[spec.gap] = 0.0
        weight[spec.gap] = 0.0

    image_path = tmpdir / f"{name}.fits"
    weight_path = tmpdir / f"{name}_wht.fits"
    torchsky.write(str(image_path), noisy, header=spec.header, overwrite=True)
    torchsky.write(str(weight_path), weight, header=spec.header, overwrite=True)
    center = 0.5 * (spec.spectral_min + spec.spectral_max)
    return SimulatedObservation(
        observation=ImageObservation(
            str(image_path),
            weight_path=str(weight_path),
            wavelength=center,
            spectral_min=spec.spectral_min,
            spectral_max=spec.spectral_max,
            transmission_center=spec.transmission_center,
            transmission_sigma=spec.transmission_sigma,
            band_name=spec.band_name,
        ),
        header=spec.header,
        exposure_time=float(spec.exposure_time),
        transmission=float(spec.transmission),
        psf_sigma_pix=float(spec.psf_sigma_pix),
        wavelength=float(center),
    )


def _case_spec(
    case: str,
    *,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int = 1234,
) -> CaseSpec:
    target_width, target_height = _resolve_dims(size=size, width=width, height=height)
    target_header = _tan_header(
        target_width,
        target_height,
        crval1=12.0,
        crval2=-3.0,
        scale_deg=0.0045,
        rotation_deg=0.0,
    )
    target_edges = (4300.0, 5000.0, 5700.0, 6500.0, 7400.0)
    target_centers = tuple(
        0.5 * (target_edges[idx] + target_edges[idx + 1])
        for idx in range(len(target_edges) - 1)
    )
    cases = {
        "tan_multifield_cube": CaseSpec(
            case="tan_multifield_cube",
            projection="TAN",
            description="Multiwavelength TAN latent-field cube with mixed resolution and bandpass overlap",
            field_model="analytic_basis",
            base_noise_sb=0.03,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=(
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0045,
                    ),
                    80.0,
                    0.92,
                    0.55,
                    4300.0,
                    5200.0,
                    "blue_wide",
                    4700.0,
                    170.0,
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.018,
                        crval2=-3.010,
                        scale_deg=0.0052,
                        rotation_deg=6.0,
                    ),
                    120.0,
                    0.88,
                    0.85,
                    5000.0,
                    6100.0,
                    "green_wide",
                    5520.0,
                    210.0,
                    (
                        slice(None),
                        slice(
                            target_width // 3,
                            target_width // 3 + max(2, target_width // 18),
                        ),
                    ),
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=11.986,
                        crval2=-2.984,
                        scale_deg=0.0038,
                        rotation_deg=-8.0,
                    ),
                    160.0,
                    0.83,
                    1.10,
                    5900.0,
                    7400.0,
                    "red_wide",
                    6750.0,
                    260.0,
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.004,
                        crval2=-2.996,
                        scale_deg=0.0041,
                        rotation_deg=2.0,
                    ),
                    90.0,
                    0.95,
                    0.65,
                    5400.0,
                    6600.0,
                    "mid_overlap",
                    6020.0,
                    190.0,
                ),
            ),
        ),
        "tan_multifield_cube_inr": CaseSpec(
            case="tan_multifield_cube_inr",
            projection="TAN",
            description="Multiwavelength TAN latent-field cube driven by a frozen implicit neural field",
            field_model="implicit_siren",
            base_noise_sb=0.03,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=(
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0045,
                    ),
                    80.0,
                    0.92,
                    0.55,
                    4300.0,
                    5200.0,
                    "blue_wide",
                    4700.0,
                    170.0,
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.018,
                        crval2=-3.010,
                        scale_deg=0.0052,
                        rotation_deg=6.0,
                    ),
                    120.0,
                    0.88,
                    0.85,
                    5000.0,
                    6100.0,
                    "green_wide",
                    5520.0,
                    210.0,
                    (
                        slice(None),
                        slice(
                            target_width // 3,
                            target_width // 3 + max(2, target_width // 18),
                        ),
                    ),
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=11.986,
                        crval2=-2.984,
                        scale_deg=0.0038,
                        rotation_deg=-8.0,
                    ),
                    160.0,
                    0.83,
                    1.10,
                    5900.0,
                    7400.0,
                    "red_wide",
                    6750.0,
                    260.0,
                ),
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.004,
                        crval2=-2.996,
                        scale_deg=0.0041,
                        rotation_deg=2.0,
                    ),
                    90.0,
                    0.95,
                    0.65,
                    5400.0,
                    6600.0,
                    "mid_overlap",
                    6020.0,
                    190.0,
                ),
            ),
        ),
        "tpv_multifield_cube": CaseSpec(
            case="tpv_multifield_cube",
            projection="TPV",
            description="Multiwavelength TPV latent-field cube with distortion, low overlap, and mixed PSFs",
            field_model="analytic_basis",
            base_noise_sb=0.035,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=(
                ObservationSpec(
                    _tan_header(
                        target_width,
                        target_height,
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0043,
                    ),
                    70.0,
                    0.90,
                    0.6,
                    4300.0,
                    5200.0,
                    "blue_tan",
                    4680.0,
                    150.0,
                ),
                ObservationSpec(
                    _tpv_header(
                        target_width,
                        target_height,
                        crval1=12.012,
                        crval2=-3.018,
                        scale_deg=0.0051,
                        rotation_deg=8.0,
                        pv_scale=0.18,
                    ),
                    110.0,
                    0.89,
                    0.95,
                    5000.0,
                    6100.0,
                    "green_tpv",
                    5560.0,
                    180.0,
                ),
                ObservationSpec(
                    _tpv_header(
                        target_width,
                        target_height,
                        crval1=11.988,
                        crval2=-2.982,
                        scale_deg=0.0039,
                        rotation_deg=-10.0,
                        pv_scale=0.12,
                    ),
                    150.0,
                    0.78,
                    1.20,
                    6000.0,
                    7400.0,
                    "red_tpv",
                    6830.0,
                    230.0,
                    (
                        slice(
                            target_height // 3,
                            target_height // 3 + max(2, target_height // 20),
                        ),
                        slice(None),
                    ),
                ),
                ObservationSpec(
                    _tpv_header(
                        target_width,
                        target_height,
                        crval1=12.020,
                        crval2=-3.030,
                        scale_deg=0.0048,
                        rotation_deg=14.0,
                        pv_scale=0.22,
                    ),
                    95.0,
                    0.84,
                    1.05,
                    5400.0,
                    6800.0,
                    "mid_tpv_overlap",
                    6100.0,
                    220.0,
                ),
            ),
        ),
        "tan_multifield_rectangular_scale": CaseSpec(
            case="tan_multifield_rectangular_scale",
            projection="TAN",
            description=(
                "Rectangular large-scale TAN multifield cube with mixed image sizes, "
                "rotation, low overlap, and resolution ratios"
            ),
            field_model="analytic_basis",
            base_noise_sb=0.04,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=(
                ObservationSpec(
                    _tan_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=0.70,
                            height_scale=1.35,
                        ),
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0044,
                    ),
                    75.0,
                    0.84,
                    0.50,
                    4300.0,
                    5200.0,
                    "blue_rect",
                    4600.0,
                    120.0,
                ),
                ObservationSpec(
                    _tan_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=1.85,
                            height_scale=0.72,
                        ),
                        crval1=12.055,
                        crval2=-3.018,
                        scale_deg=0.0061,
                        rotation_deg=12.0,
                    ),
                    120.0,
                    0.89,
                    0.92,
                    4920.0,
                    5600.0,
                    "green_rect",
                    5260.0,
                    145.0,
                    (
                        slice(max(1, target_height // 6), max(2, target_height // 2)),
                        slice(max(1, target_width // 2), max(2, target_width // 2 + 9)),
                    ),
                ),
                ObservationSpec(
                    _tan_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=1.25,
                            height_scale=1.60,
                        ),
                        crval1=11.945,
                        crval2=-2.962,
                        scale_deg=0.0035,
                        rotation_deg=-17.0,
                    ),
                    210.0,
                    0.76,
                    1.35,
                    5600.0,
                    6400.0,
                    "orange_rect",
                    6030.0,
                    165.0,
                ),
                ObservationSpec(
                    _tan_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=0.92,
                            height_scale=0.58,
                        ),
                        crval1=12.022,
                        crval2=-2.990,
                        scale_deg=0.0050,
                        rotation_deg=5.0,
                    ),
                    165.0,
                    1.04,
                    0.72,
                    6480.0,
                    7360.0,
                    "red_rect",
                    6920.0,
                    185.0,
                ),
            ),
        ),
        "tpv_multifield_rectangular_scale": CaseSpec(
            case="tpv_multifield_rectangular_scale",
            projection="TPV",
            description=(
                "Rectangular large-scale TPV multifield cube with mixed image sizes, "
                "distortion, rotation, and low-overlap pointings"
            ),
            field_model="analytic_basis",
            base_noise_sb=0.05,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=(
                ObservationSpec(
                    _tan_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=0.82,
                            height_scale=1.10,
                        ),
                        crval1=12.000,
                        crval2=-3.000,
                        scale_deg=0.0042,
                    ),
                    70.0,
                    0.80,
                    0.58,
                    4300.0,
                    5200.0,
                    "blue_tan_rect",
                    4600.0,
                    120.0,
                ),
                ObservationSpec(
                    _tpv_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=1.45,
                            height_scale=0.68,
                        ),
                        crval1=12.048,
                        crval2=-3.026,
                        scale_deg=0.0058,
                        rotation_deg=13.0,
                        pv_scale=1.1,
                    ),
                    115.0,
                    0.88,
                    1.00,
                    4920.0,
                    5600.0,
                    "green_tpv_rect",
                    5260.0,
                    145.0,
                    (
                        slice(max(1, target_height // 5), max(2, target_height // 2)),
                        slice(max(1, target_width // 3), max(2, target_width // 3 + 8)),
                    ),
                ),
                ObservationSpec(
                    _tpv_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=1.10,
                            height_scale=1.72,
                        ),
                        crval1=11.952,
                        crval2=-2.956,
                        scale_deg=0.0036,
                        rotation_deg=-16.0,
                        pv_scale=1.25,
                    ),
                    220.0,
                    0.74,
                    1.42,
                    5600.0,
                    6400.0,
                    "orange_tpv_rect",
                    6030.0,
                    165.0,
                ),
                ObservationSpec(
                    _tpv_header(
                        *_scaled_dims(
                            target_width,
                            target_height,
                            width_scale=0.96,
                            height_scale=0.54,
                        ),
                        crval1=12.026,
                        crval2=-2.985,
                        scale_deg=0.0049,
                        rotation_deg=18.0,
                        pv_scale=1.4,
                    ),
                    170.0,
                    1.02,
                    0.78,
                    6480.0,
                    7360.0,
                    "red_tpv_rect",
                    6920.0,
                    185.0,
                ),
            ),
        ),
        "mixed_multifield_random10": CaseSpec(
            case="mixed_multifield_random10",
            projection="MIXED",
            description=(
                "Ten-image large-scale multifield stress case with random rectangular input sizes, "
                "mixed TAN/TPV geometry, mixed resolution, random shifts/rotations, and partial non-overlap"
            ),
            field_model="analytic_basis",
            base_noise_sb=0.055,
            target_header=target_header,
            target_centers=target_centers,
            target_edges=target_edges,
            observations=_random_large_rect_observations(
                target_width,
                target_height,
                seed=seed,
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
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int,
) -> tuple[
    list[SimulatedObservation], dict[str, Any], SpectralAxis, torch.Tensor, torch.Tensor
]:
    case_spec = _case_spec(case, size=size, width=width, height=height, seed=seed)
    field_model = _make_field_model(case_spec.field_model, seed=seed)
    observations = [
        _simulate_observation(
            tmpdir,
            name=f"{case}_{idx}",
            spec=spec,
            base_noise_sb=case_spec.base_noise_sb,
            seed=seed + idx,
            field_model=field_model,
        )
        for idx, spec in enumerate(case_spec.observations)
    ]
    spectral_axis = SpectralAxis(
        values=torch.tensor(case_spec.target_centers, dtype=torch.float32),
        unit="Angstrom",
        type="WAVE",
    )
    spectral_edges = torch.tensor(case_spec.target_edges, dtype=torch.float32)
    truth_planes = []
    for lo, hi in zip(
        case_spec.target_edges[:-1], case_spec.target_edges[1:], strict=False
    ):
        truth_planes.append(
            _integrated_field_image(
                case_spec.target_header,
                spectral_min=float(lo),
                spectral_max=float(hi),
                field_model=field_model,
            )
        )
    truth = torch.stack(truth_planes, dim=0)
    return observations, case_spec.target_header, spectral_axis, spectral_edges, truth


def _time_many(fn, runs: int) -> float:
    fn()
    samples = []
    for _ in range(max(1, runs)):
        t0 = time.perf_counter()
        fn()
        samples.append(time.perf_counter() - t0)
    samples.sort()
    return float(samples[len(samples) // 2])


def _cube_truth_metrics(
    data: torch.Tensor,
    weight: torch.Tensor,
    truth: torch.Tensor,
    *,
    target_header: dict[str, Any],
) -> dict[str, float]:
    metrics = evaluate_map_product(
        data,
        truth,
        weight=weight,
        header=target_header,
        value_semantics="surface_brightness",
    )
    coverage = (weight > 0.0).to(torch.float32).mean(dim=(-1, -2))
    residual = data - truth
    support = weight > 0.0
    weighted_rms_values: list[float] = []
    background_rms_values: list[float] = []
    detrended_rms_values: list[float] = []
    background_leakage_values: list[float] = []
    for band_idx in range(int(data.shape[0])):
        mask = support[band_idx]
        if not bool(mask.any()):
            weighted_rms_values.append(0.0)
            background_rms_values.append(0.0)
            detrended_rms_values.append(0.0)
            background_leakage_values.append(0.0)
            continue
        residual_supported = residual[band_idx][mask].to(dtype=torch.float32)
        weight_supported = weight[band_idx][mask].to(dtype=torch.float32)
        weight_sum = torch.clamp(weight_supported.sum(), min=1.0e-8)
        weighted_rms_values.append(
            float(
                torch.sqrt(
                    torch.sum(weight_supported * torch.square(residual_supported))
                    / weight_sum
                ).item()
            )
        )
        background_fit = fit_background_plane(residual[band_idx], mask)
        background_rms_values.append(background_fit.model_rms)
        detrended_rms_values.append(background_fit.detrended_rms)
        background_leakage_values.append(background_fit.leakage_fraction)
    metrics.update(
        {
            "cube_rmse_vs_truth": float(metrics["reconstruction_rmse"]),
            "cube_mae_vs_truth": float(metrics["reconstruction_mae"]),
            "weighted_residual_rms_mean": float(
                sum(weighted_rms_values) / max(len(weighted_rms_values), 1)
            ),
            "weighted_residual_rms_max": float(max(weighted_rms_values, default=0.0)),
            "background_residual_rms_mean": float(
                sum(background_rms_values) / max(len(background_rms_values), 1)
            ),
            "background_residual_rms_max": float(
                max(background_rms_values, default=0.0)
            ),
            "background_leakage_fraction_mean": float(
                sum(background_leakage_values) / max(len(background_leakage_values), 1)
            ),
            "background_leakage_fraction_max": float(
                max(background_leakage_values, default=0.0)
            ),
            "detrended_residual_rms_mean": float(
                sum(detrended_rms_values) / max(len(detrended_rms_values), 1)
            ),
            "per_band_rmse_mean": float(metrics["band_reconstruction_rmse_mean"]),
            "per_band_rmse_max": float(metrics["band_reconstruction_rmse_max"]),
            "coverage_frac_mean": float(coverage.mean().item()),
            "coverage_frac_min": float(coverage.min().item()),
            "output_snr": float(metrics["signal_to_residual_ratio"]),
            "output_snr_db": float(metrics["signal_to_residual_db"]),
        }
    )
    return metrics


def evaluate_torchsky_case(
    tmpdir: Path,
    *,
    case: str,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int,
    runs: int,
    resampling_kernel: str,
) -> dict[str, Any]:
    observations, target_header, spectral_axis, spectral_edges, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        width=width,
        height=height,
        seed=seed,
    )
    obs_list = [item.observation for item in observations]
    product = plan_map_product(
        obs_list,
        header_overrides=target_header,
        spectral_axis=spectral_axis,
        spectral_bin_edges=spectral_edges,
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
    case_spec = _case_spec(case, size=size, width=width, height=height, seed=seed)
    return {
        "case": case,
        "library": "torchsky",
        "method": "coadd",
        "variant": resampling_kernel,
        "geometry_family": map_compat._geometry_family_for_torchsky_kernel(
            resampling_kernel
        ),
        "projection": case_spec.projection,
        "description": case_spec.description,
        "field_model": case_spec.field_model,
        "bandpass_model": "normalized_gaussian_filter",
        "n_obs": len(observations),
        "n_band": int(spectral_axis.values.numel()),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(obs.header["NAXIS1"]) for obs in observations),
        "max_input_width": max(int(obs.header["NAXIS1"]) for obs in observations),
        "min_input_height": min(int(obs.header["NAXIS2"]) for obs in observations),
        "max_input_height": max(int(obs.header["NAXIS2"]) for obs in observations),
        "time_ms": dt * 1000.0,
        "mpix_s": float(result.data.numel()) / dt / 1e6,
        "mean_exposure_time": float(
            sum(item.exposure_time for item in observations) / max(len(observations), 1)
        ),
        "mean_transmission": float(
            sum(item.transmission for item in observations) / max(len(observations), 1)
        ),
        "mean_psf_sigma_pix": float(
            sum(item.psf_sigma_pix for item in observations) / max(len(observations), 1)
        ),
        **_cube_truth_metrics(
            result.data,
            result.weight,
            truth,
            target_header=target_header,
        ),
    }


def evaluate_torchsky_field_case(
    tmpdir: Path,
    *,
    case: str,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int,
    runs: int,
    field_model_kind: str = "wire_implicit",
    fit_steps: int = 10,
    fit_lr: float = 1.0e-2,
    max_observations: int = 4,
) -> dict[str, Any]:
    observations, target_header, spectral_axis, spectral_edges, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        width=width,
        height=height,
        seed=seed,
    )
    case_spec = _case_spec(case, size=size, width=width, height=height, seed=seed)
    field_observations = _field_observations_from_simulated_inputs(
        observations,
        max_observations=max_observations,
    )
    product = MapProductSpec(
        target_header=target_header,
        bands=tuple(
            MapBandpassSpec(
                label=f"band_{idx}",
                spectral_min=float(spectral_edges[idx].item()),
                spectral_max=float(spectral_edges[idx + 1].item()),
                transmission_center=float(spectral_axis.values[idx].item()),
            )
            for idx in range(int(spectral_axis.values.numel()))
        ),
        value_semantics="surface_brightness",
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
            num_spectral_samples=5,
            loss_kind="mse",
            loss_reduction="global",
        )
        rendered = render_scene_product(
            fit.model,
            product,
            num_spectral_samples=5,
        )
        return fit, rendered

    dt = _time_many(_fit_and_render, runs)
    fit, result = _fit_and_render()
    return {
        "case": case,
        "library": "torchsky_field",
        "method": "field_fit",
        "variant": field_model_kind,
        "geometry_family": "latent_scene_field",
        "projection": case_spec.projection,
        "description": case_spec.description,
        "field_model": case_spec.field_model,
        "field_model_kind": field_model_kind,
        "bandpass_model": "normalized_gaussian_filter",
        "n_obs": len(observations),
        "fit_observations": len(field_observations),
        "n_band": int(spectral_axis.values.numel()),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(obs.header["NAXIS1"]) for obs in observations),
        "max_input_width": max(int(obs.header["NAXIS1"]) for obs in observations),
        "min_input_height": min(int(obs.header["NAXIS2"]) for obs in observations),
        "max_input_height": max(int(obs.header["NAXIS2"]) for obs in observations),
        "time_ms": dt * 1000.0,
        "mpix_s": float(result.data.numel()) / dt / 1e6,
        "fit_steps": int(fit_steps),
        "fit_lr": float(fit_lr),
        "field_initial_loss": float(fit.initial_loss),
        "field_final_loss": float(fit.final_loss),
        "mean_exposure_time": float(
            sum(item.exposure_time for item in observations) / max(len(observations), 1)
        ),
        "mean_transmission": float(
            sum(item.transmission for item in observations) / max(len(observations), 1)
        ),
        "mean_psf_sigma_pix": float(
            sum(item.psf_sigma_pix for item in observations) / max(len(observations), 1)
        ),
        **_cube_truth_metrics(
            result.data,
            result.weight,
            truth,
            target_header=target_header,
        ),
    }


def evaluate_torchsky_spherical_case(
    tmpdir: Path,
    *,
    case: str,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int,
    runs: int,
    spherical_mode: str = "harmonic_direct",
    healpix_oversample: float = 2.0,
) -> dict[str, Any]:
    observations, target_header, spectral_axis, spectral_edges, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        width=width,
        height=height,
        seed=seed,
    )
    case_spec = _case_spec(case, size=size, width=width, height=height, seed=seed)
    nside = _benchmark_healpix_nside_for_target(
        target_header,
        oversample=healpix_oversample,
    )
    hp_maps = _truth_cube_to_healpix_maps(
        truth,
        target_header=target_header,
        nside=nside,
    )
    bundle = spherical_spectral_operator_bundle(mode=spherical_mode)
    lmax = min(3 * int(nside) - 1, 32)
    if spherical_mode not in {"harmonic_direct", "healpix_bridge"}:
        raise ValueError("spherical_mode must be 'harmonic_direct' or 'healpix_bridge'")
    geometry_family = (
        "spherical_harmonic_direct"
        if spherical_mode == "harmonic_direct"
        else "spherical_spectral_healpix_bridge"
    )
    product = MapProductSpec(
        target_header=target_header,
        bands=tuple(
            MapBandpassSpec(
                label=f"band_{idx}",
                spectral_min=float(spectral_edges[idx].item()),
                spectral_max=float(spectral_edges[idx + 1].item()),
                transmission_center=float(spectral_axis.values[idx].item()),
            )
            for idx in range(int(spectral_axis.values.numel()))
        ),
        value_semantics="surface_brightness",
    )
    scene = HealpixMapScene(
        hp_maps,
        nside=nside,
        nest=False,
        band_labels=tuple(f"band_{idx}" for idx in range(int(hp_maps.shape[0]))),
        lmax=lmax,
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
    return {
        "case": case,
        "library": "torchsky_spherical",
        "method": "spherical_reproject",
        "variant": spherical_mode,
        "geometry_family": geometry_family,
        "projection": case_spec.projection,
        "description": case_spec.description,
        "field_model": case_spec.field_model,
        "bandpass_model": "normalized_gaussian_filter",
        "n_obs": len(observations),
        "n_band": int(spectral_axis.values.numel()),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(obs.header["NAXIS1"]) for obs in observations),
        "max_input_width": max(int(obs.header["NAXIS1"]) for obs in observations),
        "min_input_height": min(int(obs.header["NAXIS2"]) for obs in observations),
        "max_input_height": max(int(obs.header["NAXIS2"]) for obs in observations),
        "time_ms": dt * 1000.0,
        "mpix_s": float(result.data.numel()) / dt / 1e6,
        "mean_exposure_time": float(
            sum(item.exposure_time for item in observations) / max(len(observations), 1)
        ),
        "mean_transmission": float(
            sum(item.transmission for item in observations) / max(len(observations), 1)
        ),
        "mean_psf_sigma_pix": float(
            sum(item.psf_sigma_pix for item in observations) / max(len(observations), 1)
        ),
        "healpix_nside": int(nside),
        "operator_bundle_geometry": bundle.geometry.name
        if bundle.geometry is not None
        else None,
        "spherical_mode": spherical_mode,
        **_cube_truth_metrics(
            result.data,
            result.weight,
            truth,
            target_header=target_header,
        ),
    }


def evaluate_torchsky_multiband_spherical_inverse_case(
    tmpdir: Path,
    *,
    case: str,
    size: int | None = None,
    width: int | None = None,
    height: int | None = None,
    seed: int,
    runs: int,
    healpix_oversample: float = 2.0,
    lmax: int | None = None,
) -> dict[str, Any]:
    observations, target_header, spectral_axis, spectral_edges, truth = make_case(
        tmpdir,
        case=case,
        size=size,
        width=width,
        height=height,
        seed=seed,
    )
    case_spec = _case_spec(case, size=size, width=width, height=height, seed=seed)
    nside = _benchmark_healpix_nside_for_target(
        target_header,
        oversample=healpix_oversample,
    )
    joint_lmax = min(3 * int(nside) - 1, 8) if lmax is None else int(lmax)
    hp_maps = _truth_cube_to_healpix_maps(
        truth,
        target_header=target_header,
        nside=nside,
    )
    joint_truth = torch.stack(
        [
            map2alm(
                hp_maps[idx],
                nside=int(nside),
                lmax=joint_lmax,
                pol=False,
            ).to(torch.complex64)
            for idx in range(int(hp_maps.shape[0]))
        ],
        dim=0,
    )
    nband = int(joint_truth.shape[0])
    band_values = spectral_axis.values.to(dtype=torch.float64)
    if nband > 1:
        band_span = torch.clamp(band_values.max() - band_values.min(), min=1.0)
        band_coord = (band_values - band_values.mean()) / band_span
        delta = band_coord[:, None] - band_coord[None, :]
        spectral_cov = torch.exp(-0.5 * torch.square(delta))
        spectral_cov = spectral_cov + 1.0e-3 * torch.eye(
            nband,
            dtype=spectral_cov.dtype,
            device=spectral_cov.device,
        )
    else:
        spectral_cov = torch.ones((1, 1), dtype=torch.float64)
    spectral_precision = torch.linalg.inv(spectral_cov)
    spectral_factor = torch.linalg.cholesky(spectral_precision).to(
        dtype=joint_truth.dtype
    )
    ell = torch.arange(joint_lmax + 1, dtype=torch.float64)
    cls = 5.0 / torch.square(ell + 1.0) + 0.1
    harmonic_diag = harmonic_cl_diagonal(
        lmax=joint_lmax,
        cls=cls,
        dtype=joint_truth.real.dtype,
        device=joint_truth.device,
    )
    harmonic_factor = torch.diag(harmonic_diag.to(dtype=joint_truth.dtype))

    hooks = build_spherical_harmonic_solver_hooks(lmax=joint_lmax)
    band_blocks = []
    render_blocks = []
    for band_idx in range(nband):
        targets = []
        band_center = float(band_values[band_idx].item())
        for obs_idx, item in enumerate(observations):
            nearest = int(
                torch.argmin(torch.abs(band_values - float(item.wavelength))).item()
            )
            if nearest != band_idx:
                continue
            offset_scale = 0.2 * float(obs_idx)
            targets.append(
                TargetRenderSpec.from_header(
                    _shift_header_center(
                        item.header,
                        dra_deg=0.1 * offset_scale,
                        ddec_deg=-0.05 * offset_scale,
                    ),
                    band_label=f"band_{band_idx}",
                    spectral_min=float(spectral_edges[band_idx].item()),
                    spectral_max=float(spectral_edges[band_idx + 1].item()),
                    transmission_center=band_center,
                )
            )
        if not targets:
            targets.append(
                TargetRenderSpec.from_header(
                    target_header,
                    band_label=f"band_{band_idx}",
                    spectral_min=float(spectral_edges[band_idx].item()),
                    spectral_max=float(spectral_edges[band_idx + 1].item()),
                    transmission_center=band_center,
                )
            )
        band_blocks.append(
            build_state_subspace_operator(
                joint_truth,
                hooks.bind_targets(tuple(targets)),
                selector=band_idx,
            )
        )
        render_blocks.append(
            build_state_subspace_operator(
                joint_truth,
                hooks.bind_target(
                    TargetRenderSpec.from_header(
                        target_header,
                        band_label=f"band_{band_idx}",
                        spectral_min=float(spectral_edges[band_idx].item()),
                        spectral_max=float(spectral_edges[band_idx + 1].item()),
                        transmission_center=band_center,
                    )
                ),
                selector=band_idx,
            )
        )

    prior_block = build_state_kronecker_operator(
        joint_truth,
        spectral_factor,
        harmonic_factor,
        apply_noise=lambda tensor: tensor * 0.08,
    )
    operator_set = compose_bound_operators(*band_blocks, prior_block)
    torch.manual_seed(seed)
    observed_blocks = []
    for block in band_blocks:
        for cutout in block.apply(joint_truth):
            sigma = max(float(cutout.to(torch.float32).std().item()) * 0.05, 1.0e-3)
            observed_blocks.append(cutout + sigma * torch.randn_like(cutout))
    data = (*observed_blocks, torch.zeros_like(joint_truth))
    initial_outputs = operator_set.apply(torch.zeros_like(joint_truth))
    initial_residual_norm = math.sqrt(
        sum(
            float(torch.sum(torch.abs(pred - obs) ** 2).item())
            for pred, obs in zip(initial_outputs, data, strict=True)
        )
    )

    def solve_fn():
        diagonal = operator_set.estimate_normal_operator_diagonal(
            data,
            initial_probe=torch.zeros_like(joint_truth),
            max_entries=max(int(joint_truth.numel()), 128),
            damping=1.0e-4,
        )
        return operator_set.solve_normal_equation_cg(
            data,
            initial_solution=torch.zeros_like(joint_truth),
            iterations=12,
            damping=1.0e-4,
            preconditioner=make_diagonal_preconditioner(diagonal),
        )

    dt = _time_many(solve_fn, runs)
    fit = solve_fn()
    reconstructed_cube = torch.stack(
        [render_block.apply(fit.solution) for render_block in render_blocks],
        dim=0,
    ).to(dtype=torch.float32)
    weight = torch.ones_like(reconstructed_cube)
    final_residual_norm = (
        float(fit.residual_norm_history[-1])
        if fit.residual_norm_history
        else initial_residual_norm
    )
    prior_norm = float(torch.linalg.vector_norm(prior_block.apply(fit.solution)).item())
    return {
        "case": case,
        "library": "torchsky_spherical",
        "method": "spherical_inverse",
        "variant": "multiband_kronecker_prior",
        "geometry_family": "spherical_harmonic_multiband_joint",
        "projection": case_spec.projection,
        "description": case_spec.description,
        "field_model": case_spec.field_model,
        "bandpass_model": "normalized_gaussian_filter",
        "n_obs": len(observations),
        "n_band": int(spectral_axis.values.numel()),
        "target_width": int(target_header["NAXIS1"]),
        "target_height": int(target_header["NAXIS2"]),
        "min_input_width": min(int(obs.header["NAXIS1"]) for obs in observations),
        "max_input_width": max(int(obs.header["NAXIS1"]) for obs in observations),
        "min_input_height": min(int(obs.header["NAXIS2"]) for obs in observations),
        "max_input_height": max(int(obs.header["NAXIS2"]) for obs in observations),
        "time_ms": dt * 1000.0,
        "mpix_s": float(reconstructed_cube.numel()) / dt / 1e6,
        "mean_exposure_time": float(
            sum(item.exposure_time for item in observations) / max(len(observations), 1)
        ),
        "mean_transmission": float(
            sum(item.transmission for item in observations) / max(len(observations), 1)
        ),
        "mean_psf_sigma_pix": float(
            sum(item.psf_sigma_pix for item in observations) / max(len(observations), 1)
        ),
        "healpix_nside": int(nside),
        "operator_bundle_geometry": hooks.operators.geometry.name,
        "joint_lmax": int(joint_lmax),
        "residual_norm_initial": float(initial_residual_norm),
        "residual_norm_final": float(final_residual_norm),
        "spectral_harmonic_prior_norm": prior_norm,
        **_cube_truth_metrics(
            reconstructed_cube,
            weight,
            truth,
            target_header=target_header,
        ),
    }


def _field_observations_from_simulated_inputs(
    observations: list[SimulatedObservation],
    *,
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
                wavelength=float(item.observation.wavelength)
                if item.observation.wavelength is not None
                else float(item.wavelength),
                spectral_min=item.observation.spectral_min,
                spectral_max=item.observation.spectral_max,
                transmission_center=item.observation.transmission_center,
                transmission_sigma=item.observation.transmission_sigma,
                psf_sigma_pix=float(item.psf_sigma_pix),
                band_name=item.observation.band_name,
            )
        )
    return field_observations


def _recommendation_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    by_case: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        if row.get("library") == "torchsky":
            by_case.setdefault(str(row["case"]), []).append(row)
    out: list[dict[str, Any]] = []
    for case, case_rows in by_case.items():
        best_rmse = min(
            case_rows,
            key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
        )
        rmse_limit = float(best_rmse["reconstruction_rmse"]) * 1.01
        near_best = [
            row for row in case_rows if float(row["reconstruction_rmse"]) <= rmse_limit
        ]
        default = min(near_best, key=lambda row: row["time_ms"])
        interpolatory = min(
            [row for row in case_rows if row["variant"] in _INTERPOLATORY_KERNELS],
            key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
        )
        for profile, basis, note in [
            (
                "default_general",
                default,
                "Fastest kernel within 1% of best cube RMSE on this case",
            ),
            (
                "high_accuracy_interpolatory",
                interpolatory,
                "Lowest-RMSE interpolatory kernel on this case",
            ),
        ]:
            out.append(
                {
                    "case": case,
                    "library": "torchsky_recommendation",
                    "method": "torchsky_recommendation",
                    "profile": profile,
                    "recommended_variant": basis["variant"],
                    "recommended_reconstruction_rmse": basis["reconstruction_rmse"],
                    "recommended_time_ms": basis["time_ms"],
                    "recommended_signal_to_residual_db": basis["signal_to_residual_db"],
                    "projection": basis["projection"],
                    "field_model": basis["field_model"],
                    "note": note,
                    "status": "INFO",
                }
            )
        smooth = [
            row for row in case_rows if row["variant"] in _SMOOTH_RECONSTRUCTION_KERNELS
        ]
        if smooth:
            best_smooth = min(
                smooth,
                key=lambda row: (row["reconstruction_rmse"], row["time_ms"]),
            )
            out.append(
                {
                    "case": case,
                    "library": "torchsky_recommendation",
                    "method": "torchsky_recommendation",
                    "profile": "smooth_reconstruction",
                    "recommended_variant": best_smooth["variant"],
                    "recommended_reconstruction_rmse": best_smooth[
                        "reconstruction_rmse"
                    ],
                    "recommended_time_ms": best_smooth["time_ms"],
                    "recommended_signal_to_residual_db": best_smooth[
                        "signal_to_residual_db"
                    ],
                    "projection": best_smooth["projection"],
                    "field_model": best_smooth["field_model"],
                    "note": "Best smooth reconstruction kernel on this case",
                    "status": "INFO",
                }
            )
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases", default=",".join(DEFAULT_CASES))
    parser.add_argument("--size", type=int, default=96)
    parser.add_argument("--width", type=int, default=None)
    parser.add_argument("--height", type=int, default=None)
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument(
        "--torchsky-kernels",
        default=",".join(DEFAULT_TORCHSKY_KERNELS),
        help="Comma-separated torchsky resampling kernels",
    )
    parser.add_argument(
        "--include-spherical-inverse",
        action="store_true",
        help="Also score the native multiband spherical inverse benchmark row",
    )
    parser.add_argument("--json-out", type=Path, default=None)
    parser.add_argument("--markdown-out", type=Path, default=None)
    parser.add_argument(
        "--baseline-json",
        type=Path,
        default=None,
        help="Optional baseline JSON artifact to enforce regression gates against",
    )
    args = parser.parse_args()

    rows: list[dict[str, Any]] = []
    with tempfile.TemporaryDirectory(prefix="torchsky_multifield_truth_") as tmp:
        tmpdir = Path(tmp)
        kernels = [
            part.strip() for part in args.torchsky_kernels.split(",") if part.strip()
        ]
        cases = [part.strip() for part in args.cases.split(",") if part.strip()]
        for case in cases:
            for kernel in kernels:
                rows.append(
                    evaluate_torchsky_case(
                        tmpdir,
                        case=case,
                        size=None
                        if args.width is not None or args.height is not None
                        else args.size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=args.runs,
                        resampling_kernel=kernel,
                    )
                )
            if args.include_spherical_inverse:
                rows.append(
                    evaluate_torchsky_multiband_spherical_inverse_case(
                        tmpdir,
                        case=case,
                        size=None
                        if args.width is not None or args.height is not None
                        else args.size,
                        width=args.width,
                        height=args.height,
                        seed=args.seed,
                        runs=args.runs,
                    )
                )
    rows.extend(_recommendation_rows(rows))
    markdown_path = args.markdown_out
    if markdown_path is None and args.json_out is not None:
        markdown_path = args.json_out.with_suffix(".md")
    markdown = write_map_benchmark_artifacts(
        rows,
        json_out=args.json_out,
        markdown_out=markdown_path,
    )
    if args.baseline_json is not None:
        baseline_rows = json.loads(args.baseline_json.read_text(encoding="utf-8"))
        errors = check_map_benchmark_regressions(rows, baseline_rows)
        if errors:
            for error in errors:
                print(f"REGRESSION: {error}", file=sys.stderr)
            return 1
    print(markdown)
    if args.json_out is not None:
        print(f"JSON: {args.json_out}")
    if markdown_path is not None:
        print(f"Markdown: {markdown_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
