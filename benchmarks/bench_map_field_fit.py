#!/usr/bin/env python3
"""Benchmark differentiable latent-field fitting against synthetic truth."""

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
from torch import Tensor

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
    SkyEncoderMLPField,
    SkyEncoderResMLPField,
    SineImplicitField,
    SpectralBasisField,
    estimate_field_domain,
    evaluate_map_product,
    fit_field_model,
    make_field_model,
    check_map_benchmark_regressions,
    render_field_cube,
    render_field_observation,
    render_field_product,
    plan_map_product,
    stack_map_product,
    write_map_benchmark_artifacts,
)
from torchsky.spectral import SpectralAxis  # noqa: E402


DEFAULT_CASES = (
    "tan_sip_field_fit_multicolor",
    "tan_field_fit_exact_transmission",
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
    "sky_lambda_planes_field",
    "conditioned_spectral_field",
    "hashgrid_conditioned_field",
    "hashgrid_spectral_field",
    "sky_encoder_mlp",
    "sky_encoder_resmlp",
    "coadd_bilinear",
    "coadd_bicubic",
)


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


def _field_fit_kwargs_for_model(model_kind: str, *, case: str | None = None) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "lr": 5.0e-3,
        "num_spectral_samples": 9,
        "loss_kind": "charbonnier",
        "loss_reduction": "per_observation",
        "lr_schedule": "cosine",
        "grad_clip_norm": 1.0,
        "curriculum": "none",
    }
    if case in {"tan_field_fit_exact_transmission", "tpv_field_fit_deep_multiband_exact"}:
        kwargs["loss_kind"] = "mse"
        kwargs["loss_reduction"] = "global"
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
    elif model_kind == "sky_lambda_planes_field":
        kwargs["lr"] = 4.0e-3
        kwargs["grad_clip_norm"] = 1.0
    elif model_kind == "conditioned_spectral_field":
        kwargs["lr"] = 3.5e-3
        kwargs["grad_clip_norm"] = 0.9
    elif model_kind == "hashgrid_conditioned_field":
        kwargs["lr"] = 3.5e-3
        kwargs["grad_clip_norm"] = 0.85
    elif model_kind == "hashgrid_spectral_field":
        kwargs["lr"] = 3.0e-3
        kwargs["grad_clip_norm"] = 0.8
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
        if case == "tpv_field_fit_deep_multiband_exact":
            kwargs["lr"] = 3.0e-3
            kwargs["loss_kind"] = "charbonnier"
            kwargs["loss_reduction"] = "global"
            kwargs["grad_clip_norm"] = 1.0
    return kwargs


def _field_model_kwargs_for_case(model_kind: str, *, case: str | None = None) -> dict[str, Any]:
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
        if case in {"tan_field_fit_exact_transmission", "tpv_field_fit_deep_multiband_exact"}:
            return {
                "hidden_dim": 128,
                "hidden_layers": 2,
                "num_spatial_frequencies": 6,
                "basis_dim": 16,
                "spectral_hidden_dim": 32,
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
    if model_kind == "sky_encoder_resmlp":
        if case == "tpv_field_fit_deep_multiband_exact":
            return {
                "hidden_dim": 160,
                "num_blocks": 4,
                "num_fourier_frequencies": 10,
                "spectral_frequencies": 4,
            }
    return {}


def _make_field_model_for_case(
    model_kind: str,
    domain: FieldDomain,
    *,
    case: str | None = None,
):
    kwargs = _field_model_kwargs_for_case(model_kind, case=case)
    if model_kind == "spectral_basis_field":
        return SpectralBasisField(domain, **kwargs)
    if model_kind == "learned_spectral_basis_field":
        return LearnedSpectralBasisField(domain, **kwargs)
    if model_kind == "sky_encoder_mlp":
        return SkyEncoderMLPField(domain, **kwargs)
    if model_kind == "sky_encoder_resmlp":
        return SkyEncoderResMLPField(domain, **kwargs)
    return make_field_model(model_kind, domain)


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
    deep_exact_domain = FieldDomain(
        center_ra_deg=210.0,
        center_dec_deg=-6.0,
        spatial_scale_deg=0.06,
        wavelength_center=14000.0,
        wavelength_scale=9000.0,
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
                    gap=(slice(size // 5, size // 5 + 5), slice(size // 3, size - size // 4)),
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
                    gap=(slice(size // 3, size - size // 5), slice(size // 4, size // 4 + 4)),
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
                    gap=(slice(size // 4, size // 2), slice(size // 3, size - size // 6)),
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
                    gap=(slice(size // 5, size // 5 + 6), slice(size // 2, size - size // 8)),
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
                    gap=(slice(size // 3, size - size // 5), slice(size // 4, size // 4 + 7)),
                ),
            ),
        ),
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
                values=torch.tensor([4700.0, 5550.0, 6400.0, 7150.0], dtype=torch.float64),
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
                    gap=(slice(size // 4, size // 4 + 7), slice(size // 3, size - size // 5)),
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
                    gap=(slice(size // 3, size // 3 + 7), slice(size // 2, size - size // 5)),
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
        ridge = 0.18 * torch.sin(130.0 * dra + 0.5 * lam) * torch.cos(
            110.0 * ddec - 0.25 * lam
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
            num_spectral_samples=9,
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
    within = [
        row for row in rows if float(row[metric_name]) <= 1.05 * best_rmse
    ]
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


def evaluate_case(
    *,
    case: str,
    model_kind: str,
    size: int,
    seed: int,
    steps: int,
    runs: int,
) -> dict[str, Any]:
    case_spec = _build_case_specs(size)[case]
    with tempfile.TemporaryDirectory(prefix="torchsky-field-fit-") as tmpdir_str:
        observations, image_observations, truth_cube = _simulate_observations(
            case_spec,
            seed=seed,
            tmpdir=Path(tmpdir_str),
        )
        product = plan_map_product(
            image_observations,
            header_overrides=case_spec.target_header,
            spectral_axis=case_spec.spectral_axis,
            spectral_bin_edges=case_spec.spectral_bin_edges,
        )
        support_reference_product = stack_map_product(
            image_observations,
            product,
            resampling_kernel="bilinear",
        )
        domain = estimate_field_domain(observations)
        fit_times = []
        result = None
        product_pred = None
        observation_residuals = []
        model_kwargs: dict[str, Any] = {}
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
            fit_kwargs = _field_fit_kwargs_for_model(model_kind, case=case_spec.case)
            model_kwargs = _field_model_kwargs_for_case(model_kind, case=case_spec.case)
            torch.manual_seed(int(seed) + 999)
            warmup_model = _make_field_model_for_case(
                model_kind,
                domain,
                case=case_spec.case,
            )
            fit_field_model(
                observations,
                warmup_model,
                steps=steps,
                **fit_kwargs,
            )
            for run_idx in range(int(runs)):
                torch.manual_seed(int(seed) + 1000 + run_idx)
                model = _make_field_model_for_case(
                    model_kind,
                    domain,
                    case=case_spec.case,
                )
                start = time.perf_counter()
                result = fit_field_model(
                    observations,
                    model,
                    steps=steps,
                    **fit_kwargs,
                )
                fit_times.append((time.perf_counter() - start) * 1000.0)
            assert result is not None
            product_pred = render_field_product(
                result.model,
                product,
            )
            for observation in observations:
                pred = render_field_observation(
                    result.model,
                    observation,
                    num_spectral_samples=9,
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
        "fit_method": "weighted_mse_adam" if result is not None else "weighted_cube_coadd",
        "num_observations": len(observations),
        "num_bands": int(case_spec.spectral_axis.values.numel()),
        "steps": int(steps),
        "fit_time_ms": fit_time_ms,
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
        "band_centers_angstrom": [
            float(value)
            for value in case_spec.spectral_axis.to_wavelength().detach().cpu().tolist()
        ],
        "signal_rms_truth": signal_rms,
        "residual_rms": residual_rms,
        "signal_to_residual_ratio": signal_rms / max(residual_rms, 1.0e-8),
        "signal_to_residual_db": 20.0 * math.log10(signal_rms / max(residual_rms, 1.0e-8)),
        "observation_rmse": (
            float(torch.stack(observation_residuals).mean().item())
            if observation_residuals
            else None
        ),
        "mean_noise_sigma": float(
            sum(spec.noise_sigma for spec in case_spec.observations)
            / len(case_spec.observations)
        ),
        "min_noise_sigma": float(min(spec.noise_sigma for spec in case_spec.observations)),
        "max_noise_sigma": float(max(spec.noise_sigma for spec in case_spec.observations)),
        "min_psf_sigma_pix": float(min(spec.psf_sigma_pix for spec in case_spec.observations)),
        "max_psf_sigma_pix": float(max(spec.psf_sigma_pix for spec in case_spec.observations)),
        "min_resolution_arcsec_pix": float(
            min(_pixel_scale_arcsec(spec.header) for spec in case_spec.observations)
        ),
        "max_resolution_arcsec_pix": float(
            max(_pixel_scale_arcsec(spec.header) for spec in case_spec.observations)
        ),
        "min_bandwidth_angstrom": float(
            min(spec.spectral_max - spec.spectral_min for spec in case_spec.observations)
        ),
        "max_bandwidth_angstrom": float(
            max(spec.spectral_max - spec.spectral_min for spec in case_spec.observations)
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=48)
    parser.add_argument("--steps", type=int, default=120)
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
    args = parser.parse_args()

    cases = tuple(item for item in args.cases.split(",") if item)
    models = tuple(item for item in args.models.split(",") if item)
    rows = []
    for case in cases:
        case_rows = [
            evaluate_case(
                case=case,
                model_kind=model_kind,
                size=args.size,
                seed=args.seed,
                steps=args.steps,
                runs=args.runs,
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
