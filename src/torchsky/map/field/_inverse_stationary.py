"""Stationary calibrated band observations for explicit-grid inverse solves."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING


import torch
import torch.nn as nn
from torch import Tensor
from ...device import _to


if TYPE_CHECKING:
    pass

from ..operators.background import ConditionedBackgroundField, LearnedBackgroundOp
from ..operators.bundle import (
    ObservationOperatorBundle,
)
from ..operators.learned_kernel import (
    ConditionedKernelPSF,
    SpatioSpectralTemporalEncoder,
)
from ..operators.psf import ExplicitPSFOp, LearnedPSFOp
from .types import FieldObservation, ForwardContext


def _extract_stationary_effective_psf(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    forward_context: ForwardContext | None,
) -> tuple[Tensor, str]:
    psf_op = operator_bundle.psf
    if psf_op is None:
        raise ValueError(
            "stationary calibration adapter requires a calibrated PSF operator"
        )
    if not psf_op.frozen:
        raise ValueError(
            "stationary calibration adapter requires a frozen PSF operator"
        )
    if psf_op.pixel_response_included is not True:
        raise ValueError(
            "stationary calibration adapter requires PSF pixel_response_included=True"
        )
    provenance = psf_op.pixel_response_provenance
    if not isinstance(provenance, str) or not provenance.strip():
        raise ValueError(
            "stationary calibration adapter requires non-empty PSF pixel_response_provenance"
        )
    if isinstance(psf_op, ExplicitPSFOp):
        spec = observation.psf_spec
        if spec is None or spec.mode != "kernel2d" or spec.kernel2d is None:
            raise ValueError(
                "ExplicitPSFOp stationary adapter requires observation.psf_spec kernel2d"
            )
        if not spec.frozen or spec.pixel_response_included is not True:
            raise ValueError(
                "ExplicitPSFOp observation.psf_spec must be frozen with pixel_response_included=True"
            )
        kernel = _to(
            spec.kernel2d.detach(),
            device=observation.image.device,
            dtype=observation.image.dtype,
        )
    elif isinstance(psf_op, LearnedPSFOp):
        model = psf_op.model
        if not isinstance(model, ConditionedKernelPSF):
            raise TypeError(
                "stationary LearnedPSFOp must wrap a ConditionedKernelPSF model"
            )
        if any(p.requires_grad for p in model.parameters()):
            raise ValueError(
                "stationary ConditionedKernelPSF parameters must be frozen"
            )
        height, width = map(int, observation.image.shape)
        param = next(model.parameters())
        probe_x = torch.tensor(
            [0.0, float(max(width - 1, 0)), 0.5 * float(max(width - 1, 0))],
            device=param.device,
            dtype=param.dtype,
        ) + float(observation.detector_x0_pix)
        probe_y = torch.tensor(
            [0.0, float(max(height - 1, 0)), 0.5 * float(max(height - 1, 0))],
            device=param.device,
            dtype=param.dtype,
        ) + float(observation.detector_y0_pix)
        wl = float(
            observation.wavelength
            if observation.wavelength is not None
            else (
                0.5
                * (float(observation.spectral_min) + float(observation.spectral_max))
                if observation.spectral_min is not None
                and observation.spectral_max is not None
                else 0.0
            )
        )
        time_mjd = (
            None
            if forward_context is None or forward_context.time_mjd is None
            else float(forward_context.time_mjd)
        )
        with torch.no_grad():
            kernels = model.effective_kernels(probe_x, probe_y, wl, time_mjd)
        ref_kernel = kernels[0]
        if not torch.allclose(
            kernels, ref_kernel.expand_as(kernels), rtol=1.0e-6, atol=1.0e-7
        ):
            raise ValueError(
                "calibrated PSF varies across detector coordinates; stationary adapter requires a coordinate-invariant PSF"
            )
        kernel = _to(
            ref_kernel.detach(),
            device=observation.image.device,
            dtype=observation.image.dtype,
        )
    else:
        raise TypeError(f"unsupported PSF operator type {type(psf_op)!r}")

    if (
        kernel.ndim != 2
        or kernel.shape[0] != kernel.shape[1]
        or int(kernel.shape[0]) % 2 != 1
    ):
        raise ValueError("stationary effective PSF kernel must be odd square 2D")
    if not bool(torch.isfinite(kernel).all()):
        raise ValueError("stationary effective PSF kernel must be finite")
    if bool((kernel < -1.0e-7).any()):
        raise ValueError("stationary effective PSF kernel must be non-negative")
    kernel = torch.clamp(kernel, min=0.0)
    kernel_sum = float(kernel.sum().item())
    if not math.isfinite(kernel_sum) or kernel_sum <= 0.0:
        raise ValueError("stationary effective PSF kernel must have positive sum")
    kernel = kernel / kernel.sum()
    return kernel, provenance


def _extract_stationary_calibrated_background(
    observation: FieldObservation,
    operator_bundle: ObservationOperatorBundle,
    *,
    forward_context: ForwardContext | None,
    background_gauge_policy: str,
) -> tuple[Tensor, float]:
    if (
        not isinstance(background_gauge_policy, str)
        or not background_gauge_policy.strip()
    ):
        raise ValueError("background_gauge_policy must be an explicit non-empty string")
    bg_op = operator_bundle.background
    if bg_op is None:
        return torch.zeros_like(observation.image), 0.0
    if not isinstance(bg_op, LearnedBackgroundOp):
        raise TypeError(
            "stationary calibration adapter requires LearnedBackgroundOp or None in bundle.background"
        )
    if bg_op.coordinate_frame != "detector":
        raise ValueError(
            "stationary calibrated background must use detector coordinate_frame"
        )
    model = bg_op.model
    if not isinstance(model, ConditionedBackgroundField):
        raise TypeError(
            "stationary LearnedBackgroundOp must wrap a ConditionedBackgroundField"
        )
    if any(p.requires_grad for p in model.parameters()):
        raise ValueError(
            "stationary ConditionedBackgroundField parameters must be frozen"
        )
    height, width = map(int, observation.image.shape)
    param = next(model.parameters())
    probe_x = torch.tensor(
        [0.0, float(max(width - 1, 0)), 0.5 * float(max(width - 1, 0))],
        device=param.device,
        dtype=param.dtype,
    ) + float(observation.detector_x0_pix)
    probe_y = torch.tensor(
        [0.0, float(max(height - 1, 0)), 0.5 * float(max(height - 1, 0))],
        device=param.device,
        dtype=param.dtype,
    ) + float(observation.detector_y0_pix)
    wl = float(
        observation.wavelength
        if observation.wavelength is not None
        else (
            0.5 * (float(observation.spectral_min) + float(observation.spectral_max))
            if observation.spectral_min is not None
            and observation.spectral_max is not None
            else 0.0
        )
    )
    time_mjd = (
        None
        if forward_context is None or forward_context.time_mjd is None
        else float(forward_context.time_mjd)
    )
    with torch.no_grad():
        values = model(probe_x, probe_y, wl, time_mjd)
    if not bool(torch.isfinite(values).all()):
        raise ValueError("stationary calibrated background evaluation must be finite")
    scalar_val = float(values[0].item())
    if not torch.allclose(
        values, values[:1].expand_as(values), rtol=1.0e-6, atol=1.0e-7
    ):
        raise ValueError(
            "calibrated background varies across detector coordinates; stationary adapter requires a constant pedestal"
        )
    bg_map = torch.full_like(observation.image, scalar_val)
    return bg_map, scalar_val


def _build_stationary_model_from_split(
    arrays: dict[str, Tensor],
    *,
    role: str,
) -> nn.Module:
    def _center_scale(values: Tensor) -> tuple[float, float]:
        vals = values.to(dtype=torch.float64)
        return float(vals.mean()), max(float(vals.max() - vals.min()), 1.0)

    x_center, x_scale = _center_scale(arrays["detector_x"])
    y_center, y_scale = _center_scale(arrays["detector_y"])
    wl_center, wl_scale = _center_scale(arrays["wavelength"])
    t_center, t_scale = _center_scale(arrays["time_mjd"])
    encoder = SpatioSpectralTemporalEncoder(
        spatial_center=(x_center, y_center),
        spatial_scale=(x_scale, y_scale),
        wavelength_center=wl_center,
        spectral_scale=wl_scale,
        time_center_mjd=t_center,
        temporal_scale_days=t_scale,
        spatial_frequencies=0,
        spectral_frequencies=0,
        temporal_frequencies=0,
    )
    if role == "psf":
        kernel_size = int(arrays["target_kernels"].shape[-1])
        model: nn.Module = ConditionedKernelPSF(
            encoder,
            n_basis=1,
            kernel_size=kernel_size,
            init_fwhm_pix=2.0,
            hidden_dim=32,
        )
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    if role == "background":
        model = ConditionedBackgroundField(encoder, hidden_dim=32, hidden_layers=2)
        for p in model.parameters():
            p.requires_grad_(False)
        return model
    raise ValueError(f"unsupported calibration role {role!r}")
