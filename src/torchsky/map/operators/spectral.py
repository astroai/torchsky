"""Shared spectral-response helpers for image-plane and field rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING, Sequence

import torch
from torch import Tensor

from .base import SpectralResponseOp

if TYPE_CHECKING:
    from ..field.types import FieldObservation


@dataclass
class ExplicitSpectralResponseOp(SpectralResponseOp):
    def __init__(self, *, name: str = "explicit_spectral_response") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedSpectralResponseOp(SpectralResponseOp):
    def __init__(self, *, name: str = "learned_spectral_response") -> None:
        super().__init__(name=name, learned=True)


def validate_override_quadrature(
    wavelengths: Tensor,
    weights: Tensor,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    """Validate caller-supplied spectral nodes and quadrature weights."""
    if not dtype.is_floating_point:
        raise TypeError("override quadrature dtype must be floating point")
    if not isinstance(wavelengths, Tensor) or not isinstance(weights, Tensor):
        raise TypeError("override wavelengths and weights must be tensors")
    if wavelengths.ndim != 1 or weights.ndim != 1:
        raise ValueError("override wavelengths and weights must be 1D")
    if wavelengths.shape != weights.shape or wavelengths.numel() == 0:
        raise ValueError(
            "override wavelengths and weights must be matching non-empty vectors"
        )
    nodes = wavelengths.to(device=device, dtype=dtype)
    quadrature = weights.to(device=device, dtype=dtype)
    if not bool(torch.isfinite(nodes).all()):
        raise ValueError("override wavelengths must be finite")
    if not bool(torch.isfinite(quadrature).all()) or bool((quadrature < 0.0).any()):
        raise ValueError("override weights must be finite and non-negative")
    if not bool(quadrature.sum() > 0.0):
        raise ValueError("override weights must have positive total mass")
    return nodes, quadrature


def bandpass_bounds(observation: FieldObservation) -> tuple[float, float]:
    if observation.spectral_min is not None and observation.spectral_max is not None:
        lower = float(observation.spectral_min)
        upper = float(observation.spectral_max)
    elif observation.passband is not None:
        lower = float(observation.passband.wavelength[0].item())
        upper = float(observation.passband.wavelength[-1].item())
    elif observation.wavelength is not None:
        lower = float(observation.wavelength)
        upper = float(observation.wavelength)
    else:
        raise ValueError(
            "FieldObservation must define wavelength or spectral_min/spectral_max"
        )
    if not math.isfinite(lower) or not math.isfinite(upper) or upper < lower:
        raise ValueError("FieldObservation spectral bounds must be finite and ordered")
    return lower, upper


def has_tabulated_transmission(observation: FieldObservation) -> bool:
    return observation.passband is not None or (
        observation.transmission_wavelength is not None
        and observation.transmission_throughput is not None
    )


def sorted_tabulated_transmission(
    observation: FieldObservation,
    *,
    device: torch.device,
    dtype: torch.dtype,
) -> tuple[Tensor, Tensor]:
    if observation.passband is not None:
        wl = observation.passband.wavelength
        tt = observation.passband.throughput
    else:
        wl = observation.transmission_wavelength
        tt = observation.transmission_throughput
        assert wl is not None and tt is not None
    wl_d = wl.to(device=device, dtype=dtype).reshape(-1)
    tt_d = tt.to(device=device, dtype=dtype).reshape(-1)
    wl_s, perm = torch.sort(wl_d)
    tt_s = torch.clamp(tt_d[perm], min=0.0)
    return wl_s, tt_s


def linear_interp_transmission(
    wl_tab: Tensor, tt_tab: Tensor, wl_query: Tensor
) -> Tensor:
    xq = wl_query
    if not xq.is_floating_point():
        xq = xq.to(dtype=torch.get_default_dtype())
    x = wl_tab.to(device=xq.device, dtype=xq.dtype)
    y = tt_tab.to(device=xq.device, dtype=xq.dtype)
    lo = x[0]
    hi = x[-1]
    clamped = torch.clamp(xq, lo, hi)
    idx = torch.searchsorted(x, clamped, right=False) - 1
    idx = torch.clamp(idx, 0, x.numel() - 2)
    x0 = x[idx]
    x1 = x[idx + 1]
    denom = x1 - x0
    t = torch.where(denom > 0.0, (clamped - x0) / denom, torch.zeros_like(clamped))
    y0 = y[idx]
    y1 = y[idx + 1]
    out = torch.lerp(y0, y1, t)
    outside = (xq < lo) | (xq > hi)
    return torch.where(outside, torch.zeros_like(out), out)


def tabulated_response_or_uniform(
    response: Tensor,
    *,
    mass: Tensor,
    mass_eps: float = 1.0e-30,
) -> Tensor:
    return torch.where(mass > mass_eps, response, torch.ones_like(response))


def bandpass_quadrature(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    work_dtype = dtype or (
        observation.image.dtype
        if observation.image.is_floating_point()
        else torch.get_default_dtype()
    )
    if not work_dtype.is_floating_point:
        raise TypeError("bandpass quadrature dtype must be floating point")
    lower, upper = bandpass_bounds(observation)
    if lower == upper or num_spectral_samples == 1:
        samples = torch.tensor([0.5 * (lower + upper)], dtype=work_dtype, device=device)
        return samples, torch.ones_like(samples)
    n = int(num_spectral_samples)
    samples = torch.linspace(lower, upper, steps=n, dtype=work_dtype, device=device)
    span = float(upper - lower)
    h = span / float(n - 1)
    trap = torch.full((n,), h, dtype=work_dtype, device=device)
    trap[0] = h * 0.5
    trap[-1] = h * 0.5
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(
            observation, device=device, dtype=work_dtype
        )
        response = linear_interp_transmission(wl_tab, tt_tab, samples)
        mass = (response * trap).sum()
        response = tabulated_response_or_uniform(response, mass=mass)
    else:
        center = (
            float(observation.transmission_center)
            if observation.transmission_center is not None
            else 0.5 * (lower + upper)
        )
        sigma = observation.transmission_sigma
        if sigma is None:
            response = torch.ones_like(samples)
        else:
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    "FieldObservation transmission_sigma must be finite and positive"
                )
            response = torch.exp(
                -0.5 * (((samples - center) / sigma) * ((samples - center) / sigma))
            )
    weights = response * trap
    return samples, weights / torch.clamp(weights.sum(), min=1e-8)


def bandpass_response(observation: FieldObservation, wavelengths: Tensor) -> Tensor:
    lower, upper = bandpass_bounds(observation)
    if not wavelengths.is_floating_point():
        wavelengths = wavelengths.to(dtype=torch.get_default_dtype())
    if lower == upper:
        atol = max(1.0e-3, 1.0e-6 * max(abs(lower), 1.0))
        return (torch.abs(wavelengths - float(lower)) <= atol).to(
            dtype=wavelengths.dtype
        )
    inside = (wavelengths >= float(lower)) & (wavelengths <= float(upper))
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(
            observation,
            device=wavelengths.device,
            dtype=wavelengths.dtype,
        )
        response = linear_interp_transmission(wl_tab, tt_tab, wavelengths)
        return response * inside.to(dtype=response.dtype)
    sigma = observation.transmission_sigma
    if sigma is None:
        return inside.to(dtype=wavelengths.dtype)
    sigma = float(sigma)
    if not math.isfinite(sigma) or sigma <= 0.0:
        raise ValueError(
            "FieldObservation transmission_sigma must be finite and positive"
        )
    center = (
        float(observation.transmission_center)
        if observation.transmission_center is not None
        else 0.5 * (float(lower) + float(upper))
    )
    response = torch.exp(
        -0.5 * (((wavelengths - center) / sigma) * ((wavelengths - center) / sigma))
    )
    return response * inside.to(dtype=response.dtype)


def bandpass_mc_samples(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    work_dtype = dtype or (
        observation.image.dtype
        if observation.image.is_floating_point()
        else torch.get_default_dtype()
    )
    if not work_dtype.is_floating_point:
        raise TypeError("bandpass sampling dtype must be floating point")
    lower, upper = bandpass_bounds(observation)
    if lower == upper or num_spectral_samples == 1:
        sample = torch.tensor([0.5 * (lower + upper)], dtype=work_dtype, device=device)
        return sample, torch.ones_like(sample)
    proposal_size = (
        int(proposal_grid_size)
        if proposal_grid_size is not None
        else max(65, 8 * int(num_spectral_samples) + 1)
    )
    if proposal_size < 2:
        proposal_size = 2
    proposal_samples = torch.linspace(
        lower, upper, steps=proposal_size, dtype=work_dtype, device=device
    )
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(
            observation, device=device, dtype=work_dtype
        )
        proposal_weights = linear_interp_transmission(wl_tab, tt_tab, proposal_samples)
        proposal_weights = tabulated_response_or_uniform(
            proposal_weights, mass=proposal_weights.sum()
        )
    else:
        center = (
            float(observation.transmission_center)
            if observation.transmission_center is not None
            else 0.5 * (lower + upper)
        )
        sigma = observation.transmission_sigma
        if sigma is None:
            proposal_weights = torch.ones_like(proposal_samples)
        else:
            sigma = float(sigma)
            if not math.isfinite(sigma) or sigma <= 0.0:
                raise ValueError(
                    "FieldObservation transmission_sigma must be finite and positive"
                )
            proposal_weights = torch.exp(
                -0.5
                * (
                    ((proposal_samples - center) / sigma)
                    * ((proposal_samples - center) / sigma)
                )
            )
    proposal_weights = proposal_weights / torch.clamp(
        proposal_weights.sum(), min=1.0e-8
    )
    sample_ids = torch.multinomial(
        proposal_weights,
        int(num_spectral_samples),
        replacement=True,
        generator=generator,
    )
    samples = proposal_samples[sample_ids]
    samples, _ = torch.sort(samples)
    weights = torch.full_like(samples, 1.0 / float(num_spectral_samples))
    return samples, weights


def mixture_mc_samples(
    observations: Sequence[FieldObservation],
    *,
    num_spectral_samples: int,
    device: torch.device,
    dtype: torch.dtype | None = None,
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    if not observations:
        raise ValueError("mixture_mc_samples requires at least one observation")
    work_dtype = dtype or (
        observations[0].image.dtype
        if observations and observations[0].image.is_floating_point()
        else torch.get_default_dtype()
    )
    if not work_dtype.is_floating_point:
        raise TypeError("mixture sampling dtype must be floating point")
    bounds = [bandpass_bounds(observation) for observation in observations]
    lower = min(bound[0] for bound in bounds)
    upper = max(bound[1] for bound in bounds)
    if lower == upper or num_spectral_samples == 1:
        sample = torch.tensor([0.5 * (lower + upper)], dtype=work_dtype, device=device)
        return sample, torch.ones_like(sample)
    proposal_size = (
        int(proposal_grid_size)
        if proposal_grid_size is not None
        else max(65, 8 * int(num_spectral_samples) + 1)
    )
    if proposal_size < 2:
        proposal_size = 2
    proposal_samples = torch.linspace(
        lower, upper, steps=proposal_size, dtype=work_dtype, device=device
    )
    proposal = torch.zeros_like(proposal_samples)
    for observation in observations:
        response = bandpass_response(observation, proposal_samples)
        response_sum = torch.clamp(response.sum(), min=1.0e-8)
        proposal = proposal + response / response_sum
    proposal = proposal / torch.clamp(proposal.sum(), min=1.0e-8)
    sample_ids = torch.multinomial(
        proposal,
        int(num_spectral_samples),
        replacement=True,
        generator=generator,
    )
    samples = proposal_samples[sample_ids]
    samples, order = torch.sort(samples)
    sample_probs = proposal[sample_ids][order]
    return samples, sample_probs


def mixture_importance_weights(
    observation: FieldObservation,
    wavelengths: Tensor,
    proposal_probs: Tensor,
) -> Tensor:
    response = bandpass_response(observation, wavelengths)
    weights = response / torch.clamp(
        proposal_probs.to(dtype=response.dtype), min=1.0e-8
    )
    return weights / torch.clamp(weights.sum(), min=1.0e-8)
