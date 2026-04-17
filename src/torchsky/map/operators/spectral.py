"""Shared spectral-response helpers for image-plane and field rendering."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import Tensor

from ..field.types import FieldObservation
from .base import SpectralResponseOp


@dataclass
class ExplicitSpectralResponseOp(SpectralResponseOp):
    def __init__(self, *, name: str = "explicit_spectral_response") -> None:
        super().__init__(name=name, learned=False)


@dataclass
class LearnedSpectralResponseOp(SpectralResponseOp):
    def __init__(self, *, name: str = "learned_spectral_response") -> None:
        super().__init__(name=name, learned=True)


def bandpass_bounds(observation: FieldObservation) -> tuple[float, float]:
    if observation.spectral_min is not None and observation.spectral_max is not None:
        lower = float(observation.spectral_min)
        upper = float(observation.spectral_max)
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
    return (
        observation.transmission_wavelength is not None
        and observation.transmission_throughput is not None
    )


def sorted_tabulated_transmission(
    observation: FieldObservation,
    *,
    device: torch.device,
) -> tuple[Tensor, Tensor]:
    wl = observation.transmission_wavelength
    tt = observation.transmission_throughput
    assert wl is not None and tt is not None
    wl_d = wl.to(device=device, dtype=torch.float32).reshape(-1)
    tt_d = tt.to(device=device, dtype=torch.float32).reshape(-1)
    wl_s, perm = torch.sort(wl_d)
    tt_s = torch.clamp(tt_d[perm], min=0.0)
    return wl_s, tt_s


def linear_interp_transmission(wl_tab: Tensor, tt_tab: Tensor, wl_query: Tensor) -> Tensor:
    xq = wl_query.to(dtype=torch.float32)
    x = wl_tab.to(dtype=torch.float32)
    y = tt_tab.to(dtype=torch.float32)
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
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    lower, upper = bandpass_bounds(observation)
    if lower == upper or num_spectral_samples == 1:
        samples = torch.tensor([0.5 * (lower + upper)], dtype=torch.float32, device=device)
        return samples, torch.ones_like(samples)
    n = int(num_spectral_samples)
    samples = torch.linspace(lower, upper, steps=n, dtype=torch.float32, device=device)
    span = float(upper - lower)
    h = span / float(n - 1)
    trap = torch.full((n,), h, dtype=torch.float32, device=device)
    trap[0] = h * 0.5
    trap[-1] = h * 0.5
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(observation, device=device)
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
            response = torch.exp(-0.5 * torch.square((samples - center) / sigma))
    weights = response * trap
    return samples, weights / torch.clamp(weights.sum(), min=1e-8)


def bandpass_response(observation: FieldObservation, wavelengths: Tensor) -> Tensor:
    lower, upper = bandpass_bounds(observation)
    wavelengths = wavelengths.to(dtype=torch.float32)
    if lower == upper:
        atol = max(1.0e-3, 1.0e-6 * max(abs(lower), 1.0))
        return (torch.abs(wavelengths - float(lower)) <= atol).to(dtype=wavelengths.dtype)
    inside = (wavelengths >= float(lower)) & (wavelengths <= float(upper))
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(
            observation, device=wavelengths.device
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
    response = torch.exp(-0.5 * torch.square((wavelengths - center) / sigma))
    return response * inside.to(dtype=response.dtype)


def bandpass_mc_samples(
    observation: FieldObservation,
    *,
    num_spectral_samples: int,
    device: torch.device,
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    lower, upper = bandpass_bounds(observation)
    if lower == upper or num_spectral_samples == 1:
        sample = torch.tensor([0.5 * (lower + upper)], dtype=torch.float32, device=device)
        return sample, torch.ones_like(sample)
    proposal_size = (
        int(proposal_grid_size)
        if proposal_grid_size is not None
        else max(65, 8 * int(num_spectral_samples) + 1)
    )
    if proposal_size < 2:
        proposal_size = 2
    proposal_samples = torch.linspace(
        lower, upper, steps=proposal_size, dtype=torch.float32, device=device
    )
    if has_tabulated_transmission(observation):
        wl_tab, tt_tab = sorted_tabulated_transmission(observation, device=device)
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
                -0.5 * torch.square((proposal_samples - center) / sigma)
            )
    proposal_weights = proposal_weights / torch.clamp(proposal_weights.sum(), min=1.0e-8)
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
    proposal_grid_size: int | None = None,
    generator: torch.Generator | None = None,
) -> tuple[Tensor, Tensor]:
    if num_spectral_samples < 1:
        raise ValueError("num_spectral_samples must be positive")
    bounds = [bandpass_bounds(observation) for observation in observations]
    lower = min(bound[0] for bound in bounds)
    upper = max(bound[1] for bound in bounds)
    if lower == upper or num_spectral_samples == 1:
        sample = torch.tensor([0.5 * (lower + upper)], dtype=torch.float32, device=device)
        return sample, torch.ones_like(sample)
    proposal_size = (
        int(proposal_grid_size)
        if proposal_grid_size is not None
        else max(65, 8 * int(num_spectral_samples) + 1)
    )
    if proposal_size < 2:
        proposal_size = 2
    proposal_samples = torch.linspace(
        lower, upper, steps=proposal_size, dtype=torch.float32, device=device
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
    weights = response / torch.clamp(proposal_probs.to(dtype=response.dtype), min=1.0e-8)
    return weights / torch.clamp(weights.sum(), min=1.0e-8)

