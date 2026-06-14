"""Synthetic noise helpers for field pretraining."""

from __future__ import annotations

import math

import torch
from torch import Tensor

from .types import FieldNoisePretrainKind


def _synthetic_noise_target(
    coords: Tensor,
    *,
    noise_kind: FieldNoisePretrainKind,
    num_terms: int,
    max_frequency: float,
    generator: torch.Generator,
) -> Tensor:
    flat = coords.reshape(-1, coords.shape[-1]).to(dtype=torch.float32)
    device = flat.device
    coord_dim = flat.shape[-1]
    term_count = max(int(num_terms), 1)
    freq_direction = torch.randn(term_count, coord_dim, generator=generator)
    freq_direction = freq_direction / torch.clamp(
        torch.linalg.norm(freq_direction, dim=-1, keepdim=True),
        min=1.0e-6,
    )
    radial = torch.rand(term_count, 1, generator=generator) * float(max_frequency)
    freqs = (1.0 + radial) * freq_direction
    phases = 2.0 * math.pi * torch.rand(term_count, generator=generator)
    if noise_kind == "uniform":
        amplitudes = 2.0 * torch.rand(term_count, generator=generator) - 1.0
    elif noise_kind in {"gaussian", "spectrum"}:
        amplitudes = torch.randn(term_count, generator=generator)
    else:
        raise ValueError(f"unsupported noise pretraining kind {noise_kind!r}")
    if noise_kind == "spectrum":
        spectral_slope = torch.pow(
            1.0 + torch.linalg.norm(freqs, dim=-1),
            1.5,
        )
        amplitudes = amplitudes / spectral_slope
    freqs = freqs.to(device=device, dtype=flat.dtype)
    phases = phases.to(device=device, dtype=flat.dtype)
    amplitudes = amplitudes.to(device=device, dtype=flat.dtype)
    signal = torch.sin(flat @ freqs.transpose(0, 1) + phases) @ amplitudes
    signal = signal.reshape(coords.shape[:-1])
    centered = signal - signal.mean()
    scaled = centered / torch.clamp(centered.std(unbiased=False), min=1.0e-5)
    return torch.clamp(0.5 + 0.18 * scaled, min=1.0e-4)


__all__ = ["_synthetic_noise_target"]
