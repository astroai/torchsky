"""Spectral-axis validation and projection helpers for cube mapmaking."""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import Tensor

from ..spectral import SpectralAxis
from .observation import BandMetadataTable, ImageObservation


def _validated_spectral_axis(
    axis: SpectralAxis | Sequence[float] | Tensor,
    *,
    default_unit: str = "Angstrom",
    default_type: str = "WAVE",
) -> SpectralAxis:
    if not isinstance(axis, SpectralAxis):
        axis = SpectralAxis(
            values=torch.as_tensor(axis, dtype=torch.float64),
            unit=default_unit,
            type=default_type,
        )
    values = axis.values.to(dtype=torch.float64)
    if values.ndim != 1 or values.numel() == 0:
        raise ValueError("Spectral axis must be one-dimensional and non-empty")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("Spectral axis values must be finite")
    if values.numel() > 1 and not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("Spectral axis values must be strictly increasing")
    return SpectralAxis(
        values=values,
        unit=axis.unit,
        type=axis.type,
        rest_frequency=axis.rest_frequency,
        redshift=axis.redshift,
    )


def _spectral_fractional_index(wave_centers: Tensor, wavelengths: Tensor) -> Tensor:
    """Map wavelengths to fractional indices on a strictly increasing axis.

    ``k + f`` means: the wavelength lies between ``wave_centers[k]`` and
    ``wave_centers[k + 1]``, a fraction ``f`` of the way. Values outside the axis
    map outside ``[0, count - 1]`` so downstream bounds checks reject them.
    This is the non-uniform generalization of ``(lam - lam_0) / d_lam``.
    """
    count = int(wave_centers.numel())
    if count < 2:
        return torch.zeros_like(wavelengths)
    index = torch.searchsorted(wave_centers, wavelengths, right=True) - 1
    index = index.clamp(0, count - 2)
    lower = wave_centers[index]
    upper = wave_centers[index + 1]
    fraction = (wavelengths - lower) / torch.clamp(
        upper - lower, min=torch.finfo(wavelengths.dtype).tiny
    )
    return index.to(wavelengths.dtype) + fraction


def _observation_spectral_columns(
    observations: Sequence[ImageObservation],
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor, tuple[str | None, ...]]:
    centers: list[float] = []
    lower: list[float] = []
    upper: list[float] = []
    filter_center: list[float] = []
    filter_sigma: list[float] = []
    names: list[str | None] = []
    for obs in observations:
        if obs.wavelength is None:
            raise ValueError(
                "All observations must define wavelength for spectral cube coaddition"
            )
        center = float(obs.wavelength)
        if not torch.isfinite(torch.tensor(center, dtype=torch.float64)):
            raise ValueError("Observation wavelengths must be finite")
        if (obs.spectral_min is None) != (obs.spectral_max is None):
            raise ValueError(
                "Observations must define both spectral_min and spectral_max together"
            )
        band_lower = center if obs.spectral_min is None else float(obs.spectral_min)
        band_upper = center if obs.spectral_max is None else float(obs.spectral_max)
        if not torch.isfinite(
            torch.tensor(band_lower, dtype=torch.float64)
        ) or not torch.isfinite(torch.tensor(band_upper, dtype=torch.float64)):
            raise ValueError("Observation spectral bounds must be finite")
        if band_upper < band_lower:
            raise ValueError("Observation spectral_max must be >= spectral_min")
        filt_center = (
            center
            if obs.transmission_center is None
            else float(obs.transmission_center)
        )
        if not torch.isfinite(torch.tensor(filt_center, dtype=torch.float64)):
            raise ValueError("Observation transmission_center must be finite")
        filt_sigma = (
            float("nan")
            if obs.transmission_sigma is None
            else float(obs.transmission_sigma)
        )
        if obs.transmission_sigma is not None:
            if (
                not torch.isfinite(torch.tensor(filt_sigma, dtype=torch.float64))
                or filt_sigma <= 0.0
            ):
                raise ValueError(
                    "Observation transmission_sigma must be finite and > 0"
                )
        centers.append(center)
        lower.append(band_lower)
        upper.append(band_upper)
        filter_center.append(filt_center)
        filter_sigma.append(filt_sigma)
        names.append(obs.band_name)
    return (
        torch.tensor(centers, dtype=torch.float64),
        torch.tensor(lower, dtype=torch.float64),
        torch.tensor(upper, dtype=torch.float64),
        torch.tensor(filter_center, dtype=torch.float64),
        torch.tensor(filter_sigma, dtype=torch.float64),
        tuple(names),
    )


def _validated_spectral_bin_edges(edges: Tensor, *, n_band: int) -> Tensor:
    values = edges.to(dtype=torch.float64)
    if values.ndim != 1 or values.numel() != n_band + 1:
        raise ValueError("spectral_bin_edges must have length n_band + 1")
    if not bool(torch.isfinite(values).all()):
        raise ValueError("spectral_bin_edges must be finite")
    if not bool(torch.all(values[1:] > values[:-1])):
        raise ValueError("spectral_bin_edges must be strictly increasing")
    return values


def _normal_cdf(x: Tensor) -> Tensor:
    return 0.5 * (1.0 + torch.erf(x / math.sqrt(2.0)))


def _build_gaussian_filter_projection(
    *,
    lower: Tensor,
    upper: Tensor,
    center: float,
    sigma: float,
) -> Tensor:
    widths = max(float(sigma), torch.finfo(torch.float64).tiny)
    z_upper = (upper - float(center)) / widths
    z_lower = (lower - float(center)) / widths
    weights = _normal_cdf(z_upper) - _normal_cdf(z_lower)
    weights = torch.clamp(weights, min=0.0)
    row_sum = weights.sum()
    return weights / torch.clamp(row_sum, min=torch.finfo(weights.dtype).tiny)


def _build_spectral_projection(
    observation_center: Tensor,
    observation_lower: Tensor,
    observation_upper: Tensor,
    observation_filter_center: Tensor,
    observation_filter_sigma: Tensor,
    *,
    target_axis: SpectralAxis,
    target_edges: Tensor | None,
) -> Tensor:
    n_obs = int(observation_center.numel())
    n_band = int(target_axis.values.numel())
    projection = torch.zeros((n_obs, n_band), dtype=torch.float64)
    target_values = target_axis.values.to(dtype=torch.float64)

    if target_edges is None:
        for idx in range(n_obs):
            matches = target_values == observation_center[idx]
            if int(matches.sum().item()) != 1:
                raise ValueError(
                    "Observation wavelengths must match the planned spectral axis exactly; "
                    "provide spectral_bin_edges for interval reprojection"
                )
            projection[idx, int(torch.nonzero(matches, as_tuple=False).item())] = 1.0
        return projection

    left = target_edges[:-1]
    right = target_edges[1:]
    widths = observation_upper - observation_lower

    for idx in range(n_obs):
        if torch.isfinite(observation_filter_sigma[idx]):
            effective_left = torch.maximum(left, observation_lower[idx])
            effective_right = torch.minimum(right, observation_upper[idx])
            positive = effective_right > effective_left
            if bool(positive.any()):
                projection[idx] = _build_gaussian_filter_projection(
                    lower=effective_left,
                    upper=effective_right,
                    center=float(observation_filter_center[idx].item()),
                    sigma=float(observation_filter_sigma[idx].item()),
                )
        elif widths[idx] > 0.0:
            overlap = torch.clamp(
                torch.minimum(right, observation_upper[idx])
                - torch.maximum(left, observation_lower[idx]),
                min=0.0,
            )
            projection[idx] = overlap / widths[idx]
        else:
            center = observation_center[idx]
            in_bin = (center >= left) & (center < right)
            if center == right[-1]:
                in_bin[-1] = True
            projection[idx, in_bin] = 1.0
        if not bool((projection[idx] > 0.0).any()):
            raise ValueError(
                "Observation spectral support does not overlap the planned spectral grid"
            )

    return projection


def _band_names_from_projection(
    projection: Tensor,
    observation_names: tuple[str | None, ...],
) -> tuple[str | None, ...]:
    names: list[str | None] = []
    for band_idx in range(int(projection.shape[1])):
        band_names = {
            observation_names[obs_idx]
            for obs_idx in range(int(projection.shape[0]))
            if projection[obs_idx, band_idx] > 0.0
            and observation_names[obs_idx] is not None
        }
        names.append(next(iter(band_names)) if len(band_names) == 1 else None)
    return tuple(names)


def _build_band_metadata(
    spectral_axis: SpectralAxis,
    spectral_bin_edges: Tensor | None,
    projection: Tensor,
    observation_names: tuple[str | None, ...],
) -> BandMetadataTable:
    centers = spectral_axis.values.to(dtype=torch.float64)
    if spectral_bin_edges is None:
        lower = centers.clone()
        upper = centers.clone()
    else:
        lower = spectral_bin_edges[:-1].clone()
        upper = spectral_bin_edges[1:].clone()
    return BandMetadataTable(
        band_index=torch.arange(int(centers.numel()), dtype=torch.int64),
        center=centers,
        lower=lower,
        upper=upper,
        observation_count=(projection > 0.0).sum(dim=0).to(torch.int64),
        spectral_weight_sum=projection.sum(dim=0),
        band_names=_band_names_from_projection(projection, observation_names),
    )
