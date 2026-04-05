"""Observation-derived spectral quadrature and model-capacity suggestions.

See :mod:`torchsky.map.field.observation_builders` for binned 1D spectra, IFU cubes,
and energy-channel data.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

from .core import _bandpass_bounds
from .types import (
    FieldObservation,
    ObservationSpectralSummary,
    SpectralCapacityProfile,
    SpectralDataBudget,
    SpectralModelCapacityHints,
)


def _variation_scale(
    span: float,
    lower: float,
    transmission_sigma: float | None,
) -> float:
    if span <= 0.0:
        return max(1.0e-9 * max(abs(lower), 1.0), 1.0e-12)
    sig = transmission_sigma
    if sig is not None and math.isfinite(sig) and sig > 0.0:
        return min(float(sig), float(span))
    return float(span)


def spectral_data_budget(
    observations: Sequence[FieldObservation],
) -> SpectralDataBudget:
    if not observations:
        raise ValueError("spectral_data_budget requires at least one observation")
    summaries: list[ObservationSpectralSummary] = []
    lowers: list[float] = []
    uppers: list[float] = []
    band_keys: set[tuple[object, ...]] = set()
    for obs in observations:
        lo, up = _bandpass_bounds(obs)
        span = float(up - lo)
        sig = obs.transmission_sigma
        if sig is not None and (not math.isfinite(sig) or sig <= 0.0):
            sig = None
        vs = _variation_scale(span, lo, sig)
        summaries.append(
            ObservationSpectralSummary(
                lower=lo,
                upper=up,
                span=span,
                transmission_sigma=sig,
                variation_scale=vs,
                band_name=obs.band_name,
            )
        )
        lowers.append(lo)
        uppers.append(up)
        band_keys.add(
            (
                obs.band_name,
                round(lo, 9),
                round(up, 9),
                None if sig is None else round(sig, 9),
            )
        )
    wmin = min(lowers)
    wmax = max(uppers)
    span_global = float(wmax - wmin)
    min_vs = min(s.variation_scale for s in summaries)
    max_vs = max(s.variation_scale for s in summaries)
    return SpectralDataBudget(
        observations=tuple(summaries),
        wavelength_min=wmin,
        wavelength_max=wmax,
        wavelength_span=span_global,
        min_variation_scale=min_vs,
        max_variation_scale=max_vs,
        n_distinct_band_specs=len(band_keys),
    )


def suggest_num_spectral_samples(
    budget: SpectralDataBudget,
    *,
    spacing_to_scale_ratio: float = 0.35,
    floor: int = 3,
    cap: int = 256,
) -> int:
    """Suggest a single ``num_spectral_samples`` for :func:`fit_field_model` / rendering.

    For each observation with box span ``S`` and variation scale ``L`` (Gaussian σ
    capped by ``S``, or ``S`` for uniform-in-box), require trapezoid node spacing
    ``h ≲ spacing_to_scale_ratio * L`` across the integration interval, i.e.

    ``n ≳ ceil(S / h) + 1`` when ``S > 0``.

    Default ``spacing_to_scale_ratio=0.35`` targets on the order of three nodes per
    correlation scale (documented tradeoff, not a hard error bound).

    Delta-band observations contribute ``floor`` (minimum meaningful count).
    """
    if spacing_to_scale_ratio <= 0.0:
        raise ValueError("spacing_to_scale_ratio must be positive")
    floor_i = max(int(floor), 2)
    cap_i = max(int(cap), floor_i)
    n_need = floor_i
    for s in budget.observations:
        if s.span <= 0.0:
            # Point-like band: integrator collapses to one λ; still respect ``floor``.
            n_need = max(n_need, floor_i)
            continue
        h_max = spacing_to_scale_ratio * max(s.variation_scale, 1.0e-30)
        n_i = int(math.ceil(s.span / h_max)) + 1
        n_i = max(n_i, floor_i)
        n_need = max(n_need, n_i)
    return min(n_need, cap_i)


def suggest_spectral_model_capacity(
    budget: SpectralDataBudget,
) -> SpectralModelCapacityHints:
    """Map global span vs finest scale to encoder width and model-family hints.

    ``spectroscopicity = wavelength_span / min_variation_scale`` grows when many
    narrow bands or small σ sit inside a wide global λ range.
    """
    eps = max(1.0e-30, budget.min_variation_scale)
    R = budget.wavelength_span / eps if budget.wavelength_span > 0.0 else 1.0

    if R < 5.0 and budget.n_distinct_band_specs <= 4:
        profile: SpectralCapacityProfile = "photometric_wide"
        sf = 4
        basis_waves = 4
        kinds = (
            "sky_encoder_bandpass_field",
            "sky_encoder_mlp",
            "hashgrid_conditioned_field",
        )
        rationale = "Few wide bands and low spectroscopicity: prefer band-aggregated or compact λ encoding."
    elif R < 25.0:
        profile = "multiband"
        sf = 6
        basis_waves = 6
        kinds = (
            "sky_encoder_resmlp",
            "sky_lambda_planes_field",
            "conditioned_spectral_field",
            "sky_encoder_mlp",
        )
        rationale = "Moderate spectroscopicity: increase log-λ Fourier features or factorized sky–λ fields."
    elif R < 120.0:
        profile = "narrowband_sweep"
        sf = 8
        basis_waves = 8
        kinds = (
            "sky_lambda_feature_field",
            "sky_lambda_kplanes_field",
            "operator_constrained_feature_field",
            "spectral_basis_field",
        )
        rationale = "Many λ degrees of freedom across span: structured λ fields or explicit spectral bases."
    else:
        profile = "spectroscopic"
        sf = 11
        basis_waves = 12
        kinds = (
            "spectral_basis_field",
            "learned_spectral_basis_field",
            "sky_lambda_feature_field",
            "operator_constrained_feature_field",
        )
        rationale = "High spectroscopicity: prioritize spectral-basis and explicit multiscale λ representations."

    return SpectralModelCapacityHints(
        spectroscopicity=float(R),
        profile=profile,
        spectral_frequencies=sf,
        num_spectral_frequencies_basis=basis_waves,
        recommended_model_kinds=kinds,
        rationale=rationale,
    )
