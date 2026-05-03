"""Implicit forward-modeling shell aligned with ``docs/implicit_forward.md``.

This module is the **code counterpart** to the architectural note: shared scene INR
``F_θ``, optional kinematics ``ζ``, context ``c``, and instrument-side rendering
(``torchsky.map.field.core``). It does **not** replace :func:`fit_field_model`; it
composes the same render path with explicit **rest-frame spectral queries** and a
**multi-survey loss** scaffold (§8 modular ``Σ_s L_s``).

Design goals:

- Keep the public surface **small** and **tensor-native**.
- Avoid duplicating physics inside this file—only **coordinate remaps** and **calls**
  to :func:`render_field_observation`.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

import torch
from torch import Tensor, nn

from .core import render_field_observation
from .types import FieldObservation, ForwardContext


@dataclass(frozen=True)
class KinematicState:
    """Scalar spectral kinematics ``ζ`` for rest ↔ observed wavelength (§4.2).

    ``one_plus_z`` encodes **observed / rest** wavelength ratio:

    ``λ_obs = one_plus_z * λ_emit`` (non-relativistic Doppler: ``one_plus_z ≈ 1 + v/c``).

    Set ``one_plus_z=1.0`` or ``None`` for identity. No cosmology is implied—only a
    user-chosen linear map on the spectral axis.
    """

    one_plus_z: float | None = None

    def effective_one_plus_z(self) -> float:
        z = self.one_plus_z
        if z is None:
            return 1.0
        if not math.isfinite(z) or z <= 0.0:
            raise ValueError("KinematicState.one_plus_z must be finite and positive")
        return float(z)


@dataclass(frozen=True)
class SurveyDataTerm:
    """One modular likelihood term ``L_s`` (``implicit_forward.md`` §8.2)."""

    name: str
    observations: tuple[FieldObservation, ...]
    loss_weight: float = 1.0
    kinematics_per_observation: tuple[KinematicState | None, ...] | None = None
    forward_context_per_observation: tuple[ForwardContext | None, ...] | None = None

    def __post_init__(self) -> None:
        if self.loss_weight < 0.0:
            raise ValueError("loss_weight must be non-negative")
        if self.kinematics_per_observation is not None:
            if len(self.kinematics_per_observation) != len(self.observations):
                raise ValueError(
                    "kinematics_per_observation length must match observations"
                )
        if self.forward_context_per_observation is not None:
            if len(self.forward_context_per_observation) != len(self.observations):
                raise ValueError(
                    "forward_context_per_observation length must match observations"
                )


def scale_observation_for_rest_frame_scene(
    observation: FieldObservation,
    *,
    one_plus_z: float,
) -> FieldObservation:
    """Map **observed** band limits to **rest** coordinates for querying a rest-frame INR.

    If the scene models ``F(λ_emit)`` and the observation carries ``λ_obs`` bounds,
    pass ``one_plus_z = λ_obs / λ_emit`` so internal queries use
    ``λ_emit = λ_obs / one_plus_z``.
    """
    if not math.isfinite(one_plus_z) or one_plus_z <= 0.0:
        raise ValueError("one_plus_z must be finite and positive")
    inv = 1.0 / float(one_plus_z)
    tc = observation.transmission_center
    ts = observation.transmission_sigma
    twl = observation.transmission_wavelength
    ttt = observation.transmission_throughput
    twl_rest = None if twl is None else (twl * inv)
    return FieldObservation(
        image=observation.image,
        header=dict(observation.header),
        weight=observation.weight,
        mask=observation.mask,
        wavelength=(
            None
            if observation.wavelength is None
            else float(observation.wavelength) * inv
        ),
        spectral_min=(
            None
            if observation.spectral_min is None
            else float(observation.spectral_min) * inv
        ),
        spectral_max=(
            None
            if observation.spectral_max is None
            else float(observation.spectral_max) * inv
        ),
        transmission_center=None if tc is None else float(tc) * inv,
        transmission_sigma=None if ts is None else float(ts) * inv,
        transmission_wavelength=twl_rest,
        transmission_throughput=ttt,
        psf_sigma_pix=float(observation.psf_sigma_pix),
        psf_spec=observation.psf_spec,
        background_offset=float(observation.background_offset),
        band_name=observation.band_name,
        spectral_quantity=observation.spectral_quantity,
    )


def predict_observation_mean(
    model: nn.Module,
    observation: FieldObservation,
    *,
    kinematics: KinematicState | None = None,
    rest_frame_scene: bool = False,
    forward_context: ForwardContext | None = None,
    **render_kw: Any,
) -> Tensor:
    """Predict mean sky patch for one :class:`FieldObservation` (deterministic ``G_ψ`` path).

    Wraps :func:`render_field_observation`. When ``rest_frame_scene`` is True and
    ``kinematics.one_plus_z`` is set, spectral metadata is scaled with
    :func:`scale_observation_for_rest_frame_scene` before rendering.

    ``forward_context`` is forwarded to :func:`render_field_observation` for models that
    accept ``forward_context`` in ``forward`` / ``render_bandpass``.
    """
    obs = observation
    if rest_frame_scene:
        opz = kinematics.effective_one_plus_z() if kinematics is not None else 1.0
        if opz != 1.0:
            obs = scale_observation_for_rest_frame_scene(observation, one_plus_z=opz)
    return render_field_observation(
        model, obs, forward_context=forward_context, **render_kw
    )


def _charbonnier(x: Tensor, *, eps: float) -> Tensor:
    return torch.sqrt(torch.square(x) + float(eps) ** 2) - float(eps)


def observation_charbonnier_loss(
    prediction: Tensor,
    observation: FieldObservation,
    *,
    eps: float = 1.0e-3,
) -> Tensor:
    """Per-observation Charbonnier loss with optional ``weight`` and ``mask``."""
    diff = prediction - observation.image
    rho = _charbonnier(diff, eps=eps)
    m = (
        torch.ones_like(diff, dtype=rho.dtype)
        if observation.mask is None
        else observation.mask.to(dtype=rho.dtype)
    )
    w = (
        torch.ones_like(diff, dtype=rho.dtype)
        if observation.weight is None
        else observation.weight.to(dtype=rho.dtype)
    )
    wm = (w * m).clamp_min(0.0)
    return (rho * wm).sum() / wm.sum().clamp_min(1.0e-8)


def multi_survey_charbonnier_loss(
    model: nn.Module,
    terms: Sequence[SurveyDataTerm],
    *,
    rest_frame_scene: bool = False,
    charbonnier_eps: float = 1.0e-3,
    **render_kw: Any,
) -> Tensor:
    """Sum weighted Charbonnier losses ``Σ_s w_s L_s`` over survey terms (§8.2 scaffold).

    Each term renders predictions with :func:`predict_observation_mean` and accumulates
    a scalar loss. Different surveys can share the same ``model`` (shared ``θ``) while
    using different observations and optional per-observation kinematics.
    """
    total = None
    for term in terms:
        kin_list = term.kinematics_per_observation
        ctx_list = term.forward_context_per_observation
        for i, obs in enumerate(term.observations):
            kin = None if kin_list is None else kin_list[i]
            fctx = None if ctx_list is None else ctx_list[i]
            pred = predict_observation_mean(
                model,
                obs,
                kinematics=kin,
                rest_frame_scene=rest_frame_scene,
                forward_context=fctx,
                **render_kw,
            )
            li = observation_charbonnier_loss(
                pred,
                obs,
                eps=charbonnier_eps,
            )
            weighted = float(term.loss_weight) * li
            total = weighted if total is None else total + weighted
    if total is None:
        raise ValueError(
            "multi_survey_charbonnier_loss requires at least one observation"
        )
    return total
