"""Shared measurement encoders and conditional distribution primitives.

These modules do not establish source identity, calibrate probabilities, or
decide which catalogue evidence is admissible for a training task.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch
from torch import Tensor, nn


def _mask(mask: Tensor | None, shape: torch.Size, device: torch.device) -> Tensor:
    if mask is None:
        return torch.ones(shape, dtype=torch.bool, device=device)
    if mask.shape != shape or mask.dtype != torch.bool or mask.device != device:
        raise ValueError("mask must have matching shape/device and boolean dtype")
    return mask


def _positive_int(value: int, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{name} must be a positive integer")


class MaskedSetFusion(nn.Module):
    """Permutation-invariant learned pooling of ``(..., items, latent_dim)``.

    Returns ``(latent, supported)``. Empty sets have a zero latent and false
    support. Padding can contain arbitrary values, including NaNs. Attention
    weights are representation weights, not association probabilities.
    """

    def __init__(self, latent_dim: int) -> None:
        super().__init__()
        _positive_int(latent_dim, "latent_dim")
        self.latent_dim = latent_dim
        self.score = nn.Linear(latent_dim, 1)
        self.output = nn.Sequential(nn.Linear(latent_dim + 1, latent_dim), nn.SiLU())

    def forward(
        self, tokens: Tensor, mask: Tensor | None = None
    ) -> tuple[Tensor, Tensor]:
        if tokens.ndim < 2 or tokens.shape[-1] != self.latent_dim:
            raise ValueError("tokens must have shape (..., items, latent_dim)")
        if not tokens.is_floating_point() or tokens.device != self.score.weight.device:
            raise ValueError("tokens must be floating point on the model device")
        active = _mask(mask, tokens.shape[:-1], tokens.device)
        if not torch.isfinite(tokens[active]).all():
            raise ValueError("observed tokens must be finite")
        clean = torch.where(active.unsqueeze(-1), tokens, 0).to(self.score.weight.dtype)
        supported = active.any(dim=-1)
        scores = self.score(clean).squeeze(-1).masked_fill(~active, -torch.inf)
        scores = torch.where(supported.unsqueeze(-1), scores, 0)
        weights = torch.softmax(scores, dim=-1).masked_fill(~active, 0)
        pooled = (clean * weights.unsqueeze(-1)).sum(dim=-2)
        count = active.sum(dim=-1, keepdim=True).to(pooled.dtype).log1p()
        latent = self.output(torch.cat((pooled, count), dim=-1))
        return latent.masked_fill(~supported.unsqueeze(-1), 0), supported


class PhotometryEncoder(nn.Module):
    """Shared measurement-token encoder with explicit vocabulary and scaling.

    All inputs have shape ``(..., measurements)``. Flux and ``flux_scale`` use
    the same declared unit per passband. Negative measured flux is valid.
    ``status=0`` denotes a detection and ``status=1`` an upper-limit threshold;
    neither is padding. ``error=None`` or NaN means unknown uncertainty. Known
    errors must be finite and positive. Epochs use decimal years with NaN/None
    meaning unknown. Vocabulary IDs are caller-owned nonnegative integers.

    Returns a latent of shape ``(..., latent_dim)`` and a boolean support mask.
    No absolute sky position enters this content encoder. Scaling buffers are
    saved in ``state_dict``; save the corresponding vocabulary alongside it.
    """

    flux_scale: Tensor
    epoch_origin: Tensor
    epoch_scale: Tensor

    def __init__(
        self,
        num_passbands: int,
        num_surveys: int,
        *,
        flux_scale: Sequence[float] | Tensor,
        num_methods: int = 1,
        latent_dim: int = 64,
        epoch_origin: float = 2000.0,
        epoch_scale: float = 100.0,
    ) -> None:
        super().__init__()
        for name, value in (
            ("num_passbands", num_passbands),
            ("num_surveys", num_surveys),
            ("num_methods", num_methods),
            ("latent_dim", latent_dim),
        ):
            _positive_int(value, name)
        scale = (
            torch.as_tensor(flux_scale, dtype=torch.get_default_dtype())
            .detach()
            .clone()
        )
        if (
            scale.shape != (num_passbands,)
            or not torch.isfinite(scale).all()
            or (scale <= 0).any()
        ):
            raise ValueError(
                "flux_scale must contain one finite positive value per passband"
            )
        if not math.isfinite(epoch_origin):
            raise ValueError("epoch_origin must be finite")
        if not math.isfinite(epoch_scale) or epoch_scale <= 0:
            raise ValueError("epoch_scale must be finite and positive")
        self.register_buffer("flux_scale", scale)
        self.register_buffer("epoch_origin", torch.tensor(float(epoch_origin)))
        self.register_buffer("epoch_scale", torch.tensor(float(epoch_scale)))
        self.passband_embedding = nn.Embedding(num_passbands, latent_dim)
        self.survey_embedding = nn.Embedding(num_surveys, latent_dim)
        self.method_embedding = nn.Embedding(num_methods, latent_dim)
        self.status_embedding = nn.Embedding(2, latent_dim)
        self.token_encoder = nn.Sequential(
            nn.Linear(latent_dim + 5, latent_dim),
            nn.SiLU(),
            nn.Linear(latent_dim, latent_dim),
        )
        self.fusion = MaskedSetFusion(latent_dim)

    def forward(
        self,
        flux: Tensor,
        error: Tensor | None,
        passband_id: Tensor,
        survey_id: Tensor,
        *,
        mask: Tensor | None = None,
        method_id: Tensor | None = None,
        status: Tensor | None = None,
        epoch: Tensor | None = None,
    ) -> tuple[Tensor, Tensor]:
        if flux.ndim < 1 or not flux.is_floating_point():
            raise ValueError(
                "flux must be floating point with shape (..., measurements)"
            )
        if flux.device != self.flux_scale.device:
            raise ValueError("flux must be on the model device")
        active = _mask(mask, flux.shape, flux.device)
        if not torch.isfinite(flux[active]).all():
            raise ValueError("observed flux must be finite")

        embedded: list[Tensor] = []
        ids: list[Tensor] = []
        for name, value, embedding in (
            ("passband_id", passband_id, self.passband_embedding),
            ("survey_id", survey_id, self.survey_embedding),
            ("method_id", method_id, self.method_embedding),
            ("status", status, self.status_embedding),
        ):
            if value is None:
                value = torch.zeros_like(flux, dtype=torch.int64)
            if value.shape != flux.shape or value.device != flux.device:
                raise ValueError(f"{name} must have the same shape/device as flux")
            if value.dtype not in (
                torch.int8,
                torch.int16,
                torch.int32,
                torch.int64,
                torch.uint8,
            ):
                raise ValueError(f"{name} must contain integer vocabulary IDs")
            if (
                (value[active] < 0) | (value[active] >= embedding.num_embeddings)
            ).any():
                raise ValueError(f"{name} contains an out-of-range vocabulary ID")
            clean_id = value.masked_fill(~active, 0).long()
            ids.append(clean_id)
            embedded.append(embedding(clean_id))

        optional: list[Tensor] = []
        known: list[Tensor] = []
        for name, value in (("error", error), ("epoch", epoch)):
            if value is None:
                value = torch.full_like(flux, torch.nan)
            if (
                value.shape != flux.shape
                or value.device != flux.device
                or not value.is_floating_point()
            ):
                raise ValueError(
                    f"{name} must be floating point with the same shape/device as flux"
                )
            if torch.isinf(value[active]).any():
                raise ValueError(f"{name} must be finite or NaN for unknown values")
            available = active & torch.isfinite(value)
            if name == "error" and (value[available] <= 0).any():
                raise ValueError("known error must be positive")
            optional.append(torch.where(available, value, 0).to(self.flux_scale.dtype))
            known.append(available)

        scale = self.flux_scale[ids[0]]
        clean_flux = torch.where(active, flux, 0).to(scale.dtype)
        error_value, epoch_value = optional
        error_known, epoch_known = known
        epoch_value = torch.where(
            epoch_known, (epoch_value - self.epoch_origin) / self.epoch_scale, 0
        )
        continuous = torch.stack(
            (
                torch.asinh(clean_flux / scale),
                torch.log1p(error_value / scale),
                error_known.to(scale.dtype),
                epoch_value,
                epoch_known.to(scale.dtype),
            ),
            dim=-1,
        )
        metadata = torch.stack(embedded).sum(dim=0)
        tokens = self.token_encoder(torch.cat((continuous, metadata), dim=-1))
        return self.fusion(tokens, active)


class BinnedRedshiftHead(nn.Module):
    """Conditional logits with a piecewise-uniform density on explicit z bins.

    Output logits represent bin probability masses; ``log_prob`` includes bin
    width, so nonuniform bins still define a normalized density. Targets outside
    the closed support have log probability ``-inf``. Applications must apply
    their evidence/support policy; passing ``supported`` rejects empty objects.
    Training does not by itself establish probability calibration.
    """

    bin_edges: Tensor

    def __init__(self, latent_dim: int, bin_edges: Sequence[float] | Tensor) -> None:
        super().__init__()
        _positive_int(latent_dim, "latent_dim")
        edges = (
            torch.as_tensor(bin_edges, dtype=torch.get_default_dtype()).detach().clone()
        )
        if (
            edges.ndim != 1
            or edges.numel() < 2
            or not torch.isfinite(edges).all()
            or (edges.diff() <= 0).any()
        ):
            raise ValueError("bin_edges must be a finite, strictly increasing vector")
        self.register_buffer("bin_edges", edges)
        self.linear = nn.Linear(latent_dim, edges.numel() - 1)

    def forward(self, latent: Tensor, *, supported: Tensor | None = None) -> Tensor:
        if latent.ndim < 1 or latent.shape[-1] != self.linear.in_features:
            raise ValueError("latent has an incompatible shape")
        if not latent.is_floating_point() or not torch.isfinite(latent).all():
            raise ValueError("latent must be finite floating point")
        if latent.device != self.linear.weight.device:
            raise ValueError("latent and model parameters must share a device")
        if supported is not None:
            support = _mask(supported, latent.shape[:-1], latent.device)
            if not support.all():
                raise ValueError("cannot predict for unsupported objects")
        return self.linear(latent)

    def log_prob(self, logits: Tensor, redshift: Tensor) -> Tensor:
        if (
            logits.ndim < 1
            or logits.shape[-1] != self.linear.out_features
            or redshift.shape != logits.shape[:-1]
        ):
            raise ValueError("logits/redshift have an incompatible shape")
        if (
            not logits.is_floating_point()
            or not redshift.is_floating_point()
            or not torch.isfinite(logits).all()
            or not torch.isfinite(redshift).all()
        ):
            raise ValueError("logits and redshift must be finite floating point")
        if logits.device != self.bin_edges.device or redshift.device != logits.device:
            raise ValueError("logits, redshift and bin_edges must share a device")
        edges = self.bin_edges.to(redshift.dtype)
        index = torch.bucketize(redshift.contiguous(), edges, right=True) - 1
        inside = (redshift >= edges[0]) & (redshift <= edges[-1])
        index = index.clamp(0, self.linear.out_features - 1)
        log_mass = (
            torch.log_softmax(logits, dim=-1)
            .gather(-1, index.unsqueeze(-1))
            .squeeze(-1)
        )
        log_width = edges.diff()[index].log().to(log_mass.dtype)
        log_density = log_mass - log_width
        return log_density.masked_fill(~inside, -torch.inf)


def marginalize_association_logits(
    logits: Tensor,
    association_weights: Tensor,
    hypothesis_mask: Tensor | None = None,
) -> Tensor:
    """Mix conditional bin masses over mutually exclusive association hypotheses.

    Inputs have shapes ``(..., hypotheses, bins)`` and ``(..., hypotheses)``;
    returns normalized log bin masses ``(..., bins)``. Nonnegative input weights
    are normalized over the supplied hypotheses. Include a no-match hypothesis
    when applicable: omitted hypotheses produce a conditional distribution.
    This algebra does not turn matcher ranking scores into calibrated weights.
    """
    if (
        logits.ndim < 2
        or logits.shape[-1] == 0
        or association_weights.shape != logits.shape[:-1]
    ):
        raise ValueError("logits/association_weights have an incompatible shape")
    if (
        not logits.is_floating_point()
        or not association_weights.is_floating_point()
        or logits.device != association_weights.device
    ):
        raise ValueError(
            "logits/association_weights must be floating point on the same device"
        )
    active = _mask(hypothesis_mask, association_weights.shape, logits.device)
    weights = association_weights.masked_fill(~active, 0)
    if not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("association_weights must be finite and nonnegative")
    nonzero = weights > 0
    valid_hyp = active & torch.isfinite(logits).all(dim=-1)
    if (nonzero & ~valid_hyp).any():
        raise ValueError("positive-weight hypothesis logits must be finite")
    if not nonzero.any(dim=-1).all():
        raise ValueError("each object needs positive association weight")
    clean_logits = torch.where(valid_hyp.unsqueeze(-1), logits, 0)
    log_weights = (
        weights.masked_fill(~nonzero, 1).log().masked_fill(~nonzero, -torch.inf)
    )
    log_weights = log_weights - torch.logsumexp(log_weights, dim=-1, keepdim=True)
    log_mass = torch.log_softmax(clean_logits, dim=-1)
    out_lse = torch.logsumexp(log_mass + log_weights.detach().unsqueeze(-1), dim=-2)
    norm_weights = weights / weights.sum(dim=-1, keepdim=True)
    ratio = torch.where(
        valid_hyp.unsqueeze(-1),
        (log_mass.detach() - out_lse.detach().unsqueeze(-2)).exp(),
        0,
    )
    weight_term = ((norm_weights - norm_weights.detach()).unsqueeze(-1) * ratio).sum(
        dim=-2
    )
    return out_lse + weight_term


__all__ = [
    "BinnedRedshiftHead",
    "MaskedSetFusion",
    "PhotometryEncoder",
    "marginalize_association_logits",
]
