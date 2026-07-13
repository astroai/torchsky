"""Torch-native quality, inverse-variance, and confidence contracts."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag
from typing import Any, Mapping

import torch

CONTRACT_VERSION = "torchsky.quality.v1"
MASK_POLARITY = "set_means_flagged"
INVERSE_VARIANCE_SEMANTICS = "inverse_variance_per_pixel"
CONFIDENCE_SEMANTICS = "normalized_weight_0_to_1"
MAX_CONFIDENCE_SAMPLES = 100_000


class QualityBit(IntFlag):
    """Named quality conditions stored as set bits in an integer mask."""

    BAD_PIXEL = 1 << 0
    SATURATED = 1 << 1
    COSMIC_RAY = 1 << 2
    DETECTED = 1 << 3
    STREAK = 1 << 4
    INVALID_VARIANCE = 1 << 5


DEFAULT_ZERO_WEIGHT_BITS = (
    QualityBit.BAD_PIXEL
    | QualityBit.SATURATED
    | QualityBit.COSMIC_RAY
    | QualityBit.STREAK
    | QualityBit.INVALID_VARIANCE
)


@dataclass(frozen=True)
class WeightProduct:
    """Classical quality products with reserved producer metadata."""

    quality_mask: torch.Tensor
    inverse_variance: torch.Tensor
    weight: torch.Tensor
    confidence: torch.Tensor
    metadata: Mapping[str, Any]


def build_weight_product(
    inverse_variance: torch.Tensor,
    quality_mask: torch.Tensor | None = None,
    *,
    exclude_detected: bool = False,
    confidence_percentile: float = 99.0,
    producer: Mapping[str, Any] | None = None,
) -> WeightProduct:
    """Sanitize inverse variance and combine it with named quality bits.

    Invalid variance is represented by zero inverse variance and zero weight,
    and is recorded as ``INVALID_VARIANCE``.  ``producer`` reserves a stable
    artifact field for future learned mask/weight generators without changing
    downstream consumers.
    """
    ivar_raw = torch.as_tensor(inverse_variance)
    if ivar_raw.ndim == 0:
        raise ValueError("inverse_variance must be an array tensor")
    if not ivar_raw.is_floating_point():
        ivar_raw = ivar_raw.to(torch.float32)
    if not 0.0 < confidence_percentile <= 100.0:
        raise ValueError("confidence_percentile must be in (0, 100]")
    if quality_mask is None:
        mask = torch.zeros_like(ivar_raw, dtype=torch.int64)
    else:
        mask = torch.as_tensor(quality_mask, device=ivar_raw.device)
        if mask.shape != ivar_raw.shape:
            raise ValueError("quality_mask shape must match inverse_variance")
        if mask.dtype.is_floating_point:
            raise TypeError("quality_mask must have an integer dtype")
        mask = mask.to(torch.int64).clone()
    valid = torch.isfinite(ivar_raw) & (ivar_raw > 0)
    mask = mask | torch.where(
        valid, torch.zeros_like(mask), int(QualityBit.INVALID_VARIANCE)
    )
    excluded = int(DEFAULT_ZERO_WEIGHT_BITS)
    if exclude_detected:
        excluded |= int(QualityBit.DETECTED)
    usable = valid & ((mask & excluded) == 0)
    ivar = torch.where(valid, ivar_raw, torch.zeros_like(ivar_raw))
    weight = torch.where(usable, ivar, torch.zeros_like(ivar))
    confidence = torch.zeros_like(weight)
    positive = weight[weight > 0]
    if positive.numel():
        step = max(
            1, (positive.numel() + MAX_CONFIDENCE_SAMPLES - 1) // MAX_CONFIDENCE_SAMPLES
        )
        # ponytail: striding bounds quantile memory at 100k values; upgrade to a streaming estimator if needed.
        normalization = torch.quantile(positive[::step], confidence_percentile / 100.0)
        if bool(torch.isfinite(normalization) & (normalization > 0)):
            confidence = (weight / normalization).clamp(0.0, 1.0)
    metadata = {
        "contract_version": CONTRACT_VERSION,
        "mask_polarity": MASK_POLARITY,
        "inverse_variance_semantics": INVERSE_VARIANCE_SEMANTICS,
        "confidence_semantics": CONFIDENCE_SEMANTICS,
        "producer": dict(producer or {"name": "torchsky", "kind": "classical"}),
    }
    return WeightProduct(mask, ivar, weight, confidence, metadata)


def quality_bit_names(value: int | QualityBit) -> frozenset[str]:
    """Decode a bitset into stable public names."""
    integer = int(value)
    return frozenset(
        bit.name
        for bit in QualityBit
        if bit.name is not None and integer & int(bit)
    )


def quality_bits_from_names(
    names: Mapping[str, Any] | list[str] | tuple[str, ...] | set[str],
) -> QualityBit:
    """Encode stable public quality names into a bitset."""
    value = QualityBit(0)
    for name in names:
        try:
            value |= QualityBit[name]
        except KeyError as exc:
            raise ValueError(f"unknown quality-bit name: {name}") from exc
    return value
