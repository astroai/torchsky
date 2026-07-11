"""Analytic checks for the torchsky quality contract."""

from __future__ import annotations

import pytest
import torch

from torchsky.quality import (
    CONTRACT_VERSION,
    MASK_POLARITY,
    QualityBit,
    build_weight_product,
    quality_bit_names,
    quality_bits_from_names,
)


def test_invalid_variance_is_zero_weight_and_flagged() -> None:
    ivar = torch.tensor([[1.0, 0.0], [float("nan"), 4.0]])
    product = build_weight_product(ivar)
    invalid = ~torch.isfinite(ivar) | (ivar <= 0)
    assert torch.all(product.inverse_variance[invalid] == 0)
    assert torch.all(product.weight[invalid] == 0)
    assert torch.all(product.quality_mask[invalid] & int(QualityBit.INVALID_VARIANCE))
    assert product.metadata["contract_version"] == CONTRACT_VERSION
    assert product.metadata["mask_polarity"] == MASK_POLARITY


def test_detected_bit_is_informational_by_default_and_excludable() -> None:
    mask = torch.tensor([int(QualityBit.DETECTED), 0])
    ivar = torch.ones(2)
    keep = build_weight_product(ivar, mask)
    drop = build_weight_product(ivar, mask, exclude_detected=True)
    assert keep.weight[0] == 1
    assert drop.weight[0] == 0


def test_quality_names_round_trip() -> None:
    bits = QualityBit.BAD_PIXEL | QualityBit.COSMIC_RAY | QualityBit.STREAK
    names = quality_bit_names(bits)
    assert names == {"BAD_PIXEL", "COSMIC_RAY", "STREAK"}
    assert quality_bits_from_names(names) == bits


def test_producer_metadata_is_reserved_for_future_ml() -> None:
    product = build_weight_product(
        torch.ones(3), producer={"name": "future-model", "kind": "ml"}
    )
    assert product.metadata["producer"]["kind"] == "ml"
    assert product.confidence.min() == pytest.approx(1.0)
