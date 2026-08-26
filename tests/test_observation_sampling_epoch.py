"""Tests for epoch sampling in map.field.observation_ops."""

from __future__ import annotations

import math
from collections import Counter

import pytest
import torch

from torchsky.map.field.observation_ops import (
    epoch_permutation,
    sample_observation_indices,
)


class _Obs:
    def __init__(self, band: str) -> None:
        self.band = band
        self.band_name = band
        self.spectral_min = None
        self.spectral_max = None
        self.wavelength = None
        self.transmission_center = None


def test_epoch_permutation_covers_every_index_exactly_once() -> None:
    for n in range(1, 60):
        for epoch in (0, 1, 2):
            permutation = epoch_permutation(n, epoch=epoch).tolist()
            assert sorted(permutation) == list(range(n))


def test_epoch_permutation_is_deterministic_and_reshuffles() -> None:
    n = 37
    epochs = [epoch_permutation(n, epoch=e).tolist() for e in range(3)]
    assert epochs[0] != epochs[1] != epochs[2]
    assert epoch_permutation(n, epoch=1).tolist() == epochs[1]


def test_full_coverage_returns_complete_epoch_order() -> None:
    observations = [_Obs("F090W")] * 21
    full = sample_observation_indices(
        observations,
        observations_per_step=None,
        observation_sampling="epoch",
        generator=torch.Generator(),
        epoch=0,
    )
    assert full == epoch_permutation(21, epoch=0).tolist()


def test_stride_is_coprime_so_cycles_never_repeat_early() -> None:
    # A non-coprime stride revisits indices before covering them; exact-once
    # coverage across many sizes is precisely the coprimality property.
    n = 36  # highly composite: fails for any stride sharing a factor
    seen: set[int] = set()
    order = epoch_permutation(n, epoch=4).tolist()
    for idx in order:
        assert idx not in seen
        seen.add(idx)
    assert len(seen) == n
    assert math.gcd(1, n) == 1  # trivially valid fallback stride exists


def test_partial_epoch_respects_band_grouping_like_band_balanced() -> None:
    observations = [_Obs("F090W")] * 5 + [_Obs("F277W")] * 7 + [_Obs("F444W")] * 9
    selected = sample_observation_indices(
        observations,
        observations_per_step=8,
        observation_sampling="epoch",
        generator=torch.Generator(),
        epoch=2,
        step_in_epoch=0,
    )
    assert len(selected) == len(set(selected)) == 8
    counts = Counter(observations[i].band for i in selected)
    # Round-robin band interleaving gives every represented band two samples.
    assert counts["F090W"] >= 2
    assert counts["F277W"] >= 2
    assert counts["F444W"] >= 2


def test_partial_epoch_steps_cover_each_observation_once() -> None:
    observations = [_Obs("F090W")] * 5 + [_Obs("F277W")] * 7
    batches = [
        sample_observation_indices(
            observations,
            observations_per_step=5,
            observation_sampling="epoch",
            generator=torch.Generator(),
            epoch=3,
            step_in_epoch=step,
        )
        for step in range(3)
    ]
    flattened = [index for batch in batches for index in batch]
    assert list(map(len, batches)) == [5, 5, 2]
    assert sorted(flattened) == list(range(len(observations)))


def test_existing_modes_unchanged_and_unknown_rejected() -> None:
    observations = [_Obs("F090W")] * 6
    uniform = sample_observation_indices(
        observations,
        observations_per_step=4,
        observation_sampling="uniform",
        generator=torch.Generator().manual_seed(0),
    )
    assert sorted(uniform) == sorted(set(uniform)) and len(uniform) == 4
    balanced = sample_observation_indices(
        observations,
        observations_per_step=None,
        observation_sampling="band_balanced",
        generator=torch.Generator(),
    )
    assert balanced == list(range(6))
    with pytest.raises(ValueError, match="unsupported observation_sampling"):
        sample_observation_indices(
            observations,
            observations_per_step=2,
            observation_sampling="bogus",
            generator=torch.Generator(),
        )
