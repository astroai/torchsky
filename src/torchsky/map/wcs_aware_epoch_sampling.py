"""Without-replacement epoch sampling for WCS-aware INR exposure pools.

The historical sampler draws with replacement on every optimization step.  This
module instead walks through a shuffled permutation of every loaded pool: every
pixel is used exactly once before that exposure starts a new epoch.

A full ``torch.randperm`` for every native exposure would add several gigabytes
of host memory for the SMACS J0723 run.  We therefore use an affine permutation

    index(k) = (offset + stride * k) mod N,

with ``gcd(stride, N) = 1``.  It is a true permutation of ``0, ..., N - 1``, is
reshuffled at every epoch boundary, and needs only O(1) state per exposure.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import torch
from torch import Tensor

from .wcs_aware_training import ExposurePool, SHORT_WAVE_BANDS


BatchSpec = int | Sequence[int]


@dataclass
class _ExposureCycle:
    """State of one exposure's current without-replacement pass."""

    size: int
    offset: int = 0
    stride: int = 1
    cursor: int = 0
    epoch: int = -1


class EpochBalancedBatchSampler:
    """Balanced mini-batches that exhaust each exposure before reshuffling.

    ``batch_size`` keeps the existing command-line meaning:

    * one integer: exact total batch size, distributed as evenly as possible
      over exposures;
    * two integers: exact short-wave and long-wave samples *per exposure*.

    Different exposure sizes and SW/LW batch counts are allowed, so each
    exposure has its own epoch counter and reshuffles independently when its
    own valid-pixel pool is exhausted.
    """

    def __init__(
        self,
        pools: Sequence[ExposurePool],
        *,
        batch_size: BatchSpec,
        generator: torch.Generator,
    ) -> None:
        if not pools:
            raise ValueError("at least one exposure pool is required")

        self.pools = tuple(pools)
        self.generator = generator
        self.batch_size = self._normalize_batch_spec(batch_size)
        self._cycles = [_ExposureCycle(pool.size) for pool in self.pools]
        self._remainder_start = 0

        for index, cycle in enumerate(self._cycles):
            if cycle.size <= 0:
                raise ValueError(f"exposure {index} has no training pixels")
            self._start_epoch(index)

    @staticmethod
    def _normalize_batch_spec(batch_size: BatchSpec) -> int | tuple[int, int]:
        if isinstance(batch_size, int):
            if batch_size <= 0:
                raise ValueError("batch_size must be positive")
            return int(batch_size)

        values = tuple(int(value) for value in batch_size)
        if len(values) != 2 or min(values) <= 0:
            raise ValueError(
                "batch_size must be one total or two SW/LW positive integers"
            )
        return values

    def _random_int(self, low: int, high: int) -> int:
        """Draw one deterministic CPU integer from ``[low, high)``."""

        if high <= low:
            return low
        return int(
            torch.randint(
                low,
                high,
                (1,),
                generator=self.generator,
                device="cpu",
            ).item()
        )

    def _coprime_stride(self, size: int) -> int:
        if size <= 1:
            return 1
        while True:
            candidate = self._random_int(1, size)
            if math.gcd(candidate, size) == 1:
                return candidate

    def _start_epoch(self, exposure_index: int) -> None:
        cycle = self._cycles[exposure_index]
        cycle.offset = self._random_int(0, cycle.size)
        cycle.stride = self._coprime_stride(cycle.size)
        cycle.cursor = 0
        cycle.epoch += 1

    def _take_indices(self, exposure_index: int, count: int) -> Tensor:
        """Take ``count`` indices, crossing/reshuffling epochs if necessary."""

        if count <= 0:
            return torch.empty(0, dtype=torch.long)

        cycle = self._cycles[exposure_index]
        pieces: list[Tensor] = []
        remaining = int(count)

        while remaining > 0:
            if cycle.cursor == cycle.size:
                self._start_epoch(exposure_index)

            take = min(remaining, cycle.size - cycle.cursor)
            logical = torch.arange(
                cycle.cursor,
                cycle.cursor + take,
                dtype=torch.int64,
                device="cpu",
            )
            physical = (cycle.offset + cycle.stride * logical).remainder(cycle.size)
            pieces.append(physical)
            cycle.cursor += take
            remaining -= take

        return pieces[0] if len(pieces) == 1 else torch.cat(pieces)

    def _counts_for_step(self) -> list[int]:
        if not isinstance(self.batch_size, int):
            short_count, long_count = self.batch_size
            return [
                short_count if pool.band in SHORT_WAVE_BANDS else long_count
                for pool in self.pools
            ]

        total = self.batch_size
        exposure_count = len(self.pools)
        if total < exposure_count:
            raise ValueError("batch_size must be at least the number of exposures")

        base, remainder = divmod(total, exposure_count)
        counts = [base] * exposure_count

        # Rotate the +1 assignments so the same early exposures are not always
        # favored when total_batch is not divisible by number_of_exposures.
        for offset in range(remainder):
            counts[(self._remainder_start + offset) % exposure_count] += 1
        self._remainder_start = (
            self._remainder_start + remainder
        ) % exposure_count
        return counts

    def sample(self) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        uv: list[Tensor] = []
        exposure_id: list[Tensor] = []
        value: list[Tensor] = []
        err: list[Tensor] = []

        for index, (pool, count) in enumerate(
            zip(self.pools, self._counts_for_step(), strict=True)
        ):
            chosen = self._take_indices(index, count)
            uv.append(pool.uv[chosen])
            value.append(pool.value[chosen])
            err.append(pool.err[chosen])
            exposure_id.append(torch.full((count,), index, dtype=torch.long))

        uv_t = torch.cat(uv)
        exposure_t = torch.cat(exposure_id)
        value_t = torch.cat(value)
        err_t = torch.cat(err)

        # Mixing exposure blocks changes no coverage accounting and keeps the
        # mini-batch order stochastic for kernels that are order-sensitive.
        order = torch.randperm(uv_t.shape[0], generator=self.generator)
        uv_t = uv_t[order]
        exposure_t = exposure_t[order]
        value_t = value_t[order]
        err_t = err_t[order]

        return (
            uv_t,
            exposure_t,
            value_t,
            err_t,
            torch.ones_like(value_t, dtype=torch.bool),
        )

    def state_summary(self) -> list[dict[str, int]]:
        """Small diagnostic snapshot; no pixel permutations are materialized."""

        return [
            {
                "exposure": index,
                "size": cycle.size,
                "epoch": cycle.epoch,
                "cursor": cycle.cursor,
            }
            for index, cycle in enumerate(self._cycles)
        ]


# ``train_model`` creates one CPU generator per run and repeatedly calls the
# module-level sampling function.  The cache gives that unchanged training loop
# a persistent epoch cursor after this function is installed as a monkey patch.
_SAMPLER_CACHE: dict[
    int,
    tuple[
        torch.Generator,
        tuple[int, ...],
        int | tuple[int, int],
        EpochBalancedBatchSampler,
    ],
] = {}


def _batch_key(batch_size: BatchSpec) -> int | tuple[int, int]:
    return EpochBalancedBatchSampler._normalize_batch_spec(batch_size)


def sample_epoch_balanced_batch(
    pools: Sequence[ExposurePool],
    *,
    batch_size: BatchSpec,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Drop-in replacement for ``sample_balanced_batch``."""

    cache_key = id(generator)
    pool_signature = tuple(id(pool) for pool in pools)
    batch_key = _batch_key(batch_size)
    cached = _SAMPLER_CACHE.get(cache_key)

    if (
        cached is None
        or cached[0] is not generator
        or cached[1] != pool_signature
        or cached[2] != batch_key
    ):
        sampler = EpochBalancedBatchSampler(
            pools,
            batch_size=batch_key,
            generator=generator,
        )
        _SAMPLER_CACHE[cache_key] = (
            generator,
            pool_signature,
            batch_key,
            sampler,
        )
    else:
        sampler = cached[3]

    return sampler.sample()


def install_epoch_sampler() -> None:
    """Install the sampler into the existing training module at runtime."""

    from . import wcs_aware_training

    wcs_aware_training.sample_balanced_batch = sample_epoch_balanced_batch


__all__ = [
    "EpochBalancedBatchSampler",
    "install_epoch_sampler",
    "sample_epoch_balanced_batch",
]
