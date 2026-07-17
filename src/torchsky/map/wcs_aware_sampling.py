"""Band-balanced sampling for WCS-aware multi-band INR training."""

from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def _balanced_counts(
    total: int,
    groups: int,
    *,
    generator: torch.Generator,
) -> list[int]:
    if groups <= 0:
        raise ValueError("groups must be positive")
    base, remainder = divmod(int(total), int(groups))
    counts = torch.full((groups,), base, dtype=torch.long)
    if remainder:
        order = torch.randperm(groups, generator=generator)
        counts[order[:remainder]] += 1
    return [int(value) for value in counts.tolist()]


def sample_band_balanced_batch(
    pools: Sequence[object],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Draw an exact-size batch balanced first by band, then by exposure.

    Each distinct band receives an equal share of the requested batch budget.
    Within a band, that share is divided as evenly as possible across its
    exposure pools. Remainders are randomized so no fixed band or exposure is
    systematically favored.
    """

    if not pools:
        raise ValueError("at least one exposure pool is required")

    band_order = list(dict.fromkeys(str(pool.band) for pool in pools))
    if batch_size < len(band_order):
        raise ValueError("batch_size must be at least the number of bands")

    indices_by_band: dict[str, list[int]] = {band: [] for band in band_order}
    for index, pool in enumerate(pools):
        if int(pool.size) <= 0:
            raise ValueError(f"exposure pool {index} is empty")
        indices_by_band[str(pool.band)].append(index)

    band_counts = _balanced_counts(
        batch_size,
        len(band_order),
        generator=generator,
    )

    uv: list[Tensor] = []
    exposure_id: list[Tensor] = []
    value: list[Tensor] = []
    err: list[Tensor] = []

    for band, band_count in zip(band_order, band_counts):
        exposure_indices = indices_by_band[band]
        exposure_counts = _balanced_counts(
            band_count,
            len(exposure_indices),
            generator=generator,
        )
        for index, count in zip(exposure_indices, exposure_counts):
            if count == 0:
                continue
            pool = pools[index]
            chosen = torch.randint(0, int(pool.size), (count,), generator=generator)
            uv.append(pool.uv[chosen])
            value.append(pool.value[chosen])
            err.append(pool.err[chosen])
            exposure_id.append(torch.full((count,), index, dtype=torch.long))

    uv_t = torch.cat(uv)
    exposure_t = torch.cat(exposure_id)
    value_t = torch.cat(value)
    err_t = torch.cat(err)
    if int(uv_t.shape[0]) != int(batch_size):
        raise RuntimeError(
            f"band-balanced sampler produced {uv_t.shape[0]} samples, "
            f"expected {batch_size}"
        )

    order = torch.randperm(batch_size, generator=generator)
    uv_t = uv_t[order]
    exposure_t = exposure_t[order]
    value_t = value_t[order]
    err_t = err_t[order]
    mask = torch.ones_like(value_t, dtype=torch.bool)
    return uv_t, exposure_t, value_t, err_t, mask


__all__ = ["sample_band_balanced_batch"]
