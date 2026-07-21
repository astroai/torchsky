#!/usr/bin/env python3
"""Diagnostic source-stratified sampling for the WCS-aware INR.

This entry point is intentionally an ablation, not the final scientific
objective.  It reuses the loss variants from
``smacs0723_wcs_aware_loss_ablation.py`` and oversamples increasingly bright
native pixels within every exposure.  The purpose is to test whether the
current amplitude collapse is mainly caused by source-gradient starvation
under uniform random pixel sampling.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys
from typing import Sequence

import torch
from torch import Tensor

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

import scripts.smacs0723_wcs_aware_loss_ablation as ablation  # noqa: E402
from torchsky.map.wcs_aware_training import ExposurePool  # noqa: E402


# Cache source bins once per in-memory ExposurePool.  Pool values are already
# normalized by the common intensity scale, but quantile membership is
# invariant to that global scaling.
_BIN_CACHE: dict[int, tuple[Tensor, Tensor, Tensor, Tensor]] = {}


def _source_bins(pool: ExposurePool) -> tuple[Tensor, Tensor, Tensor, Tensor]:
    key = id(pool)
    cached = _BIN_CACHE.get(key)
    if cached is not None:
        return cached

    values = pool.value.float()
    finite = torch.isfinite(values)
    all_indices = torch.nonzero(finite, as_tuple=False).squeeze(1)
    if all_indices.numel() == 0:
        raise RuntimeError(f"exposure pool {pool.path} has no finite values")

    finite_values = values[all_indices]
    q90, q99, q999 = torch.quantile(
        finite_values,
        torch.tensor([0.90, 0.99, 0.999], dtype=finite_values.dtype),
    )

    q90_q99 = all_indices[(finite_values >= q90) & (finite_values < q99)]
    q99_q999 = all_indices[(finite_values >= q99) & (finite_values < q999)]
    top_q999 = all_indices[finite_values >= q999]

    # Quantile ties can occasionally empty a narrow bin.  Falling back to the
    # full valid set keeps the diagnostic robust rather than crashing.
    if q90_q99.numel() == 0:
        q90_q99 = all_indices
    if q99_q999.numel() == 0:
        q99_q999 = all_indices
    if top_q999.numel() == 0:
        top_q999 = all_indices

    output = (all_indices, q90_q99, q99_q999, top_q999)
    _BIN_CACHE[key] = output
    return output


def _draw(indices: Tensor, count: int, generator: torch.Generator) -> Tensor:
    if count <= 0:
        return indices[:0]
    choose = torch.randint(
        0,
        int(indices.numel()),
        (count,),
        generator=generator,
    )
    return indices[choose]


def sample_source_stratified_batch(
    pools: Sequence[ExposurePool],
    *,
    batch_size: int,
    generator: torch.Generator,
) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
    """Exposure-balanced diagnostic sampler with bright-source oversampling.

    Per exposure, approximately 50% of samples remain uniform over all valid
    training-pool pixels.  The remaining 50% is divided across the 90--99,
    99--99.9, and top-0.1 percentile brightness strata in proportions
    25% / 15% / 10% of the total exposure quota.

    This deliberately changes the optimized sampling distribution and is only
    intended to diagnose whether source pixels are starved of gradient signal.
    It must not be interpreted as an unbiased final likelihood objective.
    """

    if batch_size < len(pools):
        raise ValueError("batch_size must be at least the number of exposures")

    each = math.ceil(batch_size / len(pools))
    n_uniform = max(1, int(round(0.50 * each)))
    n_q90_q99 = max(1, int(round(0.25 * each)))
    n_q99_q999 = max(1, int(round(0.15 * each)))
    n_top = each - n_uniform - n_q90_q99 - n_q99_q999
    if n_top < 1:
        n_top = 1
        n_uniform = max(1, each - n_q90_q99 - n_q99_q999 - n_top)

    uv_parts: list[Tensor] = []
    exposure_parts: list[Tensor] = []
    value_parts: list[Tensor] = []
    err_parts: list[Tensor] = []

    for exposure_index, pool in enumerate(pools):
        all_idx, q90_q99, q99_q999, top_q999 = _source_bins(pool)
        chosen = torch.cat(
            (
                _draw(all_idx, n_uniform, generator),
                _draw(q90_q99, n_q90_q99, generator),
                _draw(q99_q999, n_q99_q999, generator),
                _draw(top_q999, n_top, generator),
            )
        )

        uv_parts.append(pool.uv[chosen])
        value_parts.append(pool.value[chosen])
        err_parts.append(pool.err[chosen])
        exposure_parts.append(
            torch.full((chosen.numel(),), exposure_index, dtype=torch.long)
        )

    uv = torch.cat(uv_parts)
    exposure_id = torch.cat(exposure_parts)
    value = torch.cat(value_parts)
    err = torch.cat(err_parts)

    if uv.shape[0] > batch_size:
        order = torch.randperm(uv.shape[0], generator=generator)[:batch_size]
        uv = uv[order]
        exposure_id = exposure_id[order]
        value = value[order]
        err = err[order]

    mask = torch.ones_like(value, dtype=torch.bool)
    return uv, exposure_id, value, err, mask


ablation.training.sample_balanced_batch = sample_source_stratified_batch
ablation.base.train_model = ablation.training.train_model


if __name__ == "__main__":
    raise SystemExit(ablation.base.main())
