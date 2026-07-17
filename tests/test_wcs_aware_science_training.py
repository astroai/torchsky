from __future__ import annotations

from dataclasses import dataclass

import torch

from torchsky.map.wcs_aware_objectives import err_whitened_robust_loss
from torchsky.map.wcs_aware_sampling import sample_band_balanced_batch


@dataclass
class _Pool:
    band: str
    uv: torch.Tensor
    value: torch.Tensor
    err: torch.Tensor

    @property
    def size(self) -> int:
        return int(self.value.numel())


def _pool(band: str, offset: float) -> _Pool:
    value = torch.arange(20, dtype=torch.float32) + offset
    uv = torch.stack((value, value), dim=-1)
    return _Pool(
        band=band,
        uv=uv,
        value=value,
        err=torch.ones_like(value),
    )


def test_err_whitened_huber_uses_sigma_residuals() -> None:
    prediction = torch.tensor([2.0, 8.0])
    target = torch.tensor([0.0, 0.0])
    err = torch.tensor([2.0, 4.0])
    mask = torch.ones(2, dtype=torch.bool)

    loss, diagnostics = err_whitened_robust_loss(
        prediction,
        target,
        err,
        mask,
        kind="huber",
        delta=3.0,
        max_relative_weight=100.0,
    )

    # Whitened residuals are [1, 2], both in the quadratic Huber region.
    assert torch.allclose(loss, torch.tensor(1.25))
    assert torch.allclose(diagnostics["whitened_rmse"], torch.sqrt(torch.tensor(2.5)))
    assert torch.allclose(diagnostics["whitened_mae"], torch.tensor(1.5))


def test_sampler_balances_bands_before_exposures() -> None:
    pools = [
        _pool("F090W", 0.0),
        _pool("F090W", 100.0),
        _pool("F150W", 200.0),
        _pool("F277W", 300.0),
        _pool("F277W", 400.0),
        _pool("F277W", 500.0),
    ]
    generator = torch.Generator(device="cpu").manual_seed(7)

    _, exposure_id, _, _, mask = sample_band_balanced_batch(
        pools,
        batch_size=60,
        generator=generator,
    )

    bands = [pools[int(index)].band for index in exposure_id.tolist()]
    assert bands.count("F090W") == 20
    assert bands.count("F150W") == 20
    assert bands.count("F277W") == 20
    assert int(mask.sum()) == 60

    counts = torch.bincount(exposure_id, minlength=len(pools))
    assert counts[2].item() == 20
    assert sorted(counts[:2].tolist()) == [10, 10]
    assert max(counts[3:].tolist()) - min(counts[3:].tolist()) <= 1
