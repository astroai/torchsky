from __future__ import annotations

import math

import pytest
import torch

from torchsky.map.field.core import _BaseFieldModel
from torchsky.map.field.losses import _field_residual_loss
from torchsky.map.field.training import (
    _compute_full_observation_losses,
    fit_field_model,
)
from torchsky.map.field.types import FieldDomain, FieldObservation
from torchsky.map.operators.bundle import ObservationOperatorBundle


def _loss(
    residual: torch.Tensor,
    logvar: torch.Tensor,
    *,
    observation_invvar: torch.Tensor | None = None,
    sample_weight: torch.Tensor | None = None,
) -> torch.Tensor:
    return _field_residual_loss(
        residual,
        loss_kind="gaussian_nll",
        charbonnier_eps=1.0e-3,
        huber_delta=5.0e-2,
        logvar=logvar,
        observation_invvar=observation_invvar,
        sample_weight=sample_weight,
    )


def _header(width: int = 2, height: int = 2) -> dict:
    return {
        "NAXIS": 2,
        "NAXIS1": width,
        "NAXIS2": height,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": 1.0,
        "CRPIX2": 1.0,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -1.0 / 3600.0,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 1.0 / 3600.0,
    }


class _ConstantGaussianField(_BaseFieldModel):
    def __init__(self, *, dtype: torch.dtype = torch.float64) -> None:
        super().__init__(
            FieldDomain(
                center_ra_deg=150.0,
                center_dec_deg=2.0,
                spatial_scale_deg=0.01,
                wavelength_center=5000.0,
                wavelength_scale=1.0,
            )
        )
        self.mean = torch.nn.Parameter(torch.tensor(0.0, dtype=dtype))
        self.logvar = torch.nn.Parameter(torch.tensor(math.log(3.75), dtype=dtype))

    def forward(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        lam: torch.Tensor | float,
    ) -> torch.Tensor:
        del dec, lam
        return self.mean.expand_as(ra)

    def forward_distribution(
        self,
        ra: torch.Tensor,
        dec: torch.Tensor,
        lam: torch.Tensor | float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(ra, dec, lam), self.logvar.expand_as(ra)


def test_gaussian_nll_total_variance_has_analytic_predictive_optimum() -> None:
    logvar = torch.tensor(math.log(3.75), dtype=torch.float64, requires_grad=True)
    loss = _loss(
        torch.tensor(2.0, dtype=torch.float64),
        logvar,
        observation_invvar=torch.tensor(4.0, dtype=torch.float64),
    )
    gradient = torch.autograd.grad(loss, logvar)[0]
    expected = 0.5 * (1.0 + math.log(4.0))
    assert loss.item() == pytest.approx(expected)
    assert gradient.item() == pytest.approx(0.0, abs=1.0e-12)

    old_optimum = _loss(
        torch.tensor(2.0, dtype=torch.float64),
        torch.tensor(math.log(4.0), dtype=torch.float64),
        observation_invvar=torch.tensor(4.0, dtype=torch.float64),
    )
    assert loss < old_optimum


def test_gaussian_nll_precision_and_sample_weight_are_distinct() -> None:
    residual = torch.tensor([2.0, 2.0], dtype=torch.float64)
    logvar = torch.log(torch.tensor([3.0, 3.0], dtype=torch.float64))
    precision = torch.tensor([1.0, 4.0], dtype=torch.float64)
    actual = _loss(residual, logvar, observation_invvar=precision)
    total = torch.tensor([4.0, 3.25], dtype=torch.float64)
    expected = 0.5 * (residual.square() / total + total.log())
    torch.testing.assert_close(actual, expected)

    sample_weight = torch.tensor([2.0, 3.0], dtype=torch.float64)
    weighted = _loss(
        residual,
        logvar,
        observation_invvar=precision,
        sample_weight=sample_weight,
    )
    torch.testing.assert_close(weighted, sample_weight * expected)


def test_gaussian_nll_zero_and_invalid_weights_are_excluded() -> None:
    residual = torch.ones(6, dtype=torch.float64)
    logvar = torch.tensor(
        [float("nan"), float("inf"), 0.0, 0.0, 0.0, 0.0],
        dtype=torch.float64,
        requires_grad=True,
    )
    invalid = torch.tensor([0.0, -1.0, float("nan"), float("inf"), 2.0, 2.0])
    actual = _loss(residual, logvar, observation_invvar=invalid)
    assert torch.equal(actual[:4], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(actual).all()
    gradient = torch.autograd.grad(actual.sum(), logvar)[0]
    assert torch.equal(gradient[:4], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(gradient).all()

    generic = _loss(
        residual,
        logvar,
        observation_invvar=torch.full_like(residual, 2.0),
        sample_weight=invalid,
    )
    assert torch.equal(generic[:4], torch.zeros(4, dtype=torch.float64))
    assert torch.isfinite(generic).all()


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize(
    "device",
    [
        torch.device("cpu"),
        pytest.param(
            torch.device("cuda"),
            marks=pytest.mark.skipif(
                not torch.cuda.is_available(), reason="CUDA is unavailable"
            ),
        ),
        pytest.param(
            torch.device("mps"),
            marks=pytest.mark.skipif(
                not torch.backends.mps.is_available(), reason="MPS is unavailable"
            ),
        ),
    ],
)
def test_gaussian_nll_gradient_dtype_and_device(
    dtype: torch.dtype, device: torch.device
) -> None:
    if device.type == "mps" and dtype == torch.float64:
        pytest.skip("MPS does not support float64")
    residual = torch.tensor([0.5, 2.0], dtype=dtype, device=device, requires_grad=True)
    logvar = torch.tensor([-0.4, 0.7], dtype=dtype, device=device, requires_grad=True)
    invvar = torch.tensor([3.0, 0.5], dtype=dtype, device=device)
    loss = _loss(residual, logvar, observation_invvar=invvar).sum()
    residual_grad, logvar_grad = torch.autograd.grad(loss, (residual, logvar))
    assert loss.dtype == dtype and loss.device.type == device.type
    assert residual_grad.dtype == dtype and residual_grad.device.type == device.type
    assert logvar_grad.dtype == dtype and logvar_grad.device.type == device.type
    assert torch.isfinite(residual_grad).all()
    assert torch.isfinite(logvar_grad).all()


@pytest.mark.parametrize(
    ("reduction", "expected"),
    [
        ("per_observation", 1.7292130679473608),
        ("global", 0.8646065339736804),
        ("per_band", 1.7292130679473608),
        ("band_global_mix", 1.2969098009605206),
    ],
)
def test_full_observation_gaussian_masks_and_reductions(
    monkeypatch: pytest.MonkeyPatch,
    reduction: str,
    expected: float,
) -> None:
    observations = (
        FieldObservation(
            image=torch.tensor([[2.0, 100.0]], dtype=torch.float64),
            header=_header(2, 1),
            weight=torch.tensor([[4.0, 4.0]], dtype=torch.float64),
            mask=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            wavelength=5000.0,
            band_name="a",
        ),
        FieldObservation(
            image=torch.tensor([[1.0, 3.0]], dtype=torch.float64),
            header=_header(2, 1),
            weight=torch.tensor([[1.0, 0.0]], dtype=torch.float64),
            wavelength=5000.0,
            band_name="b",
        ),
    )
    predictions = iter(
        (
            (
                torch.zeros((1, 2), dtype=torch.float64),
                torch.full((1, 2), math.log(3.75), dtype=torch.float64),
            ),
            (
                torch.zeros((1, 2), dtype=torch.float64),
                torch.full((1, 2), math.log(0.5), dtype=torch.float64),
            ),
        )
    )

    def render(*args: object, **kwargs: object) -> tuple[torch.Tensor, torch.Tensor]:
        del args, kwargs
        return next(predictions)

    monkeypatch.setattr(
        "torchsky.map.field.training.render_field_observation_distribution", render
    )
    actual = _compute_full_observation_losses(
        observations,
        torch.nn.Linear(1, 1),
        (None, None),
        (None, None),
        (ObservationOperatorBundle(), ObservationOperatorBundle()),
        num_spectral_samples=1,
        loss_kind="gaussian_nll",
        charbonnier_eps=1.0e-3,
        huber_delta=5.0e-2,
        loss_reduction=reduction,
    )
    assert actual == pytest.approx(expected)


def test_fit_field_model_uses_total_variance_in_diagnostics_and_training() -> None:
    observation = FieldObservation(
        image=torch.full((2, 2), 2.0, dtype=torch.float64),
        header=_header(),
        weight=torch.full((2, 2), 4.0, dtype=torch.float64),
        mask=torch.tensor([[1.0, 1.0], [1.0, 0.0]], dtype=torch.float64),
        wavelength=5000.0,
    )
    result = fit_field_model(
        [observation],
        _ConstantGaussianField(),
        steps=1,
        lr=0.0,
        num_spectral_samples=1,
        loss_kind="gaussian_nll",
        device="cpu",
    )
    expected = 0.5 * (1.0 + math.log(4.0))
    assert result.initial_loss == pytest.approx(expected, rel=1.0e-6)
    assert result.loss_history == pytest.approx([expected], rel=1.0e-6)
    assert result.final_loss == pytest.approx(expected, rel=1.0e-6)
