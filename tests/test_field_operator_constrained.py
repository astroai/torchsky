"""Regression tests for the operator-constrained feature field."""

from __future__ import annotations

import torch

from torchsky.map.field import FieldDomain, FieldObservation
from torchsky.map.field.models_skylambda import OperatorConstrainedFeatureField

from map_field_support import _tan_header

_DOMAIN = FieldDomain(
    center_ra_deg=10.0,
    center_dec_deg=0.0,
    spatial_scale_deg=0.1,
    wavelength_center=5000.0,
    wavelength_scale=1000.0,
)
_CPU = torch.device("cpu")
_GRID = torch.linspace(4000.0, 6000.0, 21)
_BAND = {"spectral_min": 4000.0, "spectral_max": 6000.0, "band_name": "x"}


def _observation(**kwargs) -> FieldObservation:
    header = _tan_header(4, crval1=10.0, crval2=0.0, scale_deg=0.01)
    return FieldObservation(image=torch.zeros(4, 4), header=header, **kwargs)


def _tabulated(center: float) -> FieldObservation:
    throughput = torch.exp(-0.5 * ((_GRID - center) / 200.0) ** 2)
    return _observation(
        **_BAND, transmission_wavelength=_GRID, transmission_throughput=throughput
    )


def _weights(model, observations) -> torch.Tensor:
    return model._observability_mode_weights(
        observations, device=_CPU, dtype=torch.float32
    )


def test_observability_weights_are_not_reused_across_band_responses() -> None:
    # The cache key held only scalar band metadata, so a second transmission
    # table (or delta-band wavelength) with the same metadata reused the
    # weights computed for the first one.
    model = OperatorConstrainedFeatureField(_DOMAIN)
    cases = (
        [_tabulated(4300.0)],
        [_tabulated(5700.0)],
        [_observation(wavelength=4500.0)],
        [_observation(wavelength=5500.0)],
    )
    for observations in cases:
        fresh = _weights(OperatorConstrainedFeatureField(_DOMAIN), observations)
        torch.testing.assert_close(_weights(model, observations), fresh)
    blue, red = (_weights(model, case) for case in cases[:2])
    assert not torch.allclose(blue, red)


def test_observability_penalty_accepts_named_and_unnamed_bands() -> None:
    # Sorting a key that mixed band_name=None with a string raised TypeError.
    model = OperatorConstrainedFeatureField(_DOMAIN)
    unnamed = {key: value for key, value in _BAND.items() if key != "band_name"}
    observations = [_observation(**_BAND), _observation(**unnamed)]
    penalty = model.spectral_observability_penalty(observations)
    assert bool(torch.isfinite(penalty))
    assert torch.equal(
        model.spectral_observability_penalty(observations[::-1]), penalty
    )
