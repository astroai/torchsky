"""Explicit constant conditioning must not introduce unobserved variation."""

import pytest
import torch

from torchsky.map.operators.adaptive_axes import (
    AdaptiveAxisFeatures,
    AdaptiveSpatioSpectralTemporalEncoder,
)


def test_constant_axis_is_invariant_and_checkpoint_preserves_mode():
    axis = AdaptiveAxisFeatures.from_samples(torch.tensor([1.5]), constant=True)
    values = torch.tensor([1.0, 1.5, 2.0], requires_grad=True)
    output = axis(values)
    assert axis.num_bands == 0 and axis.output_dim == 1
    torch.testing.assert_close(output, torch.zeros(3, 1), rtol=0, atol=0)
    output.sum().backward()
    torch.testing.assert_close(values.grad, torch.zeros(3), rtol=0, atol=0)
    restored = AdaptiveAxisFeatures(center=0, reference_scale=2,
                                    length_bounds=(1, 4), num_bands=0)
    restored.load_state_dict(axis.state_dict())
    assert restored.constant
    torch.testing.assert_close(restored(values), output)
    legacy = restored.state_dict()
    del legacy["_extra_state"]["constant"]
    restored.load_state_dict(legacy)
    assert not restored.constant
    assert torch.count_nonzero(restored(values)) == 2


def test_single_band_encoder_is_spatial_only_across_passband_and_time():
    coordinates = torch.linspace(0, 10, 9)
    kwargs = dict(spatial_x=coordinates, spatial_y=coordinates,
                  wavelength=torch.full((9,), 1.5), time_mjd=torch.full((9,), 60000.0))
    with pytest.raises(ValueError, match="three distinct"):
        AdaptiveSpatioSpectralTemporalEncoder.from_samples(**kwargs)
    encoder = AdaptiveSpatioSpectralTemporalEncoder.from_samples(
        **kwargs, spectral_constant=True, temporal_constant=True)
    first = encoder(coordinates, coordinates, 1.1, 59999.0)
    second = encoder(coordinates, coordinates, 1.9, 60001.0)
    torch.testing.assert_close(first, second, rtol=0, atol=0)
    assert not torch.equal(first[0], first[-1])
    assert encoder.scale_diagnostics()["spectral"]["constant"]
    with pytest.raises(ValueError, match="three distinct"):
        AdaptiveSpatioSpectralTemporalEncoder.from_samples(
            **dict(kwargs, spatial_x=torch.tensor([1.0, 2.0])),
            spectral_constant=True, temporal_constant=True)


def test_constant_axis_requires_finite_evidence():
    with pytest.raises(ValueError, match="finite calibration"):
        AdaptiveAxisFeatures.from_samples(torch.tensor([float("nan")]), constant=True)
