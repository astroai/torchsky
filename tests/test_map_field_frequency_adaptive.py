from __future__ import annotations

import pytest
import torch

from torchsky.field.frequency_adaptive import BaconField, SASNetField
from torchsky.map.field.types import FieldDomain


def _domain(*, coord_mode: str = "local_tangent") -> FieldDomain:
    return FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=-20.0,
        spatial_scale_deg=0.01,
        wavelength_center=20000.0,
        wavelength_scale=15000.0,
        coord_mode=coord_mode,
    )


def _coordinates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ra = torch.tensor([[9.999, 10.0], [10.001, 10.002]], dtype=torch.float32)
    dec = torch.tensor([[-20.001, -20.0], [-19.999, -19.998]], dtype=torch.float32)
    wavelength = torch.tensor(
        [[10000.0, 15000.0], [25000.0, 35000.0]], dtype=torch.float32
    )
    return ra, dec, wavelength


def test_bacon_field_is_finite_positive_and_multiscale() -> None:
    torch.manual_seed(2)
    model = BaconField(
        _domain(), hidden_dim=16, bandwidths=(1.0, 2.0, 4.0), positive=True
    )
    ra, dec, wavelength = _coordinates()
    coarse = model.forward_level(ra, dec, wavelength, level=0)
    fine = model(ra, dec, wavelength)
    assert coarse.shape == ra.shape
    assert fine.shape == ra.shape
    assert torch.isfinite(coarse).all()
    assert torch.isfinite(fine).all()
    assert (coarse > 0).all()
    assert (fine > 0).all()
    assert not torch.allclose(coarse, fine)


def test_bacon_field_validates_bandwidths_and_level() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BaconField(_domain(), hidden_dim=8, bandwidths=(1.0, 1.0))
    model = BaconField(_domain(), hidden_dim=8, bandwidths=(1.0, 2.0))
    ra, dec, wavelength = _coordinates()
    with pytest.raises(ValueError, match="level"):
        model.forward_level(ra, dec, wavelength, level=2)


def test_sasnet_field_masks_are_local_finite_and_differentiable() -> None:
    torch.manual_seed(3)
    model = SASNetField(
        _domain(),
        num_frequencies=16,
        max_frequency=8.0,
        hidden_dim=16,
        mask_hidden_dim=12,
        hash_levels=3,
        hash_features_per_level=2,
        hash_log2_size=6,
        hash_base_resolution=4,
        hash_finest_resolution=16,
    )
    ra, dec, wavelength = _coordinates()
    masks = model.spatial_masks(ra, dec, wavelength)
    prediction = model(ra, dec, wavelength)
    assert masks.shape == (*ra.shape, 16)
    assert prediction.shape == ra.shape
    assert torch.isfinite(masks).all()
    assert torch.isfinite(prediction).all()
    assert ((masks > 0) & (masks < 1)).all()
    assert (prediction > 0).all()
    prediction.sum().backward()
    gradients = [p.grad for p in model.parameters() if p.requires_grad]
    assert gradients
    assert all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_sasnet_field_supports_spherical_coordinates() -> None:
    model = SASNetField(
        _domain(coord_mode="sphere_xyz"),
        num_frequencies=8,
        hidden_dim=8,
        mask_hidden_dim=8,
        hash_levels=2,
        hash_features_per_level=2,
        hash_log2_size=5,
        hash_base_resolution=4,
        hash_finest_resolution=8,
    )
    ra, dec, wavelength = _coordinates()
    output = model(ra, dec, wavelength)
    assert output.shape == ra.shape
    assert torch.isfinite(output).all()
