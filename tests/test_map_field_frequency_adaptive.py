from __future__ import annotations

import io
import math
from pathlib import Path

import pytest
import torch

from torchsky.field.frequency_adaptive import BaconField, SASNetField
from torchsky.map.field.artifacts import load_field_artifact, save_field_artifact
from torchsky.map.field.core import make_field_model
from torchsky.map.field.registry import FIELD_MODEL_KINDS
from torchsky.map.field.types import FieldDomain


def _domain(*, coord_mode: str = "local_tangent") -> FieldDomain:
    return FieldDomain(
        center_ra_deg=10.0,
        center_dec_deg=0.0,
        spatial_scale_deg=0.01,
        wavelength_center=20_000.0,
        wavelength_scale=15_000.0,
        coord_mode=coord_mode,
    )


def _coordinates() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ra = torch.tensor([[9.999, 10.0], [10.001, 10.002]], dtype=torch.float32)
    dec = torch.tensor([[-0.001, 0.0], [0.001, 0.002]], dtype=torch.float32)
    wavelength = torch.tensor(
        [[10_000.0, 15_000.0], [25_000.0, 35_000.0]], dtype=torch.float32
    )
    return ra, dec, wavelength


def test_bacon_sum_of_squares_preserves_declared_output_bandwidth() -> None:
    domain = FieldDomain(
        center_ra_deg=0.0,
        center_dec_deg=0.0,
        spatial_scale_deg=1.0,
        wavelength_center=1.0,
        wavelength_scale=1.0,
    )
    model = BaconField(
        domain,
        hidden_dim=1,
        bandwidths=(2.0,),
        readout_terms=1,
    )
    assert model.output_bandwidths == (2.0,)
    assert model.latent_bandwidths == (1.0,)
    with torch.no_grad():
        model.omega_0.zero_()
        model.omega_0[0, 0] = 1.0
        model.phases[0].zero_()
        model.readouts[0].weight.fill_(1.0)
        model.readouts[0].bias.zero_()
    x = torch.linspace(-0.5, 0.5, 31)
    zeros = torch.zeros_like(x)
    ones = torch.ones_like(x)
    encoded_x = model._encoded_coords(x, zeros, ones)[..., 0]
    assert torch.allclose(
        model(x, zeros, ones),
        torch.sin(encoded_x).square(),
        atol=1.0e-6,
    )


def test_bacon_is_nonnegative_multiscale_and_differentiable() -> None:
    torch.manual_seed(2)
    model = BaconField(
        _domain(),
        hidden_dim=16,
        bandwidths=(1.0, 2.0, 4.0),
        readout_terms=2,
    )
    ra, dec, wavelength = _coordinates()
    coarse = model.forward_level(ra, dec, wavelength, level=0)
    fine = model(ra, dec, wavelength)
    assert coarse.shape == fine.shape == ra.shape
    assert torch.isfinite(coarse).all() and torch.isfinite(fine).all()
    assert (coarse >= 0.0).all() and (fine >= 0.0).all()
    assert not torch.allclose(coarse, fine)
    multiscale_loss = sum(
        model.forward_level(ra, dec, wavelength, level=level).sum()
        for level in range(model.num_levels)
    )
    multiscale_loss.backward()
    gradients = [parameter.grad for parameter in model.parameters()]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)


def test_bacon_validates_its_spectral_contract() -> None:
    with pytest.raises(ValueError, match="strictly increasing"):
        BaconField(_domain(), hidden_dim=8, bandwidths=(1.0, 1.0))
    with pytest.raises(ValueError, match="readout_terms"):
        BaconField(_domain(), hidden_dim=8, readout_terms=0)
    with pytest.raises(ValueError, match="intensity_floor"):
        BaconField(_domain(), hidden_dim=8, intensity_floor=-1.0)
    model = BaconField(_domain(), hidden_dim=8, bandwidths=(1.0, 2.0))
    ra, dec, wavelength = _coordinates()
    with pytest.raises(ValueError, match="level"):
        model.forward_level(ra, dec, wavelength, level=2)


def test_sasnet_caps_frequency_and_hash_scale_from_pixel_and_psf() -> None:
    model = SASNetField(
        _domain(),
        pixel_footprint_deg=0.001,
        psf_support_deg=0.002,
        num_frequencies=16,
        max_frequency=100.0,
        hidden_dim=16,
        mask_hidden_dim=12,
        hash_levels=3,
        hash_features_per_level=2,
        hash_log2_size=6,
        hash_base_resolution=8,
        hash_finest_resolution=128,
    )
    expected_frequency = math.pi * 0.01 / 0.002
    assert model.requested_max_frequency == 100.0
    assert model.max_supported_angular_frequency == pytest.approx(expected_frequency)
    assert model.max_frequency == pytest.approx(expected_frequency)
    assert model.requested_hash_finest_resolution == 128
    assert model.mask_encoder.base_resolution == 8
    assert model.mask_encoder.finest_resolution == 11
    assert max(model.mask_encoder.resolutions) == 11
    assert torch.linalg.vector_norm(model.frequencies, dim=-1).max() <= (
        expected_frequency * (1.0 + 1.0e-6)
    )


def test_sasnet_is_local_finite_differentiable_and_tensor_serializable() -> None:
    torch.manual_seed(3)
    kwargs = {
        "pixel_footprint_deg": 0.001,
        "psf_support_deg": 0.002,
        "num_frequencies": 16,
        "max_frequency": 8.0,
        "hidden_dim": 16,
        "mask_hidden_dim": 12,
        "hash_levels": 3,
        "hash_features_per_level": 2,
        "hash_log2_size": 6,
        "hash_base_resolution": 4,
        "hash_finest_resolution": 16,
    }
    model = SASNetField(_domain(), **kwargs)
    ra, dec, wavelength = _coordinates()
    masks = model.spatial_masks(ra, dec, wavelength)
    prediction = model(ra, dec, wavelength)
    assert masks.shape == (*ra.shape, 16)
    assert prediction.shape == ra.shape
    assert torch.isfinite(masks).all() and torch.isfinite(prediction).all()
    assert ((masks > 0.0) & (masks < 1.0)).all()
    assert (prediction > 0.0).all()
    prediction.sum().backward()
    gradients = [parameter.grad for parameter in model.parameters() if parameter.requires_grad]
    assert gradients and all(gradient is not None for gradient in gradients)
    assert all(torch.isfinite(gradient).all() for gradient in gradients if gradient is not None)

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = SASNetField(_domain(), **kwargs)
    restored.load_state_dict(state)
    assert torch.allclose(restored(ra, dec, wavelength), prediction.detach())


def test_sasnet_requires_resolved_observation_support() -> None:
    with pytest.raises(ValueError, match="pixel_footprint_deg"):
        SASNetField(_domain(), pixel_footprint_deg=0.0)
    with pytest.raises(ValueError, match="psf_support_deg"):
        SASNetField(
            _domain(),
            pixel_footprint_deg=0.001,
            psf_support_deg=float("nan"),
        )
    coarse = SASNetField(_domain(), pixel_footprint_deg=0.1)
    assert coarse.mask_encoder.base_resolution == 2
    assert coarse.mask_encoder.finest_resolution == 2


@pytest.mark.parametrize("coord_mode", ["local_tangent", "sphere_xyz"])
def test_sasnet_factory_artifact_roundtrip(tmp_path: Path, coord_mode: str) -> None:
    kwargs = {
        "pixel_footprint_deg": 0.001,
        "psf_support_deg": 0.002,
        "num_frequencies": 8,
        "hidden_dim": 8,
        "mask_hidden_dim": 8,
        "hash_levels": 2,
        "hash_features_per_level": 2,
        "hash_log2_size": 5,
        "hash_base_resolution": 4,
        "hash_finest_resolution": 16,
    }
    assert "sasnet" in FIELD_MODEL_KINDS
    model = make_field_model("sasnet", _domain(coord_mode=coord_mode), **kwargs)
    assert isinstance(model, SASNetField)
    coordinates = _coordinates()
    prediction = model(*coordinates).detach()
    save_field_artifact(
        tmp_path, model=model, model_kind="sasnet", model_kwargs=kwargs
    )
    loaded = load_field_artifact(tmp_path, device="cpu")
    restored = loaded.model
    assert isinstance(restored, SASNetField)
    assert loaded.manifest.model_kind == "sasnet"
    assert restored.domain == model.domain
    assert restored.max_frequency == model.max_frequency
    assert restored.mask_encoder.resolutions == model.mask_encoder.resolutions
    assert torch.equal(restored.frequencies, model.frequencies)
    assert torch.equal(restored(*coordinates), prediction)


def test_explicit_grid_factory_artifact_roundtrip(tmp_path: Path) -> None:
    kwargs = {"spatial_resolution": 4, "spectral_resolution": 3}
    model = make_field_model("explicit_spectral_grid_field", _domain(), **kwargs)
    coordinates = _coordinates()
    prediction = model(*coordinates)
    assert prediction.shape == coordinates[0].shape
    assert torch.isfinite(prediction).all() and (prediction > 0).all()
    prediction.sum().backward()
    assert model.grid.grad is not None and torch.isfinite(model.grid.grad).all()
    assert model.grid.grad.abs().sum() > 0
    save_field_artifact(
        tmp_path,
        model=model,
        model_kind="explicit_spectral_grid_field",
        model_kwargs=kwargs,
    )
    restored = load_field_artifact(tmp_path, device="cpu").model
    assert torch.equal(restored(*coordinates), prediction.detach())
    with pytest.raises(ValueError, match="local_tangent"):
        make_field_model(
            "explicit_spectral_grid_field", _domain(coord_mode="sphere_xyz"), **kwargs
        )
    with pytest.raises(ValueError, match="at least 2"):
        make_field_model("explicit_spectral_grid_field", _domain(), spatial_resolution=1)


def test_sasnet_supports_spherical_coordinates() -> None:
    model = SASNetField(
        _domain(coord_mode="sphere_xyz"),
        pixel_footprint_deg=0.001,
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
    assert model.max_supported_angular_frequency == pytest.approx(
        math.pi / math.radians(0.001)
    )


@pytest.mark.parametrize("model_name", ["bacon", "sasnet"])
@pytest.mark.parametrize("device_name", ["cuda", "mps"])
def test_frequency_adaptive_models_accelerator_contract(
    model_name: str,
    device_name: str,
) -> None:
    if device_name == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA is unavailable")
    if device_name == "mps" and not torch.backends.mps.is_available():
        pytest.skip("MPS is unavailable")
    device = torch.device(device_name)
    if model_name == "bacon":
        model = BaconField(
            _domain(),
            hidden_dim=8,
            bandwidths=(1.0, 2.0),
        )
    else:
        model = SASNetField(
            _domain(),
            pixel_footprint_deg=0.001,
            psf_support_deg=0.002,
            num_frequencies=8,
            hidden_dim=8,
            mask_hidden_dim=8,
            hash_levels=2,
            hash_features_per_level=2,
            hash_log2_size=5,
            hash_base_resolution=4,
            hash_finest_resolution=8,
        )
    model = model.to(device)
    ra, dec, wavelength = (
        value.to(device) for value in _coordinates()
    )
    output = model(ra, dec, wavelength)
    assert output.device.type == device_name
    assert torch.isfinite(output).all()
    output.sum().backward()
    assert any(
        parameter.grad is not None and torch.isfinite(parameter.grad).all()
        for parameter in model.parameters()
        if parameter.requires_grad
    )
