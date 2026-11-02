from __future__ import annotations

import io

import pytest
import torch

from torchsky.map.field import FIELD_MODEL_KINDS, FieldDomain, make_field_model


@pytest.fixture(scope="module")
def domain() -> FieldDomain:
    return FieldDomain(
        center_ra_deg=150.0,
        center_dec_deg=2.0,
        spatial_scale_deg=0.05,
        wavelength_center=5600.0,
        wavelength_scale=1000.0,
        coord_mode="local_tangent",
    )


def _make_contract_model(kind: str, domain: FieldDomain):
    if kind == "sasnet":
        return make_field_model(kind, domain, pixel_footprint_deg=0.001)
    if kind in {
        "spectral_power_foam",
        "foam_initialized_inr",
        "diffuse_foam_hybrid",
    }:
        foam = make_field_model(
            "spectral_power_foam",
            domain,
            sites=torch.tensor([[-0.25, 0.0], [0.25, 0.0]]),
            radii=torch.tensor([0.8, 0.8]),
            cell_coefficients=torch.tensor([[0.5, 0.1], [0.8, -0.1]]),
            log_intensity_spectra=kind == "diffuse_foam_hybrid",
        )
        if kind == "spectral_power_foam":
            return foam
        if kind == "diffuse_foam_hybrid":
            return make_field_model(
                "diffuse_foam_hybrid",
                domain,
                foam=foam,
                diffuse_hidden_dim=8,
                diffuse_hidden_layers=1,
            )
        return make_field_model(
            "foam_initialized_inr",
            domain,
            foam=foam,
            hidden_dim=8,
            hidden_layers=1,
        )
    return make_field_model(kind, domain)


@pytest.mark.parametrize("kind", FIELD_MODEL_KINDS)
@pytest.mark.parametrize("coordinate_dtype", [torch.float32, torch.float64])
def test_registered_field_model_contract(
    kind: str, domain: FieldDomain, coordinate_dtype
) -> None:
    """Every registered candidate has the same minimum evaluation contract."""
    torch.manual_seed(8912)
    model = _make_contract_model(kind, domain)
    ra = torch.tensor([149.999, 150.0, 150.001], dtype=coordinate_dtype)
    dec = torch.tensor([1.999, 2.0, 2.001], dtype=coordinate_dtype)
    wavelength = torch.tensor([5200.0, 5600.0, 6000.0], dtype=coordinate_dtype)

    output = model(ra, dec, wavelength)
    assert isinstance(output, torch.Tensor)
    assert output.shape == ra.shape
    assert torch.isfinite(output).all()
    output.sum().backward()
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.requires_grad and parameter.grad is not None
    ]
    assert gradients
    assert all(torch.isfinite(gradient).all() for gradient in gradients)

    stream = io.BytesIO()
    torch.save(model.state_dict(), stream)
    stream.seek(0)
    state = torch.load(stream, weights_only=True)
    restored = _make_contract_model(kind, domain)
    restored.load_state_dict(state, strict=True)
    with torch.no_grad():
        torch.testing.assert_close(restored(ra, dec, wavelength), output.detach())
