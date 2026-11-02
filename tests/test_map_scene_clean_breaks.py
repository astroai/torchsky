from __future__ import annotations

import importlib
import inspect

import pytest
import torch
from torch import Tensor, nn

import torchsky.map as map_api
from torchsky.map.field.forwarding import (
    call_forward_distribution,
    call_render_bandpass,
    call_render_bandpass_distribution,
    call_scalar_field_forward,
)
from torchsky.map.field.models_encoder import StrainerSkyEncoderMLPField
from torchsky.map.field.types import FieldDomain, ForwardContext
from torchsky.map.observation import CubeResult
from torchsky.scene import SourceCatalogScene


class _ContextBlindField(nn.Module):
    def forward(self, ra: Tensor, dec: Tensor, wavelength: Tensor | float) -> Tensor:
        return ra + dec + torch.as_tensor(wavelength)

    def render_bandpass(
        self, ra: Tensor, dec: Tensor, wavelengths: Tensor, weights: Tensor
    ) -> Tensor:
        return ra + dec + torch.sum(wavelengths * weights)

    def render_bandpass_distribution(
        self, ra: Tensor, dec: Tensor, wavelengths: Tensor, weights: Tensor
    ) -> tuple[Tensor, Tensor]:
        value = self.render_bandpass(ra, dec, wavelengths, weights)
        return value, torch.zeros_like(value)

    def forward_distribution(
        self, ra: Tensor, dec: Tensor, wavelength: Tensor | float
    ) -> tuple[Tensor, Tensor]:
        value = self(ra, dec, wavelength)
        return value, torch.zeros_like(value)


@pytest.mark.parametrize(
    "module_name",
    [
        "torchsky.map.image_plane",
        "torchsky.map.operators.provenance",
        "torchsky.map.field.models",
        "torchsky.map.field.foam",
    ],
)
def test_removed_forwarding_modules_are_unimportable(module_name: str) -> None:
    with pytest.raises((ModuleNotFoundError, FileNotFoundError)):
        importlib.import_module(module_name)


def test_removed_aliases_are_absent_from_public_surfaces() -> None:
    assert not hasattr(map_api, "evaluate_image_plane_aperture_uncertainty")
    assert not hasattr(CubeResult, "line_uncertainty")
    assert "templates" not in inspect.signature(SourceCatalogScene).parameters
    assert (
        "spectral_coefficients" not in inspect.signature(SourceCatalogScene).parameters
    )
    assert (
        "per_band_coefficients" not in inspect.signature(SourceCatalogScene).parameters
    )
    model = StrainerSkyEncoderMLPField(FieldDomain(0.0, 0.0, 1.0, 5000.0, 1000.0))
    assert not hasattr(model, "output")


def test_forwarding_helpers_reject_unconsumed_context() -> None:
    model = _ContextBlindField()
    ra = torch.zeros(2)
    dec = torch.zeros(2)
    wavelengths = torch.tensor([5000.0, 6000.0])
    weights = torch.tensor([0.5, 0.5])
    context = ForwardContext(survey="test")

    calls = (
        lambda: call_scalar_field_forward(
            model, ra, dec, 5500.0, forward_context=context
        ),
        lambda: call_render_bandpass(
            model, ra, dec, wavelengths, weights, forward_context=context
        ),
        lambda: call_render_bandpass_distribution(
            model, ra, dec, wavelengths, weights, forward_context=context
        ),
        lambda: call_forward_distribution(
            model, ra, dec, 5500.0, forward_context=context
        ),
    )
    for call in calls:
        with pytest.raises(TypeError, match="cannot consume ForwardContext"):
            call()
