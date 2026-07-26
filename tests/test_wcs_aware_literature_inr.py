from __future__ import annotations

import torch

from torchsky.map.wcs_aware_inr import UnionCanvas
from torchsky.map.wcs_aware_literature_inr import (
    NeRFStyleMLP,
    SirenMLP,
    WCSAwareLiteratureINR,
)


def _simple_header(size: int = 8) -> dict[str, float | int | str]:
    scale = 0.031 / 3600.0
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CUNIT1": "deg",
        "CUNIT2": "deg",
        "CRVAL1": 110.8,
        "CRVAL2": -73.45,
        "CRPIX1": (size + 1) / 2.0,
        "CRPIX2": (size + 1) / 2.0,
        "CD1_1": -scale,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": scale,
    }


def _canvas(size: int = 8) -> UnionCanvas:
    return UnionCanvas(
        header=_simple_header(size),
        shape=(size, size),
        pixel_scale_arcsec=0.031,
    )


def test_nerf_style_mlp_has_expected_skip_input_dimension() -> None:
    model = NeRFStyleMLP(
        in_dim=3,
        num_frequencies=4,
        hidden_dim=32,
        depth=8,
        skip_after_layer=4,
    )
    encoded_dim = 3 * (1 + 2 * 4)
    assert model.layers[0].in_features == encoded_dim
    assert model.layers[5].in_features == 32 + encoded_dim
    output = model(torch.zeros(5, 3))
    assert output.shape == (5, 1)
    assert torch.isfinite(output).all()


def test_siren_mlp_is_finite_and_differentiable() -> None:
    model = SirenMLP(
        in_dim=3,
        hidden_dim=32,
        hidden_layers=2,
        first_omega_0=30.0,
        hidden_omega_0=30.0,
    )
    coordinate = torch.randn(7, 3, requires_grad=True)
    output = model(coordinate)
    assert output.shape == (7, 1)
    assert torch.isfinite(output).all()
    output.square().mean().backward()
    assert coordinate.grad is not None
    assert torch.isfinite(coordinate.grad).all()


def test_literature_models_use_existing_observation_operator() -> None:
    header = _simple_header()
    canvas = _canvas()
    uv = torch.tensor([[2.0, 2.0], [3.0, 4.0], [5.0, 5.0]])
    exposure_id = torch.zeros(3, dtype=torch.long)
    wavelength = torch.tensor([[9000.0, 10000.0]])
    weight = torch.tensor([[0.4, 0.6]])

    for architecture in ("A1_nerf", "A2_siren"):
        model = WCSAwareLiteratureINR(
            [header],
            canvas,
            architecture=architecture,
            wavelength_min=9000.0,
            wavelength_max=10000.0,
            alignment_mode="original_wcs",
            learn_background=True,
            output_activation="softplus",
            nerf_num_frequencies=4,
            nerf_hidden_dim=32,
            siren_hidden_dim=32,
            siren_hidden_layers=2,
        )
        prediction, valid = model.predict_observation(
            uv,
            exposure_id,
            wavelength,
            weight,
            psf=None,
        )
        assert prediction.shape == (3,)
        assert valid.shape == (3,)
        assert valid.all()
        assert torch.isfinite(prediction).all()
        assert (prediction >= 0).all()
