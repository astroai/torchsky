"""Tiled field rendering must match untiled rendering.

The exact Foam refinement renders through native detector frames; on small
MIG slices an untiled forward overflows device memory. spatial_tile_hw
chunks that forward, so tile assembly must not change results.
"""

from __future__ import annotations

import math

import torch

from torchsky.map.field.core import estimate_field_domain, make_field_model
from torchsky.map.field.rendering import render_field_observation
from torchsky.map.field.types import FieldObservation


def _tiny_observation(side: int = 48) -> FieldObservation:
    header = {
        "NAXIS1": side,
        "NAXIS2": side,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": side / 2,
        "CRPIX2": side / 2,
        "CRVAL1": 150.0,
        "CRVAL2": 2.0,
        "CD1_1": -2.5e-8,
        "CD1_2": 0.0,
        "CD2_1": 0.0,
        "CD2_2": 2.5e-8,
    }
    image = torch.rand(side, side) * 0.1
    return FieldObservation(image=image, header=header, wavelength=3.5e4)


def _tiny_foam_model(observation: FieldObservation) -> torch.nn.Module:
    domain = estimate_field_domain([observation], coord_mode="local_tangent")
    side = 4
    step = 2.0 * 0.55 / side
    centers = torch.linspace(-0.55 + 0.5 * step, 0.55 - 0.5 * step, side)
    yy, xx = torch.meshgrid(centers, centers, indexing="ij")
    sites = torch.stack((xx.reshape(-1), yy.reshape(-1)), dim=-1)
    radii = torch.full((side * side,), math.hypot(step, step))
    coefficients = torch.zeros((side * side, 2))
    return make_field_model(
        "spectral_power_foam",
        domain,
        sites=sites,
        radii=radii,
        cell_coefficients=coefficients,
        minimum_radius=0.25 * step,
    )


def test_tiled_render_matches_untiled() -> None:
    observation = _tiny_observation()
    model = _tiny_foam_model(observation).eval()
    with torch.no_grad():
        full = render_field_observation(model, observation, num_spectral_samples=3)
        tiled = render_field_observation(
            model,
            observation,
            num_spectral_samples=3,
            spatial_tile_hw=(16, 16),
        )
    assert full.shape == tiled.shape == (48, 48)
    max_abs = float((full - tiled).abs().max())
    assert torch.allclose(full, tiled, rtol=1.0e-5, atol=1.0e-6), max_abs
