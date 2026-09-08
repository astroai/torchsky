"""Foam photometry against an analytic continuum, independent of Foam truth."""

from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest
import torch

pytest.importorskip("PIL")

BENCHMARKS = Path(__file__).resolve().parents[1] / "benchmarks"
if str(BENCHMARKS) not in sys.path:
    sys.path.insert(0, str(BENCHMARKS))

from bench_map_smacs0723_field_fit_joint import (  # noqa: E402
    _initialize_foam_from_band_coadds,
    _joint_real_fit_kwargs,
    _refine_foam_from_observations_exact,
    _regular_foam_model_kwargs,
    _set_foam_geometry_trainable,
)
from torchsky.map.field import (  # noqa: E402
    FieldDomain,
    FieldObservation,
    SpectralPowerFoamField,
    fit_field_model,
    make_tangent_plane_header,
    render_spectral_power_foam_observation_exact,
)
from torchsky.map.operators import apply_psf_spec  # noqa: E402


@pytest.mark.parametrize("initialize_from_coadd", [False, True])
def test_linear_foam_recovers_independent_continuum_source_flux(
    initialize_from_coadd: bool,
) -> None:
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        size = 24
        domain = FieldDomain(
            center_ra_deg=10.0,
            center_dec_deg=20.0,
            spatial_scale_deg=0.024,
            wavelength_center=2.0,
            wavelength_scale=1.0,
        )
        header = make_tangent_plane_header(
            size, size, crval_ra_deg=10.0, crval_dec_deg=20.0
        )
        axis = torch.arange(size, dtype=torch.float64)
        sigma = 1.4

        def pixel_integral(center: float) -> torch.Tensor:
            # Analytic top-hat integral, without any Foam renderer or coefficients.
            scale = sigma * math.sqrt(2.0)
            return sigma * math.sqrt(math.pi / 2.0) * (
                torch.erf((axis + 0.5 - center) / scale)
                - torch.erf((axis - 0.5 - center) / scale)
            )

        source = pixel_integral(11.1)[:, None] * pixel_integral(12.2)[None, :]
        yy, xx = torch.meshgrid(axis, axis, indexing="ij")
        aperture = (xx - 12.2).square() + (yy - 11.1).square() < 6.0**2
        observations = []
        source_images = []
        for wavelength, amplitude in ((1.0, 0.7), (3.0, 1.3)):
            observation = FieldObservation(
                image=torch.zeros((size, size)),
                header=dict(header),
                weight=torch.ones((size, size)),
                mask=torch.ones((size, size), dtype=torch.bool),
                wavelength=wavelength,
                psf_sigma_pix=0.7,
                band_name=f"B{wavelength:g}",
            )
            source_image = apply_psf_spec(
                (amplitude * source).float(), observation.psf_spec
            )
            observation.image = apply_psf_spec(
                (0.02 + amplitude * source).float(), observation.psf_spec
            )
            observations.append(observation)
            source_images.append(source_image)

        model = SpectralPowerFoamField(
            domain,
            **_regular_foam_model_kwargs(
                grid_side=12,
                basis_count=2,
                affine_spatial_basis=True,
                extent=0.55,
            ),
            compact_fixed_geometry=True,
        )
        _set_foam_geometry_trainable(model, False)
        if initialize_from_coadd:
            _initialize_foam_from_band_coadds(
                model,
                [
                    {
                        "coadd_pred": observation.image.numpy(),
                        "support_mask": observation.mask.numpy(),
                        "target_header": observation.header,
                        "target_observation": observation,
                    }
                    for observation in observations
                ],
                intensity_scale=1.0,
                num_spectral_samples=5,
                whiten_spectral_basis=True,
            )
        else:
            fit_field_model(
                observations,
                model,
                steps=200,
                device="cpu",
                **_joint_real_fit_kwargs(
                    "spectral_power_foam",
                    crop_size=size,
                    steps=200,
                    num_observations=len(observations),
                    foam_grid_side=12,
                    foam_fixed_geometry=True,
                ),
            )

        for observation, source_image in zip(observations, source_images, strict=True):
            prediction = render_spectral_power_foam_observation_exact(
                model, observation, num_spectral_samples=5
            ).detach()
            background = observation.image - source_image
            flux_ratio = (prediction - background)[aperture].sum() / source_image[
                aperture
            ].sum()
            assert float(flux_ratio) == pytest.approx(1.0, abs=0.05)

        diagnostics = _refine_foam_from_observations_exact(
            model, observations, num_spectral_samples=5, max_iterations=100
        )
        assert diagnostics["foam_exact_refinement_converged"] == 1.0
        for observation, source_image in zip(observations, source_images, strict=True):
            prediction = render_spectral_power_foam_observation_exact(
                model, observation, num_spectral_samples=5
            ).detach()
            background = observation.image - source_image
            flux_ratio = (prediction - background)[aperture].sum() / source_image[
                aperture
            ].sum()
            assert float(flux_ratio) == pytest.approx(1.0, abs=0.01)
            assert float((prediction - observation.image).square().mean().sqrt()) < 0.002
    finally:
        torch.set_num_threads(previous_threads)
