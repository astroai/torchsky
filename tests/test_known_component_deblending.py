from __future__ import annotations

from dataclasses import replace

import pytest
import torch

from torchsky.catalogs import (
    KnownComponentFitError,
    ParametricBounds,
    fit_morphology_spectrum_deblending,
    fit_parametric_deblending,
)
from torchsky.catalogs.deblending import _catastrophic_swap, _project_morphology
from torchsky.map.field import FieldObservation, PSFSpec
from torchsky.scene import SourceCatalogScene
from torchsky.wcs import WCS


def _header(size: int = 25) -> dict[str, object]:
    center = 0.5 * (size + 1)
    return {
        "NAXIS": 2,
        "NAXIS1": size,
        "NAXIS2": size,
        "CTYPE1": "RA---TAN",
        "CTYPE2": "DEC--TAN",
        "CRPIX1": center,
        "CRPIX2": center,
        "CRVAL1": 180.0,
        "CRVAL2": 0.0,
        "CDELT1": -1.0 / 3600.0,
        "CDELT2": 1.0 / 3600.0,
    }


def _positions(
    header: dict[str, object], pixels: list[tuple[float, float]]
) -> tuple[torch.Tensor, torch.Tensor]:
    value = torch.tensor(pixels, dtype=torch.float64)
    return WCS(header).pixel_to_world(value[:, 0], value[:, 1])


def _blank(
    header: dict[str, object],
    *,
    band: str | None = None,
    sigma: float = 0.8,
    device: str = "cpu",
) -> FieldObservation:
    dtype = torch.float32 if device == "mps" else torch.float64
    image = torch.zeros(
        (int(header["NAXIS2"]), int(header["NAXIS1"])),
        dtype=dtype,
        device=device,
    )
    return FieldObservation(
        image=image,
        weight=torch.ones_like(image),
        header=header,
        band_name=band,
        psf_spec=PSFSpec.gaussian(sigma),
    )


def test_parametric_variable_projection_refines_position_and_preserves_ids() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.3, 12.2), (14.1, 12.0)])
    truth = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor([41, 73]),
        coefficients=torch.tensor([[8.0], [0.8]], dtype=torch.float64),
    )
    blank = _blank(header)
    observation = replace(blank, image=truth.render_observation(blank))
    initial = SourceCatalogScene(
        lon + torch.tensor([0.25, -0.2], dtype=torch.float64) / 3600.0,
        lat + torch.tensor([-0.15, 0.1], dtype=torch.float64) / 3600.0,
        component_ids=torch.tensor([41, 73]),
        coefficients=torch.ones((2, 1), dtype=torch.float64),
    )

    result = fit_parametric_deblending(
        [observation],
        initial,
        refine_shapes=False,
        background="none",
        max_iterations=12,
        failure_policy="return",
    )

    torch.testing.assert_close(result.scene.component_ids, torch.tensor([41, 73]))
    torch.testing.assert_close(result.flux[:, 0], torch.tensor([8.0, 0.8], dtype=torch.float64), rtol=2e-2, atol=2e-2)
    assert torch.linalg.vector_norm(result.residuals[0]) < 2.0e-3
    assert result.provenance["source_detection"] is False
    assert result.provenance["source_birth_death"] is False


def test_bounded_sersic_render_and_refinement_are_finite() -> None:
    header = _header(size=31)
    lon, lat = _positions(header, [(15.1, 14.8)])
    truth_parameters = torch.tensor([[1.6, 0.65, 25.0, 2.0]], dtype=torch.float64)
    truth = SourceCatalogScene(
        lon,
        lat,
        morphology="sersic",
        sersic_parameters=truth_parameters,
        coefficients=torch.tensor([[3.0]], dtype=torch.float64),
    )
    blank = _blank(header, sigma=0.6)
    stamp = truth.render_component(0, blank)
    assert stamp is not None
    assert bool(torch.isfinite(stamp.data).all())
    observation = replace(blank, image=truth.render_observation(blank))
    initial = SourceCatalogScene(
        lon,
        lat,
        morphology="sersic",
        sersic_parameters=torch.tensor([[1.3, 0.75, 20.0, 1.7]], dtype=torch.float64),
        coefficients=torch.tensor([[1.0]], dtype=torch.float64),
        support_radius_arcsec=truth.support_radius_arcsec,
    )

    result = fit_parametric_deblending(
        [observation],
        initial,
        refine_positions=False,
        bounds=ParametricBounds(position_arcsec=0.5),
        background="none",
        max_iterations=10,
        failure_policy="return",
    )

    assert bool(torch.isfinite(result.scene.sersic_parameters).all())
    assert result.diagnostics[0].final_objective < result.diagnostics[0].initial_objective
    assert result.scene.sersic_parameters[0, 3] >= 0.3
    assert result.scene.sersic_parameters[0, 3] <= 8.0


def test_morphology_spectrum_factorization_recovers_color_and_constraints() -> None:
    header = _header(size=29)
    lon, lat = _positions(header, [(13.0, 14.0), (15.0, 14.0)])
    truth_templates = torch.zeros((2, 5, 5), dtype=torch.float64)
    truth_templates[0, 2, 2] = 4.0
    truth_templates[0, 2, 1:4] += 1.0
    truth_templates[1, 1:4, 1:4] = torch.tensor(
        [[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]],
        dtype=torch.float64,
    )
    truth = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor([3, 9]),
        morphology="template",
        fixed_templates=truth_templates,
        template_pixel_scale_arcsec=0.5,
        coefficients=torch.tensor([[5.0, 1.0], [1.0, 4.0]], dtype=torch.float64),
        band_labels=("g", "r"),
    )
    blanks = [_blank(header, band="g"), _blank(header, band="r", sigma=1.0)]
    observations = [replace(item, image=truth.render_observation(item)) for item in blanks]
    initial_templates = (truth_templates + 0.2).clone()
    initial = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor([3, 9]),
        morphology="template",
        fixed_templates=initial_templates,
        template_pixel_scale_arcsec=0.5,
        coefficients=torch.ones((2, 2), dtype=torch.float64),
        band_labels=("g", "r"),
        support_radius_arcsec=truth.support_radius_arcsec,
    )

    result = fit_morphology_spectrum_deblending(
        observations,
        initial,
        symmetric=True,
        background="none",
        max_iterations=12,
        step_size=0.02,
        failure_policy="return",
    )

    assert result.scene.component_count == 2
    torch.testing.assert_close(result.scene.component_ids, torch.tensor([3, 9]))
    assert bool((result.spectra >= 0).all())
    for morphology in result.morphologies:
        torch.testing.assert_close(morphology.sum(), torch.tensor(1.0, dtype=morphology.dtype))
        torch.testing.assert_close(morphology, torch.flip(morphology, dims=(0, 1)))
    assert result.diagnostics[0].final_objective < result.diagnostics[0].initial_objective


def test_missing_band_reports_spectrum_gauge_without_source_creation() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.0, 12.0)])
    template = torch.ones((1, 3, 3), dtype=torch.float64)
    truth = SourceCatalogScene(
        lon,
        lat,
        morphology="template",
        fixed_templates=template,
        template_pixel_scale_arcsec=0.5,
        coefficients=torch.tensor([[2.0, 7.0]], dtype=torch.float64),
        band_labels=("g", "r"),
    )
    blank = _blank(header, band="g")
    observation = replace(blank, image=truth.render_observation(blank))

    result = fit_morphology_spectrum_deblending(
        [observation],
        truth,
        background="none",
        max_iterations=2,
        failure_policy="return",
    )

    assert torch.isnan(result.spectra[0, 1])
    assert result.linear_diagnostics[0].gauge_parameter_indices.tolist() == [1]
    assert result.scene.component_count == 1


def test_inadequate_psf_fails_closed_and_carries_diagnostics() -> None:
    header = _header()
    lon, lat = _positions(header, [(12.0, 12.0)])
    truth = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[50.0]], dtype=torch.float64),
    )
    blank = _blank(header, sigma=0.5)
    image = truth.render_observation(blank)
    data = replace(blank, image=image, psf_spec=PSFSpec.gaussian(2.0), weight=torch.full_like(image, 1.0e5))

    with pytest.raises(KnownComponentFitError) as caught:
        fit_parametric_deblending(
            [data],
            truth,
            refine_positions=False,
            refine_shapes=False,
            background="none",
        )

    assert caught.value.result.adequacy.inadequate
    assert caught.value.result.scene.component_count == 1


def test_heldout_likelihood_covariance_and_image_gradient() -> None:
    header = _header(size=19)
    lon, lat = _positions(header, [(9.0, 9.0)])
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[2.0]], dtype=torch.float64),
    )
    blank = _blank(header)
    image = scene.render_observation(blank).detach().requires_grad_(True)
    observation = replace(blank, image=image, weight=torch.ones_like(image))
    holdout = torch.zeros_like(image, dtype=torch.bool)
    holdout[::2, ::2] = True

    result = fit_parametric_deblending(
        [observation],
        scene,
        refine_positions=False,
        refine_shapes=False,
        background="none",
        heldout_masks=[holdout],
        failure_policy="return",
    )
    gradient = torch.autograd.grad(result.coefficients.sum(), image)[0]

    assert result.adequacy.heldout_log_likelihood is not None
    assert result.covariance.to_dense().shape == (1, 1)
    assert bool(torch.isfinite(gradient).all())
    assert torch.count_nonzero(gradient[~holdout]) > 0


def test_tiny_dense_flux_and_covariance_oracle() -> None:
    header = _header(size=17)
    lon, lat = _positions(header, [(7.5, 8.0), (9.0, 8.0)])
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[3.0], [0.4]], dtype=torch.float64),
    )
    blank = _blank(header, sigma=1.1)
    observation = replace(
        blank,
        image=scene.render_observation(blank),
        weight=torch.linspace(0.5, 2.0, 17 * 17, dtype=torch.float64).reshape(17, 17),
    )
    fit_scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.zeros((2, 1), dtype=torch.float64),
    )

    result = fit_parametric_deblending(
        [observation],
        fit_scene,
        refine_positions=False,
        refine_shapes=False,
        background="none",
        linear_solver="signed",
        failure_policy="return",
    )
    columns = []
    for parameter in range(2):
        state = torch.zeros(2, dtype=torch.float64)
        state[parameter] = 1.0
        columns.append(
            fit_scene.render_observation(blank, parameter_values=state).reshape(-1)
        )
    design = torch.stack(columns, dim=1)
    weight = observation.weight.reshape(-1)
    expected_covariance = torch.linalg.inv(design.T @ (design * weight[:, None]))
    expected_flux = expected_covariance @ (
        design.T @ (weight * observation.image.reshape(-1))
    )

    torch.testing.assert_close(result.coefficients, expected_flux, rtol=1e-9, atol=1e-9)
    torch.testing.assert_close(
        result.covariance.to_dense(), expected_covariance, rtol=1e-8, atol=1e-8
    )


def test_catastrophic_swap_and_monotonic_projection_oracles() -> None:
    header = _header()
    lon, lat = _positions(header, [(10.0, 12.0), (14.0, 12.0)])
    initial = SourceCatalogScene(
        lon,
        lat,
        component_ids=torch.tensor([101, 202]),
        coefficients=torch.ones((2, 1), dtype=torch.float64),
    )
    swapped = SourceCatalogScene(
        lon.flip(0),
        lat.flip(0),
        component_ids=torch.tensor([101, 202]),
        coefficients=torch.ones((2, 1), dtype=torch.float64),
    )
    assert _catastrophic_swap(initial, swapped, torch.tensor([0, 1]))

    morphology = torch.tensor(
        [[0.0, 0.0, 5.0], [0.0, 1.0, 0.0], [4.0, 0.0, 0.0]],
        dtype=torch.float64,
    )
    projected = _project_morphology(morphology, monotonic=True, symmetric=False)
    center = projected[1, 1]
    assert bool((projected <= center).all())
    assert bool((projected >= 0).all())
    torch.testing.assert_close(projected.sum(), torch.tensor(1.0, dtype=torch.float64))


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_parametric_fixed_geometry_preserves_device(device: str) -> None:
    header = _header(size=17)
    lon, lat = _positions(header, [(8.0, 8.0)])
    dtype = torch.float32 if device == "mps" else torch.float64
    scene = SourceCatalogScene(
        lon.to(device=device, dtype=dtype),
        lat.to(device=device, dtype=dtype),
        coefficients=torch.tensor([[1.5]], dtype=dtype, device=device),
    )
    blank = _blank(header, device=device)
    observation = replace(blank, image=scene.render_observation(blank))

    result = fit_parametric_deblending(
        [observation],
        scene,
        refine_positions=False,
        refine_shapes=False,
        background="none",
        chunk_size=7,
        failure_policy="return",
    )

    assert result.coefficients.device.type == device
    assert result.predictions[0].device.type == device


@pytest.mark.parametrize(
    "device",
    [
        "cpu",
        pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")),
        pytest.param("mps", marks=pytest.mark.skipif(not torch.backends.mps.is_available(), reason="MPS unavailable")),
    ],
)
def test_factorization_preserves_device_and_nonnegative_gauge(device: str) -> None:
    header = _header(size=15)
    lon, lat = _positions(header, [(7.0, 7.0)])
    dtype = torch.float32 if device == "mps" else torch.float64
    template = torch.tensor(
        [[[1.0, 2.0, 1.0], [2.0, 5.0, 2.0], [1.0, 2.0, 1.0]]],
        dtype=dtype,
        device=device,
    )
    scene = SourceCatalogScene(
        lon.to(device=device, dtype=dtype),
        lat.to(device=device, dtype=dtype),
        morphology="template",
        fixed_templates=template,
        template_pixel_scale_arcsec=0.5,
        coefficients=torch.tensor([[2.0]], dtype=dtype, device=device),
    )
    blank = _blank(header, device=device)
    observation = replace(blank, image=scene.render_observation(blank))

    result = fit_morphology_spectrum_deblending(
        [observation],
        scene,
        background="none",
        max_iterations=1,
        chunk_size=2,
        failure_policy="return",
    )

    assert result.spectra.device.type == device
    assert result.morphologies[0].device.type == device
    assert bool((result.morphologies[0] >= 0).all())
    torch.testing.assert_close(
        result.morphologies[0].sum(), torch.tensor(1.0, dtype=dtype, device=device)
    )


def test_separated_components_stay_in_bounded_single_source_groups() -> None:
    header = _header(size=97)
    lon, lat = _positions(header, [(10.0, 48.0), (48.0, 48.0), (86.0, 48.0)])
    scene = SourceCatalogScene(
        lon,
        lat,
        coefficients=torch.tensor([[1.0], [2.0], [3.0]], dtype=torch.float64),
    )
    blank = _blank(header)
    observation = replace(blank, image=scene.render_observation(blank))

    result = fit_parametric_deblending(
        [observation],
        scene,
        refine_positions=False,
        refine_shapes=False,
        background="none",
        max_group_size=1,
        chunk_size=11,
        failure_policy="return",
    )

    assert [group.numel() for group in result.grouping.blend_groups] == [1, 1, 1]
    assert all(block.numel() == 0 for block in result.nonlinear_covariance)


def test_chunked_and_unchunked_factorization_match() -> None:
    header = _header(size=21)
    lon, lat = _positions(header, [(10.0, 10.0)])
    template = torch.arange(1, 26, dtype=torch.float64).reshape(1, 5, 5)
    scene = SourceCatalogScene(
        lon,
        lat,
        morphology="template",
        fixed_templates=template,
        template_pixel_scale_arcsec=0.4,
        coefficients=torch.tensor([[2.0]], dtype=torch.float64),
    )
    blank = _blank(header)
    observation = replace(blank, image=scene.render_observation(blank))

    unchunked = fit_morphology_spectrum_deblending(
        [observation],
        scene,
        background="none",
        max_iterations=2,
        failure_policy="return",
    )
    chunked = fit_morphology_spectrum_deblending(
        [observation],
        scene,
        background="none",
        max_iterations=2,
        chunk_size=3,
        failure_policy="return",
    )

    torch.testing.assert_close(chunked.spectra, unchunked.spectra)
    torch.testing.assert_close(chunked.morphologies[0], unchunked.morphologies[0])
    torch.testing.assert_close(chunked.predictions[0], unchunked.predictions[0])


def test_sersic_edge_clipped_source_loses_off_image_flux() -> None:
    header = _header(size=41)
    blank = _blank(header, sigma=0.01)
    coefficients = torch.tensor([[1.0]], dtype=torch.float64)
    parameters = torch.tensor([[2.0, 1.0, 0.0, 1.0]], dtype=torch.float64)
    totals = []
    for x_pixel in (20.0, 0.0):
        lon, lat = _positions(header, [(x_pixel, 20.0)])
        scene = SourceCatalogScene(
            lon,
            lat,
            morphology="sersic",
            sersic_parameters=parameters,
            coefficients=coefficients,
        )
        totals.append(float(scene.render_observation(blank).sum()))
    assert totals[0] == pytest.approx(1.0, abs=0.05)
    assert totals[1] < 0.75 * totals[0]
